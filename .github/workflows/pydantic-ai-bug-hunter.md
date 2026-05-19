---
emoji: "🐛"
name: "Pydantic AI Bug Hunter"
description: "Find a reproducible, user-impacting bug in pydantic-ai and file a report issue. Runs on the Pydantic AI harness engine; the task prompt is iterable from a Logfire managed variable."
# Fuzzy schedule: gh-aw scatters the daily run to a repo-stable off-peak time
# (avoids the top-of-hour stampede) and auto-adds workflow_dispatch.
on: daily
permissions:
  contents: read
  issues: read
  pull-requests: read
concurrency:
  group: ${{ github.workflow }}-bug-hunter
  cancel-in-progress: true
network:
  allowed:
    - defaults
    # ANTHROPIC_BASE_URL is a compile-time literal (below) so gh-aw already
    # auto-allowlists the host; this explicit entry is a harmless safety net.
    - api.minimax.io
# We register as the built-in `claude` engine and only override `command`, so
# gh-aw runs its full Claude proxy + credential-injection machinery for us.
# ANTHROPIC_BASE_URL MUST be a compile-time literal (not a ${{ vars.* }}
# expression): gh-aw derives the api-proxy target host AND the
# `--anthropic-api-base-path` from its parsed URL path at compile time. With a
# vars expression the path can't be parsed, so the proxy drops the `/anthropic`
# prefix and the upstream returns 404. Only ANTHROPIC_API_KEY stays a secret
# (injected by the AWF api-proxy, excluded from the agent container). MiniMax
# exposes an Anthropic-compatible API at https://api.minimax.io/anthropic.
runtimes:
  uv: {}
engine:
  id: claude
  # The checked-out workspace is mounted no-exec in the AWF sandbox, so a
  # pre-step stages a launcher in gh-aw's exec-able /tmp/gh-aw/bin that runs
  # `uv run --script` against the workspace harness.
  command: /tmp/gh-aw/bin/pydantic-ai-runner-launch
  env:
    ANTHROPIC_BASE_URL: https://api.minimax.io/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    GH_AW_HARNESS_MODEL: ${{ vars.MODEL }}
tools:
  github:
    mode: gh-proxy
    toolsets: [default]
safe-outputs:
  activation-comments: false
  noop:
  create-issue:
    max: 1
    title-prefix: "[bug-hunter] "
    close-older-key: "[bug-hunter]"
    close-older-issues: false
    expires: 7d
timeout-minutes: 90
imports:
  - shared/otel-logfire.md
pre-steps:
  # Setting engine.command makes gh-aw skip ALL engine installation steps,
  # which also drops the bundled AWF firewall binary install. Re-run gh-aw's
  # own installer (the same call it makes for non-custom-command jobs).
  - name: Install AWF firewall binary (skipped by custom engine.command)
    run: bash "${RUNNER_TEMP}/gh-aw/actions/install_awf_binary.sh" v0.25.46
  # Stage (not install) a launcher at gh-aw's exec-able /tmp/gh-aw/bin path.
  # uv itself is installed by runtimes.uv; this only writes a wrapper file
  # that runs `uv run --script` on the workspace harness at agent time.
  - name: Stage Pydantic AI harness launcher
    run: |
      set -euo pipefail
      mkdir -p /tmp/gh-aw/bin
      cat > /tmp/gh-aw/bin/pydantic-ai-runner-launch <<'WRAP'
      #!/usr/bin/env bash
      set -euo pipefail
      # setup-uv points UV_CACHE_DIR at ${RUNNER_TEMP}/setup-uv-cache, which is
      # not writable by the chrooted sandbox user (UID 1001). Only /tmp/gh-aw
      # is owned by that user, so redirect every uv-writable dir there.
      export UV_CACHE_DIR=/tmp/gh-aw/uv/cache
      export UV_PYTHON_INSTALL_DIR=/tmp/gh-aw/uv/python
      export UV_TOOL_DIR=/tmp/gh-aw/uv/tool
      export XDG_DATA_HOME=/tmp/gh-aw/uv/data
      export XDG_CACHE_HOME=/tmp/gh-aw/uv/xdg-cache
      mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
      runner="${GITHUB_WORKSPACE}/.github/scripts/pydantic-ai-runner"
      echo "[harness-launch] cwd=$(pwd) GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-unset} UV_CACHE_DIR=${UV_CACHE_DIR}" >&2
      echo "[harness-launch] runner=${runner} exists=$([ -f "${runner}" ] && echo yes || echo no)" >&2
      uv_bin=""
      if command -v uv >/dev/null 2>&1; then
        uv_bin="$(command -v uv)"
      else
        for c in "${HOME}/.local/bin/uv" "${RUNNER_TOOL_CACHE:-/opt/hostedtoolcache}"/uv/*/*/uv /opt/hostedtoolcache/uv/*/*/uv /home/runner/work/_tool/uv/*/*/uv /usr/local/bin/uv; do
          [ -x "$c" ] && uv_bin="$c" && break
        done
      fi
      if [ -z "${uv_bin}" ]; then
        echo "[harness-launch] FATAL: uv not found; PATH=${PATH}" >&2
        exit 127
      fi
      echo "[harness-launch] using uv=${uv_bin}" >&2
      exec "${uv_bin}" run --script "${runner}" "$@"
      WRAP
      chmod +x /tmp/gh-aw/bin/pydantic-ai-runner-launch

jobs:
  fetch_dynamic_prompt:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      contents: none
    outputs:
      dynamic_prompt: ${{ steps.logfire.outputs.dynamic_prompt }}
    steps:
      - name: Fetch agent prompt from Logfire managed variables
        id: logfire
        env:
          LOGFIRE_READ_KEY: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES_KEY }}
          LOGFIRE_API_HOST: logfire-api.pydantic.dev
          LOGFIRE_VARIABLE_KEY: gh_aw_pydantic_ai_bug_hunter_prompt
          TARGETING_KEY: gh-aw-${{ github.repository }}
        run: |
          set -euo pipefail

          emit_prompt() {
            {
              echo "dynamic_prompt<<__GH_AW_DYNAMIC_PROMPT_EOF__"
              printf '%s\n' "$1"
              echo "__GH_AW_DYNAMIC_PROMPT_EOF__"
            } >> "$GITHUB_OUTPUT"
          }

          # Prompt iteration via Logfire is opt-in. If the read key is not
          # configured, fall back to the baked-in instructions below so the
          # workflow still runs (just without live prompt overrides).
          if [ -z "${LOGFIRE_READ_KEY:-}" ]; then
            echo "::notice::LOGFIRE_READ_EXTERNAL_VARIABLES_KEY not set — using baked-in static prompt only."
            emit_prompt ""
            exit 0
          fi

          RESPONSE="$(curl --fail --silent --show-error \
            --max-time 20 \
            -X POST \
            -H "Authorization: Bearer ${LOGFIRE_READ_KEY}" \
            -H "Content-Type: application/json" \
            -d "{\"context\":{\"targetingKey\":\"${TARGETING_KEY}\"}}" \
            "https://${LOGFIRE_API_HOST}/v1/ofrep/v1/evaluate/flags/${LOGFIRE_VARIABLE_KEY}")" || {
              echo "::warning::Logfire OFREP request failed — using baked-in static prompt only."
              emit_prompt ""
              exit 0
            }

          PROMPT="$(printf '%s' "$RESPONSE" | jq -r '.value // empty')"

          if [ -z "$PROMPT" ]; then
            REASON="$(printf '%s' "$RESPONSE" | jq -r '.reason // "UNKNOWN"')"
            echo "::notice::No Logfire value for ${LOGFIRE_VARIABLE_KEY} (reason: ${REASON}) — using baked-in static prompt only."
            emit_prompt ""
            exit 0
          fi

          emit_prompt "$PROMPT"
          echo "Loaded dynamic prompt (${#PROMPT} chars) from Logfire variable '${LOGFIRE_VARIABLE_KEY}'."
---

# Pydantic AI Bug Hunter

You are running under the **Pydantic AI harness engine** (not the Claude Code
CLI), driving a model through gh-aw's AWF firewall and credential-injecting
proxy. You have native `bash`, `read_file`, `grep`, `list_dir` tools plus the
gh-aw GitHub tools and the `create_issue` / `noop` safe-output tools.

Repository: `${{ github.repository }}` — [Pydantic AI](https://ai.pydantic.dev/),
a provider-agnostic GenAI agent framework for Python. It is a `uv` workspace:
`pydantic_ai_slim/` (the agent framework), `pydantic_graph/`, `pydantic_evals/`,
`clai/`, with tests in `tests/`.

## Objective

Find a single reproducible, user-impacting bug that can be covered by a minimal
failing test. Not a number field accepting `"ABC"` — a real, impactful bug.

**The bar is high: you must actually reproduce the bug before filing.** Most
runs should end with `noop` — that means the codebase is healthy. Filing a
weak or speculative issue is worse than filing nothing.

### Data Gathering

1. Review recent changes: run `git log --since="28 days ago" --stat` and
   identify candidates with user-facing impact. Read the diffs and related
   files for each candidate.
2. Investigate from multiple angles — different subsystems (model providers,
   the agent loop, tools, output handling, message history), different bug
   categories (logic errors, type-safety gaps, async edge cases), and
   different recent commits.
3. Reproduce locally — **mandatory, not optional**:
   - Write a **new** minimal reproduction: a small script or test that directly
     triggers the specific bug you identified. Do **not** run the existing
     suite (`make test`, `pytest`) and report its failures — if you did not
     write the test, a failure is not your finding.
   - Capture the exact steps and output from your reproduction.
   - If you cannot write a concrete reproduction that fails due to the bug, do
     **not** file it. Call `noop` instead.

### What to Look For

- Logic errors: incorrect conditionals, off-by-one, wrong variable, missing
  edge-case handling.
- Clear user impact: wrong output, raised/swallowed exception, broken agent
  run, incorrect tool dispatch, mis-serialized message history.
- Deterministic reproduction (not flaky) that you trigger yourself.
- Expressible as a minimal failing test (unit or integration).

### What to Skip

- Theoretical concerns without a reproduction — no "this looks like it could break."
- Code that "looks wrong" but works correctly in practice.
- Existing test-suite failures you did not cause.
- Edge cases needing unusual or undocumented inputs.
- Issues requiring large refactors or design changes.
- Behavior already tracked by an open issue.
- **By-design behavior.** Check for nearby comments explaining the choice,
  consistent patterns across the codebase, and recent PRs/commits for context.
  If the "bug" requires assuming an error despite an established pattern, it is
  probably by-design.

### Quality Gate — When to Noop

Call `noop` if any of these are true:
- You could not write a concrete reproduction that triggers the bug.
- Your only evidence is an existing test failure you did not cause.
- The bug is speculative — inferred from reading code, not triggered.
- A similar issue is already open.
- The impact is cosmetic or low-severity (e.g., a typo in a log message).
- The bug is already fixed in a recently merged PR (search before filing).

### Issue Format

**Title:** Short bug summary

**Body:**

> ## Impact
> [Who/what is affected, why it matters]
>
> ## Reproduction Steps
> 1. [Exact commands you ran, including the new test or script you wrote]
>
> ## Expected vs Actual
> **Expected:** ...
> **Actual:** ... [include actual command output]
>
> ## Failing Test
> [The new test/script you wrote — include the full code]
>
> ## Evidence
> - [Commands/output captured, file references with `path:line`]

## Dynamic instructions

The following are loaded at run time from the Logfire managed variable
`gh_aw_pydantic_ai_bug_hunter_prompt`, so the task can be tuned, A/B-tested, or
rolled back from the Logfire UI without recompiling or committing this
workflow. They **override or extend** the guidance above. If empty, the
baked-in instructions above stand on their own.

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}

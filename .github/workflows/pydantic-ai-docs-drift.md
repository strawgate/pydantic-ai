---
emoji: "📚"
name: "Pydantic AI Docs Drift"
description: "Detect code changes that require documentation updates and file an issue. Runs on the Pydantic AI harness engine; the task prompt is iterable from a Logfire managed variable."
# Fuzzy schedule: docs drift accumulates slowly, so weekly (not daily).
# gh-aw scatters the run to a repo-stable time and auto-adds workflow_dispatch.
on: weekly on monday
permissions:
  contents: read
  issues: read
  pull-requests: read
concurrency:
  group: ${{ github.workflow }}-docs-drift
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
# `--anthropic-api-base-path` from its parsed URL path at compile time. Only
# ANTHROPIC_API_KEY stays a secret (injected by the AWF api-proxy, excluded
# from the agent container). MiniMax exposes an Anthropic-compatible API at
# https://api.minimax.io/anthropic.
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
    ANTHROPIC_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    GH_AW_HARNESS_MODEL: ${{ vars.GH_AW_HARNESS_MODEL }}
tools:
  github:
    mode: gh-proxy
    toolsets: [default]
safe-outputs:
  activation-comments: false
  noop:
  create-issue:
    max: 1
    title-prefix: "[docs-drift] "
    close-older-key: "[docs-drift]"
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
          LOGFIRE_VARIABLE_KEY: gh_aw_pydantic_ai_docs_drift_prompt
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

# Pydantic AI Docs Drift

You are running under the **Pydantic AI harness engine** (not the Claude Code
CLI), driving a model through gh-aw's AWF firewall and credential-injecting
proxy. You have native `bash`, `read_file`, `grep`, `list_dir` tools plus the
gh-aw GitHub tools and the `create_issue` / `noop` safe-output tools.

Repository: `${{ github.repository }}` — [Pydantic AI](https://ai.pydantic.dev/).
Documentation lives in `docs/` (built with `mkdocs`, configured in
`mkdocs.yml`), plus `README.md`, `CONTRIBUTING.md`, and per-package
`AGENTS.md` files. Doc code examples are tested by `tests/test_examples.py`.

## Objective

Detect documentation drift — code changes that require corresponding
documentation updates.

**Noop is the expected outcome most days.** Only file an issue when the
documentation is concretely wrong, a new public feature has zero docs, or a
removed/renamed public interface is still referenced in docs.

### Data Gathering

1. Run `git log --since="7 days ago" --oneline --stat` for a summary of recent
   commits. If there are no commits in the window, call `noop` and stop.
2. Inventory documentation: scan `docs/`, `mkdocs.yml`, `README.md`,
   `CONTRIBUTING.md`, and `AGENTS.md` files. Do not assume a fixed structure.

### What to Look For

For each commit (or group of related commits), determine whether the change
could require documentation updates:

1. **Public API changes** — new/renamed/removed classes, methods, function
   signatures, `Agent` options, model/provider classes, CLI flags.
2. **Behavioral changes** — altered defaults, changed exceptions/messages,
   modified control flow affecting user-facing behavior.
3. **New features** — anything a user or contributor needs to know about.
4. **Dependency/tooling changes** — version bumps, new optional dependency
   groups, changed build/test commands.
5. **Structural changes** — moved/renamed/deleted files referenced in docs or
   in `mkdocs.yml` nav.
6. **Doc code examples** — code blocks in `docs/` that no longer match the API.

### How to Analyze

For each potentially impactful change: read the full diff, read the current
docs, check whether docs were already updated in the same or a later commit in
the window, and check whether an open issue/PR already tracks it.

### What to Skip

- Purely internal refactors with no user-facing impact.
- Changes where docs were already updated in the same/later commit.
- Changes already tracked by an open issue or PR.
- Test-only changes.
- Minor changes where existing docs are still substantially correct.

### Issue Format

**Title:** Brief summary (e.g., "Update agent.md for new `Agent` output option")

**Body:**

> Recent code changes have introduced documentation drift. The following
> changes need corresponding documentation updates.
>
> ## Changes Requiring Documentation Updates
>
> ### 1. [Brief description]
> **Commit(s):** [SHA(s)]
> **What changed:** [Concise description]
> **Documentation impact:** [Which doc file(s) and what specifically]
>
> ## Suggested Actions
> - [ ] [Specific, actionable checkbox per doc update needed]

## Dynamic instructions

The following are loaded at run time from the Logfire managed variable
`gh_aw_pydantic_ai_docs_drift_prompt`, so the task can be tuned, A/B-tested, or
rolled back from the Logfire UI without recompiling or committing this
workflow. They **override or extend** the guidance above. If empty, the
baked-in instructions above stand on their own.

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}

---
emoji: "🔁"
name: "Pydantic AI Harness PR Self-Test"
description: "PR self-test: run the Pydantic AI harness engine on this PR so we can iterate via the job log"
on:
  pull_request:
    types: [opened, synchronize]
    paths:
      - .github/scripts/pydantic-ai-runner
      - .github/scripts/test_pydantic_ai_runner.py
      - .github/workflows/pydantic-ai-pr-selftest.md
  workflow_dispatch:
permissions:
  contents: read
  pull-requests: read
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
  # The checked-out workspace is mounted no-exec in the AWF sandbox (spawning
  # a repo script gives EACCES). gh-aw's exec-able convention is /tmp/gh-aw/bin
  # — a pre-step stages a launcher there that runs `uv run --script` against
  # the workspace harness (uv READS the file, so no-exec/exec-bit is moot).
  command: /tmp/gh-aw/bin/pydantic-ai-runner-launch
  env:
    ANTHROPIC_BASE_URL: https://api.minimax.io/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    GH_AW_HARNESS_MODEL: ${{ vars.GH_AW_HARNESS_MODEL }}
tools:
  github:
    mode: gh-proxy
    # Self-test only fetches this PR's title — least privilege, no issues toolset.
    toolsets: [pull_requests]
safe-outputs:
  add-comment:
    max: 1
  noop:
timeout-minutes: 15
imports:
  - shared/otel-logfire.md
pre-steps:
  # Setting engine.command makes gh-aw skip ALL engine installation steps
  # (claude_engine.go GetInstallationSteps returns []), which also drops the
  # bundled AWF firewall binary install — so the agent step's `sudo -E awf`
  # fails with "awf: command not found". Re-run gh-aw's own installer (the
  # exact call gh-aw makes for non-custom-command jobs). The helper is staged
  # by the preceding "Setup Scripts" step and needs no repo checkout.
  - name: Install AWF firewall binary (skipped by custom engine.command)
    run: bash "${RUNNER_TEMP}/gh-aw/actions/install_awf_binary.sh" v0.25.46
  # Stage (not install) a launcher at gh-aw's exec-able /tmp/gh-aw/bin path.
  # uv itself is installed by runtimes.uv; this only writes a wrapper file.
  # It runs `uv run --script` on the workspace harness at agent time, after
  # gh-aw's checkout, so the no-exec workspace mount is not a problem.
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
---

# Pydantic AI Harness PR Self-Test

You are running under the **Pydantic AI harness engine** (not the Claude Code
CLI), backed by the configured MiniMax Anthropic-compatible endpoint. This
workflow verifies, on a real PR, that the harness supports a multi-step agent
loop with native tools (bash/read/grep/...) plus gh-aw safe-outputs.

## Task

Work step by step, using your tools (do not guess — actually call them):

1. Print a recognizable banner line: `PYDANTIC_AI_HARNESS_SELFTEST_OK`.
2. State the repository (`${{ github.repository }}`).
3. Use the `bash` tool to count workflow files:
   `ls .github/workflows/*.md | wc -l`. Report the number.
4. Use the `grep` tool to find the line in `.github/scripts/pydantic-ai-runner`
   that defines `NATIVE_TOOLS` and quote it.
5. Use the `read_file` tool to read `pyproject.toml` and quote the `name = ...`
   line under `[project]`.
6. Use a GitHub tool to fetch this pull request's title and quote it back.
7. You **must** call the `add_comment` safe-output tool exactly once to post a
   short PR comment summarizing the above (banner, repo, the workflow count,
   the project name, the PR title). Do not just print the summary — the run
   only counts as passing if the `add_comment` tool is actually called.

Keep the final message under ~250 words. Do not modify any tracked files
(scratch writes under `/tmp` are fine).

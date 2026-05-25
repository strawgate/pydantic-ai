#!/usr/bin/env bash
# Pre-warm the harness's uv script environment on the OPEN network.
#
# Runs in each workflow's `pre-agent-steps` — after checkout + Setup uv but
# before the firewalled agent step — into the same uv dirs the in-sandbox
# launcher uses, so the agent run reuses the warm cache instead of depending
# on PyPI access through the AWF firewall.
#
# Strictly non-fatal: on any failure the sandboxed `uv run --script` (with
# the `python` allowlist in each workflow) still installs from scratch.
set -uo pipefail
export UV_CACHE_DIR=/tmp/gh-aw/uv/cache
export UV_PYTHON_INSTALL_DIR=/tmp/gh-aw/uv/python
export UV_TOOL_DIR=/tmp/gh-aw/uv/tool
export XDG_DATA_HOME=/tmp/gh-aw/uv/data
export XDG_CACHE_HOME=/tmp/gh-aw/uv/xdg-cache
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
runner="${GITHUB_WORKSPACE}/.github/scripts/pydantic-ai-runner"
uv_bin=""
if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
else
  for c in "${HOME}/.local/bin/uv" "${RUNNER_TOOL_CACHE:-/opt/hostedtoolcache}"/uv/*/*/uv /opt/hostedtoolcache/uv/*/*/uv /home/runner/work/_tool/uv/*/*/uv /usr/local/bin/uv; do
    [ -x "$c" ] && uv_bin="$c" && break
  done
fi
if [ -z "${uv_bin}" ]; then
  echo "::warning::uv not found for pre-warm; agent will install under the firewall"
  exit 0
fi
echo "[harness-prewarm] using uv=${uv_bin} cache=${UV_CACHE_DIR}"
"${uv_bin}" sync --script "${runner}" \
  || echo "::warning::harness uv pre-warm failed; agent will install under the firewall"

# Stage uv and rg in /tmp/gh-aw/bin so the agent can call them by name.
# /tmp/gh-aw/bin is added to PATH by the launcher before exec'ing the shim.
mkdir -p /tmp/gh-aw/bin

# Symlink the resolved uv binary so `uv` works inside the firewalled agent.
if [ ! -e /tmp/gh-aw/bin/uv ]; then
  ln -sf "${uv_bin}" /tmp/gh-aw/bin/uv
  echo "[harness-prewarm] symlinked uv -> /tmp/gh-aw/bin/uv"
fi

# Install ripgrep (pre-firewall; open network) so the native Grep tool works.
export UV_TOOL_BIN_DIR=/tmp/gh-aw/bin
"${uv_bin}" tool install ripgrep --force 2>&1 \
  | head -5 \
  || echo "::warning::ripgrep install failed; Grep tool will fall back to Bash grep"
[ -x /tmp/gh-aw/bin/rg ] && echo "[harness-prewarm] rg installed -> /tmp/gh-aw/bin/rg" \
  || echo "::warning::rg not found at /tmp/gh-aw/bin/rg after install"

"""Tests for the Pydantic AI gh-aw harness (.github/scripts/pydantic-ai-runner).

Offline tests cover the gh-aw compatibility surface (argv tolerance, prompt
recovery, model resolution, MCP-config translation, stream-json schema) and
need no network or credentials.

The live test is skipped unless an OpenAI-compatible endpoint is provided via
env:  GH_AW_HARNESS_LIVE_API_KEY, GH_AW_HARNESS_LIVE_BASE_URL,
GH_AW_HARNESS_LIVE_MODEL.  No credentials are stored in this repo.

Run:  uv run --with pytest pytest .github/scripts/test_pydantic_ai_runner.py
"""

import importlib.machinery
import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent / "pydantic-ai-runner"


def _load_harness():
    # The harness has no .py extension, so use an explicit source loader.
    loader = importlib.machinery.SourceFileLoader("par_harness", str(_SCRIPT))
    spec = importlib.util.spec_from_loader("par_harness", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


har = _load_harness()


# The exact argv gh-aw's claude_harness.cjs passes, with the prompt appended
# as the final positional argument.
GHAW_ARGV = [
    "--print",
    "--no-chrome",
    "--allowed-tools",
    "Bash,Read,mcp__github__get_me,mcp__safeoutputs",
    "--debug-file",
    "/tmp/gh-aw/agent-stdio.log",
    "--verbose",
    "--permission-mode",
    "bypassPermissions",
    "--output-format",
    "stream-json",
    "--mcp-config",
    "/tmp/mcp-servers.json",
    "--prompt-file",
    "/tmp/gh-aw/aw-prompts/prompt.txt",
]


def test_parses_full_claude_argv_without_error():
    args = har.parse_args([*GHAW_ARGV, "do the thing"])
    assert args.mcp_config == "/tmp/mcp-servers.json"
    assert args.prompt_file == "/tmp/gh-aw/aw-prompts/prompt.txt"
    assert args.prompt_positional == "do the thing"


def test_unknown_future_claude_flags_are_tolerated():
    # gh-aw / Claude may add flags later; the harness must not crash, and the
    # trailing prompt positional is still recovered.
    args = har.parse_args([*GHAW_ARGV, "--some-future-flag", "x", "prompt"])
    assert args.prompt_positional == "prompt"


def test_prompt_recovered_from_trailing_positional():
    args = har.parse_args([*GHAW_ARGV, "Investigate the failing CI run."])
    assert har.resolve_prompt(args) == "Investigate the failing CI run."


def test_prompt_falls_back_to_prompt_file(tmp_path):
    pf = tmp_path / "prompt.txt"
    pf.write_text("from file", encoding="utf-8")
    args = har.parse_args(["--prompt-file", str(pf)])
    assert har.resolve_prompt(args) == "from file"


def test_prompt_falls_back_to_env(tmp_path, monkeypatch):
    pf = tmp_path / "p.txt"
    pf.write_text("from env path", encoding="utf-8")
    monkeypatch.setenv("GH_AW_PROMPT", str(pf))
    args = har.parse_args(["--print"])
    assert har.resolve_prompt(args) == "from env path"


def test_model_defaults_to_anthropic(monkeypatch):
    for v in (
        "GH_AW_HARNESS_MODEL",
        "GH_AW_MODEL_AGENT_CLAUDE",
        "ANTHROPIC_MODEL",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(v, raising=False)
    args = har.parse_args(["--print"])
    model, label = har.build_model(args)
    assert label == "anthropic:claude-sonnet-4-5"
    assert model.__class__.__name__ == "AnthropicModel"


def test_model_openai_prefix_builds_openai_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    args = har.parse_args(["--model", "openai:gpt-4o-mini"])
    model, label = har.build_model(args)
    assert label == "openai-compatible:gpt-4o-mini"
    assert model.__class__.__name__ == "OpenAIChatModel"


def test_openai_base_url_triggers_openai_compatible(monkeypatch):
    # An OpenAI-compatible endpoint (vLLM/Together/etc.) with a bare model id.
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GH_AW_HARNESS_MODEL", "some/model")
    args = har.parse_args(["--print"])
    model, label = har.build_model(args)
    assert label == "openai-compatible:some/model"
    assert model.__class__.__name__ == "OpenAIChatModel"


def test_anthropic_base_url_builds_anthropic_model(monkeypatch):
    # Under the gh-aw claude engine the custom endpoint arrives as
    # ANTHROPIC_BASE_URL and the key is proxy-injected (no *_API_KEY in env).
    for v in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GH_AW_HARNESS_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "placeholder")
    monkeypatch.setenv("GH_AW_HARNESS_MODEL", "MiniMax-M2.7-highspeed")
    args = har.parse_args(["--print"])
    model, label = har.build_model(args)
    assert label == "anthropic:MiniMax-M2.7-highspeed"
    assert model.__class__.__name__ == "AnthropicModel"


def test_mcp_missing_config_degrades_gracefully():
    assert har.build_mcp_servers(har.Args(mcp_config="/no/such/file.json")) == []


def test_mcp_translates_stdio_and_http(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "github": {"command": "docker", "args": ["run"], "env": {"X": "1"}},
                    "safeoutputs": {
                        "type": "http",
                        "url": "http://host.docker.internal:1234",
                        "headers": {"Authorization": "k"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    servers = har.build_mcp_servers(har.Args(mcp_config=str(cfg)))
    assert len(servers) == 2
    # Both stdio and http now use the non-deprecated MCPToolset (no prefix).
    assert {s.__class__.__name__ for s in servers} == {"MCPToolset"}


def test_native_tools_registered():
    names = [t.__name__ for t in har.NATIVE_TOOLS]
    assert names == ["bash", "read_file", "write_file", "edit_file", "list_dir", "grep"]


def test_native_file_tools_roundtrip(tmp_path):
    f = tmp_path / "sub" / "note.txt"
    assert "wrote" in har.write_file(str(f), "hello\nworld\n")
    assert har.read_file(str(f)) == "hello\nworld\n"
    assert "edited" in har.edit_file(str(f), "world", "there")
    assert "there" in har.read_file(str(f))
    assert "note.txt" in har.list_dir(str(tmp_path / "sub"))
    assert "error: `old` text not found" == har.edit_file(str(f), "absent", "x")


def test_native_bash_tool(tmp_path):
    out = har.bash("echo hello-from-bash")
    assert "exit=0" in out and "hello-from-bash" in out


def test_native_grep_tool(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nNEEDLE here\n", encoding="utf-8")
    res = har.grep("NEEDLE", str(tmp_path))
    assert "NEEDLE here" in res


def test_emit_result_matches_claude_stream_json_schema():
    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("answer", usage=None, session_id="run-1")
    obj = json.loads(buf.getvalue().strip())
    assert obj["type"] == "result"
    assert obj["subtype"] == "success"
    assert obj["is_error"] is False
    assert obj["result"] == "answer"
    for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        assert k in obj["usage"]


def test_emit_result_passes_through_turns_and_duration():
    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("x", usage=None, session_id="s", num_turns=3, duration_ms=1234)
    obj = json.loads(buf.getvalue().strip())
    assert obj["num_turns"] == 3
    assert obj["duration_ms"] == 1234


def test_first_env_precedence(monkeypatch):
    for v in ("A", "B", "C"):
        monkeypatch.delenv(v, raising=False)
    assert har._first_env("A", "B", "C") is None
    monkeypatch.setenv("B", "vb")
    monkeypatch.setenv("C", "vc")
    assert har._first_env("A", "B", "C") == "vb"


def test_use_openai_provider_selection():
    # explicit prefixes win
    assert har._use_openai_provider("openai", None, None) is True
    assert har._use_openai_provider("anthropic", "http://x", None) is False
    # no prefix: only OPENAI_BASE_URL set -> openai; gh-aw (anthropic set) -> anthropic
    assert har._use_openai_provider("", "http://x", None) is True
    assert har._use_openai_provider("", None, "http://proxy") is False
    assert har._use_openai_provider("", None, None) is False


def test_emit_result_error_subtype():
    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("boom", usage=None, session_id="run-1", is_error=True)
    obj = json.loads(buf.getvalue().strip())
    assert obj["subtype"] == "error"
    assert obj["is_error"] is True


def test_emit_result_reads_usage_attributes():
    class U:
        input_tokens = 22
        output_tokens = 292

    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("x", usage=U(), session_id="s")
    usage = json.loads(buf.getvalue().strip())["usage"]
    assert usage["input_tokens"] == 22
    assert usage["output_tokens"] == 292


@pytest.mark.skipif(
    not os.environ.get("GH_AW_HARNESS_LIVE_API_KEY"),
    reason="set GH_AW_HARNESS_LIVE_API_KEY/_BASE_URL/_MODEL to run the live test",
)
def test_live_openai_compatible_endpoint(monkeypatch):
    """End-to-end against a real OpenAI-compatible endpoint, using the exact
    argv gh-aw passes. Credentials come from env only — never committed."""
    monkeypatch.setenv("OPENAI_API_KEY", os.environ["GH_AW_HARNESS_LIVE_API_KEY"])
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        os.environ.get("GH_AW_HARNESS_LIVE_BASE_URL", "https://example.test/v1"),
    )
    model = os.environ.get("GH_AW_HARNESS_LIVE_MODEL", "gpt-4o-mini")
    # Exercise the model path without the MCP gateway (not available outside a
    # gh-aw run): drop the --mcp-config <path> pair from the gh-aw argv.
    argv = list(GHAW_ARGV)
    i = argv.index("--mcp-config")
    del argv[i : i + 2]
    argv += ["--model", f"openai:{model}", "Reply with exactly: HARNESS_OK"]
    monkeypatch.setattr(sys, "argv", ["pydantic-ai-runner", *argv])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = har.main()
    assert rc == 0
    lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
    result = next(x for x in lines if x["type"] == "result")
    assert result["is_error"] is False
    assert "HARNESS_OK" in result["result"]
    assert result["usage"]["output_tokens"] > 0

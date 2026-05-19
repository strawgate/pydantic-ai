"""Offline tests for the Pydantic AI gh-aw harness (.github/scripts/pydantic-ai-runner).

These cover the gh-aw compatibility surface with no network or credentials:
argv tolerance, prompt recovery, model resolution, MCP-config translation and
allow-list filtering, Claude-named native tools, ``--allowed-tools`` /
``--permission-mode`` enforcement, structured-error guarantees, and the
stream-json schema.

The single live test is skipped unless an OpenAI-compatible endpoint is given
via env: GH_AW_HARNESS_LIVE_API_KEY / _BASE_URL / _MODEL.

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
    # Register in sys.modules before exec so dataclasses can resolve the
    # module (required with `from __future__ import annotations`).
    loader = importlib.machinery.SourceFileLoader("par_harness", str(_SCRIPT))
    spec = importlib.util.spec_from_loader("par_harness", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    loader.exec_module(mod)
    return mod


har = _load_harness()


# The exact argv shape gh-aw's claude_harness.cjs passes, prompt appended last.
GHAW_ARGV = [
    "--print",
    "--no-chrome",
    "--allowed-tools",
    "Bash,Read,Edit(/tmp/*),mcp__github__get_me,mcp__safeoutputs",
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


# --------------------------------------------------------------------------- #
# argv / prompt
# --------------------------------------------------------------------------- #
def test_parses_full_claude_argv_without_error():
    args = har.parse_args([*GHAW_ARGV, "do the thing"])
    assert args.mcp_config == "/tmp/mcp-servers.json"
    assert args.prompt_file == "/tmp/gh-aw/aw-prompts/prompt.txt"
    assert args.prompt_positional == "do the thing"
    assert args.permission_mode == "bypassPermissions"


def test_unknown_future_claude_flags_are_tolerated():
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
    assert har.resolve_prompt(har.parse_args(["--print"])) == "from env path"


# --------------------------------------------------------------------------- #
# --allowed-tools parsing & enforcement
# --------------------------------------------------------------------------- #
def test_allowed_tools_absent_is_none():
    assert har.parse_args(["--print"]).allowed_tools is None
    assert har._split_allowed_tools(None) is None


def test_allowed_tools_parsed_and_scope_stripped():
    args = har.parse_args([*GHAW_ARGV, "p"])
    assert args.allowed_tools == frozenset(
        {"Bash", "Read", "Edit", "mcp__github__get_me", "mcp__safeoutputs"}
    )


def test_select_native_tools_no_allowlist_keeps_all():
    tools = har.select_native_tools(None, None)
    assert [t.name for t in tools] == list(har.NATIVE_TOOL_NAMES)


def test_select_native_tools_enforces_allowlist():
    tools = har.select_native_tools(frozenset({"Bash", "Read", "mcp__safeoutputs"}), None)
    assert [t.name for t in tools] == ["Bash", "Read"]


def test_plan_mode_withholds_mutating_tools():
    tools = har.select_native_tools(None, "plan")
    names = {t.name for t in tools}
    assert names.isdisjoint(har.MUTATING_TOOLS)
    assert "Read" in names and "Grep" in names and "Glob" in names


def test_plan_mode_and_allowlist_compose():
    tools = har.select_native_tools(frozenset({"Bash", "Read"}), "plan")
    assert [t.name for t in tools] == ["Read"]  # Bash dropped by plan mode


def test_native_tools_registry_uses_claude_names():
    assert tuple(har.NATIVE_TOOLS) == har.NATIVE_TOOL_NAMES
    assert har.NATIVE_TOOL_NAMES == ("Bash", "Read", "Write", "Edit", "Grep", "Glob", "LS")


# --------------------------------------------------------------------------- #
# native tool behavior
# --------------------------------------------------------------------------- #
def test_native_file_tools_roundtrip(tmp_path):
    f = tmp_path / "sub" / "note.txt"
    assert "wrote" in har.write_file(str(f), "hello\nworld\n")
    assert har.read_file(str(f)) == "hello\nworld\n"
    assert "edited" in har.edit_file(str(f), "world", "there")
    assert "there" in har.read_file(str(f))
    assert "note.txt" in har.list_dir(str(tmp_path / "sub"))
    assert har.edit_file(str(f), "absent", "x") == "error: `old_string` not found"


def test_read_file_offset_and_limit(tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")
    assert har.read_file(str(f), offset=2, limit=2) == "l2\nl3"


def test_edit_file_replace_all(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("a a a", encoding="utf-8")
    har.edit_file(str(f), "a", "b", replace_all=True)
    assert f.read_text(encoding="utf-8") == "b b b"


def test_native_bash_tool():
    out = har.bash("echo hello-from-bash")
    assert "exit=0" in out and "hello-from-bash" in out


def test_native_grep_tool(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nNEEDLE here\n", encoding="utf-8")
    assert "NEEDLE here" in har.grep("NEEDLE", str(tmp_path))


def test_native_glob_tool(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "x" / "b.txt").write_text("", encoding="utf-8")
    res = har.glob_search("**/*.py", str(tmp_path))
    assert "x/a.py" in res and "b.txt" not in res


# --------------------------------------------------------------------------- #
# model resolution (proxy semantics — unchanged)
# --------------------------------------------------------------------------- #
def test_model_defaults_to_anthropic(monkeypatch):
    for v in (
        "GH_AW_HARNESS_MODEL",
        "GH_AW_MODEL_AGENT_CLAUDE",
        "ANTHROPIC_MODEL",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(v, raising=False)
    model, label = har.build_model(har.parse_args(["--print"]))
    assert label == "anthropic:claude-sonnet-4-5"
    assert model.__class__.__name__ == "AnthropicModel"


def test_model_openai_prefix_builds_openai_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    model, label = har.build_model(har.parse_args(["--model", "openai:gpt-4o-mini"]))
    assert label == "openai-compatible:gpt-4o-mini"
    assert model.__class__.__name__ == "OpenAIChatModel"


def test_anthropic_base_url_builds_anthropic_model(monkeypatch):
    for v in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GH_AW_HARNESS_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "placeholder")
    monkeypatch.setenv("GH_AW_HARNESS_MODEL", "MiniMax-M2.7-highspeed")
    model, label = har.build_model(har.parse_args(["--print"]))
    assert label == "anthropic:MiniMax-M2.7-highspeed"
    assert model.__class__.__name__ == "AnthropicModel"


def test_first_env_precedence(monkeypatch):
    for v in ("A", "B", "C"):
        monkeypatch.delenv(v, raising=False)
    assert har._first_env("A", "B", "C") is None
    monkeypatch.setenv("B", "vb")
    monkeypatch.setenv("C", "vc")
    assert har._first_env("A", "B", "C") == "vb"


def test_use_openai_provider_selection():
    assert har._use_openai_provider("openai", None, None) is True
    assert har._use_openai_provider("anthropic", "http://x", None) is False
    assert har._use_openai_provider("", "http://x", None) is True
    assert har._use_openai_provider("", None, "http://proxy") is False
    assert har._use_openai_provider("", None, None) is False


# --------------------------------------------------------------------------- #
# MCP translation & allow-list filtering
# --------------------------------------------------------------------------- #
def test_mcp_missing_config_degrades_gracefully():
    assert har.build_mcp_servers(har.Args(mcp_config="/no/such/file.json")) == []


def _mcp_cfg(tmp_path):
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
    return cfg


def test_mcp_translates_stdio_and_http_unfiltered(tmp_path):
    servers = har.build_mcp_servers(har.Args(mcp_config=str(_mcp_cfg(tmp_path))))
    assert len(servers) == 2
    assert {s.__class__.__name__ for s in servers} == {"MCPToolset"}


def test_mcp_wrapped_in_filter_when_allowlist_present(tmp_path):
    servers = har.build_mcp_servers(
        har.Args(mcp_config=str(_mcp_cfg(tmp_path)), allowed_tools=frozenset({"mcp__safeoutputs"}))
    )
    assert len(servers) == 2
    assert {s.__class__.__name__ for s in servers} == {"FilteredToolset"}


def test_mcp_allow_predicate_server_wildcard_vs_specific():
    class _TD:  # minimal stand-in for ToolDefinition (predicate only reads .name)
        def __init__(self, name):
            self.name = name

    # whole-server allow
    pred = har._mcp_tool_allowed("safeoutputs", frozenset({"mcp__safeoutputs"}))
    assert pred(None, _TD("add_comment")) is True
    assert pred(None, _TD("create_issue")) is True

    # specific-tool allow only
    pred = har._mcp_tool_allowed("github", frozenset({"mcp__github__get_me"}))
    assert pred(None, _TD("get_me")) is True
    assert pred(None, _TD("delete_repo")) is False


# --------------------------------------------------------------------------- #
# stream-json schema & structured-error guarantee
# --------------------------------------------------------------------------- #
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
    assert obj["num_turns"] == 3 and obj["duration_ms"] == 1234


def test_emit_result_error_subtype():
    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("boom", usage=None, session_id="run-1", is_error=True)
    obj = json.loads(buf.getvalue().strip())
    assert obj["subtype"] == "error" and obj["is_error"] is True


def test_emit_result_reads_usage_attributes():
    class U:
        input_tokens = 22
        output_tokens = 292

    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("x", usage=U(), session_id="s")
    usage = json.loads(buf.getvalue().strip())["usage"]
    assert usage["input_tokens"] == 22 and usage["output_tokens"] == 292


def test_main_emits_structured_error_on_empty_prompt(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pydantic-ai-runner", "--print"])
    monkeypatch.delenv("GH_AW_PROMPT", raising=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = har.main()
    assert rc == 1
    obj = json.loads(buf.getvalue().strip())
    assert obj["type"] == "result" and obj["is_error"] is True


def test_main_emits_structured_error_on_startup_failure(monkeypatch):
    # A failure *before* the agent loop (e.g. model build) must still produce a
    # parseable stream-json result, never an opaque "no entries" run.
    def boom(_args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(har, "build_model", boom)
    monkeypatch.setattr(sys, "argv", ["pydantic-ai-runner", "--print", "hello"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = har.main()
    assert rc == 1
    obj = json.loads(buf.getvalue().strip())
    assert obj["is_error"] is True
    assert "harness startup failed" in obj["result"]
    assert "kaboom" in obj["result"]


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
    argv = list(GHAW_ARGV)
    i = argv.index("--mcp-config")
    del argv[i : i + 2]  # no MCP gateway outside a gh-aw run
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

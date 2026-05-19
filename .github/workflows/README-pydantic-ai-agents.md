# Pydantic AI agentic workflows (gh-aw prototype)

A prototype of [gh-aw](https://github.com/githubnext/gh-aw) continuous-improvement
agents running on **Pydantic AI** instead of the Claude Code CLI, with prompts
iterable from **Logfire managed variables**. Combines:

- the Pydantic AI harness as a drop-in gh-aw engine (`engine.command`),
- runtime prompt fetch from a Logfire managed variable (OFREP),
- task designs adapted from `elastic/ai-github-actions` (bug-hunter, docs-patrol),

targeting this repo (Python `uv` workspace).

## Workflows

| Source (`.md`) | Trigger | Output |
| --- | --- | --- |
| `pydantic-ai-bug-hunter.md` | daily + manual | files a `[bug-hunter]` issue (or noop) |
| `pydantic-ai-docs-drift.md` | weekly + manual | files a `[docs-drift]` issue (or noop) |
| `pydantic-ai-pr-selftest.md` | PR touching the harness | comments harness health on the PR |

Each `.md` is the source of truth; the adjacent `.lock.yml` is the compiled
GitHub Actions workflow. The harness is `.github/scripts/pydantic-ai-runner`
(a `uv run --script` self-contained Pydantic AI agent).

## Required configuration

Repository **secrets**:

- `OPENAI_API_KEY` — key for the MiniMax Anthropic-compatible endpoint
  (`https://api.minimax.io/anthropic`); injected by gh-aw's AWF proxy.
- `LOGFIRE_WRITE_TOKEN` — Logfire project write token for OTLP trace export.
- `LOGFIRE_READ_EXTERNAL_VARIABLES_KEY` — *optional*; enables runtime prompt
  overrides. If unset, workflows run on the baked-in static prompts.

Repository **variables**:

- `GH_AW_HARNESS_MODEL` — model name passed to the harness (e.g. a MiniMax
  model id). Defaults to `claude-sonnet-4-5` inside the harness if unset.

## Iterating prompts from Logfire

Create these Logfire managed variables (string values), targeting key
`gh-aw-<owner>/<repo>`:

- `gh_aw_pydantic_ai_bug_hunter_prompt`
- `gh_aw_pydantic_ai_docs_drift_prompt`

Their content is appended to (and overrides) the static instructions in the
workflow body — no recompile or commit needed to tune the agent.

## Editing & recompiling

After editing any `.md`, recompile the lockfiles with the `gh aw` CLI:

```bash
gh extension install githubnext/gh-aw   # or build from source
GHAW_SHA=27f07dd9efd43d19f8b0f6928285d4efd97c3b12  # pin matching the compiler
gh aw compile --actions-repo github/gh-aw --action-tag "$GHAW_SHA" \
  pydantic-ai-bug-hunter pydantic-ai-docs-drift pydantic-ai-pr-selftest
```

`--action-tag` (a pinned `github/gh-aw` commit SHA) makes the lockfiles
reference `github/gh-aw/actions/setup@<sha>` as a normal **remote** composite
action. The default/`dev`/`script` modes instead emit a local `./actions/setup`
that only resolves in repos that vendor gh-aw's 4–11MB `actions/` bundle (the
gh-aw repo itself does; this one does not) — without it the runner's auto
"Post Setup Scripts" step fails. Pin the SHA to the gh-aw build used to
compile so the action code matches the generated YAML; bump it together with
the compiler. Commit the regenerated `*.lock.yml` — they are generated
artifacts (excluded from the whitespace pre-commit hooks and the
`secrets-outside-env` zizmor audit, like `uv.lock`).

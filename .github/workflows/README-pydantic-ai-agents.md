# Pydantic AI agentic workflows (gh-aw prototype)

A prototype of [gh-aw](https://github.com/githubnext/gh-aw) continuous-improvement
agents running on **Pydantic AI** instead of the Claude Code CLI, with prompts
iterable from **Logfire managed variables**. Combines:

- the Pydantic AI harness as a drop-in gh-aw engine (`engine.command`),
- runtime prompt fetch from a Logfire managed variable (OFREP),
- task designs grounded in the upstream pydantic-ai issue corpus (bug-hunter, docs-drift, provider-mapping/parity, round-trip, regression, streaming, PR review),

targeting this repo (Python `uv` workspace).

## Workflows

| Source (`.md`) | Trigger | Output |
| --- | --- | --- |
| `pydantic-ai-bug-hunter.md` | daily + manual | files a `[bug-hunter]` issue (or noop) |
| `pydantic-ai-docs-drift.md` | weekly + manual | files a `[docs-drift]` issue (or noop) |
| `pydantic-ai-provider-mapping-sweep.md` | daily + manual | files a `[provider-mapping-sweep]` issue (or noop) — rotates one provider/run |
| `pydantic-ai-roundtrip-sweep.md` | daily + manual | files a `[roundtrip-sweep]` issue (or noop) — serialize/deserialize state loss |
| `pydantic-ai-regression-detector.md` | weekly + manual | files a `[regression-detector]` issue (or noop) — old-passes/new-fails |
| `pydantic-ai-provider-parity-explore.md` | weekly + manual | files a `[provider-parity-explore]` issue (or noop) — rotates one capability/run |
| `pydantic-ai-streaming-resilience-sweep.md` | weekly + manual | files a `[streaming-resilience-sweep]` issue (or noop) |
| `pydantic-ai-pr-review.md` | PR opened/synchronize/ready_for_review + manual | inline review comments + a single `submit_pull_request_review` verdict (or noop) — uses read-only `Task` sub-agents for breadth; PR context is pre-fetched by `scripts/gather-review-context.sh` |
| `pydantic-ai-pr-selftest.md` | PR touching the harness | comments harness health on the PR |

Each `.md` is the source of truth; the adjacent `.lock.yml` is the compiled
GitHub Actions workflow. The harness is `.github/scripts/pydantic-ai-runner`
(a `uv run --script` self-contained Pydantic AI agent).

`pydantic-ai-agents-orchestrator.yml` is a plain (non-gh-aw) helper:
`workflow_dispatch` it to fan a dispatch out to every scheduled agent above,
sequentially with a short delay between each (input `delay-seconds`, 1–120,
default 30), so they don't all hit the model/API at once. It excludes the
PR-triggered self-test. Useful for an initial smoke run or a coordinated
on-demand sweep without waiting for the crons.

## Required configuration

Repository **secrets**:

- `ANTHROPIC_API_KEY` — key for the MiniMax Anthropic-compatible endpoint
  (`https://api.minimax.io/anthropic`); injected by gh-aw's AWF proxy.
- `LOGFIRE_WRITE_TOKEN` — Logfire project write token for OTLP trace export.
- `LOGFIRE_READ_EXTERNAL_VARIABLES` — *optional*; enables runtime prompt
  overrides. If unset, workflows run on the baked-in static prompts.

Repository **variables**:

- `MODEL` — model name passed to the harness (e.g. a MiniMax model id).
  Defaults to `claude-sonnet-4-5` inside the harness if unset.
- `LOGFIRE_URL` — *optional*; base URL for the dynamic-prompt OFREP fetch.
  May be set as a variable or secret. Defaults to
  `https://logfire-api.pydantic.dev`. Note: the **trace-export** endpoint in
  `shared/otel-logfire.md` is a separate baked compile-time literal (gh-aw
  adds its host to the AWF firewall allowlist at compile time, so it can't
  read a variable) — if you use a non-standard/self-hosted Logfire, that
  literal must be updated to match and the workflows recompiled.

## Iterating prompts from Logfire

Each agent's prompt is **the entire content** of a Logfire managed variable
(not a suffix appended to a baked-in prompt). Create these string variables,
targeting key `gh-aw-<owner>/<repo>`:

- `gh_aw_pydantic_ai_bug_hunter_prompt`
- `gh_aw_pydantic_ai_docs_drift_prompt`
- `gh_aw_pydantic_ai_provider_mapping_sweep_prompt`
- `gh_aw_pydantic_ai_roundtrip_sweep_prompt`
- `gh_aw_pydantic_ai_regression_detector_prompt`
- `gh_aw_pydantic_ai_provider_parity_explore_prompt`
- `gh_aw_pydantic_ai_streaming_resilience_sweep_prompt`
- `gh_aw_pydantic_ai_pr_review_prompt`

Seed each one with the corresponding committed default, which is the complete
canonical prompt:

- `.github/workflows/shared/prompts/pydantic-ai-bug-hunter.md`
- `.github/workflows/shared/prompts/pydantic-ai-docs-drift.md`
- `.github/workflows/shared/prompts/pydantic-ai-provider-mapping-sweep.md`
- `.github/workflows/shared/prompts/pydantic-ai-roundtrip-sweep.md`
- `.github/workflows/shared/prompts/pydantic-ai-regression-detector.md`
- `.github/workflows/shared/prompts/pydantic-ai-provider-parity-explore.md`
- `.github/workflows/shared/prompts/pydantic-ai-streaming-resilience-sweep.md`
- `.github/workflows/shared/prompts/pydantic-ai-pr-review.md`

(Paste the file content below its leading HTML comment.) Whatever the variable
holds **replaces** the prompt wholesale at run time — iterate freely in the
Logfire UI, no recompile or commit needed. If the variable is unset or Logfire
is unreachable, the workflow falls back to the committed default file so the
agent always receives a complete prompt. Keep the default files in sync with
the live variables as the reviewed baseline.

## Editing & recompiling

After editing any `.md`, recompile the lockfiles with the `gh aw` CLI:

```bash
gh extension install githubnext/gh-aw   # or build from source
GHAW_SHA=27f07dd9efd43d19f8b0f6928285d4efd97c3b12  # pin matching the compiler
gh aw compile --actions-repo github/gh-aw --action-tag "$GHAW_SHA" \
  pydantic-ai-bug-hunter pydantic-ai-docs-drift pydantic-ai-pr-selftest \
  pydantic-ai-provider-mapping-sweep pydantic-ai-roundtrip-sweep \
  pydantic-ai-regression-detector pydantic-ai-provider-parity-explore \
  pydantic-ai-streaming-resilience-sweep pydantic-ai-pr-review
# Then discard gh-aw's reformat of .github/dependabot.yml (it drops `version: 2`):
git checkout -- .github/dependabot.yml
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

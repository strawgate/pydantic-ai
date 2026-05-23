---
# Logfire OTLP observability shared import
# Exports gh-aw distributed traces (agent GenAI spans, setup/conclusion spans)
# to Pydantic Logfire via OTLP/HTTP.
#
# gh-aw POSTs OTLP/HTTP JSON to {endpoint}/v1/traces, so the endpoint below is
# the bare Logfire ingest base URL (no /v1/traces path). It MUST be a
# compile-time literal: gh-aw adds the parsed static hostname to the AWF
# network firewall allowlist at compile time, so it cannot be a ${{ vars.* }}
# expression. This is the EU Pydantic Logfire ingest URL; if you use the US
# region or a custom LOGFIRE_URL, update the literal below to match and recompile.
#
# Required secret:
#   LOGFIRE_WRITE_TOKEN — a Logfire project write token. Used as the
#   Authorization header value for OTLP ingest and passed directly to the agent
#   container so the Logfire Python SDK can also use it natively.
#
# Usage:
#   imports:
#     - shared/otel-logfire.md
observability:
  otlp:
    endpoint: https://logfire-eu.pydantic.dev
    headers:
      Authorization: ${{ secrets.LOGFIRE_WRITE_TOKEN }}
    if-missing: warn
---

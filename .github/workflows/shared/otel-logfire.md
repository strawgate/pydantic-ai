---
# Logfire OTLP observability shared import
# Exports gh-aw distributed traces (agent GenAI spans, setup/conclusion spans)
# to Pydantic Logfire via OTLP/HTTP.
#
# gh-aw POSTs OTLP/HTTP JSON to {endpoint}/v1/traces, so the endpoint below is
# the bare Logfire ingest base URL (no /v1/traces path). It MUST be a
# compile-time literal: gh-aw adds the parsed static hostname to the AWF
# network firewall allowlist at compile time, so it cannot be a ${{ vars.* }}
# expression. Update the literal below to match your Logfire region and recompile.
#
# Logfire regions:
#   US: https://logfire-us.pydantic.dev
#   EU: https://logfire-eu.pydantic.dev
#
# Required secret:
#   LOGFIRE_WRITE_TOKEN — a Logfire project write token (pylf_v1_us_... or
#   pylf_v1_eu_...). Used as the Authorization header value for OTLP ingest
#   and passed directly to the agent container so the Logfire Python SDK can
#   also use it natively. The token region must match the endpoint below.
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

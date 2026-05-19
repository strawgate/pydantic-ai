---
# Logfire OTLP observability shared import
# Exports gh-aw distributed traces (agent GenAI spans, setup/conclusion spans)
# to Pydantic Logfire via OTLP/HTTP.
#
# gh-aw POSTs OTLP/HTTP JSON to {endpoint}/v1/traces, so the endpoint below is
# the bare Logfire ingest base URL (no /v1/traces path). The static hostname is
# automatically added to the network firewall allowlist.
#
# Required secret:
#   LOGFIRE_WRITE_TOKEN — a Logfire project *write* token (distinct from the
#   read-variables key used for dynamic prompts). Used as the Authorization
#   header value for OTLP ingest.
#
# Usage:
#   imports:
#     - shared/otel-logfire.md
observability:
  otlp:
    endpoint: https://logfire-api.pydantic.dev
    headers:
      Authorization: ${{ secrets.LOGFIRE_WRITE_TOKEN }}
    if-missing: warn
---

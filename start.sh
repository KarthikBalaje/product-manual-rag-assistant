#!/bin/bash
set -euo pipefail

export ANONYMIZED_TELEMETRY="${ANONYMIZED_TELEMETRY:-FALSE}"
export STREAMLIT_SERVER_HEADLESS=true

exec streamlit run streamlit_app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true

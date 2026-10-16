#!/usr/bin/env bash
set -e

python api.py &
API_PID=$!

cleanup() {
    kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python -m streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true

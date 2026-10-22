#!/usr/bin/env bash
# PubMedQA evidence demo — port 8510 (avoids other Streamlit apps on 8501)
cd "$(dirname "$0")"
# shellcheck source=/dev/null
[ -f .venv/bin/activate ] && source .venv/bin/activate
exec streamlit run pubmedqa_evidence_demo.py --server.port 8510

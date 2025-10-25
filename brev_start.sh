#!/usr/bin/env bash
# brev_start.sh — one-liner to train (if needed) and start the API on Brev

set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train if no model export yet
if [ ! -f outputs/model_export.pkl ]; then
  echo "No model export found — training with default args..."
  python survival_hackathon.py --model cox --horizons 90 180 365
fi

# Serve
export MODEL_PATH="outputs/model_export.pkl"
uvicorn serve:app --host 0.0.0.0 --port 8000

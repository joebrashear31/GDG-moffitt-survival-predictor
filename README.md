# Moffitt Survival — Brev-Ready (Local + Cloud)

Research-only survival probabilities at clinical time horizons.
- Local train/test folders or Hugging Face fallback.
- Trains Cox (and optional RSF).
- Exports model for FastAPI serving.
- Runs on NVIDIA Brev or locally.

## Local Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python survival_hackathon.py --data-dir "~/Desktop/moffitt hackathon/hackathon" --model cox --horizons 90 180 365
uvicorn serve:app --host 0.0.0.0 --port 8000

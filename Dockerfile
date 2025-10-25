# Dockerfile
FROM python:3.11-slim

# System deps (build tools; if scikit-survival needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    liblapack-dev libblas-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY survival_hackathon.py serve.py /app/

# Expose FastAPI port
EXPOSE 8000

# Default command: start API (expects model at outputs/model_export.pkl)
ENV MODEL_PATH=/app/outputs/model_export.pkl
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

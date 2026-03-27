# ---------------------------------------------------------------
# SupportEnv — Hugging Face Spaces Dockerfile
# Space SDK: Docker  |  Port: 7860
# ---------------------------------------------------------------
FROM python:3.11-slim

# HF Spaces requirement: non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=appuser:appuser . .

USER appuser

# Expose the port HF Spaces expects
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
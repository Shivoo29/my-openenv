# ---------------------------------------------------------------
# DevOpsEnv — Hugging Face Spaces Docker container
# Space SDK: Docker  |  Port: 7860
# ---------------------------------------------------------------
FROM python:3.11-slim

# Install system utilities for DevOps tasks
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    docker.io \
    systemctl \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# HF Spaces compatibility
RUN chmod +x /app/app.py 2>/dev/null || true

USER appuser

# Expose the port HF Spaces expects
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--reload"]
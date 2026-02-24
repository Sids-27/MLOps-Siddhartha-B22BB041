# ============================================================
# Dockerfile — Training & Development Environment
# ============================================================
# Base: Official Python 3.10 slim image (Debian Bookworm)
# Includes all dependencies needed to download data, train
# the DistilBERT model, and evaluate it.
# ============================================================

FROM python:3.10-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Default command: run the training pipeline
CMD ["python", "src/train.py"]

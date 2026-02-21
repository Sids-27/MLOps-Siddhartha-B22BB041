# Use Python 3.9
FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY train.py .
COPY evaluate.py .

# Copy model file
COPY setA.pth .

# Create data directories
RUN mkdir -p data/train data/test

# Copy data directories
COPY data/ ./data/

# Set environment variables
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]

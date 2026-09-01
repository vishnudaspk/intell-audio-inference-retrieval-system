# Dockerfile for Intell Audio Intelligence Platform Backend
FROM python:3.11-slim

# Install system audio dependencies (ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source tree
COPY . .

# Expose FastAPI backend port
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Use NVIDIA CUDA base for GPU inference support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set non-interactive install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (OpenCV needs libgl1 and libglib2.0)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up Python symlink
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create working directory
WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Default command for API
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

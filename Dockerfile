
# Use lightweight Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and Poppler (for PDF)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to keep image size down
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version explicitly if needed (optional optimization)
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set enviroment variables
ENV PYTHONUNBUFFERED=1
ENV HEADLESS=true

# Command to run the app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

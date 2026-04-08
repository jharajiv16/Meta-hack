FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    rsync \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Gradio/FastAPI uses
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

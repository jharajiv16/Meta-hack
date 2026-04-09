FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    rsync \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock requirements.txt ./
COPY server/ ./server/
COPY inference.py ./
COPY openenv.yaml ./
COPY README.md ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 7860

# Launch the OpenEnv-compliant server (server is a package, relative imports work)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]


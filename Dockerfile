# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TRANSFORMERS_CACHE=/tmp/transformers_cache \
    HF_HOME=/tmp/hf_home

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install Python dependencies
RUN uv pip install --system -e .

# Install insanely-fast-whisper
RUN uv pip install --system insanely-fast-whisper

# Optional: Pre-download Whisper model (comment out if not needed)
# RUN python3 -c "from transformers import AutoModelForSpeechSeq2Seq; AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uv", "run", "mcp", "dev", "src/whisper_mcp/server.py:create_server"]

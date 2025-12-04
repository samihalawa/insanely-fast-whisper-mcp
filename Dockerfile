FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install dependencies
RUN uv pip install --system .

# Set environment
ENV PYTHONUNBUFFERED=1

# Run the MCP server
CMD ["python", "-m", "insanely_fast_whisper_mcp"]

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package itself
RUN pip install --no-cache-dir -e .

# Set environment
ENV PYTHONUNBUFFERED=1

# Run the MCP server
CMD ["python", "-m", "insanely_fast_whisper_mcp"]

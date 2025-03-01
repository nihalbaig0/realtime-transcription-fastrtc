# Stage 1: Get uv installer
FROM ghcr.io/astral-sh/uv:0.2.12 as uv

# Stage 2: Main application image
FROM python:3.10.12-slim-bookworm

# Copy uv from first stage
COPY --from=uv /uv /uv

# Create virtual environment with uv
RUN --mount=type=cache,target=/root/.cache/uv \
    /uv venv /opt/venv

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages with uv caching
RUN --mount=type=cache,target=/root/.cache/uv \
    /uv pip install -r requirements.txt

# Copy application code
COPY . .

# Expose FastRTC port
EXPOSE 8000

# Start the application
CMD ["python", "main.py"]
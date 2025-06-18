# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in stages
COPY requirements.txt .

# Install dependencies in stages to handle conflicts
RUN pip install --upgrade pip && \
    # Install base dependencies first
    pip install --no-cache-dir setuptools wheel && \
    # Install transformers with its required tokenizers version
    pip install --no-cache-dir "tokenizers>=0.14,<0.19" && \
    pip install --no-cache-dir transformers==4.37.2 && \
    # Install core dependencies
    pip install --no-cache-dir fastapi uvicorn sqlalchemy psycopg2-binary redis && \
    # Install remaining dependencies
    pip install --no-cache-dir -r requirements.txt && \
    # Install crewai with compatible tokenizers
    pip install --no-cache-dir --no-deps crewai==0.130.0

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs

# Security scanning
RUN pip install safety bandit && \
    safety check && \
    bandit -r src/ -f json -o bandit-results.json || true

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 
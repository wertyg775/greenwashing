# Stage 1: build environment
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Use a specific directory for the prefix to keep the runtime clean
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: runtime environment
FROM python:3.10-slim

# Set environment variables for Python and Transformers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/tmp/.cache

WORKDIR /app

# Create a non-root user for security (required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy the large model first (to utilize layer caching)
COPY --chown=user:user ./models ./models

# Copy comparison images
COPY --chown=user:user ./image/comparison ./image/comparison

# Copy code and frontend
COPY --chown=user:user app.py greenwashing_analyzer.py ./ 
COPY --chown=user:user frontend ./frontend

# Expose the port (Note: HF Spaces uses 7860 by default)
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
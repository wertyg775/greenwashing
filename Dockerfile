# -----------------------------
# Dockerfile for Greenwashing FYP
# Python 3.10 + FastAPI + Torch + Transformers
# -----------------------------

# Stage 1: build environment
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements and install to a local directory
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: runtime environment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy backend app, model, and frontend
COPY ./models ./models 
COPY app.py greenwashing_analyzer.py ./
COPY frontend ./frontend

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

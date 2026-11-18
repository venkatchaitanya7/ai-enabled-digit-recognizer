# Multi-stage build for smaller final image
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY . .

# Train model (creates digit_model.pth and digit_model.onnx)
RUN python train.py


FROM python:3.11-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local
# Copy application code and trained models
COPY --from=builder /app/digit_model.pth .
COPY --from=builder /app/digit_model.onnx .
COPY --from=builder /app/app.py .
COPY --from=builder /app/model.py .
COPY --from=builder /app/templates ./templates

# Set PATH to include user packages
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run with gunicorn
CMD ["gunicorn", "app:app", "--workers", "2", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:8000"]
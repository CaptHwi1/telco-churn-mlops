# docker/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy frontend
COPY frontend/ ./frontend/

# Copy artifacts (models, preprocessing)
COPY artifacts/ ./artifacts/

# Configure nginx
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Create directories with proper permissions
RUN mkdir -p /app/logs /app/tmp /var/log/nginx /var/lib/nginx/body /var/run && \
    chmod -R 755 /app /var/log/nginx /var/lib/nginx /var/run /app/tmp

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ARTIFACTS_PATH=/app/artifacts \
    TMPDIR=/app/tmp

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

CMD ["sh", "-c", "gunicorn src.serving.api:app --bind 127.0.0.1:8000 --workers 2 --worker-class uvicorn.workers.UvicornWorker --access-logfile /app/tmp/gunicorn_access.log --error-logfile /app/tmp/gunicorn_error.log & nginx -g 'daemon off;'"]
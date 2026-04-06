# docker/Dockerfile (with nginx, but NO config copy)
FROM python:3.10-slim

WORKDIR /app

# Install nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx curl \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY artifacts/ ./artifacts/

# ✅ FIX: Modify default nginx.conf in-place (DON'T copy over it)
RUN echo 'server { \
    listen 80; \
    server_name localhost; \
    location / { root /app/frontend; try_files $uri $uri/ /index.html; } \
    location /api/ { proxy_pass http://127.0.0.1:8000/; proxy_set_header Host $host; } \
    location /health { proxy_pass http://127.0.0.1:8000/health; } \
}' > /etc/nginx/conf.d/default.conf && \
    sed -i 's/daemon off;/# daemon off;/' /etc/nginx/nginx.conf

# Create directories
RUN mkdir -p /app/logs /app/tmp /var/log/nginx /var/lib/nginx/body /var/run && \
    chmod -R 755 /app /var/log/nginx /var/lib/nginx /var/run /app/tmp

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ARTIFACTS_PATH=/app/artifacts \
    TMPDIR=/app/tmp

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

CMD ["sh", "-c", "gunicorn src.serving.api:app --bind 127.0.0.1:8000 --workers 2 --worker-class uvicorn.workers.UvicornWorker & nginx -g 'daemon off;'"]
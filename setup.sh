#!/bin/bash

# =============================================================================
# Telco Churn MLOps - Project Setup Script
# =============================================================================
# This script creates the complete project structure with all necessary files
# Run from the project root directory
# =============================================================================

set -e  # Exit on error

echo "🚀 Starting Telco Churn MLOps Project Setup..."
echo "============================================================"

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# =============================================================================
# 1. Create Directory Structure
# =============================================================================
echo "📁 Creating directory structure..."

mkdir -p src/data
mkdir -p src/features
mkdir -p src/models
mkdir -p src/serving
mkdir -p notebooks/experiments
mkdir -p notebooks/exploration
mkdir -p frontend
mkdir -p docker
mkdir -p k8s
mkdir -p .github/workflows
mkdir -p artifacts
mkdir -p models
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs
mkdir -p tests

# Create .gitkeep files to preserve empty directories in git
touch artifacts/.gitkeep
touch models/.gitkeep
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch logs/.gitkeep

echo "✅ Directory structure created"

# =============================================================================
# 2. Create Python Package Files
# =============================================================================
echo "📦 Creating Python package files..."

# src/__init__.py
cat > src/__init__.py << 'EOF'
"""Telco Churn MLOps Project Package."""

__version__ = "1.0.0"
EOF

# src/data/__init__.py
cat > src/data/__init__.py << 'EOF'
"""Data loading and validation module."""

from src.data.validation import validate_data_quality, validate_raw_data

__all__ = [
    'validate_data_quality',
    'validate_raw_data'
]
EOF

# src/features/__init__.py
cat > src/features/__init__.py << 'EOF'
"""Feature engineering module for Telco Churn prediction."""

from src.features.preprocessing import (
    standardize_columns,
    fit_preprocessing_pipeline,
    transform_data,
    save_preprocessing_artifacts,
    load_preprocessing_artifacts,
    TargetEncoder
)

__all__ = [
    'standardize_columns',
    'fit_preprocessing_pipeline',
    'transform_data',
    'save_preprocessing_artifacts',
    'load_preprocessing_artifacts',
    'TargetEncoder'
]
EOF

# src/models/__init__.py
cat > src/models/__init__.py << 'EOF'
"""Model training and evaluation module."""

from src.models.evaluate import calculate_business_metrics

__all__ = [
    'calculate_business_metrics'
]
EOF

# src/serving/__init__.py
cat > src/serving/__init__.py << 'EOF'
"""Serving module for Telco Churn API."""

from src.serving.api import app

__all__ = ['app']
EOF

echo "✅ Python package files created"

# =============================================================================
# 3. Create Configuration Files
# =============================================================================
echo "⚙️  Creating configuration files..."

# .env (template - user should customize)
cat > .env << 'EOF'
# =============================================================================
# Telco Churn MLOps - Environment Configuration
# Copy this file to .env and update values for your environment
# =============================================================================

# Project Settings
PROJECT_NAME=telco_churn
ENVIRONMENT=dev
DEBUG=true

# Paths (relative to project root)
ARTIFACTS_PATH=artifacts
DATA_RAW_PATH=data/raw
DATA_PROCESSED_PATH=data/processed
MODELS_PATH=models

# MLflow Settings
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=telco_churn_baseline
MODEL_REGISTRY_NAME=telco_churn_model

# Model Settings
RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# Huawei Cloud Settings (Optional - for deployment)
# HUAWEI_REGION=cn-north-4
# HUAWEI_PROJECT_ID=your_project_id_here
# SWR_REGISTRY=swr.cn-north-4.myhuaweicloud.com
# CCE_CLUSTER_NAME=telco-churn-cluster
EOF

# .gitignore
cat > .gitignore << 'EOF'
# =============================================================================
# Telco Churn MLOps - Git Ignore Rules
# =============================================================================

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/
mlops_env/
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb_checkpoints
*.nbconvert.ipynb
*.pynb
*.nbconvert.html
*.nbconvert.pdf

# IDE & Editors
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store
Thumbs.db
.project
.pydevproject
.settings/

# Logs & Temporary Files
*.log
logs/
tmp/
temp/
*.tmp
*.bak
*.cache

# MLflow
mlruns/
mlflow_runs/
*.mlflow
mlflow.db
mlflow_artifacts/

# Artifacts (DO NOT COMMIT - Generated at runtime)
artifacts/*.joblib
artifacts/*.pkl
artifacts/*.json
artifacts/*.png
!artifacts/.gitkeep

# Models
models/*.pkl
models/*.joblib
!models/.gitkeep

# Data (DO NOT COMMIT - Sensitive/Large)
data/raw/*.csv
data/raw/*.parquet
data/raw/*.feather
data/processed/
data/interim/
!data/raw/.gitkeep
!data/processed/.gitkeep

# Environment & Secrets (CRITICAL - NEVER COMMIT)
.env
.env.local
.env.*.local
*.env
secrets/
credentials/
*.pem
*.key
kubeconfig
kubeconfig.json
*.kubeconfig

# Docker
*.dockerfile.local
docker-compose.override.yml

# Kubernetes
k8s/*secret*.yaml
k8s/*-secret.yaml
*.kubeconfig
*.kubeconfig.json

# GitHub Actions
.github/workflows/local-*.yml

# Documentation Build
docs/_build/
docs/.doctrees/

# Testing
.coverage
coverage.xml
htmlcov/
.pytest_cache/
nosetests.xml

# OS Files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOF

echo "✅ Configuration files created"

# =============================================================================
# 4. Create Requirements Files
# =============================================================================
echo "📋 Creating requirements files..."

# requirements.txt
cat > requirements.txt << 'EOF'
# =============================================================================
# Telco Churn MLOps - Python Dependencies
# Generated for Docker & CI/CD compatibility
# =============================================================================

# Core Data Science
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Machine Learning
xgboost>=3.2.0
optuna>=4.8.0

# Experiment Tracking
mlflow>=2.10.0

# API & Serving
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
gunicorn>=21.0.0

# Configuration & Validation
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Utilities
joblib>=1.3.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Data Validation (optional, for serving)
pandera>=0.18.0

# Jupyter (for notebooks)
jupyter>=1.0.0
ipykernel>=6.0.0

# HTTP Client (for testing)
requests>=2.31.0

# Environment Variables
python-dotenv>=1.0.0
EOF

# pyproject.toml
cat > pyproject.toml << 'EOF'
# =============================================================================
# Telco Churn MLOps - Project Configuration
# =============================================================================

[tool.poetry]
name = "telco-churn-mlops"
version = "1.0.0"
description = "End-to-end MLOps pipeline for Telco Churn prediction with Huawei Cloud deployment"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
repository = "https://github.com/your-username/telco-churn-mlops"
keywords = ["mlops", "churn", "xgboost", "fastapi", "huawei-cloud"]

[tool.poetry.dependencies]
python = "^3.10"

# Core Data Science
pandas = "^2.0.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"

# Machine Learning
xgboost = "^3.2.0"
optuna = "^4.8.0"

# Experiment Tracking
mlflow = "^2.10.0"

# API & Serving
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
gunicorn = "^21.0.0"

# Configuration & Validation
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"

# Utilities
joblib = "^1.3.0"

# Visualization
matplotlib = "^3.8.0"
seaborn = "^0.13.0"

# Data Validation (optional)
pandera = {version = "^0.18.0", optional = true}

# Jupyter (for notebooks)
jupyter = "^1.0.0"
ipykernel = "^6.0.0"

[tool.poetry.extras]
serve = ["pandera"]
dev = ["jupyter", "ipykernel"]

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.12.0"
isort = "^5.13.0"
flake8 = "^7.0.0"
mypy = "^1.8.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

# =============================================================================
# Tool Configurations
# =============================================================================

[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
skip = [".venv", "build", "dist"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --tb=short"
EOF

echo "✅ Requirements files created"

# =============================================================================
# 5. Create Docker Files
# =============================================================================
echo "🐳 Creating Docker files..."

# docker/Dockerfile
cat > docker/Dockerfile << 'EOF'
# =============================================================================
# TELCO CHURN MLOPS - PRODUCTION DOCKERFILE
# Multi-stage build for minimal production image
# Designed for Huawei Cloud CCE deployment
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency files first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# -----------------------------------------------------------------------------
# Stage 2: Production
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    ARTIFACTS_PATH="/app/artifacts"

# Copy application code
COPY src/ ./src/

# Copy frontend
COPY frontend/ ./frontend/

# Copy artifacts (models, preprocessing)
COPY artifacts/ ./artifacts/

# Configure nginx to serve frontend and proxy API
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data/raw /app/data/processed /app/tmp \
    /var/log/nginx /var/lib/nginx/body /var/run && \
    chown -R appuser:appuser /app /var/log/nginx /var/lib/nginx /var/run /app/tmp

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    TMPDIR="/app/tmp"

# Run with gunicorn + nginx
CMD ["sh", "-c", "gunicorn src.serving.api:app --bind 127.0.0.1:8000 --workers 2 --worker-class uvicorn.workers.UvicornWorker --worker-tmp-dir /app/tmp --chdir /app --access-logfile /app/tmp/gunicorn_access.log --error-logfile /app/tmp/gunicorn_error.log & nginx -g 'daemon off;'"]
EOF

# docker/nginx.conf
cat > docker/nginx.conf << 'EOF'
# =============================================================================
# NGINX CONFIGURATION FOR TELCO CHURN API
# Serves frontend static files + proxies API requests to Gunicorn
# =============================================================================

worker_processes auto;
daemon off;
error_log /app/tmp/nginx_error.log warn;
pid /app/tmp/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Access log to writable location
    access_log /app/tmp/nginx_access.log;
    
    sendfile on;
    keepalive_timeout 65;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;

    server {
        listen 80;
        server_name localhost;

        # Serve frontend static files
        location / {
            root /app/frontend;
            index index.html;
            try_files $uri $uri/ /index.html;
        }

        # Proxy API requests to FastAPI backend
        location /api/ {
            proxy_pass http://127.0.0.1:8000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # CORS headers
            add_header Access-Control-Allow-Origin *;
            add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
            add_header Access-Control-Allow-Headers 'Content-Type';
            
            # Handle preflight requests
            if ($request_method = 'OPTIONS') {
                add_header Access-Control-Allow-Origin *;
                add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
                add_header Access-Control-Allow-Headers 'Content-Type';
                add_header Content-Length 0;
                add_header Content-Type text/plain;
                return 204;
            }
        }

        # Health check endpoint
        location /health {
            proxy_pass http://127.0.0.1:8000/health;
            proxy_set_header Host $host;
        }

        # Model info endpoint
        location /model/info {
            proxy_pass http://127.0.0.1:8000/model/info;
            proxy_set_header Host $host;
        }
    }
}
EOF

echo "✅ Docker files created"

# =============================================================================
# 6. Create Kubernetes Files
# =============================================================================
echo "☸️  Creating Kubernetes files..."

# k8s/namespace.yaml
cat > k8s/namespace.yaml << 'EOF'
# =============================================================================
# KUBERNETES NAMESPACE FOR TELCO CHURN
# =============================================================================

apiVersion: v1
kind: Namespace
metadata:
  name: telco-churn
  labels:
    app: telco-churn
    environment: production
    team: data-science
  annotations:
    description: "Namespace for Telco Churn Prediction Application"
    owner: "mlops-team"
EOF

# k8s/configmap.yaml
cat > k8s/configmap.yaml << 'EOF'
# =============================================================================
# KUBERNETES CONFIGMAP FOR TELCO CHURN
# Application configuration (non-sensitive)
# =============================================================================

apiVersion: v1
kind: ConfigMap
metadata:
  name: telco-churn-config
  namespace: telco-churn
  labels:
    app: telco-churn
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  ARTIFACTS_PATH: "/app/artifacts"
  # MLFLOW_TRACKING_URI: "http://mlflow-service:5000"
  PYTHONUNBUFFERED: "1"
  PYTHONDONTWRITEBYTECODE: "1"
EOF

# k8s/deployment.yaml
cat > k8s/deployment.yaml << 'EOF'
# =============================================================================
# KUBERNETES DEPLOYMENT FOR TELCO CHURN API
# Huawei Cloud CCE Compatible
# =============================================================================

apiVersion: apps/v1
kind: Deployment
metadata:
  name: telco-churn-api
  namespace: telco-churn
  labels:
    app: telco-churn-api
    version: v1
    environment: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: telco-churn-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: telco-churn-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      imagePullSecrets:
      - name: swr-secret
      
      containers:
      - name: api
        image: swr.cn-north-4.myhuaweicloud.com/telco-churn/telco-churn-api:latest
        imagePullPolicy: Always
        
        ports:
        - containerPort: 80
          name: http
          protocol: TCP
        
        envFrom:
        - configMapRef:
            name: telco-churn-config
        
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        volumeMounts:
        - name: tmp-volume
          mountPath: /app/tmp
        - name: log-volume
          mountPath: /app/logs
      
      volumes:
      - name: tmp-volume
        emptyDir: {}
      - name: log-volume
        emptyDir: {}
      
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - telco-churn-api
              topologyKey: kubernetes.io/hostname
      
      restartPolicy: Always
      terminationGracePeriodSeconds: 30
EOF

# k8s/service.yaml
cat > k8s/service.yaml << 'EOF'
# =============================================================================
# KUBERNETES SERVICE FOR TELCO CHURN API
# Huawei Cloud ELB Configuration
# =============================================================================

apiVersion: v1
kind: Service
metadata:
  name: telco-churn-api
  namespace: telco-churn
  labels:
    app: telco-churn-api
  annotations:
    kubernetes.io/elb.class: union
    kubernetes.io/elb.autocreate: |
      {
        "type": "public",
        "bandwidth_charge_mode": "traffic",
        "bandwidth_size": 10,
        "share_type": "PER",
        "ip_type": "5_bgp",
        "eip_type": "5_bgp"
      }
    kubernetes.io/elb.health-check: |
      {
        "protocol": "HTTP",
        "path": "/health",
        "delay": 5,
        "timeout": 3,
        "max_retries": 3
      }
spec:
  type: LoadBalancer
  selector:
    app: telco-churn-api
  ports:
  - name: http
    port: 80
    targetPort: 80
    protocol: TCP
  sessionAffinity: None
EOF

echo "✅ Kubernetes files created"

# =============================================================================
# 7. Create GitHub Actions Workflow
# =============================================================================
echo "🔄 Creating GitHub Actions workflow..."

# .github/workflows/deploy.yml
cat > .github/workflows/deploy.yml << 'EOF'
# =============================================================================
# GitHub Actions CI/CD Pipeline for Telco Churn MLOps
# Builds Docker image and pushes to Huawei Cloud SWR
# =============================================================================

name: Build & Deploy to Huawei SWR

on:
  push:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'artifacts/**'
      - 'docker/**'
      - 'k8s/**'
      - 'requirements.txt'
      - '.github/workflows/deploy.yml'
  workflow_dispatch:

env:
  REGISTRY: swr.cn-north-4.myhuaweicloud.com
  ORG: telco-churn
  APP: telco-churn-api
  REGION: cn-north-4

jobs:
  test:
    name: Validate & Test
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Validate artifacts exist
        run: |
          echo "🔍 Checking required artifacts..."
          test -f artifacts/preprocessing_pipeline.joblib || exit 1
          test -f artifacts/model.pkl || exit 1
          test -f artifacts/serving_config.json || exit 1
          echo "✅ All artifacts present"
      
      - name: Test API imports
        run: |
          python -c "from src.serving.api import app; print('✅ API imports OK')"
          python -c "from src.config import settings; print('✅ Config loads OK')"

  build-and-push:
    name: Build & Push to Huawei SWR
    runs-on: ubuntu-latest
    needs: test
    permissions:
      contents: read
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Huawei SWR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ secrets.HUAWEI_SWR_USERNAME }}
          password: ${{ secrets.HUAWEI_SWR_PASSWORD }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.ORG }}/${{ env.APP }}
          tags: |
            type=sha,prefix=sha-
            type=raw,value=latest,enable={{is_default_branch}}
            type=raw,value=prod,enable={{is_default_branch}}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            BUILD_DATE=${{ github.event.head_commit.timestamp }}
            VCS_REF=${{ github.sha }}
      
      - name: Deployment summary
        run: |
          echo "✅ Image pushed successfully!"
          echo "📦 Image: ${{ env.REGISTRY }}/${{ env.ORG }}/${{ env.APP }}:${{ steps.meta.outputs.version }}"
          echo "🔗 SWR Console: https://console.huaweicloud.com/swr/"
EOF

echo "✅ GitHub Actions workflow created"

# =============================================================================
# 8. Create README.md
# =============================================================================
echo "📖 Creating README.md..."

cat > README.md << 'EOF'
# 📊 Telco Churn MLOps Project

End-to-end MLOps pipeline for customer churn prediction with Huawei Cloud deployment.

## 🎯 Project Overview

This project implements a complete MLOps pipeline for predicting customer churn in telecommunications:

- **Data Processing**: Automated preprocessing pipeline with sklearn
- **Model Training**: XGBoost classifier with hyperparameter tuning (Optuna)
- **Experiment Tracking**: MLflow for tracking experiments and models
- **API Serving**: FastAPI REST API for real-time predictions
- **Containerization**: Docker for portable deployment
- **Orchestration**: Kubernetes (Huawei Cloud CCE) for production deployment
- **CI/CD**: GitHub Actions for automated builds and deployments

## 📁 Project Structure

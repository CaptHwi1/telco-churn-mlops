# 📊 Telco Churn Prediction MLOps Platform

[![Build & Deploy](https://github.com/your-username/telco-churn-mlops/actions/workflows/deploy.yml/badge.svg)](https://github.com/CaptHwi1/telco-churn-mlops/actions)
[![API Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/your-username/telco-churn-mlops/releases)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade MLOps platform for predicting customer churn with explainable AI, deployed on Huawei Cloud.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)
- [Common Mistakes to Avoid](#common-mistakes-to-avoid)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project provides an end-to-end MLOps solution for predicting customer churn in telecommunications. It combines **machine learning best practices** with **production-grade deployment** on Huawei Cloud, featuring:

- ✅ **Explainable AI**: Understand *why* customers are at risk with detailed factor analysis
- ✅ **Real-time Predictions**: FastAPI-based REST API with <100ms response time
- ✅ **Batch Processing**: Process up to 100 predictions per request
- ✅ **Production Deployment**: Docker + Kubernetes on Huawei Cloud CCE
- ✅ **Automated CI/CD**: GitHub Actions for automated builds and deployments
- ✅ **Model Versioning**: Track experiments with MLflow
- ✅ **Scalable Architecture**: Auto-scaling, load balancing, and high availability

### Business Impact

- **Reduce Churn**: Identify at-risk customers 30+ days before they leave
- **Actionable Insights**: Personalized recommendations for retention teams
- **Cost-Effective**: Automated model retraining and deployment
- **Transparent Decisions**: Explainable predictions build trust with stakeholders

---

## ✨ Features

### 🔮 Prediction Capabilities

| Feature | Description |
|---------|-------------|
| **Real-time API** | RESTful API for instant churn predictions |
| **Batch Processing** | Process up to 100 customers per request |
| **Explainable AI** | Detailed risk factors and personalized recommendations |
| **Confidence Scores** | Model confidence metrics for every prediction |
| **Risk Levels** | Categorized as Low/Medium/High with actionable insights |

### 🛠️ MLOps Features

| Feature | Tool/Technology |
|---------|----------------|
| **Experiment Tracking** | MLflow for model versioning and metrics |
| **Hyperparameter Tuning** | Optuna for automated optimization |
| **Model Registry** | Huawei SWR for containerized models |
| **CI/CD** | GitHub Actions for automated deployments |
| **Containerization** | Docker for consistent environments |
| **Orchestration** | Kubernetes (Huawei CCE) for scaling |

### 📊 Model Performance

| Metric | Baseline | Tuned Model |
|--------|----------|-------------|
| **Recall** | 0.978 | **0.994** (+1.6%) |
| **Precision** | 0.438 | 0.379 |
| **F1 Score** | 0.605 | 0.549 |
| **AUC-ROC** | 0.910 | **0.923** (+1.3%) |
| **Inference Time** | - | **<100ms** (p95) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (Web Frontend / API)                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Huawei Cloud ELB (Load Balancer)                │
│                   Public IP: 190.92.238.236                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Kubernetes Cluster (Huawei CCE)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              telco-churn-api Deployment              │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │   Pod 1    │  │   Pod 2    │  │   Pod 3    │    │  │
│  │  │ Gunicorn   │  │ Gunicorn   │  │ Gunicorn   │    │  │
│  │  │ FastAPI    │  │ FastAPI    │  │ FastAPI    │    │  │
│  │  │ :8000      │  │ :8000      │  │ :8000      │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  │         ReplicaSet with Auto-Scaling (2-5 pods)     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   Service (ClusterIP)                 │  │
│  │              Internal: 10.247.224.94                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Logic                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Preprocessing│  │   XGBoost    │  │   Explain    │      │
│  │   Pipeline   │  │    Model     │  │   Engine     │      │
│  │              │  │  (v2.0.0)    │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage & Artifacts                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Model.pkl  │  │  Preprocess  │  │    Config    │      │
│  │              │  │   Pipeline   │  │     JSON     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘

External Services:
┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Huawei SWR     │    │   GitHub Actions │    │   MLflow     │
│ (Model Registry) │    │   (CI/CD)        │    │ (Tracking)   │
└──────────────────┘    └──────────────────┘    └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker Desktop (for local development)
- kubectl (for Kubernetes deployment)
- Huawei Cloud account (for production deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/telco-churn-mlops.git
cd telco-churn-mlops
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv mlops_env

# Activate environment
source mlops_env/bin/activate  # Linux/Mac
# or
mlops_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

### 4. Run Locally

```bash
# Start the API server
python -m src.serving.api

# Or use uvicorn directly
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "contract": "Month-to-Month",
    "tenure_in_months": 6,
    "monthly_charge": 95.0,
    "internet_service": "Fiber Optic",
    "online_security": "No"
  }' | jq
```

### 6. Access Web UI

Open your browser to: **http://localhost:8000**

---

## 📥 Installation

### Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/telco-churn-mlops.git
cd telco-churn-mlops

# 2. Create and activate virtual environment
python -m venv mlops_env
source mlops_env/bin/activate  # Linux/Mac
# or
mlops_env\Scripts\activate  # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install development dependencies (optional)
pip install pytest pytest-cov black flake8 mypy

# 5. Verify installation
python -c "from src.serving.api import app; print('✅ Installation successful')"
```

### Docker Setup

```bash
# Build Docker image
docker build -t telco-churn-app:latest -f docker/Dockerfile .

# Run container
docker run -d --name telco-churn \
  -p 8000:8000 \
  -e ARTIFACTS_PATH=/app/artifacts \
  telco-churn-app:latest

# Test
curl http://localhost:8000/health

# Stop container
docker stop telco-churn && docker rm telco-churn
```

### Kubernetes Setup (Huawei CCE)

```bash
# 1. Download kubeconfig from Huawei CCE Console
#    CCE → Your Cluster → More → Download kubeconfig

# 2. Configure kubectl
export KUBECONFIG=./kubeconfig.json

# 3. Verify connection
kubectl cluster-info
kubectl get nodes

# 4. Create namespace
kubectl apply -f k8s/namespace.yaml

# 5. Create SWR pull secret
kubectl create secret docker-registry swr-secret \
  --docker-server=swr.ap-southeast-1.myhuaweicloud.com \
  --docker-username=<your-swr-username> \
  --docker-password=<your-swr-password> \
  --docker-email=your@email.com \
  -n telco-churn

# 6. Deploy application
kubectl apply -f k8s/

# 7. Verify deployment
kubectl get pods -n telco-churn -w
kubectl get svc -n telco-churn
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Project Settings
PROJECT_NAME=telco_churn
ENVIRONMENT=dev  # dev, staging, production
DEBUG=true

# Paths
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
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=logs/api.log

# Huawei Cloud (Optional)
HUAWEI_REGION=ap-southeast-1
HUAWEI_PROJECT_ID=your_project_id
SWR_REGISTRY=swr.ap-southeast-1.myhuaweicloud.com
```

### Configuration Priority

1. **Environment variables** (highest priority)
2. **.env file**
3. **Default values** in `src/config.py` (lowest priority)

---

## 📖 Usage

### API Endpoints

#### Health Check

```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "pipeline_loaded": true,
  "features_loaded": 39,
  "timestamp": "2026-04-06T15:30:00.123456"
}
```

#### Single Prediction

```bash
POST /api/predict

Request Body:
{
  "gender": "Male",
  "age": 35,
  "tenure_in_months": 6,
  "contract": "Month-to-Month",
  "monthly_charge": 95.0,
  "total_charges": 570.0,
  "internet_service": "Fiber Optic",
  "online_security": "No",
  "online_backup": "Yes",
  "streaming_tv": "Yes",
  "customer_id": "CUST-001"
}

Response:
{
  "customer_id": "CUST-001",
  "churn_probability": 0.8234,
  "prediction": "Yes",
  "risk_level": "High",
  "risk_explanation": {
    "risk_level": "High",
    "primary_factors": [
      "Contract: increases risk",
      "Tenure In Months: increases risk",
      "Internet Service: increases risk"
    ],
    "factor_details": {
      "contract": {
        "value": "Month-to-Month",
        "contribution": 0.1523,
        "impact": "increases risk",
        "context": "Short-term contracts have higher churn rates"
      }
    },
    "recommendation": "Offer retention discount immediately...",
    "confidence_score": 0.8234
  },
  "confidence": 0.8234,
  "recommended_action": "Offer retention discount immediately",
  "timestamp": "2026-04-06T15:30:00.123456",
  "feature_contributions": {...}
}
```

#### Batch Prediction

```bash
POST /api/predict/batch

Request Body:
{
  "requests": [
    { "contract": "Month-to-Month", "tenure_in_months": 6, ... },
    { "contract": "Two Year", "tenure_in_months": 48, ... }
  ]
}

Response:
{
  "results": [...],
  "processed_count": 2,
  "error_count": 0,
  "processing_time_ms": 145.23,
  "timestamp": "2026-04-06T15:30:00.123456"
}
```

#### Model Information

```bash
GET /model/info

Response:
{
  "model_type": "XGBClassifier",
  "feature_count": 39,
  "optimal_threshold": 0.35,
  "model_version": "2.0.0-tuned",
  "explainable": true
}
```

### Using the Web Interface

1. Navigate to **http://your-server:8000**
2. Fill in customer details in the form
3. Click **"Predict Churn"**
4. View prediction results with:
   - Churn probability
   - Risk level (Low/Medium/High)
   - Primary risk factors
   - Detailed factor analysis
   - Personalized recommendations
   - Model confidence score

---

## 📚 API Documentation

### Interactive API Docs

FastAPI provides interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Request Schema

All prediction endpoints accept the following fields:

| Field | Type | Required | Example | Description |
|-------|------|----------|---------|-------------|
| `gender` | string | Yes | "Male" | Customer gender |
| `age` | integer | Yes | 35 | Customer age (0-120) |
| `tenure_in_months` | integer | Yes | 24 | Months as customer |
| `contract` | string | Yes | "Month-to-Month" | Contract type |
| `monthly_charge` | float | Yes | 79.5 | Monthly charge ($) |
| `total_charges` | float | Yes | 1868.0 | Total charges ($) |
| `internet_service` | string | Yes | "Fiber Optic" | Internet service type |
| `online_security` | string | Yes | "No" | Has online security |
| `online_backup` | string | Yes | "Yes" | Has online backup |
| `streaming_tv` | string | Yes | "Yes" | Has streaming TV |
| `customer_id` | string | No | "CUST-001" | Unique customer ID |
| *(+ 25+ other optional fields with defaults)* | | | | |

### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `customer_id` | string | Customer ID from request |
| `churn_probability` | float | Probability of churn (0-1) |
| `prediction` | string | "Yes" or "No" |
| `risk_level` | string | "Low", "Medium", or "High" |
| `risk_explanation` | object | Detailed risk analysis |
| `confidence` | float | Model confidence (0-1) |
| `recommended_action` | string | Actionable recommendation |
| `timestamp` | string | ISO 8601 timestamp |

---

## 🚢 Deployment

### Docker Deployment

```bash
# Build
docker build -t telco-churn-app:v1.1.0 -f docker/Dockerfile .

# Test locally
docker run -p 8000:8000 telco-churn-app:v1.1.0

# Push to Huawei SWR
docker tag telco-churn-app:v1.1.0 swr.ap-southeast-1.myhuaweicloud.com/telco-churn/telco-churn-app:v1.1.0
docker push swr.ap-southeast-1.myhuaweicloud.com/telco-churn/telco-churn-app:v1.1.0
```

### Kubernetes Deployment (Huawei CCE)

```bash
# 1. Update deployment image
kubectl set image deployment/telco-churn-api \
  api=swr.ap-southeast-1.myhuaweicloud.com/telco-churn/telco-churn-app:v1.1.0 \
  -n telco-churn

# 2. Watch rollout
kubectl rollout status deployment/telco-churn-api -n telco-churn

# 3. Verify
kubectl get pods -n telco-churn
kubectl get svc -n telco-churn

# 4. Rollback if needed
kubectl rollout undo deployment/telco-churn-api -n telco-churn
```

### Production Checklist

- [ ] Update `api.py` version to match deployment
- [ ] Build and test Docker image locally
- [ ] Push to Huawei SWR with semantic version tag
- [ ] Update Kubernetes deployment manifest
- [ ] Apply changes: `kubectl apply -f k8s/`
- [ ] Verify health endpoint: `curl http://<ELB-IP>/health`
- [ ] Test prediction endpoint with sample data
- [ ] Check pod logs: `kubectl logs -n telco-churn -l app=telco-churn-api`
- [ ] Monitor ELB health checks in Huawei Console
- [ ] Update DNS/CDN if applicable

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline automates:

1. **Code Validation**: Linting, type checking, tests
2. **Docker Build**: Multi-platform image building
3. **Security Scan**: Vulnerability scanning
4. **Image Push**: Push to Huawei SWR
5. **Deployment**: Update Kubernetes deployment (optional)

### Workflow Triggers

```yaml
# Triggers:
- Push to main branch
- Push to release tags (v*.*.*)
- Manual trigger via GitHub UI
- Pull request to main
```

### Pipeline Stages

```
┌─────────────────┐
│    Validate     │
│  - Lint code    │
│  - Run tests    │
│  - Check types  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Build Image   │
│  - Docker build │
│  - Multi-arch   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Security Scan  │
│  - Trivy scan   │
│  - Check CVEs   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Push to SWR   │
│  - Auth to SWR  │
│  - Push tags    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Deploy       │
│  - Update K8s   │
│  - Health check │
└─────────────────┘
```

### Manual Deployment

```bash
# 1. Create version tag
git tag -a v1.1.0 -m "Release v1.1.0"

# 2. Push tag
git push origin v1.1.0

# 3. GitHub Actions triggers automatically
# 4. Monitor: GitHub → Actions → Build & Deploy
```

---

## 📊 Monitoring & Logging

### Application Logs

```bash
# View pod logs
kubectl logs -n telco-churn -l app=telco-churn-api -f

# View specific pod logs
kubectl logs -n telco-churn <pod-name> --tail=100

# View logs with timestamps
kubectl logs -n telco-churn <pod-name> --timestamps
```

### Metrics

The API exposes metrics at `/metrics` (Prometheus format):

- Request count
- Request latency (p50, p95, p99)
- Error rate
- Model inference time
- Active connections

### Health Checks

- **Liveness Probe**: `/health` every 30s
- **Readiness Probe**: `/health` every 10s
- **Startup Probe**: `/health` every 5s (first 60s)

### Huawei Cloud Monitoring

1. **CCE Dashboard**: Pod status, resource usage
2. **ELB Dashboard**: Request count, latency, errors
3. **Cloud Eye**: Custom metrics and alerts
4. **AOM**: Application Operations Management

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. **ELB Health Check Failing**

**Symptom**: ELB shows "Unhealthy" backend servers

**Causes**:
- Security group blocking NodePort
- Pod not listening on expected port
- Health check path misconfigured

**Solutions**:
```bash
# Check security group
# Huawei Console → VPC → Security Groups
# Add inbound rule: TCP 30000-32767 from 0.0.0.0/0

# Verify pod is listening
kubectl exec -n telco-churn <pod-name> -- ss -tlnp | grep 8000

# Test health endpoint
kubectl exec -n telco-churn <pod-name> -- curl http://localhost:8000/health

# Check ELB health check config
# ELB Console → Listener → Backend Servers → Health Check
# Path: /health, Port: 8000, Interval: 15s
```

#### 2. **502 Bad Gateway**

**Symptom**: ELB returns 502 error

**Causes**:
- App binding to 127.0.0.1 instead of 0.0.0.0
- Gunicorn timeout too short
- Pod crashed or not ready

**Solutions**:
```bash
# Check Dockerfile CMD
# Ensure: --bind "0.0.0.0:8000" (not 127.0.0.1)

# Check pod status
kubectl get pods -n telco-churn
kubectl describe pod -n telco-churn <pod-name>

# Check logs
kubectl logs -n telco-churn <pod-name> --tail=50

# Test directly
EXTERNAL_IP=$(kubectl get svc -n telco-churn -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -v http://${EXTERNAL_IP}/health
```

#### 3. **ImagePullBackOff**

**Symptom**: Pod stuck in ImagePullBackOff state

**Causes**:
- SWR secret missing or incorrect
- Image tag doesn't exist
- Network connectivity issues

**Solutions**:
```bash
# Verify SWR secret
kubectl get secret swr-secret -n telco-churn -o yaml

# Recreate secret if needed
kubectl delete secret swr-secret -n telco-churn
kubectl create secret docker-registry swr-secret \
  --docker-server=swr.ap-southeast-1.myhuaweicloud.com \
  --docker-username=<username> \
  --docker-password=<password> \
  -n telco-churn

# Verify image exists
# Huawei Console → SWR → telco-churn → telco-churn-app → Tags

# Check pod events
kubectl describe pod -n telco-churn <pod-name> | grep -A 10 "Events:"
```

#### 4. **Model Not Loaded**

**Symptom**: `/health` returns `"model_loaded": false`

**Causes**:
- Artifacts not in Docker image
- Wrong artifact path
- Corrupted model file

**Solutions**:
```bash
# Check artifacts in container
kubectl exec -n telco-churn <pod-name> -- ls -la /app/artifacts/

# Verify model file exists
kubectl exec -n telco-churn <pod-name> -- test -f /app/artifacts/model.pkl && echo "✅ Model exists" || echo "❌ Model missing"

# Check logs for errors
kubectl logs -n telco-churn <pod-name> | grep -i "model\|artifact"

# Rebuild Docker image with artifacts
docker build -t telco-churn-app:latest -f docker/Dockerfile .
```

#### 5. **Slow Response Times**

**Symptom**: API response time > 1 second

**Causes**:
- Model too large/slow
- Insufficient resources
- Network latency

**Solutions**:
```bash
# Check pod resources
kubectl top pods -n telco-churn

# Scale up replicas
kubectl scale deployment telco-churn-api --replicas=5 -n telco-churn

# Check resource limits in deployment.yaml
# Increase CPU/memory if needed

# Enable horizontal pod autoscaler
kubectl autoscale deployment telco-churn-api \
  --min=2 --max=10 --cpu-percent=80 -n telco-churn
```

#### 6. **GitHub Actions Build Fails**

**Symptom**: Workflow fails at "Build and Push" step

**Causes**:
- SWR credentials expired
- Docker buildx compatibility issues
- Insufficient disk space

**Solutions**:
```yaml
# Update .github/workflows/deploy.yml:
# - Use classic Docker instead of buildx
# - Add --no-cache flag
# - Push tags individually

# Refresh SWR credentials
# Huawei Console → SWR → Access Management → Generate Temporary Password

# Update GitHub Secrets
# Settings → Secrets and variables → Actions
# Update HUAWEI_SWR_PASSWORD
```

#### 7. **Security Group Blocking Traffic**

**Symptom**: Connection timeout to NodePort

**Solutions**:
```bash
# Check current security group rules
# Huawei Console → VPC → Security Groups → Inbound Rules

# Add rule via CLI (if you have access)
hcloud vpc security-group rule add \
  --direction ingress \
  --protocol tcp \
  --port-range 30000-32767 \
  --remote-ip 0.0.0.0/0 \
  --description "Allow Kubernetes NodePorts"

# Or manually in console:
# Protocol: TCP
# Port: 30000-32767
# Source: 0.0.0.0/0
```

---

## ⚠️ Common Mistakes to Avoid

### 1. **Hardcoding Credentials**

❌ **Wrong**:
```python
# In code
password = "my-secret-password"
```

✅ **Correct**:
```python
# Use environment variables
import os
password = os.getenv("DATABASE_PASSWORD")
```

### 2. **Using `latest` Tag in Production**

❌ **Wrong**:
```yaml
image: telco-churn-app:latest  # Can change unexpectedly
```

✅ **Correct**:
```yaml
image: telco-churn-app:v1.1.0  # Pinned version
```

### 3. **Not Validating Input Data**

❌ **Wrong**:
```python
# No validation
age = request.age  # Could be -1 or 1000
```

✅ **Correct**:
```python
# Use Pydantic validation
class Request(BaseModel):
    age: int = Field(..., ge=0, le=120)
```

### 4. **Ignoring Model Drift**

❌ **Wrong**: Deploy model once and never retrain

✅ **Correct**:
- Monitor prediction distributions
- Set up drift detection
- Retrain monthly/quarterly
- A/B test new models

### 5. **No Health Checks**

❌ **Wrong**:
```yaml
# No probes defined
containers:
- name: api
  image: telco-churn-app
```

✅ **Correct**:
```yaml
containers:
- name: api
  image: telco-churn-app
  livenessProbe:
    httpGet:
      path: /health
      port: 8000
  readinessProbe:
    httpGet:
      path: /health
      port: 8000
```

### 6. **Exposing Sensitive Data in Logs**

❌ **Wrong**:
```python
logger.info(f"User data: {request.dict()}")  # Logs PII
```

✅ **Correct**:
```python
logger.info(f"Prediction request: customer_id={request.customer_id}")
```

### 7. **Not Setting Resource Limits**

❌ **Wrong**:
```yaml
# No resource limits
resources: {}
```

✅ **Correct**:
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### 8. **Skipping Tests Before Deploy**

❌ **Wrong**: Push directly to main

✅ **Correct**:
```bash
# Run tests locally first
pytest tests/ -v --cov=src

# Check code quality
black --check src/
flake8 src/

# Then push
git push origin main
```

### 9. **Not Monitoring API Performance**

❌ **Wrong**: Deploy and forget

✅ **Correct**:
- Set up alerts for error rate > 1%
- Monitor p95 latency < 500ms
- Track model prediction distributions
- Set up log aggregation (ELK/Loki)

### 10. **Using Default Docker Base Image**

❌ **Wrong**:
```dockerfile
FROM python:3.10  # Large image (~900MB)
```

✅ **Correct**:
```dockerfile
FROM python:3.10-slim  # Smaller image (~150MB)
```

---

## 🛠️ Development

### Setting Up Development Environment

```bash
# 1. Clone repository
git clone https://github.com/CaptHwi1/telco-churn-mlops.git
cd telco-churn-mlops

# 2. Create virtual environment
python -m venv mlops_env
source mlops_env/bin/activate

# 3. Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy jupyter

# 4. Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Style

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/ --max-line-length=100

# Type checking
mypy src/
```



### Building Documentation

```bash
# Install docs dependencies
pip install mkdocs mkdocs-material

# Serve docs locally
mkdocs serve

# Build static docs
mkdocs build
```

---

## 🧪 Testing

### Test Coverage

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Expected coverage: >80%
```

### Test Types

| Type | Location | Purpose |
|------|----------|---------|

| **Integration Tests** | `tests/test_pipeline.py` | Test component interactions |


### Writing Tests

```python
# tests/test_example.py
import pytest
from src.serving.api import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    payload = {
        "contract": "Month-to-Month",
        "tenure_in_months": 6,
        "monthly_charge": 95.0,
        "internet_service": "Fiber Optic",
        "online_security": "No"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert "churn_probability" in response.json()
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `pytest tests/ -v`
5. **Format code**: `black src/ tests/`
6. **Commit changes**: `git commit -m "Add amazing feature"`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Code Review Process

- All PRs require at least 1 approval
- CI/CD pipeline must pass
- Code must follow PEP 8 style guide
- New features require tests
- Documentation must be updated

### Commit Message Guidelines

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Example**:
```
feat(api): Add batch prediction endpoint

- Add /api/predict/batch endpoint
- Support up to 100 predictions per request
- Include error tracking and reporting

Closes #123
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Abdulmuiz Abdullateef

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **FastAPI**: Modern web framework for building APIs
- **XGBoost**: Gradient boosting library
- **MLflow**: Machine learning lifecycle management
- **Optuna**: Hyperparameter optimization framework
- **Huawei Cloud**: Cloud infrastructure and services
- **Kubernetes**: Container orchestration
- **GitHub Actions**: CI/CD automation

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/CaptHwi1/telco-churn-mlops/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CaptHwi1/telco-churn-mlops/discussions)
- **Email**: abdulmuiz0a@gmail.com.com
- **Documentation**: [Wiki](https://github.com/CaptHwi1/telco-churn-mlops/wiki)

---

## 📈 Roadmap

- [ ] Add model retraining automation
- [ ] Implement A/B testing framework
- [ ] Add SHAP integration for better explanations
- [ ] Support multi-model deployments
- [ ] Add real-time monitoring dashboard
- [ ] Implement feature store
- [ ] Add drift detection alerts
- [ ] Support batch inference on S3/GCS data



---

*Last Updated: April 2026*

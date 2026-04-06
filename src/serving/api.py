"""
FastAPI application for Telco Churn prediction serving.
Serves both API endpoints and Frontend static files for local development.
Supports single and batch predictions.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from datetime import datetime
import logging
import json
import os
import sys
import time

from src.config import settings
from src.features.preprocessing import standardize_columns, load_preprocessing_artifacts

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(settings.artifacts_path) / 'api.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Production API for predicting customer churn with batch support",
    version="1.0.0"
)

# Global variables
_model = None
_pipeline = None
_metadata = None
_config = None
_feature_names = None

# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and artifacts on startup."""
    global _model, _pipeline, _metadata, _config, _feature_names
    
    try:
        artifacts_path = Path(settings.artifacts_path).resolve()
        logger.info(f"Loading artifacts from: {artifacts_path}")
        
        # Load preprocessing
        _pipeline, _metadata = load_preprocessing_artifacts(artifacts_path)
        _feature_names = _metadata.get('feature_names', [])
        logger.info(f"✅ Preprocessing pipeline loaded | Features: {len(_feature_names)}")
        
        # Load serving config
        config_path = artifacts_path / "serving_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                _config = json.load(f)
            logger.info("✅ Serving config loaded")
        else:
            _config = {"optimal_threshold": 0.5}
            logger.warning("⚠️  Serving config not found, using defaults")
        
        # Load model
        model_pkl = artifacts_path / "model.pkl"
        model_json = artifacts_path / "model" / "model.json"
        
        if model_pkl.exists():
            _model = joblib.load(model_pkl)
            logger.info("✅ Model loaded from model.pkl")
        elif model_json.exists():
            _model = xgb.XGBClassifier()
            _model.load_model(str(model_json))
            logger.info("✅ Model loaded from model.json")
        else:
            logger.error("❌ No model found in artifacts!")
            raise RuntimeError("No model found for serving")
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise

# =============================================================================
# FRONTEND SERVING
# =============================================================================

@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html for local development."""
    frontend_path = Path("frontend") / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return JSONResponse(content={"message": "Frontend not found"})

@app.get("/static/{path:path}")
async def serve_static(path: str):
    """Serve static files from frontend folder."""
    file_path = Path("frontend") / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    raise HTTPException(status_code=404, detail="File not found")

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChurnPredictionRequest(BaseModel):
    # Required fields (user provides these)
    gender: str = Field(..., example="Male")
    age: int = Field(..., ge=0, le=120, example=35)
    tenure_in_months: int = Field(..., ge=0, example=24)
    contract: str = Field(..., example="Month-to-Month")
    monthly_charge: float = Field(..., ge=0, example=79.5)
    total_charges: float = Field(..., ge=0, example=1868.0)
    internet_service: str = Field(..., example="Fiber Optic")
    online_security: str = Field(..., example="No")
    online_backup: str = Field(..., example="Yes")
    streaming_tv: str = Field(..., example="Yes")
    
    # Optional fields with defaults
    senior_citizen: str = "No"
    married: str = "Yes"
    dependents: str = "No"
    payment_method: str = "Bank Withdrawal"
    customer_id: Optional[str] = None
    under_30: str = "No"
    phone_service: str = "Yes"
    multiple_lines: str = "No"
    internet_type: str = "Fiber Optic"
    device_protection_plan: str = "No"
    premium_tech_support: str = "No"
    streaming_movies: str = "No"
    streaming_music: str = "No"
    paperless_billing: str = "Yes"
    referred_a_friend: str = "No"
    number_of_referrals: int = 0
    offer: str = "Offer A"
    unlimited_data: str = "No"
    satisfaction_score: int = 3
    avg_monthly_long_distance_charges: float = 0.0
    total_long_distance_charges: float = 0.0
    avg_monthly_gb_download: float = 50.0
    total_extra_data_charges: float = 0.0


class ChurnPredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    prediction: str
    risk_level: str
    confidence: float
    recommended_action: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    requests: List[ChurnPredictionRequest] = Field(..., min_items=1, max_items=100)


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    results: List[ChurnPredictionResponse]
    processed_count: int
    error_count: int
    processing_time_ms: float
    timestamp: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _add_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing features with sensible defaults."""
    defaults = {
        # Demographics
        'gender': 'Male', 'age': 35, 'senior_citizen': 'No',
        'married': 'Yes', 'dependents': 'No', 'under_30': 'No',
        # Contract & Tenure
        'tenure_in_months': 24, 'contract': 'Month-to-Month',
        'payment_method': 'Bank Withdrawal',
        # Services
        'phone_service': 'Yes', 'multiple_lines': 'No',
        'internet_service': 'Fiber Optic', 'internet_type': 'Fiber Optic',
        'online_security': 'No', 'online_backup': 'Yes',
        'device_protection_plan': 'No', 'premium_tech_support': 'No',
        'streaming_tv': 'Yes', 'streaming_movies': 'No',
        'streaming_music': 'No',
        # Billing
        'paperless_billing': 'Yes', 'monthly_charge': 79.5,
        'total_charges': 1868.0, 'avg_monthly_long_distance_charges': 0.0,
        'total_long_distance_charges': 0.0, 'avg_monthly_gb_download': 50.0,
        'total_extra_data_charges': 0.0,
        # Other
        'referred_a_friend': 'No', 'number_of_referrals': 0,
        'offer': 'Offer A', 'unlimited_data': 'No', 'satisfaction_score': 3,
    }
    
    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value
    
    return df


def _transform_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Transform input data using the loaded preprocessing pipeline."""
    try:
        df_std = standardize_columns(df.copy())
        df_complete = _add_missing_features(df_std)
        X_transformed = _pipeline.transform(df_complete)
        
        # Use metadata feature names as fallback
        feature_names_out = _feature_names if _feature_names else []
        return pd.DataFrame(X_transformed, columns=feature_names_out)
            
    except Exception as e:
        logger.error(f"Transform error: {e}", exc_info=True)
        raise


def _predict_single(input_data: dict) -> ChurnPredictionResponse:
    """Make a single prediction."""
    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Transform data
    X_transformed = _transform_for_prediction(df_input)
    
    # Predict
    if hasattr(_model, 'predict_proba'):
        proba = float(_model.predict_proba(X_transformed)[0, 1])
    else:
        pred = _model.predict(X_transformed)
        proba = float(pred[0]) if len(pred) > 0 else 0.5
    
    # Apply threshold
    threshold = _config.get('optimal_threshold', 0.5) if _config else 0.5
    prediction = int(proba >= threshold)
    
    # Business logic
    if proba >= 0.7:
        risk_level = "High"
        action = "Offer retention discount immediately"
    elif proba >= 0.4:
        risk_level = "Medium"
        action = "Send engagement email"
    else:
        risk_level = "Low"
        action = "No action needed"
    
    return ChurnPredictionResponse(
        customer_id=input_data.get('customer_id'),
        churn_probability=round(proba, 4),
        prediction="Yes" if prediction == 1 else "No",
        risk_level=risk_level,
        confidence=round(max(proba, 1 - proba), 4),
        recommended_action=action,
        timestamp=datetime.now().isoformat()
    )


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if _model is not None else "unhealthy",
        "model_loaded": _model is not None,
        "pipeline_loaded": _pipeline is not None,
        "features_loaded": len(_feature_names) if _feature_names else 0,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    return {
        "model_type": type(_model).__name__ if _model else None,
        "feature_count": len(_feature_names) if _feature_names else 0,
        "optimal_threshold": _config.get('optimal_threshold', 0.5) if _config else 0.5,
        "model_version": _config.get('model_version', 'unknown') if _config else 'unknown'
    }


@app.post("/api/predict", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    """Predict churn probability for a single customer."""
    try:
        input_data = request.dict(exclude_unset=True)
        result = _predict_single(input_data)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid input", "detail": str(e)}
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "detail": str(e)}
        )


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(request: BatchPredictionRequest):
    """
    Predict churn probability for multiple customers in batch.
    
    - Accepts up to 100 predictions per request
    - Processes requests in parallel for efficiency
    - Returns results with error tracking
    """
    start_time = time.time()
    results = []
    errors = []
    
    logger.info(f"📦 Batch prediction request: {len(request.requests)} items")
    
    for idx, req in enumerate(request.requests):
        try:
            input_data = req.dict(exclude_unset=True)
            result = _predict_single(input_data)
            results.append(result)
            
        except ValidationError as e:
            logger.warning(f"Batch item {idx} validation error: {e}")
            errors.append({
                "index": idx,
                "customer_id": req.customer_id,
                "error": "validation_error",
                "detail": str(e)
            })
            # Add error response to maintain order
            results.append(ChurnPredictionResponse(
                customer_id=req.customer_id,
                churn_probability=0.0,
                prediction="Error",
                risk_level="Unknown",
                confidence=0.0,
                recommended_action="Validation failed",
                timestamp=datetime.now().isoformat()
            ))
            
        except Exception as e:
            logger.error(f"Batch item {idx} prediction error: {e}", exc_info=True)
            errors.append({
                "index": idx,
                "customer_id": req.customer_id,
                "error": "prediction_error",
                "detail": str(e)
            })
            # Add error response to maintain order
            results.append(ChurnPredictionResponse(
                customer_id=req.customer_id,
                churn_probability=0.0,
                prediction="Error",
                risk_level="Unknown",
                confidence=0.0,
                recommended_action="Prediction failed",
                timestamp=datetime.now().isoformat()
            ))
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Log batch metrics
    logger.info(f"✅ Batch complete: {len(results)} processed, {len(errors)} errors, {processing_time:.2f}ms")
    
    return BatchPredictionResponse(
        results=results,
        processed_count=len(results),
        error_count=len(errors),
        processing_time_ms=round(processing_time, 2),
        timestamp=datetime.now().isoformat()
    )


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper JSONResponse."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "detail": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with proper JSONResponse."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "status_code": 500,
                "detail": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }
        }
    )
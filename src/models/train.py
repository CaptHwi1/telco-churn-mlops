"""
Model training module for Telco Churn prediction.
Handles XGBoost training, MLflow logging, and artifact saving.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import joblib
import json
from datetime import datetime

from src.config import settings
from src.features import (
    fit_preprocessing_pipeline,
    transform_data,
    save_preprocessing_artifacts,
    standardize_columns
)


def train_model(
    df: pd.DataFrame,
    target_col: str = 'churn_label',
    experiment_name: Optional[str] = None,
    run_name: str = "xgboost_baseline"
) -> Dict:
    """
    Train XGBoost model with MLflow tracking.
    
    Args:
        df: Input DataFrame (raw or preprocessed)
        target_col: Target column name
        experiment_name: MLflow experiment name
        run_name: MLflow run name
    
    Returns:
        Dictionary with training results and artifact paths
    """
    # Set MLflow experiment
    if experiment_name is None:
        experiment_name = settings.mlflow_experiment_name
    
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Standardize columns
    df_std = standardize_columns(df)
    
    # Fit preprocessing pipeline
    pipeline, feature_names, metadata = fit_preprocessing_pipeline(
        df_std, target_col=target_col
    )
    
    # Transform data
    X, y = transform_data(df_std, pipeline, metadata)
    
    # Calculate class weight for XGBoost imbalance handling
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # Model parameters
    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': settings.random_state,
        'n_estimators': 200,
        'n_jobs': -1
    }
    
    # Cross-validation
    cv = StratifiedKFold(
        n_splits=settings.cv_folds, 
        shuffle=True, 
        random_state=settings.random_state
    )
    
    cv_results = {'recall': [], 'precision': [], 'f1': [], 'auc': []}
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("cv_folds", settings.cv_folds)
        mlflow.log_param("feature_count", len(feature_names))
        
        # CV Loop
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            y_val_pred = model.predict(X_val)
            y_val_pred_proba = model.predict_proba(X_val)[:, 1]
            
            fold_metrics = {
                'recall': recall_score(y_val, y_val_pred, zero_division=0),
                'precision': precision_score(y_val, y_val_pred, zero_division=0),
                'f1': f1_score(y_val, y_val_pred, zero_division=0),
                'auc': roc_auc_score(y_val, y_val_pred_proba),
            }
            
            for metric, value in fold_metrics.items():
                cv_results[metric].append(value)
                mlflow.log_metric(f"fold_{fold}_{metric}", value)
        
        # Aggregate metrics
        avg_metrics = {k: np.mean(v) for k, v in cv_results.items()}
        for k, v in avg_metrics.items():
            mlflow.log_metric(f"cv_{k}_mean", v)
        
        # Train final model on full data
        final_model = xgb.XGBClassifier(**model_params)
        final_model.fit(X, y, verbose=False)
        
        # Log model to MLflow
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X.head(1), final_model.predict_proba(X.head(1))[:, 1])
        
        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=settings.model_registry_name,
            input_example=X.head(1)
        )
        
        # Save preprocessing artifacts
        artifacts_dir = settings.artifacts_path
        save_preprocessing_artifacts(pipeline, metadata, artifacts_dir)
        
        # Save serving config
        serving_config = {
            "optimal_threshold": 0.5,  # Can be optimized further
            "feature_names": feature_names,
            "target_mapping": {0: "No", 1: "Yes"},
            "model_type": "xgboost",
            "mlflow_run_id": run.info.run_id,
            "cv_metrics": avg_metrics,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(artifacts_dir / "serving_config.json", "w") as f:
            json.dump(serving_config, f, indent=2)
        
        return {
            "run_id": run.info.run_id,
            "model": final_model,
            "pipeline": pipeline,
            "metrics": avg_metrics,
            "artifacts_dir": artifacts_dir,
            "feature_names": feature_names
        }


def load_model(
    model_uri: Optional[str] = None,
    run_id: Optional[str] = None
) -> Tuple:
    """
    Load trained model and preprocessing pipeline.
    
    Args:
        model_uri: MLflow model URI (e.g., "models:/telco_churn_model/Production")
        run_id: MLflow run ID to load from
    
    Returns:
        Tuple of (model, pipeline, metadata)
    """
    from src.features import load_preprocessing_artifacts
    
    # Load model
    if model_uri:
        model = mlflow.pyfunc.load_model(model_uri)
    elif run_id:
        model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    else:
        # Load latest from registry
        model = mlflow.pyfunc.load_model(f"models:/{settings.model_registry_name}/Production")
    
    # Load preprocessing
    pipeline, metadata = load_preprocessing_artifacts()
    
    return model, pipeline, metadata
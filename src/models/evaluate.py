"""
Model evaluation module for Telco Churn prediction.
Handles metrics calculation, business impact analysis, and reporting.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime

from src.config import settings


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Calculate comprehensive model evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (0/1)
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold
    
    Returns:
        Dictionary with all evaluation metrics
    """
    # Apply threshold if needed
    if y_pred is None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        # Classification Metrics
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_pred_proba)),
        
        # Confusion Matrix
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        
        # Counts
        'true_positives': int(np.sum((y_true == 1) & (y_pred == 1))),
        'true_negatives': int(np.sum((y_true == 0) & (y_pred == 0))),
        'false_positives': int(np.sum((y_true == 0) & (y_pred == 1))),
        'false_negatives': int(np.sum((y_true == 1) & (y_pred == 0))),
        
        # Threshold
        'threshold': float(threshold),
        'total_samples': int(len(y_true)),
        
        # Timestamp
        'evaluated_at': datetime.now().isoformat()
    }
    
    # Add class distribution
    metrics['class_distribution'] = {
        'churn': int(np.sum(y_true == 1)),
        'no_churn': int(np.sum(y_true == 0)),
        'churn_rate': float(np.mean(y_true) * 100)
    }
    
    return metrics


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    offer_cost: float = 50.0,
    clv: float = 500.0,
    offer_efficacy: float = 0.30
) -> Dict:
    """
    Calculate business impact metrics for churn prediction.
    
    Args:
        y_true: True churn labels
        y_pred: Predicted churn labels
        offer_cost: Cost per retention offer ($)
        clv: Customer lifetime value ($)
        offer_efficacy: Percentage of offered customers who stay
    
    Returns:
        Dictionary with business metrics
    """
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # Churners caught
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False alarms
    fn = np.sum((y_true == 1) & (y_pred == 0))  # Missed churners
    tn = np.sum((y_true == 0) & (y_pred == 0))  # Correct negatives
    
    n_customers = len(y_true)
    
    # Business calculations
    retained_customers = tp * offer_efficacy
    campaign_cost = (tp + fp) * offer_cost
    revenue_saved = retained_customers * clv
    net_benefit = revenue_saved - campaign_cost
    
    # ROI
    roi = (net_benefit / campaign_cost * 100) if campaign_cost > 0 else 0
    
    business_metrics = {
        'customers_analyzed': int(n_customers),
        'churners_caught': int(tp),
        'churners_missed': int(fn),
        'false_alarms': int(fp),
        'correct_negatives': int(tn),
        'retention_rate': float(tp / np.sum(y_true == 1) * 100) if np.sum(y_true == 1) > 0 else 0,
        'retained_customers': float(retained_customers),
        'campaign_cost': float(campaign_cost),
        'revenue_saved': float(revenue_saved),
        'net_benefit': float(net_benefit),
        'roi_percent': float(roi),
        'offer_cost': float(offer_cost),
        'clv': float(clv),
        'offer_efficacy': float(offer_efficacy),
        'calculated_at': datetime.now().isoformat()
    }
    
    return business_metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = 'recall',
    min_precision: float = 0.35,
    threshold_range: Tuple[float, float, float] = (0.1, 0.9, 0.05)
) -> Dict:
    """
    Find optimal decision threshold based on business requirements.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('recall', 'f1', 'balanced')
        min_precision: Minimum precision constraint
        threshold_range: (start, end, step) for threshold search
    
    Returns:
        Dictionary with optimal threshold and metrics
    """
    start, end, step = threshold_range
    thresholds = np.arange(start, end + step, step)
    
    results = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Apply precision constraint
        if precision < min_precision:
            continue
        
        results.append({
            'threshold': float(thresh),
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1)
        })
    
    if not results:
        # Fallback to default threshold
        return {
            'optimal_threshold': 0.5,
            'recall': float(recall_score(y_true, (y_pred_proba >= 0.5).astype(int), zero_division=0)),
            'precision': float(precision_score(y_true, (y_pred_proba >= 0.5).astype(int), zero_division=0)),
            'f1': float(f1_score(y_true, (y_pred_proba >= 0.5).astype(int), zero_division=0)),
            'warning': 'No threshold met precision constraint, using default 0.5'
        }
    
    # Find optimal based on metric
    if metric == 'recall':
        best = max(results, key=lambda x: x['recall'])
    elif metric == 'f1':
        best = max(results, key=lambda x: x['f1'])
    elif metric == 'balanced':
        best = max(results, key=lambda x: x['recall'] * x['precision'])
    else:
        best = max(results, key=lambda x: x['recall'])
    
    best['all_thresholds'] = results
    best['search_range'] = {'start': start, 'end': end, 'step': step}
    best['min_precision_constraint'] = min_precision
    best['optimized_at'] = datetime.now().isoformat()
    
    return best


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dict: bool = False
) -> str:
    """
    Generate sklearn-style classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dict: Return as dictionary if True
    
    Returns:
        Classification report string or dictionary
    """
    report = classification_report(
        y_true, y_pred,
        target_names=['No Churn', 'Churn'],
        output_dict=output_dict,
        zero_division=0
    )
    
    return report


def save_evaluation_results(
    metrics: Dict,
    business_metrics: Dict,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save evaluation results to JSON files.
    
    Args:
        metrics: Model evaluation metrics
        business_metrics: Business impact metrics
        output_dir: Output directory
    
    Returns:
        Path to output directory
    """
    if output_dir is None:
        output_dir = settings.artifacts_path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model metrics
    with open(output_path / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save business metrics
    with open(output_path / 'business_metrics.json', 'w') as f:
        json.dump(business_metrics, f, indent=2)
    
    # Save combined report
    combined_report = {
        'model_metrics': metrics,
        'business_metrics': business_metrics,
        'generated_at': datetime.now().isoformat()
    }
    with open(output_path / 'evaluation_report.json', 'w') as f:
        json.dump(combined_report, f, indent=2)
    
    return output_path


def print_evaluation_summary(metrics: Dict, business_metrics: Dict) -> None:
    """
    Print formatted evaluation summary to console.
    
    Args:
        metrics: Model evaluation metrics
        business_metrics: Business impact metrics
    """
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n  Classification Metrics:")
    print(f"   - Recall: {metrics['recall']:.3f}")
    print(f"   - Precision: {metrics['precision']:.3f}")
    print(f"   - F1 Score: {metrics['f1']:.3f}")
    print(f"   - AUC-ROC: {metrics['auc']:.3f}")
    
    print(f"\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   [[{cm[0][0]:5d} {cm[0][1]:5d}]  [TN  FP]")
    print(f"    [{cm[1][0]:5d} {cm[1][1]:5d}]]  [FN  TP]")
    
    print(f"\n Business Impact:")
    print(f"   - Churners Caught: {business_metrics['churners_caught']:,}")
    print(f"   - Churners Missed: {business_metrics['churners_missed']:,}")
    print(f"   - Retention Rate: {business_metrics['retention_rate']:.1f}%")
    print(f"   - Campaign Cost: ${business_metrics['campaign_cost']:,.0f}")
    print(f"   - Revenue Saved: ${business_metrics['revenue_saved']:,.0f}")
    print(f"   -   Net Benefit: ${business_metrics['net_benefit']:,.0f}")
    print(f"   - ROI: {business_metrics['roi_percent']:.1f}%")
    
    print("\n" + "=" * 60)
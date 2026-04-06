"""Model evaluation and business metrics calculation."""

import numpy as np
from typing import Dict

def calculate_business_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    customer_value: float = 500.0,
    retention_cost: float = 50.0,
    retention_efficacy: float = 0.30
) -> Dict:
    """
    Calculate business impact of churn predictions.
    
    Args:
        y_true: Actual churn labels (0/1)
        y_pred: Predicted churn labels (0/1)
        customer_value: Average lifetime value of a customer
        retention_cost: Cost of retention offer per customer
        retention_efficacy: Percentage of offered customers who stay
        
    Returns:
        Dictionary with business metrics
    """
    # True Positives: Predicted churn, actually churned
    tp = np.sum((y_pred == 1) & (y_true == 1))
    # False Positives: Predicted churn, actually stayed
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    # Revenue saved from retained customers (TP * efficacy * value)
    revenue_saved = tp * retention_efficacy * customer_value
    
    # Cost of offers (TP + FP) * cost
    offer_cost = (tp + fp) * retention_cost
    
    # Net benefit
    net_benefit = revenue_saved - offer_cost
    
    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'revenue_saved': float(revenue_saved),
        'offer_cost': float(offer_cost),
        'net_benefit': float(net_benefit),
        'roi': float(net_benefit / offer_cost) if offer_cost > 0 else 0.0
    }
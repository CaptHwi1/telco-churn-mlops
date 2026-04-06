"""Data validation utilities for Telco Churn dataset."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

def validate_raw_data(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Validate raw data quality.
    
    Returns:
        Tuple of (is_valid, validation_report)
    """
    report = {
        'is_valid': True,
        'issues': []
    }
    
    # Check for empty dataframe
    if df.empty:
        report['is_valid'] = False
        report['issues'].append("Dataframe is empty")
        return False, report
    
    # Check for required columns
    required_cols = ['Churn Label', 'gender', 'age', 'tenure_in_months', 'contract']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        report['is_valid'] = False
        report['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check for duplicate Customer IDs
    if 'Customer ID' in df.columns:
        duplicates = df['Customer ID'].duplicated().sum()
        if duplicates > 0:
            report['issues'].append(f"Found {duplicates} duplicate Customer IDs")
    
    return report['is_valid'], report


def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    Generate data quality report.
    
    Returns:
        Dictionary with quality metrics
    """
    total_rows = len(df)
    total_cells = df.size
    
    # Missing values
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    duplicate_pct = (duplicate_rows / total_rows) * 100
    
    # Calculate quality score (0-100)
    quality_score = 100 - missing_pct - (duplicate_pct * 0.5)
    quality_score = max(0, min(100, quality_score))
    
    return {
        'total_rows': total_rows,
        'total_columns': len(df.columns),
        'missing_cells': missing_cells,
        'missing_percentage': round(missing_pct, 2),
        'duplicate_rows': duplicate_rows,
        'duplicate_percentage': round(duplicate_pct, 2),
        'quality_score': round(quality_score, 1)
    }
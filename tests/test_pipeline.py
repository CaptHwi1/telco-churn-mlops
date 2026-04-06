"""
Integration tests for complete ML pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.validation import validate_data_quality
from src.features.preprocessing import fit_preprocessing_pipeline, transform_data
from src.models.evaluate import calculate_business_metrics


class TestEndToEndPipeline:
    """Test end-to-end ML pipeline."""
    
    def test_complete_pipeline(self, sample_data):
        """Test complete pipeline from raw data to metrics."""
        # 1. Validate data
        is_valid, validation_report = validate_raw_data(sample_data)
        assert is_valid == True
        
        # 2. Check data quality
        quality_report = validate_data_quality(sample_data)
        assert quality_report['quality_score'] > 0
        
        # 3. Preprocess data
        pipeline, feature_names, metadata = fit_preprocessing_pipeline(
            sample_data,
            target_col='Churn Label',
            validate=False
        )
        X, y = transform_data(sample_data, pipeline, metadata)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(sample_data)
        
        # 4. Train model (simplified for testing)
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y, verbose=False)
        
        # 5. Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # 6. Calculate business metrics
        metrics = calculate_business_metrics(y, predictions)
        
        assert 'net_benefit' in metrics
        assert 'roi' in metrics
    
    def test_pipeline_with_missing_values(self, sample_data):
        """Test pipeline handles missing values."""
        # Introduce missing values
        sample_data.loc[0, 'age'] = np.nan
        sample_data.loc[1, 'monthly_charge'] = np.nan
        
        # Pipeline should handle this
        pipeline, feature_names, metadata = fit_preprocessing_pipeline(
            sample_data,
            target_col='Churn Label',
            validate=False
        )
        X, y = transform_data(sample_data, pipeline, metadata)
        
        # No missing values after transform
        assert X.isnull().sum().sum() == 0
    
    def test_pipeline_reproducibility(self, sample_data):
        """Test pipeline produces reproducible results."""
        # Run pipeline twice
        pipeline1, features1, metadata1 = fit_preprocessing_pipeline(
            sample_data,
            target_col='Churn Label',
            validate=False
        )
        X1, y1 = transform_data(sample_data, pipeline1, metadata1)
        
        pipeline2, features2, metadata2 = fit_preprocessing_pipeline(
            sample_data,
            target_col='Churn Label',
            validate=False
        )
        X2, y2 = transform_data(sample_data, pipeline2, metadata2)
        
        # Results should be identical
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)
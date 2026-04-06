"""
Data preprocessing pipeline for Telco Churn prediction.
Handles column standardization, missing values, encoding, scaling, and pipeline serialization.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
import joblib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings
import json
from datetime import datetime

from src.config import settings
from src.data.validation import validate_raw_data, validate_data_quality

warnings.filterwarnings('ignore')


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: lowercase, replace spaces with underscores."""
    df_copy = df.copy()
    df_copy.columns = (
        df_copy.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('/', '_')
    )
    return df_copy


class TotalChargesConverter(BaseEstimator, TransformerMixin):
    """Convert total_charges from string to numeric."""
    
    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if 'total_charges' in X_copy.columns:
            X_copy['total_charges'] = pd.to_numeric(
                X_copy['total_charges'].astype(str).str.strip(), 
                errors='coerce'
            )
            if X_copy['total_charges'].isna().any():
                median_val = X_copy['total_charges'].median()
                X_copy['total_charges'] = X_copy['total_charges'].fillna(median_val)
        return X_copy


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encode target variable: Yes/No → 1/0."""
    
    def fit(self, X: pd.Series, y=None):
        self.mapping_ = {'Yes': 1, 'No': 0}
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.Series) -> pd.Series:
        return X.map(self.mapping_).fillna(0).astype(int)
    
    def inverse_transform(self, X: pd.Series) -> pd.Series:
        reverse_mapping = {v: k for k, v in self.mapping_.items()}
        return X.map(reverse_mapping)


def get_feature_categories(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize features for appropriate preprocessing."""
    exclude_cols = [
        'customer_id', 'churn_label', 'customer_status', 'churn_score', 
        'cltv', 'churn_category', 'churn_reason',
        'country', 'state', 'city', 'zip_code', 'latitude', 'longitude', 
        'population', 'total_revenue', 'total_refunds', 'quarter'
    ]
    
    available_cols = [c for c in df.columns if c not in exclude_cols]
    
    numeric_features = [
        c for c in available_cols 
        if df[c].dtype in ['int64', 'float64'] and df[c].nunique() > 10
    ]
    
    ordinal_features = ['contract']
    
    binary_features = [
        c for c in available_cols 
        if df[c].nunique() == 2 and c not in ordinal_features + ['churn_label']
        and df[c].dtype == 'object'
    ]
    
    categorical_features = [
        c for c in available_cols 
        if c not in numeric_features + ordinal_features + binary_features
        and df[c].dtype == 'object' and df[c].nunique() < 20
    ]
    
    return {
        'numeric': numeric_features,
        'ordinal': ordinal_features,
        'binary': binary_features,
        'categorical': categorical_features,
        'all_features': numeric_features + ordinal_features + binary_features + categorical_features
    }


def create_preprocessing_pipeline(
    df: pd.DataFrame,
    feature_categories: Dict[str, List[str]]
) -> ColumnTransformer:
    """Create sklearn ColumnTransformer pipeline for preprocessing."""
    numeric_features = feature_categories['numeric']
    ordinal_features = feature_categories['ordinal']
    binary_features = feature_categories['binary']
    categorical_features = feature_categories['categorical']
    
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=[['Month-to-Month', 'One Year', 'Two Year']],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])
    
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            drop='if_binary'
        ))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('ord', ordinal_pipeline, ordinal_features),
            ('bin', binary_pipeline, binary_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def fit_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str = 'churn_label',
    validate: bool = True
) -> Tuple[Pipeline, List[str], Dict]:
    """Fit complete preprocessing pipeline on training data."""
    if validate:
        try:
            is_valid, validation_report = validate_raw_data(df)
            if not is_valid:
                warnings.warn(f"Data validation failed: {validation_report}")
            quality_report = validate_data_quality(df)
            if quality_report['quality_score'] < 80:
                warnings.warn(f"Low data quality score: {quality_report['quality_score']}")
        except Exception as e:
            warnings.warn(f"Validation error (continuing): {e}")
    
    df_std = standardize_columns(df)
    print(f"   Standardized columns: {list(df_std.columns)[:5]}...")
    
    feature_categories = get_feature_categories(df_std)
    print(f"   Numeric features: {len(feature_categories['numeric'])}")
    print(f"   Categorical features: {len(feature_categories['categorical'])}")
    
    preprocessor = create_preprocessing_pipeline(df_std, feature_categories)
    preprocessor.fit(df_std)
    
    feature_names = []
    feature_names.extend(feature_categories['numeric'])
    feature_names.extend(feature_categories['ordinal'])
    feature_names.extend(feature_categories['binary'])
    
    if feature_categories['categorical']:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(feature_categories['categorical'])
        feature_names.extend(cat_feature_names)
    
    print(f"   Total features after encoding: {len(feature_names)}")
    
    pipeline = Pipeline([
        ('converter', TotalChargesConverter()),
        ('preprocessor', preprocessor),
    ])
    
    print("   Fitting pipeline...")
    pipeline.fit(df_std)
    
    try:
        check_is_fitted(pipeline)
        print(f"   ✅ Pipeline fitted successfully")
    except Exception as e:
        raise RuntimeError(f"Pipeline fitting failed: {e}")
    
    target_encoder = TargetEncoder()
    if target_col and target_col in df_std.columns:
        target_encoder.fit(df_std[target_col])
    
    metadata = {
        'feature_categories': feature_categories,
        'feature_names': feature_names,
        'target_col': target_col,
        'target_encoder': target_encoder,
        'validation_enabled': validate,
        'fitted_at': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    return pipeline, feature_names, metadata


def transform_data(
    df: pd.DataFrame,
    pipeline: Pipeline,
    metadata: Dict,
    validate: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Transform data using fitted pipeline."""
    df_std = standardize_columns(df)
    X_transformed = pipeline.transform(df_std)
    
    # Convert numpy array to DataFrame with proper column names
    X_df = pd.DataFrame(X_transformed, columns=metadata['feature_names'])
    
    y_transformed = None
    if metadata['target_col'] and metadata['target_col'] in df_std.columns:
        y_transformed = metadata['target_encoder'].transform(df_std[metadata['target_col']])
    
    return X_df, y_transformed


def save_preprocessing_artifacts(
    pipeline: Pipeline,
    metadata: Dict,
    output_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Save preprocessing pipeline and metadata for serving."""
    if output_dir is None:
        output_dir = settings.artifacts_path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, output_path / 'preprocessing_pipeline.joblib')
    
    serializable_metadata = {
        'feature_categories': metadata['feature_categories'],
        'feature_names': metadata['feature_names'],
        'target_col': metadata['target_col'],
        'target_mapping': getattr(metadata['target_encoder'], 'mapping_', None),
        'validation_enabled': metadata.get('validation_enabled', True),
        'fitted_at': metadata.get('fitted_at', datetime.now().isoformat()),
        'version': metadata.get('version', '1.0.0')
    }
    joblib.dump(serializable_metadata, output_path / 'preprocessing_metadata.joblib')
    
    with open(output_path / 'preprocessing_metadata.json', 'w') as f:
        json.dump(serializable_metadata, f, indent=2, default=str)
    
    return output_path


def load_preprocessing_artifacts(
    input_dir: Optional[Union[str, Path]] = None
) -> Tuple[Pipeline, Dict]:
    """Load preprocessing pipeline and metadata for inference."""
    if input_dir is None:
        input_dir = settings.artifacts_path
    
    input_path = Path(input_dir)
    
    pipeline = joblib.load(input_path / 'preprocessing_pipeline.joblib')
    metadata = joblib.load(input_path / 'preprocessing_metadata.joblib')
    
    target_encoder = TargetEncoder()
    if metadata.get('target_mapping') is not None:
        target_encoder.mapping_ = metadata['target_mapping']
        target_encoder.is_fitted_ = True
    metadata['target_encoder'] = target_encoder
    
    return pipeline, metadata
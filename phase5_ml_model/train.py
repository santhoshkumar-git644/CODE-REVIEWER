import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time
import uuid

# Set up simple logging (instead of MLFlow to avoid heavy dependencies, though MLFlow is supported)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train and validation splits."""
    logger.info(f"Loading data from {train_path} and {val_path}")
    
    train_df = pd.read_csv(train_path) if train_path.endswith('.csv') else pd.read_parquet(train_path)
    val_df = pd.read_csv(val_path) if val_path.endswith('.csv') else pd.read_parquet(val_path)
    
    return train_df, val_df

def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Extracts features and target, returns X, y, and feature names."""
    if 'target' not in df.columns:
        raise ValueError("Target column missing in dataset.")
        
    # Drop non-feature columns
    drop_cols = ['id', 'target', 'func', 'project', 'commit_id', 'language']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0)
    
    return X, y, feature_cols

def train_random_forest(X: np.ndarray, y: np.ndarray, params: Optional[Dict] = None) -> Pipeline:
    """Trains a Random Forest classifier within a pipeline."""
    logger.info("Training Random Forest model...")
    default_params = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    if params:
        default_params.update(params)
        
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**default_params))
    ])
    
    pipeline.fit(X, y)
    logger.info("Random Forest training complete.")
    return pipeline

def train_xgboost(X: np.ndarray, y: np.ndarray, params: Optional[Dict] = None) -> Pipeline:
    """Trains an XGBoost classifier within a pipeline."""
    logger.info("Training XGBoost model...")
    
    # Calculate scale_pos_weight for imbalanced classes
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    scale_pos_weight = n_neg / max(1, n_pos)
    
    default_params = {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    if params:
        default_params.update(params)
        
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(**default_params))
    ])
    
    pipeline.fit(X, y)
    logger.info("XGBoost training complete.")
    return pipeline

def train_with_smote(X: np.ndarray, y: np.ndarray, model_type: str = 'rf', params: Optional[Dict] = None) -> ImbPipeline:
    """Trains a model handling class imbalance with SMOTE."""
    logger.info(f"Training {model_type.upper()} with SMOTE...")
    
    if model_type.lower() == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, **(params or {}))
    elif model_type.lower() == 'xgb':
        clf = xgb.XGBClassifier(n_estimators=100, random_state=42, **(params or {}))
    else:
        raise ValueError("model_type must be 'rf' or 'xgb'")
        
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', clf)
    ])
    
    pipeline.fit(X, y)
    logger.info("SMOTE training complete.")
    return pipeline

def cross_validate_model(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
    """Performs k-fold cross validation."""
    logger.info(f"Running {cv}-fold cross validation...")
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    
    results = {}
    for metric in scoring:
        results[f"mean_{metric}"] = np.mean(scores[f"test_{metric}"])
        results[f"std_{metric}"] = np.std(scores[f"test_{metric}"])
        
    logger.info(f"CV Results: F1={results['mean_f1']:.3f}, AUC={results['mean_roc_auc']:.3f}")
    return results

def log_experiment(model_name: str, params: Dict, metrics: Dict, output_dir: str):
    """Saves experiment results to a JSON log file."""
    os.makedirs(output_dir, exist_ok=True)
    
    exp_id = str(uuid.uuid4())[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    log_data = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "model": model_name,
        "parameters": params,
        "metrics": metrics
    }
    
    log_path = os.path.join(output_dir, f"exp_{model_name}_{timestamp}.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
        
    logger.info(f"Experiment logged to {log_path}")
    return exp_id

def main():
    """Main training pipeline entry point."""
    print("Training pipeline initialized. Awaiting dataset.")
    # Placeholder for running the script directly
    
if __name__ == "__main__":
    main()

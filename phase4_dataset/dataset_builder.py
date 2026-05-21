import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split

def build_dataset(features_dir: str, labels_path: str = None) -> pd.DataFrame:
    """Builds a unified dataset from features and optionally labels."""
    df_list = []
    
    # Load all feature csvs
    if os.path.exists(features_dir):
        for file in os.listdir(features_dir):
            if file.endswith('.csv'):
                path = os.path.join(features_dir, file)
                try:
                    df = pd.read_csv(path)
                    df_list.append(df)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    
    if not df_list:
        print(f"No feature CSVs found in {features_dir}")
        return pd.DataFrame()
        
    final_df = pd.concat(df_list, ignore_index=True)
    
    # Optional label loading would go here
    # If the CSVs already contain 'target', we're good
    
    return final_df

def create_splits(df: pd.DataFrame, train_ratio: float = 0.7, 
                 val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset into train/val/test using stratified sampling if possible."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    if 'target' in df.columns:
        # Stratified split
        y = df['target']
        X = df.drop(columns=['target'])
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=42
        )
        
        # Calculate test_ratio relative to temp size
        test_ratio_relative = test_ratio / (val_ratio + test_ratio)
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_relative, stratify=y_temp, random_state=42
        )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
    else:
        # Random split
        train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=42)
        test_ratio_relative = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(temp_df, test_size=test_ratio_relative, random_state=42)
        
    return train_df, val_df, test_df

def save_dataset(df: pd.DataFrame, output_path: str, format: str = 'csv'):
    """Saves dataset to CSV or Parquet format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format.lower() == 'csv':
        if not output_path.endswith('.csv'):
            output_path += '.csv'
        df.to_csv(output_path, index=False)
    elif format.lower() == 'parquet':
        if not output_path.endswith('.parquet'):
            output_path += '.parquet'
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    print(f"Saved dataset ({len(df)} rows) to {output_path}")

def load_dataset(path: str) -> pd.DataFrame:
    """Loads a dataset from CSV or Parquet."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """Returns summary statistics for the dataset."""
    stats = {
        'total_rows': len(df),
        'features': [c for c in df.columns if c not in ['id', 'target', 'func']],
        'missing_values': df.isnull().sum().to_dict()
    }
    
    if 'target' in df.columns:
        counts = df['target'].value_counts()
        stats['class_distribution'] = counts.to_dict()
        stats['buggy_ratio'] = counts.get(1, 0) / len(df)
        
    return stats

def validate_dataset(df: pd.DataFrame) -> List[str]:
    """Checks dataset for potential issues."""
    issues = []
    
    if len(df) == 0:
        issues.append("Dataset is empty.")
        return issues
        
    if 'target' not in df.columns:
        issues.append("Missing 'target' column for labels.")
    else:
        val_counts = df['target'].value_counts()
        if len(val_counts) < 2:
            issues.append("Only one class present in 'target'. ML models need at least two classes.")
        elif val_counts.get(1, 0) / len(df) < 0.05:
            issues.append("Severe class imbalance: 'buggy' class is < 5%. Consider SMOTE.")
            
    # Check for NaNs
    nans = df.isnull().sum().sum()
    if nans > 0:
        issues.append(f"Found {nans} missing values across all columns.")
        
    # Check numeric columns variance
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['id', 'target'] and df[col].std() == 0:
            issues.append(f"Feature '{col}' has zero variance (constant value).")
            
    return issues

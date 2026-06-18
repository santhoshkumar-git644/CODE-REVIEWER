import json
import pandas as pd
from typing import Dict, List, Any

def load_codenet_labels(metadata_path: str) -> Dict[str, int]:
    """Loads Project CodeNet metadata and returns {submission_id: target}."""
    labels = {}
    try:
        df = pd.read_csv(metadata_path)
        # Assuming 'status' is the column where 'Accepted' means clean (0) and others mean buggy (1)
        for _, row in df.iterrows():
            sub_id = str(row['submission_id'])
            status = row['status']
            labels[sub_id] = 0 if status == 'Accepted' else 1
    except Exception as e:
        print(f"Error loading CodeNet labels: {e}")
    return labels

def load_codexglue_labels(jsonl_path: str) -> Dict[str, int]:
    """Loads CodeXGLUE defect detection labels."""
    labels = {}
    try:
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                # Use project+commit_id as unique key, or index if unavailable
                key = f"{data.get('project', 'unk')}_{data.get('commit_id', i)}"
                labels[key] = int(data.get('target', 0))
    except Exception as e:
        print(f"Error loading CodeXGLUE labels: {e}")
    return labels

def assign_binary_labels(verdicts: Dict[str, Any]) -> Dict[str, int]:
    """Maps various textual verdicts to binary buggy(1) / clean(0) labels."""
    binary_labels = {}
    bug_keywords = ['error', 'fail', 'bug', 'vulnerable', 'reject', 'wrong']
    
    for key, verdict in verdicts.items():
        if isinstance(verdict, int):
            binary_labels[key] = verdict
        elif isinstance(verdict, str):
            v_lower = verdict.lower()
            is_buggy = any(kw in v_lower for kw in bug_keywords)
            binary_labels[key] = 1 if is_buggy else 0
        else:
            binary_labels[key] = 0 # Default to clean
            
    return binary_labels

def align_labels_with_features(features_df: pd.DataFrame, labels: Dict[str, int]) -> pd.DataFrame:
    """Merges labels into the features DataFrame based on 'id'."""
    if 'id' not in features_df.columns:
        print("Warning: 'id' column not found in features. Cannot align labels.")
        return features_df
        
    df = features_df.copy()
    
    def get_label(row_id):
        return labels.get(str(row_id), 0)
        
    df['target'] = df['id'].apply(get_label)
    return df

def compute_label_statistics(labels: Dict[str, int]) -> dict:
    """Computes class distribution for labels."""
    total = len(labels)
    if total == 0:
        return {'total': 0, 'buggy': 0, 'clean': 0, 'buggy_ratio': 0.0}
        
    buggy = sum(1 for v in labels.values() if v == 1)
    clean = total - buggy
    
    return {
        'total': total,
        'buggy': buggy,
        'clean': clean,
        'buggy_ratio': buggy / total
    }

def generate_synthetic_labels(n_samples: int, bug_ratio: float = 0.3) -> Dict[str, int]:
    """Generates synthetic labels for testing."""
    import random
    labels = {}
    for i in range(n_samples):
        key = f"syn_{i:04d}"
        labels[key] = 1 if random.random() < bug_ratio else 0
    return labels

import os
import time
import json
import joblib
from typing import Dict, Any, Tuple, List

import logging
logger = logging.getLogger(__name__)

def save_model(model: Any, path: str, metadata: Dict[str, Any] = None):
    """Saves a model to disk along with metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Save the model
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    
    # Save metadata if provided
    if metadata:
        meta_path = path + ".meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {meta_path}")

def load_model(path: str) -> Tuple[Any, Dict[str, Any]]:
    """Loads a model and its metadata from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
        
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    
    metadata = {}
    meta_path = path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
    return model, metadata

def version_model(model: Any, base_dir: str, model_name: str, metadata: Dict[str, Any] = None) -> str:
    """Saves a model with a timestamped version directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(base_dir, model_name, timestamp)
    os.makedirs(version_dir, exist_ok=True)
    
    model_path = os.path.join(version_dir, "model.joblib")
    
    if metadata is None:
        metadata = {}
    metadata['version'] = timestamp
    metadata['model_name'] = model_name
    
    save_model(model, model_path, metadata)
    
    # Update latest symlink/pointer
    latest_pointer = os.path.join(base_dir, model_name, "latest.txt")
    with open(latest_pointer, 'w') as f:
        f.write(timestamp)
        
    return version_dir

def load_best_model(base_dir: str, model_name: str) -> Any:
    """Loads the latest/best model version."""
    model_dir = os.path.join(base_dir, model_name)
    latest_pointer = os.path.join(model_dir, "latest.txt")
    
    if os.path.exists(latest_pointer):
        with open(latest_pointer, 'r') as f:
            version = f.read().strip()
    else:
        # Fallback to finding the newest directory
        versions = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if not versions:
            raise FileNotFoundError(f"No versions found for model {model_name}")
        version = sorted(versions)[-1]
        
    model_path = os.path.join(model_dir, version, "model.joblib")
    model, _ = load_model(model_path)
    return model

def list_model_versions(base_dir: str, model_name: str) -> List[Dict[str, Any]]:
    """Lists all available versions for a model with their metadata."""
    model_dir = os.path.join(base_dir, model_name)
    if not os.path.exists(model_dir):
        return []
        
    versions = []
    for d in os.listdir(model_dir):
        version_dir = os.path.join(model_dir, d)
        if os.path.isdir(version_dir):
            meta_path = os.path.join(version_dir, "model.joblib.meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    versions.append(meta)
            else:
                versions.append({'version': d})
                
    return sorted(versions, key=lambda x: x.get('version', ''), reverse=True)

def delete_old_versions(base_dir: str, model_name: str, keep: int = 3):
    """Deletes old versions, keeping the newest `keep` versions."""
    import shutil
    
    model_dir = os.path.join(base_dir, model_name)
    if not os.path.exists(model_dir):
        return
        
    versions = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    versions.sort()
    
    if len(versions) > keep:
        for old_version in versions[:-keep]:
            old_dir = os.path.join(model_dir, old_version)
            try:
                shutil.rmtree(old_dir)
                logger.info(f"Deleted old model version: {old_version}")
            except Exception as e:
                logger.error(f"Failed to delete {old_dir}: {e}")

def export_model_card(model: Any, metrics: Dict[str, float], output_path: str):
    """Generates a markdown model card."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_type = type(model).__name__
    if hasattr(model, 'named_steps'):
        model_type = type(model.named_steps.get('classifier')).__name__
        
    card = f"""# Model Card

## Model Details
- **Type**: {model_type}
- **Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Framework**: Scikit-Learn / XGBoost

## Performance Metrics
- **Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}
- **Precision**: {metrics.get('precision', 'N/A'):.4f}
- **Recall**: {metrics.get('recall', 'N/A'):.4f}
- **F1 Score**: {metrics.get('f1', 'N/A'):.4f}
- **ROC AUC**: {metrics.get('roc_auc', 'N/A'):.4f}

## Intended Use
Bug risk prediction for code analysis. Intended to identify potentially vulnerable or defective code snippets.
"""
    with open(output_path, 'w') as f:
        f.write(card)

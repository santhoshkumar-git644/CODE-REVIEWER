import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Computes standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            # Handle case where only one class is present in y_true
            metrics["roc_auc"] = 0.5
            
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
    """Plots and saves confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Clean (0)', 'Buggy (1)'],
                yticklabels=['Clean (0)', 'Buggy (1)'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: str):
    """Plots and saves ROC curve."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    except ValueError:
        print("Cannot plot ROC curve: Need both classes in y_true")

def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: str):
    """Plots and saves Precision-Recall curve."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    except ValueError:
        print("Cannot plot PR curve.")

def plot_feature_importance(model: Any, feature_names: List[str], output_path: str, top_n: int = 15):
    """Plots feature importance for tree-based models."""
    importances = None
    
    # Try to extract importances from pipeline or direct model
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier')
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
    if importances is None:
        print("Model does not support feature importances.")
        return
        
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances[::-1], align='center')
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Returns formatted classification report."""
    return classification_report(y_true, y_pred, target_names=['Clean (0)', 'Buggy (1)'])

def save_results(metrics: Dict, output_path: str):
    """Saves metrics dictionary to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
def compare_models(results_list: List[Dict]) -> pd.DataFrame:
    """Creates a comparison DataFrame from multiple results dicts."""
    df = pd.DataFrame(results_list)
    return df.sort_values(by='roc_auc' if 'roc_auc' in df.columns else 'f1', ascending=False)

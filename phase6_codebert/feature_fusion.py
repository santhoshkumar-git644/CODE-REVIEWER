import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from typing import Tuple, Dict, List, Any

import logging
logger = logging.getLogger(__name__)

class MLPClassifier(nn.Module):
    """Simple Multi-Layer Perceptron for classification."""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128], dropout_rate: float = 0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        # We output logits; BCEWithLogitsLoss will handle the sigmoid
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(1)

def fuse_features(ast_features: np.ndarray, embeddings: np.ndarray, reduce_dim: int = None) -> np.ndarray:
    """Concatenates AST features with CodeBERT embeddings."""
    logger.info(f"Fusing features: AST shape={ast_features.shape}, Embeddings shape={embeddings.shape}")
    
    # Ensure both have same number of samples
    assert ast_features.shape[0] == embeddings.shape[0], "Feature dimensions mismatch"
    
    # Optional: reduce dimensionality of embeddings
    if reduce_dim and reduce_dim < embeddings.shape[1]:
        logger.info(f"Reducing embedding dimensions to {reduce_dim} using PCA")
        pca = PCA(n_components=reduce_dim)
        embeddings = pca.fit_transform(embeddings)
        
    # Standardize AST features (simple robust scaling)
    ast_means = np.mean(ast_features, axis=0)
    ast_stds = np.std(ast_features, axis=0)
    # Avoid division by zero
    ast_stds[ast_stds == 0] = 1.0
    ast_scaled = (ast_features - ast_means) / ast_stds
    
    # Concatenate
    fused = np.hstack([ast_scaled, embeddings])
    logger.info(f"Fused features shape: {fused.shape}")
    return fused

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def train_mlp_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray = None, y_val: np.ndarray = None,
                        hidden_dims: List[int] = [256, 128], 
                        epochs: int = 50, batch_size: int = 64) -> MLPClassifier:
    """Trains a PyTorch MLP on fused features."""
    device = get_device()
    logger.info(f"Training MLP on device: {device}")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    
    # Calculate class weights for imbalanced data
    n_neg = np.sum(y_train == 0)
    n_pos = max(1, np.sum(y_train == 1))
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
                
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                
                # Calculate metrics
                val_preds = (torch.sigmoid(val_outputs) >= 0.5).float()
                
                # Simple F1 calculation
                tp = (val_preds * y_val_t).sum().item()
                fp = (val_preds * (1 - y_val_t)).sum().item()
                fn = ((1 - val_preds) * y_val_t).sum().item()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                val_f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
                
                scheduler.step(val_f1)
                
                # Early stopping logic / model saving
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict()
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with Val F1: {best_val_f1:.4f}")
        
    return model

def train_fused_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    """Trains an XGBoost model on fused features."""
    logger.info("Training XGBoost on fused features...")
    
    n_neg = np.sum(y_train == 0)
    n_pos = max(1, np.sum(y_train == 1))
    scale_pos_weight = n_neg / n_pos
    
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    return clf

def evaluate_fused_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluates a model (MLP or XGBoost) on test data."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    if isinstance(model, nn.Module):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            logits = model(X_test_t)
            y_proba = torch.sigmoid(logits).cpu().numpy()
            y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0))
    }
    
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        metrics["roc_auc"] = 0.5
        
    return metrics

def compare_with_baseline(baseline_metrics: Dict[str, float], fused_metrics: Dict[str, float]) -> Dict[str, float]:
    """Computes improvement of fused model over baseline."""
    improvements = {}
    for metric in baseline_metrics:
        if metric in fused_metrics:
            diff = fused_metrics[metric] - baseline_metrics[metric]
            improvements[f"{metric}_diff"] = diff
            
            # Prevent division by zero
            if baseline_metrics[metric] > 0:
                improvements[f"{metric}_percent_improvement"] = (diff / baseline_metrics[metric]) * 100
            else:
                improvements[f"{metric}_percent_improvement"] = 0.0
                
    return improvements

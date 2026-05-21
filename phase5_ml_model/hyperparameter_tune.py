import os
import json
import numpy as np
from typing import Dict, Any, Callable
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history

import logging
logger = logging.getLogger(__name__)

def create_optuna_objective(model_type: str, X: np.ndarray, y: np.ndarray) -> Callable:
    """Returns an objective function for Optuna."""
    
    def objective(trial):
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': 42,
                'n_jobs': -1
            }
            clf = RandomForestClassifier(**params)
        elif model_type == 'xgb':
            # Calculate class weight ratio for XGBoost
            n_neg = np.sum(y == 0)
            n_pos = max(1, np.sum(y == 1))
            scale_pos_weight = n_neg / n_pos
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
            clf = xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        # We use simple CV inside the trial
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(pipeline, X, y, cv=3, scoring='f1', n_jobs=-1)
        return scores.mean()
        
    return objective

def tune_random_forest(X: np.ndarray, y: np.ndarray, n_trials: int = 20) -> optuna.Study:
    """Uses Optuna to tune Random Forest."""
    logger.info("Starting Optuna optimization for Random Forest...")
    study = optuna.create_study(direction='maximize', study_name='rf_tuning')
    objective = create_optuna_objective('rf', X, y)
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best RF trial: F1 = {study.best_value:.4f}")
    logger.info(f"Best RF params: {study.best_params}")
    return study

def tune_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 20) -> optuna.Study:
    """Uses Optuna to tune XGBoost."""
    logger.info("Starting Optuna optimization for XGBoost...")
    study = optuna.create_study(direction='maximize', study_name='xgb_tuning')
    objective = create_optuna_objective('xgb', X, y)
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best XGB trial: F1 = {study.best_value:.4f}")
    logger.info(f"Best XGB params: {study.best_params}")
    return study

def grid_search_rf(X: np.ndarray, y: np.ndarray) -> Dict:
    """Fallback using GridSearchCV for Random Forest."""
    logger.info("Running GridSearchCV for Random Forest...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__class_weight': ['balanced', None]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    
    logger.info(f"Best RF Grid params: {grid.best_params_}")
    return {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'best_estimator': grid.best_estimator_
    }

def grid_search_xgb(X: np.ndarray, y: np.ndarray) -> Dict:
    """Fallback using GridSearchCV for XGBoost."""
    logger.info("Running GridSearchCV for XGBoost...")
    
    n_neg = np.sum(y == 0)
    n_pos = max(1, np.sum(y == 1))
    scale_pos_weight = n_neg / n_pos
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [4, 6, 8],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    
    logger.info(f"Best XGB Grid params: {grid.best_params_}")
    return {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'best_estimator': grid.best_estimator_
    }

def save_tuning_results(study: optuna.Study, output_path: str):
    """Saves best parameters from an Optuna study."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def visualize_optimization_history(study: optuna.Study, output_path: str):
    """Plots and saves optimization history."""
    try:
        fig = plot_optimization_history(study)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_image(output_path)
    except Exception as e:
        logger.error(f"Failed to visualize optimization history: {e}")

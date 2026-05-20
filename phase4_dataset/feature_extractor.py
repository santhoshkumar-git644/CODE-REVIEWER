import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict

# Add project root to path to import phase1 modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from phase1_static_analyzer.metrics import compute_all_metrics
    from phase1_static_analyzer.parser import parse_code
    from phase1_static_analyzer.ast_walker import walk_tree, find_functions, find_loops, find_conditions, find_variables
except ImportError:
    # Handle case where phase1 isn't fully set up during testing
    pass

def extract_features(code: str, language: str = 'python') -> dict:
    """Extracts all AST and lexical features from a code string."""
    features = {
        'loop_count': 0,
        'condition_count': 0,
        'function_count': 0,
        'avg_function_length': 0.0,
        'max_nesting_depth': 0,
        'cyclomatic_complexity': 0,
        'line_count': 0,
        'comment_ratio': 0.0,
        'import_count': 0,
        'variable_count': 0,
        'has_recursion': 0,
        'string_literal_count': 0,
        'numeric_literal_count': 0,
        'operator_count': 0,
        'error': False
    }
    
    if not code or not code.strip():
        features['error'] = True
        return features
        
    try:
        # Lexical features
        lines = code.split('\n')
        features['line_count'] = len(lines)
        
        comments = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//'))
        features['comment_ratio'] = comments / max(1, len(lines))
        
        # We try to use phase1 module if available
        if 'compute_all_metrics' in globals():
            metrics = compute_all_metrics(code, language)
            features['function_count'] = metrics.function_count
            features['max_nesting_depth'] = metrics.max_nesting_depth
            features['avg_function_length'] = metrics.avg_function_length
            features['cyclomatic_complexity'] = metrics.cyclomatic_complexity
            
            tree = parse_code(code, language)
            if tree:
                root = tree.root_node if hasattr(tree, 'root_node') else tree
                features['loop_count'] = len(find_loops(root))
                features['condition_count'] = len(find_conditions(root))
                features['variable_count'] = len(find_variables(root))
        else:
            # Fallback for simple estimation if phase1 is missing
            features['cyclomatic_complexity'] = code.count('if ') + code.count('for ') + code.count('while ') + 1
            features['loop_count'] = code.count('for ') + code.count('while ')
            features['condition_count'] = code.count('if ') + code.count('elif ')
            features['function_count'] = code.count('def ') + code.count('function ')
            features['max_nesting_depth'] = max([len(line) - len(line.lstrip()) for line in lines]) // 4
        
        # Simple text-based features
        features['has_recursion'] = 1 if 'return' in code and features['function_count'] > 0 else 0
        features['import_count'] = code.count('import ') + code.count('from ')
        features['operator_count'] = sum(code.count(op) for op in ['+', '-', '*', '/', '==', '!=', '>', '<'])
        features['string_literal_count'] = code.count('"') // 2 + code.count("'") // 2
        features['numeric_literal_count'] = sum(1 for word in code.split() if word.isdigit())
        
    except Exception as e:
        features['error'] = True
        
    return features

def extract_features_batch(samples: List[dict]) -> pd.DataFrame:
    """Extracts features for a batch of code samples.
    Expects samples to have 'func' (code) and optionally 'target' and 'language'.
    """
    results = []
    
    for i, sample in enumerate(samples):
        code = sample.get('func', '')
        language = sample.get('language', 'python')
        target = sample.get('target', None)
        
        features = extract_features(code, language)
        
        # Add metadata and labels
        row = {'id': sample.get('id', i)}
        row.update(features)
        if target is not None:
            row['target'] = target
            
        results.append(row)
        
    return pd.DataFrame(results)

def validate_features(features: dict) -> bool:
    """Checks if extracted features are valid."""
    if features.get('error', False):
        return False
    if features.get('line_count', 0) == 0:
        return False
    return True

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values and outliers in feature DataFrame."""
    # Drop rows with parsing errors
    if 'error' in df.columns:
        df = df[~df['error']].copy()
        df = df.drop(columns=['error'])
        
    # Fill NaN values
    df = df.fillna(0)
    
    # Clip extreme outliers (e.g., 99th percentile)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['id', 'target']:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)
            
    return df

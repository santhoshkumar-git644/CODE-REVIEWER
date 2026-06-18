from transformers import AutoTokenizer
from typing import List, Dict, Any
import numpy as np

import logging
logger = logging.getLogger(__name__)

def load_tokenizer(model_name: str = 'microsoft/codebert-base') -> AutoTokenizer:
    """Loads the CodeBERT tokenizer."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_code(code: str, tokenizer: AutoTokenizer, max_length: int = 512) -> Dict[str, Any]:
    """Tokenizes a single code string."""
    if not code:
        code = ""
        
    return tokenizer(
        code,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )

def batch_tokenize(codes: List[str], tokenizer: AutoTokenizer, max_length: int = 512, batch_size: int = 32) -> Dict[str, Any]:
    """Tokenizes a batch of code strings."""
    logger.info(f"Tokenizing {len(codes)} code snippets...")
    
    # Replace None with empty string
    safe_codes = [c if c is not None else "" for c in codes]
    
    return tokenizer(
        safe_codes,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )

def truncate_code(code: str, max_tokens: int = 512) -> str:
    """Smart truncation of code preserving structure."""
    if not code:
        return ""
        
    # Simple heuristic: code lines are usually around 10 tokens on average
    # We keep the start and end of the function if it's too long
    lines = code.split('\n')
    estimated_tokens = len(code.split()) * 1.5
    
    if estimated_tokens <= max_tokens:
        return code
        
    # If too long, keep the first 60% and last 40% of lines that fit
    max_lines = int(max_tokens / 15) # conservative estimate
    if len(lines) <= max_lines:
        return code
        
    head_lines = int(max_lines * 0.6)
    tail_lines = max_lines - head_lines
    
    head = '\n'.join(lines[:head_lines])
    tail = '\n'.join(lines[-tail_lines:])
    
    return f"{head}\n... [TRUNCATED] ...\n{tail}"

def get_token_statistics(codes: List[str], tokenizer: AutoTokenizer) -> Dict[str, float]:
    """Calculates token length distribution for a dataset."""
    safe_codes = [c if c is not None else "" for c in codes]
    
    # Tokenize without padding or truncation to get true lengths
    encodings = tokenizer(safe_codes, truncation=False, padding=False)
    lengths = [len(ids) for ids in encodings['input_ids']]
    
    stats = {
        'min': float(np.min(lengths)),
        'max': float(np.max(lengths)),
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'p90': float(np.percentile(lengths, 90)),
        'p95': float(np.percentile(lengths, 95)),
        'p99': float(np.percentile(lengths, 99))
    }
    
    return stats

def create_attention_mask(input_ids, pad_token_id: int):
    """Creates attention mask from input_ids."""
    import torch
    return (input_ids != pad_token_id).long()

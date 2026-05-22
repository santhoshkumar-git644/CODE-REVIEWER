import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Union
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def load_codebert_model(model_name: str = 'microsoft/codebert-base') -> AutoModel:
    """Loads the CodeBERT model to the best available device."""
    logger.info(f"Loading model: {model_name}")
    device = get_device()
    logger.info(f"Using device: {device}")
    
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval() # Set to evaluation mode
    
    return model

def extract_cls_embedding(code: str, model: AutoModel, tokenizer: AutoTokenizer) -> np.ndarray:
    """Extracts the [CLS] token embedding for a single code string."""
    device = next(model.parameters()).device
    
    inputs = tokenizer(
        code if code else "",
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.last_hidden_state shape: [batch_size, sequence_length, hidden_size]
        # [CLS] token is at index 0
        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        
    return cls_embedding

def extract_mean_pooled_embedding(code: str, model: AutoModel, tokenizer: AutoTokenizer) -> np.ndarray:
    """Extracts mean-pooled embeddings across all tokens."""
    device = next(model.parameters()).device
    
    inputs = tokenizer(
        code if code else "",
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
    return mean_pooled[0].cpu().numpy()

def extract_embeddings_batch(codes: List[str], model: AutoModel, tokenizer: AutoTokenizer, 
                           batch_size: int = 16, pooling: str = 'cls') -> np.ndarray:
    """Extracts embeddings for a batch of code strings."""
    device = next(model.parameters()).device
    all_embeddings = []
    
    logger.info(f"Extracting embeddings for {len(codes)} samples in batches of {batch_size}")
    
    for i in tqdm(range(0, len(codes), batch_size), desc="Extracting"):
        batch_codes = [c if c is not None else "" for c in codes[i:i+batch_size]]
        
        inputs = tokenizer(
            batch_codes,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            if pooling == 'cls':
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif pooling == 'mean':
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
                
            all_embeddings.append(embeddings)
            
    # Concatenate all batches
    if all_embeddings:
        return np.vstack(all_embeddings)
    return np.array([])

def save_embeddings(embeddings: np.ndarray, file_ids: List[str], output_path: str):
    """Saves embeddings and their corresponding IDs to a numpy archive."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if not output_path.endswith('.npz'):
        output_path += '.npz'
        
    np.savez_compressed(output_path, embeddings=embeddings, ids=file_ids)
    logger.info(f"Saved {len(file_ids)} embeddings to {output_path}")

def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """Loads saved embeddings from a numpy archive."""
    if not path.endswith('.npz'):
        path += '.npz'
        
    data = np.load(path)
    return data['embeddings'], data['ids'].tolist()

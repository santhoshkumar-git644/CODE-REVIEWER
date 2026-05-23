import os
import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import List, Dict, Any, Tuple

import logging
logger = logging.getLogger(__name__)

def compute_metrics(pred):
    """Computes metrics for HuggingFace Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5
        
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': auc
    }

def create_code_dataset(codes: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 512) -> Dataset:
    """Creates a HuggingFace Dataset from code strings and labels."""
    logger.info(f"Creating dataset with {len(codes)} samples")
    
    # Clean None values
    safe_codes = [c if c is not None else "" for c in codes]
    
    # Tokenize
    encodings = tokenizer(
        safe_codes,
        truncation=True,
        max_length=max_length,
        padding=False # We use DataCollatorWithPadding instead of padding here
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    return dataset

def create_training_args(output_dir: str, epochs: int = 3, batch_size: int = 8) -> TrainingArguments:
    """Creates standard training arguments for fine-tuning."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        report_to="none" # Disable wandb/tensorboard for simplicity unless configured
    )

def fine_tune_codebert(train_dataset: Dataset, val_dataset: Dataset, 
                       model_name: str = 'microsoft/codebert-base',
                       output_dir: str = './models/fine_tuned_codebert',
                       epochs: int = 3,
                       batch_size: int = 8) -> Trainer:
    """Fine-tunes CodeBERT for sequence classification."""
    logger.info(f"Fine-tuning {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    training_args = create_training_args(output_dir, epochs, batch_size)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    
    trainer.train()
    return trainer

def evaluate_fine_tuned(trainer: Trainer, test_dataset: Dataset) -> Dict[str, float]:
    """Evaluates the fine-tuned model on a test set."""
    logger.info("Evaluating fine-tuned model...")
    results = trainer.evaluate(test_dataset)
    return results

def save_fine_tuned_model(trainer: Trainer, output_dir: str):
    """Saves the fine-tuned model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

def load_fine_tuned_model(model_dir: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Loads a fine-tuned model for inference."""
    logger.info(f"Loading fine-tuned model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, tokenizer

def predict_single_code(code: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> Tuple[int, float]:
    """Predicts vulnerability probability for a single code snippet."""
    device = next(model.parameters()).device
    
    inputs = tokenizer(
        code if code else "",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        prob_buggy = probs[0, 1].item()
        pred_class = 1 if prob_buggy > 0.5 else 0
        
    return pred_class, prob_buggy

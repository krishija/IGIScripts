#!/usr/bin/env python3
"""
DNA Methylation Binary Classifier - Training Script
Loads pre-processed data and fine-tunes a DNABERT model.
"""
import os
import sys
import pickle
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Configuration ---
PROCESSED_DATA_FILE = "data/processed/processed_data.pkl"
MODEL_NAME = "zhihan1996/DNA_bert_6"

def load_preprocessed_data():
    """Load the processed data DataFrame from the pickle file."""
    print(f"Loading pre-processed data from {PROCESSED_DATA_FILE}...")
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"Error: Processed data file not found at '{PROCESSED_DATA_FILE}'")
        print("Please run the 'process_data_once.py' script first.")
        sys.exit(1)
    
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        df = pickle.load(f)
    print(f"Loaded {len(df)} samples.")
    return df

def prepare_dataset_balanced(df):
    """Prepare a balanced Hugging Face Dataset."""
    print("Preparing balanced dataset...")
    methylated = df[df['is_methylated'] == 1]
    unmethylated = df[df['is_methylated'] == 0]
    
    min_samples = min(len(methylated), len(unmethylated))
    print(f"Balancing dataset to {min_samples} samples per class.")
    
    methylated_balanced = methylated.sample(n=min_samples, random_state=42)
    unmethylated_balanced = unmethylated.sample(n=min_samples, random_state=42)
    
    balanced_df = pd.concat([methylated_balanced, unmethylated_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    dataset = Dataset.from_pandas(balanced_df[['sequence', 'is_methylated', 'original_context']])
    return dataset.train_test_split(test_size=0.2, seed=42)

def tokenize_data(dataset, tokenizer):
    """Tokenize the sequences."""
    print("Tokenizing data...")
    def tokenize_function(examples):
        tokenized = tokenizer(examples['sequence'], padding='max_length', truncation=True, max_length=256)
        tokenized['labels'] = examples['is_methylated']
        return tokenized
    
    tokenized_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_test = dataset['test'].map(tokenize_function, batched=True)
    return tokenized_train, tokenized_test

def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def evaluate_by_context(trainer, tokenized_test, dataset_test):
    """Evaluate the model separately for each methylation context."""
    print("\n=== Context-Specific Evaluation ===")
    for context in ['CG', 'CHG', 'CHH']:
        print(f"\nEvaluating {context} context...")
        context_indices = [i for i, ctx in enumerate(dataset_test['original_context']) if ctx == context]
        
        if not context_indices:
            print(f"No {context} samples in test set.")
            continue
            
        context_test_dataset = tokenized_test.select(context_indices)
        results = trainer.evaluate(context_test_dataset)
        
        print(f"  {context} Performance ({len(context_indices)} samples):")
        print(f"    Accuracy: {results['eval_accuracy']:.3f}, F1-Score: {results['eval_f1']:.3f}")

def main():
    """Main training and evaluation function."""
    print("=== DNA Methylation Classifier - Training ===")
    print(f"Model: {MODEL_NAME}")

    df = load_preprocessed_data()
    dataset = prepare_dataset_balanced(df)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train, tokenized_test = tokenize_data(dataset, tokenizer)
    
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: 'Unmethylated', 1: 'Methylated'},
        label2id={'Unmethylated': 0, 'Methylated': 1},
        problem_type="single_label_classification"
    )

    training_args = TrainingArguments(
        output_dir="results/dnabert6",
        num_train_epochs=2,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="logs/dnabert6",
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,
        fp16=True if torch.cuda.is_available() else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    print("\nStarting model training...")
    trainer.train()

    print("\nEvaluating model on overall test set...")
    full_results = trainer.evaluate()
    print(f"Overall Test Set Performance: Accuracy: {full_results['eval_accuracy']:.4f}, F1-Score: {full_results['eval_f1']:.4f}")

    evaluate_by_context(trainer, tokenized_test, dataset['test'])

    trainer.save_model("./final_model")
    print("\nModel saved to ./final_model")
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
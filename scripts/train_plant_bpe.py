#!/usr/bin/env python3
"""
Plant DNABERT-BPE Binary Classifier - Training Script
Loads pre-processed data (data/processed/processed_data.pkl) and fine-tunes zhangtaolab/plant-dnabert-BPE
for binary methylation classification, with context-stratified evaluation.
"""

import os
import pickle
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROCESSED_DATA_FILE = "data/processed/processed_data.pkl"
MODEL_NAME = "zhangtaolab/plant-dnabert-BPE"


def load_preprocessed_data():
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_FILE}. Run process_data_once.py first.")
    with open(PROCESSED_DATA_FILE, "rb") as f:
        df = pickle.load(f)
    return df


def prepare_dataset_balanced(df: pd.DataFrame):
    methylated = df[df['is_methylated'] == 1]
    unmethylated = df[df['is_methylated'] == 0]
    n = min(len(methylated), len(unmethylated))
    methylated_balanced = methylated.sample(n=n, random_state=42)
    unmethylated_balanced = unmethylated.sample(n=n, random_state=42)
    balanced_df = pd.concat([methylated_balanced, unmethylated_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset = Dataset.from_pandas(balanced_df[['sequence', 'is_methylated', 'original_context']])
    return dataset.train_test_split(test_size=0.2, seed=42)


def tokenize_data(dataset, tokenizer):
    def tokenize_function(examples):
        tok = tokenizer(examples['sequence'], padding='max_length', truncation=True, max_length=256)
        tok['labels'] = examples['is_methylated']
        return tok
    tokenized_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_test = dataset['test'].map(tokenize_function, batched=True)
    return tokenized_train, tokenized_test


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def evaluate_by_context(trainer, tokenized_test, test_df):
    print("\n=== Context-Specific Evaluation ===")
    for ctx in ['CG', 'CHG', 'CHH']:
        idx = [i for i, c in enumerate(test_df['original_context']) if c == ctx]
        if not idx:
            print(f"No {ctx} samples in test set.")
            continue
        results = trainer.evaluate(tokenized_test.select(idx))
        print(f"  {ctx} ({len(idx)} samples) -> Acc: {results['eval_accuracy']:.3f}, F1: {results['eval_f1']:.3f}")


def main():
    print("=== Plant DNABERT-BPE - Training ===")
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    df = load_preprocessed_data()
    dataset = prepare_dataset_balanced(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenized_train, tokenized_test = tokenize_data(dataset, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type='single_label_classification'
    )

    training_args = TrainingArguments(
        output_dir="results/plant_bpe",
        num_train_epochs=2,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="logs/plant_bpe",
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
        tokenizer=tokenizer,
    )

    print("\nStarting model training...")
    trainer.train()

    print("\nEvaluating on overall test set...")
    overall = trainer.evaluate()
    print(f"Overall -> Acc: {overall['eval_accuracy']:.4f}, F1: {overall['eval_f1']:.4f}")

    # Context evaluation
    evaluate_by_context(trainer, tokenized_test, dataset['test'].to_pandas())

    trainer.save_model("models/saved/plant_bpe")
    print("\nSaved model to models/saved/plant_bpe")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified fine-tuning on MuLan-Methyl processed TSVs.

- Expects TSVs in a structure like:
  {data_root}/{split}/processed_{mark}.tsv where split in {train,test}
  Columns: id, text, label

- Supports multiple models:
  * DNABERT-6 (zhihan1996/DNA_bert_6)
  * Plant-DNABERT-BPE (zhangtaolab/plant-dnabert-BPE)
  * HyenaDNA (LongSafari/hyenadna-large-1m-seqlen-hf)
  * GENA-LM (AIRI-Institute/gena-lm-bert-base-t2t)

- Trains a binary classifier for each (model, mark) pair and reports AUPRC, F1

Usage example:
  python scripts/finetune_from_tsv.py \
    --data_root data/benchmark/processed_dataset \
    --models DNABERT-6 Plant-DNABERT-BPE HyenaDNA GENA-LM \
    --marks 4mC 5hmC 6mA \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --output_csv results/benchmark_results.csv
"""

import os
import re
import sys
import json
import time
import math
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

warnings.filterwarnings("ignore")


MODEL_CATALOG: Dict[str, str] = {
    "DNABERT-6": "zhihan1996/DNA_bert_6",
    "Plant-DNABERT-BPE": "zhangtaolab/plant-dnabert-BPE",
    "HyenaDNA": "LongSafari/hyenadna-large-1m-seqlen-hf",
    "GENA-LM": "AIRI-Institute/gena-lm-bert-base-t2t",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DNA models on MuLan-Methyl TSVs")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing train/test TSVs (processed_dataset)")
    parser.add_argument("--models", nargs="+", default=["DNABERT-6", "Plant-DNABERT-BPE"],
                        help="List of model keys from MODEL_CATALOG")
    parser.add_argument("--marks", nargs="+", default=["4mC", "5hmC", "6mA"],
                        help="List of methylation marks to train on")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--output_csv", type=str, default="results/benchmark_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Subsample training set for quick runs")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Subsample eval set for quick runs")
    return parser.parse_args()


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_sequence_text(text: str) -> str:
    """Extract the k-mer sequence segment from MuLan-Methyl sentence.
    Example: "The DNA sequence is AAA... . For this organism, ..."
    Returns the content between the fixed prefix and the first period.
    Falls back to original text if pattern not found.
    """
    if not isinstance(text, str):
        return str(text)
    m = re.search(r"The DNA sequence is\s+(.*?)\.", text)
    if m:
        return m.group(1).strip()
    return text


def load_split(data_root: str, mark: str, split: str) -> pd.DataFrame:
    path = os.path.join(data_root, split, f"processed_{mark}.tsv")
    df = pd.read_csv(path, sep="\t")
    # Ensure required cols
    assert {"text", "label"}.issubset(df.columns), f"Missing columns in {path}"
    # Extract only the sequence segment for DNA models
    df = df.copy()
    df["sequence"] = df["text"].apply(extract_sequence_text)
    # Normalize labels to int 0/1
    df["labels"] = df["label"].astype(int)
    return df[["sequence", "labels"]]


def make_hf_dataset(df: pd.DataFrame) -> Dict[str, List]:
    return {"sequence": df["sequence"].tolist(), "labels": df["labels"].tolist()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits[:, 1] > logits[:, 0]).astype(int)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    # AUPRC uses scores for positive class
    probs_pos = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
    auprc = average_precision_score(labels, probs_pos)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
    }


def tokenize_function(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["sequence"],
        max_length=max_length,
        padding=False,
        truncation=True,
    )


def train_one_model_mark(model_key: str, model_id: str, mark: str, args: argparse.Namespace, device: torch.device) -> Tuple[Dict, Dict]:
    # Load data
    train_df = load_split(args.data_root, mark, "train")
    test_df = load_split(args.data_root, mark, "test")

    if args.max_train_samples is not None:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed)
    if args.max_eval_samples is not None:
        test_df = test_df.sample(n=min(args.max_eval_samples, len(test_df)), random_state=args.seed)

    # Tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            problem_type="single_label_classification",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    except Exception as e:
        # If the model does not provide a classification head with AutoModelForSequenceClassification,
        # surface the error and skip in the caller.
        raise RuntimeError(f"Failed to create classification model for {model_key} ({model_id}): {e}")

    model.to(device)

    # Build HF datasets
    train_ds = make_hf_dataset(train_df)
    test_ds = make_hf_dataset(test_df)

    # Map-style datasets using lambda wrappers
    from datasets import Dataset
    ds_train = Dataset.from_dict(train_ds)
    ds_test = Dataset.from_dict(test_ds)

    def tok_fn(batch):
        return tokenize_function(batch, tokenizer, args.max_length)

    ds_train = ds_train.map(tok_fn, batched=True, remove_columns=["sequence"])  # keep labels
    ds_test = ds_test.map(tok_fn, batched=True, remove_columns=["sequence"])    # keep labels

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training setup
    out_dir = os.path.join(args.output_dir, f"{model_key.replace(' ', '_')}_{mark}")
    log_dir = os.path.join(args.logging_dir, f"{model_key.replace(' ', '_')}_{mark}")
    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
        greater_is_better=True,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        logging_dir=log_dir,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    # Extract metrics of interest
    metrics = {
        "Model": model_key,
        "HF_ID": model_id,
        "Mark": mark,
        "Train_Steps": int(train_result.global_step),
        "Eval_Accuracy": float(eval_result.get("eval_accuracy", float("nan"))),
        "Eval_F1": float(eval_result.get("eval_f1", float("nan"))),
        "Eval_AUPRC": float(eval_result.get("eval_auprc", float("nan"))),
    }

    # Persist metrics JSON per run
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Free memory
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return metrics, eval_result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    device = detect_device()
    print(f"Device: {device}")

    all_results: List[Dict] = []

    for model_key in args.models:
        if model_key not in MODEL_CATALOG:
            print(f"[WARN] Unknown model key '{model_key}', skipping.")
            continue
        model_id = MODEL_CATALOG[model_key]

        for mark in args.marks:
            print(f"\n=== Fine-tuning {model_key} on {mark} ===")
            try:
                metrics, _ = train_one_model_mark(model_key, model_id, mark, args, device)
                print(f"-> Eval AUPRC: {metrics['Eval_AUPRC']:.4f} | F1: {metrics['Eval_F1']:.4f}")
                all_results.append(metrics)
            except Exception as e:
                print(f"[ERROR] Failed: {model_key} x {mark}: {e}")
                all_results.append({
                    "Model": model_key,
                    "HF_ID": model_id,
                    "Mark": mark,
                    "Train_Steps": 0,
                    "Eval_Accuracy": float("nan"),
                    "Eval_F1": float("nan"),
                    "Eval_AUPRC": float("nan"),
                })

    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved aggregated results to: {args.output_csv}")
        print(df[["Model", "Mark", "Eval_AUPRC", "Eval_F1"]])
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()



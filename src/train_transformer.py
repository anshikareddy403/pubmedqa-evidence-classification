"""Fine-tune Hugging Face encoders for 3-class PubMedQA evidence classification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

from .config import ID2LABEL, LABEL2ID, LABELS, RunConfig
from .evaluate import compute_metrics as sk_compute_metrics
from .utils import set_seed, timer


def _encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label_id"] = out["label"].map(LABEL2ID)
    return out


def build_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize text column; attach numeric labels."""
    df = _encode_labels(df)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds = Dataset.from_pandas(df[["text", "label_id"]])
    ds = ds.rename_column("label_id", "labels")
    return ds.map(tokenize_batch, batched=True, remove_columns=["text"])


def _metrics_from_eval(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = sk_compute_metrics(labels, preds)
    return {
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
    }


def train_transformer(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: RunConfig,
    output_dir: Path,
    run_name: str,
) -> Tuple[Trainer, Dict[str, Any]]:
    """
    Fine-tune a sequence classification head with early stopping on val macro-F1.

    Supports resume if output_dir contains a prior checkpoint (Trainer auto-detects).
    """
    set_seed(config.seed)
    hf_set_seed(config.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {"model_name": model_name, "run_name": run_name}

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = build_tokenized_dataset(train_df, tokenizer, config.max_length)
    val_ds = build_tokenized_dataset(val_df, tokenizer, config.max_length)

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.effective_num_epochs(),
        per_device_train_batch_size=config.effective_batch_size(),
        per_device_eval_batch_size=config.effective_batch_size(),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        save_total_limit=2,
        seed=config.seed,
        report_to="none",
        fp16=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=_metrics_from_eval,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    resume = any(output_dir.glob("checkpoint-*"))
    with timer("transformer_training") as t:
        trainer.train(resume_from_checkpoint=resume)
    meta["training"] = dict(t)
    meta["resumed_from_checkpoint"] = resume

    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    meta["best_model_dir"] = str(output_dir / "best")

    return trainer, meta


def load_transformer_for_inference(model_dir: Path):
    """Load saved classifier and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def predict_transformer(
    model,
    tokenizer,
    device: torch.device,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return predicted label ids and confidence (softmax max)."""
    all_preds: List[int] = []
    all_conf: List[float] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            proba = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(proba, dim=-1)
        all_preds.extend(pred.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())

    return np.array(all_preds), np.array(all_conf)

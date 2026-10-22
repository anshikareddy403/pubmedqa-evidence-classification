"""Load PubMedQA pqa_labeled and build question + abstract inputs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
from datasets import Dataset, load_dataset

from .config import (
    HF_CONFIG,
    HF_DATASET_NAME,
    OFFICIAL_TEST_GROUND_TRUTH_URL,
    RunConfig,
)


def context_to_abstract(context: Dict[str, Any]) -> str:
    """Join PubMedQA context sentences into a single abstract string."""
    contexts = context.get("contexts") or []
    if not contexts:
        return ""
    return " ".join(str(c).strip() for c in contexts if c)


def build_input_text(question: str, abstract: str) -> str:
    """Combine question and abstract for classification."""
    q = (question or "").strip()
    a = (abstract or "").strip()
    return f"question: {q} abstract: {a}"


def fetch_official_test_pubids(url: str = OFFICIAL_TEST_GROUND_TRUTH_URL) -> set[int]:
    """Download official PubMedQA test set pubids (500 expert-labelled)."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    ground_truth = resp.json()
    return {int(pubid) for pubid in ground_truth.keys()}


def load_labeled_dataframe(config: RunConfig | None = None) -> pd.DataFrame:
    """
    Load expert-labelled PubMedQA only (pqa_labeled, 1000 examples).

    Does not use pqa_artificial (211k) for training or evaluation.
    """
    config = config or RunConfig()
    ds = load_dataset(HF_DATASET_NAME, HF_CONFIG)
    rows: List[Dict[str, Any]] = []
    for ex in ds["train"]:
        abstract = context_to_abstract(ex["context"])
        rows.append(
            {
                "pubid": int(ex["pubid"]),
                "question": ex["question"],
                "abstract": abstract,
                "text": build_input_text(ex["question"], abstract),
                "label": ex["final_decision"].lower().strip(),
                "long_answer": ex.get("long_answer", ""),
            }
        )
    df = pd.DataFrame(rows)
    invalid = ~df["label"].isin(["yes", "no", "maybe"])
    if invalid.any():
        raise ValueError(f"Unexpected labels: {df.loc[invalid, 'label'].unique()}")

    # Optional cap for custom experiments only (not used for smoke_test — see load_train_test).
    if config.max_samples is not None and len(df) > config.max_samples:
        df = df.sample(n=config.max_samples, random_state=config.seed).reset_index(drop=True)

    return df


def official_train_test_split(
    df: pd.DataFrame,
    test_pubids: set[int] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split using official PubMedQA test pubids (500 test / 500 train).

    See: https://github.com/pubmedqa/pubmedqa — data/test_ground_truth.json
    """
    if test_pubids is None:
        test_pubids = fetch_official_test_pubids()

    is_test = df["pubid"].isin(test_pubids)
    test_df = df.loc[is_test].copy()
    train_df = df.loc[~is_test].copy()

    missing = test_pubids - set(test_df["pubid"].tolist())
    if missing:
        raise ValueError(
            f"Official test split missing {len(missing)} pubids from pqa_labeled. "
            "Dataset version may differ from the paper split."
        )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_train_test(config: RunConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load labelled data and return official train/test DataFrames."""
    config = config or RunConfig()
    df = load_labeled_dataframe(config)
    train_df, test_df = official_train_test_split(df)

    # CPU smoke mode: subsample after official split so test pubids stay valid.
    if config.smoke_test:
        train_n = min(80, len(train_df))
        test_n = min(40, len(test_df))
        train_df = train_df.sample(n=train_n, random_state=config.seed).reset_index(drop=True)
        test_df = test_df.sample(n=test_n, random_state=config.seed).reset_index(drop=True)

    return train_df, test_df


def dataframe_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Convert project DataFrame to Hugging Face Dataset for Trainer."""
    return Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))

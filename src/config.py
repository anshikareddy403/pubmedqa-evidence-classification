"""Central configuration and reproducibility settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Project root: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Hugging Face dataset (expert-labelled subset only)
HF_DATASET_NAME = "qiaojin/PubMedQA"
HF_CONFIG = "pqa_labeled"

# Official test pubids (500) from https://github.com/pubmedqa/pubmedqa
OFFICIAL_TEST_GROUND_TRUTH_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
)

LABELS: List[str] = ["yes", "no", "maybe"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Primary biomedical encoder (abstract + full-text pretraining on PubMed)
BIOMED_MODEL_NAME = (
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
GENERAL_MODEL_NAME = "bert-base-uncased"

DEFAULT_SEED = 42


@dataclass
class RunConfig:
    """Runtime flags for training and evaluation."""

    seed: int = DEFAULT_SEED
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    # Smoke / CPU-friendly mode: subsample and fewer epochs
    smoke_test: bool = field(
        default_factory=lambda: os.environ.get("SMOKE_TEST", "0") == "1"
    )
    max_samples: int | None = field(default=None)  # if set, cap total examples
    smoke_max_samples: int = 120
    # Transformer training
    max_length: int = 512
    batch_size: int = 8
    smoke_batch_size: int = 4
    num_epochs: int = 3
    smoke_num_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    # Baseline
    tfidf_max_features: int = 50000
    smoke_tfidf_max_features: int = 5000

    def effective_max_samples(self) -> int | None:
        if self.max_samples is not None:
            return self.max_samples
        if self.smoke_test:
            return self.smoke_max_samples
        return None

    def effective_batch_size(self) -> int:
        return self.smoke_batch_size if self.smoke_test else self.batch_size

    def effective_num_epochs(self) -> int:
        return self.smoke_num_epochs if self.smoke_test else self.num_epochs

    def effective_tfidf_max_features(self) -> int:
        return (
            self.smoke_tfidf_max_features
            if self.smoke_test
            else self.tfidf_max_features
        )

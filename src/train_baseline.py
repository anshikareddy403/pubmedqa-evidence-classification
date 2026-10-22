"""TF-IDF + multinomial logistic regression baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import LABEL2ID, RunConfig
from .sklearn_compat import ensure_baseline_pipeline_compat
from .utils import set_seed, timer


def build_baseline_pipeline(config: RunConfig) -> Pipeline:
    """Create sklearn pipeline with TF-IDF and 3-class logistic regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=config.effective_tfidf_max_features(),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.seed,
                    multi_class="multinomial",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_baseline(
    train_df: pd.DataFrame,
    config: RunConfig,
    model_path: Path | None = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Fit baseline on training texts; optionally persist pipeline."""
    set_seed(config.seed)
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].map(LABEL2ID).tolist()

    pipe = build_baseline_pipeline(config)
    meta: Dict[str, Any] = {"model_name": "tfidf_logistic_regression"}

    with timer("baseline_training") as t:
        pipe.fit(X_train, y_train)
    meta["training"] = dict(t)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_path)
        meta["saved_path"] = str(model_path)

    return pipe, meta


def load_baseline_pipeline(model_path: Path) -> Pipeline:
    """Load baseline joblib and patch sklearn version mismatches."""
    pipe = joblib.load(model_path)
    return ensure_baseline_pipeline_compat(pipe)


def predict_baseline(pipe: Pipeline, texts: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return predicted class ids and max prob scores."""
    pipe = ensure_baseline_pipeline_compat(pipe)
    pred_ids = pipe.predict(texts)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(texts)
        conf = proba.max(axis=1)
    else:
        conf = np.ones(len(pred_ids))
    return np.asarray(pred_ids), np.asarray(conf)

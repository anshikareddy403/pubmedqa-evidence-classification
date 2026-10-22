"""Metrics, confusion matrices, error analysis, and results export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import ID2LABEL, LABEL2ID, LABELS, RunConfig
from .utils import ensure_dir, save_json


def labels_to_ids(labels: Sequence[str]) -> np.ndarray:
    return np.array([LABEL2ID[l] for l in labels])


def ids_to_labels(ids: Sequence[int]) -> List[str]:
    return [ID2LABEL[int(i)] for i in ids]


def compute_metrics(
    y_true: Sequence[str] | Sequence[int],
    y_pred: Sequence[str] | Sequence[int],
) -> Dict[str, Any]:
    """Accuracy, macro-F1, weighted-F1, and sklearn classification report."""
    if isinstance(y_true[0], str):
        yt = labels_to_ids(y_true)  # type: ignore[index]
        yp = labels_to_ids(y_pred)  # type: ignore[index]
    else:
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)

    report = classification_report(
        yt, yp, target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(yt, yp, labels=list(range(len(LABELS)))).tolist(),
    }


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_path: Path,
    title: str = "Confusion matrix",
) -> None:
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def maybe_class_error_analysis(
    test_df: pd.DataFrame,
    y_pred: Sequence[str],
    out_path: Path,
    max_rows: int = 50,
) -> pd.DataFrame:
    """
    Emphasize errors involving the minority 'maybe' class.

    Includes false negatives/positives for maybe and confusions yes/no ↔ maybe.
    """
    df = test_df.copy()
    df["predicted"] = list(y_pred)
    df["correct"] = df["label"] == df["predicted"]

    maybe_mask = (df["label"] == "maybe") | (df["predicted"] == "maybe")
    errors = df.loc[maybe_mask & ~df["correct"]].head(max_rows)
    errors.to_csv(out_path, index=False)
    return errors


def measure_inference_latency(
    predict_fn,
    texts: List[str],
    n_warmup: int = 3,
    repeat: int = 1,
) -> Dict[str, float]:
    """Wall-clock latency per example (seconds), single-threaded batch=1."""
    import time

    sample = texts[: min(20, len(texts))]
    for t in sample[:n_warmup]:
        predict_fn([t])

    times: List[float] = []
    for t in sample:
        start = time.perf_counter()
        predict_fn([t])
        times.append(time.perf_counter() - start)

    arr = np.array(times)
    return {
        "n_samples": len(times),
        "mean_seconds_per_example": float(arr.mean()),
        "std_seconds_per_example": float(arr.std()),
        "total_seconds": float(arr.sum() * repeat),
    }


def export_run_results(
    run_name: str,
    metrics: Dict[str, Any],
    config: RunConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write JSON (+ flat CSV row) under results/."""
    ensure_dir(config.results_dir)
    payload: Dict[str, Any] = {
        "run_name": run_name,
        "seed": config.seed,
        "smoke_test": config.smoke_test,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)

    json_path = config.results_dir / f"{run_name}_metrics.json"
    save_json(json_path, payload)

    flat = {
        "run_name": run_name,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "smoke_test": config.smoke_test,
    }
    if "classification_report" in metrics:
        for lab in LABELS:
            flat[f"f1_{lab}"] = metrics["classification_report"].get(lab, {}).get(
                "f1-score"
            )

    csv_path = config.results_dir / f"{run_name}_summary.csv"
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    return json_path

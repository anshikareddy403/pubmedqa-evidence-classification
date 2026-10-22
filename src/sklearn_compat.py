"""Compatibility shims for sklearn models pickled across versions."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def ensure_logistic_regression_compat(clf: LogisticRegression) -> LogisticRegression:
    """
    Old joblib checkpoints may lack `multi_class` after unpickling on a newer sklearn,
    which breaks predict_proba with AttributeError.
    """
    if not isinstance(clf, LogisticRegression):
        return clf

    if not hasattr(clf, "multi_class") or getattr(clf, "multi_class", None) is None:
        clf.multi_class = getattr(clf, "multi_class_", "multinomial")

    # Some pickles miss the public solver name while fitted attrs exist.
    if not hasattr(clf, "solver") or clf.solver is None:
        clf.solver = getattr(clf, "solver_", "lbfgs")

    return clf


def ensure_baseline_pipeline_compat(pipe: Pipeline) -> Pipeline:
    """Patch baseline pipeline after joblib.load."""
    if isinstance(pipe, Pipeline) and "clf" in pipe.named_steps:
        ensure_logistic_regression_compat(pipe.named_steps["clf"])
    return pipe

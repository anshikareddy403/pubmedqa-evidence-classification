#!/usr/bin/env python3
"""
Reproducible end-to-end experiment runner (CPU/GPU).

Usage:
  python run_experiment.py --smoke-test          # fast CPU smoke test
  python run_experiment.py --models baseline     # baseline only
  SMOKE_TEST=1 python run_experiment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without pip install -e .
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    BIOMED_MODEL_NAME,
    GENERAL_MODEL_NAME,
    PROJECT_ROOT,
    RunConfig,
)
from src.data_loading import load_train_test
from src.evaluate import (
    compute_metrics,
    export_run_results,
    ids_to_labels,
    measure_inference_latency,
    maybe_class_error_analysis,
    plot_confusion_matrix,
)
from src.train_baseline import predict_baseline, train_baseline
from src.train_transformer import (
    load_transformer_for_inference,
    predict_transformer,
    train_transformer,
)
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PubMedQA evidence classification")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Subsample data and shorten training (CPU-friendly)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "bert", "pubmedbert"],
        choices=["baseline", "bert", "pubmedbert"],
    )
    p.add_argument(
        "--skip-transformers",
        action="store_true",
        help="Skip HF fine-tuning (baseline only)",
    )
    return p.parse_args()


def evaluate_predictions(
    run_name: str,
    test_df,
    y_pred_labels,
    config: RunConfig,
    training_meta: dict,
) -> None:
    metrics = compute_metrics(test_df["label"].tolist(), y_pred_labels)
    export_run_results(run_name, metrics, config, extra={"training": training_meta})

    from src.config import LABEL2ID

    yt = [LABEL2ID[x] for x in test_df["label"]]
    yp = [LABEL2ID[x] for x in y_pred_labels]
    plot_confusion_matrix(
        yt,
        yp,
        config.results_dir / f"{run_name}_confusion_matrix.png",
        title=run_name,
    )
    maybe_class_error_analysis(
        test_df,
        y_pred_labels,
        config.results_dir / f"{run_name}_maybe_errors.csv",
    )
    print(f"[{run_name}] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


def main() -> None:
    args = parse_args()
    config = RunConfig(seed=args.seed, smoke_test=args.smoke_test)
    set_seed(config.seed)

    train_df, test_df = load_train_test(config)
    # Use a slice of official train as validation for early stopping (stratified by label)
    val_df = train_df.sample(frac=0.15, random_state=config.seed)
    train_fit_df = train_df.drop(val_df.index)

    print(
        f"Loaded pqa_labeled — train_fit={len(train_fit_df)} val={len(val_df)} "
        f"test={len(test_df)} smoke={config.smoke_test}"
    )

    if "baseline" in args.models:
        pipe, meta = train_baseline(
            train_fit_df,
            config,
            model_path=config.checkpoint_dir / "baseline_tfidf_lr.joblib",
        )
        pred_ids, _ = predict_baseline(pipe, test_df["text"].tolist())
        preds = ids_to_labels(pred_ids)
        lat = measure_inference_latency(lambda xs: predict_baseline(pipe, xs)[0], test_df["text"].tolist())
        meta["inference_latency"] = lat
        evaluate_predictions("baseline_tfidf_lr", test_df, preds, config, meta)

    if args.skip_transformers:
        return

    model_specs = []
    if "bert" in args.models:
        model_specs.append(("bert_base", GENERAL_MODEL_NAME))
    if "pubmedbert" in args.models:
        model_specs.append(("pubmedbert", BIOMED_MODEL_NAME))

    for run_name, hf_name in model_specs:
        out = config.checkpoint_dir / run_name
        trainer, meta = train_transformer(
            hf_name,
            train_fit_df,
            val_df,
            config,
            out,
            run_name,
        )
        model, tokenizer, device = load_transformer_for_inference(out / "best")
        pred_ids, _ = predict_transformer(
            model,
            tokenizer,
            device,
            test_df["text"].tolist(),
            max_length=config.max_length,
            batch_size=config.effective_batch_size(),
        )
        preds = ids_to_labels(pred_ids)

        def _predict_one(batch_texts):
            pids, _ = predict_transformer(
                model,
                tokenizer,
                device,
                batch_texts,
                max_length=config.max_length,
                batch_size=1,
            )
            return pids

        meta["inference_latency"] = measure_inference_latency(
            _predict_one, test_df["text"].tolist()
        )
        evaluate_predictions(run_name, test_df, preds, config, meta)


if __name__ == "__main__":
    main()

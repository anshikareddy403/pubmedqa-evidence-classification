"""
PubMedQA evidence classification — Streamlit demo.

  streamlit run pubmedqa_evidence_demo.py --server.port 8510
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LABELS, PROJECT_ROOT, RunConfig  # noqa: E402
from src.data_loading import build_input_text  # noqa: E402
from src.train_baseline import load_baseline_pipeline  # noqa: E402
from src.train_transformer import load_transformer_for_inference, predict_transformer  # noqa: E402

CHECKPOINTS = PROJECT_ROOT / "checkpoints"
BASELINE_PATH = CHECKPOINTS / "baseline_tfidf_lr.joblib"
BERT_PATH = CHECKPOINTS / "bert_base" / "best"
PUBMEDBERT_PATH = CHECKPOINTS / "pubmedbert" / "best"

# Stable ids — do not infer from display-name prefixes (PubMedBERT vs BERT-base).
MODEL_SPECS: list[dict[str, Any]] = [
    {
        "id": "baseline",
        "label": "TF-IDF + Logistic Regression",
        "short": "Baseline",
        "path": BASELINE_PATH,
        "is_dir": False,
    },
    {
        "id": "bert",
        "label": "BERT-base-uncased",
        "short": "BERT-base",
        "path": BERT_PATH,
        "is_dir": True,
    },
    {
        "id": "pubmedbert",
        "label": "PubMedBERT (biomedical)",
        "short": "PubMedBERT",
        "path": PUBMEDBERT_PATH,
        "is_dir": True,
    },
]

EXAMPLE = {
    "question": "Do preoperative statins reduce atrial fibrillation after cardiac surgery?",
    "abstract": (
        "We reviewed randomized trials of statin therapy before cardiac surgery. "
        "Results were mixed; several studies reported no significant reduction in "
        "postoperative atrial fibrillation. Further high-quality evidence is needed."
    ),
}

CUSTOM_CSS = """
<style>
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #0f766e 0%, #115e59 45%, #134e4a 100%);
        color: #f0fdfa;
        margin-bottom: 1rem;
    }
    .hero h1 { color: #f0fdfa !important; font-size: 1.75rem !important; margin: 0 0 0.35rem 0; }
    .hero p { margin: 0; opacity: 0.92; font-size: 0.95rem; }
    .badge-yes { background:#ccfbf1; color:#115e59; padding:0.35rem 0.75rem; border-radius:999px; font-weight:600; }
    .badge-no { background:#fee2e2; color:#991b1b; padding:0.35rem 0.75rem; border-radius:999px; font-weight:600; }
    .badge-maybe { background:#ffedd5; color:#9a3412; padding:0.35rem 0.75rem; border-radius:999px; font-weight:600; }
    .model-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        min-height: 200px;
    }
    .model-card-off {
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
        background: #f8fafc;
        min-height: 200px;
        color: #64748b;
    }
    .model-card h4 { margin-top: 0; color: #0f172a; }
</style>
"""


def _model_ready(spec: dict) -> bool:
    p = spec["path"]
    return p.is_dir() if spec["is_dir"] else p.is_file()


def _checkpoint_status() -> dict[str, bool]:
    return {spec["id"]: _model_ready(spec) for spec in MODEL_SPECS}


@st.cache_resource(show_spinner="Loading baseline…")
def _load_baseline():
    return load_baseline_pipeline(BASELINE_PATH)


@st.cache_resource(show_spinner="Loading BERT-base…")
def _load_bert():
    return load_transformer_for_inference(BERT_PATH)


@st.cache_resource(show_spinner="Loading PubMedBERT…")
def _load_pubmedbert():
    return load_transformer_for_inference(PUBMEDBERT_PATH)


def _proba_baseline(pipe, text: str) -> dict[str, float]:
    proba = pipe.predict_proba([text])[0]
    return {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}


def _proba_transformer(model, tokenizer, device, text: str, max_length: int) -> dict[str, float]:
    import torch

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}


def _predict_label(proba: dict[str, float]) -> str:
    return max(proba, key=proba.get)


def _label_badge(label: str) -> str:
    cls = {"yes": "badge-yes", "no": "badge-no", "maybe": "badge-maybe"}.get(label, "badge-maybe")
    return f'<span class="{cls}">{label.upper()}</span>'


def _run_one_model(model_id: str, text: str, config: RunConfig) -> dict[str, Any]:
    spec = next(s for s in MODEL_SPECS if s["id"] == model_id)
    if model_id == "baseline":
        proba = _proba_baseline(_load_baseline(), text)
    elif model_id == "bert":
        model, tok, device = _load_bert()
        proba = _proba_transformer(model, tok, device, text, config.max_length)
    elif model_id == "pubmedbert":
        model, tok, device = _load_pubmedbert()
        proba = _proba_transformer(model, tok, device, text, config.max_length)
    else:
        raise ValueError(model_id)
    label = _predict_label(proba)
    return {
        "id": model_id,
        "short": spec["short"],
        "label": spec["label"],
        "prediction": label,
        **proba,
    }


def _run_inference(question: str, abstract: str, selected_ids: list[str], status: dict[str, bool]):
    text = build_input_text(question, abstract)
    config = RunConfig()
    results: list[dict] = []
    errors: list[str] = []

    for model_id in selected_ids:
        if not status.get(model_id):
            continue
        try:
            results.append(_run_one_model(model_id, text, config))
        except Exception as exc:
            errors.append(f"**{model_id}:** {exc}")
    for msg in errors:
        st.error(msg)
    return results


def _render_all_model_slots(results: list[dict], status: dict[str, bool]) -> None:
    """Always show three columns — missing checkpoints get a clear placeholder."""
    by_id = {r["id"]: r for r in results}
    cols = st.columns(3)

    for col, spec in zip(cols, MODEL_SPECS):
        mid = spec["id"]
        with col:
            if not status.get(mid):
                st.markdown(
                    f'<div class="model-card-off"><h4>{spec["short"]}</h4>'
                    f"<p><strong>Not loaded</strong></p>"
                    f"<p>Expected under <code>checkpoints/</code>: "
                    f"<code>{'baseline_tfidf_lr.joblib' if mid == 'baseline' else mid + '/best/'}</code></p>"
                    f"<p>Train in Colab (Sections 3–4) and copy this folder into "
                    f"<code>checkpoints/</code> on your PC.</p></div>",
                    unsafe_allow_html=True,
                )
            elif mid in by_id:
                row = by_id[mid]
                st.markdown(
                    f'<div class="model-card"><h4>{row["short"]}</h4>'
                    f'<p>Prediction: {_label_badge(row["prediction"])}</p></div>',
                    unsafe_allow_html=True,
                )
                chart_df = pd.DataFrame(
                    {"probability": [row[l] for l in LABELS]},
                    index=LABELS,
                )
                st.bar_chart(chart_df, color="#0f766e", height=160)
                st.caption(f"Confidence: **{row[row['prediction']]:.1%}**")
            else:
                st.markdown(
                    f'<div class="model-card-off"><h4>{spec["short"]}</h4>'
                    f"<p>Checkpoint found but not run — enable in sidebar.</p></div>",
                    unsafe_allow_html=True,
                )

    if results:
        st.markdown("#### Comparison table")
        table = pd.DataFrame(results)[["short", "prediction", "yes", "no", "maybe"]]
        table = table.rename(columns={"short": "Model"})
        st.dataframe(
            table.style.format({"yes": "{:.2%}", "no": "{:.2%}", "maybe": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
        preds = [r["prediction"] for r in results]
        if len(preds) > 1:
            if len(set(preds)) == 1:
                st.success(f"All running models agree: **{preds[0].upper()}**.")
            else:
                st.info("Models disagree — check **maybe** for inconclusive evidence.")


st.set_page_config(
    page_title="PubMedQA · Evidence Classifier (NLP Project)",
    page_icon="🧬",
    layout="wide",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>PubMedQA Evidence Classifier</h1>
      <p>Biomedical NLP · yes / no / maybe · Anshika Reddy · Applied AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("About this project", expanded=False):
    st.markdown(
        """
        This demo supports a master's **NLP and LLMs** project on **PubMedQA** (expert-labelled subset only).
        Given a biomedical **question** and **abstract**, three models estimate whether the text supports
        **yes**, **no**, or **maybe** (uncertain evidence): a TF-IDF baseline, **BERT-base**, and **PubMedBERT**.
        It is for **research evidence classification**, not clinical diagnosis.
        """
    )

status = _checkpoint_status()
ready_count = sum(status.values())
labels_by_id = {s["id"]: s["label"] for s in MODEL_SPECS}

with st.sidebar:
    st.markdown("### Models (checkpoints)")
    st.caption(f"{ready_count} of 3 ready on this machine")
    for spec in MODEL_SPECS:
        ok = status[spec["id"]]
        st.markdown(f"{'🟢' if ok else '🔴'} **{spec['short']}** — {spec['label']}")

    if ready_count < 3:
        st.warning(
            "Only loaded models can run. Missing folders/files must be copied from Colab after "
            "fine-tuning (see `checkpoints/bert_base/best` and `checkpoints/pubmedbert/best`)."
        )

    st.divider()
    selected = st.multiselect(
        "Run these models",
        options=[s["id"] for s in MODEL_SPECS],
        default=[s["id"] for s in MODEL_SPECS if status[s["id"]]],
        format_func=lambda mid: labels_by_id[mid],
    )
    st.link_button(
        "GitHub repository",
        "https://github.com/anshikareddy403/pubmedqa-evidence-classification",
        use_container_width=True,
    )

st.info("**Not for clinical use.** Research and education only.")

c1, c2 = st.columns(2)
with c1:
    question = st.text_area("Biomedical question", height=140, value=EXAMPLE["question"])
with c2:
    abstract = st.text_area("Abstract text", height=140, value=EXAMPLE["abstract"])

if st.button("Classify evidence", type="primary"):
    if not question.strip() or not abstract.strip():
        st.error("Enter both a question and an abstract.")
    elif not selected:
        st.warning("Select at least one model in the sidebar (with a green checkpoint).")
    else:
        with st.spinner("Running models…"):
            st.session_state["last_results"] = _run_inference(question, abstract, selected, status)

st.divider()
st.subheader("Model outputs")
if ready_count == 0:
    st.error(
        "No checkpoints found. Train in Colab or run "
        "`python run_experiment.py`, then place files under `checkpoints/`."
    )
_render_all_model_slots(st.session_state.get("last_results") or [], status)

"""
Streamlit demo: PubMedQA evidence classification (yes / no / maybe).

Run locally:
  pip install -r requirements.txt
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BIOMED_MODEL_NAME,
    GENERAL_MODEL_NAME,
    LABELS,
    PROJECT_ROOT,
    RunConfig,
)
from src.data_loading import build_input_text  # noqa: E402
from src.train_baseline import predict_baseline  # noqa: E402
from src.train_transformer import load_transformer_for_inference, predict_transformer  # noqa: E402

CHECKPOINTS = PROJECT_ROOT / "checkpoints"
BASELINE_PATH = CHECKPOINTS / "baseline_tfidf_lr.joblib"
BERT_PATH = CHECKPOINTS / "bert_base" / "best"
PUBMEDBERT_PATH = CHECKPOINTS / "pubmedbert" / "best"

MODEL_KEYS = {
    "baseline": "TF-IDF + Logistic Regression",
    "bert": f"BERT-base",
    "pubmedbert": "PubMedBERT",
}

LABEL_COLORS = {
    "yes": "#0d9488",
    "no": "#dc2626",
    "maybe": "#d97706",
}

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
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
    }
    .model-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        min-height: 220px;
    }
    .model-card h4 { margin-top: 0; color: #0f172a; }
</style>
"""


def _checkpoint_status() -> dict[str, bool]:
    return {
        MODEL_KEYS["baseline"]: BASELINE_PATH.is_file(),
        MODEL_KEYS["bert"]: BERT_PATH.is_dir(),
        MODEL_KEYS["pubmedbert"]: PUBMEDBERT_PATH.is_dir(),
    }


@st.cache_resource(show_spinner="Loading TF-IDF baseline…")
def _load_baseline():
    return joblib.load(BASELINE_PATH)


@st.cache_resource(show_spinner="Loading BERT…")
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


def _run_inference(question: str, abstract: str, selected: list[str], status: dict[str, bool]):
    text = build_input_text(question, abstract)
    config = RunConfig()
    results = []

    for model_name in selected:
        if not status.get(model_name):
            continue
        try:
            if model_name.startswith("TF-IDF"):
                proba = _proba_baseline(_load_baseline(), text)
            elif model_name.startswith("BERT-base"):
                model, tok, device = _load_bert()
                proba = _proba_transformer(model, tok, device, text, config.max_length)
            else:
                model, tok, device = _load_pubmedbert()
                proba = _proba_transformer(model, tok, device, text, config.max_length)
            label = _predict_label(proba)
            short = model_name.split()[0]
            if "PubMed" in model_name:
                short = "PubMedBERT"
            elif "BERT" in model_name:
                short = "BERT-base"
            else:
                short = "Baseline"
            results.append({"model": model_name, "short": short, "prediction": label, **proba})
        except Exception as exc:
            st.error(f"**{model_name}:** {exc}")
    return results


def _render_result_cards(results: list[dict]) -> None:
    preds = [r["prediction"] for r in results]
    agree = len(set(preds)) == 1

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Models run", len(results))
    with m2:
        st.metric("Agreement", "Unanimous" if agree else "Split")
    with m3:
        st.metric("Distinct labels", len(set(preds)))

    if agree:
        st.success(f"All models classify evidence as **{preds[0].upper()}**.")
    else:
        st.info("Models disagree — review probabilities, especially for **maybe** (inconclusive evidence).")

    cols = st.columns(len(results))
    for col, row in zip(cols, results):
        with col:
            st.markdown(
                f'<div class="model-card"><h4>{row["short"]}</h4>'
                f'<p>Prediction: {_label_badge(row["prediction"])}</p></div>',
                unsafe_allow_html=True,
            )
            chart_df = pd.DataFrame(
                {"label": list(LABELS), "probability": [row[l] for l in LABELS]}
            ).set_index("label")
            st.bar_chart(chart_df, color="#0f766e", height=180)
            conf = row[row["prediction"]]
            st.caption(f"Top-class confidence: **{conf:.1%}**")

    st.markdown("#### Detailed scores")
    table = pd.DataFrame(results)[["short", "prediction", "yes", "no", "maybe"]]
    table = table.rename(columns={"short": "Model"})
    st.dataframe(
        table.style.format({"yes": "{:.2%}", "no": "{:.2%}", "maybe": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )


st.set_page_config(
    page_title="PubMedQA Evidence Classification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>PubMedQA Evidence Classifier</h1>
      <p>Biomedical NLP · yes / no / maybe · expert-labelled PubMedQA · Anshika Reddy · Applied AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status = _checkpoint_status()
ready_count = sum(status.values())

with st.sidebar:
    st.markdown("## 🧬")
    st.markdown("### System status")
    st.progress(ready_count / max(len(status), 1), text=f"{ready_count}/{len(status)} models ready")
    for name, ok in status.items():
        icon = "🟢" if ok else "🔴"
        st.markdown(f"{icon} **{name.split()[0] if 'TF-IDF' not in name else 'Baseline'}**")
        st.caption(name if len(name) < 40 else name[:37] + "…")

    if ready_count == 0:
        st.error("No checkpoints in `checkpoints/`. Train via Colab or `run_experiment.py`.")

    st.divider()
    selected = st.multiselect(
        "Models to compare",
        options=list(status.keys()),
        default=[k for k, v in status.items() if v],
    )
    st.divider()
    st.link_button(
        "View GitHub repo",
        "https://github.com/anshikareddy403/pubmedqa-evidence-classification",
        use_container_width=True,
    )

tab_classify, tab_help, tab_about = st.tabs(["Classify", "How to use", "About"])

with tab_classify:
    st.markdown(
        "Enter a **research question** and **PubMed abstract**. The system estimates whether the "
        "abstract supports **yes**, **no**, or **maybe** (insufficient / inconclusive evidence)."
    )
    st.info(
        "**Not for clinical use.** Research and education only — not diagnosis or treatment advice."
    )

    c1, c2 = st.columns(2)
    with c1:
        question = st.text_area(
            "Biomedical question",
            height=140,
            value=EXAMPLE["question"],
            placeholder="e.g. Does X improve outcome Y in population Z?",
        )
    with c2:
        abstract = st.text_area(
            "Abstract text",
            height=140,
            value=EXAMPLE["abstract"],
            placeholder="Paste the PubMed abstract or supporting context…",
        )

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        classify = st.button("Classify evidence", type="primary", use_container_width=True)

    if classify:
        if not question.strip() or not abstract.strip():
            st.error("Please provide both a question and an abstract.")
        elif not selected:
            st.warning("Select at least one model in the sidebar.")
        else:
            with st.spinner("Running models…"):
                st.session_state["last_results"] = _run_inference(
                    question, abstract, selected, status
                )
                st.session_state["last_question"] = question
                st.session_state["last_abstract"] = abstract

    if st.session_state.get("last_results"):
        st.divider()
        _render_result_cards(st.session_state["last_results"])

with tab_help:
    st.markdown(
        """
        ### Local setup
        1. Train models (Colab notebook or `python run_experiment.py`).
        2. Place checkpoints under `checkpoints/`.
        3. Run `streamlit run streamlit_app.py`.

        ### Demo script (2–3 min)
        - Explain PubMedQA **expert labels only** (no synthetic eval).
        - Show one question + abstract → three models.
        - Highlight **maybe** when wording is hedged or inconclusive.
        - Mention limitations: English PubMed bias, small training set.

        ### Label meanings
        | Label | Meaning |
        |-------|---------|
        | **yes** | Abstract supports a positive answer |
        | **no** | Abstract supports a negative answer |
        | **maybe** | Evidence is insufficient or uncertain |
        """
    )

with tab_about:
    st.markdown(
        f"""
        ### Project
        Master's individual project comparing classical and transformer models on **PubMedQA**
        (`pqa_labeled`, n≈1,000 expert annotations).

        ### Models
        - **Baseline:** TF-IDF + balanced logistic regression
        - **BERT-base:** `{GENERAL_MODEL_NAME}`
        - **PubMedBERT:** `{BIOMED_MODEL_NAME}`

        ### Ethics
        Human experts remain authoritative for clinical decisions. This UI supports
        **literature triage** and coursework demonstration only.
        """
    )

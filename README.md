# PubMedQA Evidence Classification

Master's project scaffold for **Anshika Reddy**: classify biomedical evidence as **yes / no / maybe** from a research question plus PubMed abstract context.

## Task

- **Input:** `question` + abstract (sentences from `context.contexts` joined)
- **Output:** 3-way label (`final_decision`)
- **Models:**
  1. TF-IDF (1–2 grams) + balanced logistic regression (scikit-learn)
  2. Fine-tuned `bert-base-uncased`
  3. Fine-tuned **`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`** (primary biomedical encoder)

**Biomedical model choice:** PubMedBERT was selected over BioBERT because it is trained on PubMed abstracts and full-text PMC, matching this task’s input domain. BioBERT (`dmis-lab/biobert-base-cased-v1.1`) is a reasonable alternative; swap `BIOMED_MODEL_NAME` in `src/config.py` to compare.

## Dataset (expert labels only)

| Source | Config | Size | Use |
|--------|--------|------|-----|
| [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | `pqa_labeled` | 1,000 | **Training & evaluation** |
| Same repo | `pqa_artificial` | 211k | **Not used** (machine-generated labels) |

Load in Python:

```python
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
```

**Official split:** 500 test questions are defined by pubids in [pubmedqa/pubmedqa `test_ground_truth.json`](https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json). The remaining 500 examples are training pool; this project holds out 15% of train for validation (early stopping) and fits on the rest.

### Small-data caveats

- Only **1,000** expert-labelled examples exist; metrics have high variance.
- Class imbalance: ~55% yes, ~34% no, ~11% maybe — **macro-F1** and **maybe-class errors** are emphasized.
- Do not tune on `pqa_artificial` if the goal is expert-aligned evaluation.

## Ethics & license

- PubMedQA is MIT-licensed; respect [PubMed / PMC terms](https://www.ncbi.nlm.nih.gov/home/about/policies/) when using abstracts.
- This is a **research/education** baseline, not clinical decision support. Do not use model outputs for patient care.

## Setup

```bash
cd pubmedqa-evidence-classification
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Google Colab (optional GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USER/pubmedqa-evidence-classification/blob/main/notebooks/pubmedqa_evidence_classification.ipynb)

Open `notebooks/pubmedqa_evidence_classification.ipynb` (update the badge URL after you push to GitHub). Enable **Runtime → Change runtime type → GPU** for transformer fine-tuning.

## Reproduce

**Full pipeline (GPU recommended for BERT models):**

```bash
python run_experiment.py --seed 42
```

**CPU smoke test** (subsampled train/test, 1 epoch, smaller TF-IDF):

```bash
python run_experiment.py --smoke-test --models baseline
# or train all models slowly:
SMOKE_TEST=1 python run_experiment.py
```

**Baseline only:**

```bash
python run_experiment.py --models baseline --skip-transformers
```

### Outputs

Metrics and artifacts are written under `results/`:

- `{run_name}_metrics.json` — accuracy, macro-F1, classification report, confusion matrix
- `{run_name}_summary.csv` — one-row summary for tables
- `{run_name}_confusion_matrix.png`
- `{run_name}_maybe_errors.csv` — error analysis focused on `maybe`

Checkpoints: `checkpoints/` (gitignored).

### Notebook

`notebooks/pubmedqa_evidence_classification.ipynb` mirrors the pipeline: EDA, training, evaluation, timing comparison, and export to `results/`.

## Project layout

```
pubmedqa-evidence-classification/
├── README.md
├── requirements.txt
├── run_experiment.py
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   └── evaluate.py
├── notebooks/
│   └── pubmedqa_evidence_classification.ipynb
├── results/
└── checkpoints/   # created at runtime
```

## Random seed

Default seed: **42** (`src/config.py`). Seeds are set for Python, NumPy, PyTorch, and Hugging Face Trainer.

## Limitations

- Abstracts are truncated to **512** subword tokens for transformers.
- No hyperparameter search (fixed epochs + early stopping on validation macro-F1).
- Inference latency reported for single-example batch size 1 on available hardware (CPU vs GPU will differ).

## Author

Prepared for **Anshika Reddy** — push this folder to GitHub when ready (no credentials are included in this bundle).

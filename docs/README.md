# CBCB Project Documentation

**Content-Based Captivation Behavior** — Predicting user behaviour on video streaming platforms using watch-time duration analysis.

## Document Structure (Diátaxis Framework)

| Quadrant | Directory | Purpose |
|---|---|---|
| **Tutorial** | `01-tutorials/` | Learning-oriented — walk through the pipeline step by step |
| **How-to** | `02-how-to/` | Task-oriented — solve specific problems |
| **Reference** | `03-reference/` | Information-oriented — module APIs, config, schemas |
| **Explanation** | `04-explanation/` | Understanding-oriented — concepts, rationale, design decisions |

## Quick Links

- [Getting Started Tutorial](01-tutorials/getting-started.md)
- [Running the Full Pipeline](02-how-to/run-pipeline.md)
- [Using the Streamlit App](02-how-to/use-streamlit-app.md)
- [Training Custom Models](02-how-to/train-custom-models.md)
- [API Reference](03-reference/api-reference.md)
- [Configuration Reference](03-reference/config-reference.md)
- [Data Schema Reference](03-reference/data-schema.md)
- [CBCB Concept](04-explanation/cbcb-concept.md)
- [Architecture Overview](04-explanation/architecture.md)
- [Label Design Rationale](04-explanation/label-design.md)
- [Model Selection Rationale](04-explanation/model-selection.md)

## Project at a Glance

```
CBCB_Project/
├── main.py                   # CLI orchestrator
├── src/                      # Core Python library
│   ├── config.py             # All constants, paths, hyperparameter grids
│   ├── utils.py              # Logging, seeding, model/metrics persistence
│   ├── dataset_generator.py  # Synthetic STC-like data generator
│   ├── preprocessing.py      # Clean → IQR → one-hot → min-max pipeline
│   ├── cbcb_s.py             # CBCB-S binary labels (Algorithm 2)
│   ├── cbcb_r.py             # CBCB-R ternary labels (Algorithm 3)
│   ├── feature_engineering.py# Feature matrix builder
│   ├── train.py              # DT/RF/GB + GridSearchCV + optional boosters
│   ├── evaluate.py           # Metrics, ROC/AUC, comparison tables
│   ├── visualization.py      # Matplotlib/Seaborn/Plotly charts
│   └── deep_learning.py      # LSTM/GRU/Transformer (optional, torch)
├── app/                      # Streamlit multi-page application
│   ├── Home.py               # Entry point (7 pages)
│   ├── _shared.py            # CSS styling, model loading, path setup
│   └── pages/                # 6 numbered pages
├── data/                     # Datasets (CSV)
├── models/                   # Trained models (.joblib) + metrics.json
├── assets/                   # Generated figures (PNG)
├── notebooks/                # Jupyter exploration notebook
└── reports/                  # IEEE report + presentation outline
```

## Two Core Behaviours

| Behaviour | Task Flag | Label Type | Classes | What It Predicts |
|---|---|---|---|---|
| **CBCB-S** (Sequential) | `cbcb_s` | Binary (0/1) | `No Repeat`, `Repeat` | Will the user watch the same genre next? |
| **CBCB-R** (Revert) | `cbcb_r` | Ternary (0/1/2) | `No Repeat`, `Immediate Repeat`, `Revert` | Will the user repeat, revert, or diverge? |

## How to Use This Documentation

1. **New to the project?** Start with the [Tutorial](01-tutorials/getting-started.md).
2. **Need to do something specific?** Check the [How-to Guides](02-how-to/).
3. **Looking for a function or config key?** Use the [Reference](03-reference/).
4. **Want to understand why?** Read the [Explanations](04-explanation/).

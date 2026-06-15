# Getting Started with CBCB

> **Audience:** New team members who need to set up the project, generate data, and run their first experiment.
>
> **Goal:** By the end of this tutorial you will have installed the project, generated a synthetic dataset, trained both CBCB-S and CBCB-R models, and explored results in the Streamlit app.

---

## Prerequisites

- Python 3.9+
- pip
- (Optional) Jupyter for the notebook

## Step 1: Set Up the Environment

Create a virtual environment and install dependencies.

```bash
# From the project root
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

The `requirements.txt` installs core dependencies:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
streamlit>=1.30
joblib>=1.3
openpyxl>=3.1
```

Optional boosters (uncomment to install):

```
xgboost>=1.7
lightgbm>=4.0
catboost>=1.2
torch>=2.0
```

## Step 2: Generate the Dataset

Run the synthetic data generator. This creates 50,000 viewing records across 2,000 users — realistic viewing histories with embedded repeat and revert patterns.

```bash
python main.py --generate --rows 50000 --users 2000
```

**What happens:**

1. Each user receives a power-law distributed share of interactions.
2. Genre transitions follow a per-user Markov chain with high self-loop probability (~0.45).
3. Watch durations are drawn from genre-specific log-normal distributions, modulated by engagement.
4. About 1% of rows are injected as outliers (duration 60,000–90,000 seconds) for robustness testing.
5. The CSV is saved to `data/cbcb_synthetic.csv`.

**Generated columns:**

| Column | Example | Description |
|---|---|---|
| `User_ID` | `U_42` | Anonymous user identifier |
| `Date` | `2025-01-15` | Interaction date |
| `Program_Name` | `The Action Chronicles` | Auto-generated title from genre |
| `Program_Genre` | `Action` | One of 16 genres |
| `Program_Class` | `Movie` | `Movie` or `Series` |
| `Watch_Duration` | 2745 | Seconds watched (log-normal) |
| `Season` | 1 | Season number (or 0 for movies) |
| `Episode` | 3 | Episode number (or 0 for movies) |

## Step 3: Run the Full Pipeline

Train CBCB-S and CBCB-R models, evaluate, and save results:

```bash
python main.py --all
```

This executes the entire pipeline:

```
Raw CSV → Preprocess → Label → Feature Engineering → Train → Evaluate → Save
```

**What you get:**

| Output | Location | Description |
|---|---|---|
| Trained models | `models/*.joblib` | 8 files (2 tasks × 4 model types including meta) |
| Metrics | `models/metrics.json` | Accuracy, Precision, Recall, F1, confusion matrices, ROC |
| Figures | `assets/*.png` | 16 charts (metrics bars, confusion heatmaps, outlier analysis) |

**Console output (abbreviated):**

```
[INFO] [main] Pipeline: BOTH tasks | tune=True | boost=False
[INFO] [dataset_generator] Generating 50000 rows, 2000 users...
[INFO] [cbcb_s] CBCB-S labels: 0=15523, 1=14477 → ~0.48 repeat rate
[INFO] [cbcb_r] CBCB-R labels: 0=10371, 1=9596, 2=9585 → well-balanced
[INFO] [train] CBCB-S Decision Tree  — Accuracy: 0.7834
[INFO] [train] CBCB-S Random Forest  — Accuracy: 0.8102
[INFO] [train] CBCB-S Gradient Boosting — Accuracy: 0.7941
[INFO] [train] CBCB-R Decision Tree  — Accuracy: 0.8215
[INFO] [train] CBCB-R Random Forest  — Accuracy: 0.8473
[INFO] [train] CBCB-R Gradient Boosting — Accuracy: 0.8329
```

### What Each Step Does

| Pipeline Step | Module | What Happens |
|---|---|---|
| **Preprocess** | `src/preprocessing.py` | `clean()` → drop NAs/duplicates → `remove_outliers_iqr()` → `one_hot_encode()` → `min_max_scale()` |
| **Label (CBCB-S)** | `src/cbcb_s.py` | Per user, ordered by date: `label = 1` if `genre[i] == genre[i+1]` |
| **Label (CBCB-R)** | `src/cbcb_r.py` | Per user: `label = 2` if `genre[i] == genre[i+2]` (revert), `1` if repeat, `0` otherwise |
| **Feature Engineering** | `src/feature_engineering.py` | 6 sequential features + one-hot genre/class columns |
| **Training** | `src/train.py` | 70/30 stratified split → GridSearchCV with 5-fold CV → fit |
| **Evaluation** | `src/evaluate.py` | Accuracy, Precision, Recall, F1, ROC/AUC, confusion matrix |
| **Visualization** | `src/visualization.py` | Metric bar charts, confusion heatmaps, feature importance |

## Step 4: Explore in the Streamlit App

Launch the interactive dashboard:

```bash
streamlit run app/Home.py
```

The app opens in your browser with 7 pages:

| Page | What You Can Do |
|---|---|
| **Home** | Project overview, workflow diagram |
| **Dataset Manager** | Generate data, preview stats, preprocess, train models |
| **CBCB Architecture** | Visual pipeline walkthrough, labelling rules with examples |
| **Prediction** | Live inference — input user context, get predictions |
| **Experimental Results** | Performance tables, bar charts, confusion matrices, feature importance |
| **Model Comparison** | Radar chart, grouped bars, leaderboard across tasks |
| **About** | Paper summary, methodology, references |

## Step 5: Try the Jupyter Notebook

For an interactive walkthrough of every step with inline visualisations:

```bash
jupyter notebook notebooks/CBCB_Exploration.ipynb
```

The notebook covers the same pipeline in 7 sections:

1. **Generate dataset** — with distribution checks
2. **EDA** — watch-time distribution, genre frequency, user activity
3. **Preprocess** — IQR outlier removal visualised
4. **Generate Labels** — CBCB-S + CBCB-R with validation
5. **Build Features & Train CBCB-S** — full training with GridSearch
6. **CBCB-R Training** — with confusion matrices
7. **Takeaways** — summary of results

## Expected Results

With the default synthetic dataset, you should expect:

| Task | Metric | Decision Tree | Random Forest | Gradient Boosting |
|---|---|---|---|---|
| **CBCB-S** | Accuracy | ~0.78 | ~0.81 | ~0.79 |
| **CBCB-S** | F1 Score | ~0.78 | ~0.81 | ~0.79 |
| **CBCB-R** | Accuracy | ~0.82 | ~0.85 | ~0.83 |
| **CBCB-R** | F1 Score | ~0.82 | ~0.85 | ~0.83 |

> Random Forest consistently performs best. GridSearch often finds `n_estimators=200`, `max_depth=20` for CBCB-R.

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `No module named 'src'` | Running from wrong directory | Run from project root: `CBCB_Project/` |
| Streamlit shows no models | Models not trained yet | Run `python main.py --all` first |
| `torch` import error | PyTorch not installed | Deep learning module is optional; ignore or `pip install torch` |
| GridSearch is slow | Default 5-fold CV on 35k rows | Use `--no-tune` flag to skip GridSearch |
| Booster import errors | xgboost/lightgbm/catboost not installed | Use `--no-boosting` or install the package |

## Next Steps

- [Run the pipeline with custom options](02-how-to/run-pipeline.md)
- [Train models from the Streamlit app](02-how-to/use-streamlit-app.md)
- [Add deep learning models](02-how-to/train-custom-models.md)
- [Understand the CBCB concept](04-explanation/cbcb-concept.md)

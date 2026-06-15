# Architecture Overview

> **Audience:** Developers onboarding to the codebase or planning extensions.
>
> **Goal:** Understand the project's structure, data flow, module dependencies, and design decisions.

---

## High-Level Architecture

The project follows a **modular pipeline architecture** with two user-facing interfaces (CLI and Streamlit) sharing a common core library (`src/`).

```
┌──────────────────────────────────────────────────────────┐
│                     User Interfaces                       │
│                                                          │
│  main.py (CLI)           app/ (Streamlit GUI)            │
│  ─────────────           ───────────────────             │
│  argparse-based          multi-page with sidebar          │
│  headless execution      interactive visualisations       │
└──────────────────────┬───────────────────────────────────┘
                       │ both import
                       ▼
┌──────────────────────────────────────────────────────────┐
│                    src/ (Core Library)                     │
│                                                          │
│  config.py ←──── all modules read constants here          │
│  utils.py  ←──── all modules use persistence helpers      │
│                                                          │
│  dataset_generator.py → preprocessing.py                  │
│                             ↓                             │
│                        cbcb_s.py / cbcb_r.py              │
│                             ↓                             │
│                     feature_engineering.py                │
│                             ↓                             │
│                        train.py                           │
│                        ├── evaluate.py                    │
│                        └── visualization.py               │
│                                                          │
│  deep_learning.py (optional, torch)                       │
└──────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
main.py
  ├── src.config
  ├── src.utils
  ├── src.dataset_generator
  ├── src.preprocessing ──→ src.config, src.utils
  ├── src.cbcb_s ──────────→ src.config, src.utils
  ├── src.cbcb_r ──────────→ src.config, src.utils
  ├── src.feature_engineering ──→ src.config, src.utils
  ├── src.train ───────────→ src.config, src.utils, src.evaluate
  ├── src.evaluate ────────→ src.config, src.utils
  └── src.visualization ───→ src.config

app/
  ├── Home.py ─────────────→ app._shared
  ├── _shared.py ──────────→ src.config, src.utils
  └── pages/
      ├── 1_Dataset_Manager ──→ src.* (all)
      ├── 2_CBCB_Architecture (standalone, markdown/text)
      ├── 3_Prediction ───────→ src.preprocessing, src.feature_engineering
      ├── 4_Experimental_Results ──→ src.evaluate, src.visualization
      ├── 5_Model_Comparison ──────→ src.evaluate, src.visualization
      └── 6_About ────────── (standalone, markdown)
```

## Data Flow (End-to-End)

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Raw CSV │───→│Preprocess│───→│  Label   │───→│Features  │───→│  Train   │───→│Evaluate  │
│          │    │ X'=fs(fe │    │ CBCB-S/R │    │ 6 seq +  │    │ DT/RF/GB │    │ Metrics  │
│ 50k rows │    │  (fc(X)))│    │ per user │    │ one-hots │    │ +GridSearch│  │ +Figures │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │   Persist    │
                                                               │ *.joblib     │
                                                               │ metrics.json │
                                                               │ *.png        │
                                                               └──────────────┘
```

## Preprocessing Pipeline Design

The preprocessing pipeline is expressed as function composition:

```python
X' = fs(fe(fc(X)))
```

Where:

| Function | Name | Action |
|---|---|---|
| `fc` | **Clean** | Drop nulls, duplicates, invalid durations |
| `fe` | **Encode** | One-hot encode `Program_Genre` and `Program_Class` |
| `fs` | **Scale** | Min-max scale `Watch_Duration` to [0, 1] |
| `X'` | **Output** | Preprocessed DataFrame ready for labelling |

**Why IQR for outliers?** Tukey fences (Q1 − 1.5×IQR, Q3 + 1.5×IQR) are non-parametric and robust to skewed distributions — perfect for watch-duration data which is log-normal.

**Why min-max scaling?** Tree-based models are scale-invariant, but it makes the `watch_duration_scaled` feature interpretable as normalised engagement (0 = least, 1 = most).

## Labelling Design

Labels are generated per-user, ordered by date. This ensures temporal causality — a label at position `i` only uses information from positions `i` and ahead (never past).

### CBCB-S: Sliding Window of 2

```python
for user in users:
    for i in range(len(user_history) - 1):
        label[i] = 1 if genre[i] == genre[i+1] else 0
```

The last row of each user is dropped (no successor to compare against).

### CBCB-R: Sliding Window of 3

```python
for user in users:
    for i in range(len(user_history) - 2):
        if genre[i] == genre[i+1]:
            label = 1  # Immediate Repeat
        elif genre[i] == genre[i+2]:
            label = 2  # Revert
        else:
            label = 0  # No Repeat
```

The last two rows per user are dropped.

## Feature Engineering Rationale

The 6 sequential features capture distinct behavioural signals:

| Feature | Signal Captured | Why It Matters |
|---|---|---|
| `watch_duration_scaled` | Current engagement intensity | Core hypothesis — longer watch = more captivated |
| `prev_genre_code` | Genre of previous interaction | Baseline for genre transition patterns |
| `genre_repeat_run` | Current binge length | Binge state signals high captivation |
| `user_avg_duration` | User baseline engagement | Normalises against heavy/light users |
| `user_session_index` | Position in user history | New vs established users behave differently |
| `time_since_last_day` | Recency of last interaction | Daily vs weekly users have different patterns |

## Model Training Pipeline

```
X (features), y (labels)
    │
    ▼
Train/Test Split (70/30, stratified)
    │
    ├──► Decision Tree ──→ GridSearchCV ──→ fit ──→ evaluate
    ├──► Random Forest ──→ GridSearchCV ──→ fit ──→ evaluate
    ├──► Gradient Boosting ──→ GridSearchCV ──→ fit ──→ evaluate
    ├──► [XGBoost] ────→ fit (default params) ──→ evaluate
    ├──► [LightGBM] ───→ fit (default params) ──→ evaluate
    └──► [CatBoost] ───→ fit (default params) ──→ evaluate
```

GridSearchCV uses 5-fold cross-validation on the training set only (test set is held out).

## Streamlit App Architecture

```
app/Home.py (entry point)
    │
    ├── st.set_page_config(layout="wide")
    ├── st.sidebar (auto-populated from pages/)
    │
    ├── app/_shared.py
    │   ├── inject_css()    → custom CSS (hero, cards, pills)
    │   ├── hero()          → gradient banner
    │   ├── load_metrics()   → read models/metrics.json
    │   └── load_trained_model() → load joblib file
    │
    ├── pages/1_Dataset_Manager  ← full src pipeline with progress bars
    ├── pages/2_CBCB_Architecture ← Graphviz + markdown (read-only)
    ├── pages/3_Prediction        ← live inference form
    ├── pages/4_Experimental_Results ← metrics + charts
    ├── pages/5_Model_Comparison     ← radar + leaderboard
    └── pages/6_About               ← project info + references
```

State is managed through three mechanisms:

1. **File system** — `models/metrics.json`, `*.joblib` files (persisted across sessions)
2. **`st.session_state`** — `raw_df`, `processed_df`, `outlier_stats` (within a session)
3. **Files on disk** — uploaded CSVs are read into DataFrames (not persisted unless explicitly saved)

## Why No Database or API?

The project is designed as a **research/demo** tool, not a production service:

- No REST API — interaction is through CLI or local GUI
- No database — everything is file-based
- No user authentication — single-user local app
- This keeps the barrier to entry low and the codebase focused on the ML pipeline.

If deployed, a production version would add:

- A web API (FastAPI/Flask) wrapping `src/` for inference
- A database (PostgreSQL) for user and model data
- Authentication and multi-tenancy
- Model versioning and A/B testing

## File Layout Principles

- **`src/`** — Pure functions (no side effects except `save_model`/`save_json`). Every function takes data in, returns data out.
- **`app/`** — Streamlit-specific code. `_shared.py` for cross-cutting concerns, pages are self-contained.
- **`data/`**, **`models/`**, **`assets/`** — Generated artifacts. `.gitkeep` files ensure directories are tracked.
- **`reports/`** — Human-readable documents (IEEE report, presentation outline).
- **`notebooks/`** — Jupyter notebook for exploration (complements, doesn't duplicate, the formal pipeline).

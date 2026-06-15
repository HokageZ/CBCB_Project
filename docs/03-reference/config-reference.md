# Configuration Reference

> **Audience:** Developers needing to modify project behaviour via `src/config.py`.
>
> **Scope:** Every constant, path, and parameter grid in `src/config.py`.

---

## File Location

`src/config.py` — imported by every module in the project.

## All Constants

### Paths

| Constant | Value | Used By |
|---|---|---|
| `PROJECT_ROOT` | Parent of `src/` | All modules |
| `DATA_DIR` | `PROJECT_ROOT / "data"` | `dataset_generator`, `preprocessing` |
| `MODELS_DIR` | `PROJECT_ROOT / "models"` | `train`, `utils` |
| `ASSETS_DIR` | `PROJECT_ROOT / "assets"` | `visualization` |
| `REPORTS_DIR` | `PROJECT_ROOT / "reports"` | (reserved) |
| `RAW_DATA_PATH` | `DATA_DIR / "cbcb_synthetic.csv"` | Default dataset location |
| `PROCESSED_DATA_PATH` | `DATA_DIR / "processed.csv"` | Preprocessed output |
| `METRICS_PATH` | `MODELS_DIR / "metrics.json"` | Evaluation results |

### Dataset Schema

```python
COLUMNS = [
    "User_ID",           # str  — anonymous user identifier (U_XXXX)
    "Date",              # str  — ISO date (YYYY-MM-DD)
    "Program_Name",      # str  — auto-generated title
    "Program_Genre",     # str  — one of 16 GENRES
    "Program_Class",     # str  — "Movie" or "Series"
    "Watch_Duration",    # int  — seconds watched
    "Season",            # int  — season number (0 for movies)
    "Episode",           # int  — episode number (0 for movies)
]
```

### Genre System

16 genres with per-genre watch-duration profiles:

| Genre | Median (s) | Sigma | Typical Duration |
|---|---|---|---|
| Action | 7200 | 0.6 | ~2h movie |
| Documentary | 5400 | 0.5 | ~1.5h |
| Comedy | 3600 | 0.7 | ~1h (series episode) |
| Drama | 5400 | 0.6 | ~1.5h |
| Horror | 4200 | 0.7 | ~70min |
| Romance | 4500 | 0.6 | ~75min |
| Science_Fiction | 6000 | 0.6 | ~100min |
| Thriller | 5400 | 0.6 | ~90min |
| Animation | 3600 | 0.7 | ~1h (series episode) |
| Adventure | 6000 | 0.6 | ~100min |
| Crime | 5400 | 0.6 | ~90min |
| Fantasy | 6000 | 0.6 | ~100min |
| Mystery | 4800 | 0.6 | ~80min |
| War | 6000 | 0.6 | ~100min |
| Western | 5400 | 0.6 | ~90min |
| Musical | 7200 | 0.6 | ~2h |

### Data Generation

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_N_ROWS` | 50000 | Target row count for synthetic data |
| `DEFAULT_N_USERS` | 2000 | Target user count |
| `GENRE_REPEAT_PROB` | 0.45 | Probability of same-genre transition |
| `GENRE_REVERT_PROB` | 0.25 | Probability of revert pattern |
| `OUTLIER_FRACTION` | 0.01 | Fraction of rows injected as outliers |
| `OUTLIER_DURATION_RANGE` | (60000, 90000) | Outlier duration bounds (seconds) |

### Categorical & Scaling Features

```python
CATEGORICAL_FEATURES = ["Program_Genre", "Program_Class"]
SCALE_FEATURE = "Watch_Duration"
```

### Label Columns

| Constant | Value | Task |
|---|---|---|
| `LABEL_COL_S` | `"cbcb_s_label"` | CBCB-S (binary: 0/1) |
| `LABEL_COL_R` | `"cbcb_r_label"` | CBCB-R (ternary: 0/1/2) |

```python
CBCB_S_CLASSES = ["No Repeat", "Repeat"]
CBCB_R_CLASSES = ["No Repeat", "Immediate Repeat", "Revert"]
```

### Training

| Constant | Default | Description |
|---|---|---|
| `RANDOM_SEED` | 42 | Global seed for reproducibility |
| `TEST_SIZE` | 0.30 | Fraction of data held out for testing |
| `CV_FOLDS` | 5 | Cross-validation folds for GridSearchCV |

### Model Names

```python
MODEL_NAMES = [
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
]

METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]
```

### Hyperparameter Grids

#### Decision Tree (`DT_PARAM_GRID`)

```python
{
    'max_depth':        [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 5, 10],
    'criterion':        ['gini', 'entropy'],
}
```

Best found (synthetic data): `max_depth=15, min_samples_split=2, min_samples_leaf=1, criterion='gini'`

#### Random Forest (`RF_PARAM_GRID`)

```python
{
    'n_estimators':      [100, 200, 300],
    'max_depth':         [10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 5],
}
```

Best found (synthetic data): `n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1`

#### Gradient Boosting (`GB_PARAM_GRID`)

```python
{
    'n_estimators':     [100, 200, 300],
    'max_depth':        [3, 5, 7, 10],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'subsample':        [0.8, 1.0],
}
```

Best found (synthetic data): `n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8`

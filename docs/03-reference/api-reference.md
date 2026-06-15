# API Reference

> **Audience:** Developers working directly with `src/` modules and `app/` components.
>
> **Scope:** Every public function, class, and constant exported by each module, with code snippets.

---

## `src/config.py` — Central Configuration

All project constants, paths, and hyperparameter grids. Every other module imports from here.

```python
from src.config import *
```

### Paths

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # → CBCB_Project/
DATA_DIR    = PROJECT_ROOT / "data"                    # → CBCB_Project/data/
MODELS_DIR  = PROJECT_ROOT / "models"                  # → CBCB_Project/models/
ASSETS_DIR  = PROJECT_ROOT / "assets"                  # → CBCB_Project/assets/
REPORTS_DIR = PROJECT_ROOT / "reports"                 # → CBCB_Project/reports/

RAW_DATA_PATH      = DATA_DIR / "cbcb_synthetic.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed.csv"
METRICS_PATH       = MODELS_DIR / "metrics.json"
```

### Dataset Schema

```python
COLUMNS = [
    "User_ID", "Date", "Program_Name", "Program_Genre",
    "Program_Class", "Watch_Duration", "Season", "Episode"
]
```

### Labels

```python
LABEL_COL_S = "cbcb_s_label"     # Binary: 0 = No Repeat, 1 = Repeat
LABEL_COL_R = "cbcb_r_label"     # Ternary: 0 = No Repeat, 1 = Immediate Repeat, 2 = Revert

CBCB_S_CLASSES = ["No Repeat", "Repeat"]
CBCB_R_CLASSES = ["No Repeat", "Immediate Repeat", "Revert"]
```

### Genre System (16 genres)

```python
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
    "Romance", "Science_Fiction", "Thriller", "War", "Western",
    "Musical"
]
```

Each genre has a watch-duration profile:

```python
GENRE_DURATION_PROFILE = {
    "Action":           {"median_seconds": 7200, "sigma": 0.6},
    "Documentary":      {"median_seconds": 5400, "sigma": 0.5},
    "Comedy":           {"median_seconds": 3600, "sigma": 0.7},
    # ... 13 more genres
}
```

### Data Generation Parameters

```python
DEFAULT_N_ROWS     = 50000
DEFAULT_N_USERS    = 2000
GENRE_REPEAT_PROB  = 0.45   # Probability of same-genre transition
GENRE_REVERT_PROB  = 0.25   # Probability of revert pattern
OUTLIER_FRACTION   = 0.01
OUTLIER_DURATION_RANGE = (60000, 90000)
```

### Training Parameters

```python
TEST_SIZE = 0.30
CV_FOLDS  = 5
RANDOM_SEED = 42
```

### Hyperparameter Grids

```python
DT_PARAM_GRID = {
    'max_depth':        [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 5, 10],
    'criterion':        ['gini', 'entropy'],
}

RF_PARAM_GRID = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 5],
}

GB_PARAM_GRID = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [3, 5, 7, 10],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'subsample':        [0.8, 1.0],
}
```

### Model Names

```python
MODEL_NAMES = [
    "decision_tree", "random_forest", "gradient_boosting",
    "xgboost", "lightgbm", "catboost"
]

METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]
```

---

## `src/utils.py` — Utility Helpers

Logging, reproducibility, and model/metrics persistence.

```python
from src.utils import get_logger, set_seed, save_model, load_model
from src.utils import save_json, load_json, model_path
```

### Logging

```python
def get_logger(name: str) -> logging.Logger:
    """Creates a module-level logger with a stream handler.
    Format: [LEVEL] [name] message
    """
    # Example usage
    logger = get_logger("train")
    logger.info("Training started...")
    # Output: [INFO] [train] Training started...
```

### Reproducibility

```python
def set_seed(seed: int) -> None:
    """Seeds both `random` and `numpy.random`.
    Call this before any data generation or model training.
    """
    # Used at the top of main.py and each training function
    set_seed(RANDOM_SEED)
```

### Persistence

```python
def save_model(obj: Any, path: Path) -> None:
    """Serialise object to path using joblib.dump()."""

def load_model(path: Path) -> Any:
    """Load a joblib-serialised object.
    Raises FileNotFoundError if path doesn't exist.
    """

def save_json(data: dict, path: Path) -> None:
    """Serialise to JSON with numpy type coercion.
    Converts np.integer → int, np.floating → float, np.ndarray → list.
    """

def load_json(path: Path) -> dict | None:
    """Load JSON, returns None if file missing."""

def model_path(task: str, model_key: str) -> Path:
    """Returns standardised path for model joblib files.
    Example: model_path('cbcb_s', 'random_forest')
    # → models/cbcb_s__random_forest.joblib
    """
```

Json helper:

```python
# Internal — handles numpy type coercion
def _json_default(value):
    if isinstance(value, (np.integer,)):       return int(value)
    if isinstance(value, (np.floating,)):      return float(value)
    if isinstance(value, (np.ndarray,)):       return value.tolist()
    raise TypeError(f"Type {type(value)} not JSON-serialisable")
```

---

## `src/dataset_generator.py` — Synthetic Data Generator

Generates a realistic STC-like dataset with embedded sequential and revert patterns.

```python
from src.dataset_generator import generate_dataset, generate_and_save
```

### `generate_dataset(n_rows, n_users, seed)`

```python
def generate_dataset(
    n_rows: int = DEFAULT_N_ROWS,
    n_users: int = DEFAULT_N_USERS,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate synthetic viewing records.

    Algorithm:
    1. Allocate interactions per user via power-law distribution.
    2. For each user, sample a favourite genre and engagement threshold θ.
    3. Build a per-user Markov transition matrix with high self-loop.
    4. For each interaction:
       a. Choose genre via Markov chain (modulated by engagement).
       b. Generate watch duration from genre-specific log-normal.
       c. Modulate duration by user engagement (Beta distribution).
    5. Inject ~1% outliers.

    Returns: DataFrame with COLUMNS schema, ~50k rows by default.
    """
```

### `generate_and_save(path, n_rows, n_users, seed)`

```python
def generate_and_save(
    path: Path = RAW_DATA_PATH,
    n_rows: int = DEFAULT_N_ROWS,
    n_users: int = DEFAULT_N_USERS,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate and write to CSV. Also returns the DataFrame."""
```

### Markov Transition Matrix

```python
def _build_transition_matrix(rng: np.random.Generator) -> np.ndarray:
    """Per-user genre transition matrix.
    - Zero diagonal (no self-transition in matrix — handled separately)
    - Off-diagonal rows sum to 1
    - Shape: (len(GENRES), len(GENRES))
    """
```

### Title Generation

```python
def _make_title(rng: np.random.Generator, genre: str) -> str:
    """Generates a plausible program title from genre-specific templates.
    Examples: 'The Action Chronicles', 'Mystery of the Lost City'
    """
```

---

## `src/preprocessing.py` — Preprocessing Pipeline

Implements the full `X' = fs(fe(fc(X)))` pipeline.

```python
from src.preprocessing import clean, remove_outliers_iqr, Transformers
from src.preprocessing import one_hot_encode, min_max_scale, preprocess_pipeline
```

### `Transformers` Dataclass

```python
@dataclass
class Transformers:
    encoder: OneHotEncoder            # Fitted one-hot encoder
    scaler: MinMaxScaler              # Fitted min-max scaler
    encoded_columns: list[str]        # Column names after encoding
```

### Preprocessing Functions

```python
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """(fc) Drops missing values, duplicates, and rows with invalid
    Watch_Duration (≤ 0 or missing). Returns cleaned DataFrame."""

def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Tukey fences: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]"""

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
    """Removes rows where column value is outside Tukey fences.
    Returns (cleaned_df, stats_dict) where stats includes:
    - n_removed: count of outliers removed
    - fraction: proportion removed
    - bounds: (lower, upper) fences
    """

def _make_encoder() -> OneHotEncoder:
    """OneHotEncoder with handle_unknown='ignore', sparse_output=False."""

def one_hot_encode(df: pd.DataFrame, columns: list[str],
                   encoder: OneHotEncoder | None = None) -> tuple[pd.DataFrame, OneHotEncoder, list[str]]:
    """(fe) One-hot encode specified columns.
    If encoder is None, a new one is fitted.
    Returns (df with original columns dropped + dummies added, encoder, encoded_columns).
    """

def min_max_scale(df: pd.DataFrame, column: str,
                  scaler: MinMaxScaler | None = None) -> tuple[pd.DataFrame, MinMaxScaler]:
    """(fs) Min-max scale a numeric column to [0, 1].
    If scaler is None, a new one is fitted.
    """
```

### Pipeline Composition

```python
def preprocess_pipeline(df: pd.DataFrame,
                       transformers: Transformers | None = None,
                       drop_outliers: bool = True) -> tuple[pd.DataFrame, Transformers]:
    """Full preprocessing pipeline: clean → IQR → one-hot → scale.

    Args:
        df: Raw input data.
        transformers: Optional pre-fitted transformers (for inference).
        drop_outliers: Whether to remove IQR outliers.

    Returns:
        (processed_df, transformers) where transformers contains
        the fitted encoder, scaler, and encoded column names.
    """
```

### Correlation Analysis

```python
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute numeric-only correlation matrix for EDA."""
```

---

## `src/cbcb_s.py` — CBCB-S Sequential Labels

Implements Algorithm 2 from the paper.

```python
from src.cbcb_s import generate_labels
```

### `generate_labels(df)`

```python
def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Appends binary CBCB-S label column.

    For each user sorted by date:
      y_i = 1 if genre[i] == genre[i+1] (Repeat)
      y_i = 0 otherwise (No Repeat)

    Drops the last row of each user (no successor).

    Label column name: LABEL_COL_S = "cbcb_s_label"
    """
```

**Example:**

```python
import pandas as pd
from src.cbcb_s import generate_labels

df = pd.DataFrame({
    'User_ID': ['U1', 'U1', 'U1', 'U1'],
    'Date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']),
    'Program_Genre': ['Action', 'Action', 'Comedy', 'Action'],
})
result = generate_labels(df)
# result['cbcb_s_label'] = [1, 0, 0]  → last row dropped
```

**Self-test runs on direct execution:**

```bash
python src/cbcb_s.py
# Validates on sequence: [G1, G1, G2, G1] → labels [1, 0, 0]
```

---

## `src/cbcb_r.py` — CBCB-R Revert Labels

Implements Algorithm 3 from the paper.

```python
from src.cbcb_r import generate_labels
```

### `generate_labels(df)`

```python
def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Appends ternary CBCB-R label column.

    For each user sorted by date:
      y_i = 1 if genre[i] == genre[i+1]        (Immediate Repeat)
      y_i = 2 if genre[i] != genre[i+1] AND genre[i] == genre[i+2]  (Revert)
      y_i = 0 otherwise                         (No Repeat)

    Drops the last two rows of each user (insufficient lookahead).

    Label column name: LABEL_COL_R = "cbcb_r_label"
    """
```

**Example:**

```python
import pandas as pd
from src.cbcb_r import generate_labels

df = pd.DataFrame({
    'User_ID': ['U1', 'U1', 'U1', 'U1', 'U1'],
    'Date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05']),
    'Program_Genre': ['Action', 'Comedy', 'Action', 'Action', 'Drama'],
})
result = generate_labels(df)
# result['cbcb_r_label'] = [2, 0, 1]  → last 2 rows dropped
```

**Interpretation:**

| Label | Meaning | Condition |
|---|---|---|
| 0 | No Repeat / Diverge | G1 → G2 (and no revert) |
| 1 | Immediate Repeat | G1 → G1 (same genre consecutively) |
| 2 | Revert (Return) | G1 → G2 → G1 (diverged then returned) |

---

## `src/feature_engineering.py` — Feature Matrix Builder

```python
from src.feature_engineering import build_feature_matrix
```

### `build_feature_matrix(df, label_col)`

```python
def build_feature_matrix(df: pd.DataFrame, label_col: str
                         ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build feature matrix X and target vector y.

    1. Sort by User_ID, Date.
    2. Add 6 sequential features via _add_sequence_features():
       - watch_duration_scaled: min-max scaled duration [0, 1]
       - prev_genre_code: label-encoded previous genre
       - genre_repeat_run: current consecutive same-genre streak
       - user_avg_duration: user's mean scaled watch time
       - user_session_index: normalised position [0, 1] in user's history
       - time_since_last_day: days since previous interaction
    3. Select numeric features only (drops categorical/text/date).
    4. One-hot columns (Program_Genre_*, Program_Class_*) are included.

    Returns: (X, y, feature_names)
    """
```

**Feature vector (~24 features):**

```
watch_duration_scaled                   float   [0, 1]
prev_genre_code                         int     [0..15]
genre_repeat_run                        int     [1..n]
user_avg_duration                       float   [0, 1]
user_session_index                      float   [0, 1]
time_since_last_day                     float   days
Program_Genre_Action                    bool    0/1
Program_Genre_Adventure                 bool    0/1
... (16 genre one-hots)
Program_Class_Movie                     bool    0/1
Program_Class_Series                    bool    0/1
```

---

## `src/train.py` — Model Training

```python
from src.train import train_all
```

### `train_all(X, y, task, feature_names, tune, include_boosting, persist)`

```python
def train_all(X: pd.DataFrame, y: pd.Series,
              task: str, feature_names: list[str],
              tune: bool = True,
              include_boosting: bool = False,
              persist: bool = True
              ) -> dict[str, Any]:
    """Train all models for a given task.

    1. Stratified 70/30 train-test split.
    2. Train each model (DT, RF, GB, optional boosters).
    3. Evaluate each on test set.
    4. If persist=True, save model and metrics to disk.

    Returns dict with:
    - models: {model_key: fitted_model_object}
    - metrics: {model_key: {accuracy, precision, recall, f1, ...}}
    - best: {metric: model_key} — best model per metric
    - meta: {task, feature_names, classes, test_size}
    """
```

### Individual Fit Functions

```python
def _fit_decision_tree(X_train, y_train, tune: bool = True):
    """DecisionTreeClassifier with optional GridSearchCV."""

def _fit_random_forest(X_train, y_train, tune: bool = True):
    """RandomForestClassifier with optional GridSearchCV."""

def _fit_gradient_boosting(X_train, y_train, tune: bool = True):
    """HistGradientBoostingClassifier with optional GridSearchCV."""

def _optional_boosters() -> dict[str, Callable]:
    """Returns factory dict for installed boosters:
    {'xgboost': XGBClassifier, 'lightgbm': LGBMClassifier, 'catboost': CatBoostClassifier}
    Empty dict if none installed.
    """
```

---

## `src/evaluate.py` — Model Evaluation

```python
from src.evaluate import evaluate_model, comparison_table, best_models, recommend_model
```

### `evaluate_model(model, X_test, y_test, name)`

```python
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   name: str) -> dict[str, Any]:
    """Compute all evaluation metrics for a fitted model.

    Returns dict:
    - accuracy: float
    - precision: float (macro)
    - recall: float (macro)
    - f1: float (macro)
    - per_class_precision: list[float]
    - per_class_recall: list[float]
    - confusion_matrix: list[list[int]]
    - roc_auc: float or None (None for sklearn < 1.6 multi-class)
    - roc_curves: list[{"fpr": list, "tpr": list, "class": str}]
    - feature_importance: dict{feature: importance} or None
    """
```

### `comparison_table(results)`

```python
def comparison_table(results: dict) -> pd.DataFrame:
    """Tidy DataFrame of headline metrics per model.
    Columns: Model, Accuracy, Precision, Recall, F1
    """
```

### `best_models(results)`

```python
def best_models(results: dict) -> dict[str, str]:
    """Returns {metric: model_key} of top scorer per metric."""
```

### `recommend_model(results)`

```python
def recommend_model(results: dict) -> str:
    """Model with highest mean of 4 headline metrics.
    Uses: mean(accuracy + precision + recall + f1) / 4
    """
```

---

## `src/visualization.py` — Chart Builders

```python
from src.visualization import (
    watch_time_distribution, outliers_before_after, genre_distribution,
    correlation_heatmap, user_activity, metric_bar, confusion_heatmap,
    feature_importance_bar, radar_chart, interactive_metric_bar, gauge, save_fig
)
```

### EDA Charts

```python
def watch_time_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram with KDE overlay of Watch_Duration."""

def outliers_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame,
                         column: str) -> plt.Figure:
    """Side-by-side box plots before/after outlier removal."""

def genre_distribution(df: pd.DataFrame) -> plt.Figure:
    """Count plot of Program_Genre with viridis palette."""

def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Seaborn heatmap of numeric feature correlations."""

def user_activity(df: pd.DataFrame) -> plt.Figure:
    """Histogram of interactions per user (log-log scale)."""
```

### Results Charts

```python
def metric_bar(metrics: dict, task: str, metric_key: str) -> plt.Figure:
    """Bar chart of one metric (accuracy, precision, recall, f1)
    across all models. Saves to assets/{task}_{metric_key}.png."""

def confusion_heatmap(cm: np.ndarray, classes: list[str],
                      title: str) -> plt.Figure:
    """Normalised confusion matrix heatmap with annotations."""

def feature_importance_bar(importances: dict, title: str,
                          top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
```

### Interactive Charts (Plotly)

```python
def radar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Plotly radar chart, one trace per model, 4 axes
    (Accuracy, Precision, Recall, F1)."""

def interactive_metric_bar(metrics_df: pd.DataFrame,
                          title: str) -> go.Figure:
    """Plotly grouped bar chart with hover labels."""

def gauge(value: float, title: str) -> go.Figure:
    """Plotly gauge chart (speedometer style) for single values."""
```

### File Saving

```python
def save_fig(fig: plt.Figure, path: Path) -> None:
    """Save matplotlib figure to PNG. Calls fig.savefig() with
    bbox_inches='tight', dpi=150."""
```

---

## `src/deep_learning.py` — Deep Learning (Optional)

PyTorch-based sequence models. All imports guarded by `TORCH_AVAILABLE`.

```python
from src.deep_learning import build_sequences, train_sequence_model
```

### `build_sequences(df, seq_len)`

```python
def build_sequences(df: pd.DataFrame, seq_len: int = 5
                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convert DataFrame into sliding-window genre sequences per user.
    Returns (X_seq, y_seq, vocab) where vocab maps genre → int.
    """
```

### Architectures

```python
class LSTMRecommender(nn.Module):
    """Embedding → LSTM → Dropout → Linear"""

class GRURecommender(nn.Module):
    """Embedding → GRU → Dropout → Linear"""

class TransformerRecommender(nn.Module):
    """Embedding + PositionalEncoding → TransformerEncoder → Linear"""
```

### `train_sequence_model(df, arch, seq_len, epochs, batch_size, lr)`

```python
def train_sequence_model(df: pd.DataFrame,
                         arch: str = 'lstm',
                         seq_len: int = 5,
                         epochs: int = 20,
                         batch_size: int = 64,
                         lr: float = 0.001
                         ) -> nn.Module | None:
    """Train and evaluate a sequence model.
    Returns None if torch is not available.
    """
```

---

## `app/_shared.py` — Streamlit Shared Utilities

```python
from app._shared import inject_css, hero, load_metrics, available_tasks, load_trained_model
```

### `inject_css()`

```python
def inject_css() -> None:
    """Injects custom CSS into Streamlit via st.markdown.
    Styles: gradient hero banner, metric cards, pills, spacing.
    """
```

### `hero(title, subtitle)`

```python
def hero(title: str, subtitle: str = "") -> None:
    """Renders a gradient hero banner at the top of each page.
    Uses st.markdown with inline CSS.
    """
```

### Model Loading

```python
def load_metrics() -> dict | None:
    """Loads models/metrics.json. Returns None if missing."""

def available_tasks(metrics: dict) -> list[str]:
    """Returns list of trained task keys, e.g. ['cbcb_s', 'cbcb_r']."""

def load_trained_model(task: str, model_key: str):
    """Load a trained model from models/{task}__{model_key}.joblib."""
```

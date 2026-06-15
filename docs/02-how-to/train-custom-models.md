# How to Train Custom Models

> **Audience:** Team members extending the project with new models or datasets.
>
> **Goal:** Add a new classifier, use a real-world dataset, or enable deep learning.

---

## Adding a New Classifier

To add a new sklearn-compatible model (e.g., SVM, KNN, MLP):

### 1. Add the fit function in `src/train.py`

```python
from sklearn.svm import SVC

def _fit_svm(X_train, y_train, tune):
    logger = get_logger("train.svm")
    logger.info("Training SVM...")
    model = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model
```

### 2. Register it in `train_all()`

```python
def train_all(X, y, task, feature_names, tune, include_boosting, persist=True):
    models = {
        MODEL_NAMES[0]: _fit_decision_tree(X_train, y_train, tune),
        MODEL_NAMES[1]: _fit_random_forest(X_train, y_train, tune),
        MODEL_NAMES[2]: _fit_gradient_boosting(X_train, y_train, tune),
        "svm": _fit_svm(X_train, y_train, tune),  # ← new
    }
```

### 3. Use GridSearch (optional)

```python
from sklearn.svm import SVC

SVM_PARAM_GRID = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly'],
}

def _fit_svm(X_train, y_train, tune):
    model = SVC(probability=True, random_state=RANDOM_SEED)
    if tune:
        gs = GridSearchCV(model, SVM_PARAM_GRID, cv=CV_FOLDS,
                          scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        logger.info(f"Best params: {gs.best_params_}")
        return gs.best_estimator_
    model.fit(X_train, y_train)
    return model
```

## Using a Real-World Dataset

The expected schema is defined in `src/config.py`:

```python
COLUMNS = [
    "User_ID", "Date", "Program_Name", "Program_Genre",
    "Program_Class", "Watch_Duration", "Season", "Episode"
]
```

Your CSV must have these exact column names. If your data uses different column names, map them before loading:

```python
import pandas as pd

column_map = {
    'user': 'User_ID',
    'timestamp': 'Date',
    'title': 'Program_Name',
    'genre': 'Program_Genre',
    'type': 'Program_Class',
    'duration_sec': 'Watch_Duration',
    'season_num': 'Season',
    'ep_num': 'Episode',
}

df = pd.read_csv('your_data.csv')
df = df.rename(columns=column_map)
df = df[COLUMNS]  # reorder + drop extras
df.to_csv('data/cbcb_synthetic.csv', index=False)
```

Then run the pipeline normally:

```bash
python main.py --train
```

### Data requirements:

| Requirement | Why | Check |
|---|---|---|
| At least 2 interactions per user | CBCB-S needs consecutive pairs | `df.groupby('User_ID').size().min() >= 2` |
| At least 3 interactions per user for CBCB-R | CBCB-R needs lookahead of 2 | `df.groupby('User_ID').size().min() >= 3` |
| Realistic watch durations | Min-max scaling works best with bounded values | 30–20,000 seconds typical |
| Sorted within user-group | Labels assume chronological order | `df.sort_values(['User_ID', 'Date'])` |

## Enabling Deep Learning

Install PyTorch:

```bash
pip install torch>=2.0
```

The `src/deep_learning.py` module auto-detects torch availability:

```python
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

### Training a sequence model

```python
from src.deep_learning import train_sequence_model
from src.dataset_generator import generate_dataset

df = generate_dataset(50000, 2000)
model = train_sequence_model(
    df,
    arch='lstm',        # 'lstm', 'gru', or 'transformer'
    seq_len=5,
    epochs=20,
    batch_size=64,
    lr=0.001
)
```

### Available architectures

| Architecture | Class | Use Case |
|---|---|---|
| LSTM | `LSTMRecommender` | General sequence modelling |
| GRU | `GRURecommender` | Faster training, similar quality |
| Transformer | `TransformerRecommender` | Capturing long-range dependencies |

The deep learning module predicts the **next genre** directly (sequence-to-class), rather than using the engineered feature matrix.

## Using Custom Features

To add new features, edit `src/feature_engineering.py`:

```python
def _add_user_total_watch_time(df):
    """Add total watch time per user as a feature."""
    user_total = df.groupby('User_ID')['watch_duration_scaled'].transform('sum')
    df['user_total_duration'] = user_total
    return df
```

Then call it inside `build_feature_matrix()`:

```python
def build_feature_matrix(df, label_col):
    df = _add_sequence_features(df)
    df = _add_user_total_watch_time(df)  # ← new
    # ... rest of the function
```

## See Also

- [API Reference: train.py](../03-reference/api-reference.md#srctrainpy)
- [API Reference: deep_learning.py](../03-reference/api-reference.md#srcdeep_learningpy)
- [Configuration Reference: model params](../03-reference/config-reference.md)
- [Label Design Rationale](../04-explanation/label-design.md)

# Data Schema Reference

> **Audience:** Developers working with datasets, feature engineering, or model inputs.
>
> **Scope:** Raw data schema, preprocessed output, feature matrix layout, label formats.

---

## Raw Input Schema

Every raw dataset must have exactly these 8 columns in any order:

```python
COLUMNS = [
    "User_ID",         # str  — anonymous user identifier
    "Date",            # str  — ISO date (YYYY-MM-DD)
    "Program_Name",    # str  — program title
    "Program_Genre",   # str  — one of 16 genres
    "Program_Class",   # str  — "Movie" or "Series"
    "Watch_Duration",  # int  — seconds watched
    "Season",          # int  — season number (0 for movies)
    "Episode",         # int  — episode number (0 for movies)
]
```

### Example Row

```
User_ID,Date,Program_Name,Program_Genre,Program_Class,Watch_Duration,Season,Episode
U_42,2025-01-15,The Action Chronicles,Action,Movie,2745,0,0
U_42,2025-01-16,Laugh Factory,Comedy,Series,1560,3,7
```

### Constraints

| Column | Type | Range | Notes |
|---|---|---|---|
| User_ID | string | Any non-empty | No nulls |
| Date | string | ISO format YYYY-MM-DD | Must be parseable by pd.to_datetime |
| Program_Name | string | Any non-empty | Not used as a feature |
| Program_Genre | string | One of 16 genres | See Genre System below |
| Program_Class | string | "Movie" or "Series" | Must match exactly |
| Watch_Duration | int | > 0 | Seconds. 0 or negative = invalid |
| Season | int | ≥ 0 | 0 for movies |
| Episode | int | ≥ 0 | 0 for movies |

### Genre System

```
Action, Adventure, Animation, Comedy, Crime, Documentary, Drama,
Fantasy, Horror, Mystery, Romance, Science_Fiction, Thriller,
War, Western, Musical
```

Any genre outside this list will be encoded as a new one-hot column by `handle_unknown='ignore'`.

---

## Preprocessed Output Schema

After `preprocess_pipeline()`:

| Column | Type | Description |
|---|---|---|
| User_ID | str | Preserved from raw |
| Date | datetime64 | Parsed from raw |
| Program_Name | str | Preserved |
| watch_duration_scaled | float [0, 1] | Min-max scaled Watch_Duration |
| Season | int | Preserved |
| Episode | int | Preserved |
| Program_Genre_Action | uint8 | One-hot encoded genre |
| Program_Genre_Adventure | uint8 | One-hot encoded genre |
| ... | uint8 | 16 total genre one-hots |
| Program_Class_Movie | uint8 | One-hot encoded class |
| Program_Class_Series | uint8 | One-hot encoded class |
| cbcb_s_label | int {0, 1} | (after CBCB-S label generation) |
| cbcb_r_label | int {0, 1, 2} | (after CBCB-R label generation) |

Rows removed during preprocessing:

1. **Clean pass:** rows with nulls, duplicates, Watch_Duration ≤ 0
2. **IQR pass:** rows with Watch_Duration outside Tukey fences

---

## Feature Matrix Schema

After `build_feature_matrix()`:

### X (Features) — ~24 columns

| Feature | Type | Range | Description |
|---|---|---|---|
| `watch_duration_scaled` | float64 | [0, 1] | Normalised engagement |
| `prev_genre_code` | int64 | [0, 15] | Label-encoded previous genre |
| `genre_repeat_run` | int64 | [1, n] | Current consecutive same-genre streak |
| `user_avg_duration` | float64 | [0, 1] | User's mean scaled watch time |
| `user_session_index` | float64 | [0, 1] | 0 = first interaction, 1 = last |
| `time_since_last_day` | float64 | [0, ∞) | Days since previous interaction |
| `Program_Genre_Action` | bool | 0/1 | One-hot genre flags |
| `Program_Genre_Adventure` | bool | 0/1 | ... |
| `Program_Genre_Comedy` | bool | 0/1 | ... |
| `Program_Genre_Drama` | bool | 0/1 | ... |
| `Program_Genre_Horror` | bool | 0/1 | ... |
| `Program_Genre_Romance` | bool | 0/1 | ... |
| `Program_Genre_Science_Fiction` | bool | 0/1 | ... |
| `Program_Genre_Thriller` | bool | 0/1 | ... |
| `Program_Genre_Animation` | bool | 0/1 | ... |
| `Program_Genre_Adventure` | bool | 0/1 | ... |
| `Program_Genre_Crime` | bool | 0/1 | ... |
| `Program_Genre_Fantasy` | bool | 0/1 | ... |
| `Program_Genre_Mystery` | bool | 0/1 | ... |
| `Program_Genre_War` | bool | 0/1 | ... |
| `Program_Genre_Western` | bool | 0/1 | ... |
| `Program_Genre_Musical` | bool | 0/1 | ... |
| `Program_Class_Movie` | bool | 0/1 | Class one-hot |
| `Program_Class_Series` | bool | 0/1 | Class one-hot |

### y (Labels)

| Task | Type | Values | Description |
|---|---|---|---|
| CBCB-S | int64 | {0, 1} | 0 = No Repeat, 1 = Repeat |
| CBCB-R | int64 | {0, 1, 2} | 0 = No Repeat, 1 = Immediate Repeat, 2 = Revert |

### Feature Names

```python
feature_names = [
    "watch_duration_scaled",
    "prev_genre_code",
    "genre_repeat_run",
    "user_avg_duration",
    "user_session_index",
    "time_since_last_day",
    # + all one-hot columns
]
```

---

## Target Label Distributions (Synthetic Data)

### CBCB-S (Binary)

```
Label 0 (No Repeat):  ~52%
Label 1 (Repeat):     ~48%
```

The ~0.45 repeat probability in data generation produces a near-balanced distribution.

### CBCB-R (Ternary)

```
Label 0 (No Repeat):        ~34%
Label 1 (Immediate Repeat): ~33%
Label 2 (Revert):           ~33%
```

The embedded revert structure creates a well-balanced 3-class problem.

---

## Sequence Data (Deep Learning)

For `src/deep_learning.py`, sequences are built from the raw dataframe:

```python
# build_sequences(df, seq_len=5) builds sliding windows per user:
# Input sequence (5 genres):  [Action, Action, Comedy, Drama, Action]
# Target (next genre):         Comedy
```

| Variable | Shape | Description |
|---|---|---|
| X_seq | (n_sequences, seq_len) | Integer-encoded genre sequence |
| y_seq | (n_sequences,) | Next genre (integer-encoded) |
| vocab | dict | Genre → int mapping (16 entries) |

## Persisted Formats

### Model Files (`.joblib`)

```
models/cbcb_s__decision_tree.joblib     # Fitted sklearn model
models/cbcb_s__random_forest.joblib
models/cbcb_s__gradient_boosting.joblib
models/cbcb_s__meta.joblib              # {feature_names, classes, task, ...}
models/cbcb_r__decision_tree.joblib
models/cbcb_r__random_forest.joblib
models/cbcb_r__gradient_boosting.joblib
models/cbcb_r__meta.joblib
```

Metrics JSON keys:

```json
{
  "cbcb_s": {
    "decision_tree": {
      "accuracy": 0.7834,
      "precision": 0.7821,
      "recall": 0.7834,
      "f1": 0.7828,
      "per_class_precision": [0.78, 0.79],
      "per_class_recall": [0.79, 0.78],
      "confusion_matrix": [[...]],
      "roc_auc": null,
      "roc_curves": [...],
      "feature_importance": {"watch_duration_scaled": 0.32, ...}
    },
    "random_forest": {...},
    "gradient_boosting": {...}
  },
  "cbcb_r": {
    "decision_tree": {...},
    "random_forest": {...},
    "gradient_boosting": {...}
  }
}
```

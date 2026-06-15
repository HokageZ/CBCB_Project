"""Feature engineering: build the model matrix (X, y) from labelled data.

Combines the preprocessed features (scaled watch-time + one-hot genre/class)
with per-user sequential signals that give the CBCB models temporal context:

    * watch_duration_scaled   normalised engagement (Eq. 4)
    * prev_genre_code         the previous interaction's genre (label-encoded)
    * genre_repeat_run        length of the current same-genre streak
    * user_avg_duration       user's mean (scaled) watch time
    * user_session_index      0..1 position of the row in the user's history
    * time_since_last_day     days since the user's previous interaction
    * one-hot Program_Genre / Program_Class columns

The label column (CBCB-S or CBCB-R) is separated out as y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import LOG


def _add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-user sequential features. Assumes sorted by (User_ID, Date)."""
    df = df.sort_values(["User_ID", "Date"]).reset_index(drop=True)

    genre_codes = df["Program_Genre"].astype("category").cat.codes
    df["_genre_code"] = genre_codes
    df["prev_genre_code"] = df.groupby("User_ID")["_genre_code"].shift(1).fillna(-1)

    # Same-genre run length (consecutive repeats up to and including this row).
    same_as_prev = df["_genre_code"] == df.groupby("User_ID")["_genre_code"].shift(1)
    run = same_as_prev.groupby((~same_as_prev).cumsum()).cumsum()
    df["genre_repeat_run"] = run.astype(int)

    scaled_col = f"{config.SCALE_FEATURE}_scaled"
    if scaled_col in df.columns:
        df["user_avg_duration"] = df.groupby("User_ID")[scaled_col].transform("mean")
    else:
        df["user_avg_duration"] = 0.0

    # Position of the row within the user's history, normalised to [0, 1].
    df["_seq"] = df.groupby("User_ID").cumcount()
    counts = df.groupby("User_ID")["_seq"].transform("max").replace(0, 1)
    df["user_session_index"] = df["_seq"] / counts

    # Days since previous interaction for the same user.
    if "Date" in df.columns:
        date = pd.to_datetime(df["Date"])
        delta = date.groupby(df["User_ID"]).diff().dt.days
        df["time_since_last_day"] = delta.fillna(0.0)
    else:
        df["time_since_last_day"] = 0.0

    return df.drop(columns=["_genre_code", "_seq"])


# Engineered (non-one-hot) numeric feature names, in stable order.
SEQUENCE_FEATURES = [
    f"{config.SCALE_FEATURE}_scaled",
    "prev_genre_code",
    "genre_repeat_run",
    "user_avg_duration",
    "user_session_index",
    "time_since_last_day",
]


def build_feature_matrix(
    df: pd.DataFrame, label_col: str
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Assemble (X, y, feature_names) for a CBCB task.

    Parameters
    ----------
    df         A *preprocessed* frame (one-hot + scaled columns present) that
               already carries ``label_col`` from cbcb_s/cbcb_r.
    label_col  config.LABEL_COL_S or config.LABEL_COL_R.

    Returns
    -------
    X  Feature DataFrame (numeric only).
    y  Label Series.
    feature_names  Ordered list of columns in X.
    """
    if label_col not in df.columns:
        raise ValueError(f"Expected label column '{label_col}' in dataframe.")

    df = _add_sequence_features(df)

    # One-hot columns produced by preprocessing.one_hot_encode().
    onehot_cols = [
        c for c in df.columns
        if c.startswith("Program_Genre_") or c.startswith("Program_Class_")
    ]
    feature_names = [c for c in SEQUENCE_FEATURES if c in df.columns] + onehot_cols

    X = df[feature_names].astype(float).fillna(0.0)
    y = df[label_col].astype(int)

    LOG.info(
        "build_feature_matrix(%s): X=%s, classes=%s",
        label_col, X.shape, sorted(y.unique().tolist()),
    )
    return X, y, feature_names


if __name__ == "__main__":
    from .dataset_generator import generate_dataset
    from .preprocessing import preprocess_pipeline
    from . import cbcb_s

    raw = generate_dataset(n_rows=5_000, n_users=200)
    processed, _, _ = preprocess_pipeline(raw)
    labelled = cbcb_s.generate_labels(processed)
    X, y, names = build_feature_matrix(labelled, config.LABEL_COL_S)
    print("X:", X.shape, "| features:", len(names))
    print("y distribution:", y.value_counts().to_dict())

"""Data preprocessing pipeline (paper §4.1).

Implements the three transformation functions composed as X' = fs(fe(fc(X))):

    fc  cleaning            -> clean(), remove_outliers_iqr()
    fe  one-hot encoding    -> one_hot_encode()
    fs  min-max scaling     -> min_max_scale()

Fitted encoders/scalers are returned so the exact same transformation can be
re-applied at prediction time inside the Streamlit app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from . import config
from .utils import LOG


# --------------------------------------------------------------------------- #
# Container for fitted transformers (so prediction reuses train-time fits)
# --------------------------------------------------------------------------- #
@dataclass
class Transformers:
    encoder: OneHotEncoder
    scaler: MinMaxScaler
    encoded_columns: list[str]


# --------------------------------------------------------------------------- #
# fc — cleaning
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing/duplicate/invalid rows (the fc function, Eq. 1).

    * Missing values in required columns are dropped.
    * Exact duplicate rows are removed.
    * Non-positive watch durations are invalid and dropped.
    """
    before = len(df)
    df = df.copy()

    required = [c for c in config.COLUMNS if c in df.columns]
    df = df.dropna(subset=required)
    df = df.drop_duplicates()
    if config.SCALE_FEATURE in df.columns:
        df = df[df[config.SCALE_FEATURE] > 0]

    df = df.reset_index(drop=True)
    LOG.info("clean(): %d -> %d rows (removed %d)", before, len(df), before - len(df))
    return df


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Return (lower, upper) Tukey fences from the inter-quartile range."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def remove_outliers_iqr(
    df: pd.DataFrame, column: str = config.SCALE_FEATURE
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove outliers in ``column`` using the IQR method (paper §4.1.1).

    Returns the filtered frame plus a stats dict (bounds and before/after
    descriptives) used by the outlier before/after visualisation.
    """
    before_series = df[column].copy()
    lower, upper = iqr_bounds(df[column])
    mask = (df[column] >= lower) & (df[column] <= upper)
    cleaned = df[mask].reset_index(drop=True)

    stats = {
        "column": column,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "n_before": int(len(df)),
        "n_after": int(len(cleaned)),
        "n_removed": int(len(df) - len(cleaned)),
        "before_values": before_series,           # Series, for boxplots
        "after_values": cleaned[column].copy(),
    }
    LOG.info(
        "remove_outliers_iqr(%s): removed %d rows outside [%.1f, %.1f]",
        column, stats["n_removed"], lower, upper,
    )
    return cleaned, stats


# --------------------------------------------------------------------------- #
# fe — one-hot encoding
# --------------------------------------------------------------------------- #
def _make_encoder() -> OneHotEncoder:
    """OneHotEncoder that tolerates unseen categories at predict time."""
    try:  # sklearn >= 1.2
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # older sklearn
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def one_hot_encode(
    df: pd.DataFrame,
    columns: list[str] = config.CATEGORICAL_FEATURES,
    encoder: OneHotEncoder | None = None,
) -> tuple[pd.DataFrame, OneHotEncoder, list[str]]:
    """One-hot encode categorical columns (the fe function, Eqs. 2–3).

    If ``encoder`` is provided it is reused (transform only); otherwise a new
    encoder is fitted. Returns (encoded_df, encoder, encoded_column_names).
    """
    df = df.copy()
    if encoder is None:
        encoder = _make_encoder()
        encoded = encoder.fit_transform(df[columns])
    else:
        encoded = encoder.transform(df[columns])

    encoded_cols = list(encoder.get_feature_names_out(columns))
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    # Keep the original categorical columns: downstream CBCB labelling and
    # feature engineering still need the raw Program_Genre sequence. The numeric
    # feature matrix later selects only numeric + one-hot columns, so the string
    # columns never leak into the models.
    out = pd.concat([df, encoded_df], axis=1)
    return out, encoder, encoded_cols


# --------------------------------------------------------------------------- #
# fs — min-max scaling
# --------------------------------------------------------------------------- #
def min_max_scale(
    df: pd.DataFrame,
    column: str = config.SCALE_FEATURE,
    scaler: MinMaxScaler | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Min-Max scale ``column`` to [0, 1] (the fs function, Eq. 4).

    A new ``<column>_scaled`` column is added; the original is kept so the
    raw watch-time remains available for EDA and engagement scoring.
    """
    df = df.copy()
    values = df[[column]].to_numpy()
    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)
    else:
        scaled = scaler.transform(values)

    df[f"{column}_scaled"] = scaled.ravel()
    return df, scaler


# --------------------------------------------------------------------------- #
# Full composition: X' = fs(fe(fc(X)))
# --------------------------------------------------------------------------- #
def preprocess_pipeline(
    df: pd.DataFrame,
    transformers: Transformers | None = None,
    drop_outliers: bool = True,
) -> tuple[pd.DataFrame, Transformers, dict[str, Any]]:
    """Run the full preprocessing composition (Eq. 5).

    Parameters
    ----------
    df            Raw dataset.
    transformers  Optional pre-fitted transformers (for prediction-time reuse).
    drop_outliers Whether to apply IQR outlier removal (skip when transforming
                  a single prediction row).

    Returns
    -------
    (processed_df, fitted_transformers, outlier_stats)
    """
    # fc — clean (+ optional IQR outlier removal)
    cleaned = clean(df)
    outlier_stats: dict[str, Any] = {}
    if drop_outliers and config.SCALE_FEATURE in cleaned.columns:
        cleaned, outlier_stats = remove_outliers_iqr(cleaned)

    # fe — one-hot encode
    enc = transformers.encoder if transformers else None
    encoded_df, encoder, encoded_cols = one_hot_encode(cleaned, encoder=enc)

    # fs — min-max scale
    scl = transformers.scaler if transformers else None
    scaled_df, scaler = min_max_scale(encoded_df, scaler=scl)

    fitted = transformers or Transformers(
        encoder=encoder, scaler=scaler, encoded_columns=encoded_cols
    )
    LOG.info("preprocess_pipeline(): final shape %s", scaled_df.shape)
    return scaled_df, fitted, outlier_stats


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric correlation matrix for the EDA / heatmap visualisation."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr()


if __name__ == "__main__":
    from .dataset_generator import generate_dataset

    raw = generate_dataset(n_rows=5_000, n_users=200)
    processed, t, stats = preprocess_pipeline(raw)
    print("Raw:", raw.shape, "-> Processed:", processed.shape)
    print("Encoded columns:", len(t.encoded_columns))
    print("Outliers removed:", stats.get("n_removed"))
    print(processed.head().to_string(index=False))

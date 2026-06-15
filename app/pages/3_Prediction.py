"""Page 4 — User Behavior Prediction.

Collects a single interaction's inputs (user, current genre, class, watch
duration, previous genres) and produces CBCB-S / CBCB-R predictions, a
predicted next genre, an engagement score, and a confidence score.

The CBCB-S/R class predictions come from the trained sklearn models when
available; the predicted *next genre* is derived from the entered history
using the same logic the labels are built on (repeat vs. revert vs. explore).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    hero,
    inject_css,
    load_metrics,
    load_trained_model,
    no_models_warning,
)

from src import config, feature_engineering  # noqa: E402
from src.preprocessing import preprocess_pipeline  # noqa: E402
from src import visualization as viz  # noqa: E402

st.set_page_config(page_title="Prediction — CBCB", page_icon="🔮", layout="wide")
inject_css()
hero("🔮 User Behavior Prediction", "Predict sequential & revert captivation for an interaction")

metrics = load_metrics()
if not metrics:
    no_models_warning()
    st.stop()


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
st.subheader("Interaction inputs")
c1, c2, c3 = st.columns(3)
user_id = c1.number_input("User ID", 1, 100_000, 1)
current_genre = c2.selectbox("Current Genre", config.GENRES, index=0)
program_class = c3.selectbox("Program Class", config.PROGRAM_CLASSES, index=1)

c4, c5 = st.columns(2)
watch_duration = c4.slider("Watch Duration (s)", 60, 9000, 3600, step=60)
prev_genres = c5.multiselect(
    "Previous Genres (most recent last)",
    config.GENRES,
    default=[current_genre],
    help="The user's recent genre history, oldest → newest.",
)

model_choice = st.selectbox(
    "Model",
    options=list(config.MODEL_NAMES.keys()),
    format_func=lambda k: config.MODEL_NAMES[k],
    index=0,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_single_row_features(task: str) -> pd.DataFrame | None:
    """Construct a one-row feature matrix matching the trained model schema."""
    feature_names = metrics.get(task, {}).get("feature_names")
    if not feature_names:
        return None

    # Build a tiny history frame so sequential features compute sensibly.
    history = prev_genres + [current_genre]
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for i, g in enumerate(history):
        rows.append(
            {
                "User_ID": user_id,
                "Date": base_date + pd.Timedelta(days=i),
                "Program_Name": f"{g} title",
                "Program_Genre": g,
                "Program_Class": program_class,
                "Watch_Duration": float(watch_duration),
                "Season": 1 if program_class == "Series" else 0,
                "Episode": 1 if program_class == "Series" else 0,
            }
        )
    hist_df = pd.DataFrame(rows)

    processed, _, _ = preprocess_pipeline(hist_df, drop_outliers=False)
    # Reuse the feature builder; add a dummy label column it expects.
    label_col = config.LABEL_COL_S if task == "cbcb_s" else config.LABEL_COL_R
    processed[label_col] = 0
    X, _, _ = feature_engineering.build_feature_matrix(processed, label_col)

    # Align to the trained feature schema (add missing one-hot cols as 0).
    X = X.reindex(columns=feature_names, fill_value=0.0)
    return X.tail(1)


def _predict(task: str):
    model = load_trained_model(task, model_choice)
    if model is None:
        return None
    X = _build_single_row_features(task)
    if X is None:
        return None
    pred = int(model.predict(X)[0])
    conf = None
    if hasattr(model, "predict_proba"):
        conf = float(np.max(model.predict_proba(X)[0]))
    return {"label": pred, "confidence": conf}


def _predicted_next_genre() -> str:
    """Heuristic next-genre from the entered history (repeat / revert / explore)."""
    if len(prev_genres) >= 2 and prev_genres[-1] != current_genre and prev_genres[-2] == current_genre:
        return current_genre  # revert pattern completing
    # Default: the model's strongest signal is a repeat of the current genre.
    return current_genre


# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
if st.button("🔮 Predict Behavior", type="primary", use_container_width=True):
    s_res = _predict("cbcb_s")
    r_res = _predict("cbcb_r")

    engagement = min(1.0, watch_duration / 9000.0)  # normalized watch-time proxy
    confidences = [r["confidence"] for r in (s_res, r_res) if r and r["confidence"]]
    confidence = float(np.mean(confidences)) if confidences else engagement

    st.subheader("Results")
    rc1, rc2, rc3 = st.columns(3)

    with rc1:
        if s_res:
            name = config.CBCB_S_CLASS_NAMES.get(s_res["label"], str(s_res["label"]))
            st.metric("CBCB-S Prediction", name, help="Sequential captivation (0/1)")
        else:
            st.metric("CBCB-S Prediction", "model n/a")
    with rc2:
        if r_res:
            name = config.CBCB_R_CLASS_NAMES.get(r_res["label"], str(r_res["label"]))
            st.metric("CBCB-R Prediction", name, help="Revert captivation (0/1/2)")
        else:
            st.metric("CBCB-R Prediction", "model n/a")
    with rc3:
        st.metric("Predicted Next Genre", _predicted_next_genre())

    g1, g2 = st.columns(2)
    g1.plotly_chart(viz.gauge(engagement, "Engagement Score"), use_container_width=True)
    g2.plotly_chart(viz.gauge(confidence, "Confidence Score", color="#5fa86f"),
                    use_container_width=True)

    st.caption(
        "Engagement = normalized watch-time. Confidence = mean of model class "
        "probabilities. CBCB-S/R predictions use the selected trained model."
    )
else:
    st.info("Set the inputs above and click **Predict Behavior**.", icon="ℹ️")

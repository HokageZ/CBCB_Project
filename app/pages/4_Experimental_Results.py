"""Page 5 — Experimental Results.

Shows the model performance table, per-metric bar charts, per-model confusion
matrices, feature importance, and an automatically-selected best model per
metric plus a deployment recommendation — for the chosen CBCB task.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import hero, inject_css, load_metrics, no_models_warning  # noqa: E402

from src import config, evaluate  # noqa: E402
from src import visualization as viz  # noqa: E402

st.set_page_config(page_title="Experimental Results — CBCB", page_icon="📊", layout="wide")
inject_css()
hero("📊 Experimental Results", "Model performance, confusion matrices & feature importance")

metrics = load_metrics()
if not metrics:
    no_models_warning()
    st.stop()

task = st.selectbox(
    "CBCB task",
    options=list(metrics.keys()),
    format_func=lambda t: "CBCB-S (Sequential)" if t == "cbcb_s" else "CBCB-R (Revert)",
)
results = metrics[task]["results"]
feature_names = metrics[task].get("feature_names", [])


# --------------------------------------------------------------------------- #
# Performance table
# --------------------------------------------------------------------------- #
st.subheader("Model Performance")
table = evaluate.comparison_table(results)
st.dataframe(
    table.style.format(
        {c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]}
    ).background_gradient(cmap="Greens", subset=["Accuracy", "F1-Score"]),
    use_container_width=True,
)


# --------------------------------------------------------------------------- #
# Metric bar charts
# --------------------------------------------------------------------------- #
st.subheader("Visual Comparisons")
bc1, bc2 = st.columns(2)
bc3, bc4 = st.columns(2)
for col, metric in zip([bc1, bc2, bc3, bc4], config.METRIC_KEYS):
    col.pyplot(viz.metric_bar(results, metric), use_container_width=True)


# --------------------------------------------------------------------------- #
# Confusion matrices
# --------------------------------------------------------------------------- #
st.subheader("Confusion Matrices")
cols = st.columns(min(3, len(results)))
for i, (key, m) in enumerate(results.items()):
    with cols[i % len(cols)]:
        st.pyplot(
            viz.confusion_heatmap(m["confusion_matrix"], m["classes"], title=m["name"]),
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
st.subheader("Feature Importance")
fi_models = {k: m for k, m in results.items() if m.get("feature_importances")}
if fi_models and feature_names:
    fi_choice = st.selectbox(
        "Model",
        list(fi_models.keys()),
        format_func=lambda k: config.MODEL_NAMES.get(k, k),
    )
    importances = fi_models[fi_choice]["feature_importances"]
    st.pyplot(
        viz.feature_importance_bar(importances, feature_names, top_n=15),
        use_container_width=True,
    )

    # Watch-time & genre importance call-outs.
    s = pd.Series(importances, index=feature_names)
    wt = s.get(f"{config.SCALE_FEATURE}_scaled", 0.0)
    genre_imp = s[[c for c in s.index if c.startswith("Program_Genre_")]].sum()
    ic1, ic2 = st.columns(2)
    ic1.metric("Watch-Time Importance", f"{wt:.3f}")
    ic2.metric("Total Genre Importance", f"{genre_imp:.3f}")
else:
    st.info("Feature importances are available after training tree-based models.")


# --------------------------------------------------------------------------- #
# Best model + recommendation
# --------------------------------------------------------------------------- #
st.subheader("🏆 Best Model & Recommendation")
best = metrics[task].get("best", evaluate.best_models(results))
b1, b2, b3, b4 = st.columns(4)
for col, metric in zip([b1, b2, b3, b4], config.METRIC_KEYS):
    key = best[metric]
    col.metric(
        f"Best {metric.capitalize()}",
        config.MODEL_NAMES.get(key, key),
        f"{results[key][metric]:.3f}",
    )

recommended = metrics[task].get("recommended", evaluate.recommend_model(results))
st.success(
    f"**Recommended deployment model: "
    f"{config.MODEL_NAMES.get(recommended, recommended)}** "
    f"(highest mean across Accuracy/Precision/Recall/F1).",
    icon="✅",
)

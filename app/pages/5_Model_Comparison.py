"""Page 6 — Model Comparison.

Advanced cross-model visuals: a radar chart, an interactive grouped bar chart,
and a sortable leaderboard combining both CBCB tasks.
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

st.set_page_config(page_title="Model Comparison — CBCB", page_icon="📈", layout="wide")
inject_css()
hero("📈 Model Comparison", "Radar, interactive charts & leaderboard")

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


# --------------------------------------------------------------------------- #
# Radar + interactive bar
# --------------------------------------------------------------------------- #
rc1, rc2 = st.columns(2)
with rc1:
    st.subheader("Radar")
    st.plotly_chart(viz.radar_chart(results), use_container_width=True)
with rc2:
    st.subheader("Grouped Metrics")
    table = evaluate.comparison_table(results)
    st.plotly_chart(viz.interactive_metric_bar(table), use_container_width=True)


# --------------------------------------------------------------------------- #
# Leaderboard (combined across tasks)
# --------------------------------------------------------------------------- #
st.subheader("🏅 Leaderboard (all tasks)")
rows = []
for t, payload in metrics.items():
    label = "CBCB-S" if t == "cbcb_s" else "CBCB-R"
    for key, m in payload["results"].items():
        rows.append(
            {
                "Task": label,
                "Model": m["name"],
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1"],
                "AUC": m.get("auc_macro", float("nan")),
                "Mean": sum(m[k] for k in config.METRIC_KEYS) / 4,
            }
        )
board = pd.DataFrame(rows).sort_values("Mean", ascending=False).reset_index(drop=True)
board.index += 1
st.dataframe(
    board.style.format(
        {c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Mean"]}
    ).background_gradient(cmap="Blues", subset=["Mean"]),
    use_container_width=True,
)

top = board.iloc[0]
st.success(
    f"Top overall: **{top['Model']}** on **{top['Task']}** "
    f"(mean score {top['Mean']:.3f}).",
    icon="🏆",
)

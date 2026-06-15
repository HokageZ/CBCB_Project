"""Chart builders for EDA, results, and the Streamlit app.

Every function is a pure builder that *returns* a figure (matplotlib Figure or
Plotly Figure) without calling plt.show(), so the same code serves both
main.py (which saves PNGs to assets/) and Streamlit (which renders them).
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend; safe for servers and Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid")
_PALETTE = "viridis"


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def watch_time_distribution(df: pd.DataFrame, column: str = config.SCALE_FEATURE):
    """Histogram + KDE of watch-time durations."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df[column], bins=60, kde=True, color="#3b6ea5", ax=ax)
    ax.set_title("Watch-Time Duration Distribution")
    ax.set_xlabel("Watch Duration (s)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def outliers_before_after(stats: dict[str, Any]):
    """Side-by-side boxplots of watch-time before/after IQR cleaning."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    sns.boxplot(y=stats["before_values"], color="#d98c5f", ax=axes[0])
    axes[0].set_title(f"Before IQR (n={stats['n_before']:,})")
    axes[0].set_ylabel("Watch Duration (s)")

    sns.boxplot(y=stats["after_values"], color="#5fa86f", ax=axes[1])
    axes[1].set_title(f"After IQR (n={stats['n_after']:,})")
    axes[1].set_ylabel("")
    fig.suptitle(
        f"IQR Outlier Removal — bounds [{stats['lower_bound']:.0f}, "
        f"{stats['upper_bound']:.0f}] s, removed {stats['n_removed']:,}"
    )
    fig.tight_layout()
    return fig


def genre_distribution(df: pd.DataFrame, column: str = "Program_Genre"):
    """Count of interactions per genre."""
    order = df[column].value_counts().index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.countplot(data=df, y=column, order=order, hue=column,
                  palette=_PALETTE, legend=False, ax=ax)
    ax.set_title("Genre Distribution")
    ax.set_xlabel("Interactions")
    ax.set_ylabel("Genre")
    fig.tight_layout()
    return fig


def correlation_heatmap(corr: pd.DataFrame):
    """Heatmap of a numeric correlation matrix."""
    n = max(6, min(14, len(corr) * 0.6))
    fig, ax = plt.subplots(figsize=(n, n * 0.8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax,
                square=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return fig


def user_activity(df: pd.DataFrame):
    """Distribution of interactions per user."""
    counts = df.groupby("User_ID").size()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(counts, bins=50, color="#7b5ea7", ax=ax)
    ax.set_title("User Activity (interactions per user)")
    ax.set_xlabel("Interactions")
    ax.set_ylabel("Users")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def metric_bar(results: dict[str, dict], metric: str):
    """Bar chart of a single metric across models (matplotlib)."""
    names = [results[k]["name"] for k in results]
    values = [results[k][metric] for k in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=sns.color_palette(_PALETTE, len(names)))
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric.capitalize()} by Model")
    ax.set_ylabel(metric.capitalize())
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def confusion_heatmap(cm: list[list[int]], classes: list[int], title: str = ""):
    """Confusion-matrix heatmap (matplotlib)."""
    cm_arr = np.asarray(cm)
    labels = [config.CBCB_R_CLASS_NAMES.get(c, str(c)) if len(classes) > 2
              else config.CBCB_S_CLASS_NAMES.get(c, str(c)) for c in classes]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title or "Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def feature_importance_bar(importances: list[float], feature_names: list[str], top_n: int = 15):
    """Top-N feature importances (matplotlib)."""
    s = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    s.plot(kind="barh", color="#3b6ea5", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Interactive (Plotly) — used by Model Comparison page
# --------------------------------------------------------------------------- #
def radar_chart(results: dict[str, dict]) -> go.Figure:
    """Plotly radar chart comparing all models across the four metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    keys = config.METRIC_KEYS
    fig = go.Figure()
    for model_key, m in results.items():
        values = [m[k] for k in keys]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=m["name"],
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Comparison — Radar",
        showlegend=True,
    )
    return fig


def interactive_metric_bar(table: pd.DataFrame) -> go.Figure:
    """Grouped Plotly bar chart from a comparison table."""
    fig = go.Figure()
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        fig.add_trace(go.Bar(name=metric, x=table["Model"], y=table[metric]))
    fig.update_layout(
        barmode="group", title="Model Metrics Comparison",
        yaxis=dict(range=[0, 1], title="Score"), xaxis_title="Model",
    )
    return fig


def gauge(value: float, title: str, color: str = "#3b6ea5") -> go.Figure:
    """Plotly gauge for engagement/confidence scores on the Prediction page."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#f2dede"},
                    {"range": [40, 70], "color": "#fcf8e3"},
                    {"range": [70, 100], "color": "#dff0d8"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def save_fig(fig, path) -> None:
    """Persist a matplotlib figure to PNG and close it."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

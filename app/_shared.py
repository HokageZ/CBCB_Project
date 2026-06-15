"""Shared helpers for the Streamlit app (path setup, styling, model loading).

Imported by Home.py and every page so the project root is on sys.path and the
look-and-feel stays consistent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src` importable regardless of where Streamlit is launched from.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.utils import load_json, load_model  # noqa: E402


PRIMARY = "#3b6ea5"


def inject_css() -> None:
    """Apply a small amount of custom CSS for a polished look."""
    st.markdown(
        f"""
        <style>
        .main {{ background-color: #fafbfc; }}
        .cbcb-hero {{
            background: linear-gradient(120deg, {PRIMARY} 0%, #2c3e63 100%);
            padding: 1.6rem 2rem; border-radius: 14px; color: white;
            margin-bottom: 1.2rem;
        }}
        .cbcb-hero h1 {{ color: white; margin-bottom: .3rem; font-size: 1.9rem; }}
        .cbcb-hero p {{ color: #e8eef7; margin: 0; }}
        .metric-card {{
            background: white; border: 1px solid #e6e9ef; border-radius: 12px;
            padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,.04);
        }}
        .pill {{
            display:inline-block; padding:.2rem .7rem; border-radius:999px;
            background:#eef3fa; color:{PRIMARY}; font-size:.8rem; margin:.15rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    """Render the gradient hero banner used at the top of each page."""
    st.markdown(
        f'<div class="cbcb-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def load_metrics() -> dict | None:
    """Load models/metrics.json (written by main.py), or None if absent."""
    return load_json(config.METRICS_PATH)


def available_tasks(metrics: dict | None) -> list[str]:
    return list(metrics.keys()) if metrics else []


def load_trained_model(task: str, model_key: str):
    """Load a persisted model for a task, or None if not on disk."""
    from src.utils import model_path

    try:
        return load_model(model_path(task, model_key))
    except FileNotFoundError:
        return None


def no_models_warning() -> None:
    st.warning(
        "No trained models found yet. Go to **Dataset Manager** and click "
        "**Train Models**, or run `python main.py --all` from a terminal.",
        icon="⚠️",
    )

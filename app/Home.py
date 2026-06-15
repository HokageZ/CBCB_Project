"""CBCB Streamlit app — entry point (Home / Page 1).

Run with:
    streamlit run app/Home.py

Uses the modern multipage pattern: the `pages/` directory is auto-discovered
by Streamlit and rendered in the sidebar in filename order.
"""
from __future__ import annotations

import streamlit as st

from _shared import hero, inject_css

st.set_page_config(
    page_title="CBCB — Content-Based Captivation Behavior",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

hero(
    "🎬 CBCB — Content-Based Captivation Behavior",
    "Predicting user behavior on video streaming via watch-time duration analysis",
)

st.markdown(
    """
This application reproduces and extends the framework from
**“Predicting User Behavior on Video Streaming by Using Watch-Time Duration
Analysis”** (Knowledge-Based Systems, 2025). It models two complementary
short-term behaviors — **sequential captivation (CBCB-S)** and **revert
captivation (CBCB-R)** — and trains machine-learning models to predict them
from viewing history.
"""
)

# --------------------------------------------------------------------------- #
# Abstract
# --------------------------------------------------------------------------- #
st.subheader("📄 Abstract")
st.markdown(
    """
Existing video recommenders often rely solely on user–item interaction history
or optimize video-level watch-time, overlooking implicit factors such as
**watch-time duration** and intrinsic content characteristics. The **CBCB**
framework addresses this with a two-step approach: **CBCB-S** analyzes
historical viewing patterns to track engagement trends, while **CBCB-R**
identifies instances where users *return* to previously viewed content. By
integrating watch-time duration with historical behavior, CBCB captures both
sequential and revert behaviors at the user level — enabling more precise
personalized recommendations than video-level watch-time approaches.
"""
)

# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
st.subheader("🎯 Objectives")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
- Model **sequential** short-term behavior (CBCB-S)
- Model **revert** short-term behavior (CBCB-R)
- Use **watch-time duration** as an engagement signal
- Reproduce the paper's preprocessing pipeline
"""
    )
with col2:
    st.markdown(
        """
- Train & compare **Decision Tree / Random Forest / Gradient Boosting**
- Evaluate with **Accuracy, Precision, Recall, F1, ROC/AUC**
- Provide an interactive **prediction** interface
- Recommend a **deployment model** automatically
"""
    )

# --------------------------------------------------------------------------- #
# CBCB overview
# --------------------------------------------------------------------------- #
st.subheader("🧠 CBCB Overview")
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("#### CBCB-S — Sequential Captivation")
    st.markdown(
        """
Binary signal. For each interaction, label **1** if the user watches the
**same genre** on the very next interaction (`G1 → G1`), else **0**.

> *A user repeating a genre immediately is strongly engaged.*
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("#### CBCB-R — Revert Captivation")
    st.markdown(
        """
Three-class signal:
- **1** — immediate repeat (`G1 → G1`)
- **2** — revert (`G1 → G2 → G1`)
- **0** — neither

> *Captures users who diverge then return to a prior genre.*
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #
st.subheader("🔁 Project Workflow")
st.graphviz_chart(
    """
    digraph {
        rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#eef3fa"
                          color="#3b6ea5" fontname="Helvetica"];
        Dataset -> Cleaning -> Encoding -> Scaling -> "CBCB-S / CBCB-R"
        "CBCB-S / CBCB-R" -> "Feature Eng." -> "ML Models" -> Prediction -> Evaluation
    }
    """,
    use_container_width=True,
)

st.info(
    "Use the sidebar to navigate: **Dataset Manager** to generate/train, "
    "**CBCB Architecture** for the framework, **Prediction** for live inference, "
    "**Experimental Results** & **Model Comparison** for evaluation.",
    icon="👈",
)

"""Page 7 — About Project.

Paper summary, authors, methodology recap, references, and future work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import hero, inject_css  # noqa: E402

st.set_page_config(page_title="About — CBCB", page_icon="ℹ️", layout="wide")
inject_css()
hero("ℹ️ About this Project", "Research summary, methodology, and references")

st.subheader("📄 Research Paper")
st.markdown(
    """
**Predicting User Behavior on Video Streaming by Using Watch-Time Duration
Analysis.** *Knowledge-Based Systems* **332** (2025) 114779.
[doi:10.1016/j.knosys.2025.114779](https://doi.org/10.1016/j.knosys.2025.114779)

The paper introduces the **Content-Based Captivation Behavior (CBCB)**
framework — a two-step approach for short-term behavior prediction in video
streaming. It integrates **watch-time duration** with historical viewing
patterns to capture **sequential (CBCB-S)** and **revert (CBCB-R)** behaviors
at the user level, evaluated on the real-world STC/JAWWY dataset.
"""
)

st.subheader("👥 Authors")
st.markdown(
    """
- **Amir Monem**
- **Maha Elkohely**
- **Marwa Ashraf**
- **Sara Alaa**
"""
)

st.subheader("🔬 Methodology")
st.markdown(
    """
1. **Data collection** — STC/JAWWY viewing history (here: a realistic synthetic
   stand-in with the same schema).
2. **Preprocessing** — cleaning, IQR outlier removal, one-hot encoding,
   min-max scaling: `X' = fs(fe(fc(X)))`.
3. **Behavior labelling** — CBCB-S (binary) and CBCB-R (ternary) from each
   user's ordered history.
4. **Modelling** — Decision Tree, Random Forest, Gradient Boosting with
   GridSearchCV tuning (Decision Tree was the paper's strongest model).
5. **Evaluation** — Accuracy, Precision, Recall, F1, plus ROC/AUC and
   confusion matrices.
"""
)

st.subheader("🧭 This Implementation Extends the Paper With")
st.markdown(
    """
- A **synthetic dataset generator** with genuine sequential/revert structure.
- Optional **boosting** models (XGBoost, LightGBM, CatBoost).
- An optional **deep-learning** sequence module (LSTM, GRU, Transformer).
- A full **multi-page Streamlit** application for exploration and demo.
"""
)

st.subheader("🚀 Future Work")
st.markdown(
    """
- Reinforcement-learning–based recommendation policies.
- Graph-based methods over the user–item interaction graph.
- Real-time, on-device inference for streaming clients.
- Incorporating richer multimodal content features (audio/visual/text).
"""
)

st.subheader("📚 Selected References")
st.markdown(
    """
- Zheng et al. (2022). *DVR: micro-video recommendation optimizing
  watch-time-gain under duration bias.* ACM MM.
- Wu, Rizoiu, Xie (2018). *Beyond views: measuring and predicting engagement
  in online videos.* ICWSM.
- Liu et al. (2019). *User-video co-attention network for personalized
  micro-video recommendation.* WWW.
- Safavian & Landgrebe (1991). *A survey of decision tree classifier
  methodology.* IEEE TSMC.
"""
)

st.caption("Built as a Master's project — reproduction + extension of the CBCB framework.")

"""Page 3 — CBCB Architecture.

A visual walkthrough of the framework: the end-to-end flowchart, an
architecture/data-flow diagram, and detailed explanations of CBCB-S and
CBCB-R with the labelling rules from the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import hero, inject_css  # noqa: E402

st.set_page_config(page_title="CBCB Architecture", page_icon="🧩", layout="wide")
inject_css()
hero("🧩 CBCB Architecture", "From raw viewing history to behavior prediction")


# --------------------------------------------------------------------------- #
# End-to-end vertical flowchart
# --------------------------------------------------------------------------- #
st.subheader("End-to-End Pipeline")
flow_col, expl_col = st.columns([1, 1])

with flow_col:
    st.graphviz_chart(
        """
        digraph {
            node [shape=box style="rounded,filled" fontname="Helvetica"
                  fillcolor="#eef3fa" color="#3b6ea5"];
            Dataset -> "Data Cleaning" -> "Encoding (one-hot)" -> "Scaling (min-max)"
            "Scaling (min-max)" -> "CBCB-S"
            "Scaling (min-max)" -> "CBCB-R"
            "CBCB-S" -> "Feature Engineering"
            "CBCB-R" -> "Feature Engineering"
            "Feature Engineering" -> "ML Models" -> "Prediction" -> "Evaluation"
            "ML Models" [fillcolor="#dfe9f7"];
            "CBCB-S" [fillcolor="#dff0d8"];
            "CBCB-R" [fillcolor="#fcf3d8"];
        }
        """,
        use_container_width=True,
    )

with expl_col:
    st.markdown(
        """
**Stage by stage**

1. **Dataset** — viewing history `(User_ID, Date, Genre, Class, Watch_Duration, …)`
2. **Data Cleaning** — drop missing/duplicates; **IQR** removes watch-time outliers
3. **Encoding** — one-hot encode `Program_Genre`, `Program_Class` *(fe)*
4. **Scaling** — min-max scale `Watch_Duration` to [0, 1] *(fs)*
5. **CBCB-S / CBCB-R** — derive behavior labels from each user's ordered history
6. **Feature Engineering** — combine scaled + encoded + sequential features
7. **ML Models** — Decision Tree, Random Forest, Gradient Boosting
8. **Prediction** — next-genre / captivation prediction
9. **Evaluation** — Accuracy, Precision, Recall, F1, ROC/AUC
"""
    )
    st.latex(r"X' = f_s(f_e(f_c(X)))")


st.divider()

# --------------------------------------------------------------------------- #
# CBCB-S and CBCB-R explanations
# --------------------------------------------------------------------------- #
s_col, r_col = st.columns(2)

with s_col:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### CBCB-S — Sequential Captivation")
    st.markdown(
        """
For each user (ordered by date), compare the current genre to the **next** one:
"""
    )
    st.code(
        "y = 1   if genre[i] == genre[i+1]   (G1 → G1)\n"
        "y = 0   otherwise",
        language="text",
    )
    st.markdown("**Example** `Action → Action → Comedy → Action`")
    st.table(
        {
            "Step": ["Action→Action", "Action→Comedy", "Comedy→Action"],
            "Label": [1, 0, 0],
        }
    )
    st.caption("Binary cross-entropy loss (Eq. 16). Last row per user is dropped.")
    st.markdown("</div>", unsafe_allow_html=True)

with r_col:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### CBCB-R — Revert Captivation")
    st.markdown(
        "Looks **one and two** steps ahead to detect a return to a prior genre:"
    )
    st.code(
        "y = 1   if genre[i] == genre[i+1]                    (G1 → G1)\n"
        "y = 2   if genre[i] != genre[i+1] and genre[i]==genre[i+2]  (G1 → G2 → G1)\n"
        "y = 0   otherwise",
        language="text",
    )
    st.markdown("**Example** `Action → Comedy → Action`")
    st.table(
        {
            "Pattern": ["Action→Action", "Action→Comedy→Action", "Action→Comedy→Horror"],
            "Label": [1, 2, 0],
        }
    )
    st.caption("Multi-class loss (Eq. 19). Last two rows per user are dropped.")
    st.markdown("</div>", unsafe_allow_html=True)


st.divider()

# --------------------------------------------------------------------------- #
# Data-flow diagram
# --------------------------------------------------------------------------- #
st.subheader("Data-Flow Diagram")
st.graphviz_chart(
    """
    digraph {
        rankdir=LR; node [fontname="Helvetica"];
        raw [shape=cylinder label="Raw\\nViewing History" fillcolor="#eef3fa" style=filled];
        pre [shape=box style="rounded,filled" fillcolor="#dfe9f7" label="Preprocess\\nfc→fe→fs"];
        lab [shape=box style="rounded,filled" fillcolor="#dff0d8" label="Label\\nCBCB-S / CBCB-R"];
        feat [shape=box style="rounded,filled" fillcolor="#fcf3d8" label="Feature\\nMatrix (X, y)"];
        model [shape=component label="Trained\\nModels" fillcolor="#f7dfe9" style=filled];
        pred [shape=box style="rounded,filled" fillcolor="#eef3fa" label="Prediction\\n+ Scores"];
        raw -> pre -> lab -> feat -> model -> pred;
        model -> "models/*.joblib" [style=dashed];
    }
    """,
    use_container_width=True,
)

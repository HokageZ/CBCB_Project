"""Page 2 — Dataset Manager.

Generate or upload a dataset, preview it, view statistics, then preprocess and
train all models with single-click buttons and progress indicators. Trained
models and metrics are written to disk so the other pages can use them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import hero, inject_css  # noqa: E402

from src import (  # noqa: E402
    cbcb_r,
    cbcb_s,
    config,
    dataset_generator,
    evaluate,
    feature_engineering,
    train,
)
from src.preprocessing import preprocess_pipeline  # noqa: E402
from src.utils import save_json  # noqa: E402

st.set_page_config(page_title="Dataset Manager — CBCB", page_icon="🗂️", layout="wide")
inject_css()
hero("🗂️ Dataset Manager", "Generate or upload data, preprocess, and train models")


# --------------------------------------------------------------------------- #
# 1. Acquire a dataset
# --------------------------------------------------------------------------- #
st.subheader("1 · Get a dataset")
src_col, gen_col = st.columns(2)

with src_col:
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded is not None:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.session_state["raw_df"] = df
        st.success(f"Uploaded **{uploaded.name}** — {len(df):,} rows")

with gen_col:
    n_rows = st.number_input("Synthetic rows", 5_000, 300_000,
                             config.DEFAULT_N_ROWS, step=5_000)
    n_users = st.number_input("Users", 100, 20_000, config.DEFAULT_N_USERS, step=100)
    if st.button("🎲 Generate Synthetic Dataset", use_container_width=True):
        with st.status("Generating dataset...", expanded=False) as status:
            df = dataset_generator.generate_dataset(n_rows=int(n_rows), n_users=int(n_users))
            df.to_csv(config.RAW_DATA_PATH, index=False)
            st.session_state["raw_df"] = df
            status.update(label=f"Generated {len(df):,} rows", state="complete")

# Fall back to an on-disk dataset if one exists.
if "raw_df" not in st.session_state and config.RAW_DATA_PATH.exists():
    st.session_state["raw_df"] = pd.read_csv(config.RAW_DATA_PATH)

raw_df = st.session_state.get("raw_df")

if raw_df is None:
    st.info("Generate or upload a dataset to begin.", icon="ℹ️")
    st.stop()


# --------------------------------------------------------------------------- #
# 2. Preview & statistics
# --------------------------------------------------------------------------- #
st.subheader("2 · Preview & statistics")
st.dataframe(raw_df.head(20), use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(raw_df):,}")
m2.metric("Columns", raw_df.shape[1])
m3.metric("Missing values", f"{int(raw_df.isna().sum().sum()):,}")
m4.metric("Unique users", f"{raw_df['User_ID'].nunique():,}"
          if "User_ID" in raw_df.columns else "—")

with st.expander("Column data types"):
    st.dataframe(
        pd.DataFrame({"dtype": raw_df.dtypes.astype(str)}),
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# 3. Preprocess
# --------------------------------------------------------------------------- #
st.subheader("3 · Preprocess")
st.caption("Cleaning → IQR outlier removal → one-hot encoding → min-max scaling "
           "(X' = fs(fe(fc(X))))")
if st.button("🧹 Preprocess Dataset", use_container_width=True):
    progress = st.progress(0, text="Cleaning...")
    processed, transformers, outlier_stats = preprocess_pipeline(raw_df)
    progress.progress(100, text="Done")
    st.session_state["processed_df"] = processed
    st.session_state["outlier_stats"] = outlier_stats
    removed = outlier_stats.get("n_removed", 0)
    st.success(f"Processed → shape {processed.shape}. IQR removed {removed:,} outliers.")

if "processed_df" in st.session_state:
    st.download_button(
        "⬇️ Download processed dataset (CSV)",
        st.session_state["processed_df"].to_csv(index=False).encode(),
        file_name="cbcb_processed.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# 4. Train (single click)
# --------------------------------------------------------------------------- #
st.subheader("4 · Train models")
tcol1, tcol2, tcol3 = st.columns(3)
task_choice = tcol1.selectbox("Task", ["both", "cbcb_s", "cbcb_r"], index=0)
tune = tcol2.toggle("Tune (GridSearchCV)", value=False,
                    help="Slower but mirrors the paper's Table 8 tuning.")
boosting = tcol3.toggle("Include boosting (if installed)", value=True)

if st.button("🚀 Train Models", type="primary", use_container_width=True):
    tasks = ["cbcb_s", "cbcb_r"] if task_choice == "both" else [task_choice]
    summaries: dict = {}
    overall = st.progress(0, text="Starting training...")

    for i, task in enumerate(tasks):
        with st.status(f"Training {task.upper()}...", expanded=True) as status:
            st.write("Preprocessing...")
            processed, _, _ = preprocess_pipeline(raw_df)

            st.write("Generating CBCB labels...")
            if task == "cbcb_s":
                labelled = cbcb_s.generate_labels(processed)
                label_col = config.LABEL_COL_S
            else:
                labelled = cbcb_r.generate_labels(processed)
                label_col = config.LABEL_COL_R

            st.write("Building features...")
            X, y, names = feature_engineering.build_feature_matrix(labelled, label_col)

            st.write("Fitting Decision Tree / Random Forest / Gradient Boosting...")
            bundle = train.train_all(X, y, task, names, tune=tune,
                                     include_boosting=boosting, persist=True)

            summaries[task] = {
                "task": task,
                "results": bundle["results"],
                "best": bundle["best"],
                "feature_names": names,
                "recommended": evaluate.recommend_model(bundle["results"]),
            }
            status.update(label=f"{task.upper()} trained ✓", state="complete")

        overall.progress(int((i + 1) / len(tasks) * 100),
                         text=f"Finished {task.upper()}")

    # Merge with any previously-trained task summaries on disk.
    existing = save_json  # alias to avoid confusion; load below
    from src.utils import load_json
    prior = load_json(config.METRICS_PATH) or {}
    prior.update(summaries)
    save_json(prior, config.METRICS_PATH)

    st.success("Training complete — models and metrics saved to `models/`.")
    for task, s in summaries.items():
        st.markdown(f"**{task.upper()}** — recommended model: "
                    f"`{config.MODEL_NAMES.get(s['recommended'], s['recommended'])}`")
        st.dataframe(evaluate.comparison_table(s["results"]), use_container_width=True)

    st.info("Open **Experimental Results** and **Model Comparison** to explore.", icon="📊")

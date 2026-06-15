"""CBCB end-to-end CLI orchestrator.

Runs the full pipeline described in the paper:

    generate -> preprocess -> CBCB-S/CBCB-R labels -> features
             -> train (DT/RF/GB [+ boosting]) -> evaluate -> save

Examples
--------
    python main.py --all                 # 50k rows, both tasks, tuned models
    python main.py --all --rows 80000    # larger synthetic dataset
    python main.py --generate            # only (re)generate the dataset
    python main.py --train --no-tune     # train fast, skip GridSearchCV
    python main.py --task cbcb_s         # only the sequential task
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python main.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src import (
    cbcb_r,
    cbcb_s,
    config,
    dataset_generator,
    evaluate,
    feature_engineering,
    train,
    visualization as viz,
)
from src.preprocessing import preprocess_pipeline
from src.utils import LOG, save_json, set_seed


def _generate(rows: int, users: int) -> pd.DataFrame:
    return dataset_generator.generate_and_save(n_rows=rows, n_users=users)


def _load_or_generate(rows: int, users: int) -> pd.DataFrame:
    if config.RAW_DATA_PATH.exists():
        LOG.info("Loading existing dataset %s", config.RAW_DATA_PATH)
        return pd.read_csv(config.RAW_DATA_PATH, parse_dates=["Date"])
    return _generate(rows, users)


def _run_task(task: str, raw: pd.DataFrame, tune: bool, boosting: bool) -> dict:
    """Preprocess, label, train, and evaluate one CBCB task."""
    LOG.info("=" * 60)
    LOG.info("Running task: %s", task.upper())

    processed, _, outlier_stats = preprocess_pipeline(raw)

    if task == "cbcb_s":
        labelled = cbcb_s.generate_labels(processed)
        label_col = config.LABEL_COL_S
    else:
        labelled = cbcb_r.generate_labels(processed)
        label_col = config.LABEL_COL_R

    X, y, names = feature_engineering.build_feature_matrix(labelled, label_col)
    bundle = train.train_all(
        X, y, task, names, tune=tune, include_boosting=boosting, persist=True
    )

    # Save EDA outlier figure once (on the S task to avoid duplication).
    if task == "cbcb_s" and outlier_stats:
        fig = viz.outliers_before_after(outlier_stats)
        viz.save_fig(fig, config.ASSETS_DIR / "outliers_before_after.png")

    # Persist per-model confusion + metric figures and a JSON summary.
    table = evaluate.comparison_table(bundle["results"])
    print(f"\n=== {task.upper()} results ===")
    print(table.to_string(index=False))
    print("Best per metric:", bundle["best"])
    print("Recommended:", evaluate.recommend_model(bundle["results"]))

    for metric in config.METRIC_KEYS:
        fig = viz.metric_bar(bundle["results"], metric)
        viz.save_fig(fig, config.ASSETS_DIR / f"{task}_{metric}.png")

    for key, m in bundle["results"].items():
        fig = viz.confusion_heatmap(
            m["confusion_matrix"], m["classes"], title=f"{task.upper()} — {m['name']}"
        )
        viz.save_fig(fig, config.ASSETS_DIR / f"{task}_cm_{key}.png")

    # JSON-serialisable summary (drop the fitted models).
    summary = {
        "task": task,
        "results": bundle["results"],
        "best": bundle["best"],
        "feature_names": names,
        "recommended": evaluate.recommend_model(bundle["results"]),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CBCB end-to-end pipeline.")
    parser.add_argument("--all", action="store_true", help="generate + train both tasks")
    parser.add_argument("--generate", action="store_true", help="(re)generate dataset only")
    parser.add_argument("--train", action="store_true", help="train on existing dataset")
    parser.add_argument("--task", choices=["cbcb_s", "cbcb_r", "both"], default="both")
    parser.add_argument("--rows", type=int, default=config.DEFAULT_N_ROWS)
    parser.add_argument("--users", type=int, default=config.DEFAULT_N_USERS)
    parser.add_argument("--no-tune", action="store_true", help="skip GridSearchCV")
    parser.add_argument("--no-boosting", action="store_true", help="skip xgb/lgbm/catboost")
    args = parser.parse_args(argv)

    set_seed()
    tune = not args.no_tune
    boosting = not args.no_boosting

    # Decide which stages to run.
    do_generate = args.all or args.generate
    do_train = args.all or args.train or (not args.generate)

    raw = _generate(args.rows, args.users) if do_generate else _load_or_generate(args.rows, args.users)

    if args.generate and not (args.all or args.train):
        LOG.info("Dataset generated. Done.")
        return 0

    if not do_train:
        return 0

    tasks = ["cbcb_s", "cbcb_r"] if args.task == "both" else [args.task]
    all_summaries = {}
    for task in tasks:
        all_summaries[task] = _run_task(task, raw, tune, boosting)

    save_json(all_summaries, config.METRICS_PATH)
    LOG.info("All done. Metrics -> %s", config.METRICS_PATH)
    LOG.info("Figures -> %s", config.ASSETS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

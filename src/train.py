"""Model training (paper §4.3, §6).

Trains the three conventional models the paper evaluates — Decision Tree,
Random Forest, Gradient Boosting — with an optional GridSearchCV tuning pass
(the Decision Tree grid mirrors Table 8). Optionally also trains XGBoost,
LightGBM, and CatBoost if those libraries are installed (import-guarded).

A single call to :func:`train_all` fits every available model for a task,
evaluates it on the held-out test set, persists it, and returns a results
dict consumed by the Streamlit pages and main.py.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import config, evaluate
from .utils import LOG, model_path, save_model


# --------------------------------------------------------------------------- #
# Optional boosting libraries — import-guarded so the project runs without them
# --------------------------------------------------------------------------- #
def _optional_boosters() -> dict[str, Callable[..., Any]]:
    """Return {model_key: estimator_factory} for installed boosting libs only."""
    boosters: dict[str, Callable[..., Any]] = {}

    try:
        from xgboost import XGBClassifier

        def _make_xgb(num_classes: int):
            return XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.9, eval_metric="mlogloss",
                random_state=config.RANDOM_SEED, tree_method="hist",
            )

        boosters["xgboost"] = _make_xgb
    except ImportError:
        LOG.info("xgboost not installed - skipping (optional).")

    try:
        from lightgbm import LGBMClassifier

        def _make_lgbm(num_classes: int):
            return LGBMClassifier(
                n_estimators=200, max_depth=-1, learning_rate=0.1,
                random_state=config.RANDOM_SEED, verbose=-1,
            )

        boosters["lightgbm"] = _make_lgbm
    except ImportError:
        LOG.info("lightgbm not installed - skipping (optional).")

    try:
        from catboost import CatBoostClassifier

        def _make_catboost(num_classes: int):
            return CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                random_seed=config.RANDOM_SEED, verbose=False,
            )

        boosters["catboost"] = _make_catboost
    except ImportError:
        LOG.info("catboost not installed - skipping (optional).")

    return boosters


# --------------------------------------------------------------------------- #
# Core estimators
# --------------------------------------------------------------------------- #
def _fit_decision_tree(X_train, y_train, tune: bool) -> DecisionTreeClassifier:
    """Decision Tree, optionally tuned via GridSearchCV (mirrors Table 8)."""
    base = DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    if not tune:
        return base.fit(X_train, y_train)

    search = GridSearchCV(
        base, config.DT_PARAM_GRID, cv=config.CV_FOLDS,
        scoring="accuracy", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    LOG.info("DecisionTree best params: %s (CV acc=%.3f)",
             search.best_params_, search.best_score_)
    return search.best_estimator_


def _fit_random_forest(X_train, y_train, tune: bool) -> RandomForestClassifier:
    base = RandomForestClassifier(
        n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1
    )
    if not tune:
        return base.fit(X_train, y_train)

    search = GridSearchCV(
        base, config.RF_PARAM_GRID, cv=config.CV_FOLDS,
        scoring="accuracy", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    LOG.info("RandomForest best params: %s (CV acc=%.3f)",
             search.best_params_, search.best_score_)
    return search.best_estimator_


def _fit_gradient_boosting(X_train, y_train, tune: bool) -> GradientBoostingClassifier:
    base = GradientBoostingClassifier(random_state=config.RANDOM_SEED)
    if not tune:
        return base.fit(X_train, y_train)

    search = GridSearchCV(
        base, config.GB_PARAM_GRID, cv=config.CV_FOLDS,
        scoring="accuracy", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    LOG.info("GradientBoosting best params: %s (CV acc=%.3f)",
             search.best_params_, search.best_score_)
    return search.best_estimator_


_CORE_FITTERS = {
    "decision_tree": _fit_decision_tree,
    "random_forest": _fit_random_forest,
    "gradient_boosting": _fit_gradient_boosting,
}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    feature_names: list[str],
    tune: bool = True,
    include_boosting: bool = True,
    persist: bool = True,
) -> dict[str, Any]:
    """Train every available model for ``task`` and evaluate on a test split.

    Parameters
    ----------
    task  Short identifier, e.g. "cbcb_s" or "cbcb_r" — used in model filenames.
    tune  Run GridSearchCV for the core models.
    include_boosting  Also train installed boosting libraries.
    persist  Save fitted models + metrics.json to disk.

    Returns
    -------
    dict with keys:
        results       {model_key: metrics_dict}
        models        {model_key: fitted_estimator}
        feature_names list[str]
        best          {metric: model_key}
        task          str
        n_classes     int
    """
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED, stratify=stratify,
    )
    n_classes = int(y.nunique())
    LOG.info("[%s] train=%d test=%d classes=%d", task, len(X_train), len(X_test), n_classes)

    models: dict[str, Any] = {}
    results: dict[str, Any] = {}

    # Core models
    for key, fitter in _CORE_FITTERS.items():
        LOG.info("[%s] training %s ...", task, config.MODEL_NAMES[key])
        model = fitter(X_train, y_train, tune)
        models[key] = model
        results[key] = evaluate.evaluate_model(model, X_test, y_test, name=key)
        if persist:
            save_model(model, model_path(task, key))

    # Optional boosting models
    if include_boosting:
        for key, factory in _optional_boosters().items():
            LOG.info("[%s] training %s ...", task, config.MODEL_NAMES[key])
            model = factory(n_classes)
            model.fit(X_train, y_train)
            models[key] = model
            results[key] = evaluate.evaluate_model(model, X_test, y_test, name=key)
            if persist:
                save_model(model, model_path(task, key))

    best = evaluate.best_models(results)

    bundle = {
        "task": task,
        "results": results,
        "feature_names": feature_names,
        "best": best,
        "n_classes": n_classes,
    }

    if persist:
        # Persist transformers reference and a lightweight metrics summary.
        save_model(
            {"feature_names": feature_names, "best": best, "n_classes": n_classes},
            config.MODELS_DIR / f"{task}__meta.joblib",
        )

    # Keep models out of the JSON-serialisable summary.
    return {**bundle, "models": models}


if __name__ == "__main__":
    from .dataset_generator import generate_dataset
    from .preprocessing import preprocess_pipeline
    from . import cbcb_s, feature_engineering

    raw = generate_dataset(n_rows=8_000, n_users=300)
    processed, _, _ = preprocess_pipeline(raw)
    labelled = cbcb_s.generate_labels(processed)
    X, y, names = feature_engineering.build_feature_matrix(labelled, config.LABEL_COL_S)
    out = train_all(X, y, "cbcb_s", names, tune=False, persist=False)
    for k, v in out["results"].items():
        print(f"{k:18s} acc={v['accuracy']:.3f} f1={v['f1']:.3f}")
    print("Best:", out["best"])

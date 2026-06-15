"""Model evaluation (paper §4.4): metrics, ROC/AUC, confusion matrices.

Computes Accuracy, Precision, Recall, F1 (Eqs. 9–12), per-class precision/
recall, confusion matrices, and one-vs-rest ROC/AUC. Also provides helpers to
build a comparison table and to pick the best model per metric for the
"deployment recommendation" shown in the Streamlit results page.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from . import config
from .utils import LOG


def evaluate_model(model: Any, X_test, y_test, name: str = "") -> dict[str, Any]:
    """Return a metrics dict for a fitted classifier on the test set.

    Macro-averaged precision/recall/F1 keep binary (CBCB-S) and multi-class
    (CBCB-R) tasks comparable. Per-class precision/recall and the confusion
    matrix are included for the detailed views.
    """
    y_pred = model.predict(X_test)
    classes = sorted(np.unique(y_test).tolist())

    metrics: dict[str, Any] = {
        "name": config.MODEL_NAMES.get(name, name),
        "model_key": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classes": classes,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
        "precision_per_class": precision_score(
            y_test, y_pred, average=None, labels=classes, zero_division=0
        ).tolist(),
        "recall_per_class": recall_score(
            y_test, y_pred, average=None, labels=classes, zero_division=0
        ).tolist(),
    }

    # ROC / AUC (one-vs-rest for multi-class) when probabilities are available.
    metrics.update(_roc_auc(model, X_test, y_test, classes))

    # Feature importances when the estimator exposes them.
    if hasattr(model, "feature_importances_"):
        metrics["feature_importances"] = np.asarray(model.feature_importances_).tolist()

    LOG.info(
        "%-18s acc=%.3f prec=%.3f rec=%.3f f1=%.3f",
        name, metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
    )
    return metrics


def _roc_auc(model, X_test, y_test, classes) -> dict[str, Any]:
    """Compute ROC curves and AUC; returns {} if the model has no proba."""
    if not hasattr(model, "predict_proba"):
        return {}

    try:
        proba = model.predict_proba(X_test)
    except Exception:  # pragma: no cover - defensive
        return {}

    out: dict[str, Any] = {"roc": {}}
    y_arr = np.asarray(y_test)

    if len(classes) == 2:
        pos = proba[:, 1]
        fpr, tpr, _ = roc_curve(y_arr, pos, pos_label=classes[1])
        try:
            auc = float(roc_auc_score(y_arr, pos))
        except ValueError:
            auc = float("nan")
        out["roc"][str(classes[1])] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        out["auc"] = auc
        out["auc_macro"] = auc
    else:
        y_bin = label_binarize(y_arr, classes=classes)
        aucs = []
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            out["roc"][str(cls)] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            try:
                aucs.append(roc_auc_score(y_bin[:, i], proba[:, i]))
            except ValueError:
                aucs.append(float("nan"))
        out["auc_per_class"] = [float(a) for a in aucs]
        out["auc_macro"] = float(np.nanmean(aucs)) if aucs else float("nan")
        out["auc"] = out["auc_macro"]

    return out


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Tidy DataFrame of the four headline metrics for every model."""
    rows = []
    for key, m in results.items():
        rows.append(
            {
                "Model": m.get("name", key),
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1"],
                "AUC": m.get("auc_macro", float("nan")),
            }
        )
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    return df.reset_index(drop=True)


def best_models(results: dict[str, dict]) -> dict[str, str]:
    """Return {metric: model_key} of the top scorer for each headline metric."""
    best: dict[str, str] = {}
    for metric in config.METRIC_KEYS:
        best[metric] = max(results, key=lambda k: results[k].get(metric, 0.0))
    return best


def recommend_model(results: dict[str, dict]) -> str:
    """Recommend a deployment model: highest mean of the four headline metrics."""
    def score(key: str) -> float:
        m = results[key]
        return float(np.mean([m[k] for k in config.METRIC_KEYS]))

    return max(results, key=score)


if __name__ == "__main__":
    print("evaluate.py — import module; run train.py for an end-to-end demo.")

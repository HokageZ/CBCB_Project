"""Utility helpers: seeding, logging, and model/metrics persistence."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "cbcb") -> logging.Logger:
    """Return a module-level logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOG = get_logger()


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seed Python and NumPy RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_model(obj: Any, path: str | Path) -> Path:
    """Persist any picklable object (model, encoder, scaler) via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    LOG.info("Saved %s", path.name)
    return path


def load_model(path: str | Path) -> Any:
    """Load a joblib-persisted object. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def save_json(data: Any, path: str | Path) -> Path:
    """Write a JSON-serialisable object with numpy types coerced to native."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    LOG.info("Saved %s", path.name)
    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON file, or return None if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(value: Any) -> Any:
    """Coerce numpy scalars/arrays so json.dump can serialise them."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serialisable")


def model_path(task: str, model_key: str) -> Path:
    """Standard on-disk path for a trained model, e.g. cbcb_s__decision_tree.joblib."""
    return config.MODELS_DIR / f"{task}__{model_key}.joblib"

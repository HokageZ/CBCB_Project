"""Central configuration: constants, paths, and hyperparameter grids.

All other modules import from here so that the genre vocabulary, column
names, random seed, and tuning grids stay consistent across the dataset
generator, preprocessing, labelling, training, and the Streamlit app.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the project root, i.e. the parent of src/)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "cbcb_synthetic.csv"
PROCESSED_DATA_PATH = DATA_DIR / "cbcb_processed.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"

for _d in (DATA_DIR, MODELS_DIR, ASSETS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset schema (matches the STC/JAWWY columns described in the paper, §4.1)
# --------------------------------------------------------------------------- #
COLUMNS = [
    "User_ID",
    "Date",
    "Program_Name",
    "Program_Genre",
    "Program_Class",
    "Watch_Duration",
    "Season",
    "Episode",
]

# 16 genres — the paper reports 14–16 genres across its three dataset splits.
GENRES = [
    "Action",
    "Comedy",
    "Horror",
    "Drama",
    "Animation",
    "Documentary",
    "Thriller",
    "Romance",
    "SciFi",
    "Fantasy",
    "Crime",
    "Adventure",
    "Mystery",
    "Family",
    "Sport",
    "Music",
]

PROGRAM_CLASSES = ["Movie", "Series"]

# Per-genre watch-duration profile (seconds) for a log-normal draw:
# (median_seconds, sigma). Movies run longer than episodic series content.
GENRE_DURATION_PROFILE = {
    "Action": (5400, 0.45),
    "Comedy": (3600, 0.50),
    "Horror": (5100, 0.45),
    "Drama": (6000, 0.40),
    "Animation": (3000, 0.55),
    "Documentary": (3300, 0.50),
    "Thriller": (5700, 0.42),
    "Romance": (5400, 0.45),
    "SciFi": (6300, 0.42),
    "Fantasy": (6600, 0.45),
    "Crime": (3000, 0.50),
    "Adventure": (5700, 0.45),
    "Mystery": (3300, 0.50),
    "Family": (4800, 0.48),
    "Sport": (4200, 0.55),
    "Music": (1500, 0.60),
}

# Categorical columns that get one-hot encoded (paper §4.1.2)
CATEGORICAL_FEATURES = ["Program_Genre", "Program_Class"]

# Numeric column that gets Min-Max scaled (paper §4.1.3)
SCALE_FEATURE = "Watch_Duration"

# --------------------------------------------------------------------------- #
# Synthetic-generation parameters
# --------------------------------------------------------------------------- #
DEFAULT_N_ROWS = 50_000
DEFAULT_N_USERS = 2_000

# Probability a user repeats their current genre on the next interaction
# (drives CBCB-S positives). Probability of a revert (G1->G2->G1) is derived
# from the user's "stickiness" too — see dataset_generator.
GENRE_REPEAT_PROB = 0.45
GENRE_REVERT_PROB = 0.25

# Fraction of rows that receive an injected extreme outlier watch-time
# (e.g. ~80,000 s, per paper §4.1.1) so IQR cleaning is demonstrable.
OUTLIER_FRACTION = 0.01
OUTLIER_DURATION_RANGE = (60_000, 90_000)

# --------------------------------------------------------------------------- #
# CBCB labelling
# --------------------------------------------------------------------------- #
LABEL_COL_S = "cbcb_s_label"  # binary  {0, 1}
LABEL_COL_R = "cbcb_r_label"  # ternary {0, 1, 2}

CBCB_S_CLASSES = [0, 1]
CBCB_R_CLASSES = [0, 1, 2]

CBCB_S_CLASS_NAMES = {0: "No Captivation", 1: "Sequential Captivation"}
CBCB_R_CLASS_NAMES = {
    0: "No Captivation",
    1: "Sequential Repeat",
    2: "Revert Captivation",
}

# --------------------------------------------------------------------------- #
# Train / test split (paper §6.3 — 70 % train, 30 % test)
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.30
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Hyperparameter grids for GridSearchCV.
# The Decision Tree grid mirrors the tuned values reported in Table 8
# (max_depth, max_leaf_nodes, criterion, splitter).
# --------------------------------------------------------------------------- #
DT_PARAM_GRID = {
    "max_depth": [5, 7, 8, 10, 11, None],
    "max_leaf_nodes": [32, 64, 128, 254, 582, None],
    "criterion": ["gini", "entropy"],
    "splitter": ["best"],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [8, 12, None],
    "criterion": ["gini", "entropy"],
}

GB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 8],
    "learning_rate": [0.05, 0.1],
}

# Model registry key -> human-readable name
MODEL_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

# Metrics tracked everywhere (keep order stable for tables/charts)
METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]

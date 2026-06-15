"""CBCB — Content-Based Captivation Behavior.

Reproduction of the framework from:
    Anwer, Z., Qureshi, S., Iqbal, S.M.Z., Zia, A., Anwer, S. (2025).
    "Predicting user behavior on video streaming by using watch-time
     duration analysis." Knowledge-Based Systems 332, 114779.

Package modules
---------------
config              Central constants, paths, hyperparameter grids.
dataset_generator   Synthetic STC/JAWWY-like dataset generator.
preprocessing       Cleaning, IQR outlier removal, one-hot, min-max scaling.
cbcb_s              User Sequential Captivation Behavior (Algorithm 2).
cbcb_r              User Revert Captivation Behavior (Algorithm 3).
feature_engineering Build the model feature matrix from labelled data.
train               Train Decision Tree / Random Forest / Gradient Boosting.
evaluate            Metrics, ROC/AUC, confusion matrices, model selection.
visualization       Matplotlib / Seaborn / Plotly chart builders.
deep_learning       Optional LSTM / GRU / Transformer extension (torch).
utils               IO, seeding, model persistence helpers.
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "dataset_generator",
    "preprocessing",
    "cbcb_s",
    "cbcb_r",
    "feature_engineering",
    "train",
    "evaluate",
    "visualization",
    "deep_learning",
    "utils",
]

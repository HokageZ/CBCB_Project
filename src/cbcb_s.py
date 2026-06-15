"""CBCB-S — User Sequential Captivation Behavior (Algorithm 2, paper §5.2.1).

Definition
----------
For each user, interactions are ordered by Date. Row *i* is labelled by
comparing its genre to the *next* interaction's genre:

    y_i = 1   if  genre[i] == genre[i+1]      (same genre consecutively)
    y_i = 0   otherwise

This is the binary "sequential captivation" signal: a user repeating the same
genre on the very next interaction (G1 -> G1) is considered strongly engaged.
The model is trained against these labels with cross-entropy loss (Eq. 16).

The last interaction of each user has no successor, so it is dropped.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .utils import LOG


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Append the CBCB-S binary label column (``config.LABEL_COL_S``).

    Parameters
    ----------
    df  Raw or cleaned dataset containing at least User_ID, Date,
        Program_Genre.

    Returns
    -------
    A copy sorted by (User_ID, Date) with the new label column. Rows that
    have no successor within the same user (the tail row per user) are
    dropped because their label is undefined.
    """
    df = df.sort_values(["User_ID", "Date"]).reset_index(drop=True)

    # next genre within the same user
    next_genre = df.groupby("User_ID")["Program_Genre"].shift(-1)
    # Compare current vs. next genre; mask tail rows (no successor) as NA so
    # they are dropped rather than mislabelled as 0.
    label = (df["Program_Genre"] == next_genre).where(next_genre.notna())

    df[config.LABEL_COL_S] = label
    # Drop tail rows (no successor -> label is NA)
    df = df.dropna(subset=[config.LABEL_COL_S]).reset_index(drop=True)
    df[config.LABEL_COL_S] = df[config.LABEL_COL_S].astype(int)

    dist = df[config.LABEL_COL_S].value_counts().to_dict()
    LOG.info("CBCB-S labels generated: %s", dist)
    return df


def _self_test() -> None:
    """Validate labelling on a hand-built sequence: G1 G1 G2 G1.

    Expected CBCB-S labels (compare each row to the next):
        G1->G1 : 1
        G1->G2 : 0
        G2->G1 : 0
        G1     : dropped (tail)
    """
    seq = pd.DataFrame(
        {
            "User_ID": [1, 1, 1, 1],
            "Date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
            "Program_Genre": ["Action", "Action", "Comedy", "Action"],
        }
    )
    out = generate_labels(seq)
    labels = out[config.LABEL_COL_S].tolist()
    assert labels == [1, 0, 0], f"CBCB-S self-test failed: {labels}"
    print("CBCB-S self-test passed:", labels)


if __name__ == "__main__":
    _self_test()

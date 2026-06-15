"""CBCB-R — User Revert Captivation Behavior (Algorithm 3, paper §5.2.2).

Definition
----------
For each user, interactions are ordered by Date. Row *i* is given a ternary
label by looking one and two steps ahead:

    y_i = 1   if genre[i] == genre[i+1]                       (immediate repeat: G1 -> G1)
    y_i = 2   if genre[i] != genre[i+1] and genre[i] == genre[i+2]
                                                              (revert: G1 -> G2 -> G1)
    y_i = 0   otherwise                                       (no captivation)

Interpretation (paper §5.2.2):
  * label 1 — "strongly true": the user repeats the same activity at the next level.
  * label 2 — "true (revert)": the user diverges then returns to the prior genre.
  * label 0 — "weak": neither pattern holds.

The last two interactions of each user lack a full two-step lookahead and are
dropped. The model is trained against these labels with the multi-class loss
(Eq. 19).
"""
from __future__ import annotations

import pandas as pd

from . import config
from .utils import LOG


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Append the CBCB-R ternary label column (``config.LABEL_COL_R``).

    Parameters
    ----------
    df  Raw or cleaned dataset containing at least User_ID, Date,
        Program_Genre.

    Returns
    -------
    A copy sorted by (User_ID, Date) with the new label column. The final two
    rows per user (insufficient lookahead) are dropped.
    """
    df = df.sort_values(["User_ID", "Date"]).reset_index(drop=True)

    grp = df.groupby("User_ID")["Program_Genre"]
    next1 = grp.shift(-1)  # genre[i+1]
    next2 = grp.shift(-2)  # genre[i+2]

    cur = df["Program_Genre"]
    repeat = cur == next1                       # G1 -> G1
    revert = (cur != next1) & (cur == next2)    # G1 -> G2 -> G1

    # Default 0; revert (2) takes priority is irrelevant since repeat/revert
    # are mutually exclusive by construction, but assign explicitly.
    label = pd.Series(0, index=df.index, dtype="float")
    label[repeat] = 1.0
    label[revert] = 2.0

    # Rows whose two-step lookahead is missing must be dropped (label undefined).
    valid = next2.notna()
    df[config.LABEL_COL_R] = label
    df = df[valid].reset_index(drop=True)
    df[config.LABEL_COL_R] = df[config.LABEL_COL_R].astype(int)

    dist = df[config.LABEL_COL_R].value_counts().to_dict()
    LOG.info("CBCB-R labels generated: %s", dist)
    return df


def _self_test() -> None:
    """Validate labelling on a hand-built sequence: G1 G2 G1 G1 G3.

    Indices 0..4, comparing to +1 and +2 (last two rows dropped):
        i=0 G1: next1=G2 (!=), next2=G1 (==)  -> 2 (revert)
        i=1 G2: next1=G1 (!=), next2=G1 (!=)  -> 0
        i=2 G1: next1=G1 (==)                 -> 1 (repeat)
        i=3 G1: no next2                      -> dropped
        i=4 G3: no next1/next2                -> dropped
    """
    seq = pd.DataFrame(
        {
            "User_ID": [1, 1, 1, 1, 1],
            "Date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03",
                 "2023-01-04", "2023-01-05"]
            ),
            "Program_Genre": ["Action", "Comedy", "Action", "Action", "Horror"],
        }
    )
    out = generate_labels(seq)
    labels = out[config.LABEL_COL_R].tolist()
    assert labels == [2, 0, 1], f"CBCB-R self-test failed: {labels}"
    print("CBCB-R self-test passed:", labels)


if __name__ == "__main__":
    _self_test()

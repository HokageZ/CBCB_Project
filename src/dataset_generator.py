"""Synthetic STC/JAWWY-like dataset generator.

The original Saudi Telecom Company (STC) / JAWWY dataset used in the paper
(>3.5M rows) is not publicly downloadable, so this module produces a
realistic stand-in with the same schema (paper §4.1):

    User_ID, Date, Program_Name, Program_Genre, Program_Class,
    Watch_Duration, Season, Episode

Why "realistic" matters
-----------------------
The CBCB framework only has signal to learn if genuine *sequential* and
*revert* viewing patterns exist. A naive uniform-random generator would
produce near-random labels and meaningless accuracy. We therefore give each
user:

  * a personal *favourite* genre and a *stickiness* level, and
  * a per-step Markov-like choice: repeat the current genre (CBCB-S
    positive), revert to the genre two steps back (CBCB-R positive), or
    explore a new genre.

We also inject a small fraction of extreme outlier watch-times (~80,000 s,
per paper §4.1.1) so the IQR cleaning step has something to remove.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import LOG, set_seed


# Pools of program-name fragments per class, combined with the genre to form
# plausible titles such as "Action Chronicles" or "The Comedy Files".
_TITLE_PREFIX = ["The", "Great", "Dark", "Hidden", "Last", "Eternal", "Silent",
                 "Rising", "Lost", "Golden", "Crimson", "Midnight"]
_TITLE_SUFFIX = ["Chronicles", "Files", "Legacy", "Story", "Saga", "Journey",
                 "Secret", "Empire", "Horizon", "Order", "Dawn", "Realm"]


def _build_transition_matrix(rng: np.random.Generator) -> np.ndarray:
    """Per-user *explore* transition matrix with a zero diagonal.

    Immediate repeats (the CBCB-S/-R signal) are handled explicitly by the
    engagement-driven ``repeat_prob`` in the generation loop, so this matrix
    governs only the "explore a different genre" branch — its diagonal is zero
    to avoid double-counting repeats. Each user gets a skewed preference over
    other genres so some transitions are favoured (enabling reverts).
    """
    n = len(config.GENRES)
    base = rng.uniform(0.2, 1.0, size=(n, n))
    np.fill_diagonal(base, 0.0)
    base /= base.sum(axis=1, keepdims=True)  # off-diagonal rows sum to 1
    return base


def _make_title(rng: np.random.Generator, genre: str) -> str:
    return f"{rng.choice(_TITLE_PREFIX)} {genre} {rng.choice(_TITLE_SUFFIX)}"


def generate_dataset(
    n_rows: int = config.DEFAULT_N_ROWS,
    n_users: int = config.DEFAULT_N_USERS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic viewing-history dataset.

    Parameters
    ----------
    n_rows   Target number of interaction rows (>= 50,000 recommended).
    n_users  Number of distinct users to spread the rows across.
    seed     RNG seed for reproducibility.

    Returns
    -------
    DataFrame with the columns listed in ``config.COLUMNS``, sorted by
    (User_ID, Date) so that downstream sequential labelling is well-defined.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    LOG.info("Generating %d rows across %d users...", n_rows, n_users)

    genres = np.array(config.GENRES)
    n_genres = len(genres)

    # Distribute interactions across users with a power-law-ish profile:
    # a few heavy viewers, a long tail of light ones.
    weights = rng.pareto(2.0, size=n_users) + 1.0
    weights /= weights.sum()
    rows_per_user = np.maximum(1, (weights * n_rows).astype(int))
    # Adjust to hit n_rows exactly.
    diff = n_rows - rows_per_user.sum()
    rows_per_user[0] += diff

    records: list[dict] = []
    base_date = np.datetime64("2023-01-01")

    for user_id in range(1, n_users + 1):
        k = int(rows_per_user[user_id - 1])
        if k <= 0:
            continue

        transition = _build_transition_matrix(rng)
        # Per-user engagement threshold: when this watch's engagement exceeds
        # theta, the user repeats the genre next. Centring theta near 0.5 (with
        # symmetric engagement ~Beta(2,2)) keeps the repeat/non-repeat classes
        # close to balanced, while making watch-time a *strong, learnable*
        # predictor of repeats (the paper's central hypothesis).
        theta = rng.uniform(0.45, 0.55)
        revert_tendency = rng.uniform(0.30, 0.50)
        current = int(rng.integers(n_genres))  # first genre
        genre_history: list[int] = []

        # Each user's interactions land on consecutive days from a random start.
        start_offset = int(rng.integers(0, 300))
        for step in range(k):
            genre = genres[current]
            genre_history.append(current)

            # --- Engagement drives BOTH watch-time and the next-step decision.
            # This embeds the paper's hypothesis: the longer a user watches an
            # item (higher engagement), the more likely they repeat the genre.
            engagement = float(rng.beta(2.0, 2.0))  # ~U-shaped around 0.5
            median, sigma = config.GENRE_DURATION_PROFILE[genre]
            # Map engagement to a 0.4x-1.6x multiplier on the genre median,
            # with mild log-normal noise so durations stay realistic.
            duration = median * (0.4 + 1.2 * engagement) * float(
                rng.lognormal(mean=0.0, sigma=sigma * 0.35)
            )

            program_class = "Series" if rng.random() < 0.55 else "Movie"
            if program_class == "Series":
                season = int(rng.integers(1, 9))
                episode = int(rng.integers(1, 25))
            else:
                season = 0
                episode = 0

            date = base_date + np.timedelta64(start_offset + step, "D")

            records.append(
                {
                    "User_ID": user_id,
                    "Date": date,
                    "Program_Name": _make_title(rng, genre),
                    "Program_Genre": genre,
                    "Program_Class": program_class,
                    "Watch_Duration": round(duration, 1),
                    "Season": season,
                    "Episode": episode,
                }
            )

            # --- Decide the NEXT genre using this row's engagement.
            # A high-engagement watch (engagement > theta) leads to an immediate
            # repeat (G1 -> G1); a ~12% noise flip keeps the relationship
            # imperfect/realistic so the task is non-trivial. Non-repeats either
            # revert to two steps ago (G1 -> G2 -> G1) or explore a new genre.
            repeats = engagement > theta
            if rng.random() < 0.12:           # label noise -> imperfect signal
                repeats = not repeats

            if repeats:
                pass  # repeat: current genre unchanged
            elif len(genre_history) >= 2 and rng.random() < revert_tendency:
                current = genre_history[-2]  # revert to prior genre
            else:
                current = int(rng.choice(n_genres, p=transition[current]))

    df = pd.DataFrame.from_records(records, columns=config.COLUMNS)

    # Inject extreme outlier watch-times (paper §4.1.1) so IQR cleaning has
    # an obvious effect during preprocessing/visualisation.
    n_outliers = int(len(df) * config.OUTLIER_FRACTION)
    if n_outliers > 0:
        idx = rng.choice(len(df), size=n_outliers, replace=False)
        lo, hi = config.OUTLIER_DURATION_RANGE
        df.loc[df.index[idx], "Watch_Duration"] = rng.uniform(lo, hi, size=n_outliers).round(1)
        LOG.info("Injected %d outlier watch-times in [%d, %d] s", n_outliers, lo, hi)

    df = df.sort_values(["User_ID", "Date"]).reset_index(drop=True)
    LOG.info("Generated dataset: %d rows x %d cols", df.shape[0], df.shape[1])
    return df


def generate_and_save(
    path=None,
    n_rows: int = config.DEFAULT_N_ROWS,
    n_users: int = config.DEFAULT_N_USERS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a dataset and write it to CSV (defaults to config.RAW_DATA_PATH)."""
    path = path or config.RAW_DATA_PATH
    df = generate_dataset(n_rows=n_rows, n_users=n_users, seed=seed)
    df.to_csv(path, index=False)
    LOG.info("Wrote raw dataset -> %s", path)
    return df


if __name__ == "__main__":
    # Quick smoke test when run directly.
    demo = generate_dataset(n_rows=5_000, n_users=200)
    print(demo.head(10).to_string(index=False))
    print("\nShape:", demo.shape)
    print("Genres:", demo["Program_Genre"].nunique())
    print("Users:", demo["User_ID"].nunique())
    print("Duration describe:\n", demo["Watch_Duration"].describe())

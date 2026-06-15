# Label Design Rationale

> **Audience:** Team members who need to understand why labels are constructed the way they are.
>
> **Goal:** Explain the design reasoning behind CBCB-S and CBCB-R labels, with edge cases and validation.

---

## Why These Labels?

The paper defines two specific behaviours — sequential repeat and revert — that correspond to distinct user engagement states:

| User State | Behaviour | CBCB Label |
|---|---|---|
| **Binge-watching** | G1 → G1 | CBCB-S = 1, CBCB-R = 1 |
| **Exploring** | G1 → G2 | CBCB-S = 0, CBCB-R = 0 |
| **Exploring but loyal** | G1 → G2 → G1 | CBCB-S = 0 (at G1→G2), CBCB-R = 2 (at G1) |

Each label is designed to be **locally computable** — you only need the next 1–2 interactions, not the full history.

## CBCB-S: Binary Sequential Label

### Algorithm

```python
# src/cbcb_s.py
def generate_labels(df):
    df = df.sort_values(['User_ID', 'Date'])
    labels = []
    for user in df['User_ID'].unique():
        user_df = df[df['User_ID'] == user]
        for i in range(len(user_df) - 1):
            if user_df.iloc[i]['Program_Genre'] == user_df.iloc[i+1]['Program_Genre']:
                labels.append(1)  # Repeat
            else:
                labels.append(0)  # No Repeat
    # Drop last row of each user
    result = df.groupby('User_ID').apply(lambda g: g.iloc[:-1]).reset_index(drop=True)
    result[LABEL_COL_S] = labels
    return result
```

### Why Binary?

- The question is inherently yes/no: "same genre next?"
- Binary classification is simpler to train and evaluate.
- The paper found that binary CBCB-S was sufficient for the sequential captivation signal.
- Multi-class would require distinguishing *which* genre, which is a different (harder) problem.

### Why Drop the Last Row?

The last row has no successor, so no label can be assigned. This is a standard "shifted target" pattern in time-series supervised learning.

### Edge Cases

| Scenario | Label | Rationale |
|---|---|---|
| Single interaction per user | Dropped | No successor to compare |
| Consecutive same genre | 1 | Canonical repeat |
| Consecutive different genre | 0 | No repeat |
| Same genre after gap | 0 | Gap means no captivation continuity |

## CBCB-R: Ternary Revert Label

### Algorithm

```python
# src/cbcb_r.py
def generate_labels(df):
    df = df.sort_values(['User_ID', 'Date'])
    labels = []
    for user in df['User_ID'].unique():
        user_df = df[df['User_ID'] == user]
        for i in range(len(user_df) - 2):
            genre_i   = user_df.iloc[i]['Program_Genre']
            genre_i1  = user_df.iloc[i+1]['Program_Genre']
            genre_i2  = user_df.iloc[i+2]['Program_Genre']

            if genre_i == genre_i1:
                labels.append(1)                    # Immediate Repeat
            elif genre_i == genre_i2:
                labels.append(2)                    # Revert
            else:
                labels.append(0)                    # No Repeat
    # Drop last two rows of each user
    result = df.groupby('User_ID').apply(lambda g: g.iloc[:-2]).reset_index(drop=True)
    result[LABEL_COL_R] = labels
    return result
```

### Why Ternary?

Two thresholds → three states:

1. **Immediate Repeat (1):** Shows strongest captivation — the user didn't even try another genre.
2. **Revert (2):** Shows genre loyalty — the user explored but returned, indicating a strong preference.
3. **No Repeat (0):** The residual — everything else (divergence without return, or continued divergence).

### Why Check Repeat Before Revert?

Priority matters in the ternary structure:

```
Check if genre[i] == genre[i+1]?    → label = 1 (Immediate Repeat)
Else check if genre[i] == genre[i+2]? → label = 2 (Revert)
Else                                  → label = 0 (No Repeat)
```

If `genre[i] == genre[i+1] == genre[i+2]` (three same in a row), the first condition matches and the label is 1 (Immediate Repeat), not 2. This is correct — the immediate repeat is the dominant signal.

### Why Drop the Last Two Rows?

CBCB-R requires lookahead of 2. Row `n-1` can't generate a label (no `n+1`), and row `n` can't either (no `n+1` or `n+2`).

### Edge Cases

| Sequence | Labels | Explanation |
|---|---|---|
| G1, G1, G2 | [1, 0] | Repeat at pos 0, diverged at pos 1 (last two dropped from CBCB-R since insufficient lookahead) |
| G1, G2, G1 | [2] | Revert — diverged then returned |
| G1, G2, G1, G1 | [2, 1] | Revert at pos 0, then immediate repeat at pos 2→3 |
| G1, G2, G3 | [0] | No repeat, no revert — chain of exploration |
| G1, G1, G1 | [1, 1] | Immediate repeat at pos 0 AND pos 1 |

## Validation Through Self-Tests

Each labelling module includes a self-test that runs on direct execution:

```bash
python src/cbcb_s.py
# Validates: G1, G1, G2, G1 → labels [1, 0, 0]

python src/cbcb_r.py
# Validates: G1, G2, G1, G1, G3 → labels [2, 0, 1]
```

## Why the Labels Work With the Synthetic Generator

The data generator creates the exact patterns these labels are designed to detect:

```python
# In dataset_generator.py — genre transitions
if rng.random() < repeat_prob:
    next_genre = current_genre  # Creates repeat patterns
elif rng.random() < revert_prob:
    next_genre = random_genre   # Then will revert back
else:
    next_genre = transition_matrix[current_idx]  # Markov chain
```

This embeds the paper's hypothesis directly into the data: engagement drives both watch duration AND genre transitions. The CBCB labels then capture this relationship as a supervised learning problem.

## Label Distribution Target

The generator is parameterised to produce roughly balanced label distributions:

```
CBCB-S:  ~48% Repeat, ~52% No Repeat
CBCB-R:  ~33% each class (No Repeat, Immediate Repeat, Revert)
```

Balanced classes mean accuracy is a meaningful metric (no class imbalance to correct for).

## Connection to the Feature Set

The label design directly motivates the 6 sequential features:

| Label Involves | Related Features |
|---|---|
| Current genre | `prev_genre_code` |
| Watch duration | `watch_duration_scaled`, `user_avg_duration` |
| Temporal context | `time_since_last_day`, `user_session_index` |
| Repeat streak | `genre_repeat_run` |

The features and labels are designed together as a coherent supervised learning formulation.

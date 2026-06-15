# Model Selection Rationale

> **Audience:** Team members evaluating model choices or considering new ones.
>
> **Goal:** Explain why Decision Tree, Random Forest, and Gradient Boosting are the core models, and when to use the optional boosters or deep learning.

---

## Core Models

### Decision Tree — The Baseline

**Why:** The paper used Decision Tree as its main model. Decision trees are:

- **Interpretable:** Feature importance, decision paths, and tree structure are directly visible.
- **Non-parametric:** No assumptions about data distribution — handles the log-normal watch durations naturally.
- **Fast to train:** Even with GridSearch, a DT completes in seconds on 35k rows.

**When it works best:** When the relationship between features and labels is simple and monotonic. The synthetic data has relatively clean patterns, so DT achieves ~0.78 accuracy on CBCB-S.

**Limitations:** Prone to overfitting (mitigated by GridSearch on `max_depth` and `min_samples_leaf`). Lower ceiling than ensemble methods.

```python
# src/train.py — Decision Tree
dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
if tune:
    gs = GridSearchCV(dt, DT_PARAM_GRID, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_
```

### Random Forest — The Workhorse

**Why:** Ensemble of 100–300 trees with bagging. Key advantages:

- **Reduced variance:** Each tree sees a bootstrap sample; predictions are averaged → much lower overfitting.
- **Robust to noise:** The injected 1% outliers have minimal impact on ensemble predictions.
- **Best performer:** Consistently scores highest on both CBCB-S (~0.81) and CBCB-R (~0.85).
- **Feature importance:** Built-in permutation importance via `feature_importances_`.

**Limitations:** Less interpretable than a single tree. Larger model size (300 trees × depth ~20).

```python
rf = RandomForestClassifier(random_state=RANDOM_SEED)
if tune:
    gs = GridSearchCV(rf, RF_PARAM_GRID, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
```

### Gradient Boosting — The Contender

**Why:** Sequential ensemble that corrects previous errors.

- **Sequential training:** Each new tree focuses on residuals from the previous ensemble.
- **Learning rate:** `learning_rate` controls how aggressively each tree corrects → can beat RF with careful tuning.
- **HistGradientBoosting:** Uses binning for faster training on large datasets.

**Performance:** Slightly behind RF (~0.79 CBCB-S, ~0.83 CBCB-R) on the synthetic dataset, but may outperform RF on real data with more complex patterns.

```python
gb = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
if tune:
    gs = GridSearchCV(gb, GB_PARAM_GRID, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
```

## Why These Three?

| Requirement | DT | RF | GB |
|---|---|---|---|
| Handles mixed feature types (numeric + one-hot) | ✓ | ✓ | ✓ |
| Scale-invariant | ✓ | ✓ | ✓ |
| Captures non-linear interactions | ✓ | ✓ | ✓ |
| Fast training | ✓ | ✓ | ~ (slower) |
| Interpretable | ✓ | ~ | ~ |
| Robust to outliers | ~ (prone to overfit) | ✓ | ✓ |
| Best accuracy potential | ✗ | ✓ | ✓ |

Together they form a **baseline (DT) + robust workhorse (RF) + potential ceiling (GB)** trio that covers the model space without over-engineering.

---

## Optional Boosters

XGBoost, LightGBM, and CatBoost are included as optional upgrades. They are **import-guarded** — the pipeline works without them.

### Performance Notes

| Booster | CBCB-S (synthetic) | CBCB-R (synthetic) | Speed |
|---|---|---|---|
| XGBoost | ~0.80 | ~0.84 | Fast |
| LightGBM | ~0.80 | ~0.84 | Fastest |
| CatBoost | ~0.79 | ~0.83 | Slower (needs tuning) |

All three are close to RF performance. The main reason to use them is **production deployment** (LightGBM is particularly well-optimised for serving).

### Import Guard Implementation

```python
# src/train.py
def _optional_boosters():
    boosters = {}
    try:
        import xgboost as xgb
        boosters['xgboost'] = xgb.XGBClassifier
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        boosters['lightgbm'] = lgb.LGBMClassifier
    except ImportError:
        pass
    try:
        import catboost as cb
        boosters['catboost'] = cb.CatBoostClassifier
    except ImportError:
        pass
    return boosters
```

---

## Deep Learning Extension

The `src/deep_learning.py` module takes a different approach: rather than using the engineered feature matrix, it treats genre prediction as a **sequence-to-class** problem.

### Architecture Comparison

| Architecture | Parameters | Strengths | Weaknesses |
|---|---|---|---|
| **LSTM** | Embedding(16, 32) → LSTM(32, 64) → Linear(64, 16) | Proven for sequence modelling | Slower training, prone to overfitting on short sequences |
| **GRU** | Same structure as LSTM with GRU cells | Faster than LSTM, similar quality | Slightly less expressive |
| **Transformer** | Embedding(16, 32) + PosEnc → TransformerEncoder ×2 → Linear | Captures long-range dependencies | Needs more data, more parameters |

### Why Deep Learning Might Not Outperform Tree Models

1. **Small vocabulary:** Only 16 genres — a shallow LSTM doesn't have much sequential structure to learn beyond what Markov features already capture.
2. **Short sequences:** Average user has ~25 interactions. The deep learning advantage appears with much longer sequences.
3. **Feature engineering already captures temporal structure:** The 6 sequential features + one-hots encode the same information the DL models would learn from raw sequences.

### When to Use Deep Learning

- **Larger datasets:** 500k+ rows where DL can learn genuine embedding structure.
- **Cross-session patterns:** DL captures long-range dependencies that engineered features miss.
- **Direct next-genre prediction:** The DL models predict *which* genre (16-class), not just the CBCB binary/ternary label.

---

## Model Recommendation System

The `recommend_model()` function selects the best overall model by averaging the four headline metrics:

```python
def recommend_model(results):
    scores = {}
    for name, metrics in results['metrics'].items():
        scores[name] = np.mean([
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ])
    return max(scores, key=scores.get)
```

On the synthetic benchmark, this consistently recommends **Random Forest** for both tasks.

## Feature Importance Observations

Across all models, the most important features are:

| Rank | Feature | Importance | Why |
|---|---|---|---|
| 1 | `watch_duration_scaled` | ~0.30–0.40 | Core hypothesis: engagement drives behaviour |
| 2 | `genre_repeat_run` | ~0.15–0.25 | Being in a binge state strongly predicts repeat |
| 3 | `user_avg_duration` | ~0.10–0.15 | User baseline engagement matters |

This confirms the paper's central claim: **watch duration is the single most predictive feature** for short-term user behaviour.

---

## Hyperparameter Tuning Strategy

GridSearchCV with 5-fold cross-validation is the default. Parameters:

```python
CV_FOLDS = 5
TEST_SIZE = 0.30
```

**Why GridSearch over RandomSearch?** The parameter spaces are small enough (4 params × 2–5 values = 8–25 combinations) that GridSearch is computationally feasible.

**Why 5 folds and 70/30 split?** With ~35k training rows after split, 5-fold CV gives 7k rows per validation fold — reliable enough for stable metrics.

**When to skip tuning:** Use `--no-tune` for quick experiments. Default params from scikit-learn are reasonable baselines.

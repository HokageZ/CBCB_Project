# CBCB Concept — Content-Based Captivation Behavior

> **Audience:** All team members who need to understand the research foundation.
>
> **Goal:** Explain what CBCB is, the problem it solves, the paper it extends, and why this approach works.

---

## The Problem

Video streaming platforms (STC, Netflix, Amazon Prime, YouTube) collect massive amounts of viewing data, but predicting short-term user behaviour — *what will this user watch next?* — remains difficult because:

- **Watch duration** is more informative than binary "watched/didn't watch" signals.
- **Genre transitions** encode user engagement patterns that raw viewing counts miss.
- **Historical context** matters: repeat viewing, genre switching, and reverting to a previous genre all signal different engagement levels.

## What Is CBCB?

**Content-Based Captivation Behavior** is a framework that predicts short-term user behaviour using **watch-time duration** and **historical genre transitions**. It was introduced in:

> *"Predicting User Behavior on Video Streaming by Using Watch-Time Duration Analysis"* — Knowledge-Based Systems, 2025.

The core hypothesis:

> **Users who watch more of a genre are more likely to repeat or return to that genre.**

Engagement (measured as watch duration) directly drives genre choice — not just content similarity, but the *intensity* of the viewing experience.

## Two Behaviours

The CBCB framework defines two specific captivation behaviours:

### CBCB-S: Sequential Captivation (Binary)

> Will the user watch the **same genre** as their current interaction, next?

```
Current             Next
G1 ────────────────► G1?  →  Repeat (label = 1)
                  ► G2?  →  No Repeat (label = 0)
```

**This measures genre persistence** — how locked-in the user is to a content category.

**Example:** If a user watches *The Action Chronicles* (Action) and then immediately watches *Explosive Pursuit* (also Action), CBCB-S = 1 (Repeat).

### CBCB-R: Revert Captivation (Ternary)

> Will the user **return to a previous genre** after diverging?

```
Current   Next    Next+1              
G1 ──────► G2 ───► G1?  →  Revert (label = 2, the diverge-then-return pattern)
         └────────► G1?  →  Immediate Repeat (label = 1)
                  ► G3?  →  No Repeat (label = 0, divergence without return)
```

**This measures genre loyalty** — after exploring elsewhere, does the user come back?

**Example:** A user watches *The Action Chronicles* (Action), then a *Comedy Roast* (Comedy), then *Explosive Pursuit* (Action). CBCB-R = 2 (Revert).

## Why Two Behaviours?

| Behaviour | Question It Answers | Business Value |
|---|---|---|
| CBCB-S | "Is the user in a genre binge?" | Recommend same-genre content immediately |
| CBCB-R | "Does the user always come back to their favourite genre?" | Identify genre loyalty, personalise recommendations |

Together they form a richer picture of user engagement than either alone.

## How Engagement Drives Behaviour

The paper's key insight: **watch duration is a proxy for captivation**. The longer you watch, the more captivated you are, and the more likely you are to repeat or revert.

```
Low engagement  →  lower repeat/revert probability (browsing, disengaged)
High engagement →  higher repeat/revert probability (captivated, loyal)
```

In our implementation, this is embedded at the synthetic data generation level:

```python
# src/dataset_generator.py — engagement modulates watch duration
engagement = rng.beta(a=theta_effective, b=1.0)  # per-user engagement
duration = rng.lognormal(mean=log_mean, sigma=sigma)
duration *= engagement  # modulated by engagement
```

And the feature matrix captures this directly with `watch_duration_scaled` as the primary feature.

## Relationship to the Original Paper

This project extends the original paper in several ways:

| Aspect | Original Paper | This Implementation |
|---|---|---|
| Models | Decision Tree (main), k-NN, SVM | DT, RF, GB + optional XGBoost/LightGBM/CatBoost + DL |
| Hyperparameter Tuning | Manual grid search | GridSearchCV with 5-fold CV |
| Dataset | Real STC dataset | Synthetic generator with configurable parameters |
| Evaluation | Accuracy, Precision, Recall, F1 | Same + ROC/AUC + confusion matrices + charts |
| Deep Learning | Not explored | LSTM, GRU, Transformer modules |
| Visualisation | Basic | Matplotlib + Seaborn + Plotly interactive |
| Interface | MATLAB scripts | Python CLI + Streamlit app + Jupyter notebook |

## Why Machine Learning?

The relationship between watch duration and next-genre choice is **non-linear and context-dependent**:

- A long watch on *Series* vs *Movie* has different implications.
- A repeat after 2 days means something different than a repeat after 2 minutes.
- User-level baselines differ (some users always repeat, some always explore).

Decision trees capture these non-linear interactions naturally. The ensemble methods (RF, GB) add robustness.

## Validation Strategy

We validate CBCB using four metrics:

| Metric | What It Measures |
|---|---|
| **Accuracy** | Overall correctness across all classes |
| **Precision** | When the model predicts Repeat/Revert, how often is it right? |
| **Recall** | What fraction of actual Repeat/Revert does the model catch? |
| **F1 Score** | Harmonic mean of precision and recall (balanced measure) |

For CBCB-S (balanced binary), accuracy and F1 are closely aligned. For CBCB-R (balanced ternary), all metrics matter equally.

## Caveats & Limitations

1. **Synthetic data** — Results may not generalise to real STC data without retraining.
2. **No user demographics** — Age, gender, location are not used; watch duration is the sole "content-based" signal.
3. **Short-term only** — Predicts 1–2 steps ahead, not long-term churn or lifetime value.
4. **Genre-level only** — Does not predict specific programs, only genre categories.
5. **Markov assumption** — The synthetic generator uses Markov transitions; real behaviour may have longer-range dependencies.

## References

1. Predicting User Behavior on Video Streaming by Using Watch-Time Duration Analysis. *Knowledge-Based Systems*, 2025.
2. Amershi, S., et al. (2019). Software Engineering for Machine Learning: A Case Study. *ICSE-SEIP*.
3. Breiman, L. (2001). Random Forests. *Machine Learning*.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.

# How to Run the Full Pipeline

> **Audience:** Team members who want to run experiments with different configurations.
>
> **Goal:** Run the end-to-end pipeline with custom parameters, skip GridSearch for speed, or include booster models.

---

## Basic Usage

```bash
python main.py --all
```

Equivalent to: generate data → preprocess → label both tasks → train both → evaluate → save figures.

## CLI Reference

```
python main.py [options]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--all` | flag | — | Run full pipeline (generate + train both tasks) |
| `--generate` | flag | — | Generate dataset only |
| `--train` | flag | — | Train using existing preprocessed data |
| `--task` | str | `both` | Task: `cbcb_s`, `cbcb_r`, or `both` |
| `--rows` | int | 50000 | Number of rows to generate |
| `--users` | int | 2000 | Number of users to generate |
| `--no-tune` | flag | — | Skip GridSearchCV hyperparameter tuning |
| `--no-boosting` | flag | — | Skip optional booster models |

## Common Workflows

### Quick experiment (no tuning, one task)

```bash
python main.py --task cbcb_s --no-tune
```

Runs all steps but skips GridSearchCV. Uses default hyperparameters from `scikit-learn`. Completes in ~30 seconds.

### Generate data only

```bash
python main.py --generate --rows 100000 --users 5000
```

Generates a larger dataset (100k rows, 5k users) and saves to `data/cbcb_synthetic.csv`.

### Train with boosters

```bash
python main.py --all --no-boosting
```

If you skip boosters, only Decision Tree, Random Forest, and Gradient Boosting are trained. To include boosters, install them first:

```bash
pip install xgboost lightgbm catboost
python main.py --all
```

### Custom dataset (manual)

1. Place your CSV/Excel file in the `data/` directory.
2. Ensure columns match the expected schema (see [Data Schema Reference](../03-reference/data-schema.md)).
3. Run:
   ```bash
   python main.py --train
   ```
   The pipeline will use whatever CSV exists at `data/cbcb_synthetic.csv`.

## Pipeline Internals

The `_run_task()` function in `main.py` orchestrates one task:

```python
def _run_task(task, raw, tune, boosting):
    logger.info("Starting preprocessing...")
    processed, transformers = preprocess_pipeline(raw, drop_outliers=True)

    if task == "cbcb_s":
        processed = cbcb_s.generate_labels(processed)
        label_col = LABEL_COL_S
    else:
        processed = cbcb_r.generate_labels(processed)
        label_col = LABEL_COL_R

    X, y, feature_names = build_feature_matrix(processed, label_col)
    results = train_all(X, y, task, feature_names, tune=tune, include_boosting=boosting)
    evaluate_and_save(task, results, X, y)
    save_figures(task, results, X, y)
    return results
```

## Output Files

| Pattern | Example | Contents |
|---|---|---|
| `models/{task}__{model}.joblib` | `cbcb_s__random_forest.joblib` | Fitted model (joblib dump) |
| `models/{task}__meta.joblib` | `cbcb_r__meta.joblib` | Feature names, label encoder, metadata |
| `models/metrics.json` | — | All evaluation metrics per model |
| `assets/{task}_{metric}.png` | `cbcb_s_accuracy.png` | Bar chart per metric |
| `assets/{task}_cm_{model}.png` | `cbcb_r_cm_gradient_boosting.png` | Confusion matrix heatmap |

## See Also

- [API Reference: train.py](../03-reference/api-reference.md#srctrainpy)
- [Configuration Reference: hyperparameter grids](../03-reference/config-reference.md)
- [Tutorial: Getting Started](../01-tutorials/getting-started.md)

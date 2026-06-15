# CBCB — Content-Based Captivation Behavior 🎬

A complete, reproducible **Master's project** implementing and extending the framework from:

> **Predicting User Behavior on Video Streaming by Using Watch-Time Duration Analysis**
> Z. Anwer, S. Qureshi, S. M. Z. Iqbal, A. Zia, S. Anwer.
> *Knowledge-Based Systems* **332** (2025) 114779.
> [doi:10.1016/j.knosys.2025.114779](https://doi.org/10.1016/j.knosys.2025.114779)

CBCB predicts short-term user behavior on video streaming platforms using
**watch-time duration** and historical viewing patterns. It models two
complementary behaviors:

- **CBCB-S (Sequential Captivation)** — the user repeats the same genre on the
  next interaction (`G1 → G1`). *Binary.*
- **CBCB-R (Revert Captivation)** — the user diverges then returns
  (`G1 → G2 → G1`). *Three-class.*

---

## ✨ Features

- 🎲 **Synthetic dataset generator** with genuine sequential/revert structure
  (the real STC/JAWWY data isn't redistributable).
- 🧹 **Full preprocessing**: cleaning, IQR outlier removal, one-hot encoding,
  min-max scaling — `X' = fs(fe(fc(X)))`.
- 🏷️ **Automatic CBCB-S / CBCB-R labelling** from user history.
- 🤖 **Models**: Decision Tree, Random Forest, Gradient Boosting with
  GridSearchCV tuning; optional **XGBoost / LightGBM / CatBoost**.
- 🧠 **Optional deep learning**: LSTM / GRU / Transformer next-genre prediction.
- 📊 **Evaluation**: Accuracy, Precision, Recall, F1, ROC/AUC, confusion matrices,
  feature importance.
- 🖥️ **7-page Streamlit app**: Home, Dataset Manager, Architecture, Prediction,
  Experimental Results, Model Comparison, About.
- 📄 **IEEE report** + **presentation outline** + **Jupyter notebook**.

---

## 📁 Project Structure

```
CBCB_Project/
├── data/                      # generated datasets (gitignored)
├── notebooks/
│   └── CBCB_Exploration.ipynb # EDA + pipeline walkthrough
├── src/
│   ├── config.py              # constants, paths, hyperparameter grids
│   ├── dataset_generator.py   # synthetic STC-like data (>=50k rows)
│   ├── preprocessing.py       # clean / IQR / one-hot / min-max
│   ├── cbcb_s.py              # Sequential labels (Algorithm 2)
│   ├── cbcb_r.py              # Revert labels (Algorithm 3)
│   ├── feature_engineering.py # feature matrix builder
│   ├── train.py              # DT/RF/GB + GridSearchCV + optional boosting
│   ├── evaluate.py           # metrics, ROC/AUC, best-model selection
│   ├── visualization.py      # matplotlib / seaborn / plotly charts
│   ├── deep_learning.py      # LSTM/GRU/Transformer (optional, torch)
│   └── utils.py              # IO, seeding, persistence
├── models/                    # trained models + metrics.json (gitignored)
├── app/
│   ├── Home.py               # Streamlit entry point
│   ├── _shared.py            # styling + helpers
│   └── pages/                # the 6 additional pages
├── assets/                    # generated figures
├── reports/
│   ├── IEEE_Report.md
│   └── Presentation.md
├── requirements.txt
├── README.md
└── main.py                    # CLI orchestrator
```

---

## 🚀 Quickstart

### 1. Install

```bash
cd CBCB_Project
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

> **Optional extras** (boosting + deep learning):
> ```bash
> pip install xgboost lightgbm catboost torch
> ```
> The project runs fine without them — those models are simply skipped.

### 2. Run the full pipeline (CLI)

```bash
python main.py --all                  # 50k rows, both tasks, saves models + figures
python main.py --all --rows 80000     # larger dataset
python main.py --train --no-tune      # fast training, skip GridSearchCV
python main.py --task cbcb_s          # only the sequential task
```

This writes trained models to `models/`, a metrics summary to
`models/metrics.json`, and figures to `assets/`.

### 3. Launch the web app

```bash
streamlit run app/Home.py
```

Then use **Dataset Manager → Train Models** (one click) if you skipped the CLI,
and explore the **Prediction**, **Experimental Results**, and **Model
Comparison** pages.

### 4. Explore the notebook

```bash
jupyter notebook notebooks/CBCB_Exploration.ipynb
```

---

## 🧪 Verify the CBCB labelling logic

The labelling algorithms ship with self-tests on hand-built sequences:

```bash
python -m src.cbcb_s    # asserts G1 G1 G2 G1 -> [1, 0, 0]
python -m src.cbcb_r    # asserts G1 G2 G1 G1 G3 -> [2, 0, 1]
```

---

## 📊 Expected Results

On the default synthetic dataset, results reproduce the paper's qualitative
findings:

| Task | Best Model | ~Accuracy |
|---|---|---|
| CBCB-S | Decision Tree | ~0.78 |
| CBCB-R | Decision Tree | ~0.82 |

- The **Decision Tree** is the strongest/most reliable conventional model.
- **CBCB-R** outperforms **CBCB-S** thanks to the richer revert signal.
- Exact numbers depend on the generator's seed and parameters.

---

## ☁️ Deployment

**Local:** `streamlit run app/Home.py`

**Streamlit Community Cloud:**
1. Push this folder to a GitHub repository.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `app/Home.py`.
3. Set the Python version and ensure `requirements.txt` is detected.
4. (Optional) Pre-train models with `python main.py --all` and commit `models/`,
   or let users train in-app via the Dataset Manager.

**Docker (sketch):**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 📚 Documentation

- **`reports/IEEE_Report.md`** — full IEEE-style technical report.
- **`reports/Presentation.md`** — slide-by-slide defense outline.
- **`notebooks/CBCB_Exploration.ipynb`** — interactive walkthrough.

---

## 📝 Notes on the Dataset

The original Saudi Telecom Company (STC) / JAWWY IPTV dataset (>3.5M rows) is
not publicly redistributable. This project ships a **synthetic generator** that
reproduces the same schema *and* embeds real CBCB structure (genre stickiness, a
Markov transition matrix, explicit reverts, and injected watch-time outliers) so
that experiments are meaningful and fully reproducible. To use real data,
provide a CSV with the columns in `src/config.COLUMNS` via the Dataset Manager's
upload control.

---

## 📄 License & Citation

Educational/research use. If you build on this, please cite the original paper
(above). This implementation was produced as a Master's project reproduction and
extension.

# CBCB — Presentation Outline

*Predicting User Behavior on Video Streaming Using Watch-Time Duration Analysis*

Slide-by-slide content for a ~15-minute Master's project defense. Each slide
lists a title, the key talking points (bullets), and a speaker note.

---

## Slide 1 — Title

- **Predicting User Behavior on Video Streaming Using Watch-Time Duration Analysis**
- Reproduction & extension of the **CBCB** framework (Knowledge-Based Systems, 2025)
- Your name · Supervisor · Institution · Date

*Speaker note:* Open with the one-line thesis — we predict short-term viewing
behavior from watch-time + history, at the user level.

---

## Slide 2 — Problem Statement

- Users abandon platforms when they can't find appealing content.
- Mainstream recommenders use interaction sequences but ignore **watch-time
  duration** and content intrinsics.
- Video-level watch-time optimization ≠ understanding **individual** behavior.

*Speaker note:* Watch-time tells us what users finished vs. abandoned — a strong
implicit interest signal that's underused.

---

## Slide 3 — Objectives

- Model **sequential** (CBCB-S) and **revert** (CBCB-R) short-term behavior.
- Use watch-time duration as an engagement feature.
- Reproduce the preprocessing + classification pipeline.
- Compare Decision Tree / Random Forest / Gradient Boosting.
- Deliver an interactive demo + automatic deployment recommendation.

---

## Slide 4 — Literature Review

- **Content-based:** item features (CBVRP, stylistic features).
- **Collaborative:** similar-user signals (Adsorption, relative engagement, ETA).
- **Hybrid:** THACIL, two-stage YouTube, DVR (watch-time-gain, but video-level).
- **Gap:** none model user-level *sequential + revert* behavior via watch-time.

*Speaker note:* Position CBCB as filling the user-level, watch-time-driven gap.

---

## Slide 5 — CBCB Framework

- Two-step, behavior-aware approach.
- **CBCB-S:** repeat the same genre next (`G1 → G1`) → binary.
- **CBCB-R:** diverge then return (`G1 → G2 → G1`) → three-class.
- Prediction `f(G_t, T)` over history `G_t` and watch-time `T`.

*Speaker note:* Emphasize "captivation" = the watch-time range that keeps users
engaged.

---

## Slide 6 — Labelling Rules (the heart of CBCB)

- **CBCB-S:** `y=1 if genre[i]==genre[i+1] else 0`.
- **CBCB-R:** `y=1` repeat; `y=2` revert (`!=next1 & ==next2`); else `0`.
- Labels generated automatically from each user's date-ordered history.

| Example | CBCB-S | CBCB-R |
|---|---|---|
| Action → Action | 1 | 1 |
| Action → Comedy → Action | 0 then 0 | 2 |
| Action → Comedy → Horror | 0 | 0 |

---

## Slide 7 — Dataset

- Original: STC/JAWWY IPTV, >3.5M rows, 13 columns (not redistributable).
- Ours: **synthetic generator**, same schema, 50k rows / 2k users / 16 genres.
- Genre stickiness + revert probability embed real CBCB structure.
- Injected outliers (~80,000 s) to demonstrate IQR cleaning.

*Speaker note:* Stress that synthetic ≠ random — patterns are intentional and
parameterized, so results are meaningful and reproducible.

---

## Slide 8 — Methodology / Preprocessing

- `X' = fs(fe(fc(X)))`
  - **fc:** clean + IQR outlier removal.
  - **fe:** one-hot encode genre & class.
  - **fs:** min-max scale watch duration.
- Feature engineering: scaled duration, previous genre, genre-run length,
  user averages, session index, time gap.

---

## Slide 9 — Architecture

- Show the flowchart: Dataset → Cleaning → Encoding → Scaling → CBCB-S/R →
  Feature Eng. → ML Models → Prediction → Evaluation.
- Models: Decision Tree, Random Forest, Gradient Boosting (+ optional boosting
  & deep learning).
- Tuning: GridSearchCV (5-fold), DT grid mirrors paper Table 8.

---

## Slide 10 — Results

- **Decision Tree** is the strongest conventional model (matches paper).
- **CBCB-R > CBCB-S** — the revert signal improves accuracy.
- Representative accuracy: CBCB-S ~0.78, CBCB-R ~0.82 (synthetic).
- Metrics: Accuracy, Precision, Recall, F1, ROC/AUC, confusion matrices.

*Speaker note:* Tie back: watch-time feature importance is non-trivial,
validating the central hypothesis.

---

## Slide 11 — Streamlit Demo

- 7 pages: Home · Dataset Manager · Architecture · Prediction · Results ·
  Comparison · About.
- Live: generate → preprocess → **train with one click** → predict.
- Auto best-model selection + deployment recommendation.

*Speaker note:* Do a 60-second live walkthrough: generate small data, train
CBCB-S, predict one interaction.

---

## Slide 12 — Extensions (beyond the paper)

- Boosting: XGBoost, LightGBM, CatBoost (optional).
- Deep learning: LSTM, GRU, Transformer next-genre prediction (optional).
- Fully modular `src/` package + CLI orchestrator.

---

## Slide 13 — Conclusion

- Reproduced CBCB end-to-end; confirmed its core findings.
- Watch-time + sequential/revert modelling → accurate, interpretable prediction.
- Decision Tree recommended for deployment.
- Delivered a reusable, demo-ready application.

---

## Slide 14 — Future Work & Q&A

- Reinforcement learning, graph-based methods, on-device inference.
- Validate on full STC data + multimodal features.
- **Thank you — questions?**

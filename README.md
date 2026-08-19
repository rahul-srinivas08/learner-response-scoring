# Automated learner-response scoring — take-home submission

**Start here: [`REPORT.md`](REPORT.md)** — the short written report (skim, a few minutes). Everything below is the supporting code and detail behind it.

## Before running anything

```
python3 setup_check.py
```

Verifies every required package is installed (`pip install -r requirements.txt` if not) and runs a smoke test against `evaluation.py`/`feature_engineering.py` — catches a broken environment before you spend time on a notebook run that'd fail partway through.

## Reading order

1. **[`Part_01_EDA.ipynb`](Part_01_EDA.ipynb)** — analysis. Data quality, class balance, the two evaluation ceilings (human-agreement QWK and the stricter text-only oracle floor), leakage risk, ASR-failure handling, feature engineering, and an initial abstention prototype.
2. **[`Part_2_Classical_ML.ipynb`](Part_2_Classical_ML.ipynb)** — Linear Regression vs. XGBoost, compared honestly on Part 1's group-safe split. Confusion matrix, per-language/CEFR/confidence breakdowns, latency benchmark, a working abstention filter, five tested improvement attempts (two adopted, three rejected with reasons), and the model-choice writeup.
3. **[`Part_3_Transformer.ipynb`](Part_3_Transformer.ipynb)** — LoRA fine-tuning of a small multilingual transformer with focal loss, compared against Part 2 on the identical held-out rows. Covers backbone justification, quantization (tried, measured, reported honestly), and a cloud/edge/mobile deployment discussion.

Not a required deliverable, but referenced throughout Part 3: **[`Part_3b_Architecture_Comparison.ipynb`](Part_3b_Architecture_Comparison.ipynb)** — a standalone, self-contained head-to-head test of full text-serialization vs. Part 3's hybrid input design, trained and evaluated independently as evidence for that architecture decision.

Each notebook has its own "how to run" line at the top and reuses `feature_engineering.py` / `evaluation.py` — shared modules, not redefined per notebook, so the three parts can't silently drift apart on what a feature or metric means.

## Files

| File | What it is |
|---|---|
| `dataset.csv`, `DATA_DICTIONARY.md`, `CANDIDATE_BRIEF.md` | Given |
| `feature_engineering.py` | The 28-feature matrix, shared by all three notebooks |
| `evaluation.py` | Metrics, group-safe CV splitting, the two ceilings, shared by Parts 2 and 3 |
| `Part_01_EDA.ipynb`, `Part_2_Classical_ML.ipynb`, `Part_3_Transformer.ipynb` | The three deliverables |
| `Part_3b_Architecture_Comparison.ipynb` | Supplementary — full text-serialization vs. hybrid input, tested head-to-head |
| `REPORT.md` | The short written report — read this first |
| `AI_COLLABORATION_LOG.md` | Which tools were used, what was directed vs. AI-proposed, and things the AI got wrong |
| `DEPLOYMENT.md` | Deployment strategy for Part 2 and Part 3 — artifacts, latency, cloud/edge/mobile, rollout |
| `requirements.txt`, `setup_check.py` | Environment setup and verification |
| `part2_predictions.csv`, `part3_predictions.csv` | Row-level predictions (id, input, true score, prediction) for external review |

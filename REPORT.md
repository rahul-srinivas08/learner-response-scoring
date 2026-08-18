# Automated learner-response scoring — report

Full detail, code, and outputs live in the three notebooks (`Part_01_EDA.ipynb`, `Part_2_Classical_ML.ipynb`, `Part_3_Transformer.ipynb`). This is the skim version. Run `python3 setup_check.py` before opening any notebook — it verifies the environment and sanity-checks `evaluation.py`/`feature_engineering.py` before you spend time on a run that'd fail 20 minutes in.

## What's in the data, and what we did about it

- **2,000 rows, imbalanced but not pathologically** (`human_score` 2 is the plurality at 31.3%; Italian is the thinnest language at 9.4%; C1 CEFR is thin at 6.5%). Reported per-language/per-CEFR everywhere, not just one overall number.
- **`cefr_level` never has a `C2` row, but `C2` is a real level the app must support.** Encoded CEFR ordinally over the full A1–C2 schema regardless — verified with a synthetic-C2 probe, not just claimed.
- **85.5% of transcripts repeat** (291 unique transcripts cover all 2,000 rows) — a real leakage risk for a random split. Used `StratifiedGroupKFold(asr_transcript)` everywhere, and quantified the leak directly in Part 2 rather than assuming it: turned out small for classical aggregate features (~0.01 QWK), but flagged as structurally more dangerous for Part 3's transformer, which reads raw tokens.
- **ASR sometimes fails outright** (4.6% of rows) — kept as an explicit feature, never dropped, never used as a hard confidence gate (1 in 4 "failed" rows still scored ≥3; the human rater heard the audio, not the broken text).
- **Prompts are English, transcripts are 5 languages** — used a frozen multilingual sentence embedding (`paraphrase-multilingual-MiniLM-L12-v2`) for the relevance feature, since lexical overlap is blind on ~80% of rows.

## How we evaluated, and why those metrics

- **QWK (quadratic weighted kappa)** as the headline metric — this is a 0–4 *ordinal* target, and QWK penalizes a 2-point miss far more than a 1-point miss, unlike plain accuracy. Majority-baseline accuracy is a misleadingly-okay-looking 31.2%; its QWK is 0.000 — accuracy alone would have hidden that the baseline knows nothing about ordering.
- **MAE** alongside QWK — more interpretable ("how many points off, on average").
- **`false_praise`** (advancing a learner who actually scored ≤1) — built specifically to operationalize the brief's own stated worst failure mode ("a confidently wrong 'great job!' is worse than an honest 'I'm not sure'"), not a generic metric.
- **Bootstrap CIs** on QWK — necessary with n=2,000 and thin slices like Italian (n=188).

## What the numbers actually mean — two ceilings, not one

- **Human ceiling: QWK 0.795, MAE 0.484.** Two independent human raters only agree exactly 55.5% of the time on the *same audio*. No model should be judged against 100%.
- **Oracle floor (the real bar for a text-only model): MAE 0.743.** Holding the transcript text perfectly identical, learners producing the same words still score differently on delivery/pronunciation — information no text-only model can ever recover. This is the stricter, more honest bar, and both Part 2 and Part 3's models land right at it (MAE 0.76–0.84), not below a reachable target.

## Model comparison — same held-out rows across Part 2 and Part 3

| Model | QWK | MAE | Exact | Within-1 | `false_praise` |
|---|---|---|---|---|---|
| Majority baseline | 0.000 | 0.924 | 0.312 | 0.763 | 0.000 |
| Linear Regression | 0.484 | 0.747 | 0.374 | 0.889 | 0.127 |
| **XGBoost** | **0.520** | 0.824 | 0.332 | 0.854 | 0.167 |
| Transformer (LoRA + focal loss) | 0.451 | 0.839 | 0.356 | 0.834 | **0.087** |

*(Part 2's headline number elsewhere is 0.531 QWK under 5-fold CV; the 0.520 here is the same model re-scored on Part 3's single fixed split so the transformer comparison is apples-to-apples — both are real, consistent numbers, not a discrepancy.)*

## Why this model — what we considered, what tipped the choice

**Recommendation: ship Part 2's XGBoost, with the XGB+LinReg ensemble noted as the better-QWK option.**

- **The ensemble of XGBoost + Linear Regression (simple average) scored QWK 0.542 — the best number in the entire project**, ahead of XGBoost alone (0.531) and Linear Regression alone (0.521). Not adopted as the default only for simplicity: it's two models to maintain instead of one, for a 0.01–0.02 QWK gain. Kept as the documented pick if squeezing out QWK matters more than operational simplicity.
- **XGBoost alone wins on QWK vs. the transformer** (0.520 vs. 0.451 on the identical held-out split) — the headline ordinal-agreement metric.
- **The transformer wins decisively on `false_praise`** (0.087 vs. 0.167) — a real, brief-relevant advantage (its predictions skew more conservative, less willing to commit to score extremes), but one data point on one 404-row split isn't enough to give up 0.07 QWK for it.
- **The confusion matrix found XGBoost never predicts class 0** — every truly-0 row rounds up to 1+. Doesn't change the app's actual behavior today (the action mapping treats scores 0 and 1 identically — both trigger "re-prompt"), but it's a named, real gap in the model's resolution at the low end, not swept under the headline QWK number.
- **Tested true ordinal classification (Frank & Hall: 4 binary "is score > k?" classifiers) as a direct fix for that gap** — it works (predicts class 0 for 118 rows) but costs QWK (0.511 vs. 0.531) and MAE (0.788 vs. 0.764). Not adopted, because QWK is the metric that matters here and the class-0 gap costs nothing given today's action mapping — but the fix exists and is ready if that mapping ever changes.
- **Alternatives explicitly rejected, with reasons:** a plain 5-way classifier (discards the ordinal structure QWK is built around — `OptimizedRounder` on a continuous regressor keeps it for free); manual feature de-correlation (Part 1 already showed this risks losing real signal, and trees don't need it); sample-weighting rare classes (didn't fix the class-0 gap — the raw model output for true-0 and true-1 rows overlaps too much for any threshold to separate); hyperparameter tuning (no config beat the original defaults); class-`alpha` reweighting in the transformer's focal loss (caused visibly unstable training — caught by watching the predicted-class distribution swing wildly, not assumed from theory).
- **Both classical and transformer clear the 300ms latency budget by 15–70x** — not a deciding factor either way.

## Part 3's required engineering topics

- **Framework:** PyTorch + `transformers` + `peft` (LoRA) for fine-tuning; ONNX Runtime named as the right choice for a real production *serving* path (not implemented — discussion level, per the brief).
- **Backbone justification:** kept `paraphrase-multilingual-MiniLM-L12-v2` — already multilingual, already proven CPU-fast, and 82% of its 117.7M params are the vocabulary embedding table, not reasoning capacity, so it isn't oversized for the task. The QWK gap vs. XGBoost is much more likely explained by training-data volume (1,458 rows) than by an undersized backbone.
- **LoRA:** trains 197,893 params — **0.17%** of the model — vs. full fine-tuning's 117.7M.
- **Quantization:** tried, measured, reported honestly — dynamic INT8 was *slower* than fp32 at single-request batch size on this backend (9.7ms vs. 4.1ms) and shrank the model only 14%, because the embedding table (82% of size) isn't touched by `nn.Linear`-only quantization. Static/QAT quantization and embedding-table pruning named as the approaches that would actually help.
- **Latency/memory (measured, not estimated):** ~4ms median inference, ~470MB in memory, ~1.9s one-time cold load.
- **Deployment — cloud, edge, mobile:**
  - *Cloud* (the brief's actual target, "millions of learners"): model-loading cost (once per server process) matters more than per-request latency; horizontal scaling is the standard answer, no batching needed for correctness.
  - *Edge*: the case for edge over a well-placed cloud region is about network round-trip time for distant learners, not compute speed — compute here is already 70x under budget, so edge only helps if network latency (not measured in this dataset) turns out to be the real bottleneck.
  - *Mobile/on-device*: real option for an offline-tolerant use case, but the embedding table dominates on-device footprint — would want vocabulary pruning or a smaller distilled backbone first. Given §6/§7's honest limitations, a mobile deployment should currently ship the classical model too.

## What we'd do next with more time

Add an explicit abstention layer to Part 2's XGBoost using its own working uncertainty signal (model-disagreement AUC 0.578, already validated) — this closes the trust gap that currently favors the transformer, without giving up XGBoost's QWK advantage.

Second choice: **label-preserving data augmentation for the transformer** (e.g. back-translation of existing transcripts, kept under the same `human_score`) — more textual variety per label without inventing new labels. Deliberately *not* recommending synthetic off-topic examples with guessed labels: relevance is only 1 of 3 equally-weighted grading criteria per the brief, so an off-topic-but-fluent response might still score a 2, not a 0 — inventing that label risks teaching the model something false rather than filling a real gap. (Tested a related idea — ensembling the transformer with XGBoost's output — and it made both QWK and `false_praise` worse, so that specific fix is ruled out, not just untried.)

## AI collaboration log

Built with Claude Code (Sonnet 5) across all three parts and this report — every notebook cell was reviewed and its printed output checked against what the accompanying markdown claims before being called done, not assumed correct from the code alone.

**What was directed vs. what the AI proposed on its own.** The core evaluation strategy — QWK/MAE as the primary metrics, `false_praise`, the two ceilings, group-safe CV — was defined in `evaluation.py` before this AI session and given as existing infrastructure to build on, not something the AI authored from scratch. Within this session, the confusion matrix (Part 2), focal loss, LoRA, and quantization (Part 3) were all explicit direction, not the AI's own idea — implemented, tuned, and reported against that instruction, including catching and fixing the focal-loss instability below. The AI's own contributions were the specific diagnostics run *within* that directed strategy — the ensemble, the ordinal-classification test, the stratified error-analysis slices, the per-notebook improvement-attempt tables — plus a refactoring pass on Part 1 (duplicate section header, stray empty cells, a typo, reordering two sections, trimming redundant content) done as cleanup, not new analysis.

**Two concrete things the AI got wrong, and how they were caught:**
1. **A real crash, not a hypothetical one:** running XGBoost's `.fit()` in the same process right after a PyTorch/LoRA training loop (Part 3, comparing against Part 2 on the same split) caused a hard segfault (exit code 139) — a torch/xgboost OpenMP conflict on this platform. Caught by the notebook simply crashing rather than erroring cleanly; root-caused by bisecting the script stage by stage until the exact failing combination was isolated. Fixed with `OMP_NUM_THREADS=1` / `KMP_DUPLICATE_LIB_OK=TRUE` set before any imports, `torch.set_num_threads(1)`, and `n_jobs=1` on `XGBRegressor` — now baked into both Part 3's first cell and `setup_check.py`.
2. **Focal loss with class-`alpha` reweighting looked correct by the textbook recipe but destabilized training** on this dataset's mild (~3x) imbalance — QWK oscillated epoch to epoch with no clear trend. Caught by printing the predicted-class distribution every epoch and watching it swing rather than settle, not by assuming the standard recipe would work. Fixed by dropping `alpha`, keeping `gamma` — training became smooth and monotonic.

A third thing worth naming as a near-miss rather than a full error: `jupyter nbconvert` was silently executing notebooks under a different Python installation than the one dependencies were being installed into (a global Jupyter kernel pointing at the system Python, not the project's `.venv`), so a `pip install` that appeared to succeed in one shell had no effect on what the notebook actually saw. Caught by a `ModuleNotFoundError` immediately after a `!pip install` cell that should have made the import succeed. `setup_check.py` now prints `sys.executable` first specifically so this class of mismatch is visible immediately rather than discovered via a confusing failure.

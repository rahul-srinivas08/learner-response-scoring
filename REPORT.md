# Automated learner-response scoring — report

Full detail, code, and outputs live in the three notebooks (`Part_01_EDA.ipynb`, `Part_2_Classical_ML.ipynb`, `Part_3_Transformer.ipynb`). This is the skim version. Run `python3 setup_check.py` first — verifies the environment and sanity-checks the shared modules before a notebook run fails 20 minutes in. See `AI_COLLABORATION_LOG.md` for how Claude Code was used across this project.

## What's in the data, and what we did about it

- **2,000 rows, mildly imbalanced** (score 2 is plurality at 31.3%; Italian thinnest language at 9.4%). Reported per-language/per-CEFR everywhere.
- **`cefr_level` never has `C2`, but `C2` is a real level the app must support.** Encoded ordinally over the full A1–C2 schema — verified with a synthetic-C2 probe.
- **85.5% of transcripts repeat** — real leakage risk. Used `StratifiedGroupKFold(asr_transcript)` everywhere; quantified the leak rather than assuming it (small for classical features, structurally riskier for Part 3's raw-token transformer).
- **ASR sometimes fails outright** (4.6%) — kept as a feature, never dropped (1 in 4 "failed" rows still scored ≥3).
- **Prompts are English, transcripts are 5 languages** — used a multilingual embedding for relevance, since lexical overlap is blind on ~80% of rows. Validated, not assumed: synthetic off-topic pairs score 0.198 vs. 0.480 for real pairs; a TF-IDF alternative correlates only 0.012 with the score vs. the embedding's 0.123 and is blind on 83% of non-English rows.

## How we evaluated, and why those metrics

- **QWK** as the headline metric — ordinal 0–4 target, penalizes a 2-point miss more than a 1-point miss. Majority baseline: 31.2% accuracy but QWK 0.000 — accuracy alone hides that it knows nothing about ordering.
- **MAE** alongside QWK for interpretability.
- **`false_praise`** (advancing a learner who scored ≤1) — operationalizes the brief's named worst failure mode directly.
- **Bootstrap CIs** on QWK — necessary at n=2,000 with thin slices like Italian (n=188).

## What the numbers mean — two ceilings, not one

- **Human ceiling: QWK 0.795, MAE 0.484.** Two raters agree exactly only 55.5% of the time on the *same audio* — not 100%.
- **Oracle floor (real bar for a text-only model): MAE 0.743.** Same transcript text, different delivery, still different scores — information no text-only model can recover. Both Part 2 and Part 3 land right at this (MAE 0.76–0.84), not below a reachable target.

## Model comparison — same held-out rows, Part 2 and Part 3

| Model | QWK | MAE | Exact | Within-1 | `false_praise` |
|---|---|---|---|---|---|
| Majority baseline | 0.000 | 0.924 | 0.312 | 0.763 | 0.000 |
| Linear Regression | 0.484 | 0.747 | 0.374 | 0.889 | 0.127 |
| **XGBoost** | **0.520** | 0.824 | 0.332 | 0.854 | 0.167 |
| Transformer (LoRA + focal loss) | 0.451 | 0.839 | 0.356 | 0.834 | **0.087** |

*(Part 2's headline number elsewhere is 0.531 under 5-fold CV; 0.520 here is the same model re-scored on Part 3's fixed split for a fair comparison — both real, not a discrepancy.)*

## Why this model

**Recommendation: ship Part 2's XGBoost.** XGB+LinReg ensemble scores higher (QWK 0.542, best in the project) but isn't the default — two models to maintain for a 0.01–0.02 gain; documented as the pick if squeezing QWK matters more than simplicity.

- **XGBoost wins QWK** vs. the transformer (0.520 vs. 0.451). **Transformer wins `false_praise`** decisively (0.087 vs. 0.167) — real, but one split isn't enough to give up 0.07 QWK for it.
- **A criterion-coverage gap also favors Part 2**: the brief's three grading criteria are equally weighted; Part 2 has some relevance signal (`relevance_cosine_sim`), Part 3's transformer has none — see "Part 3's input design" below.
- **Confusion matrix: XGBoost never predicts class 0.** Costs nothing today (0 and 1 both trigger "re-prompt"), but named honestly. A tested fix (ordinal classification, Frank & Hall) works but costs QWK (0.511 vs. 0.531) — not adopted, kept ready if the action mapping ever splits 0 from 1.
- **Rejected, with reasons:** plain 5-way classifier (discards ordinal structure); manual feature de-correlation (Part 1 already showed this loses signal); sample-weighting (didn't fix class-0 — raw scores for true-0/true-1 overlap too much); hyperparameter tuning (no gain); class-`alpha` focal-loss reweighting (destabilized training — caught by watching predictions swing, not assumed).
- **Latency, throughput, and scaling — Part 2 specifically:** ~18ms median steady-state latency (measured, `feature_engineering.build_features` + `.predict()` together), clearing the 300ms budget by ~17x. Worked from that latency (1000ms ÷ 18ms), one CPU core handles ~55 req/s; at an illustrative 10,000 req/s peak (not measured — no real traffic data in this project) that's ~182 cores. Scoring is stateless (no session state, no batching needed for correctness), so horizontal autoscaling on request volume is the standard answer — cost is driven by memory-per-process (the ~470MB multilingual embedding model for `relevance_cosine_sim`) more than CPU. Both models clear the 300ms budget by 15–70x overall — not a deciding factor between them, but Part 2's own numbers are worth stating on their own rather than only in the Part 2 vs. Part 3 comparison. Full treatment in `Part_2_Classical_ML.ipynb`'s deployment section and `DEPLOYMENT.md`.

## Part 3's required engineering topics

- **Framework:** PyTorch + `transformers` + `peft` (LoRA). ONNX named as the right serving-path choice, not implemented — discussion level per the brief.
- **Backbone:** kept `paraphrase-multilingual-MiniLM-L12-v2` — multilingual, CPU-fast, 82% of its 117.7M params are the embedding table, not reasoning capacity. Tested a 4x-larger encoder (XLM-R-base, 278M params) directly rather than assuming size would help — it scored worse on every metric (QWK 0.315 vs. 0.451, `false_praise` 0.367 vs. 0.087), confirming the QWK gap vs. XGBoost is data-volume (1,458 rows), not backbone size.
- **LoRA:** 197,893 trainable params — 0.17% of the model.
- **Quantization:** tried, reported honestly — dynamic INT8 was *slower* than fp32 here (9.7ms vs. 4.1ms) and shrank the model only 14% (embedding table untouched by `nn.Linear`-only quantization). Static/QAT named as the real next step.
- **Latency/memory (measured):** ~4ms median, ~470MB, ~1.9s cold load.
- **Deployment:** *Cloud* — model-loading cost per process matters more than per-request latency; horizontal scaling. *Edge* — only helps if network latency (not measured) is the real bottleneck, since compute is already 70x under budget. *Mobile* — embedding table dominates footprint; ship the classical model there today given Part 3's honest limitations. Full treatment in `DEPLOYMENT.md`.
- **Throughput, quantified:** from measured latency, the transformer clears ~250 req/s per CPU core vs. XGBoost's ~55 (1000ms ÷ latency) — roughly **4.5x fewer cores needed for the same load**, a real compute-cost difference distinct from the QWK-based recommendation above.

## Part 3's input design — tested, not assumed, including a correction

Grammar and intelligibility are covered by the transcript + ASR confidence; relevance has no input at all in the shipped Part 3 model — no prompt, no relevance feature. Tested directly whether to fix that, on both backbones used in this project:

| Backbone | Configuration | QWK | MAE | `false_praise` |
|---|---|---|---|---|
| MiniLM-L12 (shipped) | Transcript only | **0.451** | **0.839** | **0.087** |
| MiniLM-L12 | + prompt | 0.425 | 0.911 | 0.140 |
| MiniLM-L12 | + prompt + language | 0.370 | 0.913 | 0.153 |
| XLM-R-base | Transcript only | 0.315 | 1.198 | 0.367 |
| XLM-R-base | + prompt | 0.369 | 0.941 | 0.233 |
| XLM-R-base | + prompt + language | 0.423 | 0.955 | 0.193 |

On MiniLM, every variant that added the prompt scored worse — the finding stated earlier in this report. **Correction, found while testing whether a bigger backbone was the real bottleneck (see Backbone above): re-running the same prompt tests on XLM-R-base showed the opposite trend — adding context *improved* its results, monotonically, even though its best configuration still didn't beat MiniLM's transcript-only number.** So "adding the prompt hurts" isn't a fixed property of this dataset, it's specific to MiniLM's smaller capacity given how repetitive the transcripts are (85.5% repeat) — MiniLM doesn't need more context to fit well; XLM-R-base's larger capacity benefits from it but starts from a worse baseline. **The shipped configuration (MiniLM, transcript-only) remains the best of all 8 tested across both backbones — but the fix for the relevance gap is more accurately "more data, possibly paired with a backbone that can use extra context," not the narrower "more data alone" claimed earlier.**

## What we'd do next with more time

1. **Abstention layer on Part 2's XGBoost** using its own working uncertainty signal (model-disagreement AUC 0.578, already validated) — closes the trust gap without losing XGBoost's QWK edge.
2. **Label-preserving data augmentation for the transformer**, backed by a real scaling test (QWK 0.187 → 0.433 → 0.451 at 33%/66%/100% of data, still rising) — confirms the transformer is data-limited, not backbone-limited. Recommended form: back-translation/paraphrasing under the *same* label, group-safe (augmented rows join their source's CV fold), targeted at the thin classes 0/4, validated against the current baseline before adopting. Not recommended for Part 2 — already at its oracle-floor ceiling, a different constraint than data volume. Deliberately not recommending synthetic off-topic examples with guessed labels — relevance is only 1 of 3 equally-weighted criteria, so a wrong guessed label risks teaching something false. (Ensembling the transformer with XGBoost was tested and made both QWK and `false_praise` worse — ruled out, not just untried.)

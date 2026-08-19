# Automated learner-response scoring — report

Full detail, code, and outputs live in the three notebooks (`Part_01_EDA.ipynb`, `Part_2_Classical_ML.ipynb`, `Part_3_Transformer.ipynb`). This is the skim version. Run `python3 setup_check.py` first — verifies the environment and sanity-checks the shared modules before a notebook run fails 20 minutes in.

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
- **Both models clear the 300ms budget by 15–70x** — not a deciding factor.

## Part 3's required engineering topics

- **Framework:** PyTorch + `transformers` + `peft` (LoRA). ONNX named as the right serving-path choice, not implemented — discussion level per the brief.
- **Backbone:** kept `paraphrase-multilingual-MiniLM-L12-v2` — multilingual, CPU-fast, 82% of its 117.7M params are the embedding table, not reasoning capacity. QWK gap vs. XGBoost is data-volume (1,458 rows), not backbone size.
- **LoRA:** 197,893 trainable params — 0.17% of the model.
- **Quantization:** tried, reported honestly — dynamic INT8 was *slower* than fp32 here (9.7ms vs. 4.1ms) and shrank the model only 14% (embedding table untouched by `nn.Linear`-only quantization). Static/QAT named as the real next step.
- **Latency/memory (measured):** ~4ms median, ~470MB, ~1.9s cold load.
- **Deployment:** *Cloud* — model-loading cost per process matters more than per-request latency; horizontal scaling. *Edge* — only helps if network latency (not measured) is the real bottleneck, since compute is already 70x under budget. *Mobile* — embedding table dominates footprint; ship the classical model there today given Part 3's honest limitations. Full treatment in `DEPLOYMENT.md`.

## Part 3's input design — tested, not assumed

Grammar and intelligibility are covered by the transcript + ASR confidence; relevance has no input at all in Part 3 — no prompt, no relevance feature. Tested directly whether to fix that:

| Configuration | QWK | MAE | `false_praise` |
|---|---|---|---|
| Transcript only (shipped) | **0.451** | **0.839** | **0.087** |
| Prompt + transcript | 0.425 | 0.911 | 0.140 |
| Prompt + language + transcript | 0.370 | 0.913 | 0.153 |

Every variant that added the prompt scored worse on all three metrics. Root cause: only 10 unique prompts across 1,458 rows gives the model an easy shortcut ("prompt X scores around Y") that beats the harder skill of judging relevance — confirmed by validation QWK rising while test QWK fell, the signature of a shortcut that doesn't generalize. **Relevance stays unrepresented in Part 3 — fixable with more data, not a cleverer prompt encoding.**

## What we'd do next with more time

1. **Abstention layer on Part 2's XGBoost** using its own working uncertainty signal (model-disagreement AUC 0.578, already validated) — closes the trust gap without losing XGBoost's QWK edge.
2. **Label-preserving data augmentation for the transformer**, backed by a real scaling test (QWK 0.187 → 0.433 → 0.451 at 33%/66%/100% of data, still rising) — confirms the transformer is data-limited, not backbone-limited. Recommended form: back-translation/paraphrasing under the *same* label, group-safe (augmented rows join their source's CV fold), targeted at the thin classes 0/4, validated against the current baseline before adopting. Not recommended for Part 2 — already at its oracle-floor ceiling, a different constraint than data volume. Deliberately not recommending synthetic off-topic examples with guessed labels — relevance is only 1 of 3 equally-weighted criteria, so a wrong guessed label risks teaching something false. (Ensembling the transformer with XGBoost was tested and made both QWK and `false_praise` worse — ruled out, not just untried.)

## AI collaboration log

Built with Claude Code (Sonnet 5) across all three parts and this report — every cell's output checked against what its markdown claims, not assumed from the code alone.

**Directed vs. AI-proposed:** the core eval strategy (QWK/MAE, `false_praise`, the two ceilings, group-safe CV) predates this session, in `evaluation.py`. Within-session direction: the confusion matrix, focal loss, LoRA, and quantization were explicit asks, implemented and reported against instruction. AI-proposed: the ensemble, the ordinal-classification test, the stratified error slices, and a Part 1 refactoring pass (duplicate header, empty cells, a typo, reordering).

**Persistent direction produced more rigor than one test would have:** repeated direction to test the prompt (and language) in the transformer's text input produced four independent variants tested, not one — all four underperformed the shipped design (worst case: `false_praise` 0.340 vs. 0.087). The value wasn't the AI catching its own mistake; it was direction that kept the question open until the evidence was conclusive.

**Two concrete things the AI got wrong:**
1. **A real crash:** XGBoost's `.fit()` after a PyTorch/LoRA training loop segfaulted (torch/xgboost OpenMP conflict). Root-caused by bisecting the script; fixed with `OMP_NUM_THREADS=1`/`KMP_DUPLICATE_LIB_OK=TRUE` before any imports, `torch.set_num_threads(1)`, `n_jobs=1` — now in Part 3's first cell and `setup_check.py`.
2. **Focal loss with class-`alpha` reweighting** looked correct by the textbook recipe but destabilized training (QWK oscillating epoch to epoch). Caught by watching the predicted-class distribution swing rather than settle, not by trusting the standard recipe. Fixed by dropping `alpha`, keeping `gamma`.

**Near-miss:** `jupyter nbconvert` was silently executing under a different Python than where dependencies were installed, so a successful `pip install` had no effect on the notebook. Caught by a `ModuleNotFoundError` right after an install that should have worked. `setup_check.py` now prints `sys.executable` first so this surfaces immediately.

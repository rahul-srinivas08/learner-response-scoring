# Automated learner-response scoring — report

Full code and outputs: `Part_01_EDA.ipynb`, `Part_2_Classical_ML.ipynb`, `Part_3_Transformer.ipynb`. Run `python3 setup_check.py` first to verify the environment.

## What's in the data, and what we did about it

- **2,000 rows, mildly imbalanced** (score 2 is plurality at 31.3%; Italian thinnest at 9.4%) — reported per-language/CEFR throughout.
- **`cefr_level` never has C2, but the app must support it** — encoded ordinally over the full A1–C2 schema, verified with a synthetic-C2 probe.
- **85.5% of transcripts repeat** — real leakage risk. Used `StratifiedGroupKFold(asr_transcript)` everywhere; quantified the leak instead of assuming it (small for classical features, structurally riskier for Part 3's raw-token transformer).
- **ASR fails outright on 4.6% of rows** — kept as a feature, never dropped (1 in 4 "failed" rows still scored ≥3).
- **Prompts are English, transcripts are 5 languages** — used a multilingual embedding for relevance. Validated it actually works: synthetic off-topic pairs score 0.198 vs. 0.480 for real matched pairs.

## How we evaluated, and why

**QWK** (ordinal, penalizes a 2-point miss more than a 1-point miss) is the headline metric — the majority baseline gets 31% accuracy but QWK 0.000, so accuracy alone hides that it knows nothing about ordering. **MAE** alongside for interpretability. **`false_praise`** (advancing a learner scored ≤1) operationalizes the brief's named worst failure mode directly. **Bootstrap CIs** on QWK — necessary at thin slices like Italian (n=188).

## What the number means — two ceilings, not one

**Human ceiling: QWK 0.795.** Two raters agree exactly only 55.5% of the time on the *same audio*. **Oracle floor (the real bar for a text-only model): MAE 0.743** — same transcript, different delivery, still different scores, information no text model can recover. Both models land right at this floor (MAE 0.76–0.84), not below a reachable target.

## Model comparison — same held-out rows

| Model | QWK | MAE | `false_praise` |
|---|---|---|---|
| Majority baseline | 0.000 | 0.924 | 0.000 |
| Linear Regression | 0.484 | 0.747 | 0.127 |
| **XGBoost** | **0.520** | 0.824 | 0.167 |
| Transformer (LoRA + focal loss) | 0.451 | 0.839 | **0.087** |

## Why this model

**Ship XGBoost.** It wins QWK; the transformer wins `false_praise` (0.087 vs. 0.167) but not by enough to give up 0.07 QWK on one 404-row split. XGBoost also carries some relevance signal (`relevance_cosine_sim`); the shipped transformer has none — a real gap against the brief's three equally-weighted criteria. Confusion matrix: XGBoost never predicts class 0 — costs nothing today (0 and 1 both trigger "re-prompt") but named honestly. Rejected, with reasons: plain classifier (loses ordinal structure), manual feature de-correlation (loses signal), sample-weighting/hyperparameter tuning (no gain), true ordinal classification (fixes class-0 but costs QWK, 0.511 vs 0.531). Both models clear the 300ms budget 15–70x — not a deciding factor, but at ~55 req/s per core, XGBoost needs ~182 cores for an illustrative 10,000 req/s peak vs. the transformer's ~40 — worth having in hand for a cost conversation later.

## Part 3's engineering topics

**Framework:** PyTorch + `transformers` + `peft` (LoRA); ONNX named as the right serving path, not built. **Backbone:** `paraphrase-multilingual-MiniLM-L12-v2` — tested a 4x-larger encoder (XLM-R-base) directly rather than assuming size helps; it scored worse everywhere (QWK 0.315 vs. 0.451), confirming the gap vs. XGBoost is data volume (1,458 rows), not backbone size. **Quantization:** tried honestly — dynamic INT8 was *slower* than fp32 here (9.7ms vs. 4.1ms). **Measured:** ~4ms latency, ~470MB, ~1.9s cold load. **Deployment:** cloud is the fit today (memory-per-process, not latency, is the real constraint at scale); mobile needs the embedding table addressed first, so ship XGBoost there for now. Full treatment in `DEPLOYMENT.md`.

## Part 3's input design — tested, not assumed, including a correction

The shipped transformer has no relevance signal (transcript-only). Every attempt to add the prompt hurt on MiniLM — stated as settled after four tests. Re-running the same tests on XLM-R-base reversed it: adding context *helped* there. So "adding context hurts" isn't a property of the dataset, it's MiniLM-specific — the honest fix is more data paired with a backbone that can use it, not a fixed architectural conclusion. The shipped config (MiniLM, transcript-only) remains the best of all 8 tested across both backbones.

## What we'd do next

Wire Part 2's model-disagreement signal (AUC 0.578, already validated) into a real abstention layer — closes the trust gap without losing XGBoost's QWK edge, and is the cheapest lever available given both models are already near their respective ceilings.

## AI collaboration log

Built with Claude Code (Sonnet 5) across all three parts and this report. **Part 1 was the most direction-heavy:** `relevance_cosine_sim`'s weak importance was pushed on past the first plausible answer, which is what produced the synthetic-mismatch and TF-IDF validation tests — direction that shaped the eval framing both later parts build on. **Part 2/3:** the confusion matrix, focal loss, LoRA, and quantization were explicit asks; the ensemble and ordinal-classification test were AI-proposed within that direction. **What the AI got wrong:** after four MiniLM tests, it declared "adding the prompt always hurts" settled — direction to test a second backbone (XLM-R-base) proved that over-generalized, since the same tests reversed on the bigger model. Caught by persistence, not by the AI catching itself. Also caught: an XGBoost/PyTorch OpenMP segfault (root-caused by bisecting the script, fixed with `OMP_NUM_THREADS=1`), and a focal-loss `alpha`-reweighting instability (caught by watching the predicted-class distribution swing, not by trusting the textbook recipe).

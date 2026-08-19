# Deployment strategy — Part 2 and Part 3

Discussion-level, per the brief ("how you'd think about deploying this... not an implementation requirement"). All numbers below are measured in the notebooks, not estimated — see `Part_2_Classical_ML.ipynb`'s section 5 and `Part_3_Transformer.ipynb`'s sections 8/9 for the underlying benchmarks.

## Recommended path: ship Part 2 (XGBoost) as primary, Part 3 (transformer) as a documented alternative

Per `REPORT.md`: XGBoost wins on QWK (the headline metric), the transformer wins on `false_praise` but not by enough to override the QWK gap on current evidence. This document assumes that recommendation and describes deployment for both, since the transformer remains a real, evaluated option — not a discarded experiment.

---

## 1. What actually ships — the artifacts

**Part 2 (XGBoost):**
- The trained `XGBRegressor` (serializable via `model.save_model()`, typically a few hundred KB)
- `feature_engineering.py` — must ship as-is; the model is meaningless without the exact same 28-feature construction
- `OptimizedRounder`'s fitted thresholds (4 floats) — converts the regressor's continuous output to a 0–4 class
- The frozen `paraphrase-multilingual-MiniLM-L12-v2` weights (for `relevance_cosine_sim`) — this is the largest artifact in Part 2's pipeline despite XGBoost itself being tiny

**Part 3 (transformer):**
- LoRA adapters **merged into the base weights** (`merge_and_unload()`, already demonstrated in section 8) — the deployed artifact is a plain fine-tuned transformer, no runtime PEFT dependency, no separate adapter file to manage
- The tokenizer (ships with the model)
- The small MLP head's weights
- The `StandardScaler` fitted on the 4 auxiliary features (`cefr_ordinal`, `conf_mean`, `conf_min`, `is_asr_failure`) — must ship alongside the model; the aux vector is meaningless without the same scaling

**Both share:** `evaluation.py`'s `CEFR_ORDINAL` mapping — the hardcoded full A1–C2 schema is what guarantees correct behavior for an unseen C2 learner in production, independent of which model serves the request.

## 2. Latency and cold start — measured, not projected

| | Part 2 (XGBoost) | Part 3 (Transformer) |
|---|---|---|
| Steady-state median, single request | ~18ms | ~4ms (clean) / up to tens of ms under concurrent load |
| Cold start (model load, once per process) | ~1.9–3s (dominated by the MiniLM load for relevance) | ~1.9–3.3s |
| Budget | 300ms | 300ms |
| Headroom | ~17x | ~60-75x (clean) |

Both clear the budget with wide margin — **latency is not the deciding factor between the two models**, consistent with what both notebooks already concluded independently. The cold-start cost is paid once per server process at startup, not per request, so it doesn't affect steady-state throughput; it does affect how fast a new server instance becomes ready, relevant to autoscaling response time under load.

**Throughput, worked from that latency (1000ms ÷ latency = req/s per core):** Part 2 ≈ 55 req/s/core, Part 3 ≈ 250 req/s/core. For an illustrative 10,000 req/s peak (not measured — no real traffic data in this project), that's ~182 cores for Part 2 vs. ~40 for Part 3 — a ~4.5x compute-cost gap that doesn't change the QWK-based recommendation, but is the concrete number that would matter if cost-per-request became the dominant concern at true scale.

## 3. Serving architecture

**Minimal viable path for either model:** a stateless HTTP service (FastAPI/Flask-equivalent) wrapping `build_features()` + `model.predict()` (Part 2) or the tokenizer + forward pass (Part 3), returning a score 0–4 plus the recommended app action (`re-prompt` / `correct` / `advance`, via `evaluation.action_of`). Stateless by design — no session state, no per-user memory — which is what makes horizontal scaling trivial at the "millions of learners" scale the brief names.

**Input validation worth building in, not optional:** reject or flag CEFR values outside the known 6-level schema, empty/malformed `asr_word_confidences`, and unexpected `target_language` codes — the notebooks handle the *known* edge cases (empty transcript, C2 unseen in training) correctly by construction, but a real service sees malformed input a clean dataset never does.

## 4. Cloud / edge / mobile — three different real constraints, not one generic answer

This is the fuller version of Part 3's section 9 discussion, now covering Part 2 as well.

### Cloud (the brief's actual target — "millions of learners")
Both models' bottleneck at this scale is **memory footprint per server process**, not request latency:
- Part 2: small model file, but the MiniLM embedding model (~470MB-class) is still loaded per process for `relevance_cosine_sim`
- Part 3: ~470MB fp32 model in memory per process

Neither model needs batching for correctness (both are stateless single-request scorers), so the standard answer applies: horizontal autoscaling on request volume, with instance count driven by memory-per-pod budgets more than CPU. A managed container platform (e.g., Kubernetes HPA, or a serverless container runtime with a floor of warm instances to absorb the ~2-3s cold start) is the boring, correct choice here — no exotic infrastructure needed given the latency headroom both models have.

### Edge (regional inference nodes closer to learners)
The case for edge over a well-placed cloud region is about **network round-trip time**, not compute — both models are so far under the latency budget that moving compute physically closer only helps if network latency (not measured in this dataset — no geographic/connection data available) turns out to be the dominant cost for the actual learner population. Both models are small enough (hundreds of MB) to replicate cheaply across regions if this turns out to matter; this is a "measure with real traffic first" question, not a build decision to make now.

### Mobile / on-device
- **Part 2** is the more mobile-appropriate model of the two as-is: no transformer forward pass needed for classical features other than the embedding call for relevance (which could be dropped or cached client-side if truly latency/resource-constrained on-device — recall Part 2's section 4 found `relevance_cosine_sim` carries little unique signal once length/confidence are controlled for, so dropping it for an on-device variant costs little).
- **Part 3** would need real work before shipping to mobile: the embedding table (82% of model size, per section 1) dominates on-device footprint. Options, not implemented here: vocabulary pruning (keep only tokens the 5 target languages actually use), a smaller distilled backbone, or ONNX → Core ML / LiteRT conversion (named in section 9, not built).
- **Given both models' current honest limitations** (Part 3's weaker QWK, its rejected uncertainty signal), **an on-device deployment should ship Part 2**, not Part 3, until/unless the transformer's data-volume-limited gap closes.
- **Update cadence** matters more for mobile than either model's raw size: an on-device model is bound to app-store release cycles, not instant redeploy — favors keeping scoring server-side with on-device caching of recent results unless full offline support (e.g., practicing without connectivity) is a hard product requirement.

## 5. Trust in production — not just a notebook metric

`false_praise` (Part 2: 0.167 / ensemble: similar / Part 3: 0.087) needs a production analog, not just an offline number:
- **Abstention as a real feature, not just an evaluation exercise:** Part 2's section 6 model-disagreement signal (XGBoost vs. Linear Regression, AUC 0.578) is the one uncertainty tool validated to work at all this session — worth wiring into the served path as a genuine triage mechanism: route the top-N% most-disagreed-on predictions to human review instead of auto-scoring them, exactly as tested offline.
- **Monitor `false_praise`-shaped events in production**, not just accuracy — specifically, track the rate of predicted-advance (score ≥3) on responses a sampled human-review process later flags as genuinely poor. This is the metric that would catch model drift the brief actually cares about, not overall accuracy drift.
- **Don't deploy Part 3's softmax-entropy as a confidence signal** — tested and found worse than random (AUC 0.46, Part 3's section 7). If a transformer-based abstention signal is wanted later, it needs recalibration (e.g., temperature scaling on held-out data) before being trusted for real triage decisions.

## 6. Rollout strategy

Given XGBoost is the recommendation and the transformer has one specific documented strength (`false_praise`):
1. **Ship Part 2 as the primary scorer.**
2. **Run Part 3 in shadow mode** (score every request with both models, serve only Part 2's result) for a real-traffic comparison before committing further engineering time — the current comparison is on one 404-row offline split; shadow traffic would validate whether the `false_praise` advantage holds at scale before deciding whether to invest in closing the transformer's QWK gap (per `REPORT.md`'s next-steps: more training data via label-preserving augmentation).
3. **If shadow-mode data confirms the trust advantage is real and reproducible**, consider a hybrid production policy: Part 2 scores by default, Part 3's prediction (or disagreement between the two) triggers human review — combining Part 2's stronger QWK with Part 3's demonstrated caution, without needing to pick one model exclusively.

## 7. What's explicitly not implemented — discussion-level per the brief

ONNX export for serving (named, not built — Part 3's section 9), static/QAT quantization (dynamic quantization tested and found not to help on this backend, per section 8 — static quantization is the more promising untried alternative), embedding-table vocabulary pruning for mobile, and the shadow-mode rollout itself. All are next steps, not gaps in the current evaluation — the brief asks for this to be a discussion, and each item here has a specific, evidence-backed reason it's the right next thing to try rather than a vague "future work" list.

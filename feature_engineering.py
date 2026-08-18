"""Shared feature engineering for Part 1 (`EDA_PART_01.ipynb`) and Part 2
(`Part_2_Classical_ML_Pipeline.ipynb`).

Both notebooks import this module rather than each defining their own copy.
That's the point of pulling it out: the two notebooks used to each inline
their own version of this logic, and they drifted apart (different feature
names, different ASR-failure token sets) without either notebook's author
noticing until the numbers stopped matching. An import makes that drift
structurally impossible instead of relying on someone remembering to keep
two copies in sync by hand.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

# Full CEFR schema per DATA_DICTIONARY.md, not just the levels observed in
# dataset.csv (C2 is valid but absent from training data).
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_ORDINAL = {lvl: i for i, lvl in enumerate(CEFR_LEVELS)}

# Valid target languages per DATA_DICTIONARY.md.
LANGUAGES = ["de", "en", "es", "fr", "it"]

# Filler/noise tokens -- identical across all 5 languages (not real
# vocabulary in any of them), confirmed empirically against dataset.csv:
# e.g. "no" appears 11x with mean score 0.18 when present, in line with the
# other filler tokens, not a legitimate standalone answer in this dataset.
# "the" is deliberately excluded -- it's a real English word elsewhere
# ("the story was really moving"); "the the the" disfluency is instead
# caught by repeat_word_ratio below.
ASR_FAILURE_TOKENS = {"uh", "um", "mmm", "brrr", "klk", "xxx", "...", "???", "no",
                       "*noise*", "[inaudible]"}

# Partial-disfluency markers -- distinct from is_asr_failure, which requires
# the WHOLE transcript to be filler.
DISFLUENCY_PATTERNS = {"uh": r"\buh+\b", "um": r"\bum+\b",
                        "imean": r"\bi mean\b", "youknow": r"\byou know\b"}


def parse_word_confidences(df: pd.DataFrame) -> pd.Series:
    """`asr_word_confidences` as parsed lists of floats, one per row."""
    return df["asr_word_confidences"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x.strip() else [])


def is_asr_failure(transcript: str) -> bool:
    """True only if EVERY token is a filler/noise token -- the transcript
    carries zero real content. The single definition used everywhere in
    both notebooks (no redefining this per-section)."""
    toks = transcript.strip().lower().split()
    return len(toks) > 0 and all(t in ASR_FAILURE_TOKENS for t in toks)


_EMBEDDER = None
_EMB_CACHE: dict[str, np.ndarray] = {}


def _embed_unique(texts: pd.Series) -> dict[str, np.ndarray]:
    """Encode each unique string once (10 prompts, <=291 transcripts) rather
    than once per row -- an in-memory cache computed fresh every run,
    standing in for the features_cache.parquet the original drafts of these
    notebooks depended on (and which no longer exists in this repo)."""
    global _EMBEDDER
    uncached = [t for t in texts.unique() if t not in _EMB_CACHE]
    if uncached:
        if _EMBEDDER is None:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vecs = _EMBEDDER.encode(uncached, show_progress_bar=False, normalize_embeddings=True)
        _EMB_CACHE.update(zip(uncached, vecs))
    return {t: _EMB_CACHE[t] for t in texts.unique()}


def relevance_cosine_sim(df_in: pd.DataFrame) -> np.ndarray:
    """Prompt<->transcript cosine similarity via a frozen multilingual
    sentence embedding. Prompts are English, transcripts are 5 languages --
    lexical overlap would be blind on ~80% of rows, so a language-agnostic
    embedding is what makes this feature work at all (see EDA_PART_01.ipynb
    Section 7)."""
    p_vecs = _embed_unique(df_in["prompt"])
    r_vecs = _embed_unique(df_in["asr_transcript"])
    return np.array([float(np.dot(p_vecs[p], r_vecs[t]))
                      for p, t in zip(df_in["prompt"], df_in["asr_transcript"])])


def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """The full 28-feature matrix from a raw dataset.csv-shaped DataFrame
    (just needs prompt / asr_transcript / asr_word_confidences / cefr_level
    / target_language columns -- no pre-parsing required by the caller).
    Safe for single-row serving. Every feature traces to one of the brief's
    three grading criteria -- see the rubric table in either notebook's
    feature-engineering section."""
    word_confs = parse_word_confidences(df_in)
    rows = []
    for r, confs in zip(df_in.itertuples(index=False), word_confs):
        text = (getattr(r, "asr_transcript", "") or "")
        toks = text.lower().split()
        confs = list(confs or [])
        n = len(toks)
        f = {}

        # -- Grammar / lexical --------------------------------------------
        f["n_tok"] = n
        f["n_char"] = len(text)
        f["log_n_tok"] = np.log1p(n)
        f["mean_word_len"] = float(np.mean([len(t) for t in toks])) if toks else 0.0
        f["type_token_ratio"] = len(set(toks)) / n if n else 0.0
        f["n_sent"] = sum(text.count(c) for c in ".?!;")
        if n >= 2:
            f["repeat_word_ratio"] = sum(1 for x, y in zip(toks, toks[1:]) if x == y) / (n - 1)
        else:
            f["repeat_word_ratio"] = 0.0

        # -- Intelligibility / ASR confidence distribution -----------------
        if confs:
            a = np.asarray(confs, dtype=float)
            f["conf_mean"] = float(a.mean()); f["conf_min"] = float(a.min()); f["conf_max"] = float(a.max())
            f["conf_std"] = float(a.std()); f["conf_p25"] = float(np.percentile(a, 25))
            f["conf_frac_lt50"] = float((a < 0.50).mean()); f["conf_frac_lt70"] = float((a < 0.70).mean())
            f["conf_worst3_mean"] = float(np.sort(a)[:3].mean())
        else:
            for k in ["conf_mean", "conf_min", "conf_max", "conf_std", "conf_p25",
                      "conf_frac_lt50", "conf_frac_lt70", "conf_worst3_mean"]:
                f[k] = 0.0

        f["is_asr_failure"] = float(is_asr_failure(text))
        f["is_empty"] = float(n == 0)
        f["is_ultrashort"] = float(n <= 2)
        n_dis = sum(len(re.findall(p, text.lower())) for p in DISFLUENCY_PATTERNS.values())
        f["disfluency_rate"] = n_dis / n if n else 0.0

        # -- Relevance & completeness (filled in below, batched) ----------
        f["relevance_cosine_sim"] = 0.0

        # -- Proficiency baseline / metadata -------------------------------
        cefr = getattr(r, "cefr_level", None)
        f["cefr_ordinal"] = float(CEFR_ORDINAL.get(cefr, np.nan))
        lang = getattr(r, "target_language", None)
        for lg in LANGUAGES:
            f[f"lang_{lg}"] = float(lang == lg)
        f["lang_other"] = float(lang not in LANGUAGES)

        # -- Interaction: token count x CEFR --------------------------------
        cefr_val = f["cefr_ordinal"] if not np.isnan(f["cefr_ordinal"]) else 2.0
        f["tok_x_cefr"] = f["log_n_tok"] * cefr_val

        rows.append(f)

    X = pd.DataFrame(rows, index=df_in.index)
    X["relevance_cosine_sim"] = relevance_cosine_sim(df_in)  # one batched pass, not per-row
    return X

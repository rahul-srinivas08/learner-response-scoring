# `dataset.csv` — data dictionary

2,000 rows, one per learner utterance.

| Column | Type | Description |
|---|---|---|
| `utterance_id` | string | Unique id, e.g. `utt_00042`. |
| `prompt` | string | The speaking prompt the learner was responding to. |
| `target_language` | string | ISO code of the language being learned (`es`, `fr`, `de`, `en`, `it`). |
| `cefr_level` | string | Learner's proficiency level: `A1`, `A2`, `B1`, `B2`, `C1`, `C2`. No `C2` rows exist in this dataset (see notes below). |
| `asr_transcript` | string | Automatic transcript of the spoken response. May be imperfect. |
| `asr_mean_confidence` | float | Mean of the word-level ASR confidences (0–1). |
| `asr_word_confidences` | JSON list | Per-word ASR confidences (0–1), as a JSON-encoded list of floats. |
| `human_score` | int (0–4) | Primary human rating of grammar, relevance & completeness, and intelligibility, scored from the audio (see the brief). **This is the target.** |
| `human_score_2` | int (0–4) or empty | Second independent human rating, present on ~15% of rows; empty otherwise. |

Notes:
- The two human ratings were produced independently by different raters.
- ASR is not perfect and, like any real pipeline, sometimes fails outright.
- `target_language` reflects the language the learner was speaking; transcripts
  are in that language. You do not need to be fluent in every language to do
  this task well — most useful signals don't require reading the transcript
  content directly.
- The language and CEFR-level distributions are not balanced. Treat this as
  you would a real production dataset.
- `C2` is listed as a valid `cefr_level` because it's part of the real CEFR
  scale the app supports, even though this particular dataset happens to have
  no `C2` rows — a model/pipeline built only around the levels *observed* in
  training data is a common production bug, and this is a small nudge toward
  handling the full label space, not just what's in `dataset.csv`.

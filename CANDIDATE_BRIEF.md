# Work Sample — Automated learner-response scoring

This is the **data-science** take-home for the role. It's designed for
**about 3-4 hours** across three parts — that's a guide to the intended scope,
not a deadline; spend what you need. We care far more about your judgment and
how you **evaluate** your model than about squeezing out performance. A simple
model you understand and assess honestly beats a fancy one you don't.

**We encourage you to use AI coding tools and agents** — build the way you would
on the job. In the interview we'll ask how you directed them and how you
validated what they produced, so use whatever setup shows you at your best.

## The problem

Our AI-powered language-learning app gives learners speaking prompts and needs to
score their spoken responses so it can decide whether to move on, correct the
learner, or re-prompt. You're given ~2,000 learner utterances, across several
target languages. Each row has the target prompt, an ASR (speech-recognition)
transcript of what the learner said, word-level ASR confidences, the learner's
CEFR level (A1–C2, though this dataset happens to have no C2 rows), the language being learned, and **one human rating 0–4** of
response quality (defined below). About **15%** of rows also have a **second**
independent human rating.

Build a model that predicts the human score, and evaluate it honestly.

## What the human score measures

Human raters listened to each spoken response and scored it on three things,
roughly equally weighted:

1. **Grammar** — is the response grammatically correct for what the learner was
   trying to say?
2. **Relevance & completeness** — does it actually answer the prompt, and how
   fully?
3. **Intelligibility** — could the rater understand it without effort?

It is **not** a pronunciation score.

## Structure: three parts

Work through these in order — each builds on the last.

1. **Analysis.** Explore the dataset before you model it. It's synthetic but
   built to resemble real learner data, mess included — multiple languages,
   an unbalanced class distribution, and some transcription errors. Show us
   what you found and how you'd act on it. A breakdown of any modeling result
   by language belongs here or in Part 2, not as an afterthought.
2. **Classical ML.** Build and evaluate a non-transformer model (or several,
   compared honestly) that predicts the human score.
3. **Transformer + engineering discussion.** Fine-tune a small, CPU-friendly
   transformer and compare it against Part 2's model. This part should also
   cover the engineering side we'd otherwise ask about separately: framework
   choice, memory/latency tradeoffs on CPU, and — at a discussion level, not
   an implementation requirement — how you'd think about deploying this
   (including to a mobile/on-device context).

## What to deliver

1. **Runnable code** (scripts or notebooks) for all three parts, with a line
   on how to run each.
2. **A short written report** — bullet points, not an essay. It should be
   **concise, focused, and easy to read**: something a colleague can skim in a
   few minutes and understand your reasoning. We'd rather read one sharp page
   than ten padded ones, so please don't hand us an unedited AI dump. Cover:
   - what you found in the data and what you did about it,
   - how you evaluated the model(s) and *why those metrics*,
   - what your number actually means (what's a good score here? what's the
     ceiling?),
   - why this model — what else you considered and what tipped the choice,
   - one thing you'd do next with more time,
   - a short **AI collaboration log**: which tools/agents you used, the key
     prompts or sessions, and at least one thing the AI got wrong and how you
     caught it. (If you chose not to use AI tools, a line on why is enough.)

## What we'll look at

No hidden criteria — this is what we read your submission for:

- **How you work with the data** — what you explored and processed, and how you
  *show* it: the few tables, plots, or numbers that carry your findings.
- **How you evaluate** — metrics that fit the problem, a baseline, and an honest
  read of the result.
- **How you choose a model** — the *why* behind it: alternatives you weighed,
  and how the constraints below shaped the choice.
- **How you communicate** — issues found, limitations admitted, next steps a
  teammate could pick up.
- **How you used your tools** — including AI agents: what you asked of them and
  how you checked the result.
- **How you think about engineering tradeoffs** — framework choice, latency,
  and deployment considerations for Part 3.

## Constraints worth keeping in mind

- **Latency:** at serving time the model must score a response in **under 300 ms
  on CPU**. Let that inform your model choice; you don't need a rigorous
  benchmark.
- **Throughput/scale:** this app serves millions of learners. A model that's
  fast per-request but expensive to run at that volume (cost, hardware, or
  operational complexity) is a real constraint — worth a line of discussion in
  Part 3, not a build requirement.
- **Trust:** the app is only useful if learners trust it — a confidently wrong
  "great job!" is worse than an honest "I'm not sure."
- **CPU only:** Part 3's transformer must run on a normal laptop CPU in
  reasonable time — pick small models and justify the choice.

Use any libraries or pretrained models; just note what you used.

## Data

[`dataset.csv`](dataset.csv) — see [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) for
the columns. It's synthetic but built to resemble real learner data, mess
included. **Look at it before you model.**

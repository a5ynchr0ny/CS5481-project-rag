# Results

Frozen from the 65-question stratified evaluation (`data/eval_set.json`,
seed = 42). Everything below is reproducible with the commands in
[README.md](README.md); per-question records are in `results/`.

## Setup

| | |
|---|---|
| Corpus | QASPER **dev**: 281 papers → 3,796 sections, 11,461 citation markers removed |
| Chunking A (structured) | 9,609 chunks — mean 147 tok, median 149, max 715 |
| Chunking B (sliding) | 3,203 chunks — 512 tok window / 64 overlap, mean 493 |
| Embedding | Qwen3-Embedding-4B fp16 (local, Ollama), 2,560-d, instruction-prefixed queries |
| Vector store | ChromaDB, cosine, metadata `paper_id / section_name / chunk_id` |
| Generation | DeepSeek chat, temperature 0, hard citation rules |
| Retrieval depth | top-5 (rerank runs fetch 24 then re-order to 5) |
| Indexing cost | A ≈ 12 min, B ≈ 4 min (13–14 chunks/s on an RTX 4090) |
| Rerank cost | 7–13 ms per (query, document) pair, 24 docs per request |

## Evaluation set (65 questions)

| Type | n | Source |
|---|---:|---|
| Extractive | 20 | QASPER dev, first annotator |
| Abstractive | 12 | QASPER dev |
| Boolean | 8 | QASPER dev |
| Unanswerable | 20 | QASPER dev (`unanswerable: true`, 16.2% of dev) |
| Out-of-KB | 5 | questions from **test-split** papers never indexed |

Questions whose evidence is **only** figure/table captions (`FLOAT SELECTED`, 9% of
evidence strings) are excluded, matching the official evaluator's
`text_evidence_only` mode.

## Metric definitions

* **Hit@5** — a retrieved chunk **contains a gold evidence string in full**
  (case/punctuation-normalised substring). Stricter than paragraph-coverage
  definitions: a chunk holding *most* of the evidence does not count.
* **EvidenceRecall@5** — fraction of a question's gold evidence strings found
  anywhere in the top-5.
* **MRR** — mean reciprocal rank of the first chunk containing a gold evidence string.
* **Answer F1** — official QASPER `token_f1_score`, citations stripped, max over
  annotators; extractive and abstractive questions only (boolean scored separately).
* **Abstention accuracy** — fraction of the 25 negatives (20 unanswerable + 5
  out-of-KB) correctly refused.
* **Refusal precision / recall** — precision = refusals that were genuinely
  unanswerable; recall = negatives that were refused.
* **Citation rate / invalid citations** — answers containing at least one `[n]` /
  `[n]` values that do not resolve to a retrieved chunk.
* **Hallucination rate** — answered negatives ÷ all negatives (= 1 − abstention accuracy).

Refusal uses a **per-configuration threshold** τ = 10th percentile of answerable
top-1 scores, plus a prompt-level refusal instruction. A shared τ would bias the
comparison: sliding-window scores sit systematically lower than structured-chunk
scores.

## Main table — no reranking

| | A / paper | B / paper | A / global | B / global |
|---|---:|---:|---:|---:|
| **Hit@5** | **0.725** | **0.725** | 0.250 | 0.300 |
| Hit@3 / Hit@1 | 0.575 / 0.250 | 0.525 / 0.125 | 0.225 / 0.100 | 0.200 / 0.000 |
| EvidenceRecall@5 | 0.634 | 0.665 | 0.221 | 0.281 |
| MRR | **0.436** | 0.337 | 0.160 | 0.102 |
| Gold paper in top-5 | 1.00 | 1.00 | 0.575 | 0.500 |
| Answer F1 (n answered) | 0.228 (12) | 0.228 (15) | 0.240 (12) | 0.285 (15) |
| Boolean accuracy | 0.000 (1) | 0.500 (2) | 0.000 (1) | — (0) |
| Abstention accuracy | 0.84 | 0.84 | 0.84 | 0.84 |
| Refusal precision | 0.583 | 0.636 | 0.438 | 0.467 |
| Hallucination rate | 0.16 | 0.16 | 0.16 | 0.16 |
| Citation rate / invalid | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 |
| τ | 0.554 | 0.388 | 0.554 | 0.488 |

Per-type retrieval (Hit@5):

| Type | A / paper | B / paper | A / global | B / global |
|---|---:|---:|---:|---:|
| Extractive (n=20) | 0.850 | 0.900 | 0.250 | 0.400 |
| Abstractive (n=12) | **0.917** | 0.667 | 0.417 | 0.333 |
| Boolean (n=8) | 0.125 | 0.375 | 0.000 | 0.000 |

## Reranking — Qwen3-Reranker-0.6B, paper-scoped, strategy A

Fetch top-24 by embedding, re-order with the cross-encoder, keep top-5.

| | Baseline | + Reranker | Δ |
|---|---:|---:|---:|
| **Hit@5** | 0.725 | **0.875** | **+0.150** |
| Hit@3 | 0.575 | **0.725** | +0.150 |
| Hit@1 | 0.250 | **0.300** | +0.050 |
| EvidenceRecall@5 | 0.634 | **0.789** | +0.155 |
| MRR | 0.436 | **0.512** | +0.076 |
| Answer F1 (n) | 0.228 (12) | **0.284 (21)** | +0.056 |
| Extractive Hit@5 / F1 | 0.850 / 0.238 | **0.950 / 0.298** | +0.100 / +0.060 |
| Abstractive Hit@5 / F1 | 0.917 / 0.203 | 0.833 / **0.243** | −0.084 / +0.040 |
| Abstention accuracy | 0.84 | **0.92** | +0.080 |
| Hallucination rate | 0.16 | **0.08** | −0.080 |
| Refusal precision | 0.583 | 0.590 | +0.007 |
| Citation rate / invalid | 1.00 / 0 | 1.00 / 0 | — |
| τ | 0.554 | 0.265 | — |

Rerank scores are bimodal (min 0.012, max 1.000 across answerable questions), so τ
must be recalibrated per configuration — this is why the rerank run uses 0.265.

## Miss taxonomy — where the 40 answerable questions fail

`recompute_metrics.py`/`diagnose_hits.py` classify every non-hit:

* **paper-miss** — the gold paper is not in the top-5 at all
* **rank-miss** — the gold paper is there, but the evidence chunk ranks below 5
* **split-miss** — the evidence chunk is there but the evidence string was cut by
  the chunk boundary (first 10 tokens present, full string absent)
* **no-evidence** — the annotation carries no text evidence

| Config | Misses | paper | rank | split | no-ev |
|---|---:|---:|---:|---:|---:|
| A / paper | 11 | 0 | **9** | 0 | 2 |
| B / paper | 11 | 0 | 2 | **7** | 2 |
| A / global | 30 | **17** | 10 | 1 | 2 |
| B / global | 28 | **20** | 4 | 2 | 2 |
| A / paper + rerank | **5** | 0 | **3** | 0 | 2 |

This is the clearest finding in the study: **the two chunkers fail in different
ways.** Structured chunks fail by *ranking* (the evidence is inside the paper's
retrieved block but ranked 6–20); sliding windows fail by *splitting* (the window
boundary cuts the annotated sentence in half). Reranking fixes exactly the
ranking failure — rank-misses 9 → 3, Hit@5 0.725 → 0.875 — and cannot fix
splitting, which is structural.

## Retrieval scope is the single largest lever

Paper-scoped retrieval (the "user is reading this paper" setting) lifts Hit@5 from
**0.25–0.30 to 0.725** on the same corpus and questions. Global retrieval over
9,609 chunks misses the correct paper in 42–50% of questions; no amount of chunk
tuning repairs that. Both settings are reported because the gap is the interesting
result, not a defect to hide.

## Limitations

* **Boolean questions are mostly refused.** The per-config τ (10th percentile of
  answerable scores) lands above the short, low-lexical-overlap boolean queries:
  8 boolean questions yield only 1–3 answers, with 0–1 correct. Boolean handling
  needs its own threshold or a classification-style prompt; we report rather than
  hide it.
* **Answer F1 understates answer quality.** QASPER gold answers are often a single
  phrase, and token-F1 punishes the long, cited, explanatory answers the prompt
  asks for (an answer containing the gold phrase plus surrounding explanation
  scores ≈ 0.35). A manual semantic review — as done in the companion
  implementation (strict span 30–40% vs manual 75–80%) — is the right companion
  metric and is the next step for this system.
* **The 4B reranker is not in these numbers.** Two community Q8_0 conversions
  returned all-zero scores; the `ggml-org` 0.6B conversion is the only verified
  one. Reranking is therefore reported at 0.6B, and the 4B comparison is left as
  future work pending a trustworthy GGUF.
* **65 questions is a small sample.** Per-type cells (especially boolean, n = 8)
  carry wide error bars; differences below ~0.08 Hit@5 should be treated as ties.

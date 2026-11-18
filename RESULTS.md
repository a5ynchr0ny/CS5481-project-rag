# Results

Frozen from the 65-question stratified evaluation (`data/eval_set.json`,
seed = 42). Everything below is reproducible with the commands in
[README.md](README.md); per-question records are in `results/`.

> `results/summary.json` holds the **last** `evaluate.py` run (the rerank
> configuration). The four baseline configurations are stored per-question in
> `results/results_{A,B}_{paper,global}.json`; their aggregates are the main table
> below.

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

## Reranking — Qwen3-Reranker-0.6B, all four configurations

Fetch the top-24 candidates by embedding, re-order with the cross-encoder, keep top-5.

| | A / paper | B / paper | A / global | B / global |
|---|---:|---:|---:|---:|
| Hit@5 — baseline → rerank | 0.725 → **0.875** | 0.725 → 0.675 | 0.250 → **0.325** | 0.300 → 0.300 |
| Δ Hit@5 | **+0.150** | −0.050 | **+0.075** | 0.000 |
| Hit@3 | 0.725 | 0.625 | 0.300 | 0.250 |
| Hit@1 | 0.300 | 0.225 | 0.125 | 0.075 |
| EvidenceRecall@5 | 0.789 | 0.608 | 0.296 | 0.271 |
| MRR — baseline → rerank | 0.436 → **0.512** | 0.337 → **0.403** | 0.160 → **0.209** | 0.102 → **0.163** |
| Δ MRR | +0.076 | +0.066 | +0.049 | +0.061 |
| Answer F1 | 0.284 (21) | 0.234 (23) | 0.233 (14) | 0.234 (13) |
| Abstention accuracy | 0.92 | 0.84 | 0.84 | 0.76 |
| Hallucination rate | **0.08** | 0.16 | 0.16 | 0.24 |
| Refusal precision | 0.590 | 0.583 | 0.457 | 0.432 |
| Citation rate / invalid | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 |
| Misses (was) | **5** (11) | 13 (11) | 27 (30) | 28 (28) |
| τ | 0.265 | 0.066 | 0.927 | 0.859 |

### Reranking is not free: it helps fine-grained chunks and hurts when evidence is split

Three patterns hold across the grid:

1. **Ordering always improves.** MRR rises in all four configurations (+0.049 to
   +0.076). When the reranker keeps a hit, it puts it higher.
2. **Hit@5 only improves where chunks are small and self-contained.** Structured
   chunks (A) gain +0.150 paper-scoped and +0.075 globally. Sliding windows (B) lose
   0.050 paper-scoped and stay flat globally.
3. **Confidence calibration can invert.** Abstention improves only where retrieval
   improved (A/paper 0.84 → 0.92, hallucination 16% → 8%); for B/global the
   reranker's confidently-high scores on irrelevant windows *raise* false answers
   (hallucination 16% → 24%).

The mechanism is visible in the miss taxonomy: B/paper keeps **7 split-misses
before and after reranking** — those are chunks whose evidence was cut by the
window boundary, and no re-ordering can restore text that is not there. Meanwhile
the cross-encoder prefers a window that is *entirely* on-topic over a 512-token
window that contains the evidence sentence alongside three unrelated paragraphs, so
it demotes some evidence-bearing windows: B/paper's misses rise 11 → 13 even as its
MRR improves.

**Practical rule from this grid:** rerank when chunks are semantically focused
(structured / small); with long fixed windows, fix the chunker first (or add
sentence-level splitting), then rerank.

### Global retrieval is a candidate-generation problem, not a ranking problem

Global paper-hit is 0.575 (A) / 0.500 (B) at depth 5 and 0.600 / 0.550 at depth 24
with reranking. Roughly 40–45% of questions never see their own paper in the
candidate list, so no re-ordering stage can help them — the remaining global misses
are dominated by paper-misses (16 of 27 for A/global).

Quadrupling the rerank depth does **not** fix it:

| A / global, rerank depth | Hit@5 | paper-hit | MRR | Misses (paper / rank) |
|---|---:|---:|---:|---:|
| 24 | 0.325 | 0.600 | 0.209 | 27 (16 / 8) |
| 100 | 0.325 | 0.575 | 0.181 | 27 (17 / 7) |

The gold paper simply is not in the bi-encoder's top-100 for ~42% of questions —
widening the candidate pool only adds noise for the reranker. This rules out the
intuitive "just retrieve deeper" fix.

A likely cause is the QASPER question-collection protocol: questions were written by
NLP practitioners who saw **only the title and abstract**, while the annotated
evidence usually sits in the body (methods/results). The question therefore matches
the paper's framing, not the passage that answers it. Paper-scoped retrieval — the
"user is reading this paper" setting — bypasses exactly this mismatch, which is why
it gains ~0.45 Hit@5 and why we treat it as the primary setting rather than a
convenience.

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
| B / paper + rerank | 13 | 0 | 4 | **7** | 2 |
| A / global + rerank | 27 | **16** | 8 | 1 | 2 |
| B / global + rerank | 28 | **18** | 7 | 1 | 2 |
| A / global + rerank, depth 100 | 27 | **17** | 7 | 1 | 2 |

This is the clearest finding in the study: **the two chunkers fail in different
ways.** Structured chunks fail by *ranking* (the evidence is inside the paper's
retrieved block but ranked 6–20); sliding windows fail by *splitting* (the window
boundary cuts the annotated sentence in half). Reranking fixes exactly the
ranking failure — A/paper rank-misses 9 → 3, Hit@5 0.725 → 0.875 — and leaves the
split-misses at 7 for B, because that damage is structural.

## Retrieval scope is the single largest lever

Paper-scoped retrieval (the "user is reading this paper" setting) lifts Hit@5 from
**0.25–0.30 to 0.725** on the same corpus and questions, before any reranking.
Global retrieval over 9,609 chunks keeps the correct paper out of the top-24
candidates for 40–45% of questions; no re-ordering stage repairs that. Both settings
are reported because the gap is the interesting result, not a defect to hide.

## Limitations

* **Reranking is not a universal upgrade.** It lifts Hit@5 by +0.150 paper-scoped
  and +0.075 globally on structured chunks, but *lowers* it by 0.050 on 512-token
  sliding windows and raises B/global's hallucination rate from 16% to 24%. The
  mechanism (split evidence + holistic relevance preference) is analysed above;
  it should be reported as a conditional finding, not as "rerank works".
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
  Under that rule, A/paper's +0.150 and A/global's +0.075 sit at or just under the
  threshold — the B/paper drop (−0.050) should be called a tie, not a loss.

# Comparison with PaperScope (same group, same assignment)

Two independent implementations of CS5481 Topic 5 exist in this group:
this repository (**281-paper corpus, reranking, retrieval-scope ablations**) and
[PaperScope](https://github.com/Kevin999-sudo/paperscope-cs5481) (**20-paper corpus,
UI + statement checking, manual answer review**). They answer the same assignment
with different trade-offs; this note compares them honestly.

> **Read this before comparing any numbers.** The two systems define *Hit@5* and
> *answer correctness* differently. Cross-referencing the headline numbers without
> the definitions is meaningless — the tables below therefore keep the definitions
> attached.

## Side by side

| | PaperScope | This repo |
|---|---|---|
| Corpus | 20 QASPER dev papers | **281 QASPER dev papers** (14×) |
| Chunking | fixed 240 tok / 32 overlap; paragraph-aware (same cap) | A: structured sections, 100–450 words, 9,609 chunks<br>B: sliding 512 tok / 64 overlap, 3,203 chunks |
| Embedding | BGE-small-en-v1.5, 33M params, **CPU** | Qwen3-Embedding-4B fp16, 4B params, **GPU** (via Ollama) |
| Vector store | Qdrant | ChromaDB |
| Retrieval scope | paper-scoped only | **paper-scoped *and* global** (ablation) |
| Reranking | none | **Qwen3-Reranker-0.6B** cross-encoder, 7–13 ms/pair |
| Generation | DeepSeek Flash | DeepSeek chat, temperature 0 |
| Eval size (QA) | 24 questions (20 answerable + 4 out-of-scope) | **65 questions** (20 extractive + 12 abstractive + 8 boolean + 20 unanswerable + 5 out-of-KB) |
| Extra evaluation | 40 claim-verification statements (20 support / 12 contradict / 8 insufficient) | per-type breakdown, Hit@1/@3, EvidenceRecall, miss taxonomy, rerank ablation |
| UI | **Streamlit app** | CLI scripts |
| Engineering artifacts | 208 frozen outcomes, reproducible fingerprint, latency + token accounting | per-question records, summary.json, reproducible seed-42 eval set |

## Results, with definitions attached

| Metric | PaperScope | This repo | Comparable? |
|---|---:|---:|---|
| Hit@5 (paper-scoped) | 0.70 / 0.70 | 0.725 (A) / 0.725 (B) → **0.875** (A + rerank), 0.675 (B + rerank) | **Only roughly.** PaperScope counts a hit when ≥50% of a gold *paragraph* is covered by the **union** of the top-5 chunks; we require the full gold *evidence string* to sit inside **one** chunk. Their definition tolerates evidence split across chunks; ours does not. |
| Evidence recall | 0.675 / 0.650 | 0.634 / 0.665 (0.789 best with rerank) | Roughly — both are coverage-style, ours over evidence strings, theirs over paragraphs. |
| Strict answer accuracy | 0.30 / 0.40 | 0.228 / 0.228 token-F1 (0.284 with rerank) | **No.** Strict span match (binary, exact) vs token-F1 (graded, penalises long answers). |
| Manual semantic review | **0.75 / 0.80** (19–20 / 24 incl. refusals) | not performed | — |
| Refusal | 4/4 | 0.84 accuracy on 25 negatives (0.92 with rerank) | **No.** 4 hand-made clearly-unrelated questions vs 20 official unanswerable + 5 out-of-KB. |
| Citation validity | 0 errors (fixed) / 1 response blocked (section) | 100% citation rate, 0 invalid `[n]` | Yes — both validate that every citation resolves to a retrieved passage. |
| Latency | median 1.06–1.32 s/query | rerank 7–13 ms/pair; generation not separately profiled | Partial — ours lacks an end-to-end latency table. |
| Chunking conclusion | paragraph-aware slightly better (manual 80% vs 75%) | structured better on MRR (0.436 vs 0.337) and abstractive Hit@5 (0.917 vs 0.667); tied on Hit@5 | Complementary: ours is a retrieval-geometry result, theirs an answer-quality result. |

### Why our answer metrics look worse

PaperScope reports both strict scoring (30%/40%) and a manual semantic review
(75%/80%) — a 2× gap. Our token-F1 (0.228) sits in the same "strict" family and
would very likely rise similarly under a semantic review: QASPER gold answers are
often a single phrase, while our prompt requires a cited explanatory sentence, so an
answer that *contains* the gold phrase plus context scores around 0.35. The metric
choice, not answer quality, drives most of the difference. **Adopting their manual
review protocol is the highest-value next step for this system.**

### Where each is stronger

**PaperScope** — product completeness: a working UI, statement verification against
the paper (beyond the five required features), latency and token accounting, a
frozen 208-case outcome file, and a manual review that keeps automated metrics in
perspective.

**This repo** — system depth: a 14× larger knowledge base, a verified cross-encoder
reranking stage (+0.150 Hit@5 on structured chunks, rank-misses 9 → 3, hallucination
16% → 8%; neutral to negative on 512-token windows, which is itself a reportable
result), an explicit paper-scoped vs global retrieval ablation (0.725 vs 0.25–0.30),
a 65-question evaluation with 20 genuinely unanswerable and 5 out-of-KB questions,
the official QASPER token-F1, and a miss taxonomy that explains *why* each chunker
fails (rank-miss vs split-miss).

## What a merged report should take from each

1. **PaperScope's UI and statement checking** as the demo surface — it makes
   "verifiable citations" visible in a way a CLI cannot.
2. **This repo's reranking stage and retrieval-scope ablation** as the experimental
   core — they are the only measurements in the group that move Hit@5 by a large,
   attributable margin.
3. **PaperScope's manual-review protocol** for answer quality, run over this repo's
   20 extractive questions, so the two systems report the same second axis.
4. **A single, stated Hit@5 definition** in the joint report. Recommended: the
   coverage definition (theirs) because it does not depend on the chunker — and then
   report our 0.725/0.875 re-scored under it.

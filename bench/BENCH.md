# Full-scale retrieval benchmark — 823 questions × 8 configurations

`full_bench.json` holds per-question retrieval flags for **every answerable
QASPER dev question with text evidence** (n = 823 of 1,005; unanswerable and
figure-only-evidence questions excluded), evaluated on eight retrieval
configurations at top-5. `full_retrieval_bench.py` reproduces the run
(~5 min on an RTX 4090; embeddings + Chroma queries + optional reranking,
no LLM calls).

Metric: **strict containment** — the complete normalized gold evidence string
must appear inside a single retrieved chunk (the same definition as
[RESULTS.md](../RESULTS.md)).

## Main table (n = 823)

| Config | Hit@1 | Hit@3 | Hit@5 | 95% CI | MRR | EvidenceRecall@5 | Paper-hit |
|---|---:|---:|---:|---|---:|---:|---:|
| **ours A + rerank24** | **0.373** | **0.648** | **0.770** | [0.740, 0.798] | **0.523** | 0.710 | 1.000 |
| ours B + rerank24 | 0.334 | 0.639 | 0.748 | [0.718, 0.777] | 0.492 | 0.698 | 1.000 |
| ours B (512 tok) | 0.326 | 0.621 | 0.725 | [0.694, 0.755] | 0.480 | 0.681 | 1.000 |
| ours A (147 tok) | 0.335 | 0.597 | 0.717 | [0.685, 0.747] | 0.475 | 0.659 | 1.000 |
| PaperScope section + BGE | 0.253 | 0.469 | 0.594 | [0.560, 0.627] | 0.376 | 0.523 | 1.000 |
| PaperScope fixed 240/32 + BGE | 0.194 | 0.363 | 0.445 | [0.411, 0.479] | 0.287 | 0.379 | 1.000 |
| ours A global | 0.109 | 0.197 | 0.255 | [0.227, 0.286] | 0.161 | 0.227 | 0.465 |
| ours B global | 0.092 | 0.187 | 0.233 | [0.206, 0.263] | 0.144 | 0.211 | 0.417 |

*ours A/B = our structured / sliding-window chunking with Qwen3-Embedding-4B;
PaperScope = the companion implementation's chunker with BGE-small; rerank24 =
Qwen3-Reranker-0.6B over the top-24 candidates (paper-scoped throughout except
the last two rows).*

## Paired significance (McNemar exact, Hit@5)

| Comparison | Gained / Lost | p |
|---|---:|---:|
| ours A vs PaperScope section | +169 / −68 | **4.2e-11** |
| ours B vs PaperScope section | +192 / −84 | 6.9e-11 |
| PaperScope section vs fixed | +147 / −24 | 9.2e-23 |
| ours A vs PaperScope fixed | +276 / −52 | 5.2e-38 |
| ours A vs ours B | +124 / −131 | 0.71 (tie) |
| ours A vs ours A + rerank | +51 / −95 (rerank gains 95, loses 51) | **3.4e-04** |
| ours B vs ours B + rerank | +51 / −32 (rerank gains 51, loses 32) | 0.048 |
| ours A + rerank vs PaperScope section | +177 / −32 | 1.6e-25 |

## Findings that a 65-question sample got wrong

The same configurations evaluated on a 65-question subset (see
[RESULTS.md](../RESULTS.md)) produced two impressions that do not survive the
full sample:

1. **"Reranking hurts sliding-window retrieval" — an artefact.** At n = 65 the
   B + rerank delta was −0.050; at n = 823 it is **+0.023 and significant
   (p = 0.048)**. Reranking helps both chunkers, just more on structured chunks
   (+0.053 for A).
2. **"+0.150 from reranking" was inflated.** The true A/paper effect is
   **+0.053** (0.717 → 0.770). Small-sample winner's-curse: the subset happened
   to contain many of the questions reranking fixes.

## Effects that do hold at scale

With n = 823 the per-factor picture becomes clean, ordered by effect size:

| Factor | Δ Hit@5 | p | Note |
|---|---:|---:|---|
| Retrieval scope (paper vs global) | +0.47…+0.49 | ≪ 1e-100 | the dominant lever |
| Chunker sanity (fixed-240 → paragraph-aware, BGE) | +0.149 | 9.2e-23 | fixed-240 collapses at scale (0.445) |
| Embedding model (BGE-small → Qwen3-4B, paragraph-aware) | +0.123 | 4.2e-11 | significant once the chunker is sane |
| Reranking (0.6B, depth 24) | +0.023…+0.053 | 0.048 / 3.4e-04 | real but the smallest factor |
| Chunker choice within ours (A vs B) | — | 0.71 | tie at every depth |

## Caveats

* Strict containment favours larger chunks (they contain more of the evidence
  string); compare chunkers of similar size, or add a coverage-style metric for
  cross-size conclusions.
* The PaperScope stack is evaluated with our corpus and our metric on purpose
  (controlled comparison); its published 20-question numbers use their own
  paragraph-coverage definition and are not comparable row-by-row.
* Query embeddings are computed once per model and shared across configurations,
  so embedding noise is paired, not independent — which is what makes the paired
  tests appropriate.

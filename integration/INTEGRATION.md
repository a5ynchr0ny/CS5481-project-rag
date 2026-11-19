# Putting this reranker behind someone else's retrieval

Short answer: **yes — it is a pure `(query, [passage texts]) → [scores]` function,
so it can be inserted after any retrieval stage.** This note records the exact
contract, the measured effect on PaperScope's own benchmark, and the one
structural caveat that decides whether you will see a gain.

## Interface

```
start server (Ollama's bundled llama.cpp build, no torch):
  llama-server.exe -m qwen3-reranker-0.6b-q8_0.gguf --rerank -ngl 99 -c 16384 ^
                   -b 512 -ub 512 --host 127.0.0.1 --port 8012

POST /v1/rerank
{"query": "...", "documents": ["...", "..."], "top_n": 20,
 "instruction": "Given a user question about a scientific paper, retrieve the passages from that paper that help answer the question."}
→ {"results": [{"index": 3, "relevance_score": 0.91}, ...]}
```

`scripts/rerank.py` wraps that call (`RERANK_URL` env var) and is what the
`rerank_n` parameter of `retrieve()` uses: fetch `n` candidates, re-order, keep
`top_k`.

## Adapter into a `paperscope`-style pipeline

Their `PaperScope.retrieve(paper_id, question, strategy, top_k=5)` already
fetches passages; the only change is to fetch a wider pool and re-order:

```python
def retrieve(self, paper_id, question, strategy, top_k=5, rerank_n=20):
    hits = self.search(paper_id, question, strategy, limit=max(top_k, rerank_n))
    if rerank_n:
        scores = rerank(question, [h["text"] for h in hits])   # our /v1/rerank
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda p: -p[0])]
    return hits[:top_k]
```

Two requirements: (1) the candidate **texts** must be available at that point —
PaperScope's archived `outcomes.json` stores only chunk ids and char offsets, so
either expose texts or rebuild them from the corpus + tokenizer; (2) the
`instruction` string should match the embedding-side task description.

## Measured on PaperScope's benchmark (20 answerable questions, their metric)

We rebuilt PaperScope's chunk texts offline (their chunker is deterministic:
QASPER + BGE tokenizer offsets; 240/240 stored chunk ids resolved) and re-ordered
**their stored top-5** with Qwen3-Reranker-0.6B. Scoring uses PaperScope's own
rule — a hit means ≥50% of a gold paragraph is covered by the retrieved chunk's
character range.

| Strategy | Hit@5 | Hit@1 before | Hit@1 after | MRR before | MRR after | top-1 changed |
|---|---:|---:|---:|---:|---:|---:|
| fixed 240/32 | 0.70 (14/20) | 0.35 (7/20) | **0.40 (8/20)** | 0.467 | **0.508** | 12 / 20 |
| paragraph aware | 0.70 (14/20) | 0.35 (7/20) | **0.45 (9/20)** | 0.485 | **0.552** | 9 / 20 |

**Hit@5 cannot move in this experiment**: the reranker only re-orders the five
passages PaperScope already kept, and Hit@5 is a set property. The gain is in
Hit@1 / MRR — i.e. the right passage now arrives first, which also helps the
generator.

## The rule this implies: widen the pool before reranking

A reranker cannot recover evidence that the first stage never retrieved. In our
own 65-question grid the miss taxonomy separates the two cases directly:
rank-misses (evidence retrieved but ranked below 5) fall 9 → 3 with reranking,
while split-misses (evidence cut by the chunk boundary) stay at 7 — reordering
cannot restore text that was never in a chunk.

The same holds here, and it is visible in the numbers: re-ordering PaperScope's
top-5 yields +1 to +2 questions on Hit@1 and **no change in Hit@5**, whereas
retrieving 20 candidates and then reranking to 5 (as `rauki34/paperscope-cs5481-repro`
does with a MiniLM cross-encoder) lifts their Hit@5 from 14/20 to **17/20**
(fixed) and **18/20** (paragraph aware) at ~0.4 s per case.

So, in order of leverage:

1. make the chunker produce self-contained passages (fix split-misses, drop stub
   chunks — also the `ISS-001` finding in the repro repo);
2. widen the candidate pool to 20–24;
3. only then choose the reranker model — the pool width matters much more than
   which cross-encoder scores it.

## Reproducing the table

```sh
# chunks rebuilt from the corpus (tokenizer only, no torch needed)
python rebuild_ps_chunks.py <paperscope-repo> <qasper-dev-v0.3.json> /tmp/ps_chunks.json
# rerank server reachable at RERANK_URL (default 127.0.0.1:8012)
python rerank_after_peers.py
```

Both scripts live next to this note and read the peer repo's
`results/cases.json` + `results/outcomes.json`.

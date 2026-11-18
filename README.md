# CS5481 Project — RAG KB-QA with Verifiable Citations

Knowledge-base question answering over the **QASPER** dev corpus (281 NLP papers,
1,005 questions) in the Retrieval-Augmented Generation pattern: two chunking
strategies → instruction-tuned local embeddings → vector retrieval → DeepSeek
generation where **every claim carries an inline `[n]` citation into the retrieved
chunks**, and the system **refuses to answer** when the evidence is insufficient.

Built for CS5481 (Data Engineering) Topic 5. All numbers in [RESULTS.md](RESULTS.md)
are frozen from a 65-question stratified evaluation; a companion implementation by a
teammate is compared in [COMPARE.md](COMPARE.md).

## Architecture

```
QASPER dev JSON (281 papers)
  → clean (strip BIBREF / SECREF / TABREF / FIGREF / INLINEFORM / DISPLAYFORM markers)
  → chunk A: structured sections, merge <100-word paragraphs, 450-word cap   (9,609 chunks)
    chunk B: fixed sliding window, 512 tokens / 64 overlap                   (3,203 chunks)
  → embed with Qwen3-Embedding-4B (fp16, local via Ollama) → ChromaDB
        metadata: paper_id / section_name / chunk_id

question → embed WITH task instruction prefix  ("Instruct: ...\nQuery: ...")
         → top-k from Chroma (paper-scoped or global)
         → [optional] Qwen3-Reranker cross-encoder re-ordering via /v1/rerank
         → DeepSeek chat: context + hard citation rules + fixed refusal sentence
         → answer + [n] citations → post-validation (every [n] must resolve to a chunk)
```

## Repository layout

```
scripts/
  prepare_corpus.py     cleaning + both chunkers; exports clean_text reused by the evaluator
  build_index.py        embed chunks via Ollama /api/embed into Chroma collections
  retrieve.py           retrieval: instruction-prefixed queries, paper filter, rerank hook
  answer.py             grounded generation, citation post-validation, threshold refusal
  evaluate.py           grid runner: --scope paper,global --strategies A,B --rerank N
  make_eval_set.py      stratified 65-question eval set (seed = 42)
  recompute_metrics.py  offline metric enrichment from saved per-question records
  diagnose_hits.py      miss taxonomy (paper-miss / rank-miss / split-miss)
  rerank.py             rerank backends: llama-server (default) / transformers
  download_qasper.py    fetch QASPER v0.3 from the official AllenAI S3 bucket
  hf_get_file.py        single-file HF download helper (GGUF models, hf-mirror aware)
  qasper_evaluator.py   official QASPER evaluator (AllenAI), used for answer token-F1
data/eval_set.json      65-question evaluation set (derived from QASPER, CC BY 4.0)
results/                frozen per-question records + summary.json (see RESULTS.md)
```

## Reproducing

Requires Python 3.12+, [Ollama](https://ollama.com) on a CUDA GPU, and a DeepSeek
API key in `DEEPSEEK_API_KEY`.

```sh
# 0. dependencies + embedding model
python -m pip install -r requirements.txt
ollama pull qwen3-embedding:4b-fp16        # 8 GB; q8_0 (4.3 GB) also works

# 1. data (QASPER v0.3, ~60 MB)
python scripts/download_qasper.py

# 2. chunk (pulls the Qwen3 tokenizer for exact token counting)
python scripts/prepare_corpus.py           # -> data/chunks/chunks_{A,B}.jsonl

# 3. index
python scripts/build_index.py --chunks data/chunks --db data/chroma

# 4. ask
export DEEPSEEK_API_KEY=sk-...
python scripts/answer.py "what ner models were evaluated?" --strategy A

# 5. evaluate (writes summary.json + per-question records under results/)
python scripts/evaluate.py --eval-set data/eval_set.json --db data/chroma --out results \
                           --scope paper,global --strategies A,B
```

`build_index.py` embeds documents **without** an instruction; `retrieve.py` prepends
the task instruction to the *query* only, as required by the Qwen3-Embedding model
card (omitting it costs 1–5 retrieval points).

### Optional: cross-encoder reranking without torch

The obvious route was `sentence-transformers` + a CUDA torch wheel; on the target
machine that download was not viable. Instead we use the **llama.cpp server binary
that ships inside the Ollama installation**, which already implements reranking:

```bat
:: <Ollama>\lib\ollama\ contains llama-server.exe, ggml-*.dll and cuda_v12\*.dll
:: copy them into one folder, then:
llama-server.exe -m qwen3-reranker-0.6b-q8_0.gguf --rerank -ngl 99 -c 16384 ^
                 -b 512 -ub 512 --host 127.0.0.1 --port 8012
:: POST /v1/rerank {"query": ..., "documents": [...], "instruction": ...}
```

Then run `scripts/rerank.py` with `RERANK_URL=http://127.0.0.1:8012/v1/rerank`.
The GGUF is `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` (fetch with
`scripts/hf_get_file.py`); measured at 7–13 ms per (query, document) pair on an
RTX 4090, 24 documents per request.

Two findings worth recording:

* **Ollama 0.40.6 itself cannot serve this model.** It recognizes the rank-pooling
  GGUF (`ollama show` reports `embedding` in capabilities) but `/api/embed` returns
  an all-zero vector and `/v1/rerank` drops the connection by crashing its internal
  runner. The standalone `llama-server` binary is required.
* Two community 4B Q8_0 conversions produced all-zero rerank scores (broken
  classification head); the `ggml-org` 0.6B conversion scores correctly
  (0.94 vs 0.01 on a positive/negative pair). Prefer `ggml-org` conversions.

## Data

QASPER (Dasigi et al., NAACL 2021), CC BY 4.0 — papers, questions, annotated answers
and evidence spans. `data/eval_set.json` is a stratified subset of the official dev
split; the full corpus is downloaded from AllenAI's S3 bucket at reproduce time.
`scripts/qasper_evaluator.py` is the official evaluation script distributed with the
dataset and is used here for answer token-F1.

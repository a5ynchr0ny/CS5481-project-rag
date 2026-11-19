"""Full-scale retrieval benchmark: every answerable dev question, 8 configs, no LLM calls.

Configs (top-5 throughout):
  ours_A, ours_B          our Qwen3 embeddings, Chroma qasper_A/B, paper-scoped
  ours_A_rr24, ours_B_rr24  + Qwen3-Reranker-0.6B over 24 candidates
  ours_A_global, ours_B_global   same embeddings, global (no paper filter)
  ps_fixed, ps_section    PaperScope chunker + BGE-small, paper-scoped (their stack)

Metric: strict containment — the full gold evidence string (normalized) appears
inside ONE of the top-k chunks. Hit@1/3/5, MRR, EvidenceRecall@5, paper-hit.

Usage:
  python full_retrieval_bench.py --limit 50     # smoke + per-stage timing
  python full_retrieval_bench.py                # full run
Writes F:/AI/cs5481-project/full_bench.json (per-question flags only, no texts).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path("F:/AI/cs5481-project")
sys.path.insert(0, str(BASE / "scripts"))
from prepare_corpus import clean_text  # noqa: E402
from retrieve import QUERY_INSTRUCTION  # noqa: E402

QWEN_MODEL = "qwen3-embedding:4b-fp16"
OLLAMA = "http://localhost:11434"
RERANK_URL = "http://127.0.0.1:8012/v1/rerank"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def norm(s: str) -> str:
    return " ".join(clean_text(s).lower().split())


def build_pool() -> list[dict]:
    corpus = json.loads((BASE / "qasper-dev-v0.3.json").read_text(encoding="utf-8"))
    pool = []
    for paper_id, paper in corpus.items():
        for qa in paper["qas"]:
            a = qa["answers"][0]["answer"]
            if a["unanswerable"]:
                continue
            ev = [e for e in a["evidence"] if "FLOAT SELECTED" not in e]
            if not ev:
                continue
            pool.append({"question_id": qa["question_id"], "paper_id": paper_id,
                         "question": qa["question"], "evidence": ev})
    return pool


def qwen_embed_batch(questions: list[str]) -> list[list[float]]:
    out = []
    t0 = time.time()
    for i in range(0, len(questions), 64):
        batch = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {q}" for q in questions[i:i + 64]]
        payload = json.dumps({"model": QWEN_MODEL, "input": batch}).encode()
        req = urllib.request.Request(f"{OLLAMA}/api/embed", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            out.extend(json.loads(resp.read())["embeddings"])
    print(f"  qwen embed: {len(out)} queries in {time.time()-t0:.0f}s", flush=True)
    return out


def bge_embed_batch(questions: list[str]) -> list[list[float]]:
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    t0 = time.time()
    vecs = [v.tolist() for v in model.query_embed([BGE_QUERY_PREFIX + q for q in questions])]
    print(f"  bge embed: {len(vecs)} queries in {time.time()-t0:.0f}s", flush=True)
    return vecs


def rerank(query: str, docs: list[str]) -> list[float]:
    payload = json.dumps({"query": query, "documents": docs, "top_n": len(docs),
                          "instruction": QUERY_INSTRUCTION}).encode()
    req = urllib.request.Request(RERANK_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    scores = [0.0] * len(docs)
    for item in data.get("results", []):
        scores[item["index"]] = float(item["relevance_score"])
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pool = build_pool()
    if args.limit:
        pool = pool[:args.limit]
    print(f"pool: {len(pool)} answerable questions with text evidence", flush=True)

    import chromadb
    chroma_ours = chromadb.PersistentClient(path=str(BASE / "chroma"))
    chroma_ps = chromadb.PersistentClient(path=str(BASE / "chroma_ps"))

    questions = [q["question"] for q in pool]
    qwen_vecs = qwen_embed_batch(questions)
    bge_vecs = bge_embed_batch(questions)

    configs = []  # (name, client, collection, vecs, scope, rerank)
    for s in ("A", "B"):
        configs.append((f"ours_{s}", chroma_ours, f"qasper_{s}", qwen_vecs, "paper", False))
    for s in ("A", "B"):
        configs.append((f"ours_{s}_rr24", chroma_ours, f"qasper_{s}", qwen_vecs, "paper", True))
    for s in ("A", "B"):
        configs.append((f"ours_{s}_global", chroma_ours, f"qasper_{s}", qwen_vecs, "global", False))
    for strat in ("fixed", "section"):
        configs.append((f"ps_{strat}", chroma_ps, f"ps_{strat}", bge_vecs, "paper", False))

    results = {q["question_id"]: {"paper_id": q["paper_id"],
                                  "n_evidence": len(q["evidence"])} for q in pool}

    for name, client, col_name, vecs, scope, use_rr in configs:
        t0 = time.time()
        col = client.get_collection(col_name)
        n_fetch = 24 if use_rr else 5
        for idx, q in enumerate(pool):
            where = {"paper_id": q["paper_id"]} if scope == "paper" else None
            res = col.query(query_embeddings=[vecs[idx]], n_results=n_fetch, where=where)
            ids = res["ids"][0]
            texts = res["documents"][0]
            if use_rr and texts:
                scores = rerank(q["question"], texts)
                order = sorted(range(len(texts)), key=lambda i: -scores[i])
                ids = [ids[i] for i in order[:5]]
                texts = [texts[i] for i in order[:5]]
            normed = [norm(t) for t in texts]
            hit_flags = [any(norm(e) in nt for e in q["evidence"]) for nt in normed]
            rec = {
                "hit1": hit_flags[0] if hit_flags else False,
                "hit3": any(hit_flags[:3]),
                "hit5": any(hit_flags[:5]),
                "mrr": next((1 / (r + 1) for r, h in enumerate(hit_flags) if h), 0.0),
                "er5": (sum(1 for e in q["evidence"]
                            if any(norm(e) in nt for nt in normed)) / len(q["evidence"])
                        if q["evidence"] else 0.0),
                "paper_hit": any(i.split(":")[1] == q["paper_id"] or
                                 i.split(":")[0] == q["paper_id"] for i in ids[:5]),
            }
            results[q["question_id"]][name] = rec
        dt = time.time() - t0
        h5 = sum(results[q["question_id"]][name]["hit5"] for q in pool) / len(pool)
        print(f"[{name}] {len(pool)} qs in {dt:.0f}s ({dt/len(pool)*1000:.0f} ms/q)  Hit@5={h5:.3f}", flush=True)

    out = BASE / "full_bench.json"
    out.write_text(json.dumps(
        [{"question_id": q["question_id"], **results[q["question_id"]]} for q in pool],
        ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

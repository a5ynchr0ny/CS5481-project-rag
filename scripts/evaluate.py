"""Evaluate the RAG system: retrieval hits, answer quality, abstention, citations.

Per chunking strategy (A/B):
  Phase 1  retrieval-only over the whole eval set:
             Hit@5 / MRR against gold evidence (cleaned substring match),
             top-1 score distribution -> per-strategy refusal threshold
             tau = 10th percentile of answerable top-1 scores (window
             scores are systematically lower; a shared tau would bias
             the abstention comparison).
  Phase 2  generation with per-strategy tau:
             answer F1 (max over annotators, official token_f1_score),
             boolean accuracy (word-boundary Yes/No match),
             abstention accuracy + refusal precision/recall,
             citation validity over non-refused answerable answers.

Outputs: data/results_{A,B}.json (per-question records) + printed tables.

Usage (desktop):
  python evaluate.py --eval-set F:/AI/cs5481-project/eval_set.json \
                     --db F:/AI/cs5481-project/chroma --out F:/AI/cs5481-project/results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import CITE_RE, REFUSAL, answer_question  # noqa: E402
from prepare_corpus import clean_text  # noqa: E402
from qasper_evaluator import token_f1_score  # noqa: E402


def norm(s: str) -> str:
    return " ".join(clean_text(s).lower().split())


def evidence_hit(hits: list[dict], evidence: list[str]) -> bool:
    """A chunk counts as a hit when it CONTAINS a gold evidence string."""
    chunks = [norm(h["text"]) for h in hits]
    return any(norm(e) in c for e in evidence for c in chunks)


def evidence_mrr(hits: list[dict], evidence: list[str]) -> float:
    for rank, h in enumerate(hits, 1):
        if evidence_hit([h], evidence):
            return 1.0 / rank
    return 0.0


def boolean_correct(answer: str, gold: str) -> bool:
    return re.search(rf"\b{gold.lower()}\b", answer.lower()) is not None


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(p * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_strategy(client, eval_set: list[dict], strategy: str, scope: str, k: int,
                 out_dir: Path, host: str, emb_model: str, rerank_n: int = 0) -> dict:
    col = client.get_collection(f"qasper_{strategy}")
    records: list[dict] = []

    # ---- phase 1: retrieval + tau calibration ----
    from retrieve import retrieve
    for item in eval_set:
        pid = item["paper_id"] if scope == "paper" else None
        hits = retrieve(col, item["question"], k=k, model=emb_model, host=host,
                        paper_id=pid, rerank_n=rerank_n)
        rec = dict(item)
        rec["hits"] = [{"chunk_id": h["chunk_id"], "score": round(h["score"], 4)} for h in hits]
        rec["top_score"] = hits[0]["score"] if hits else 0.0
        answerable = item["qtype"] in ("extractive", "abstractive", "boolean")
        rec["hit5"] = bool(item["evidence"]) and evidence_hit(hits, item["evidence"])
        rec["mrr"] = evidence_mrr(hits, item["evidence"]) if answerable else None
        records.append(rec)

    scores = sorted(r["top_score"] for r in records
                    if r["qtype"] in ("extractive", "abstractive", "boolean"))
    tau = round(percentile(scores, 0.10), 3)
    print(f"\n[{strategy}/{scope}] tau (p10 of answerable top-1) = {tau} "
          f"(answerable top-1: min={scores[0]:.3f} max={scores[-1]:.3f})", flush=True)

    # ---- phase 2: generation ----
    for i, rec in enumerate(records):
        pid = rec["paper_id"] if scope == "paper" else None
        res = answer_question(col, rec["question"], k=k, tau=tau, strategy=strategy,
                              host=host, emb_model=emb_model, paper_id=pid, rerank_n=rerank_n)
        rec["refused"] = res["refused"]
        rec["answer"] = res["answer"]
        rec["validation"] = res["validation"]
        if (i + 1) % 10 == 0:
            print(f"[{strategy}/{scope}] generated {i + 1}/{len(records)}", flush=True)

    # ---- metrics ----
    answerable = [r for r in records if r["qtype"] in ("extractive", "abstractive", "boolean")]
    negative = [r for r in records if r["qtype"] in ("unanswerable", "oob")]

    hits_all = [r["hit5"] for r in answerable]
    mrr_all = [r["mrr"] for r in answerable]

    f1_vals, bool_correct_vals, gen_answerable = [], [], []
    for r in answerable:
        pred = CITE_RE.sub("", r["answer"])
        if r["refused"] or pred.strip() == REFUSAL:
            continue
        gen_answerable.append(r)
        if r["qtype"] == "boolean":
            bool_correct_vals.append(boolean_correct(pred, r["gold_answer"]))
        else:
            f1_vals.append(token_f1_score(pred, r["gold_answer"]))

    abst_correct = sum(1 for r in negative if r["refused"])
    tp = sum(1 for r in negative if r["refused"])
    fp = sum(1 for r in answerable if r["refused"])
    fn = len(negative) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    cited = [r for r in gen_answerable if r["qtype"] != "boolean"]
    citation_rate = (sum(1 for r in cited if r["validation"]["has_citations"]) / len(cited)) if cited else None
    invalid_cites = sum(r["validation"]["n_invalid"] for r in gen_answerable)

    metrics = {
        "strategy": strategy,
        "scope": scope,
        "rerank_n": rerank_n,
        "tau": tau,
        "n_questions": len(records),
        "Hit@5": round(sum(hits_all) / len(hits_all), 3) if hits_all else None,
        "MRR": round(sum(mrr_all) / len(mrr_all), 3) if mrr_all else None,
        "answer_F1_extr_abs": round(sum(f1_vals) / len(f1_vals), 3) if f1_vals else None,
        "n_answer_F1": len(f1_vals),
        "boolean_accuracy": round(sum(bool_correct_vals) / len(bool_correct_vals), 3) if bool_correct_vals else None,
        "n_boolean": len(bool_correct_vals),
        "abstention_accuracy": round(abst_correct / len(negative), 3) if negative else None,
        "refusal_precision": round(precision, 3) if precision is not None else None,
        "refusal_recall": round(recall, 3) if recall is not None else None,
        "citation_rate_nonboolean": round(citation_rate, 3) if citation_rate is not None else None,
        "invalid_citations_total": invalid_cites,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{scope}" + (f"_rr{rerank_n}" if rerank_n else "")
    (out_dir / f"results_{strategy}{suffix}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{strategy}/{scope}{', rr' if rerank_n else ''}] " + json.dumps(
        {k2: v for k2, v in metrics.items() if k2 not in ("strategy", "scope", "rerank_n")},
        ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, default=Path("F:/AI/cs5481-project/eval_set.json"))
    ap.add_argument("--db", type=Path, default=Path("F:/AI/cs5481-project/chroma"))
    ap.add_argument("--out", type=Path, default=Path("F:/AI/cs5481-project/results"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--emb-model", default="qwen3-embedding:4b-fp16")
    ap.add_argument("--scope", default="paper,global", help="comma list: paper,global")
    ap.add_argument("--strategies", default="A,B", help="comma list of chunking strategies")
    ap.add_argument("--rerank", type=int, default=0,
                    help="candidate pool size for Qwen3-Reranker re-ordering (0 = off)")
    args = ap.parse_args()

    import chromadb

    eval_set = json.loads(args.eval_set.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(args.db))
    all_metrics = []
    for scope in args.scope.split(","):
        for s in args.strategies.split(","):
            all_metrics.append(run_strategy(client, eval_set, s.strip(), scope.strip(), args.k,
                                            args.out, args.host, args.emb_model, args.rerank))
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsummary -> {summary}")


if __name__ == "__main__":
    main()

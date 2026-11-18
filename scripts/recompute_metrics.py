"""Offline metric enrichment from saved evaluation records (no LLM calls).

Recomputes, per results file:
  Hit@1 / Hit@3 / Hit@5          evidence substring match on top-k prefixes
  Evidence Recall@5              fraction of gold evidence strings covered
  paper-hit                      gold paper present in top-5 at all
  per-type breakdown             Hit@5 + answer F1 by extractive/abstractive
  boolean accuracy               word-boundary Yes/No match
  hallucination rate             answered negatives / negatives
  miss taxonomy                  for missed answerable questions:
                                   paper-miss  (gold paper not in top-5)
                                   rank-miss   (paper in top-5, evidence chunk not)
                                   split-miss  (evidence head found in a top-5
                                                chunk but full string does not
                                                match -> evidence cut by chunking)

Usage: python recompute_metrics.py --results F:/AI/cs5481-project/results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import CITE_RE, REFUSAL  # noqa: E402
from evaluate import boolean_correct, evidence_hit, norm  # noqa: E402
from qasper_evaluator import token_f1_score  # noqa: E402

ANSWERABLE = {"extractive", "abstractive", "boolean"}


def load(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for enc in ("utf-8", "gbk"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot decode {path}")


def evidence_recall(hits_texts: list[str], evidence: list[str]) -> float:
    if not evidence:
        return 0.0
    joined = " || ".join(norm(t) for t in hits_texts)
    found = sum(1 for e in evidence if norm(e) in joined)
    return found / len(evidence)


def classify_miss(rec: dict, texts: dict[str, str]) -> str:
    if not rec["evidence"]:
        return "no-evidence"
    gold_paper = rec["paper_id"]
    hit_chunks = [h["chunk_id"] for h in rec["hits"]]
    if not any(c.split(":")[1] == gold_paper for c in hit_chunks):
        return "paper-miss"
    head = " ".join(norm(rec["evidence"][0]).split()[:10])
    if any(head in norm(texts[c]) for c in hit_chunks if c in texts):
        return "split-miss"
    return "rank-miss"


def enrich(path: Path, col) -> dict:
    rows = load(path)
    for r in rows:
        ids = [h["chunk_id"] for h in r["hits"]]
        texts = {}
        if ids:
            got = col.get(ids=ids)
            texts = dict(zip(got["ids"], got["documents"]))
        r["_texts"] = texts

    answerable = [r for r in rows if r["qtype"] in ANSWERABLE]
    negatives = [r for r in rows if r["qtype"] not in ANSWERABLE]

    def hit_at(r, k):
        subset = [{**h, "text": r["_texts"].get(h["chunk_id"], "")} for h in r["hits"][:k]]
        return bool(r["evidence"]) and evidence_hit(subset, r["evidence"])

    out: dict = {"file": path.name, "n_answerable": len(answerable), "n_negative": len(negatives)}
    for k in (1, 3, 5):
        out[f"Hit@{k}"] = round(sum(hit_at(r, k) for r in answerable) / len(answerable), 3)

    ev_recall = []
    for r in answerable:
        texts5 = [r["_texts"].get(h["chunk_id"], "") for h in r["hits"]]
        ev_recall.append(evidence_recall(texts5, r["evidence"]))
    out["EvidenceRecall@5"] = round(sum(ev_recall) / len(ev_recall), 3)

    out["paper_hit"] = round(sum(
        1 for r in answerable if any(h["chunk_id"].split(":")[1] == r["paper_id"] for h in r["hits"])
    ) / len(answerable), 3)

    by_type: dict[str, dict] = {}
    for t in ("extractive", "abstractive", "boolean"):
        subset = [r for r in answerable if r["qtype"] == t]
        d: dict = {"n": len(subset)}
        d["Hit@5"] = round(sum(hit_at(r, 5) for r in subset) / len(subset), 3) if subset else None
        f1s = []
        for r in subset:
            pred = CITE_RE.sub("", r["answer"])
            if r["refused"] or pred.strip() == REFUSAL:
                continue
            if t == "boolean":
                d.setdefault("bool_correct", boolean_correct(pred, r["gold_answer"]))
            else:
                f1s.append(token_f1_score(pred, r["gold_answer"]))
        if f1s:
            d["answer_F1"] = round(sum(f1s) / len(f1s), 3)
            d["n_answered"] = len(f1s)
        by_type[t] = d
    out["by_type"] = by_type

    out["hallucination_rate"] = round(sum(1 for r in negatives if not r["refused"]) / len(negatives), 3)

    misses = [r for r in answerable if not hit_at(r, 5)]
    taxonomy = {"paper-miss": 0, "rank-miss": 0, "split-miss": 0, "no-evidence": 0}
    for r in misses:
        taxonomy[classify_miss(r, r["_texts"])] += 1
    out["miss_taxonomy"] = taxonomy
    out["n_miss"] = len(misses)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("F:/AI/cs5481-project/results"))
    ap.add_argument("--db", type=Path, default=Path("F:/AI/cs5481-project/chroma"))
    args = ap.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=str(args.db))
    collections = {s: client.get_collection(f"qasper_{s}") for s in ("A", "B")}

    for path in sorted(args.results.glob("results_*.json")):
        if path.name == "summary.json":
            continue
        strategy = path.name.split("_")[1]
        if strategy not in collections:
            continue
        m = enrich(path, collections[strategy])
        m.pop("file", None)
        print(f"\n== {path.name} ==")
        print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

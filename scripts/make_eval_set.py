"""Build the stratified evaluation set from the QASPER dev split.

Composition (deterministic, seed=42):
  extractive  20   abstractive 12   boolean 8   unanswerable 20   = 60 in-KB
  oob          5   questions about papers NOT in the KB (test split),
                   a second source of "must refuse" cases.

Rules
  - answers are taken from the FIRST annotator (as in our corpus stats).
  - answerable questions whose evidence is ONLY figure/table ("FLOAT
    SELECTED") are excluded, matching the evaluator's text_evidence_only
    mode; questions with mixed evidence stay (hit = any gold evidence).
  - gold answer strings follow the official evaluator conventions:
    extractive = spans joined with ", "; abstractive = free-form text;
    boolean = "Yes"/"No".

Output: data/eval_set.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
QUOTA = {"extractive": 20, "abstractive": 12, "boolean": 8, "unanswerable": 20}
N_OOB = 5


def classify(answer_info: dict) -> str | None:
    if answer_info["unanswerable"]:
        return "unanswerable"
    if answer_info["extractive_spans"]:
        return "extractive"
    if answer_info["free_form_answer"]:
        return "abstractive"
    if answer_info["yes_no"] is not None:
        return "boolean"
    return None


def gold_answer(answer_info: dict) -> str | None:
    if answer_info["unanswerable"]:
        return None
    if answer_info["extractive_spans"]:
        return ", ".join(answer_info["extractive_spans"])
    if answer_info["free_form_answer"]:
        return answer_info["free_form_answer"]
    return "Yes" if answer_info["yes_no"] else "No"


def only_float(evidence: list[str]) -> bool:
    return bool(evidence) and all("FLOAT SELECTED" in e for e in evidence)


def pool(dev: dict) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = {k: [] for k in QUOTA}
    for paper_id, paper in dev.items():
        for qa in paper["qas"]:
            ans = qa["answers"][0]["answer"]
            t = classify(ans)
            if t is None or (t != "unanswerable" and only_float(ans["evidence"])):
                continue
            by_type[t].append({
                "question_id": qa["question_id"],
                "paper_id": paper_id,
                "question": qa["question"],
                "qtype": t,
                "gold_answer": gold_answer(ans),
                "evidence": ans["evidence"],
            })
    return by_type


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dev = json.loads((root / "data/qasper-src/qasper-dev-v0.3.json").read_text())
    test = json.loads((root / "data/qasper-src/qasper-test-v0.3.json").read_text())

    rng = random.Random(SEED)
    picked: list[dict] = []
    for t, n in QUOTA.items():
        pool_t = sorted(pool(dev)[t], key=lambda x: x["question_id"])  # deterministic
        rng.shuffle(pool_t)
        picked.extend(pool_t[:n])

    # out-of-KB: questions from test-split papers (never indexed)
    test_papers = sorted(test.keys())
    rng.shuffle(test_papers)
    for pid in test_papers:
        if len([p for p in picked if p["qtype"] == "oob"]) >= N_OOB:
            break
        qa = test[pid]["qas"][0]  # first question of an out-of-KB paper
        picked.append({
            "question_id": qa["question_id"] + ":oob",
            "paper_id": pid,
            "question": qa["question"],
            "qtype": "oob",
            "gold_answer": None,
            "evidence": [],
        })

    out = root / "data/eval_set.json"
    out.write_text(json.dumps(picked, ensure_ascii=False, indent=1))
    from collections import Counter
    print(f"eval set: {len(picked)} questions -> {out}")
    print("by type:", dict(Counter(p["qtype"] for p in picked)))


if __name__ == "__main__":
    main()

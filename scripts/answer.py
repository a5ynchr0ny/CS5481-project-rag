"""Grounded answer generation with verifiable citations (DeepSeek API).

Pipeline: retrieve top-k -> (optional) similarity-threshold refusal ->
LLM answer with mandatory [n] citations -> post-validate citations.

Design notes
  - each context block is presented as "[n] chunk_id | paper_id | section"
    followed by its text; the model must cite [n] after every claim.
  - hard refusal strings (deterministic + LLM) match the refusal sentence
    used by the evaluator, so abstention is measurable.
  - post-validation flags any citation whose [n] is missing/out-of-range
    (citation validity rate is a report metric).

CLI:
  python answer.py "question" --strategy A
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

from retrieve import DEFAULT_HOST, DEFAULT_MODEL, retrieve

REFUSAL = "The provided knowledge base does not contain sufficient information to answer this question."
CITE_RE = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = f"""You are a precise research assistant. Answer the user's question using ONLY the numbered context passages provided.

Rules:
1. After EVERY sentence that states a fact, append the citation of the passage that supports it, in the form [n] where n is the passage number. Multiple citations are allowed: [1][3].
2. If the context passages do not contain the information needed, reply with EXACTLY this sentence and nothing else:
{REFUSAL}
3. Do not use outside knowledge. Do not guess."""


def _chat(messages: list[dict], model: str) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0.0}).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] {h['chunk_id']} | {h['meta']['paper_id']} | {h['meta']['section_name']}\n{h['text']}")
    return "\n\n".join(blocks)


def postvalidate(answer: str, k: int) -> dict:
    """Citation audit: every [n] must resolve to a retrieved chunk."""
    cites = sorted({int(m) for m in CITE_RE.findall(answer)})
    valid = [n for n in cites if 1 <= n <= k]
    return {
        "citations": cites,
        "valid_citations": valid,
        "n_invalid": len(cites) - len(valid),
        "has_citations": bool(cites),
    }


def answer_question(
    collection,
    question: str,
    k: int = 5,
    tau: float = 0.30,
    strategy: str = "A",
    llm_model: str = "deepseek-chat",
    host: str = DEFAULT_HOST,
    emb_model: str = DEFAULT_MODEL,
    paper_id: str | None = None,
    rerank_n: int = 0,
) -> dict:
    hits = retrieve(collection, question, k=k, model=emb_model, host=host,
                    paper_id=paper_id, rerank_n=rerank_n)
    top = hits[0]["score"] if hits else 0.0

    result = {
        "question": question,
        "strategy": strategy,
        "top_score": round(top, 4),
        "hits": [{"chunk_id": h["chunk_id"], "score": round(h["score"], 4)} for h in hits],
    }

    # deterministic guard rail: below-threshold similarity -> refuse without an LLM call
    if top < tau:
        result.update({"refused": True, "refusal_reason": "below_threshold", "answer": REFUSAL,
                       "validation": {"citations": [], "valid_citations": [], "n_invalid": 0, "has_citations": False}})
        return result

    prompt = f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {question}"
    text = _chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], llm_model)
    is_refusal = text.strip() == REFUSAL
    result.update({
        "refused": is_refusal,
        "refusal_reason": "llm_refusal" if is_refusal else None,
        "answer": text,
        "validation": postvalidate(text, k),
    })
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--strategy", default="A", choices=["A", "B"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.30, help="top-1 similarity below this -> deterministic refusal")
    ap.add_argument("--db", default="F:/AI/cs5481-project/chroma")
    args = ap.parse_args()

    import chromadb

    col = chromadb.PersistentClient(path=args.db).get_collection(f"qasper_{args.strategy}")
    res = answer_question(col, args.question, k=args.k, tau=args.tau, strategy=args.strategy)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

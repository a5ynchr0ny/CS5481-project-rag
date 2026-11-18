"""Diagnostic: separate paper-level hits from evidence-level hits, and
inspect a few missed questions (is retrieval failing, or the matcher?)."""
import json
from pathlib import Path

sys_root = Path("F:/AI/cs5481-project/results")
ANSWERABLE = {"extractive", "abstractive", "boolean"}


def load(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for enc in ("utf-8", "gbk"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot decode {path}")


for s in ("A", "B"):
    rows = load(sys_root / f"results_{s}.json")
    ans = [r for r in rows if r["qtype"] in ANSWERABLE]
    paper_hit = sum(
        1 for r in ans
        if any(h["chunk_id"].split(":")[1] == r["paper_id"] for h in r["hits"])
    )
    ev_hit = sum(1 for r in ans if r["hit5"])
    refused = sum(1 for r in ans if r["refused"])
    print(f"[{s}] answerable={len(ans)}  paper-hit(top5 has gold paper)={paper_hit}  "
          f"evidence-hit={ev_hit}  auto-refused={refused}")

# inspect 3 evidence-missed but paper-hit cases from strategy A (matcher check)
rows = load(sys_root / "results_A.json")
shown = 0
for r in rows:
    if r["qtype"] not in ANSWERABLE or r["hit5"] or r["refused"]:
        continue
    paper_in_hits = any(h["chunk_id"].split(":")[1] == r["paper_id"] for h in r["hits"])
    if not paper_in_hits:
        continue
    print("\n--- missed despite gold paper in top5 ---")
    print("Q:", r["question"])
    print("gold evidence[0]:", (r["evidence"][0] if r["evidence"] else "")[:200])
    hit_chunks = [h["chunk_id"] for h in r["hits"] if h["chunk_id"].split(":")[1] == r["paper_id"]]
    print("gold-paper chunks in top5:", hit_chunks)
    shown += 1
    if shown >= 3:
        break

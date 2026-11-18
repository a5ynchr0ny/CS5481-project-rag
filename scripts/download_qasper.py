"""Download allenai/qasper and export one JSONL per split.

Usage: .venv/bin/python scripts/download_qasper.py
Data lands in data/qasper/*.jsonl; the HF cache is kept inside
data/hf-cache so the whole thing is self-contained and scp-able.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "qasper"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data" / "hf-cache"))

from datasets import load_dataset  # noqa: E402  (after cache env is set)


def main():
    for split in ("train", "validation", "test"):
        ds = load_dataset("allenai/qasper", split=split)
        path = OUT / f"{split}.jsonl"
        ds.to_json(str(path))
        n_q = None
        if "qas" in ds.column_names:
            n_q = sum(len(paper["qas"]) for paper in ds)
        msg = f"{split}: {len(ds)} papers -> {path.name} ({path.stat().st_size / 1e6:.1f} MB)"
        if n_q is not None:
            msg += f", {n_q} questions"
        print(msg)


if __name__ == "__main__":
    main()

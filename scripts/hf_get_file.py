"""Download one GGUF file via huggingface_hub (proper resume/verify) to F: drive."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = sys.argv[1]
FILENAME = sys.argv[2]
OUT = Path("F:/AI/cs5481-project/gguf")
OUT.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(repo_id=REPO, filename=FILENAME, local_dir=str(OUT))
size = Path(path).stat().st_size / 1e9
print(f"done: {path} ({size:.2f} GB)", flush=True)

"""Download the distractor corpus: WikiText-103 (raw) test split, CC BY-SA 3.0.

Writes data/corpus/wikitext103_test.jsonl with one cleaned paragraph per line and
a sidecar .meta.json recording the source URL and SHA-256 of both files.

Usage: python scripts/fetch_corpus.py
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import httpx
import pyarrow.parquet as pq

URL = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
       "wikitext-103-raw-v1/test-00000-of-00001.parquet")
OUT = Path("data/corpus/wikitext103_test.jsonl")
MIN_WORDS = 60

_DETOK = [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" ,", ","), (" .", "."),
          (" ;", ";"), (" :", ":"), (" 's", "'s"), ("( ", "("), (" )", ")"), (" n't", "n't")]


def clean(line: str) -> str:
    s = line.strip()
    for a, b in _DETOK:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s)


def main() -> None:
    raw = httpx.get(URL, follow_redirects=True, timeout=120).raise_for_status().content
    lines = pq.read_table(io.BytesIO(raw)).column("text").to_pylist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    article, n = None, 0
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            s = line.strip()
            if re.fullmatch(r"= [^=].*[^=] =", s):  # top-level heading starts a new article
                article = s.strip("= ").strip()
                continue
            if not s or s.startswith("="):
                continue
            text = clean(s)
            if len(text.split()) >= MIN_WORDS:
                f.write(json.dumps({"article": article, "text": text}) + "\n")
                n += 1
    meta = {
        "source_url": URL,
        "license": "CC BY-SA 3.0 (WikiText-103)",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "paragraphs": n,
        "min_words": MIN_WORDS,
    }
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

"""Streaming readers that turn each raw source into (doc_id, text, url) records.

Nothing here loads a large file whole (8 GB RAM): parquet is read one row
group at a time, CSVs row by row, and multi-GB pretty-printed JSON arrays
(BanglishRev, 2 GB) through `iter_json_array`, an incremental parser.

Run standalone to print a few records from every downloaded source:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.readers
"""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

from data.common import load_config, repo_path


class Record(NamedTuple):
    """One raw document from a source."""

    doc_id: str
    text: str
    url: str | None = None


def iter_json_array(path: Path, chunk_chars: int = 2**23) -> Iterator[Any]:
    """Yield the top-level elements of a (possibly huge) JSON array file one at a time."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = f.read(chunk_chars)
        pos = 0
        while pos < len(buf) and buf[pos] in " \t\r\n\ufeff":
            pos += 1
        if pos >= len(buf) or buf[pos] != "[":
            raise ValueError(f"{path} is not a JSON array")
        pos += 1
        eof = False
        while True:
            while True:  # skip separators, refilling as needed
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf) or eof:
                    break
                more = f.read(chunk_chars)
                eof = not more
                buf, pos = buf[pos:] + more, 0
            if pos >= len(buf):
                raise ValueError(f"{path}: unexpected end of file (no closing ']')")
            if buf[pos] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = f.read(chunk_chars)
                eof = not more
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end
            if pos > chunk_chars:  # drop consumed text so the buffer stays bounded
                buf, pos = buf[pos:], 0


def _parquet_rows(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for rg in range(pf.metadata.num_row_groups):
        yield from pf.read_row_group(rg, columns=columns).to_pylist()


def read_bnsentmix(raw: Path) -> Iterator[Record]:
    """BnSentMix rows (id = CSV row index). Used by splits.py, not directly as pretraining text."""
    cfg = load_config()["bnsentmix"]
    with (raw / "bnsentmix" / "dataset.csv").open("r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            yield Record(str(i), row[cfg["text_column"]])


def read_vashantor(raw: Path) -> Iterator[Record]:
    """All four text columns of Vashantor train + validation; the test split is held out
    for Prompt 3's transliteration evaluation."""
    z = zipfile.ZipFile(raw / "vashantor" / "Vashantor_CSV_Format.zip")
    for name in sorted(n for n in z.namelist() if n.lower().endswith(".csv")):
        if "/Test/" in name:
            continue
        text = z.read(name).decode("utf-8-sig", errors="replace")
        reader = csv.reader(io.StringIO(text))
        header = [h.strip() for h in next(reader)]
        cols = [i for i, h in enumerate(header) if h.endswith("_speech") and not h.startswith("english")]
        for r, row in enumerate(reader):
            for c in cols:
                if c < len(row) and row[c].strip():
                    yield Record(f"{Path(name).stem}:{r}:{header[c]}", row[c])


def read_banglishdepyt(raw: Path) -> Iterator[Record]:
    """Comment text from the unlabeled and labeled CSVs (video transcripts are skipped)."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    base = raw / "banglishdepyt"
    for rel in ("unlabeled/youtube_collected_raw.csv", "labeled/yt_train.csv", "labeled/yt_val.csv", "labeled/yt_test.csv"):
        with (base / rel).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            text_col = "text" if "text" in (reader.fieldnames or []) else (reader.fieldnames or ["text"])[0]
            for i, row in enumerate(reader):
                yield Record(f"{rel}:{i}", row.get(text_col) or "")


def read_banglishrev(raw: Path) -> Iterator[Record]:
    """Review texts from BanglishRev's product-level JSON array."""
    path = raw / "banglishrev" / "reviews v1.json"
    for p, product in enumerate(iter_json_array(path)):
        for r, review in enumerate(product.get("Reviews") or []):
            text = review.get("Review Content")
            if text:
                yield Record(f"{p}:{r}", text)


def read_fineweb(raw: Path, name: str) -> Iterator[Record]:
    """FineWeb-2 parquet files (downloaded files or sampled row-group parts)."""
    for path in sorted((raw / name).rglob("*.parquet")):
        for row in _parquet_rows(path, ["text", "id", "url"]):
            yield Record(str(row["id"]), row["text"], row.get("url"))


def read_wikipedia(raw: Path) -> Iterator[Record]:
    """Bengali Wikipedia articles (title + text)."""
    for path in sorted((raw / "wikipedia_bn").rglob("*.parquet")):
        for row in _parquet_rows(path, ["id", "url", "title", "text"]):
            yield Record(str(row["id"]), f"{row['title']}\n{row['text']}", row.get("url"))


def iter_source(name: str, raw: Path | None = None) -> Iterator[Record]:
    """Dispatch to the reader for a configured pretraining source."""
    raw = raw or repo_path(load_config()["paths"]["raw_dir"])
    if name == "vashantor":
        return read_vashantor(raw)
    if name == "banglishdepyt":
        return read_banglishdepyt(raw)
    if name == "banglishrev":
        return read_banglishrev(raw)
    if name in ("fineweb2_ben_latn", "fineweb2_ben_beng"):
        return read_fineweb(raw, name)
    if name == "wikipedia_bn":
        return read_wikipedia(raw)
    raise KeyError(f"no reader for source {name!r}")


if __name__ == "__main__":
    from itertools import islice

    from data.common import setup_env

    setup_env()
    raw_dir = repo_path(load_config()["paths"]["raw_dir"])
    for src in ("vashantor", "banglishdepyt", "banglishrev", "fineweb2_ben_latn", "wikipedia_bn", "fineweb2_ben_beng"):
        try:
            for rec in islice(iter_source(src, raw_dir), 2):
                print(f"[{src}] {rec.doc_id}: {rec.text[:100]!r}")
        except (FileNotFoundError, OSError) as exc:
            print(f"[{src}] not downloaded yet ({exc.__class__.__name__})")

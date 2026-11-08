"""Build the cleaned pretraining corpus: one streaming pass per source.

For every document: clean → length filter → NSFW filter (where configured) →
global exact dedup (match-normalized, across all sources) → leakage scan
against BnSentMix → append to Arrow IPC shards under data/processed/<source>/
(memory-mappable for Prompt 3). Per-source statistics and all leakage hits are
returned to audit.py, which owns the gate.

BnSentMix contributes only its TRAIN split text. Vashantor's test split is held
out (readers.py) for Prompt 3's transliteration evaluation.

Usage (normally via data.audit):
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.corpus [--only SOURCE ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from data.clean import clean_text, n_words, normalize_for_match, nsfw_reason, script_stats, word_script_counts
from data.common import load_config, peak_rss_mb, repo_path, setup_env, write_json
from data.leakage import Hit, LeakageIndex, build_index
from data.readers import Record, iter_source

PRETRAIN_SOURCES = (
    "bnsentmix_train",
    "vashantor",
    "banglishdepyt",
    "banglishrev",
    "fineweb2_ben_latn",
    "wikipedia_bn",
    "fineweb2_ben_beng",
)


class ShardWriter:
    """Append-only writer of Arrow IPC shards (uncompressed, memory-mappable)."""

    def __init__(self, out_dir: Path, max_docs: int, batch_docs: int) -> None:
        import pyarrow as pa

        self.pa = pa
        self.schema = pa.schema([
            ("doc_id", pa.string()), ("source", pa.string()), ("text", pa.large_string()),
            ("script", pa.string()), ("n_words", pa.int32()),
        ])
        self.out_dir, self.max_docs, self.batch_docs = out_dir, max_docs, batch_docs
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("shard-*.arrow"):
            old.unlink()
        self.shard, self.in_shard = 0, 0
        self.writer: Any = None
        self.buf: dict[str, list[Any]] = {k: [] for k in self.schema.names}

    def add(self, **row: Any) -> None:
        """Buffer one row; flush in batches."""
        for k in self.schema.names:
            self.buf[k].append(row[k])
        if len(self.buf["doc_id"]) >= self.batch_docs:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows, rolling to a new shard at max_docs."""
        if not self.buf["doc_id"]:
            return
        if self.writer is None:
            path = self.out_dir / f"shard-{self.shard:05d}.arrow"
            self.writer = self.pa.ipc.new_file(str(path), self.schema)
        batch = self.pa.record_batch([self.pa.array(self.buf[k], type=self.schema.field(k).type) for k in self.schema.names], schema=self.schema)
        self.writer.write_batch(batch)
        self.in_shard += batch.num_rows
        self.buf = {k: [] for k in self.schema.names}
        if self.in_shard >= self.max_docs:
            self.writer.close()
            self.writer, self.in_shard = None, 0
            self.shard += 1

    def close(self) -> None:
        """Flush and close the current shard."""
        self.flush()
        if self.writer is not None:
            self.writer.close()


def load_exclusions(cfg: dict[str, Any]) -> dict[tuple[str, str], str]:
    """Approved leakage removals: (source, doc_id) -> sha256 of the cleaned text."""
    path = repo_path(cfg["paths"]["leakage_exclusions"])
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(d["source"], str(d["doc_id"])): d["sha256"] for d in data["docs"]}


def norm_key(norm: str) -> int:
    """64-bit key of match-normalized text (shared by dedup and exclusion)."""
    return int.from_bytes(hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest(), "little")


def load_excluded_norm_keys(cfg: dict[str, Any]) -> set[int]:
    """Normalized-text keys of every excluded doc, so exact copies are excluded too.

    Excluding only by doc_id is not enough: dedup keeps the FIRST copy of a text, so
    removing that copy by id lets the next identical copy through.
    """
    path = repo_path(cfg["paths"]["leakage_exclusions"])
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {norm_key(normalize_for_match(d["text"])) for d in data["docs"]}


def is_excluded(name: str, doc_id: str, text: str, exclusions: dict[tuple[str, str], str]) -> bool:
    """True if this doc is on the approved exclusion list.

    Raises if the doc id matches but the text hash does not, which means the raw
    source or the cleaning changed and the exclusion list must be re-reviewed.
    """
    expected = exclusions.get((name, doc_id))
    if expected is None:
        return False
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual != expected:
        raise RuntimeError(f"exclusion hash mismatch for {name}:{doc_id} ({actual} != {expected}); re-review leakage")
    return True


def _records(name: str, cfg: dict[str, Any]) -> Iterator[Record]:
    if name == "bnsentmix_train":
        from data.splits import load_split_rows

        for r in load_split_rows(cfg, "train"):
            yield Record(f"train:{r['row_id']}", r["text"])
        return
    yield from iter_source(name)


def build_source(
    name: str, cfg: dict[str, Any], index: LeakageIndex, seen: dict[int, int], source_no: int
) -> tuple[dict[str, Any], list[Hit]]:
    """Process one source; returns (stats, leakage hits)."""
    exclusions = load_exclusions(cfg)
    excluded_keys = load_excluded_norm_keys(cfg)
    excluded_copies: list[str] = []
    src_cfg = cfg["sources"].get(name, {})
    nsfw_on = bool(src_cfg.get("nsfw_filter", False))
    min_words = int(cfg["cleaning"]["min_words"])
    out = repo_path(cfg["paths"]["processed_dir"]) / name
    writer = ShardWriter(out, int(cfg["output"]["shard_max_docs"]), int(cfg["output"]["write_batch_docs"]))
    rng = random.Random(int(cfg["minhash"]["seed"]) + source_no)
    k_spot = int(cfg["nsfw"]["spot_check_samples"])

    drops: Counter[str] = Counter()
    dup_with: Counter[str] = Counter()
    doc_scripts: Counter[str] = Counter()
    word_scripts: Counter[str] = Counter()
    lengths: list[int] = []
    chars = docs_in = 0
    kept_sample: list[str] = []
    nsfw_sample: list[str] = []
    hits: list[Hit] = []
    t0 = time.time()

    for rec in _records(name, cfg):
        docs_in += 1
        text = clean_text(rec.text)
        if is_excluded(name, rec.doc_id, text, exclusions):
            drops["leakage_excluded"] += 1
            continue
        nw = n_words(text)
        if nw < min_words:
            drops["too_short"] += 1
            continue
        if nsfw_on:
            reason = nsfw_reason(text, rec.url)
            if reason:
                drops[f"nsfw_{reason}"] += 1
                if len(nsfw_sample) < 20:
                    nsfw_sample.append(text[:200])
                continue
        norm = normalize_for_match(text)
        key = norm_key(norm)
        if key in excluded_keys:
            drops["leakage_excluded_copy"] += 1
            excluded_copies.append(rec.doc_id)
            continue
        first = seen.get(key)
        if first is not None:
            drops["duplicate"] += 1
            dup_with[PRETRAIN_SOURCES[first]] += 1
            continue
        seen[key] = source_no
        if name != "bnsentmix_train":
            hits.extend(index.scan(text, name, rec.doc_id, norm))
        script = script_stats(text).label
        doc_scripts[script] += 1
        word_scripts.update(word_script_counts(text))
        lengths.append(nw)
        chars += len(text)
        kept = len(lengths)
        if len(kept_sample) < k_spot:
            kept_sample.append(text[:200])
        elif rng.random() < k_spot / kept:  # reservoir sample
            kept_sample[rng.randrange(k_spot)] = text[:200]
        writer.add(doc_id=rec.doc_id, source=name, text=text, script=script, n_words=nw)
        if docs_in % 200000 == 0:
            print(f"  [{name}] {docs_in:,} docs in, {kept:,} kept, {len(hits):,} hits, "
                  f"{time.time() - t0:,.0f}s, peak RSS {peak_rss_mb():,.0f} MB", flush=True)
    writer.close()

    lengths.sort()
    words = sum(lengths)
    stats = {
        "docs_in": docs_in,
        "docs_kept": len(lengths),
        "dropped": dict(drops),
        "duplicates_first_seen_in": dict(dup_with),
        "leakage_excluded_copy_ids": excluded_copies,
        "chars": chars,
        "words": words,
        "doc_script": dict(doc_scripts),
        "word_script": dict(word_scripts),
        "words_per_doc": {
            "mean": round(words / len(lengths), 1) if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "p95": lengths[int(len(lengths) * 0.95)] if lengths else 0,
        },
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "spot_check_kept": kept_sample,
        "spot_check_nsfw_dropped": nsfw_sample,
    }
    print(f"  [{name}] done: {len(lengths):,}/{docs_in:,} docs, {words:,} words, {len(hits):,} hits, "
          f"{stats['seconds']:,.0f}s", flush=True)
    return stats, hits


def build_corpus(cfg: dict[str, Any], only: list[str] | None = None) -> tuple[dict[str, Any], list[Hit]]:
    """Build all pretraining sources in priority order (earlier sources win dedup)."""
    index = build_index(cfg)
    print(f"leakage index: {len(index.samples):,} BnSentMix samples", flush=True)
    seen: dict[int, int] = {}
    all_stats: dict[str, Any] = {}
    all_hits: list[Hit] = []
    for i, name in enumerate(PRETRAIN_SOURCES):
        if only and name not in only:
            continue
        print(f"[{name}]", flush=True)
        stats, hits = build_source(name, cfg, index, seen, i)
        all_stats[name] = stats
        all_hits.extend(hits)
    return all_stats, all_hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the cleaned pretraining corpus.")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    config = load_config()
    setup_env(config)
    s, h = build_corpus(config, args.only)
    out = repo_path(config["paths"]["processed_dir"]) / "_corpus_stats.json"
    write_json(out, {"stats": s, "hits": [x.as_dict() for x in h]})
    print(json.dumps({k: {kk: v[kk] for kk in ("docs_kept", "words")} for k, v in s.items()}, indent=2))

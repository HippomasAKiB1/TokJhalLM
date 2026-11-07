"""Detect BnSentMix samples inside pretraining documents (R5: dev/test must never leak).

Three detectors, all on match-normalized text (configs/data.yaml `leakage`):
- exact:       the whole document, or one of its short lines/sentences, equals a sample.
- containment: >= `containment_fraction` of a sample's word n-grams (n = min(8, len))
               occur in one document. Samples shorter than `min_words_for_ngrams`
               are covered by exact matching only.
- near_dup:    comment-length documents whose char-shingle Jaccard with a sample is
               >= `near_dup_jaccard` (MinHash LSH + exact verification). Matches in
               [0.7, threshold) are recorded as `near_dup_soft` for sensitivity only.

All BnSentMix rows are indexed so the report can show train overlap too; only
hits on the splits in `leakage.splits_checked` (dev, test) fail the gate.

Run standalone for a self-check on synthetic text:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.leakage
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from data.clean import clean_text, normalize_for_match
from data.minhash import LSHIndex, MinHasher, shingles

SOFT_JACCARD = 0.7
SEGMENT_RE = re.compile(r"[\n.!?।|]+")


@dataclass
class Sample:
    """One indexed BnSentMix row."""

    split: str
    row_id: int
    label: int
    text: str
    norm: str
    n_ngrams: int = 0


@dataclass
class Hit:
    """A sample found in a pretraining document."""

    split: str
    row_id: int
    sample_text: str
    source: str
    doc_id: str
    match: str
    score: float
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return self.__dict__.copy()


@dataclass
class LeakageIndex:
    """Index over BnSentMix samples for exact, containment and near-duplicate lookup."""

    cfg: dict[str, Any]
    samples: list[Sample] = field(default_factory=list)
    exact: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    ngrams: dict[int, dict[int, list[int]]] = field(default_factory=dict)
    vocab: set[str] = field(default_factory=set)
    max_exact_words: int = 0

    def __post_init__(self) -> None:
        lcfg = self.cfg["leakage"]
        self.n_max = int(lcfg["ngram_max"])
        self.min_words = int(lcfg["min_words_for_ngrams"])
        self.contain_frac = float(lcfg["containment_fraction"])
        self.nd_threshold = float(lcfg["near_dup_jaccard"])
        self.nd_max_words = int(lcfg["near_dup_max_words"])
        self.k = int(self.cfg["minhash"]["shingle_chars"])
        self.lsh = LSHIndex(MinHasher.from_config(self.cfg))

    def add(self, split: str, row_id: int, label: int, text: str) -> None:
        """Index one sample."""
        norm = normalize_for_match(text)
        if not norm:
            return
        idx = len(self.samples)
        words = norm.split()
        s = Sample(split, row_id, label, text, norm)
        self.exact[norm].append(idx)
        self.max_exact_words = max(self.max_exact_words, len(words))
        if len(words) >= self.min_words:
            n = min(self.n_max, len(words))
            grams = {hash(tuple(words[i : i + n])) for i in range(len(words) - n + 1)}
            table = self.ngrams.setdefault(n, defaultdict(list))
            for g in grams:
                table[g].append(idx)
            s.n_ngrams = len(grams)
            self.vocab.update(words)
        self.lsh.add(shingles(norm, self.k))
        self.samples.append(s)

    def _hit(self, idx: int, source: str, doc_id: str, match: str, score: float, excerpt: str) -> Hit:
        s = self.samples[idx]
        return Hit(s.split, s.row_id, s.text, source, doc_id, match, round(score, 3), excerpt[:300])

    def scan(self, text: str, source: str, doc_id: str, norm: str | None = None) -> list[Hit]:
        """All samples found in one document (deduplicated by sample, strongest match kept)."""
        norm = normalize_for_match(text) if norm is None else norm
        if not norm:
            return []
        found: dict[int, Hit] = {}

        def keep(h: Hit, idx: int) -> None:
            if idx not in found or found[idx].score < h.score:
                found[idx] = h

        # exact: whole doc, then short segments
        for idx in self.exact.get(norm, ()):
            keep(self._hit(idx, source, doc_id, "exact", 1.0, text), idx)
        words = norm.split()
        if len(words) > self.max_exact_words:
            for seg in SEGMENT_RE.split(clean_text(text)):
                if 0 < seg.count(" ") + 1 <= self.max_exact_words + 2:
                    sn = normalize_for_match(seg)
                    for idx in self.exact.get(sn, ()):
                        keep(self._hit(idx, source, doc_id, "exact", 1.0, seg), idx)

        # containment: only runs of in-vocab words can contain a sample n-gram
        matched: dict[int, set[int]] = defaultdict(set)
        run: list[str] = []
        for w in words + [""]:
            if w and w in self.vocab:
                run.append(w)
                continue
            if len(run) >= self.min_words:
                for n, table in self.ngrams.items():
                    for i in range(len(run) - n + 1):
                        g = hash(tuple(run[i : i + n]))
                        for idx in table.get(g, ()):
                            matched[idx].add(g)
            run = []
        for idx, grams in matched.items():
            total = self.samples[idx].n_ngrams
            frac = len(grams) / total if total else 0.0
            need = 1.0 if total <= 3 else self.contain_frac
            if frac >= need:
                keep(self._hit(idx, source, doc_id, "containment", frac, text), idx)

        # near-duplicate: comment-length docs only
        if len(words) <= self.nd_max_words:
            for idx, j in self.lsh.query(shingles(norm, self.k), SOFT_JACCARD):
                match = "near_dup" if j >= self.nd_threshold else "near_dup_soft"
                if match == "near_dup" or idx not in found:
                    keep(self._hit(idx, source, doc_id, match, j, text), idx)
        return list(found.values())


def build_index(cfg: dict[str, Any]) -> LeakageIndex:
    """Index every BnSentMix row from the committed split manifests."""
    from data.splits import SPLITS, load_split_rows

    index = LeakageIndex(cfg)
    for split in SPLITS:
        for r in load_split_rows(cfg, split):
            index.add(split, r["row_id"], r["label"], r["text"])
    return index


if __name__ == "__main__":
    from data.common import load_config, setup_env

    config = load_config()
    setup_env(config)
    ix = LeakageIndex(config)
    ix.add("test", 1, 0, "Vaia apnar voice ta onek shundor, ami protidin shuni")
    ix.add("dev", 2, 1, "Network khub baje")
    docs = {
        "clean": "Ei product ta khub valo, delivery o fast chilo",
        "exact": "vaia apnar voice ta onek shundor ami protidin shuni!!!",
        "contained": "Long page text here. Vaia apnar voice ta onek shundor ami protidin shuni. More text follows on the page.",
        "short_line": "Some intro\nNetwork khub baje\nother stuff and much more text in this document to make it long enough",
        "near": "vaia apnar voice ta onek sundor ami protidin shuni",
    }
    for name, doc in docs.items():
        hits = ix.scan(doc, "demo", name)
        print(f"{name:<11} -> {[(h.split, h.match, h.score) for h in hits]}")

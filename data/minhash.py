"""Deterministic MinHash + LSH for near-duplicate detection (numpy only).

Used by splits.py (cluster BnSentMix near-duplicates so they share a split)
and leakage.py (find near-copies of dev/test samples in pretraining text).
Shingles are character n-grams of the match-normalized text, hashed with
CRC32 so results are identical across runs (Python's hash() is salted).
Candidates from LSH are always verified with exact Jaccard.

Run standalone for a demo:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.minhash
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

MERSENNE = np.uint64((1 << 61) - 1)
MAX32 = np.uint64(0xFFFFFFFF)


def shingles(text: str, k: int) -> set[int]:
    """CRC32 hashes of character k-grams (the whole text if shorter than k)."""
    if len(text) <= k:
        return {zlib.crc32(text.encode("utf-8"))} if text else set()
    return {zlib.crc32(text[i : i + k].encode("utf-8")) for i in range(len(text) - k + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    """Exact Jaccard similarity of two shingle sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    """MinHash signatures with `num_perm` universal hash functions, banded for LSH."""

    def __init__(self, num_perm: int, bands: int, seed: int) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, int(MERSENNE), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(MERSENNE), size=num_perm, dtype=np.uint64)
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> MinHasher:
        """Build from the `minhash` section of configs/data.yaml."""
        m = cfg["minhash"]
        return cls(int(m["num_perm"]), int(m["bands"]), int(m["seed"]))

    def signature(self, sh: set[int]) -> np.ndarray:
        """MinHash signature (uint32 per permutation) of a shingle set."""
        if not sh:
            return np.full(self.num_perm, int(MAX32), dtype=np.uint32)
        x = np.fromiter(sh, dtype=np.uint64, count=len(sh))
        # (a*x + b) mod p, computed in uint64 with wraparound; low 32 bits kept.
        h = (np.outer(x, self.a) + self.b) % MERSENNE
        return (h.min(axis=0) & MAX32).astype(np.uint32)

    def band_keys(self, sig: np.ndarray) -> list[bytes]:
        """One hashable key per LSH band."""
        r = self.rows
        return [bytes([i]) + sig[i * r : (i + 1) * r].tobytes() for i in range(self.bands)]


class LSHIndex:
    """Banded LSH index over a fixed reference set (e.g. BnSentMix dev+test)."""

    def __init__(self, hasher: MinHasher) -> None:
        self.hasher = hasher
        self.buckets: dict[bytes, list[int]] = defaultdict(list)
        self.shingle_sets: list[set[int]] = []

    def add(self, sh: set[int]) -> int:
        """Index a shingle set; returns its integer id."""
        idx = len(self.shingle_sets)
        self.shingle_sets.append(sh)
        for key in self.hasher.band_keys(self.hasher.signature(sh)):
            self.buckets[key].append(idx)
        return idx

    def query(self, sh: set[int], threshold: float) -> list[tuple[int, float]]:
        """Indexed ids whose exact Jaccard with `sh` is >= threshold."""
        cands: set[int] = set()
        for key in self.hasher.band_keys(self.hasher.signature(sh)):
            cands.update(self.buckets.get(key, ()))
        out = []
        for c in cands:
            j = jaccard(sh, self.shingle_sets[c])
            if j >= threshold:
                out.append((c, j))
        return out


class UnionFind:
    """Disjoint sets over 0..n-1."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """Root of x, with path halving."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Merge the sets containing a and b."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_dup_clusters(texts: Iterable[str], k: int, hasher: MinHasher, threshold: float) -> list[int]:
    """Cluster id per text: texts linked by Jaccard >= threshold share a cluster."""
    index = LSHIndex(hasher)
    sets = [shingles(t, k) for t in texts]
    uf = UnionFind(len(sets))
    for i, sh in enumerate(sets):
        for j, _ in index.query(sh, threshold):
            uf.union(i, j)
        index.add(sh)
    return [uf.find(i) for i in range(len(sets))]


if __name__ == "__main__":
    hasher = MinHasher(num_perm=128, bands=32, seed=1234)
    docs = [
        "vaia apnar voice ta onek shundor",
        "vaia apnar voice ta onek sundor",
        "network khub baje atar moto faltu net r nai",
        "vaia apnar voice ta onek shundor",
    ]
    print("clusters:", near_dup_clusters(docs, 5, hasher, 0.8))
    a, b = shingles(docs[0], 5), shingles(docs[1], 5)
    print(f"jaccard(0,1) = {jaccard(a, b):.3f}")

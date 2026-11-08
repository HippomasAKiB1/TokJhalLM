"""Transliteration engine: Bengali script -> romanized Banglish (Prompt 3, D7-amended).

Every Bengali word type goes to the first tier that can handle it:
  T1  Vashantor lexicon: attested natural spellings from the parallel train+val
      sentences (word-aligned where word counts match).
  T2  character-level encoder-decoder trained from scratch (data/translit_model.py).
  T3  deterministic rules (data/translit_rules.py).
Latin-script runs, numbers, punctuation and emoji pass through unchanged (Bengali
digits and the danda become ASCII).

The Vashantor TEST split is never read here except by `load_vashantor_pairs("Test")`,
which only data/translit_eval.py calls.

Run standalone for a demo (T1 + T3; T2 if a checkpoint exists):
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.translit
"""

from __future__ import annotations

import csv
import io
import random
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from data.common import load_config, repo_path
from data.translit_rules import (
    eval_words,
    finalize_other,
    normalize_bengali_word,
    romanize_rules,
    segments,
    source_words,
)

TRANSLIT_CONFIG = "configs/translit.yaml"


def load_translit_config() -> dict[str, Any]:
    """Load configs/translit.yaml."""
    return load_config(repo_path(TRANSLIT_CONFIG))


@dataclass(frozen=True)
class Pair:
    """One parallel Vashantor sentence pair."""

    kind: str      # "std" (standard Bengali) or "dia" (dialect)
    region: str
    bengali: str
    roman: str


def load_vashantor_pairs(split: str, cfg: dict[str, Any] | None = None) -> list[Pair]:
    """Standard and dialect (Bengali, Banglish) pairs from one Vashantor split."""
    cfg = cfg or load_translit_config()
    z = zipfile.ZipFile(repo_path(cfg["paths"]["vashantor_zip"]))
    out: list[Pair] = []
    for name in sorted(z.namelist()):
        if f"/{split}/" not in name or not name.lower().endswith(".csv"):
            continue
        rows = list(csv.reader(io.StringIO(z.read(name).decode("utf-8-sig"))))
        for r in rows[1:]:
            if len(r) < 5:
                continue
            region = r[4].strip()
            out.append(Pair("std", region, r[0].strip(), r[1].strip()))
            out.append(Pair("dia", region, r[2].strip(), r[3].strip()))
    return out


def aligned_words(pairs: Iterable[Pair]) -> Iterable[tuple[str, str]]:
    """(bengali_word, roman_word) from pairs whose word counts match."""
    for p in pairs:
        src, tgt = source_words(p.bengali), eval_words(p.roman)
        if src and len(src) == len(tgt):
            yield from zip(src, tgt)


def build_lexicon(cfg: dict[str, Any] | None = None) -> dict[str, Counter[str]]:
    """T1: Bengali word -> Counter of attested romanizations (train + validation only)."""
    cfg = cfg or load_translit_config()
    lex: dict[str, Counter[str]] = defaultdict(Counter)
    for split in cfg["lexicon"]["splits"]:
        if split.lower() == "test":
            raise ValueError("the Vashantor test split must never feed the lexicon")
        for b, r in aligned_words(load_vashantor_pairs(split, cfg)):
            if any("\u0980" <= ch <= "\u09ff" for ch in b):
                lex[b][r] += 1
    return dict(lex)


def test_word_types(cfg: dict[str, Any] | None = None) -> set[str]:
    """Every Bengali word type in the Vashantor test split (excluded from T2 training)."""
    cfg = cfg or load_translit_config()
    return {w for p in load_vashantor_pairs("Test", cfg) for w in source_words(p.bengali)}


class Transliterator:
    """Routes each Bengali word to T1, then a fallback (T2 or T3)."""

    def __init__(
        self,
        lexicon: dict[str, Counter[str]] | None,
        fallback: Callable[[list[str]], list[str]] | None = None,
        sample_temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.lexicon = lexicon or {}
        self.fallback = fallback or (lambda ws: [romanize_rules(w) for w in ws])
        self.temperature = sample_temperature
        self.rng = random.Random(seed)
        self.cache: dict[str, str] = {}

    def tier(self, word: str) -> str:
        """'T1' if the word is in the lexicon, else 'fallback'."""
        return "T1" if normalize_bengali_word(word) in self.lexicon else "fallback"

    def romanize_words(self, words: list[str], sample: bool = False) -> list[str]:
        """Romanize Bengali words; T1 majority (or sampled) spelling, else the fallback."""
        keys = [normalize_bengali_word(w) for w in words]
        todo = sorted({k for k in keys if k not in self.lexicon and k not in self.cache})
        if todo:
            self.cache.update(zip(todo, self.fallback(todo)))
        out = []
        for k in keys:
            c = self.lexicon.get(k)
            if c is None:
                out.append(self.cache[k])
            elif sample:
                forms = list(c)
                weights = [c[f] ** (1.0 / self.temperature) for f in forms]
                out.append(self.rng.choices(forms, weights)[0])
            else:
                out.append(c.most_common(1)[0][0])
        return out

    def romanize_text(self, text: str, sample: bool = False) -> str:
        """Romanize running text, keeping non-Bengali runs (with ASCII digits/danda)."""
        segs = segments(text)
        words = [s for is_b, s in segs if is_b]
        roman = iter(self.romanize_words(words, sample)) if words else iter(())
        return "".join(next(roman) if is_b else finalize_other(s) for is_b, s in segs)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    lex = build_lexicon()
    print(f"T1 lexicon: {len(lex):,} Bengali word types")
    tr = Transliterator(lex)
    for s in ["তোমার আব্বু কেমন আছে?", "বাংলাদেশের স্বাধীনতা যুদ্ধ ১৯৭১ সালে শুরু হয়।"]:
        print(s, "->", tr.romanize_text(s))

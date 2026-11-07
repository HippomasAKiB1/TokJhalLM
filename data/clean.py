"""Text cleaning, match-normalization, script detection and NSFW filtering.

One cleaning function (`clean_text`) is shared by every model and every source,
BanglaBERT/BanglishBERT baselines included, so preprocessing is identical
across comparisons (PRD §9.3). A second, more aggressive function
(`normalize_for_match`) exists only for deduplication and leakage detection.

Run standalone for a demo:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.clean
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from data.common import load_config

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<![\w@])@[\w.]+")
# Zero-width chars to drop. ZWNJ (U+200C) and ZWJ (U+200D) are KEPT: Bengali uses
# them to control conjunct and ya-phala rendering.
ZERO_WIDTH_RE = re.compile("[\u200b\u2060\ufeff\u00ad]")
SPACES_RE = re.compile(r"[^\S\n]+")
WORD_RE = re.compile(r"\S+")


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return load_config()


@lru_cache(maxsize=1)
def _bengali_bounds() -> tuple[str, str]:
    lo, hi = _cfg()["script"]["bengali_range"]
    return str(lo), str(hi)


def is_bengali_char(ch: str) -> bool:
    """True if `ch` is in the Bengali Unicode block."""
    lo, hi = _bengali_bounds()
    return lo <= ch <= hi


def clean_text(text: str) -> str:
    """Corpus cleaning: NFC, drop URLs/@mentions/zero-width chars, tidy whitespace.

    Emoji are kept (see configs/data.yaml `cleaning.keep_emoji_in_corpus`).
    Line structure is preserved; long runs of blank lines are collapsed.
    """
    max_nl = int(_cfg()["cleaning"]["max_newlines"])
    t = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    t = ZERO_WIDTH_RE.sub("", t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = SPACES_RE.sub(" ", t)
    t = "\n".join(line.strip() for line in t.split("\n"))
    t = re.sub(r"\n{%d,}" % (max_nl + 1), "\n" * max_nl, t)
    return t.strip()


@lru_cache(maxsize=1)
def _regexes() -> dict[str, re.Pattern[str]]:
    lo, hi = _bengali_bounds()
    n = int(_cfg()["cleaning"]["repeat_char_squash"])
    return {
        # anything that is not a word char or Bengali-block char (covers vowel signs),
        # plus underscore, becomes a space
        "nonword": re.compile(f"[^\\w{lo}-{hi}]+|_+"),
        "repeat": re.compile(r"(.)\1{%d,}" % (n - 1)),
        "bengali": re.compile(f"[{lo}-{hi}]"),
        "latin": re.compile(r"[A-Za-z]"),
    }


def normalize_for_match(text: str) -> str:
    """Aggressive normalization for dedup and leakage matching only.

    clean_text, then lowercase, map every non letter/digit (punctuation, emoji,
    symbols, underscore) to space, squash runs of >= N identical characters to
    two ("sooooo" -> "soo"), collapse whitespace.
    """
    rx = _regexes()
    t = clean_text(text).lower()
    t = rx["nonword"].sub(" ", t)
    t = rx["repeat"].sub(r"\1\1", t)
    return " ".join(t.split())


@dataclass(frozen=True)
class ScriptStats:
    """Letter counts by script for one text."""

    bengali: int
    latin: int

    @property
    def label(self) -> str:
        """'bengali', 'latin', 'mixed', or 'none' (no letters)."""
        total = self.bengali + self.latin
        if total == 0:
            return "none"
        frac = float(_cfg()["script"]["dominant_fraction"])
        if self.bengali / total >= frac:
            return "bengali"
        if self.latin / total >= frac:
            return "latin"
        return "mixed"


def script_stats(text: str) -> ScriptStats:
    """Count Bengali-block characters vs ASCII Latin letters."""
    rx = _regexes()
    return ScriptStats(bengali=len(rx["bengali"].findall(text)), latin=len(rx["latin"].findall(text)))


def word_script_counts(text: str) -> dict[str, int]:
    """Count whitespace words by script: bengali / latin / mixed / other."""
    rx = _regexes()
    beng, lat = rx["bengali"].search, rx["latin"].search
    counts = {"bengali": 0, "latin": 0, "mixed": 0, "other": 0}
    for w in text.split():
        has_b, has_l = beng(w) is not None, lat(w) is not None
        key = "mixed" if has_b and has_l else "bengali" if has_b else "latin" if has_l else "other"
        counts[key] += 1
    return counts


@lru_cache(maxsize=1)
def _nsfw() -> tuple[tuple[str, ...], frozenset[str], float]:
    cfg = _cfg()["nsfw"]
    return (
        tuple(s.lower() for s in cfg["url_substrings"]),
        frozenset(t.lower() for t in cfg["romanized_terms"]),
        float(cfg["terms_per_100_words"]),
    )


def nsfw_reason(text: str, url: str | None = None) -> str | None:
    """Return why a document is adult content ('url' or 'terms'), or None if it passes."""
    url_subs, terms, density = _nsfw()
    if url and any(s in url.lower() for s in url_subs):
        return "url"
    words = normalize_for_match(text).split()
    if not words:
        return None
    hits = sum(1 for w in words if w in terms)
    if hits and hits * 100.0 / len(words) >= density:
        return "terms"
    return None


def n_words(text: str) -> int:
    """Whitespace word count."""
    return len(WORD_RE.findall(text))


if __name__ == "__main__":
    from data.common import setup_env

    setup_env()
    samples = [
        "Youtube ar volg gula boring hoia jaitase din din. Onk din por ai volg ta dakhe val laglo 😍😍",
        "আজ (৮-৪-২৪) হাতে পেলাম। দেখে ভালো ই মনে হচ্ছে https://example.com @someone",
        "Amr bmi besi noi amr high bp i nei amr naker problem ache sudhu অ্যালার্জি o nei",
    ]
    for s in samples:
        c = clean_text(s)
        print("raw     :", s)
        print("clean   :", c)
        print("match   :", normalize_for_match(s))
        print("script  :", script_stats(c).label, word_script_counts(c))
        print("nsfw    :", nsfw_reason(c))
        print()

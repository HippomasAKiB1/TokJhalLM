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


URL_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class NsfwRules:
    """NSFW filter settings from configs/data.yaml `nsfw` (D16)."""

    url_exact: frozenset[str]
    url_affix: tuple[str, ...]
    url_prefix: tuple[str, ...]
    url_substring: tuple[str, ...]
    terms: frozenset[str]
    density: float
    bengali_terms: frozenset[str]
    bengali_min_distinct: int
    bengali_density: float


@lru_cache(maxsize=1)
def _nsfw() -> NsfwRules:
    cfg = _cfg()["nsfw"]
    return NsfwRules(
        url_exact=frozenset(s.lower() for s in cfg["url_tokens_exact"]),
        url_affix=tuple(s.lower() for s in cfg["url_tokens_affix"]),
        url_prefix=tuple(s.lower() for s in cfg["url_tokens_prefix"]),
        url_substring=tuple(s.lower() for s in cfg["url_tokens_substring"]),
        terms=frozenset(t.lower() for t in cfg["romanized_terms"]),
        density=float(cfg["terms_per_100_words"]),
        bengali_terms=frozenset(unicodedata.normalize("NFC", t) for t in cfg["bengali_terms"]),
        bengali_min_distinct=int(cfg["bengali_min_distinct_terms"]),
        bengali_density=float(cfg["bengali_terms_per_100_words"]),
    )


def url_tokens(url: str) -> list[str]:
    """Hostname labels and path segments of a URL, split on non-alphanumerics.

    Percent-encoded bytes are left encoded, so their hex digits never form words.
    """
    u = re.sub(r"^[a-z]+://", "", url.lower())
    return URL_TOKEN_RE.findall(re.sub(r"%[0-9a-f]{2}", " ", u))


def nsfw_url_hits(url: str | None) -> list[str]:
    """URL rule terms that fire on this URL (D16 token rules)."""
    if not url:
        return []
    rules = _nsfw()
    hits: set[str] = set()
    for tok in url_tokens(url):
        if tok in rules.url_exact:
            hits.add(tok)
        hits.update(a for a in rules.url_affix if tok.startswith(a) or tok.endswith(a))
        hits.update(a for a in rules.url_prefix if tok.startswith(a))
        hits.update(a for a in rules.url_substring if a in tok)
    return sorted(hits)


def nsfw_reason(text: str, url: str | None = None) -> str | None:
    """Why a document is adult content ('url', 'terms' or 'bengali_terms'), or None.

    - url: a URL token equals an exact term, starts/ends with an affix term,
      starts with a prefix term, or contains a substring term.
    - terms: romanized explicit terms at or above the configured density.
    - bengali_terms: >= N distinct Bengali-script explicit terms AND density
      at or above the threshold (a news report naming one crime never fires).
    """
    rules = _nsfw()
    if nsfw_url_hits(url):
        return "url"
    words = normalize_for_match(text).split()
    if not words:
        return None
    hits = sum(1 for w in words if w in rules.terms)
    if hits and hits * 100.0 / len(words) >= rules.density:
        return "terms"
    beng = [w for w in words if w in rules.bengali_terms]
    if len(set(beng)) >= rules.bengali_min_distinct and len(beng) * 100.0 / len(words) >= rules.bengali_density:
        return "bengali_terms"
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

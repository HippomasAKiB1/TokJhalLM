"""NSFW filter false-positive spot check (approved by Akib, Prompt 2).

Re-reads every source with `nsfw_filter: true`, re-applies data.clean.nsfw_reason, and
reservoir-samples N drops per source (fixed seed). Writes docs/nsfw_spotcheck.md for manual
TP/FP review: URL, rule (url/terms), which substrings/terms fired, first 400 chars.

Usage:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m scripts.nsfw_spotcheck [--per-source 20]
"""

from __future__ import annotations

import argparse
import random

from data.clean import clean_text, n_words, normalize_for_match, nsfw_reason
from data.common import load_config, repo_path, setup_env
from data.readers import iter_source


def fired(text: str, url: str | None, cfg: dict) -> str:
    """Which URL substrings or terms triggered the rule."""
    n = cfg["nsfw"]
    subs = [s for s in n["url_substrings"] if url and s in url.lower()]
    words = normalize_for_match(text).split()
    terms = sorted({w for w in words if w in set(n["romanized_terms"])})
    return f"url:{','.join(subs)}" if subs else f"terms:{','.join(terms)}"


def main() -> int:
    """Sample drops per source and write the review table."""
    p = argparse.ArgumentParser()
    p.add_argument("--per-source", type=int, default=20)
    args = p.parse_args()
    cfg = load_config()
    setup_env(cfg)
    rng = random.Random(int(cfg["minhash"]["seed"]))
    lines = ["# NSFW filter spot check", "",
             f"{args.per_source} random drops per filtered source (seed {cfg['minhash']['seed']}). "
             "Fill the verdict column: TP = adult content, FP = innocent.", ""]
    for name, src in cfg["sources"].items():
        if not src.get("nsfw_filter"):
            continue
        sample: list[tuple[str, str | None, str, str]] = []
        seen = drops = 0
        for rec in iter_source(name):
            text = clean_text(rec.text)
            if n_words(text) < int(cfg["cleaning"]["min_words"]):
                continue
            seen += 1
            if not nsfw_reason(text, rec.url):
                continue
            drops += 1
            item = (rec.doc_id, rec.url, fired(text, rec.url, cfg), text[:400].replace("\n", " ").replace("|", "/"))
            if len(sample) < args.per_source:
                sample.append(item)
            elif rng.random() < args.per_source / drops:
                sample[rng.randrange(args.per_source)] = item
        lines += [f"## {name}: {drops:,} dropped of {seen:,} ({drops / max(seen, 1):.1%})", "",
                  "| # | rule fired | url | text (first 400 chars) | verdict |", "|---|---|---|---|---|"]
        for i, (doc_id, url, rule, text) in enumerate(sample, 1):
            lines.append(f"| {i} | {rule} | {url or ''} | {text} | |")
        lines.append("")
    out = repo_path("docs/nsfw_spotcheck.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

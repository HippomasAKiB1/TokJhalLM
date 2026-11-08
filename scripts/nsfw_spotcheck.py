"""NSFW filter false-positive spot check (Prompt 2) and D16 replay (Prompt 3).

Default mode re-reads every source with `nsfw_filter: true`, applies
data.clean.nsfw_reason, and reservoir-samples N drops per source (fixed seed). Writes
docs/nsfw_spotcheck.md (gitignored: it contains explicit excerpts) for TP/FP review.

--replay (D16 verification, Akib's requirement): reproduces the ORIGINAL Prompt 2 sample
exactly, using the pre-D16 rule from configs/data.yaml `nsfw.legacy` with the same seed
and reservoir order, verifies it row by row against the local docs/nsfw_spotcheck.md
(by URL), then applies the NEW rule to exactly those docs. Also counts, over every doc,
how many the old rule dropped that the new rule keeps and vice versa. Writes:
  - docs/nsfw_replay.md        (committed; redacted: no URLs, no text)
  - docs/nsfw_replay_local.md  (gitignored; URLs of changed docs for manual review)
Exit 1 if the replay sample does not match, or if any fineweb2_ben_beng FP is still
dropped (the fix is not done).

Usage:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m scripts.nsfw_spotcheck [--per-source 20]
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m scripts.nsfw_spotcheck --replay
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from typing import Any

from data.clean import clean_text, n_words, normalize_for_match, nsfw_reason, nsfw_url_hits
from data.common import load_config, repo_path, setup_env
from data.readers import iter_source



def legacy_reason(text: str, url: str | None, cfg: dict[str, Any]) -> str | None:
    """The pre-D16 rule (raw URL substrings, then romanized term density)."""
    n = cfg["nsfw"]
    if url and any(s in url.lower() for s in n["legacy"]["url_substrings"]):
        return "url"
    words = normalize_for_match(text).split()
    terms = set(n["romanized_terms"])
    hits = sum(1 for w in words if w in terms)
    if words and hits and hits * 100.0 / len(words) >= float(n["terms_per_100_words"]):
        return "terms"
    return None


def legacy_fired(text: str, url: str | None, cfg: dict[str, Any]) -> str:
    """Which legacy URL substrings or terms triggered the rule (matches Prompt 2 output)."""
    n = cfg["nsfw"]
    subs = [s for s in n["legacy"]["url_substrings"] if url and s in url.lower()]
    words = normalize_for_match(text).split()
    terms = sorted({w for w in words if w in set(n["romanized_terms"])})
    return f"url:{','.join(subs)}" if subs else f"terms:{','.join(terms)}"


def new_fired(text: str, url: str | None) -> str:
    """The new rule's verdict with the URL terms that fired, or 'kept'."""
    reason = nsfw_reason(text, url)
    if reason == "url":
        return "url:" + ",".join(nsfw_url_hits(url))
    return reason or "kept"


def parse_local_review(path_text: str) -> dict[str, list[tuple[str, str]]]:
    """(url, verdict) rows per source section from the Prompt 2 review file."""
    out: dict[str, list[tuple[str, str]]] = {}
    current = None
    for line in path_text.splitlines():
        if line.startswith("## ") and ":" in line and "dropped of" in line:
            current = line[3:].split(":")[0].strip()
            out[current] = []
        elif current and line.startswith("| ") and line[2:3].isdigit():
            cells = line.split("|")
            out[current].append((cells[3].strip(), cells[-2].strip()))
    return out


def verdict_kind(verdict: str) -> str:
    """'TP', 'FP' or 'BORDERLINE' from a free-text verdict cell."""
    v = verdict.upper()
    return "FP" if v.startswith("FP") else "BORDERLINE" if v.startswith("BORDERLINE") else "TP"


def sample_and_count(cfg: dict[str, Any], per_source: int, replay: bool) -> dict[str, dict[str, Any]]:
    """One pass per filtered source: reservoir sample of drops plus rule-change counts."""
    rng = random.Random(int(cfg["minhash"]["seed"]))
    result: dict[str, dict[str, Any]] = {}
    for name, src in cfg["sources"].items():
        if not src.get("nsfw_filter"):
            continue
        sample: list[tuple[str, str | None, str, str, str]] = []
        seen = drops = 0
        changes: Counter[str] = Counter()
        changed_urls: dict[str, list[str]] = {"now_kept": [], "now_dropped": []}
        for rec in iter_source(name):
            text = clean_text(rec.text)
            if n_words(text) < int(cfg["cleaning"]["min_words"]):
                continue
            seen += 1
            old = legacy_reason(text, rec.url, cfg) if replay else None
            new = nsfw_reason(text, rec.url)
            if replay:
                changes[f"old_{old or 'kept'}"] += 1
                changes[f"new_{new or 'kept'}"] += 1
                if old and not new:
                    changes["now_kept"] += 1
                    if len(changed_urls["now_kept"]) < 200:
                        changed_urls["now_kept"].append(f"{old} -> kept | {rec.url or ''}")
                elif new and not old:
                    changes["now_dropped"] += 1
                    if len(changed_urls["now_dropped"]) < 200:
                        changed_urls["now_dropped"].append(f"kept -> {new} | {rec.url or ''}")
            sampled_rule = old if replay else new
            if not sampled_rule:
                continue
            drops += 1
            fired = legacy_fired(text, rec.url, cfg) if replay else new_fired(text, rec.url)
            item = (rec.doc_id, rec.url, fired, text[:400].replace("\n", " ").replace("|", "/"),
                    new_fired(text, rec.url))
            if len(sample) < per_source:
                sample.append(item)
            elif rng.random() < per_source / drops:
                sample[rng.randrange(per_source)] = item
        result[name] = {"sample": sample, "seen": seen, "drops": drops, "changes": dict(changes),
                        "changed_urls": changed_urls}
        print(f"[{name}] {seen:,} docs, {drops:,} sampled-rule drops, changes {dict(changes)}", flush=True)
    return result


def write_review(cfg: dict[str, Any], res: dict[str, dict[str, Any]], per_source: int) -> None:
    """Default mode: the TP/FP review table (local only)."""
    lines = ["# NSFW filter spot check", "",
             f"{per_source} random drops per filtered source (seed {cfg['minhash']['seed']}). "
             "Fill the verdict column: TP = adult content, FP = innocent.", ""]
    for name, r in res.items():
        lines += [f"## {name}: {r['drops']:,} dropped of {r['seen']:,} ({r['drops'] / max(r['seen'], 1):.1%})", "",
                  "| # | rule fired | url | text (first 400 chars) | verdict |", "|---|---|---|---|---|"]
        for i, (_doc, url, rule, text, _new) in enumerate(r["sample"], 1):
            lines.append(f"| {i} | {rule} | {url or ''} | {text} | |")
        lines.append("")
    out = repo_path("docs/nsfw_spotcheck.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def replay(cfg: dict[str, Any], res: dict[str, dict[str, Any]]) -> int:
    """Compare the reproduced sample to the Prompt 2 review and score the new rule."""
    review_path = repo_path("docs/nsfw_spotcheck.md")
    if not review_path.is_file():
        print("docs/nsfw_spotcheck.md (local Prompt 2 review) not found; cannot replay", file=sys.stderr)
        return 1
    review = parse_local_review(review_path.read_text(encoding="utf-8"))
    code = 0
    pub = ["# NSFW filter D16 replay (redacted)", "",
           "The Prompt 2 spot-check sample, reproduced exactly with the pre-D16 rule (same seed, same "
           "reservoir order, verified row by row against the local review file by URL), re-scored with "
           "the D16 rule. Verdicts are the Prompt 2 verdicts confirmed by Akib. URLs and text are omitted; "
           "the local file docs/nsfw_spotcheck.md has them.", ""]
    local = ["# NSFW D16 replay: changed documents (LOCAL ONLY, contains URLs)", ""]
    for name, r in res.items():
        rows = review.get(name, [])
        urls = [u or "" for _d, u, *_ in r["sample"]]
        match = len(rows) == len(urls) and all(a == b for a, (b, _v) in zip(urls, rows))
        if not match:
            print(f"[{name}] replay sample does NOT match the Prompt 2 review", file=sys.stderr)
            code = 1
        ch = r["changes"]
        pub += [f"## {name}", "",
                f"Sample reproduced: {'yes, all ' + str(len(rows)) + ' rows match' if match else 'NO (mismatch)'}. "
                f"Docs checked: {r['seen']:,}. Old rule drops: {r['seen'] - ch.get('old_kept', 0):,}. "
                f"New rule drops: {r['seen'] - ch.get('new_kept', 0):,}. "
                f"Old-dropped now kept: {ch.get('now_kept', 0):,}. Old-kept now dropped: {ch.get('now_dropped', 0):,}.",
                "", "| # | old rule fired | new rule | Prompt 2 verdict | outcome |", "|---|---|---|---|---|"]
        tally: Counter[str] = Counter()
        for i, ((_d, _u, old_rule, _t, new_rule), (_url, verdict)) in enumerate(zip(r["sample"], rows), 1):
            kind = verdict_kind(verdict)
            kept = new_rule == "kept"
            if kind == "FP":
                outcome = "FP recovered (kept)" if kept else "FP STILL DROPPED"
                if not kept and name == "fineweb2_ben_beng":
                    code = 1
            elif kind == "TP":
                outcome = "TP REGRESSION (now kept)" if kept else "TP still dropped"
            else:
                outcome = "borderline, now kept" if kept else "borderline, still dropped"
            tally[outcome] += 1
            pub.append(f"| {i} | {old_rule} | {new_rule} | {kind} | {outcome} |")
        pub += ["", "Tally: " + ", ".join(f"{k}: {v}" for k, v in sorted(tally.items())), ""]
        local += [f"## {name}", "", "### old-dropped, now kept (first 200)", ""]
        local += [f"- {x}" for x in r["changed_urls"]["now_kept"]]
        local += ["", "### old-kept, now dropped (first 200)", ""]
        local += [f"- {x}" for x in r["changed_urls"]["now_dropped"]] + [""]
    repo_path("docs/nsfw_replay.md").write_text("\n".join(pub) + "\n", encoding="utf-8")
    repo_path("docs/nsfw_replay_local.md").write_text("\n".join(local) + "\n", encoding="utf-8")
    print("wrote docs/nsfw_replay.md and docs/nsfw_replay_local.md")
    print("\n".join(pub))
    return code


def main() -> int:
    """Run the spot check or the D16 replay."""
    p = argparse.ArgumentParser()
    p.add_argument("--per-source", type=int, default=20)
    p.add_argument("--replay", action="store_true", help="D16: reproduce the Prompt 2 sample and re-score it")
    args = p.parse_args()
    cfg = load_config()
    setup_env(cfg)
    res = sample_and_count(cfg, args.per_source, args.replay)
    if args.replay:
        return replay(cfg, res)
    write_review(cfg, res, args.per_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the Prompt 2 report's tables from data/audit.json (no hand-copied numbers).

Prints Markdown for: per-source counts and drop reasons, script mix, token accounting and
tokens/param, splits, leakage categorization, and the REDACTED NSFW spot-check table
(source, rule type, first 30 chars of URL, verdict; never sample text).

Usage:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m scripts.report_tables > tables.md
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from data.common import load_config, repo_path, setup_env

ORDER = ["bnsentmix_train", "vashantor", "banglishdepyt", "banglishrev",
         "fineweb2_ben_latn", "wikipedia_bn", "fineweb2_ben_beng"]


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _m(x: float) -> str:
    return f"{x / 1e6:,.1f}M"


def sources_table(a: dict[str, Any]) -> list[str]:
    """Per-source documents in/kept and drop reasons."""
    reasons = sorted({r for s in a["sources"].values() for r in s["dropped"]})
    out = ["| Source | Docs in | Docs kept | Kept % | " + " | ".join(reasons) + " | Words kept | Words/doc (median) |",
           "|---|---:|---:|---:|" + "---:|" * len(reasons) + "---:|---:|"]
    tot_in = tot_kept = tot_words = 0
    tot_drop: Counter[str] = Counter()
    for name in ORDER:
        s = a["sources"][name]
        tot_in += s["docs_in"]
        tot_kept += s["docs_kept"]
        tot_words += s["words"]
        tot_drop.update(s["dropped"])
        drops = " | ".join(_n(s["dropped"].get(r, 0)) for r in reasons)
        out.append(f"| {name} | {_n(s['docs_in'])} | {_n(s['docs_kept'])} | {s['docs_kept'] / s['docs_in']:.1%} | "
                   f"{drops} | {_n(s['words'])} | {s['words_per_doc']['median']} |")
    drops = " | ".join(f"**{_n(tot_drop[r])}**" for r in reasons)
    out.append(f"| **Total** | **{_n(tot_in)}** | **{_n(tot_kept)}** | **{tot_kept / tot_in:.1%}** | {drops} | **{_n(tot_words)}** | |")
    return out


def script_table(a: dict[str, Any]) -> list[str]:
    """Word-level and document-level script mix per source."""
    out = ["| Source | Latin words | Mixed words | Bengali words | Other | Target-script share | Docs latin / mixed / bengali |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for name in ORDER:
        ws, ds = a["sources"][name]["word_script"], a["sources"][name]["doc_script"]
        tot = sum(ws.values()) or 1
        tgt = (ws.get("latin", 0) + ws.get("mixed", 0)) / tot
        out.append(f"| {name} | {_n(ws.get('latin', 0))} | {_n(ws.get('mixed', 0))} | {_n(ws.get('bengali', 0))} | "
                   f"{_n(ws.get('other', 0))} | {tgt:.1%} | {_n(ds.get('latin', 0))} / {_n(ds.get('mixed', 0))} / {_n(ds.get('bengali', 0))} |")
    t = a["tokens"]["words"]["word_script_totals"]
    tot = sum(t.values())
    out.append(f"| **Total** | **{_n(t.get('latin', 0))}** | **{_n(t.get('mixed', 0))}** | **{_n(t.get('bengali', 0))}** | "
               f"**{_n(t.get('other', 0))}** | **{(t.get('latin', 0) + t.get('mixed', 0)) / tot:.1%}** | |")
    return out


def token_table(a: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Token estimates (low/mid/high) and tokens/param for 30M and 50M."""
    t = a["tokens"]
    est = t["tokens_est"]
    params = {k: int(v) for k, v in cfg["tokens"]["model_params"].items()}
    out = ["| Basis | Low (×1.6) | **Mid (×2.0)** | High (×2.5) | tok/param 30M (low / mid / high) | tok/param 50M (low / mid / high) |",
           "|---|---:|---:|---:|---|---|"]
    rows = [("All unique tokens", est["all"]), ("Target script (latin + mixed words)", est["target_script"]),
            ("Bengali script", est["bengali_script"])]
    for label, e in rows:
        per = {k: " / ".join(f"{e[b] / p:.2f}" for b in ("low", "mid", "high")) for k, p in params.items()}
        out.append(f"| {label} | {_m(e['low'])} | **{_m(e['mid'])}** | {_m(e['high'])} | {per['30M']} | {per['50M']} |")
    b = t["training_budget"]
    out.append(f"| Training budget (PRD commitment) | | **{_m(b)}** | | {b / params['30M']:.2f} | {b / params['50M']:.2f} |")
    return out


def per_source_tokens(a: dict[str, Any]) -> list[str]:
    """Mid token estimate per source."""
    out = ["| Source | Words | ≈ Tokens (mid) |", "|---|---:|---:|"]
    for name in ORDER:
        out.append(f"| {name} | {_n(a['sources'][name]['words'])} | {_m(a['tokens']['per_source_tokens_mid'][name])} |")
    return out


def splits_table(a: dict[str, Any]) -> list[str]:
    """BnSentMix split sizes and label shares."""
    sp = a["splits"]["splits"]
    labels = list(next(iter(sp.values()))["labels"])
    out = ["| Split | Rows | Share | " + " | ".join(labels) + " |", "|---|---:|---:|" + "---:|" * len(labels)]
    for name, v in sp.items():
        cells = " | ".join(f"{v['labels'][lab]:,} ({v['labels'][lab] / v['rows']:.1%})" for lab in labels)
        out.append(f"| {name} | {v['rows']:,} | {v['share']:.1%} | {cells} |")
    return out


def leakage_table(a: dict[str, Any]) -> list[str]:
    """Hits by source × split/match, plus formulaic (non-gated) samples."""
    lk = a["leakage"]
    keys = sorted({k for v in lk["by_source"].values() for k in v})
    out = ["| Source | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for src, v in lk["by_source"].items():
        out.append(f"| {src} | " + " | ".join(str(v.get(k, 0)) for k in keys) + " |")
    out += ["", f"Gating hits: **{lk['gating_hits']}** (unique samples: {lk['gating_unique_samples']})", "",
            "Formulaic dev/test samples (logged, not gated, D14):", "",
            "| Split | Row | Words | Doc frequency | Reason | Text |", "|---|---:|---:|---:|---|---|"]
    for f in lk["formulaic_checked_samples"]:
        out.append(f"| {f['split']} | {f['row_id']} | {f['words']} | {f['doc_frequency']} | {f['reason']} | `{f['text']}` |")
    return out


def nsfw_redacted() -> list[str]:
    """Redacted spot-check table: source, rule type, first 30 chars of URL, verdict label."""
    path = repo_path("docs/nsfw_spotcheck.md")
    out = ["| Source | # | Rule | URL (first 30 chars) | Verdict |", "|---|---:|---|---|---|"]
    source = ""
    counts: dict[str, Counter[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and ":" in line and "dropped" in line:
            source = line[3:].split(":")[0]
            counts[source] = Counter()
        elif source and line.startswith("| ") and not line.startswith("| #") and line.count("|") >= 6:
            cells = [c.strip() for c in line.split("|")]
            num, rule, url, verdict = cells[1], cells[2], cells[3], cells[-2]
            label = verdict.split(":")[0].split(" ")[0]
            counts[source][label] += 1
            out.append(f"| {source} | {num} | {rule.split(':')[0]} | `{url[:30]}` | {label} |")
    out += ["", "| Source | TP | BORDERLINE | FP |", "|---|---:|---:|---:|"]
    for src, c in counts.items():
        out.append(f"| {src} | {c['TP']} | {c['BORDERLINE']} | {c['FP']} |")
    return out


def main() -> int:
    """Print all report tables."""
    cfg = load_config()
    setup_env(cfg)
    a = json.loads(repo_path(cfg["paths"]["audit_json"]).read_text(encoding="utf-8"))
    sections = {
        "SOURCES": sources_table(a), "SCRIPT MIX": script_table(a), "TOKENS": token_table(a, cfg),
        "TOKENS PER SOURCE": per_source_tokens(a), "SPLITS": splits_table(a), "LEAKAGE": leakage_table(a),
        "NSFW (REDACTED)": nsfw_redacted(),
    }
    for title, lines in sections.items():
        print(f"<!-- {title} -->")
        print("\n".join(lines))
        print()
    print(f"<!-- META --> download_bytes={a['download_bytes']} peak_rss_mb={a['peak_rss_mb']} "
          f"majority={json.dumps(a['majority_baseline_macro_f1'])} dedup={json.dumps({k: v for k, v in a['splits']['dedup'].items() if k != 'conflicts'})} "
          f"near_dup={json.dumps(a['splits']['near_dup'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

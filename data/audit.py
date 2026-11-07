"""Prompt 2 audit: splits → corpus build → statistics → token accounting → leakage gate.

Writes:
- data/audit.json            (committed) per-source stats, token estimates, tokens/param,
                             majority baseline, leakage summary
- docs/leakage_hits.jsonl    (committed) every gating hit (dev/test samples found)
- data/processed/_leakage_all.jsonl  (gitignored) every hit incl. train and soft matches

Exits 1 if any BnSentMix dev/test sample is found in pretraining text (R5).
Nothing is removed: the hits are listed for diagnosis.

Usage:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.audit [--reuse]
      --reuse  skip the corpus rebuild and re-audit data/processed/_corpus_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Any

from data.common import load_config, peak_rss_mb, repo_path, setup_env, write_json

TARGET_SCRIPTS = ("latin", "mixed")


def macro_f1_constant(labels: list[int], predict: int, classes: list[int]) -> float:
    """Macro-F1 of always predicting `predict`."""
    f1s = []
    n_pred = len(labels)
    for c in classes:
        tp = sum(1 for y in labels if y == c) if c == predict else 0
        support = sum(1 for y in labels if y == c)
        prec = tp / n_pred if c == predict else 0.0
        rec = tp / support if support else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def token_accounting(stats: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Estimated tokens (words × tokens/word) overall and in the target script, plus tokens/param."""
    t = cfg["tokens"]
    mid, lo, hi = float(t["per_word_mid"]), float(t["per_word_low"]), float(t["per_word_high"])
    words_total = sum(s["words"] for s in stats.values())
    ws: Counter[str] = Counter()
    for s in stats.values():
        ws.update(s["word_script"])
    target_words = sum(ws[k] for k in TARGET_SCRIPTS)
    bengali_words = ws["bengali"]
    params = {k: int(v) for k, v in t["model_params"].items()}
    budget = int(t["training_budget"])

    def est(words: int) -> dict[str, int]:
        return {"mid": round(words * mid), "low": round(words * lo), "high": round(words * hi)}

    totals = {"all": est(words_total), "target_script": est(target_words), "bengali_script": est(bengali_words)}
    per_param = {
        basis: {name: round(n / p, 2) for name, p in params.items()}
        for basis, n in (
            ("training_budget", budget),
            ("all_unique_mid", totals["all"]["mid"]),
            ("target_script_unique_mid", totals["target_script"]["mid"]),
        )
    }
    return {
        "method": f"tokens = whitespace words × {mid} (range {lo}–{hi}); exact counts in Prompt 3",
        "words": {"all": words_total, "target_script": target_words, "bengali_script": bengali_words,
                  "word_script_totals": dict(ws)},
        "tokens_est": totals,
        "per_source_tokens_mid": {k: round(s["words"] * mid) for k, s in stats.items()},
        "training_budget": budget,
        "tokens_per_param": per_param,
    }


def _hard(h: dict[str, Any]) -> bool:
    return bool(h["match"] != "near_dup_soft")


def sample_doc_frequency(hits: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    """Distinct pretraining documents (source, doc_id) each sample matched (hard matches only)."""
    docs: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for h in hits:
        if _hard(h):
            docs[(h["split"], int(h["row_id"]))].add((h["source"], str(h["doc_id"])))
    return {k: len(v) for k, v in docs.items()}


def formulaic_samples(hits: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[tuple[str, int], str]:
    """D14: samples that are formulaic, mapped to the reason ('allowlist' or 'doc_frequency').

    Formulaic = on the allowlist, or matched in >= min_doc_frequency distinct docs.
    `max_words` (null = no cap) optionally limits which samples may qualify.
    """
    from data.clean import normalize_for_match

    f = cfg["leakage"]["formulaic"]
    allow = {normalize_for_match(p) for p in f["allowlist"]}
    max_words = f.get("max_words")
    min_df = int(f["min_doc_frequency"])
    df = sample_doc_frequency(hits)
    out: dict[tuple[str, int], str] = {}
    for h in hits:
        key = (h["split"], int(h["row_id"]))
        if key in out or not _hard(h):
            continue
        if max_words is not None and len(h["sample_text"].split()) > int(max_words):
            continue
        if normalize_for_match(h["sample_text"]) in allow:
            out[key] = "allowlist"
        elif df.get(key, 0) >= min_df:
            out[key] = "doc_frequency"
    return out


def gating_hits(hits: list[dict[str, Any]], checked: set[str], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Hits that fail the gate: checked split, hard match type, sample not formulaic (D14)."""
    formulaic = formulaic_samples(hits, cfg)
    return [h for h in hits if h["split"] in checked and _hard(h) and (h["split"], int(h["row_id"])) not in formulaic]


def gate_exit_code(hits: list[dict[str, Any]], checked: set[str], cfg: dict[str, Any]) -> int:
    """1 if any gating hit exists, else 0."""
    return 1 if gating_hits(hits, checked, cfg) else 0


def leakage_summary(hits: list[dict[str, Any]], checked: set[str], cfg: dict[str, Any]) -> dict[str, Any]:
    """Counts by source × split × match type, formulaic (non-gated) samples, and the gating subset."""
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for h in hits:
        table[h["source"]][f"{h['split']}/{h['match']}"] += 1
    formulaic = formulaic_samples(hits, cfg)
    df = sample_doc_frequency(hits)
    text = {(h["split"], int(h["row_id"])): h["sample_text"] for h in hits}
    checked_formulaic = {k: v for k, v in formulaic.items() if k[0] in checked}
    gating = gating_hits(hits, checked, cfg)
    unique = {(h["split"], int(h["row_id"])) for h in gating}
    return {
        "by_source": {k: dict(v) for k, v in table.items()},
        "formulaic_checked_samples": [
            {"split": k[0], "row_id": k[1], "reason": r, "doc_frequency": df.get(k, 0),
             "words": len(text[k].split()), "text": text[k]}
            for k, r in sorted(checked_formulaic.items(), key=lambda kv: -df.get(kv[0], 0))
        ],
        "gating_hits": len(gating),
        "gating_unique_samples": len(unique),
        "gating_samples_doc_frequency": {f"{k[0]}:{k[1]}": df.get(k, 0) for k in sorted(unique)},
    }


def main() -> int:
    """Run the audit; return 1 if the leakage gate fails."""
    parser = argparse.ArgumentParser(description="Prompt 2 data audit + leakage gate.")
    parser.add_argument("--reuse", action="store_true", help="reuse data/processed/_corpus_stats.json")
    args = parser.parse_args()
    cfg = load_config()
    setup_env(cfg)

    from data.splits import build_splits, load_split_rows

    split_report = build_splits(cfg)
    stats_path = repo_path(cfg["paths"]["processed_dir"]) / "_corpus_stats.json"
    if args.reuse:
        saved = json.loads(stats_path.read_text(encoding="utf-8"))
        stats, hits = saved["stats"], saved["hits"]
    else:
        from data.corpus import build_corpus

        st, hit_objs = build_corpus(cfg)
        stats, hits = st, [h.as_dict() for h in hit_objs]
        write_json(stats_path, {"stats": stats, "hits": hits})

    train_labels = [r["label"] for r in load_split_rows(cfg, "train")]
    majority = Counter(train_labels).most_common(1)[0][0]
    classes = sorted(int(k) for k in cfg["bnsentmix"]["label_names"])
    baseline = {
        "majority_class": cfg["bnsentmix"]["label_names"][majority],
        **{split: round(macro_f1_constant([r["label"] for r in load_split_rows(cfg, split)], majority, classes), 4)
           for split in ("dev", "test")},
    }

    checked = set(cfg["leakage"]["splits_checked"])
    leak = leakage_summary(hits, checked, cfg)
    manifest_path = repo_path(cfg["paths"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    public_stats = {k: {kk: vv for kk, vv in v.items() if kk != "spot_check_nsfw_dropped"} for k, v in stats.items()}
    tok: dict[str, Any] = token_accounting(stats, cfg)
    audit: dict[str, Any] = {
        "download_bytes": manifest.get("total_download_bytes"),
        "splits": split_report,
        "majority_baseline_macro_f1": baseline,
        "sources": public_stats,
        "tokens": tok,
        "leakage": leak,
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    write_json(repo_path(cfg["paths"]["audit_json"]), audit)

    all_path = repo_path(cfg["paths"]["processed_dir"]) / "_leakage_all.jsonl"
    with all_path.open("w", encoding="utf-8", newline="\n") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    gating = gating_hits(hits, checked, cfg)
    gate_path = repo_path(cfg["paths"]["leakage_hits"])
    with gate_path.open("w", encoding="utf-8", newline="\n") as f:
        for h in sorted(gating, key=lambda x: (x["split"], x["row_id"], x["source"])):
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    print("\n=== AUDIT SUMMARY ===")
    print(f"pretraining words: {tok['words']['all']:,} (target script {tok['words']['target_script']:,})")
    print(f"est. tokens (mid): {tok['tokens_est']['all']['mid']:,}; target script {tok['tokens_est']['target_script']['mid']:,}")
    print(f"tokens/param: {json.dumps(tok['tokens_per_param'])}")
    print(f"majority baseline macro-F1: {baseline}")
    print(f"leakage by source: {json.dumps(leak['by_source'])}")

    if gating:
        print(f"\nLEAKAGE GATE FAILED: {leak['gating_unique_samples']} BnSentMix {sorted(checked)} samples "
              f"found in pretraining text ({len(gating)} hits). Nothing was removed.", file=sys.stderr)
        by_sample: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for h in gating:
            by_sample[(h["split"], h["row_id"])].append(h)
        for (split, row_id), hs in sorted(by_sample.items()):
            srcs = Counter(h["source"] for h in hs)
            kinds = Counter(h["match"] for h in hs)
            print(f"  {split} row {row_id}: {hs[0]['sample_text'][:90]!r} | {len(hs)} docs {dict(srcs)} | {dict(kinds)}", file=sys.stderr)
        print(f"Full list: {gate_path.relative_to(repo_path('')).as_posix()}", file=sys.stderr)
    else:
        print(f"\nleakage gate: PASS (0 non-formulaic dev/test samples found; "
              f"{len(leak['formulaic_checked_samples'])} formulaic samples logged, not gated)")
    return gate_exit_code(hits, checked, cfg)


if __name__ == "__main__":
    raise SystemExit(main())

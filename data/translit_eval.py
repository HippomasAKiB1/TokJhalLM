"""Stage 0: transliteration quality gate (D17, as defined 2026-11-08).

Measures, on the held-out Vashantor TEST split (2,264 unique sentence pairs):
  - WER and chrF for T3-only, T2-only, T1+T3, T1+T2 (deterministic), and the chosen
    system in sampled mode; standard and dialect rows separately.
  - Per-token error rates on word-aligned standard rows: e_T1 on lexicon-covered
    tokens, e_F for the fallback applied to EVERY token (T2 never saw any test word
    type: `model.exclude_test_words`).
  - T1 token coverage c of the bulk pool (Wikipedia, FineWeb-2 ben_Beng), and the
    projected bulk scores  WER = c*e_T1 + (1-c)*e_F  (chrF projected the same way,
    using word-level chrF of each tier). The LOWEST c governs (worst case).
  - Coverage-filtered volumes at the configured levels (Prompt 3 plan §4, option 1).
The gate (configs/translit.yaml `gate`, pre-registered) takes the worse band of
"full system, standard rows" and "projected bulk".

Writes data/translit_stage0.json and docs/translit_stage0.md.

Usage:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.translit_eval
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from data.common import repo_path, setup_env, write_json
from data.translit import Pair, Transliterator, build_lexicon, load_translit_config, load_vashantor_pairs
from data.translit_rules import (
    BENGALI_WORD_RE,
    eval_words,
    normalize_bengali_word,
    romanize_rules,
    source_words,
)

SENT_SPLIT_RE = re.compile(r"[।!?\n]+")


# ----------------------------------------------------------------------------- metrics

def corpus_wer(hyps: Sequence[list[str]], refs: Sequence[list[str]]) -> float:
    """Corpus WER: total word-level Levenshtein edits / total reference words."""
    edits = total = 0
    for h, r in zip(hyps, refs):
        d = list(range(len(r) + 1))
        for i, hw in enumerate(h, 1):
            prev, d[0] = d[0], i
            for j, rw in enumerate(r, 1):
                cur = min(d[j] + 1, d[j - 1] + 1, prev + (hw != rw))
                prev, d[j] = d[j], cur
        edits += d[len(r)]
        total += len(r)
    return edits / max(total, 1)


def _ngrams(s: str, n: int) -> Counter[str]:
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def corpus_chrf(hyps: Sequence[str], refs: Sequence[str], max_order: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF on a 0-1 scale (Popović 2015; sacrebleu's corpus-statistics method).

    Whitespace is removed; for each order n, matches and totals are summed over all
    segments; precision and recall are averaged over the orders that have any
    n-grams, then combined as F-beta.
    """
    match = [0] * (max_order + 1)
    hyp_tot = [0] * (max_order + 1)
    ref_tot = [0] * (max_order + 1)
    for hyp, ref in zip(hyps, refs):
        hs, rs = "".join(hyp.split()), "".join(ref.split())
        for n in range(1, max_order + 1):
            hc, rc = _ngrams(hs, n), _ngrams(rs, n)
            match[n] += sum((hc & rc).values())
            hyp_tot[n] += sum(hc.values())
            ref_tot[n] += sum(rc.values())
    precs, recs = [], []
    for n in range(1, max_order + 1):
        if hyp_tot[n] == 0 and ref_tot[n] == 0:
            continue
        precs.append(match[n] / hyp_tot[n] if hyp_tot[n] else 0.0)
        recs.append(match[n] / ref_tot[n] if ref_tot[n] else 0.0)
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p == 0 and r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def band(wer: float, chrf: float, gate: dict[str, Any]) -> str:
    """'pass' (200M), 'middle' (150M) or 'stop' per the pre-registered thresholds."""
    if wer >= float(gate["stop_wer_at_or_above"]) or chrf < float(gate["stop_chrf_below"]):
        return "stop"
    if wer < float(gate["pass_wer_below"]) and chrf > float(gate["pass_chrf_above"]):
        return "pass"
    return "middle"


BAND_ORDER = {"pass": 0, "middle": 1, "stop": 2}


# ------------------------------------------------------------------------------ evaluation

def score(pairs: list[Pair], tr: Transliterator, sample: bool, cfg: dict[str, Any]) -> dict[str, Any]:
    """WER and chrF of `tr` on pairs, split by std / dia / all."""
    m = cfg["metrics"]
    hyp_text = [tr.romanize_text(p.bengali, sample=sample) for p in pairs]
    out: dict[str, Any] = {}
    for kind in ("std", "dia", "all"):
        idx = [i for i, p in enumerate(pairs) if kind == "all" or p.kind == kind]
        hw = [eval_words(hyp_text[i]) for i in idx]
        rw = [eval_words(pairs[i].roman) for i in idx]
        out[kind] = {
            "n": len(idx),
            "wer": round(corpus_wer(hw, rw), 4),
            "chrf": round(corpus_chrf([" ".join(x) for x in hw], [" ".join(x) for x in rw],
                                      int(m["chrf_max_order"]), float(m["chrf_beta"])), 4),
        }
    return out


def token_rates(pairs: list[Pair], lex: dict[str, Counter[str]], fallback: Callable[[list[str]], list[str]],
                kind: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Per-token error rates and word-level chrF on word-aligned pairs of one kind."""
    m = cfg["metrics"]
    t1_h, t1_r, f_h, f_r = [], [], [], []
    aligned = 0
    for p in pairs:
        if p.kind != kind:
            continue
        src, ref = source_words(p.bengali), eval_words(p.roman)
        if not src or len(src) != len(ref):
            continue
        aligned += 1
        beng = [(w, r) for w, r in zip(src, ref) if BENGALI_WORD_RE.fullmatch(w)]
        fb = fallback([w for w, _ in beng]) if beng else []
        for (w, r), fh in zip(beng, fb):
            f_h.append(fh)
            f_r.append(r)
            if w in lex:
                t1_h.append(lex[w].most_common(1)[0][0])
                t1_r.append(r)
    def err(h: list[str], r: list[str]) -> float:
        return sum(a != b for a, b in zip(h, r)) / max(len(r), 1)
    o, b = int(m["chrf_max_order"]), float(m["chrf_beta"])
    return {
        "aligned_pairs": aligned,
        "t1_tokens": len(t1_r), "fallback_tokens": len(f_r),
        "e_t1": round(err(t1_h, t1_r), 4), "e_fallback": round(err(f_h, f_r), 4),
        "chrf_t1": round(corpus_chrf(t1_h, t1_r, o, b), 4), "chrf_fallback": round(corpus_chrf(f_h, f_r, o, b), 4),
    }


def bulk_sample(source: str, n_docs: int) -> list[str]:
    """`n_docs` texts at an even stride across all shards of a processed source."""
    import pyarrow as pa

    paths = sorted(repo_path(f"data/processed/{source}").glob("shard-*.arrow"))
    counts = []
    for p in paths:
        with pa.memory_map(str(p)) as mm:
            counts.append(pa.ipc.open_file(mm).read_all().num_rows)
    total = sum(counts)
    want = set(range(0, total, max(1, total // n_docs)))
    out: list[str] = []
    base = 0
    for p, c in zip(paths, counts):
        rows = sorted(i - base for i in want if base <= i < base + c)
        if rows:
            with pa.memory_map(str(p)) as mm:
                col = pa.ipc.open_file(mm).read_all().column("text")
                out.extend(col[i].as_py() for i in rows)
        base += c
    return out[:n_docs]


def coverage(texts: list[str], lex: dict[str, Counter[str]], levels: list[float]) -> dict[str, Any]:
    """T1 token coverage of Bengali words (digit-only words excluded) and coverage-filter volumes."""
    cov_tok = tot_tok = 0
    sent_stats: list[tuple[int, int]] = []
    for t in texts:
        for sent in SENT_SPLIT_RE.split(t):
            words = [normalize_bengali_word(w) for w in BENGALI_WORD_RE.findall(sent)]
            words = [w for w in words if not all("০" <= ch <= "৯" for ch in w)]
            if not words:
                continue
            k = sum(w in lex for w in words)
            cov_tok += k
            tot_tok += len(words)
            sent_stats.append((k, len(words)))
    filt = {}
    for lv in levels:
        sel = [(k, n) for k, n in sent_stats if k / n >= lv]
        kept = sum(n for _, n in sel)
        filt[str(lv)] = {"word_share_kept": round(kept / max(tot_tok, 1), 4),
                         "coverage_in_subset": round(sum(k for k, _ in sel) / max(kept, 1), 4)}
    return {"docs": len(texts), "bengali_tokens": tot_tok, "c": round(cov_tok / max(tot_tok, 1), 4), "filtered": filt}


def project(c: float, rates: dict[str, Any]) -> dict[str, float]:
    """Linear projection of WER and chrF for bulk text with T1 coverage c."""
    return {"wer": round(c * rates["e_t1"] + (1 - c) * rates["e_fallback"], 4),
            "chrf": round(c * rates["chrf_t1"] + (1 - c) * rates["chrf_fallback"], 4)}


def mark_fallback(text: str, tr: Transliterator) -> str:
    """Romanized text with fallback-routed words marked by a trailing *."""
    from data.translit_rules import finalize_other, segments

    out = []
    for is_b, s in segments(text):
        if not is_b:
            out.append(finalize_other(s))
            continue
        r = tr.romanize_words([s])[0]
        out.append(r if tr.tier(s) == "T1" else r + "*")
    return "".join(out)


def main() -> int:
    """Run Stage 0 and write the JSON and markdown reports."""
    import sys

    cfg = load_translit_config()
    setup_env()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    m, gate = cfg["metrics"], cfg["gate"]
    lex = build_lexicon(cfg)
    test = sorted(set(load_vashantor_pairs("Test", cfg)), key=lambda p: (p.kind, p.region, p.bengali, p.roman))
    # the same standard sentence appears in all 5 regional files: keep unique (kind, bengali, roman)
    seen: set[tuple[str, str, str]] = set()
    uniq: list[Pair] = []
    for p in test:
        key = (p.kind, p.bengali, p.roman)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    print(f"test pairs: {len(uniq):,} unique ({sum(p.kind == 'std' for p in uniq)} std, "
          f"{sum(p.kind == 'dia' for p in uniq)} dia); lexicon {len(lex):,} types", flush=True)

    from data.translit_model import T2

    t2 = T2()

    def t3(ws: list[str]) -> list[str]:
        return [romanize_rules(w) for w in ws]

    systems = {
        "T3_only": Transliterator(None, t3),
        "T2_only": Transliterator(None, t2),
        "T1+T3": Transliterator(lex, t3),
        "T1+T2": Transliterator(lex, t2),
    }
    results: dict[str, Any] = {"test_pairs": len(uniq), "lexicon_types": len(lex), "t2_checkpoint_sha256": t2.sha256}
    results["systems"] = {name: score(uniq, tr, False, cfg) for name, tr in systems.items()}
    for name, r in results["systems"].items():
        print(f"  {name:8s} std WER {r['std']['wer']:.3f} chrF {r['std']['chrf']:.3f} | "
              f"dia WER {r['dia']['wer']:.3f} chrF {r['dia']['chrf']:.3f}", flush=True)

    rates = {fb: {kind: token_rates(uniq, lex, fn, kind, cfg) for kind in ("std", "dia")}
             for fb, fn in (("T2", t2), ("T3", t3))}
    results["token_rates"] = rates
    best_fb = min(("T2", "T3"), key=lambda f: rates[f]["std"]["e_fallback"])
    best = systems[f"T1+{best_fb}"]
    best.temperature = float(cfg["lexicon"]["sample_temperature"])
    best.rng = random.Random(int(m["seed"]))
    results["chosen_fallback"] = best_fb
    results["sampled_mode"] = score(uniq, best, True, cfg)

    cov = {}
    for src in m["coverage_sources"]:
        texts = bulk_sample(src, int(m["coverage_docs_per_source"]))
        cov[src] = coverage(texts, lex, [float(x) for x in m["coverage_filter_levels"]])
        cov[src]["projected"] = project(cov[src]["c"], rates[best_fb]["std"])
        for lv, f in cov[src]["filtered"].items():
            f["projected"] = project(f["coverage_in_subset"], rates[best_fb]["std"])
        print(f"  coverage {src}: c={cov[src]['c']:.3f} projected {cov[src]['projected']}", flush=True)
    results["bulk_coverage"] = cov
    worst_src = min(cov, key=lambda s: cov[s]["c"])

    full_std = results["systems"][f"T1+{best_fb}"]["std"]
    proj = cov[worst_src]["projected"]
    rows = {"full_system_standard": {**full_std, "band": band(full_std["wer"], full_std["chrf"], gate)},
            "projected_bulk": {**proj, "source": worst_src, "c": cov[worst_src]["c"], "band": band(proj["wer"], proj["chrf"], gate)}}
    worst = max((r["band"] for r in rows.values()), key=BAND_ORDER.__getitem__)
    target = {"pass": int(gate["target_tokens_pass"]), "middle": int(gate["target_tokens_middle"]), "stop": 0}[worst]
    results["gate"] = {"rows": rows, "governing_band": worst, "target_tokens": target,
                       "thresholds": {k: gate[k] for k in ("pass_wer_below", "pass_chrf_above", "stop_wer_at_or_above", "stop_chrf_below")}}
    print(f"GATE: {rows} -> {worst} ({target:,} tokens)", flush=True)

    rng = random.Random(int(m["seed"]))
    std = [p for p in uniq if p.kind == "std"]
    dia = [p for p in uniq if p.kind == "dia"]
    n_ex = int(m["examples_test"])
    picks = rng.sample(std, n_ex - n_ex // 3) + rng.sample(dia, n_ex // 3)
    results["examples_test"] = [{"kind": p.kind, "region": p.region, "bengali": p.bengali, "reference": p.roman,
                                 "output": mark_fallback(p.bengali, best)} for p in picks]
    bulk_ex = []
    for src in m["coverage_sources"]:
        sents = [s.strip() for t in bulk_sample(src, 2000) for s in SENT_SPLIT_RE.split(t)
                 if 8 <= len(BENGALI_WORD_RE.findall(s)) <= 25]
        for s in rng.sample(sents, int(m["examples_bulk"]) // len(m["coverage_sources"])):
            bulk_ex.append({"source": src, "bengali": s, "output": mark_fallback(s, best)})
    results["examples_bulk"] = bulk_ex

    stats_path = repo_path(cfg["paths"]["model_dir"]) / "t2_train_stats.json"
    if stats_path.is_file():
        st = json.loads(stats_path.read_text(encoding="utf-8"))
        st.pop("gpu_temp_log", None)
        results["t2_training"] = st
    write_json(repo_path(cfg["paths"]["stage0_json"]), results)
    repo_path(cfg["paths"]["stage0_report"]).write_text(render(results, cfg), encoding="utf-8")
    print(f"wrote {cfg['paths']['stage0_json']} and {cfg['paths']['stage0_report']}")
    return 0


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(r: dict[str, Any], cfg: dict[str, Any]) -> str:
    """The Stage 0 markdown report (numbers only; the recommendation is added by hand)."""
    g = r["gate"]
    th = g["thresholds"]
    L = ["# Stage 0: transliteration quality gate (D17)", "",
         f"Generated by `python -m data.translit_eval` from `data/translit_stage0.json`. "
         f"Thresholds pre-registered in `configs/translit.yaml` (commit 46e430e) before this run: "
         f"pass = WER < {_pct(th['pass_wer_below'])} AND chrF > {th['pass_chrf_above']}; "
         f"stop = WER ≥ {_pct(th['stop_wer_at_or_above'])} OR chrF < {th['stop_chrf_below']}; otherwise middle (150M).", "",
         "## Gate", "",
         "| Row | WER | chrF | Band |", "|---|---:|---:|---|"]
    fs, pb = g["rows"]["full_system_standard"], g["rows"]["projected_bulk"]
    L.append(f"| Full system (T1+{r['chosen_fallback']}), Vashantor test, standard rows (n={fs['n']}) | {_pct(fs['wer'])} | {fs['chrf']:.3f} | {fs['band']} |")
    L.append(f"| Projected bulk ({pb['source']}, lowest coverage c = {pb['c']:.3f}) | {_pct(pb['wer'])} | {pb['chrf']:.3f} | {pb['band']} |")
    L += ["", f"**Governing band (worst row): {g['governing_band']} → target {g['target_tokens']:,} tokens.**", "",
          "## All systems on Vashantor test (deterministic)", "",
          "| System | std WER | std chrF | dialect WER | dialect chrF | all WER | all chrF |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in r["systems"].items():
        L.append(f"| {name} | {_pct(s['std']['wer'])} | {s['std']['chrf']:.3f} | {_pct(s['dia']['wer'])} | "
                 f"{s['dia']['chrf']:.3f} | {_pct(s['all']['wer'])} | {s['all']['chrf']:.3f} |")
    sm = r["sampled_mode"]
    L += ["", f"Sampled mode (T1+{r['chosen_fallback']}, T1 spelling variants sampled ∝ count^(1/T)): "
          f"std WER {_pct(sm['std']['wer'])}, chrF {sm['std']['chrf']:.3f}; dialect WER {_pct(sm['dia']['wer'])}, chrF {sm['dia']['chrf']:.3f}. "
          "Not gated; it measures how far spelling variation moves the output from the single reference.", "",
          "## Per-tier token error rates (word-aligned test pairs)", "",
          "Fallback rates apply the fallback to every Bengali token; T2 never saw any test word type in training.", "",
          "| Fallback | Rows | aligned pairs | T1 tokens | e_T1 | chrF_T1 | fallback tokens | e_fallback | chrF_fallback |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for fb, kinds in r["token_rates"].items():
        for kind, t in kinds.items():
            L.append(f"| {fb} | {kind} | {t['aligned_pairs']} | {t['t1_tokens']} | {_pct(t['e_t1'])} | {t['chrf_t1']:.3f} | "
                     f"{t['fallback_tokens']} | {_pct(t['e_fallback'])} | {t['chrf_fallback']:.3f} |")
    L += ["", f"Chosen fallback: **{r['chosen_fallback']}** (lower e_fallback on standard rows).", "",
          "## Bulk coverage and projection", "",
          "Projected WER = c × e_T1 + (1 − c) × e_fallback (chrF the same way), using standard-row rates.", "",
          "| Source | docs | Bengali tokens | T1 coverage c | projected WER | projected chrF |", "|---|---:|---:|---:|---:|---:|"]
    for src, c in r["bulk_coverage"].items():
        L.append(f"| {src} | {c['docs']:,} | {c['bengali_tokens']:,} | {c['c']:.3f} | {_pct(c['projected']['wer'])} | {c['projected']['chrf']:.3f} |")
    L += ["", "### Coverage-filtered transliteration (plan §4 option 1)", "",
          "Keep only sentences whose T1 coverage is at least the level.", "",
          "| Source | level | share of Bengali words kept | coverage in subset | projected WER | projected chrF |", "|---|---:|---:|---:|---:|---:|"]
    for src, c in r["bulk_coverage"].items():
        for lv, f in c["filtered"].items():
            L.append(f"| {src} | {lv} | {_pct(f['word_share_kept'])} | {f['coverage_in_subset']:.3f} | {_pct(f['projected']['wer'])} | {f['projected']['chrf']:.3f} |")
    L += ["", "## Examples: Vashantor test (output marks fallback-routed words with *)", ""]
    for i, e in enumerate(r["examples_test"], 1):
        L += [f"{i}. [{e['kind']}, {e['region']}] {e['bengali']}", f"   - reference: `{e['reference']}`", f"   - output:    `{e['output']}`"]
    L += ["", "## Examples: bulk text (no reference exists; * = fallback-routed)", ""]
    for i, e in enumerate(r["examples_bulk"], 1):
        L += [f"{i}. [{e['source']}] {e['bengali']}", f"   - output: `{e['output']}`"]
    if "t2_training" in r:
        t = r["t2_training"]
        L += ["", "## T2 training", "",
              f"{t['params'] / 1e6:.2f}M params, trained from scratch. Aksharantar {t['aksharantar_pairs']:,} pairs; "
              f"Vashantor {t['vashantor_train_pairs']:,} training pairs, {t['vashantor_dev_types']:,} dev word types; "
              f"{t['excluded_test_types']:,} test word types excluded from all training. "
              f"Pretrain {t['pretrain']['seconds']:,.0f} s (dev word acc {t['pretrain']['best_dev_acc']:.3f}); "
              f"fine-tune {t['finetune']['seconds']:,.0f} s (best dev word acc {t['finetune']['best_dev_acc']:.3f}, epoch {t['finetune']['best_epoch']}). "
              f"Max GPU temperature {t['gpu_temp_max_c']} °C. Checkpoint sha256 `{r['t2_checkpoint_sha256']}` (runs/translit/, gitignored)."]
    L += ["", "## Recommendation", "", "_Added after review of the numbers above._", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

# TokJhalLM — Decision Record

Binding decisions and PRD amendments approved by Akib (execution lead). Where
this file and docs/PRD.md disagree, this file wins for the specific point
recorded; everything else in the PRD is unchanged. Evidence and alternatives
for each decision are in docs/IMPLEMENTATION_PLAN.md §3.

Approved: 2026-11-07.

---

## D1 — Training precision

D1 (precision): PRD §6.3 specifies mixed precision. Measurement on 
GTX 1650 (TU117) shows fp32 is faster with negligible memory difference 
at 50M scale. Adopt fp32 primary, fp16 fallback. Confirm at Prompt 4 
with real 50M model. Rationale: fp16's memory advantage becomes decisive 
at 1B+ params, not 50M.

Implementation:
- `configs/base.yaml: training.precision: fp32`. Training loops support both
  `fp32` and `fp16` from config; `torch.amp.GradScaler` only for `fp16`.
- Prompt 4 validation: benchmark BOTH precisions on the real 50M model
  (forward + backward + optimizer step, B=2, T=128, L=8, H=640, grad
  checkpointing on). Report peak VRAM and step time for each. fp32 fits and is
  faster → lock in. fp32 OOMs → switch to fp16 and record it here.

Note: with autocast, weights/grads/optimizer state stay fp32, so fp16's memory
saving is in activations (scales with batch × sequence × width × depth), which
is why it is negligible at micro-batch 2×128.

## D2 — Model dimensions

| Variant | Config | Layers | Hidden | Heads | FFN | Est. params |
| --- | --- | --- | --- | --- | --- | --- |
| 50M | configs/model_50m.yaml | 8 | 640 | 8 | 1920 | 50.7M |
| 30M | configs/model_30m.yaml | 8 | 480 | 6 | 1440 | 30.0M |

Depth and head_dim (80) fixed per PRD §6.1; only width varies.
`configs/phase0.yaml` references both files; later phases use them as the
primary model configs. `configs/base.yaml`'s model section mirrors the 50M file
(enforced by test).

## D3 — ELECTRA generator

2 layers at discriminator width, shared token embeddings, mask rate 0.15,
RTD loss weight 50, ~1.25× MLM compute. Only the discriminator counts toward
the reported parameter count; generator FLOPs count toward the Phase 0 winning
rule. Config: `configs/phase0.yaml: electra`.

## D4 — Phase 0 token budget

Per-run budget is a config value (`configs/phase0.yaml: budget.tokens_per_run`,
default 25M). At the end of Prompt 4, measure throughput for both 30M and 50M
on a 30-second sustained run at the target batch size, cold and after 10
minutes of sustained load (warm). Set the budget from **warm** throughput so
all 4 runs + all fine-tunes finish in ≤48 h wall time.

## D5 — Phase 0 fine-tune seeds (mandatory)

Every Phase 0 candidate is fine-tuned with ≥3 seeds
(`evaluation.finetune_seeds`). Report mean ± std. If candidates' 3-seed
intervals overlap, treat as a tie and apply the smaller/faster preference.

## D6 — BanglaC4 stream cap (amended 2026-11-07, Prompt 2 plan)

Original: cap the BanglaC4/OSCAR stream at 1 GB compressed. If the Prompt 2
audit shows total pretraining tokens < 300M after Wikipedia + task data + this
stream, the cap may be raised, but only after reporting the token total. No
further download without that report.

**Amendment:** "BanglaC4" is not a canonical dataset. The source is
**FineWeb-2 `ben_Beng`** (HuggingFaceFW/fineweb-2, ODC-By), streamed as evenly
spaced row groups across all five files, capped at **0.5 GB compressed**
(≈ 240M tokens at 4.1× compression). The "report before downloading more" rule
still applies.

## D7 — Data strategy: romanized text is the scarce resource; transliteration is central

BnSentMix is 100% Latin-script (romanized) Banglish; most Bengali corpora are
Bengali script, and a 12K vocab shares almost no subwords across scripts.

- Every verified real romanized/code-mixed source is used (BanglishRev,
  BanglishDepYT, FineWeb-2 `ben_Latn` after NSFW filtering, Vashantor,
  BnSentMix train).
- **Transliteration augmentation (PRD §7.1) is the primary mechanism** for
  scaling target-script data, and the first deliverable of Prompt 3, before
  tokenizer training (the tokenizers must see transliterated text).
- Design: lexicon-first, rules-second. Word romanizations are sampled from
  attested spellings (Vashantor's parallel Bengali↔Banglish sentences, which
  use natural Banglish spelling, e.g. "valo"; Aksharantar word pairs, which are
  more scholarly, e.g. "akalpaneey"), with a rule-based mapper (inherent-vowel
  and schwa-deletion heuristics) for unseen words, and stochastic spelling
  variants so the model sees Banglish's real spelling noise. Validated by
  character error rate against the held-out Vashantor test split.
- Framing: tokens/param is set by the compute budget (300M → 10.0 at 30M,
  6.0 at 50M). Transliteration fixes *repetition*: it lets the romanized share
  of that budget be fresh text instead of re-reading the ~60M real romanized
  tokens. Transliterated text carries formal register and no English
  code-mixing, so real Banglish stays upweighted.

## D8 — PolCSBD: skipped

No paper, no inter-annotator agreement; not a benchmark. Not downloaded.

## D9 — Regional-TinyStories: excluded by default

Synthetic, formal Bengali-script children's fiction; wrong script and register.
Revisit only if transliteration augmentation needs a clean regularization
source (then 250 MB prefix at 0–5%).

## D10 — Dialect data

The 600-instance dialect sentiment set (Jawad et al., BLP-2025) is not public;
Akib emails the authors. **Vashantor** (Mendeley DOI 10.17632/bj5jgk878b.2;
arXiv 2311.11142) is used as dialect pretraining text (train + validation
splits). Its test split is held out for transliteration evaluation (D7). No
dialect-sentiment claim until labeled data exists.

## D11 — DepressiveText

Not publicly available; Akib emails the authors. If no data before Phase 3,
compare against the paper's published numbers with explicit caveats. Always a
secondary metric with a CI; never a headline (702-sample test set).

## D12 — BanglishRev license

BanglishRev is CC BY-NC-SA 4.0. Released weights are CC BY-NC-SA 4.0 or
research-only. Model card states: "trained in part on BanglishRev (CC BY-NC-SA
4.0); released weights inherit this restriction."

## D13 — Phase 3 baselines

**BanglishBERT** (csebuetnlp/banglishbert) is the primary self-run baseline
for the romanized task, implemented in Prompt 7. BanglaBERT was pretrained on
Bengali script only; using it as the headline comparison on a romanized task
is a category error. It may be reported as context.

## D14 — Specificity-based leakage gate

The 8-word threshold (proposal 1a) would have excluded 41% of dev 
(810/1,975) and 41% of test (813/1,970) samples from leakage checking, 
including many distinctive ones. Rejected. The specificity gate (1b) 
classes a sample as formulaic only if (a) it appears on an explicit 
allowlist of common greetings, OR (b) it appears in >= 5 distinct 
pretraining documents. Formulaic text is common by definition; a real 
test-set leak is rare. Doc-frequency separates signal from noise; 
word-count does not. Every non-gated hit remains in _leakage_all.jsonl 
and is tabulated in the audit report for transparency.

Implementation notes:
- Config: `configs/data.yaml: leakage.formulaic` (`allowlist`,
  `min_doc_frequency: 5`, `max_words: null`). The 1b draft capped formulaic
  samples at 7 words; per Akib's acceptance tests (an 8-word phrase in 6 docs
  does not gate), there is no length cap: doc frequency is the only escape.
- "Distinct documents" = distinct (source, doc_id) pairs among hard matches
  (exact, containment, near_dup ≥ 0.8), counted after global exact dedup.
- Samples with doc frequency 2–4 are reported to Akib individually.
- Every 8+ word sample excused by doc frequency is listed in the audit report:
  a long, specific sentence in 5+ docs may be a widely copied leak rather than
  a common phrase.
- Remediation for real leaks: never touch the test set or split manifests;
  remove the offending pretraining docs, logged in docs/leakage_removals.md
  (doc_id, source, reason, sha256), after Akib approves the removal list.

## D15 — Phase 0 scope: 50M arm dropped

D15 (Phase 0 scope): 50M arm of Phase 0 dropped. Audit found 29.2M 
target-script tokens (0.58 tokens/param at 50M). Even with aggressive 
transliteration, 50M would need 250M+ transliterated tokens to reach 
5 tokens/param — unrealistic within the compute budget. Phase 0 
now tests MLM vs ELECTRA at 30M/BPE, then BPE vs Unigram at the 
winning objective. The 50M config remains in the repo but is not 
promoted unless transliteration massively outperforms expectations.

Implementation:
- `configs/phase0.yaml`: 3 runs, all 30M (stage a: MLM vs ELECTRA / BPE;
  stage b: Unigram at the winning objective, BPE run reused). `model_sizes`
  lists only 30M; 50M moves to `not_selected`.
- `configs/model_50m.yaml` header: "NOT SELECTED for training run; kept for
  reference and future work."
- `configs/base.yaml` model section now mirrors `model_30m.yaml` (was 50M),
  so no default points at an unselected size.
- Supersedes the size ablation in PRD §5.0 stage (b) and D2's role for 50M;
  D4's 48 h budget now covers 3 runs + 9 fine-tunes.

Corollary (Prompt 3 scope): transliteration is the critical path. Minimum
121M transliterated tokens (≈150M target-script tokens ≈ 5 tokens/param at
30M); stretch 200M (≈6.7 tokens/param). Budget 3–4 days. Validation: word
error rate on Vashantor's held-out test split vs. its ground-truth Banglish,
with a pre-set safety threshold (e.g. < 20% WER). Transliterated text is
never committed; it is regenerated from the Bengali-script source shards.

## D16 — NSFW filter fix (for Prompt 3; not applied in Prompt 2)

The current filter has 0.3% absolute loss on 
ben_Beng but systematically excludes Bengali reporting on gender, 
health, and abuse (7/20 FPs in the spot check are legitimate news). 
The bias is worse than the loss.

Fix, executed in Prompt 3 with a re-run of scripts/nsfw_spotcheck.py:
- Match URL tokens on word boundaries (hostname and path tokens), not raw
  substrings.
- Remove `sex` and `chud` from the URL list.
- Add a Bengali-script explicit-term density check for `ben_Beng` (today the
  term check only covers romanized text).

The full spot-check file (docs/nsfw_spotcheck.md) contains explicit excerpts
and stays local (gitignored); the committed report carries a redacted table.

---

## Amendment A1 — PRD §12 timeline

PRD §12 timeline amendment: Phase 0 ablation cannot precede Prompts 2–4 
because it requires their artifacts (tokenizer, data pipeline, model). 
Actual execution order: Prompt 2 (data) + Prompt 3 (tokenizer) in parallel 
where possible → Prompt 4 (model) → Phase 0 ablation (4 runs + fine-tunes) 
→ Prompt 5 (full pretraining) → Prompt 6-8 (fine-tuning) → Prompt 9 
(evaluation) → Prompt 10 (release) → Prompt 11 (buffer). Total: 9–10 weeks, 
not 8. This is a correction of PRD §12's implied ordering; the phase content 
in PRD §5.0, §8, §9 is unchanged.

Note: Prompt 3's tokenizers are trained on Prompt 2's cleaned corpus, so the
parallel overlap is limited to tooling written while downloads run.

## Constraint C1 — GPU thermals

Multi-day training logs GPU temperature every 30 seconds to wandb. If 
temperature exceeds 85 °C for >5 min, reduce batch size by 1 (minimum 1). 
If still exceeds 85 °C, pause run and report.

- Thresholds live in `configs/base.yaml: monitoring`. When the micro-batch
  drops, grad_accum rises to keep the effective batch constant
  (`keep_effective_batch: true`), so the run's optimization is unchanged.
- Code never changes GPU clocks, power limits or voltage. Physical cooling is
  handled by Akib.

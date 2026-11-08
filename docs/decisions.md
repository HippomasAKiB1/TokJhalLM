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

**Verification requirement (Akib, 2026-11-08, Prompt 2 review):** when the fix
lands in Prompt 3, re-run the spot-check on the SAME 20 samples per source and
confirm the 7 `ben_Beng` news FPs are now kept. If word-boundary matching
doesn't recover them, the fix isn't done. Record this verification in the
Prompt 3 report.

## Prompt 2 review confirmations (2026-11-08)

- Extra excluded document `banglishrev 119136:788` (exact copy of approved #5):
  confirmed. Content-based exclusion is the correct implementation.
- `configs/base.yaml` switched to 30M dims: confirmed (corollary of D15).
- NSFW spot-check verdicts: confirmed, with the D16 verification requirement
  above.

## D17 — Transliteration target is WER-gated, not fixed at 200M

Stage 0 of Prompt 3 MUST measure transliteration quality before bulk
generation:
- Use Vashantor test split (already held out from training)
- Compute WER AND chrF (character-level similarity)
- Report both, plus 10 example outputs, before generating bulk
- STOP for Akib's review

Thresholds:
- WER <20% AND chrF >0.70 → proceed with 200M target
- WER 20-30% OR chrF 0.55-0.70 → proceed with 150M, flag caveat
- WER >30% OR chrF <0.55 → STOP and propose alternative approach
  (small seq2seq trained on Vashantor+Aksharantar, or pivot to
  Bengali-script-first narrative)

Rationale: The 200M transliteration target from Prompt 2's recommendation is
conditional on transliteration quality. WER >30% means the generated text is
essentially random from the model's perspective and won't help. chrF is the
secondary metric because WER is strict — natural Banglish has variable
spellings, so chrF captures "close enough" better. The Vashantor test split is
the right evaluation set because it's parallel Bengali/Banglish with real
Banglish spellings, held out from training.

## D18 — BanglishRev shared-source control experiment

Prompt 7 (fine-tuning) must produce a THREE-row win table, not two:

| Model | Pretraining corpus | BnSentMix F1 (±CI) |
| --- | --- | --- |
| BanglishBERT baseline | N/A (external) | [measured] |
| TokJhalLM-control | Corpus WITHOUT BanglishRev | [measured] |
| TokJhalLM-full | Corpus WITH BanglishRev | [measured] |

Rationale: The audit found 34 hard train hits between BanglishRev and BnSentMix
train rows, indicating shared source sites. This is not test-set leakage
(confirmed via the D14 gate), but it means TokJhalLM-full gets free
pretraining on the target distribution that BanglishBERT does not. The
control experiment isolates the "shared-source boost" from the model's
actual capability. If the boost is large (+5 F1), report it honestly
("BanglishRev coverage accounts for X points"); if small (<2 F1), the
architectural choices stand on their own. Preempting this criticism is more
valuable than avoiding it.

Implementation choice (a: two corpus variants in Prompt 3; b: checkpoint at
the point BanglishRev enters the curriculum; c: second pretraining run) is
deferred to the Prompt 3 plan.

**Implementation (Prompt 3 plan, approved 2026-11-08): branched final phase.**
BanglishRev stays out of the shared first 80% of pretraining. At the branch
point, two final phases run with identical token counts and schedules: *full*
adds BanglishRev to the final mix; *control* replaces BanglishRev's share
proportionally with the other real code-mixed sources and transliterated text.
Extra cost: one final phase (≈ 20% of pretraining), not a second full run.
Plain (a) still needs a second pretraining run; plain (b) compares against a
checkpoint that has simply seen less training.

**Disclosure requirement (Akib).** The Prompt 7 report win-table caption and
the release README win-table caption must state, in exactly this language:
"TokJhalLM-full vs TokJhalLM-control measures the effect of adding BanglishRev
to the final 20% of pretraining. Because both models share the same first 80%
without BanglishRev, this comparison isolates the final-phase effect. A model
trained with BanglishRev across all 100% of pretraining would likely differ;
that variant is not tested here."

**Tokenizer (approved decision 4).** BanglishRev is excluded from tokenizer
training so the control is not advantaged through the vocabulary. Coverage
guard: report Bengali-script fertility on held-out BanglishRev with a tokenizer
trained before AND after the exclusion. If the exclusion degrades
Bengali-script fertility by more than 2% on that sample, revisit the decision.
This is an if-then rule, not a gate: surface the number, apply the rule,
proceed.

## D7-amended — Trained character model in the primary transliteration path

Approved 2026-11-08 (Prompt 3 plan, decision 1). Reasoning, verbatim from the
plan: "D7 says 'rules second' and D17 lists a seq2seq only as the fallback if
the gate fails. The probes show the fallback handles about half of all bulk
tokens and that raw Aksharantar scores 83% WER, so I want the trained character
model built and measured in Stage 0 alongside the rules, not after a failure.
Stage 0 reports both and picks the better one by measurement."

Evidence (read-only probes, 2026-11-08): the Vashantor lexicon covers 41.9% of
Wikipedia and 51.5% of FineWeb-2 `ben_Beng` tokens, so 48–58% of bulk tokens go
through the fallback; Aksharantar used as-is scores 82.7% WER on Vashantor test
(standard rows); BnSentMix train tokens match Vashantor's romanized vocabulary
(56.2%) far more than Aksharantar's (30.4%).

Design: T1 Vashantor lexicon (attested spelling) → T2 character-level
encoder–decoder trained from scratch (Aksharantar pretraining, Vashantor
fine-tuning, so output follows the natural convention) → T3 rule mapper. The
character model is a data tool trained from random initialization; it is not
part of the released encoder, and no pretrained transliteration model is used.

## D17 implementation — gate definition (approved 2026-11-08)

- **Gate on the worse of two rows** (decision 2): "full system, standard rows"
  on Vashantor test, and "projected bulk" = c × WER(T1) + (1 − c) ×
  WER(fallback on unseen words), with c the measured T1 token coverage of the
  bulk pool (chrF projected the same way). Dialect rows are reported, not
  gated. Reason: on the test set the fallback handles only 3.5% of words; on
  bulk text it handles about half.
- **Pre-registered thresholds** in `configs/translit.yaml`, committed before
  Stage 0 runs, not negotiated afterwards: WER < 20% AND chrF > 0.70 → 200M;
  WER 20–30% OR chrF 0.55–0.70 → 150M (with the calibration check); WER ≥ 30%
  OR chrF < 0.55 → STOP. Worst case of the two rows governs.
- **Stage 0.5 calibration check (Addition 1)**, its own gate after Stage 0
  passes: transliterate 500 randomly sampled Wikipedia docs from the pool
  (never the Vashantor test set), hand-review 50 and rate Good / Marginal /
  Bad; report counts and 10 examples. Good + Marginal ≥ 40/50 → proceed as
  planned; 30–39/50 → proceed with the bulk target cut by 25% (200M → 150M,
  150M → 112M); < 30/50 → STOP and apply the Prompt 3 plan's §4 fallbacks.
  Rationale: the projected-bulk formula is a linear approximation; calibration
  tells us whether it holds.

## D19 — Emoji: keep, normalized

Approved 2026-11-08 (Prompt 3 plan, decision 6). Emoji are kept in
pretraining, fine-tuning and inference with identical normalization
everywhere: runs of more than 3 identical emoji are capped at 3; ZWJ and
skin-tone sequences stay intact. No vocabulary slots are reserved; the
tokenizers learn frequent emoji and byte fallback covers the rest.

- Resolves PRD §6.2 (byte fallback so emoji never become `<unk>`) vs §7.2
  ("strip emojis consistently"): §7.2's real requirement is consistency.
- BnSentMix contains 0 emoji in 19,740 rows (stripped upstream), so the
  headline metric is unaffected; demo input and secondary tasks do contain
  emoji.
- The run cap lives in tokenizer-side normalization, NOT in
  `data.clean.clean_text`: changing `clean_text` would change the cleaned-text
  hashes the approved leakage exclusion list is pinned to.

## D20 — Tokenizer training sample: 25% native Bengali script

Approved 2026-11-08 (Prompt 3 plan, decision 5). The tokenizer training
sample (≈ 40M words, fixed seed) is real code-mixed 40%, transliterated 35%,
native Bengali script 25%. Overrides PRD §6.2's "5–10% Wikipedia/BanglaC4":
native Bengali is ≈ 35% of the pretraining mix, and at 5–10% a 12K vocabulary
would fragment it.

## Prompt 3 plan approvals — other points (2026-11-08)

- **No tokenizer winner in Prompt 3** (decision 7): Prompt 3 produces an
  intrinsic report only; Phase 0 run 3 decides BPE vs Unigram (D15).
- **Phase 0 subset uses identical documents for both tokenizers (Addition 3):**
  `data/pipeline.py` samples the 25M-token Phase 0 subset by doc_id once and
  tokenizes that same document set with BPE and with Unigram; a test asserts
  the two subsets have identical doc_id sets. Otherwise the Phase 0 comparison
  would be confounded by data.

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

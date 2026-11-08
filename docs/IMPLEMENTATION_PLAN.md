# TokJhalLM — Implementation Plan

Execution plan for the prompt sequence. docs/PRD.md is the source of truth for
*what*; docs/decisions.md records binding decisions and PRD amendments; this
file tracks *how and in what order*. Current state lives in
docs/EXECUTION_PROTOCOL.md.

Last updated: 2026-11-08 (decisions D1–D16, amendment A1, constraint C1 approved; Prompt 2 complete).

---

## 1. Sequence (amendment A1)

| Step | Scope | PRD phase | Status |
| --- | --- | --- | --- |
| Prompt 1 | Repo scaffold, env, configs, smoke test | — | ✅ Done |
| Prompt 2 | Data download + audit | §14 step 1, §7 | ✅ Done (report: docs/prompt2_report.md) |
| Prompt 3 | Transliteration (critical path, D15) + tokenizers (BPE + Unigram) + streaming data pipeline | Phase 1 | ⏭ Next |
| Prompt 4 | Encoder, MLM/ELECTRA pretraining loop, fine-tune harness, precision + thermal benchmarks | Phase 0 prerequisites | Planned |
| Phase 0 | 3 ablation runs at 30M (D15) + 3-seed fine-tunes, ≤48 h wall time | Phase 0 (§5.0) | Planned |
| Prompt 5 | Full pretraining | Phase 2 | Planned |
| Prompts 6–8 | Fine-tuning (baseline, distillation, secondary tasks) | Phase 3 | Planned |
| Prompt 9 | Evaluation | Phase 4 | Planned |
| Prompt 10 | Release + live demo | Phase 5 | Planned |
| Prompt 11 | Buffer | Week 8 buffer | Planned |

Total: 9–10 weeks. Phase content (PRD §5.0, §8, §9) is unchanged; only the
ordering moved. The split inside Prompts 6–8 (§3) is proposed and can change.

---

## 2. Decisions in force (details: docs/decisions.md)

| ID | Decision | Where it lives |
| --- | --- | --- |
| D1 | fp32 primary, fp16 fallback (OOM only); confirm on the real 50M model at Prompt 4 | `base.yaml: training.precision` |
| D2 | 30M = 8L/480/6h/FFN 1440 (30.0M), selected; 50M = 8L/640/8h/FFN 1920 (50.7M), not selected (D15) | `configs/model_30m.yaml`, `configs/model_50m.yaml` |
| D3 | ELECTRA generator: 2 layers, shared embeddings, mask 0.15, λ = 50 | `phase0.yaml: electra` |
| D4 | 25M tokens/run default; reset from warm throughput so Phase 0 fits ≤48 h | `phase0.yaml: budget` |
| D5 | ≥3 fine-tune seeds per candidate, mean ± std; overlap = tie → smaller wins | `phase0.yaml: evaluation.finetune_seeds` |
| D6 | BanglaC4/OSCAR stream capped at 1 GB compressed; report tokens before raising | Prompt 2 data config |
| C1 | GPU temp logged every 30 s; >85 °C for 5 min → micro-batch −1; still hot → pause + report | `base.yaml: monitoring` |
| D7 | Romanized text is the scarce resource; transliteration is central | Prompt 3 |
| D8–D13 | PolCSBD skipped; TinyStories excluded; Vashantor for dialect text; DepressiveText/dialect authors emailed; BanglishRev NC license; BanglishBERT baseline | `configs/data.yaml`, decisions.md |
| D14 | Specificity leakage gate (allowlist or ≥ 5 docs) | `data.yaml: leakage.formulaic` |
| D15 | Phase 0 = 3 runs at 30M; 50M not selected; transliteration ≥ 121M tokens | `phase0.yaml`, Prompt 3 |
| D16 | NSFW filter fix (word-boundary URL tokens, drop `sex`/`chud`, Bengali term check) | Prompt 3 |

---

## 3. Prompt-by-prompt plan

Conventions for every prompt: plan → approval (if >2 h or >3 files) →
execute → walkthrough. All commands via `E:\Anaconda\envs\tokjhallm\python.exe`.
Every module runnable as `python -m <module>`, with tests in `/tests`.

### Prompt 2 — Data download + audit
**Goal:** every source on disk (or streamed within the D6 cap), measured,
deduplicated, with frozen splits. No tokenizer yet.
- Data source config: origin, license, expected size, role (pretrain /
  finetune / both), BanglaC4 cap = 1 GB compressed (D6).
- `data/download.py`: idempotent, resumable downloads to `data/raw/`
  (gitignored), with checksums. Check free disk space first.
- `data/clean.py`: one normalization function (Unicode NFC, URL/@mention/emoji
  policy per PRD §7.2) shared by *every* model, BanglaBERT included.
- `data/audit.py` → `docs/data_audit.md`: rows, chars, script mix
  (Bengali / Latin / mixed), label distributions, char-based token estimates,
  license notes, and **total pretraining token estimate vs. the 300M target**
  (D6: report before any extra download).
- Cross-dataset dedup (BnSentMix × PolCSBD × DepressiveText): exact plus
  normalized near-duplicate matching; report collisions.
- Frozen splits: train/dev/test IDs in committed manifest files with content
  hashes; leakage check (no test text in any pretraining source).
- Majority-class macro-F1 baseline on BnSentMix dev (confirms the 0.40 floor,
  PRD §5.0).
- **Exit:** audit doc committed; split manifests committed; leakage = 0;
  RAM stays under ~6 GB throughout (stream, never load Wikipedia whole).

### Prompt 3 — Transliteration + tokenizers + data pipeline (PRD Phase 1)
- **First deliverable: transliteration augmentation (D7).** Bengali-script →
  romanized Banglish, lexicon-first (attested spellings from Vashantor
  train/val parallel sentences and Aksharantar word pairs, sampled
  stochastically) with a rule-based fallback for unseen words. **This is the
  critical path (D15): budget 3–4 days.** Target ≥ 121M transliterated tokens
  (≈150M target-script ≈ 5 tokens/param at 30M); stretch 200M. Exit: word
  error rate vs. the held-out Vashantor test split's Banglish column below a
  threshold set in the Prompt 3 plan (e.g. < 20%), plus 50 hand-checked
  outputs. Then transliterate a disjoint slice of Wikipedia/FineWeb-2
  `ben_Beng` (≈330M Bengali-script tokens available). Transliterated text is
  never committed; it is regenerated from the source shards.
- NSFW filter fix (D16) with a re-run of `scripts/nsfw_spotcheck.py`.
- Mix decision (recorded in config): real romanized (upsampled ≤ ~2 epochs),
  transliterated, native Bengali script.
- `tokenizer/train.py`: BPE (`tokenizers`) and Unigram (`sentencepiece`), both
  12K vocab with byte fallback, trained from scratch on the final mix,
  including transliterated text. Emoji policy (keep vs. strip; PRD §6.2 vs.
  §7.2) decided here.
- Exact token counts for every source with both tokenizers; the Prompt 2
  report's estimated token table is replaced.
- Tokenizer report: fertility (tokens/word) on Bengali-script vs. Romanized
  Banglish vs. dialect text; `<unk>` rate (must be 0); the 20 hand-picked
  sentence splits (PRD Phase 1 exit criterion) for both tokenizers.
- `data/pipeline.py`: pre-tokenize into uint16 memmap shards (12K vocab fits;
  300M tokens ≈ 600 MB), 128-token packed sequences, curriculum-mix sampler
  (general → code-mixed weighted in the final ~20%).
- **Exit:** both tokenizers saved; DataLoader streams faster than training
  consumes, with flat RAM; tests for encode/decode round-trip and shard indexing.

### Prompt 4 — Model + training loops + measurements
- `model/`: RMSNorm, RoPE, attention, SwiGLU, encoder block, encoder with tied
  embeddings; MLM head; ELECTRA generator + discriminator (D3). Each file
  standalone-runnable, printing shapes; parameter count asserted against
  `est_params_m` in the model configs.
- `training/pretrain.py`: grad accumulation, grad checkpointing, 8-bit AdamW,
  cosine + warmup, clipping, `training.precision` = fp32 | fp16 (GradScaler
  for fp16 only), checkpoint every `ckpt_every`, wandb +
  `runs/<run>/manifest.json` (seed, config hash, git SHA, PRD §13), C1
  thermal monitor (temp every 30 s; micro-batch step-down with grad_accum
  compensation; pause + report).
- `training/finetune.py`: classification head (mean pooling, dropout 0.1,
  class-weighted CE), multi-seed runner (D5); reused in Prompts 6–8.
- **Measurements (all must be reported):**
  1. D1: real 30M model (selected, D15; 50M optional for reference), fp32 vs.
     fp16, forward + backward + optimizer step at B=2, T=128, grad
     checkpointing on → peak VRAM + step time. fp32 fits and is faster → lock
     in; fp32 OOMs → fp16, recorded in decisions.md.
  2. D4 + C1: throughput for 30M (MLM and ELECTRA), 30 s sustained at target
     batch, **cold** (idle start) and **warm** (after 10 min sustained load).
     Warm < 80% of cold = throttling signal. Report both.
  3. Set `budget.tokens_per_run` from warm throughput so 3 runs + 9
     fine-tunes fit in ≤48 h.
- **Exit:** tests green; measurements table in `docs/phase0_report.md`;
  budget set in config.

### Phase 0 — Ablation execution
- Stages a → b per `phase0.yaml` (3 runs, all 30M, D15); each candidate
  fine-tuned with 3 seeds.
- `docs/phase0_report.md`: F1 (mean ± std) per FLOP table (6NT incl. ELECTRA
  generator), winning-rule application, tie handling (D5), floor check.
- **Exit:** winning config written into `phase0.yaml` (TBDs filled), or the
  floor fails → stop and diagnose (PRD §5.0).

### Prompt 5 — Full pretraining (PRD Phase 2)
- Winner config, commitment target 300M tokens, curriculum weighting, val
  every 500 steps, C1 thermal guard active.
- PRD §7.1 day-7 health check: < 40% of token budget done → cut the budget
  and commit.
- **Exit:** stable convergence; sensible masked-token predictions on held-out
  Bangla/Banglish.

### Prompts 6–8 — Fine-tuning (PRD Phase 3) · proposed split
- **Prompt 6:** self-run BanglaBERT baseline on the frozen splits with the
  identical preprocessing; precompute and cache teacher logits once, so the
  110M teacher never shares VRAM with the student. Pin the fine-tune spec
  (pooling, head, dropout, label smoothing, focal α/γ, class weights; PRD §11)
  in config before training.
- **Prompt 7:** BanglishBERT (csebuetnlp/banglishbert) self-run baseline on
  the frozen splits, the primary comparison for the romanized task (D13);
  BnSentMix distillation fine-tune (soft + hard loss), imbalance-aware;
  kill-criteria check (PRD §9.5).
- **Prompt 8:** secondary tasks in priority order: depression → dialect
  sentiment → counter-speech.

### Prompt 9 — Evaluation (PRD Phase 4)
- Bootstrap CIs on every headline number (mandatory for the dialect set).
- INT8 quantization; FP32/INT8 sizes; CPU latency at batch 1 on the i5
  (target ≤ 50 ms); peak memory; qualitative error analysis.

### Prompt 10 — Release (PRD Phase 5)
- README win table, model card with two-stage provenance wording (PRD §13),
  exact parameter count replacing "~30–50M" everywhere; Gradio / HF Spaces
  demo; bilingual launch post draft.

### Prompt 11 — Buffer
- Secondary-task overflow, community dialect data follow-up, polish.

---

## 4. Evidence behind D1 (2026-11-07, `scripts/bench_precision.py`)

Proxy: 8-layer FFN stack shaped like the 50M encoder, micro-batch 2×128, no
attention.

| Measure | fp32 | fp16 + GradScaler |
| --- | --- | --- |
| GEMM throughput (layer shapes) | ~1–1.8 TFLOPS | ~0.13–0.25 TFLOPS |
| Training step | 121–125 ms | 174–189 ms |
| Peak VRAM | 395 MiB | 393 MiB |

GPU at 90–91 °C after the run; one fp32 GEMM fell from 1.18 to 0.21 TFLOPS
between rounds (throttling, see C1).

---

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| GPU thermal throttling (90–91 °C observed) | C1 guard; budget from warm throughput; Akib handling physical cooling |
| PRD's 20–30 h per 200M tokens may be optimistic (~2–3k tok/s implied) | Measured in Prompt 4; at ~1k tok/s, 300M tokens ≈ 3.5 days continuous |
| Pretraining corpus under 300M tokens with the 1 GB BanglaC4 cap | Prompt 2 audit reports the total; raise the cap only after review (D6) |
| 8 GB RAM during Wikipedia/BanglaC4 processing | Stream everything; uint16 memmap shards |
| Phase 0 decision driven by fine-tune noise | 3 seeds, overlap = tie (D5) |
| CI installs the ~2.5 GB CUDA torch wheel on a GPU-less runner | Acceptable for now; revisit with a CPU-wheel constraints file if CI time hurts |

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

## D6 — BanglaC4 stream cap

Cap the BanglaC4/OSCAR stream at 1 GB compressed. If the Prompt 2 audit shows
total pretraining tokens < 300M after Wikipedia + task data + this stream,
the cap may be raised, but only after reporting the token total. No further
download without that report.

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

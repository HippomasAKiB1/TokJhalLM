# TokJhalLM

> BanglaBERT is bland. TokJhalLM has flavor.

A ~30–50M-parameter Bengali code-mixed & dialect-aware encoder, trained 
from scratch on 4GB VRAM.

## Project context
- Execution lead: Akib
- PRD at docs/PRD.md is the source of truth. When in doubt, re-read it.
- Current phase: Phase 0 ablation not yet started. Prompt 1 (scaffold) 
  is complete and verified.
- Prompt sequence in flight: 11 prompts total (amended 2026-11-07, see 
  docs/decisions.md A1). Prompts 1–2 done (Prompt 2 approved 2026-11-08). 
  Prompt 3 (transliteration + tokenizers) is in planning. Phase 0 ablation runs 
  after Prompt 4, not first.
- Binding decisions D1–D18, amendment A1, constraint C1: docs/decisions.md. 
  These override the PRD on the specific points they record.
- Execution plan: docs/IMPLEMENTATION_PLAN.md (prompt-by-prompt scope and 
  exit criteria). Current state: docs/EXECUTION_PROTOCOL.md.

## Hard rules

### Environment
- Python interpreter: `E:\Anaconda\envs\tokjhallm\python.exe` (Python 3.11.18)
- ALL commands must use this full path. Never `python`, `pip`, `pytest`, 
  `ruff`, or `mypy` bare.
- Never `conda activate` — the shell is not conda-initialized, activation 
  silently fails or resolves to C:\Python314 (CPU-only torch, wrong).
- To install: `E:\Anaconda\envs\tokjhallm\python.exe -m pip install <pkg>`

### Held pins (do not change without asking)
- torch==2.5.1+cu124
- transformers==4.46.3
- tokenizers==0.20.3
- bitsandbytes==0.44.1
- datasets==3.1.0
- pyarrow==25.0.1  ← pinned below 26 to keep NumPy 1.x compatible
- numpy==1.26.4    ← HELD at 1.x. Do NOT upgrade to 2.x. Silent CUDA 
                     failures are reported with NumPy 2 + torch.
- wandb==0.18.7
- sentencepiece==0.2.0

### Hardware truths (verified in Prompt 1)
- GPU: NVIDIA GeForce GTX 1650, 4.00 GB VRAM
- bf16: NOT supported (Turing sm_75 → needs sm_80+). 
  `torch.cuda.is_bf16_supported()` returns False.
- Tensor Cores: NONE. The GTX 1650 is a GTX 16-series Turing part (TU117); 
  it has no Tensor Cores and no RT cores.
- fp16: numerically supported but SLOW on this card (measured 2026-11-07, 
  `scripts/bench_precision.py`): fp16 GEMMs ~0.13–0.25 TFLOPS vs fp32 
  ~1–1.8 TFLOPS; an fp16+GradScaler training step is ~1.4–1.6× slower than 
  fp32 at the real micro-batch (2×128) and saves ~no VRAM there. fp16 only 
  buys memory when activations dominate (large micro-batches).
- Training precision: fp32 primary. fp16 fallback for OOM only. Measured on 
  GTX 1650 (TU117, no Tensor Cores): fp16 is 1.4–1.6× slower than fp32 with 
  negligible memory difference at 50M-scale. Confirm on real model at Prompt 4.
- `training.precision` (fp32 | fp16) controls every training loop. 
  `torch.amp.GradScaler` is required for fp16 only; the smoke test verifies 
  that fallback path.
- Thermals: GPU hit 90–91 °C under load and throughput dropped between 
  benchmark rounds (throttling). Multi-day runs log GPU temp every 30 s to 
  wandb; >85 °C for >5 min → micro-batch −1 (min 1, grad_accum raised to keep 
  effective batch); still >85 °C → pause run and report (docs/decisions.md C1, 
  configs/base.yaml `monitoring`). NEVER change GPU clocks/voltage/power from 
  code; cooling is Akib's (hardware tools only).
- System RAM: 8 GB
- CPU: Intel i5 12th gen

### From-scratch claim (load-bearing)
- Base model (tokenizer + encoder) is trained entirely from scratch. No 
  pretrained weights.
- HuggingFace `transformers` is allowed ONLY for: dataset loading, tokenizer 
  helpers, and (later, in Phase 3) loading BanglaBERT as a distillation 
  teacher. NEVER as the base model.
- If you propose initializing from bert-base-* or similar, STOP. The 
  from-scratch claim is central to the portfolio narrative.

### Style
- YAML config in /configs; no magic numbers in code.
- Type hints on public functions; docstring on every file.
- pytest tests in /tests alongside each module.
- Every module runs standalone: `python -m model.attention` should print 
  shapes and exit 0.
- If you must deviate from the PRD, STOP and ask.

## Workflow for every task
1. Plan first (Plan mode or Artifact). List every file to create/modify.
2. Wait for approval if >2 hours of work OR touches more than 3 files.
3. Execute.
4. Walkthrough: what you built, what tests pass, what failed.

## Repo state (update this section when you scan)
Last scanned: 2026-11-08 (end of Prompt 2).
- Directories: model/ tokenizer/ data/ training/ scripts/ configs/ 
  docs/ runs/ tests/
  - model/, tokenizer/, training/: empty packages (`__init__.py` + README.md).
  - data/: Prompt 2 pipeline (common, download, clean, readers, minhash, 
    splits, leakage, corpus, audit). `python -m data.audit` rebuilds the 
    corpus and runs the leakage gate (exit 1 on failure). Raw data in 
    data/raw/ and Arrow shards in data/processed/ are gitignored; 
    data/MANIFEST.json, data/audit.json, data/splits/, 
    data/leakage_exclusions.json are committed.
  - Large data lives on D: only; HF cache/TMP at D:/tokjhallm_cache 
    (data.common.setup_env). Console is cp1252: scripts force UTF-8.
  - scripts/: smoke_test.py, bench_precision.py, nsfw_spotcheck.py, 
    report_tables.py. runs/: .gitkeep + README.md (contents gitignored).
  - docs/: PRD.md (v2.0), decisions.md (D1–D18, A1, C1), 
    IMPLEMENTATION_PLAN.md, EXECUTION_PROTOCOL.md, prompt2_report.md, 
    leakage_removals.md, README.md. docs/nsfw_spotcheck.md is local only.
- Configs: base.yaml, phase0.yaml, data.yaml, model_30m.yaml, model_50m.yaml
  - model_30m.yaml = 8L/480h/6 heads/ffn 1440 (≈30.0M), SELECTED (D15); 
    model_50m.yaml = 8L/640h/8 heads/ffn 1920 (≈50.7M), NOT selected, 
    reference only. Depth 8 and head_dim 80 fixed per PRD §6.1.
  - base.yaml: `precision: fp32`, `monitoring` thermal thresholds; its model 
    section mirrors model_30m.yaml (tested).
  - phase0.yaml: 3 runs at 30M (D15); `electra` (D3); 
    `budget.tokens_per_run` 25M default (D4); `finetune_seeds` ×3 (D5).
  - data.yaml: every source with pinned revision/checksum, caps, filters, 
    D14 leakage gate (`leakage.formulaic`).
- Tests passing: 25 (6 scaffold + 19 data pipeline; 2026-11-08). ruff and 
  mypy clean.
- Last commit: "prompt2: data download + audit complete" (run 
  `git log -1` for the hash).
- Smoke test: passes, verifies fp16 on GTX 1650
  (`E:\Anaconda\envs\tokjhallm\python.exe scripts/smoke_test.py`)
- Commands:
  - Tests: `E:\Anaconda\envs\tokjhallm\python.exe -m pytest tests/ -q`
  - Lint: `E:\Anaconda\envs\tokjhallm\python.exe -m ruff check .`
  - Types: `E:\Anaconda\envs\tokjhallm\python.exe -m mypy .`
- CI (.github/workflows/ci.yml): ubuntu-latest, no GPU — runs ruff, mypy, 
  pytest; the CUDA test skips there.

## Known PRD discrepancy (do not fix unprompted)
- PRD §5.0 colloquially says "six small scoped runs" for Phase 0. After 
  D15 the structure is 3 runs, all 30M (stage a: MLM vs ELECTRA / BPE; 
  stage b: Unigram at the winning objective, BPE run reused). 
  configs/phase0.yaml is correct.
- fp16 requires torch.amp.GradScaler (bf16 would not) in EVERY fp16 
  training loop (Phase 0, pretraining, fine-tuning) whenever the fp16 
  fallback is used.
- PRD §6.3 (mixed precision) is overridden by D1 (fp32 primary). PRD §12 
  timeline is overridden by A1 (Phase 0 after Prompt 4; 11 prompts, 
  9–10 weeks). Both recorded in docs/decisions.md; do not edit PRD.md.

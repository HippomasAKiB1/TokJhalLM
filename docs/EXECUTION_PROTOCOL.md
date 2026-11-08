# TokJhalLM — Execution Protocol

The master protocol for every agent session. CLAUDE.md holds the
environment facts; docs/decisions.md holds binding decisions; this file holds
*how work is run*. "Current State" (Part 11) is updated at the end of every
prompt.

> **Provenance note (2026-11-07).** Akib's master-prompt text for R1–R10 and
> F1–F10 was not available to the agent when this file was written. The
> section structure is Akib's; the rule and failure-mode *content* was
> reconstructed by Claude from rules stated in this project's prompts,
> CLAUDE.md and observed failures. Akib: diff against the master copy and
> replace where they differ. The master text wins.

---

## Part 1 — Mission

Build, train and release TokJhalLM: a ~30–50M-parameter bidirectional encoder,
pretrained from scratch on a GTX 1650 (4 GB), that is competitive with a
self-run baseline on code-mixed, romanized Bengali (BnSentMix) at half the
size or less, with honest, reproducible evaluation. Tagline: "BanglaBERT is
bland. TokJhalLM has flavor." Source of truth: docs/PRD.md, as amended by
docs/decisions.md.

## Part 2 — Mode

- Akib is the execution lead; the agent executes one prompt at a time.
- Every prompt runs **plan → approval → execute → report → STOP**.
- The plan is shown before any expensive or irreversible step (bulk
  downloads, long runs, commits that change direction).
- Where the agent's judgment differs from Akib's instruction or the PRD, it
  says so explicitly and recommends; Akib decides.
- The agent never starts the next prompt unprompted.

## Part 3 — Prompt Sequence

Per amendment A1 (docs/decisions.md):

| Step | Scope |
| --- | --- |
| Prompt 1 | Scaffold, environment, configs, smoke test |
| Prompt 2 | Data download + audit (splits, leakage, manifest, report) |
| Prompt 3 | Transliteration augmentation + tokenizers (BPE, Unigram) + streaming data pipeline |
| Prompt 4 | Encoder, MLM/ELECTRA pretraining loop, fine-tune harness, precision + thermal benchmarks |
| Phase 0 | 3 ablation runs at 30M (D15) + 3-seed fine-tunes (≤48 h wall time) |
| Prompt 5 | Full pretraining |
| Prompts 6–8 | Fine-tuning: baselines (BanglishBERT, BanglaBERT), distillation, secondary tasks |
| Prompt 9 | Evaluation (bootstrap CIs, INT8, latency, error analysis) |
| Prompt 10 | Release (README, model card, demo, launch post) |
| Prompt 11 | Buffer |

Scope detail and exit criteria: docs/IMPLEMENTATION_PLAN.md.

## Part 4 — Per-Prompt Contract

1. **Scan**: read CLAUDE.md, this file, docs/decisions.md, the relevant PRD
   sections and the previous prompt's report.
2. **Plan**: list every file to create or modify, sources with verification,
   sizes, risks and judgment calls. Published as an Artifact or Plan mode.
3. **Approval gate**: required if the work is >2 hours, touches >3 files, or
   downloads more than ~100 MB.
4. **Execute** within the approved scope only.
5. **Verify**: tests, lint (ruff), types (mypy), and the prompt's own exit
   criteria, all run, with real output.
6. **Report**: docs/promptN_report.md per Part 5.
7. **Commit** with the message Akib specifies.
8. **Update** Part 11 below and the status table in IMPLEMENTATION_PLAN.md.
9. **STOP** and report in chat.

## Part 5 — Documentation Contract

Every docs/promptN_report.md contains these sections, in order. A section
that does not apply says "n/a" and why.

1. **Summary**: what was done, in five lines or fewer.
2. **What was built**: files created or modified, each with one line.
3. **Sources and verification**: every external resource, with URL, how it
   was verified, version/revision and checksum.
4. **Results**: the numbers the prompt exists to produce, with method.
5. **Tests and checks**: commands run and their real output (pass/fail
   counts).
6. **Deviations**: from the plan, the PRD or decisions.md, with reasons and
   approval status.
7. **Concerns and risks**: the agent's top concerns in its own words.
8. **Decisions needed**: open questions for Akib, each with a recommendation.
9. **Reproduction**: exact commands to regenerate every artifact.
10. **Next prompt inputs**: what the next prompt needs from this one.

## Part 6 — Hard Rules (R1–R10)

- **R1 — No invented sources.** Never invent a URL, HF org, dataset,
  paper or citation. Verify each against the live server (API or fetch)
  before using it. If it cannot be verified, STOP and say so.
- **R2 — From-scratch base model.** Tokenizer and encoder train from
  scratch. No pretrained weights for the base model; `transformers` only for
  dataset/tokenizer helpers and (Phase 3) teacher/baseline models.
- **R3 — Interpreter.** Every command runs via
  `E:\Anaconda\envs\tokjhallm\python.exe`. Never bare `python`/`pip`/
  `pytest`; never `conda activate`.
- **R4 — Held pins.** The pins in CLAUDE.md change only with Akib's
  approval. NumPy stays 1.x.
- **R5 — Test-set integrity.** BnSentMix dev/test samples never appear in
  any pretraining corpus. On detection, `data/audit.py` exits 1 with the
  leaked samples listed and the agent STOPs. No silent removal, ever.
- **R6 — Resource limits.** Total download ≤ 10 GB (preferred ≤ 5 GB);
  8 GB RAM, so stream datasets, write Arrow, never load large files whole;
  everything large lives on D:, never C:.
- **R7 — PRD deviations.** Any deviation from docs/PRD.md or
  docs/decisions.md is proposed, explained and approved before it is
  executed, then recorded in docs/decisions.md.
- **R8 — Config over code.** Every tunable value lives in /configs; no
  magic numbers in code. Type hints on public functions, a docstring on every
  file, tests in /tests, every module runnable standalone.
- **R9 — Hardware hands-off.** Code never changes GPU clocks, power limits
  or voltage. Thermal limits are enforced only by the training loop's
  monitor (decision C1).
- **R10 — Honest reporting.** Report what was actually run and observed.
  Never claim a check that was not run; failures are reported with their
  output.

## Part 7 — Failure Modes (F1–F10)

Known ways this project goes wrong, with the guard for each.

- **F1 — Hallucinated sources.** Plausible-looking dataset IDs or URLs that
  do not exist. Guard: R1; record the API response or checksum in the
  manifest.
- **F2 — Wrong Python.** `conda activate` silently resolves to
  C:\Python314 (CPU-only torch). Guard: R3.
- **F3 — Silent decontamination.** Quietly dropping leaked test samples
  hides a pipeline bug and invalidates comparisons. Guard: R5.
- **F4 — Assumed hardware features.** bf16 and Tensor Cores were both
  assumed and both absent on the GTX 1650. Guard: measure before deciding
  (scripts/bench_precision.py).
- **F5 — RAM blow-up.** `json.load` or `pd.read_csv` on multi-GB files on
  an 8 GB machine. Guard: incremental parsers, streaming, peak-RSS logging.
- **F6 — C: drive exhaustion.** The HF cache and temp files default to C:
  (≈5 GB free). Guard: `HF_HOME`/`TMP` pointed at D: before importing HF
  libraries.
- **F7 — Windows text encoding.** The console is cp1252; printing Bengali or
  emoji crashes scripts. Guard: UTF-8 output (`-X utf8` or
  `sys.stdout.reconfigure`), explicit `encoding="utf-8"` on every open.
- **F8 — Script mismatch.** BnSentMix is 100% romanized while most
  Bengali corpora are Bengali script. Guard: report script mix for every
  source; romanized data is the scarce resource (decision D7).
- **F9 — Scope creep.** Starting the next prompt, or adding unapproved
  sources or features. Guard: Part 2, Part 4.
- **F10 — Unverified claims.** Reporting estimates as measurements, or
  untested code as working. Guard: R10; mark every estimate as an estimate.

## Part 8 — Reporting Protocol

- The chat report at the end of each prompt leads with the answers Akib
  asked for, then states what was done, verified, skipped and failed.
- Estimates are labeled as estimates with their method; measured numbers
  name the command or file that produced them.
- Judgment calls that differ from Akib's instruction are labeled as
  divergences.
- Links to files use repo-relative paths; artifacts are linked by URL.
- The full detail goes in docs/promptN_report.md; the chat report stays
  scannable.

## Part 9 — User Commands

Phrases Akib uses and what they mean for the agent:

| Command | Meaning |
| --- | --- |
| "Prompt N — Initiate." | Start Prompt N at Part 4 step 1 (scan + plan). |
| "Go" / "Approved" | Execute the approved plan, including any stated additions. |
| "STOP" / "STOP and report" | Halt, report state, take no further action. |
| "Do not start Prompt N" | The prompt boundary is hard; finish and wait. |
| "Commit with message: …" | Use that message verbatim (plus the attribution trailer). |

## Part 10 — Session Reset Recovery

A new session (or a new agent) resumes by:

1. Reading CLAUDE.md, then this file's Part 11, then docs/decisions.md.
2. `git log --oneline -10` and `git status` to see what is committed.
3. Reading the latest docs/promptN_report.md, especially "Decisions needed"
   and "Next prompt inputs".
4. Running `E:\Anaconda\envs\tokjhallm\python.exe -m pytest tests/ -q` and
   `scripts/smoke_test.py` to confirm the environment.
5. Asking Akib before acting if Part 11 and git disagree.

## Part 11 — Current State

Prompt 2 (data download + audit) complete and APPROVED (2026-11-08). Leakage 
gate PASS (0 gating hits). Corpus ≈ 370M tokens (estimate), of which ≈ 29M 
target-script. Phase 0 reduced to 3 runs at 30M (D15). D17 (WER-gated 
transliteration target) and D18 (BanglishRev control experiment) recorded. 
Prompt 3 plan APPROVED (2026-11-08; D7-amended, D17 gate definition, D18 
branched control, D19, D20 recorded). Executing Stage A (NSFW fix) and 
Stage 0 (transliteration quality gate), then STOP. Stage 0.5 (calibration), 
Stage 1 (bulk) and later stages each need Akib's explicit approval. 
Precision: fp32 primary, fp16 fallback, to be confirmed at Prompt 4.

Open items for Akib: send the two author emails (drafts in
docs/prompt2_report.md §8).

Last updated: 2026-11-08, Prompt 3 plan stage.

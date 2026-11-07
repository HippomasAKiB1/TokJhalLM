# TokJhalLM — A \~30–50M-Parameter Bengali Code-Mixed & Dialect-Aware Language Model

*by Akib / Team Turtlers — model name: TokJhalLM (টক-ঝাল)*

| Field | Value |
| --- | --- |
| Document Version | 2.0 (Research-Reviewed) |
| Status | Approved for Execution |
| Target Completion | 8 weeks (part-time) |
| Hardware Envelope | 4 GB VRAM · 8 GB RAM · i5 12th Gen CPU |

---

## 1. Executive Summary

TokJhalLM is a \~30–50M-parameter Bengali language model trained from scratch on consumer hardware, purpose-built for the language people actually use online: **code-mixed Banglish and regional dialects**. BanglaBERT (110M params, BLUB 77.09) is strong on standardized Bengali but demonstrably underperforms on informal, code-mixed, and dialectal text. TokJhalLM does not try to beat BanglaBERT at general NLU — it targets the exact distribution BanglaBERT was never built to serve, while being roughly half the size and materially faster on CPU.

This version of the PRD corrects a critical architecture mistake in v1.0 (decoder-only for classification tasks) and adds the rigor needed for a result that survives expert scrutiny: fair re-run baselines, distillation, bootstrap confidence intervals, and a credible dialect data strategy.

---

## 2. Problem Statement

- **Code-mixed "Banglish":** Romanized Bengali mixed with English, no standard spelling. Existing systems underperform here due to irregular grammar and transliteration variance.
- **Regional dialects:** Sylheti, Chittagonian, Barishal, Noakhali diverge substantially from Standard Bangla. Public dialect resources are scarce (existing benchmarks top out around 600 labeled instances across 4 dialects).
- **Informal social sentiment:** BanglaBERT shows measurable errors on social text, including sentiment reversal and failure to separate hopeful vs. critical framing in code-mixed input.

A smaller model trained **exclusively on the target distribution** can beat a larger generalist on that distribution — this is well established in low-resource NLP (TinyStories, Regional-TinyStories).

---

## 3. Goals and Non-Goals

### 3.1 Primary Goal

Build, train, and release TokJhalLM: a \~30–50M-parameter encoder that is competitive with BanglaBERT (re-run on identical splits) on code-mixed Bengali sentiment, while being **≥2× smaller** and **materially faster at CPU inference**.

### 3.2 Secondary Goals

| Goal | Success Criterion |
| --- | --- |
| From-scratch pipeline | Custom tokenizer, custom model implementation, custom training loop for pretraining |
| Honest, reproducible evaluation | Baseline re-run on identical splits + bootstrap CIs on all headline numbers |
| Dialect awareness | Expanded, community-sourced dialect dataset (see §7.3), not just the 600-sample public set |
| Public, usable artifact | Live demo (not just a leaderboard number) that Bangla speakers can try |

### 3.3 Non-Goals

- Beating BanglaBERT on general BLUB benchmarks — TokJhalLM is a specialist, not a generalist.
- Commercial deployment.
- Instruction-following or long-form reasoning.
- Training on a massive general web corpus.

---

## 4. Target Users and Use Cases

| User | Use Case | What TokJhalLM Delivers |
| --- | --- | --- |
| Portfolio reviewer / hiring manager | Assess end-to-end ML engineering under real constraints | A reproducible, honestly-evaluated pipeline with a documented architecture decision process |
| Bengali NLP researcher | Baseline for code-mixed/dialectal Bengali SLMs | Released tokenizer, model, training code, and an expanded dialect dataset |
| Bangladeshi developer | Prototype moderation/sentiment tooling for local platforms | A small, CPU-fast model that understands Banglish and dialect input |
| Bangla speakers, general public | See their own way of writing understood by AI | A free, shareable live demo |

---

## 5. Architecture Decision (Critical Change from v1.0)

**v1.0 mistake:** proposed a decoder-only Transformer for tasks that are entirely classification (sentiment, depression detection, dialect sentiment, counter-speech). A causal LM only exposes unidirectional context for pooling; a bidirectional encoder gets full-context representations at the same parameter budget — a strictly better inductive bias for NLU at this scale. BanglaBERT itself is ELECTRA-style (bidirectional) for exactly this reason.

**Decision:** Primary model is a small **ELECTRA-style (replaced-token-detection) or MLM-style bidirectional encoder**, not a decoder-only LM.

- ELECTRA is preferred if time allows: it is markedly more sample-efficient than MLM, which matters because our pretraining corpus is small relative to compute-optimal scaling.
- **Amendment: drop the separate generative side model entirely. It adds a second training run and a second set of failure modes for no portfolio gain. The live demo (§10) is the classification model itself: a Banglish post in, sentiment/depression prediction with confidence out, plus a short 'why' example drawn from error analysi**s.

### 5.0 Phase 0 — Mandatory Pre-Commitment Ablation (NEW)

Amendment: decoder-only vs. encoder is not the open question — decoder-only will lose on classification with near-certainty, so ablating it just spends days confirming established practice. The ablation that actually reduces risk is three-way: (1) MLM vs. ELECTRA objective — ELECTRA is more sample-efficient but needs a generator + discriminator (\~1.3× compute, more implementation risk); (2) model size under the real data budget — 50M params on \~200M tokens is well below Chinchilla-optimal (\~4 tokens/param vs. 20), so a 25–30M model may generalize better; (3) BPE vs. Unigram tokenization for code-switched/Romanized text. Run these sequentially, not full-factorial, over \~5 days: (a) MLM vs. ELECTRA at 50M/BPE; (b) 30M vs. 50M at the winning objective/BPE; (c) BPE vs. Unigram at the winning objective/size — six small, scoped runs total. **Winning rule:** highest BnSentMix dev macro-F1 per FLOP of training compute, subject to fitting in 4GB VRAM at the target batch size; if two configurations are within 0.5 F1 points, prefer the smaller/faster one. **Minimum viable gate:** for the winning configuration to be promoted to full pretraining, its BnSentMix dev macro-F1 must clear a pre-registered floor (set after seeing a 4-class-majority baseline; suggest \~40% as a floor where the model is clearly learning rather than guessing). If no configuration clears the floor, Phase 0 has failed — investigate data, tokenizer, or task framing before committing to full pretraining. This is not a kill criterion for the project (§9.5); it is a 'stop and diagnose' trigger for Phase 0. This is the single highest-leverage risk-reduction step in the whole project and is **not optional**.

---

## 6. Technical Specification

### 6.1 Model Architecture

| Component | Specification | Rationale |
| --- | --- | --- |
| Architecture family | ELECTRA-style (preferred) or MLM-style bidirectional encoder. No separate decoder-only side model (dropped per §5 amendment). | Matches the NLU task family; far better sample efficiency than causal LM pretraining |
| Parameters | \~30–50M, set by Phase 0 outcome (main encoder) | Fits 4GB VRAM with mixed precision + gradient checkpointing |
| Layers | 8 | Matches successful small-model precedent (Regional-TinyStories range) |
| Hidden size | 640 | Balanced for \~30–50M budget |
| Attention heads | 8 (head dim 80) | Standard |
| Positional encoding | RoPE | Better extrapolation/stability than learned absolute embeddings |
| Normalization | RMSNorm, pre-norm | Stable, efficient |
| Activation | SwiGLU | Outperforms GeLU in modern small LMs |
| Embeddings | Tied input/output | Saves 5–10M params, critical at this budget |
| Context length | 128 (not 256/512) | Social posts are mostly <100 tokens; right-sizing frees compute for more training steps |

### 6.2 Tokenizer

| Specification | Decision |
| --- | --- |
| Type | **Ablate BPE vs. SentencePiece Unigram** — Unigram's probabilistic segmentation often handles code-switched/Romanized/morphologically rich text better than greedy BPE merges |
| Vocabulary size | 12,000, with byte-level fallback so emoji/stray Latin punctuation never produce `<unk>` |
| Training corpus | Combined BnSentMix + DepressiveText + PolCSBD text, weighted toward code-mixed/dialectal content, plus a **representative sample of Wikipedia/BanglaC4 text (e.g., 5–10% of tokenizer training corpus) to cover general Bangla vocabulary and avoid fragmenting common subwords** |
| Tools | `tokenizers` / `sentencepiece`, trained from scratch — no pretrained tokenizer |

### 6.3 Training Infrastructure

| Constraint | Mitigation |
| --- | --- |
| 4GB VRAM | Mixed precision (bf16/fp16) + gradient checkpointing + 8-bit AdamW (bitsandbytes) + gradient accumulation |
| 8GB system RAM | Memory-mapped / streaming datasets (HF `datasets` Arrow backing or custom memmap) from day one — do not wait for OOM to fix this |
| Batch size | Per-device 2–4; gradient accumulation 16–32 → effective batch 32–128 |
| Optimizer | 8-bit AdamW; cosine LR schedule with linear warmup (\~500 steps) |
| Learning rate | Peak 3e-4 to 6e-4, tuned on small validation run |
| Gradient clipping | 1.0 |
| Checkpointing | Every 500–1000 steps; keep best validation checkpoint |

---

## 7. Data Strategy

### 7.1 Pretraining Data

| Source | Size | Role |
| --- | --- | --- |
| Code-mixed/dialectal task text (BnSentMix, DepressiveText, PolCSBD, unlabeled) | \~37K+ texts | **Majority weight** — this is the target distribution |
| Regional-TinyStories Bangla | 2M stories | **Minority weight**, fluency regularization only — NOT the backbone. Overweighting this pulls the model toward generic Bangla, away from the target distribution |
| Full Bengali Wikipedia dump | \~50–100M+ tokens | General Bangla coverage |
| BanglaC4 / OSCAR Bengali subset (streamed) | 100s of M to billions of tokens | Scale to reach the 300–500M token target |
| Transliteration-augmented task text | 2–5× task text volume | Multiplies target-distribution data |

**Pretraining token target:** \~300–500M tokens (\~6–10 tokens/param at 30–50M params), with curriculum weighting — general Bangla first, code-mixed/dialectal and transliteration-augmented text heavily weighted in the final \~20% of steps. On this hardware, roughly 20–30 hours per 200M tokens is realistic; budget \~2 weeks of continuous pretraining in Weeks 2–3. **Commitment target: 300M tokens** (lower end of the range), with 400–500M as a stretch goal only if pretraining is tracking ahead of schedule — 300M tokens is Chinchilla-adequate for a 30M model. **Pretraining health check:** if by day 7 of pretraining less than \~40% of the token budget is complete, cut the target to what's actually achievable and commit; do not extend Weeks 2–3 into the Week 4 fine-tuning slot, since Phase 3 is the other protected decision point.

### 7.2 Fine-Tuning Data

| Dataset | Size | Task |
| --- | --- | --- |
| BnSentMix | 20,000 | 4-class sentiment (primary task) |
| DepressiveText | 7,019 | Binary depression detection (κ=0.84) |
| Bangla Dialect Dataset | 600 (public) + community-expanded (see 7.3) | Dialect sentiment |
| PolCSBD | 10,021 | Counter-speech detection |

**Data hygiene:** Deduplicate across datasets (BnSentMix and PolCSBD both scrape social media — check for duplicate posts before trusting splits). Strip URLs/@mentions/emojis consistently. Verify no test-set leakage.

### 7.3 Dialect Data Expansion (NEW — addresses a real weakness)

600 instances across 4 dialects (\~150/dialect) is too thin to credibly claim "dialect-aware" and is close to statistical noise at evaluation time. Plan: a simple, shareable "teach the model your dialect" data collection form, distributed in Sylheti/Chittagonian/Barishal/Noakhali-speaking community groups. This simultaneously (a) fixes the data problem and (b) builds genuine community ownership of the project — the single biggest lever for making this memorable rather than just technically correct.

---

## 8. Training Plan

**Phase 0 (\~5 days, sequential not factorial, see §5.0): Three-way ablation.** MLM vs. ELECTRA objective, 30M vs. 50M params, and BPE vs. Unigram tokenizer, on a fixed data subset; decide before committing to full pretraining.

**Phase 1 (Week 1): Tokenizer + data pipeline.**

1. Download/clean all datasets.
2. Ablate BPE vs. Unigram; train winning 12K-vocab tokenizer on the combined, dialect-weighted corpus.
3. Build streaming/memmap `Dataset`/`DataLoader` for 128-token sequences.
4. Manually verify tokenization on 20 hand-picked Banglish/dialectal sentences. *Exit criterion:* sensible subword splits; data loader runs without errors or RAM issues.

**Phase 2 (Weeks 2–3): Pretraining.**

1. Train the encoder at the Phase 0-selected size (ELECTRA/MLM) on the dialect-weighted corpus.
2. Monitor validation loss/perplexity every 500 steps. *Exit criterion:* stable convergence; sensible masked-token predictions on held-out Bangla/Banglish text.
3. Apply curriculum weighting: general Bangla (Wikipedia/BanglaC4) first, shifting to heavily code-mixed/dialectal and transliteration-augmented text for the final \~20% of steps.

**Phase 3 (Weeks 4–5): Fine-tuning with distillation (NEW).**

1. **Re-run BanglaBERT yourself** on your exact BnSentMix splits — do not compare to numbers from other papers; different preprocessing/splits make that comparison invalid.
2. Fine-tune TokJhalLM on BnSentMix using **knowledge distillation from BanglaBERT as teacher** (soft-label + hard-label loss) — the highest-leverage change for closing the performance gap without compromising the "from-scratch base model" claim.
3. Apply class-weighted or focal loss for label imbalance.
4. If time permits, repeat for secondary tasks in priority order: depression detection → dialect sentiment → counter-speech. *Exit criterion:* Primary task F1 within target range of your own re-run BanglaBERT baseline, with imbalance-aware evaluation.

**Phase 4 (Week 6): Evaluation and ablation.**

1. Compute final metrics with **bootstrap confidence intervals** on all headline numbers — mandatory for the 600-sample dialect set, where a single-point F1 comparison is close to noise.
2. Measure inference latency (CPU/GPU), model size (FP32/INT8), peak memory.
3. Quantize to INT8.
4. Qualitative error analysis: show concrete examples where TokJhalLM succeeds and where it fails — this builds more trust in the Bangla NLP community than any single leaderboard number.

**Phase 5 (Week 7): Packaging and release.**

1. README + model card + training report, including a "what this model gets wrong" section.
2. Release tokenizer, weights, code, and the expanded dialect dataset.
3. **Live demo is a primary deliverable, not optional** — a shareable web demo where someone types real Banglish/dialectal text and sees it understood in real time is the actual memorable artifact, more than the F1 table.
4. Bilingual (Bangla + English) launch post telling the human story, with real relatable examples.

**Week 8 (buffer):** Secondary tasks, community dialect data follow-up, polish.

---

## 9. Evaluation and Success Metrics

### 9.1 Primary Success Metrics

| Metric | Target | Method |
| --- | --- | --- |
| BnSentMix F1 (macro) | Within 3–5 points of **self-run** BanglaBERT baseline, with bootstrap CI | Identical splits/preprocessing for both models |
| Model size (FP32) | ≤220MB | Checkpoint size |
| Model size (INT8) | ≤60MB | Post-training quantization |
| CPU inference latency | ≤50ms/sample (batch=1) | Wall-clock on i5 12th Gen |
| Parameters | 30M–55M (Phase 0-selected size ±5M) | Trainable parameter count |

### 9.2 Secondary Metrics

| Metric | Target |
| --- | --- |
| DepressiveText accuracy | Competitive with published BanglishBERT+LSTM baseline (0.8889), re-verified where possible |
| Dialect sentiment F1 | Report with bootstrap CI; treat as directional given small sample, not a clean win/loss claim |
| Pretraining validation loss | Stable convergence, no divergence |
| Demo usability (no side model) | Live demo returns a sentiment/depression prediction with confidence and a representative error-analysis example — not a separate generative model |

### 9.3 Evaluation Rigor Requirements (NEW, non-negotiable)

- Self-run baseline on identical splits — never cite another paper's BanglaBERT number as the comparison point.
- Bootstrap confidence intervals on every headline F1, especially the 600-sample dialect set.
- Cross-dataset deduplication check (BnSentMix × PolCSBD) before trusting any split.
- Class-imbalance-aware metrics (macro F1, not just accuracy) throughout.

### 9.4 The "Win Table" for the README

| Model | Params | BnSentMix F1 (±CI) | Size (FP32) | CPU Latency |
| --- | --- | --- | --- | --- |
| BanglaBERT (self-run baseline) | 110M | \[measured\] | 442.6MB | \[measured\] |
| TokJhalLM | \~30–50M | \[measured\] | ≤220MB | \[measured\] |
| TokJhalLM (INT8) | \~30–50M | \[measured\] | ≤60MB | \[measured\] |

### 9.5 Kill Criteria (checked during Phase 3 fine-tuning)

If, after knowledge distillation and hyperparameter tuning, TokJhalLM's BnSentMix macro-F1 is **more than 10 points below** the self-run BanglaBERT baseline on identical splits with **non-overlapping bootstrap CIs**, the project does not deliver on its core claim of "competitive performance at half the size." In that case, the portfolio deliverable shifts from *"a competitive specialist"* to *"a rigorous negative result + open-source tooling for code-mixed Bengali SLM research"* — with the pipeline, tokenizer, expanded dialect data, and honest analysis as the released artifacts. This is still a valid portfolio outcome, but it is a **different story** and must be framed explicitly as such in the README.

---

## 10. Deliverables

| Deliverable | Format | Location |
| --- | --- | --- |
| Source code | Python package (`/model`, `/tokenizer`, `/data`, `/training`) | GitHub |
| Trained tokenizer | `tokenizer.json`/SentencePiece model | GitHub / HF Hub |
| Model weights | FP32 + INT8 checkpoints | HF Hub / GitHub |
| Expanded dialect dataset | CSV/JSON, community-sourced | GitHub (license-permitting) |
| Training report | Markdown, loss curves, ablations, bootstrap CIs | `/docs` |
| README | Win table + honest failure-mode section | Repo root |
| **Live demo** | Web app (Gradio/HF Spaces): text in → sentiment/depression prediction + confidence + a short error-analysis 'why' example out. No separate generative model. | HF Spaces |
| Bilingual launch post | Bangla + English | Blog / LinkedIn |

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| VRAM OOM | High | Blocking | Small batch + grad accumulation + checkpointing + 8-bit optimizer; fallback to 30M (which Phase 0 may select anyway) params if needed |
| Wrong architecture choice wastes weeks | Medium (mitigated by Phase 0) | High | Mandatory Phase 0 ablation before full commitment |
| Pretraining corpus too small/generic | Medium | High | Lift the artificial data cap: add full Bengali Wikipedia + streamed BanglaC4/OSCAR + transliteration augmentation to reach \~300–500M tokens; curriculum-weight toward code-mixed/dialectal text late in pretraining; TinyStories stays a minority share |
| Dialect results are noise (600 samples) | High | Medium | Community data expansion (§7.3) + mandatory bootstrap CIs + directional framing, not a hard claim |
| Unfair baseline comparison undermines credibility | Medium | High | Self-run BanglaBERT on identical splits — non-negotiable |
| Scope creep across secondary tasks | Medium | Medium | Primary task + Phase 0 ablation are non-negotiable; everything else is priority-ordered and droppable |
| Dataset licensing | Low | Medium | Verify before release; release model/code only if a dataset can't be redistributed |
| Fine-tuning spec too thin to implement from | Medium | Medium | Specify pooling strategy (mean vs. \[CLS\]), classification head architecture, dropout, label smoothing, focal loss α/γ, and class weights before Phase 3 starts |
| Depression/dialect data collection raises consent and sensitivity concerns | Medium | High | Add explicit consent language, anonymization policy, and sensitive-data handling statement to the §7.3 collection form |

---

## 12. Timeline

| Week | Milestone |
| --- | --- |
| 0 (\~5 days) | Architecture ablation decision made |
| 1 | Tokenizer + data pipeline ready |
| 2–3 | Pretraining complete, stable convergence |
| 4–5 | Primary task fine-tuned with distillation; self-run baseline complete |
| 6 | Full evaluation with bootstrap CIs; INT8 quantization |
| 7 | Packaging, release, live demo, bilingual launch post |
| 8 (buffer) | Secondary tasks, community dialect data, polish |

---

## 13. Portfolio & Public Narrative

> TokJhalLM is a \~30–50M-parameter Bengali encoder built from scratch on a 4GB GPU. BanglaBERT understands textbook Bengali. TokJhalLM understands the Bengali people actually type — Banglish, dialect, and the emotional nuance of social media — and does it on a fraction of the compute, with every comparison number honestly re-measured on identical splits.
>
> **BanglaBERT is bland. TokJhalLM has flavor.**
>
> **What it does well:** Banglish/dialectal sentiment and depression-signal detection, running fast enough for CPU/edge deployment. **What it does not do:** General-purpose Bengali NLU, long reasoning, or instruction following. It's a specialist, and it says so plainly — including where it still gets things wrong.

**Tagline placement:** "BanglaBERT is bland. TokJhalLM has flavor." is the single most memorable sentence in this project — treat it as a tagline, not a paragraph. Use it verbatim in four places: the block-quote above, the README title line (immediately after the project name), the HF Spaces demo banner, and the first line of the bilingual launch post.

**Post-Phase 0 size update:** once Phase 0 selects a final parameter count, every "\~30–50M" reference in this document, the model card, the README, and the demo banner must be updated to the exact selected size before launch. Do not launch with a placeholder range in the title — reviewers check.

**Claim precision:** state this as a two-stage fact, plainly, wherever the model's provenance is described: the base encoder is pretrained from scratch; the task head is fine-tuned with knowledge distillation from BanglaBERT. Never blur this into an unqualified "trained entirely from scratch" claim — reviewers will check it.

**Process note (added):** use wandb (free tier) or local TensorBoard plus a `runs/` manifest (seed, config hash, git SHA) for every training run. Reproducibility is a portfolio differentiator, not overhead. In §12, Phase 0 and Phase 3 are the two decision points worth protecting if the week-8 buffer gets consumed — everything else is droppable.

---

## 14. Approval and Immediate Next Actions

Approved for execution. **No pretraining begins before Phase 0's architecture ablation is complete.**

1. Download BnSentMix, DepressiveText, PolCSBD, Bangla Dialect Dataset, Regional-TinyStories Bangla, full Bengali Wikipedia dump, and a BanglaC4/OSCAR Bengali subset; measure actual token counts before committing to model size.
2. Stand up streaming/memmap data pipeline (don't defer this).
3. Run the three-way Phase 0 ablation (MLM vs. ELECTRA, 30M vs. 50M, BPE vs. Unigram), sequential not factorial, over \~5 days; apply the §5.0 winning rule and commit.
4. In parallel: draft and circulate the community dialect-data collection form.
5. Commit to full architecture only after Phase 0 results are in.

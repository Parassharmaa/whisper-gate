# Pulse-Whisper Project Plan

## Phase 0: Setup
- [x] Repository init (git, uv, pyproject.toml, src layout)
- [x] Copy configs from pdna (CLAUDE.md, .gitignore, .claude/settings.json, .env)
- [x] Create directory structure (models, data, eval, analysis, configs, scripts, tests)
- [x] `uv sync` — all dependencies installed and verified

## Phase 1: Implementation (Code)

### Task 1: Implement PulseModule (port from PDNA) ✅
- **File:** `src/pulse_whisper/models/pulse_module.py`
- **Blocked by:** —
- PulseLayer: learnable amplitude A, frequency ω, optional state-dependent phase φ(h)
- α mixing coefficient (initialized at 0.01)
- Forward: `hidden_states + α * A * sin(ω * t + φ(h))`
- Reference: `pdna/src/pdna/models/pulse_cfc.py`

### Task 2: Implement PulseWhisperEncoder wrapper ✅
- **File:** `src/pulse_whisper/models/pulse_whisper.py`
- **Blocked by:** Task 1
- PulseWhisperEncoder: wraps HF Whisper encoder, injects PulseLayer after each layer
- NoiseWhisper variant (random perturbation, matched magnitude)
- Variant enum (A–E) + build_variant factory
- Support freezing Whisper params, training only pulse params

### Task 3: Implement gap injection and data loading ✅
- **Files:** `src/pulse_whisper/data/gapped_audio.py`, `src/pulse_whisper/data/dataset.py`
- **Blocked by:** —
- inject_silence_gaps() for mel spectrograms — contiguous + multi-gap modes (0/5/15/30%)
- LibriSpeech loading (train-clean-100, test-clean, test-other)
- 10h subset for prototyping
- Gap augmentation during training

### Task 4: Implement evaluation and metrics ✅
- **Files:** `src/pulse_whisper/eval/gapped_eval.py`, `metrics.py`
- **Blocked by:** —
- Gapped eval: run model at each gap level, collect WER
- Hallucination: feed silence/noise, measure text output rate
- Metrics: WER (jiwer), CER, degradation curves, hallucination rate + severity

### Task 5: Implement training loop with config system ✅
- **Files:** `src/pulse_whisper/training/config.py`, `trainer.py`
- **Blocked by:** —
- YAML config → dataclasses (matching pdna pattern)
- Trainer: AdamW, cosine LR + warmup, AMP (fp16), gradient clipping
- Only train pulse params (freeze Whisper)
- TensorBoard / W&B logging, checkpoint saving

### Task 10: Implement analysis pipeline ✅
- **Files:** `src/pulse_whisper/analysis/alpha_analysis.py`, `stats.py`
- **Blocked by:** —
- α growth tracking, learned ω frequency analysis
- Paired t-tests, Cohen's d, bootstrap CIs, win rates

## Phase 1: Experiments (Run)

### Task 6: Phase 1.1 — Baseline gapped evaluation (Whisper-Tiny) ✅
- **Script:** `scripts/run_phase1_baseline.py`
- **Blocked by:** Tasks 3, 4
- Frozen Whisper-Tiny on test-clean with gaps 0/5/15/30/multi
- **Results (100 samples):**
  - gap_0: WER=0.0711 | gap_5: WER=0.0856 | gap_15: WER=0.1102
  - gap_30: WER=0.1567 | multi_gap: WER=0.2193
  - Hallucination: 100% on silence and white noise
- Clear degradation curve: WER triples from 0% to multi-gap

### Task 7: Phase 1.2 — Zero-shot pulse injection test ✅
- **Script:** `scripts/run_phase1_zeroshot.py`
- **Blocked by:** Tasks 1, 2, 6
- Fixed-parameter PulseLayer (no training) between encoder layers
- **Results:** No significant delta vs baseline (expected — params untrained)
  - pulse: ±0.001 WER across all gap levels
  - pulse_phase: ±0.001 WER across all gap levels
- Confirms pulse needs training to have effect

### Task 8: Phase 1.3 — Train pulse on Whisper-Tiny (Go/No-Go)
- **Blocked by:** Tasks 5, 7
- Train on 10h LibriSpeech with gap augmentation
- 4 variants × 1 seed: A (Baseline), B (+Noise), C (+Pulse), D (+Pulse+Phase)
- **GO:** Pulse beats Baseline AND Noise on multi-gap WER
- **PIVOT:** Pulse ≈ Baseline → try Small model or different insertion
- ~1-2 hrs compute

### Task 9: Phase 1.4 — Hallucination-specific testing
- **Blocked by:** Task 8
- Pure silence (30s), white noise, speech with long pauses
- **This is the killer metric** — hallucination rate reduction = paper
- ~30 min

## Phase 2: Full Experiment (Conditional on GO)
- Scale to Whisper-Small (12 encoder layers, 244M params)
- 5 variants × 5 seeds = 25 runs on LibriSpeech-100h
- Comprehensive eval: gapped, hallucination, noise robustness
- Full statistical analysis + paper figures
- ~125 hrs compute (~5 days on A4000)

## Dependency Graph
```
Tasks 1, 3, 4, 5, 10 — can start in parallel
Task 2 ← Task 1
Task 6 ← Tasks 3, 4
Task 7 ← Tasks 1, 2, 6
Task 8 ← Tasks 5, 7
Task 9 ← Task 8
```

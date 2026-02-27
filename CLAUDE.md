# Pulse-Whisper — Oscillatory Dynamics for Silence-Robust Speech Recognition
Learnable oscillatory pulse injection into Whisper encoder to reduce hallucinations during silence/noise gaps.
Built on the pulse mechanism from PDNA, applied to Transformer-based ASR.

## Tech Stack
- Python 3.10+ managed with `uv`
- PyTorch 2.x, `transformers` (Whisper), `torchaudio`, `datasets`, `jiwer`

## Commands
- `uv sync` — install dependencies
- `uv run pytest` — run tests
- `uv run pytest tests/test_foo.py -v` — run specific test
- `uv run python scripts/<script>.py` — run a script

## Conventions
- Source code lives in `src/pulse_whisper/`
- All model variants (A–E) share identical hyperparameters except architectural differences
- Use layered git commits — one logical change per commit
- Config-driven experiments: hyperparameters in `configs/`, not hardcoded
- Tests mirror source structure: `src/pulse_whisper/models/foo.py` → `tests/test_foo.py`

## Key Architecture (Variants A–E)
- A: Baseline — Frozen Whisper (control)
- B: + Noise — Random perturbation control (matched magnitude)
- C: + Pulse — Structured oscillation, fixed phase
- D: + Pulse + Phase — State-dependent oscillation with φ(h)
- E: + Adaptive PE — Learned oscillatory positional encoding

## Core Pulse Equation (from PDNA)
h' = h + α · A · sin(ω · t + φ(h))
- A: learned amplitude per dimension
- ω: learned frequencies
- φ(h): optional state-dependent phase shift
- α: mixing coefficient (initialized small, learned)

## Project Structure
```
src/pulse_whisper/
├── models/       # PulseModule, PulseWhisperEncoder, NoiseWhisper, variants
├── data/         # LibriSpeech loading, silence/noise gap injection
├── eval/         # Gapped evaluation, hallucination detection, WER metrics
├── analysis/     # α growth tracking, frequency analysis, statistics
configs/          # YAML experiment configs (prototype.yaml, full.yaml)
scripts/          # Training and evaluation entry points
tests/            # pytest tests mirroring src structure
```

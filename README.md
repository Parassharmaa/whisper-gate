# WhisperGate

**Silence-Aware Gating for Hallucination-Free Speech Recognition with Frozen Whisper**

[[Paper (PDF)]](paper/main.pdf)

WhisperGate is a lightweight trainable gate module (~12K parameters) that sits between Whisper's frozen encoder and decoder, learning to classify each encoder frame as speech or non-speech. It **eliminates 100% of hallucinations** on silence and noise inputs while preserving clean-speech word error rate.

## Key Results

| Method | Clean WER | Gap-30 WER | Halluc. (Silence) | Halluc. (Noise) |
|--------|-----------|------------|-------------------|-----------------|
| Vanilla Whisper-Tiny | 8.21% | 20.36% | 100% | 100% |
| WhisperGate (attn. bias) | 8.24% | 20.39% | **0%** | **0%** |
| WhisperGate (hard gate) | 8.24% | 34.49% | **0%** | **0%** |
| Energy VAD | 8.21% | 20.36% | 0% | 100% |

- **0% hallucination** on both silence and white noise (vs. 100% for vanilla Whisper)
- Only **0.03% clean WER overhead** with attention-bias gating
- Works on Whisper-Tiny (39M) and Whisper-Small (244M)
- Gate adds only **12,353 parameters** (< 0.02% of Whisper-Tiny)

## Architecture

```
Mel Spectrogram → [Frozen Encoder] → [SilenceGate (trainable)] → [Frozen Decoder] → Text
                                         ↓
                                    p_speech(t) ∈ [0,1]
```

**SilenceGate**: Two-layer MLP bottleneck (d_model → 32 → 1 → sigmoid) producing per-frame speech probability.

**Three gating strategies**:
- **Soft gate**: Multiplicative scaling (insufficient for hallucination prevention)
- **Hard gate**: Binary thresholding + silence short-circuit (0% hallucination, boundary artifacts)
- **Attention bias** (recommended): Log-probability cross-attention bias (0% hallucination, no WER degradation)

## Two-Stage Training

End-to-end training fails because ASR loss pushes the gate toward pass-through. Instead:

1. **Stage 1**: Train gate as standalone BCE classifier on frozen encoder representations (~10h of data, 10 epochs)
2. **Stage 2**: Plug trained gate into Whisper inference (no further training)

## Setup

```bash
# Install dependencies
uv sync

# Run gate training + evaluation (Whisper-Tiny)
uv run python scripts/run_silence_gate.py

# Evaluate vanilla baseline
uv run python scripts/eval_vanilla_small.py

# Compare gating strategies
uv run python scripts/eval_attention_bias.py

# Compare against VAD
uv run python scripts/eval_vad_comparison.py

# Test on Whisper-Small
uv run python scripts/eval_whisper_small.py
```

## Project Structure

```
src/pulse_whisper/
├── models/
│   ├── silence_gate.py      # SilenceGate and TemporalSilenceGate
│   └── gated_whisper.py     # GatedWhisper (frozen Whisper + gate)
├── data/
│   ├── dataset.py           # LibriSpeech loading
│   └── gapped_audio.py      # Silence gap augmentation
├── eval/
│   ├── gapped_eval.py       # Gap-level WER evaluation
│   └── metrics.py           # WER, CER, hallucination metrics
scripts/                      # Training and evaluation entry points
configs/                      # YAML experiment configs
paper/                        # LaTeX paper
results/                      # Experiment results (JSON)
```

## Citation

```bibtex
@article{sharma2026whispergate,
  title={WhisperGate: Silence-Aware Gating for Hallucination-Free Speech Recognition with Frozen Whisper},
  author={Sharma, Paras},
  year={2026}
}
```

## License

MIT

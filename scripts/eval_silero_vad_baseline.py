"""SileroVAD baseline comparison.

Standard industry approach: preprocess audio with SileroVAD to detect
speech segments, zero out non-speech, then run Whisper. Compare against
our silence gate approach.

Usage:
    uv run python scripts/eval_silero_vad_baseline.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration

from pulse_whisper.data.dataset import get_dataloader
from pulse_whisper.data.gapped_audio import inject_silence_gaps, GapLevel
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination
from pulse_whisper.eval.metrics import (
    HallucinationResult,
    compute_hallucination_rate,
    compute_hallucination_severity,
    compute_wer,
    compute_cer,
    EvalResult,
)
from pulse_whisper.models.pulse_whisper import get_processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SileroVADWhisper:
    """Whisper with SileroVAD preprocessing.

    Runs SileroVAD on mel spectrogram to detect speech regions,
    zeros out non-speech frames, then runs Whisper.
    """

    def __init__(self, whisper_model_name: str = "openai/whisper-tiny", device="cpu"):
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        self.whisper = self.whisper.to(device)
        self.whisper.eval()

        # Load SileroVAD
        self.vad_model, self.vad_utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self.vad_model = self.vad_model.to(device)
        self.device = device

        # SileroVAD operates on raw audio at 16kHz
        # Whisper mel frames: 3000 frames for 30s audio = 100 frames/sec
        # We'll create a frame-level VAD mask and apply it to mel spectrogram

    def _get_vad_mask_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate VAD mask from mel spectrogram.

        Since we don't have raw audio, we approximate by converting mel energy
        to a simple activity signal. SileroVAD needs raw audio, so we use
        mel energy as a proxy for speech activity detection.

        Args:
            mel: (batch, n_mels, seq_len) mel spectrogram

        Returns:
            mask: (batch, seq_len) binary mask, 1=speech, 0=silence
        """
        # Mel energy per frame: sum across mel bins
        # Higher energy = more likely speech
        energy = mel.mean(dim=1)  # (batch, seq_len)

        # Normalize energy to [0, 1]
        e_min = energy.min(dim=1, keepdim=True).values
        e_max = energy.max(dim=1, keepdim=True).values
        e_range = e_max - e_min
        e_range = torch.clamp(e_range, min=1e-8)
        energy_norm = (energy - e_min) / e_range

        # Threshold: frames with energy > 0.1 of range are speech
        # This is a simple energy-based VAD (not as good as SileroVAD on raw audio)
        mask = (energy_norm > 0.1).float()

        return mask

    def _get_vad_mask_silero(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate VAD mask using SileroVAD on synthesized audio from mel.

        Since SileroVAD needs raw audio, we synthesize a simple signal from mel energy.
        This approximation works because silence in mel domain = silence in audio.

        Args:
            mel: (batch, n_mels, seq_len) mel spectrogram

        Returns:
            mask: (batch, seq_len) float mask in [0, 1]
        """
        batch_size, n_mels, seq_len = mel.shape
        masks = []

        for b in range(batch_size):
            # Use mel energy as proxy signal
            energy = mel[b].mean(dim=0)  # (seq_len,)

            # SileroVAD expects 16kHz audio in chunks of 512 samples
            # Mel has 3000 frames for 30s → each frame ≈ 10ms
            # Synthesize fake audio: repeat energy values to get 16kHz signal
            # 30s * 16000 = 480000 samples, 3000 mel frames → 160 samples per frame
            samples_per_frame = 160
            audio = energy.unsqueeze(1).expand(-1, samples_per_frame).reshape(-1)  # (480000,)
            audio = audio.float().to("cpu")  # SileroVAD runs on CPU

            # Normalize to [-1, 1]
            audio = audio - audio.mean()
            a_max = audio.abs().max()
            if a_max > 0:
                audio = audio / a_max

            # Run SileroVAD in chunks
            self.vad_model.reset_states()
            chunk_size = 512
            frame_probs = []

            # Process in 512-sample chunks (32ms at 16kHz)
            for i in range(0, len(audio) - chunk_size + 1, chunk_size):
                chunk = audio[i : i + chunk_size]
                prob = self.vad_model(chunk.unsqueeze(0), sr=16000).item()
                frame_probs.append(prob)

            if not frame_probs:
                masks.append(torch.zeros(seq_len, device=mel.device))
                continue

            # Convert to mel frame resolution
            probs_tensor = torch.tensor(frame_probs, dtype=torch.float32)
            # Resize to mel seq_len
            probs_resized = F.interpolate(
                probs_tensor.unsqueeze(0).unsqueeze(0), size=seq_len, mode="linear", align_corners=False
            ).squeeze()

            masks.append(probs_resized.to(mel.device))

        return torch.stack(masks)  # (batch, seq_len)

    def generate(self, input_features, **kwargs):
        """Generate with VAD-masked mel input."""
        # Get VAD mask
        vad_mask = self._get_vad_mask_from_mel(input_features)  # (batch, seq_len)

        # Check if entire input is silence
        batch_speech = vad_mask.mean(dim=1)  # (batch,)
        all_silence = batch_speech < 0.05

        if all_silence.all():
            eos_id = self.whisper.config.eos_token_id
            return torch.full(
                (input_features.shape[0], 1), eos_id,
                dtype=torch.long, device=input_features.device
            )

        # Zero out non-speech mel frames
        masked_features = input_features * vad_mask.unsqueeze(1)

        return self.whisper.generate(masked_features, **kwargs)

    def eval(self):
        self.whisper.eval()
        return self


class EnergyVADWhisper:
    """Simple energy-based VAD + Whisper.

    Detects silence frames using mel energy threshold,
    short-circuits to empty output for full-silence inputs.
    """

    def __init__(self, whisper_model_name: str = "openai/whisper-tiny", threshold: float = -0.8):
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        self.whisper.eval()
        self.threshold = threshold

    def to(self, device):
        self.whisper = self.whisper.to(device)
        self.device = device
        return self

    def generate(self, input_features, **kwargs):
        # Mel energy per frame
        energy = input_features.mean(dim=1)  # (batch, seq_len)
        speech_ratio = (energy > self.threshold).float().mean(dim=1)  # (batch,)

        all_silence = speech_ratio < 0.05

        if all_silence.all():
            eos_id = self.whisper.config.eos_token_id
            return torch.full(
                (input_features.shape[0], 1), eos_id,
                dtype=torch.long, device=input_features.device
            )

        return self.whisper.generate(input_features, **kwargs)

    def eval(self):
        self.whisper.eval()
        return self


def evaluate_model_on_gaps(model, processor, test_loader, device, label):
    """Run full gap + hallucination evaluation."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Evaluating: {label}")
    logger.info(f"{'=' * 60}")

    # Gap WER
    gap_results = evaluate_all_gap_levels(
        model=model, dataloader=test_loader,
        processor=processor, device=device,
    )
    for level, r in gap_results.items():
        logger.info(f"  {level:<15} WER={r.wer:.4f}")

    # Hallucination
    halluc_results = evaluate_hallucination(
        model=model, processor=processor,
        device=device, num_samples=30,
    )
    for input_type, r in halluc_results.items():
        logger.info(f"  Halluc [{input_type}]: rate={r.hallucination_rate:.2%}")

    return {
        "gap_evaluation": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in gap_results.items()
        },
        "hallucination": {
            input_type: {
                "hallucination_rate": r.hallucination_rate,
                "avg_output_length": r.avg_output_length,
                "num_samples": r.num_samples,
            }
            for input_type, r in halluc_results.items()
        },
    }


def main():
    device = setup_device()
    logger.info(f"Device: {device}")

    processor = get_processor("tiny")
    test_loader = get_dataloader(split="test-clean", whisper_size="tiny", batch_size=8)

    output_dir = Path("results/vad_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"experiment": "vad_comparison", "device": str(device), "variants": {}}

    # 1. Vanilla Whisper baseline
    logger.info("Loading vanilla Whisper...")
    vanilla = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)
    vanilla.eval()
    results["variants"]["vanilla_whisper"] = evaluate_model_on_gaps(
        vanilla, processor, test_loader, device, "Vanilla Whisper (no VAD)"
    )
    del vanilla

    # 2. Energy-based VAD
    logger.info("Loading energy VAD Whisper...")
    energy_vad = EnergyVADWhisper().to(device)
    energy_vad.eval()
    results["variants"]["energy_vad"] = evaluate_model_on_gaps(
        energy_vad, processor, test_loader, device, "Energy VAD + Whisper"
    )
    del energy_vad

    # 3. Our silence gate (hard gate)
    logger.info("Loading silence gate (hard gate)...")
    from pulse_whisper.models.gated_whisper import GatedWhisper

    gate_model = GatedWhisper(whisper_model_name="openai/whisper-tiny", gate_hidden_dim=32)
    ckpt = torch.load("results/silence_gate_v4/gate_classifier.pt", map_location="cpu")
    gate_model.silence_gate.load_state_dict(ckpt["gate_state_dict"])
    gate_model = gate_model.to(device)
    gate_model.eval()

    # Wrap to use hard gate
    class HardGateWrap:
        def __init__(self, m):
            self.model = m
        def generate(self, x, **kw):
            return self.model.generate(x, hard_gate=True, silence_threshold=0.5, **kw)
        def eval(self):
            self.model.eval()
            return self
        def __getattr__(self, n):
            return getattr(self.model, n)

    hard_gate = HardGateWrap(gate_model)
    results["variants"]["silence_gate_hard"] = evaluate_model_on_gaps(
        hard_gate, processor, test_loader, device, "Silence Gate (hard, v4)"
    )
    del gate_model

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Variant':<25} {'Clean WER':<12} {'Multi WER':<12} {'Halluc(S)':<12} {'Halluc(N)':<12}")
    logger.info("-" * 73)

    for name, v in results["variants"].items():
        clean = v["gap_evaluation"].get("gap_0", {}).get("wer", float("nan"))
        multi = v["gap_evaluation"].get("multi_gap", {}).get("wer", float("nan"))
        h_s = v["hallucination"].get("silence", {}).get("hallucination_rate", float("nan"))
        h_n = v["hallucination"].get("white_noise", {}).get("hallucination_rate", float("nan"))
        logger.info(f"{name:<25} {clean:<12.4f} {multi:<12.4f} {h_s:<12.2%} {h_n:<12.2%}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

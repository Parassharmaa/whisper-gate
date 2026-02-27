"""Phase 1.4: Hallucination-specific testing on trained models.

Tests trained variants from Phase 1.3 on:
  1. Pure silence (30s)
  2. White noise
  3. Speech with long pauses (silence-injected real audio)

This is the killer metric — hallucination rate reduction = paper.

Usage:
    uv run python scripts/run_phase1_hallucination.py [--model-dir results/phase1_train]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import WhisperProcessor

from pulse_whisper.data.dataset import get_dataloader
from pulse_whisper.data.gapped_audio import GapLevel, inject_silence_gaps
from pulse_whisper.eval.metrics import compute_hallucination_rate, compute_hallucination_severity
from pulse_whisper.models.pulse_whisper import Variant, build_variant, get_processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_model(variant: Variant, model_dir: Path, whisper_size: str = "tiny") -> torch.nn.Module:
    """Load a trained variant model."""
    model = build_variant(variant=variant, whisper_size=whisper_size)

    if variant != Variant.A:
        final_path = model_dir / variant.name / "final.pt"
        if final_path.exists():
            ckpt = torch.load(final_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info(f"Loaded weights from {final_path}")
        else:
            logger.warning(f"No weights at {final_path}")

    return model


@torch.no_grad()
def test_silence(model, processor, device, num_samples: int = 50) -> dict:
    """Test on pure silence: 30s of log-mel silence value."""
    logger.info("Testing: Pure silence (30s)")
    n_mels, seq_len = 80, 3000
    outputs = []

    for i in range(num_samples):
        # Whisper log-mel silence is approximately -1.0
        features = torch.full((1, n_mels, seq_len), -1.0, device=device)
        gen_ids = model.generate(features, language="en", task="transcribe", max_new_tokens=440)
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        outputs.append(text)

    rate = compute_hallucination_rate(outputs)
    severity = compute_hallucination_severity(outputs)

    logger.info(f"  Hallucination rate: {rate:.2%}")
    logger.info(f"  Avg output length: {severity:.1f} words")
    logger.info(f"  Sample outputs: {outputs[:3]}")

    return {
        "input_type": "silence",
        "hallucination_rate": rate,
        "avg_output_length": severity,
        "num_samples": num_samples,
        "sample_outputs": outputs[:10],
    }


@torch.no_grad()
def test_white_noise(model, processor, device, num_samples: int = 50) -> dict:
    """Test on white noise."""
    logger.info("Testing: White noise")
    n_mels, seq_len = 80, 3000
    outputs = []

    for i in range(num_samples):
        features = (torch.randn(1, n_mels, seq_len) * 0.1).to(device)
        gen_ids = model.generate(features, language="en", task="transcribe", max_new_tokens=440)
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        outputs.append(text)

    rate = compute_hallucination_rate(outputs)
    severity = compute_hallucination_severity(outputs)

    logger.info(f"  Hallucination rate: {rate:.2%}")
    logger.info(f"  Avg output length: {severity:.1f} words")

    return {
        "input_type": "white_noise",
        "hallucination_rate": rate,
        "avg_output_length": severity,
        "num_samples": num_samples,
        "sample_outputs": outputs[:10],
    }


@torch.no_grad()
def test_speech_with_pauses(model, processor, device, num_samples: int = 30) -> dict:
    """Test on real speech with injected long pauses (50% silence)."""
    logger.info("Testing: Speech with long pauses (50% silence)")

    # Load a small set of real audio
    dataloader = get_dataloader(
        split="test-clean",
        whisper_size="tiny",
        batch_size=1,
        max_samples=num_samples,
    )

    outputs_gapped = []
    outputs_clean = []
    references = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_samples:
            break

        features = batch["input_features"].to(device)
        refs = batch["texts"]
        references.extend(refs)

        # Clean transcription
        gen_ids = model.generate(features, language="en", task="transcribe", max_new_tokens=440)
        clean_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        outputs_clean.append(clean_text)

        # Inject 50% silence (extreme gap)
        batch_size, n_mels, seq_len = features.shape
        gap_len = seq_len // 2
        gapped = features.clone()
        start = seq_len // 4  # center the gap
        gapped[:, :, start:start + gap_len] = -1.0

        gen_ids = model.generate(gapped, language="en", task="transcribe", max_new_tokens=440)
        gapped_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        outputs_gapped.append(gapped_text)

    # Compute metrics on gapped outputs
    # "Hallucination" here: tokens that appear in gapped output but NOT in reference
    # We measure: how much extra/fabricated text appears
    hallucinated_count = 0
    extra_words_total = 0
    for gapped_text, ref_text in zip(outputs_gapped, references):
        ref_words = set(ref_text.lower().split())
        gapped_words = gapped_text.lower().split()
        extra_words = [w for w in gapped_words if w not in ref_words]
        if extra_words:
            hallucinated_count += 1
            extra_words_total += len(extra_words)

    halluc_rate = hallucinated_count / max(1, len(outputs_gapped))
    avg_extra = extra_words_total / max(1, len(outputs_gapped))

    logger.info(f"  Hallucination rate (extra words): {halluc_rate:.2%}")
    logger.info(f"  Avg extra words per sample: {avg_extra:.1f}")
    logger.info(f"  Sample clean: {outputs_clean[:2]}")
    logger.info(f"  Sample gapped: {outputs_gapped[:2]}")

    return {
        "input_type": "speech_with_pauses",
        "hallucination_rate": halluc_rate,
        "avg_extra_words": avg_extra,
        "num_samples": len(outputs_gapped),
        "sample_clean": outputs_clean[:5],
        "sample_gapped": outputs_gapped[:5],
        "sample_references": references[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1.4: Hallucination testing")
    parser.add_argument("--model-dir", default="results/phase1_train", help="Dir with trained models")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--output", default="results/phase1_hallucination.json", help="Output file")
    parser.add_argument("--variants", default="A,B,C,D", help="Variants to test")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per test")
    parser.add_argument("--whisper-size", default="tiny", help="Whisper model size")
    args = parser.parse_args()

    device = setup_device(args.device)
    logger.info(f"Using device: {device}")

    model_dir = Path(args.model_dir)
    variants = [Variant[v.strip()] for v in args.variants.split(",")]

    all_results = {
        "experiment": "phase1_hallucination",
        "device": str(device),
        "variants": {},
    }

    for variant in variants:
        logger.info("\n" + "=" * 70)
        logger.info(f"HALLUCINATION TEST: Variant {variant.name} ({variant.value})")
        logger.info("=" * 70)

        model = load_trained_model(variant, model_dir, args.whisper_size)
        model = model.to(device)
        model.eval()

        processor = get_processor(args.whisper_size)

        variant_results = {}

        # Test 1: Pure silence
        variant_results["silence"] = test_silence(
            model, processor, device, args.num_samples
        )

        # Test 2: White noise
        variant_results["white_noise"] = test_white_noise(
            model, processor, device, args.num_samples
        )

        # Test 3: Speech with long pauses
        variant_results["speech_with_pauses"] = test_speech_with_pauses(
            model, processor, device, min(30, args.num_samples)
        )

        all_results["variants"][variant.name] = variant_results

        # Save intermediate
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("HALLUCINATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Variant':<10} {'Silence':<12} {'Noise':<12} {'Pauses':<12}")
    logger.info("-" * 46)
    for name, vr in all_results["variants"].items():
        s_rate = vr["silence"]["hallucination_rate"]
        n_rate = vr["white_noise"]["hallucination_rate"]
        p_rate = vr["speech_with_pauses"]["hallucination_rate"]
        logger.info(f"{name:<10} {s_rate:<12.2%} {n_rate:<12.2%} {p_rate:<12.2%}")

    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

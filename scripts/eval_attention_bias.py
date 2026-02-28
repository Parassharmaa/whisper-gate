"""Evaluate cross-attention bias gating vs hard gate vs soft gate.

Instead of zeroing encoder states, cross-attention bias adds log(gate_prob)
to the decoder's cross-attention scores, softly suppressing attention to
non-speech frames while preserving context at boundaries.

Usage:
    uv run python scripts/eval_attention_bias.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

from pulse_whisper.data.dataset import get_dataloader
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination
from pulse_whisper.eval.metrics import (
    compute_hallucination_rate,
    compute_hallucination_severity,
    HallucinationResult,
)
from pulse_whisper.models.gated_whisper import GatedWhisper
from pulse_whisper.models.pulse_whisper import get_processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class GateWrapper:
    """Wraps GatedWhisper with a specific gating mode."""

    def __init__(self, model, mode="soft", silence_threshold=0.5):
        self.model = model
        self.mode = mode
        self.silence_threshold = silence_threshold

    def generate(self, input_features, **kwargs):
        return self.model.generate(
            input_features,
            hard_gate=(self.mode == "hard"),
            attention_bias=(self.mode == "attention_bias"),
            silence_threshold=self.silence_threshold,
            **kwargs,
        )

    def eval(self):
        self.model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.model, name)


def evaluate_hallucination_mode(model, processor, device, mode, num_samples=30):
    """Test hallucination with specific gating mode."""
    n_mels, seq_len = 80, 3000
    results = {}

    for input_type, gen_fn in [
        ("silence", lambda: torch.zeros(1, n_mels, seq_len) - 1.0),
        ("white_noise", lambda: torch.randn(1, n_mels, seq_len) * 0.1),
    ]:
        all_outputs = []
        for i in range(num_samples):
            feats = gen_fn().to(device)
            ids = model.generate(
                feats,
                hard_gate=(mode == "hard"),
                attention_bias=(mode == "attention_bias"),
                silence_threshold=0.5,
                language="en", task="transcribe", max_new_tokens=440,
            )
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            all_outputs.append(text)

        rate = compute_hallucination_rate(all_outputs)
        results[input_type] = {
            "hallucination_rate": rate,
            "num_samples": num_samples,
        }
        logger.info(f"  {input_type}: {rate:.2%}")

    return results


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model with v4 trained gate
    model = GatedWhisper(whisper_model_name="openai/whisper-tiny", gate_hidden_dim=32)
    ckpt = torch.load("results/silence_gate_v4/gate_classifier.pt", map_location="cpu")
    model.silence_gate.load_state_dict(ckpt["gate_state_dict"])
    model = model.to(device)
    model.eval()

    processor = get_processor("tiny")
    test_loader = get_dataloader(split="test-clean", whisper_size="tiny", batch_size=8)

    output_dir = Path("results/attention_bias")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"experiment": "attention_bias_comparison", "variants": {}}

    modes = ["soft", "hard", "attention_bias"]

    for mode in modes:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"MODE: {mode}")
        logger.info(f"{'=' * 60}")

        wrapped = GateWrapper(model, mode=mode)

        # Gap WER
        logger.info("Gap WER:")
        gap_results = evaluate_all_gap_levels(
            model=wrapped, dataloader=test_loader,
            processor=processor, device=device,
        )
        for level, r in gap_results.items():
            logger.info(f"  {level:<15} WER={r.wer:.4f}")

        # Hallucination
        logger.info("Hallucination:")
        halluc_results = evaluate_hallucination_mode(model, processor, device, mode)

        results["variants"][mode] = {
            "gap_evaluation": {
                level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
                for level, r in gap_results.items()
            },
            "hallucination": halluc_results,
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: Soft vs Hard vs Attention Bias")
    logger.info("=" * 70)
    logger.info(f"{'Mode':<20} {'Clean WER':<12} {'Multi WER':<12} {'gap30 WER':<12} {'Halluc(S)':<12}")
    logger.info("-" * 68)

    for mode in modes:
        v = results["variants"][mode]
        clean = v["gap_evaluation"].get("gap_0", {}).get("wer", float("nan"))
        multi = v["gap_evaluation"].get("multi_gap", {}).get("wer", float("nan"))
        g30 = v["gap_evaluation"].get("gap_30", {}).get("wer", float("nan"))
        h_s = v["hallucination"].get("silence", {}).get("hallucination_rate", float("nan"))
        logger.info(f"{mode:<20} {clean:<12.4f} {multi:<12.4f} {g30:<12.4f} {h_s:<12.2%}")

    logger.info(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

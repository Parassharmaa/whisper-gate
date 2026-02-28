"""Evaluate hard-gated GatedWhisper on speech+silence gaps.

Tests whether hard gating preserves WER on mixed audio (speech with silence gaps).
"""

import json
import logging
import sys
from pathlib import Path
from functools import partial

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pulse_whisper.models.gated_whisper import GatedWhisper
from pulse_whisper.models.silence_gate import SilenceGate
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination
from pulse_whisper.data.dataset import get_dataloader
from pulse_whisper.models.pulse_whisper import get_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-6s %(message)s")
logger = logging.getLogger(__name__)


class HardGateWrapper:
    """Wraps GatedWhisper to force hard_gate=True during generate()."""

    def __init__(self, model: GatedWhisper, silence_threshold: float = 0.5):
        self.model = model
        self.silence_threshold = silence_threshold

    def generate(self, input_features, **kwargs):
        return self.model.generate(
            input_features,
            hard_gate=True,
            silence_threshold=self.silence_threshold,
            **kwargs,
        )

    def eval(self):
        self.model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.model, name)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    gate_path = Path("results/silence_gate_v4/gate_classifier.pt")
    if not gate_path.exists():
        logger.error(f"Gate checkpoint not found: {gate_path}")
        return

    # Build model
    model = GatedWhisper(whisper_model_name="openai/whisper-tiny", gate_hidden_dim=32)
    ckpt = torch.load(gate_path, map_location="cpu")
    model.silence_gate.load_state_dict(ckpt["gate_state_dict"])
    model = model.to(device)
    model.eval()

    processor = get_processor("tiny")
    test_loader = get_dataloader(split="test-clean", whisper_size="tiny", batch_size=8)

    # --- Soft gate WER (control) ---
    logger.info("\n" + "=" * 60)
    logger.info("SOFT GATE — gap evaluation")
    logger.info("=" * 60)
    soft_results = evaluate_all_gap_levels(
        model=model, dataloader=test_loader,
        processor=processor, device=device,
    )
    for level, r in soft_results.items():
        logger.info(f"  {level:<15} WER={r.wer:.4f}  CER={r.cer:.4f}")

    # --- Hard gate WER ---
    logger.info("\n" + "=" * 60)
    logger.info("HARD GATE (threshold=0.5) — gap evaluation")
    logger.info("=" * 60)
    hard_model = HardGateWrapper(model, silence_threshold=0.5)
    hard_results = evaluate_all_gap_levels(
        model=hard_model, dataloader=test_loader,
        processor=processor, device=device,
    )
    for level, r in hard_results.items():
        logger.info(f"  {level:<15} WER={r.wer:.4f}  CER={r.cer:.4f}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: Soft vs Hard Gate")
    logger.info("=" * 60)
    logger.info(f"{'Gap Level':<15} {'Soft WER':<12} {'Hard WER':<12} {'Delta':<12}")
    logger.info("-" * 50)
    for level in soft_results:
        sw = soft_results[level].wer
        hw = hard_results[level].wer
        delta = hw - sw
        sign = "+" if delta > 0 else ""
        logger.info(f"{level:<15} {sw:<12.4f} {hw:<12.4f} {sign}{delta:<12.4f}")

    # Save results
    output = {
        "experiment": "hard_gate_wer_comparison",
        "silence_threshold": 0.5,
        "soft_gate": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in soft_results.items()
        },
        "hard_gate": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in hard_results.items()
        },
    }
    out_path = Path("results/silence_gate_v4/hard_gate_wer.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

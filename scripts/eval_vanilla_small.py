"""Get vanilla whisper-small gap WER metrics for baseline comparison.

Usage:
    uv run python scripts/eval_vanilla_small.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from pulse_whisper.data.dataset import get_dataloader
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    whisper_model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    model = model.to(device)
    model.eval()

    batch_size = 16 if device.type == "cuda" else 4
    test_loader = get_dataloader(
        split="test-clean", whisper_size="small", batch_size=batch_size,
    )

    logger.info(f"Vanilla whisper-small gap WER (batch_size={batch_size}):")
    gap_results = evaluate_all_gap_levels(
        model=model, dataloader=test_loader,
        processor=processor, device=device,
    )
    for level, r in gap_results.items():
        logger.info(f"  {level:<15} WER={r.wer:.4f}")

    output_dir = Path("results/silence_gate_small")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "vanilla_whisper_small_gap_wer",
        "gap_evaluation": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in gap_results.items()
        },
    }

    with open(output_dir / "vanilla_gap_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_dir / 'vanilla_gap_results.json'}")


if __name__ == "__main__":
    main()

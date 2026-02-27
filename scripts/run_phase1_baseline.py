"""Phase 1.1: Baseline gapped evaluation on frozen Whisper-Tiny.

Evaluates frozen Whisper-Tiny on test-clean with gap levels 0/5/15/30/multi.
Collects WER + hallucination rate at each level.
Establishes the baseline degradation curve.

Usage:
    uv run python scripts/run_phase1_baseline.py [--max-samples N] [--device DEVICE]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from pulse_whisper.data.dataset import get_dataloader
from pulse_whisper.data.gapped_audio import GapLevel
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 1.1: Baseline gapped evaluation")
    parser.add_argument("--whisper-size", default="tiny", help="Whisper model size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples (None=all)")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per gap level")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not set)")
    parser.add_argument("--output", default="results/phase1_baseline.json", help="Output file")
    parser.add_argument("--hallucination-samples", type=int, default=20, help="Hallucination test samples")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load model and processor
    model_name = f"openai/whisper-{args.whisper_size}"
    logger.info(f"Loading {model_name}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.generation_config.max_length = None  # use max_new_tokens instead
    model.eval()
    processor = WhisperProcessor.from_pretrained(model_name)

    # Load test data
    logger.info("Loading test-clean dataset...")
    dataloader = get_dataloader(
        split="test-clean",
        whisper_size=args.whisper_size,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    logger.info(f"Test samples: {len(dataloader.dataset)}")

    # Run gapped evaluation at all levels
    logger.info("=" * 60)
    logger.info("PHASE 1.1: Baseline Gapped Evaluation")
    logger.info("=" * 60)

    gap_results = evaluate_all_gap_levels(
        model=model,
        dataloader=dataloader,
        processor=processor,
        device=device,
        max_batches=args.max_batches,
    )

    # Print results table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS: WER by Gap Level")
    logger.info("=" * 60)
    logger.info(f"{'Gap Level':<15} {'WER':<10} {'CER':<10} {'Samples':<10}")
    logger.info("-" * 45)
    for level, result in gap_results.items():
        logger.info(f"{level:<15} {result.wer:<10.4f} {result.cer:<10.4f} {result.num_samples:<10}")

    # Hallucination test
    logger.info("\n" + "=" * 60)
    logger.info("HALLUCINATION TEST")
    logger.info("=" * 60)

    halluc_results = evaluate_hallucination(
        model=model,
        processor=processor,
        device=device,
        num_samples=args.hallucination_samples,
    )

    for input_type, result in halluc_results.items():
        logger.info(f"{input_type}: rate={result.hallucination_rate:.2%}, avg_length={result.avg_output_length:.1f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "experiment": "phase1_baseline",
        "whisper_size": args.whisper_size,
        "device": str(device),
        "gap_evaluation": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in gap_results.items()
        },
        "hallucination": {
            input_type: {
                "hallucination_rate": r.hallucination_rate,
                "avg_output_length": r.avg_output_length,
                "num_samples": r.num_samples,
                "sample_outputs": r.outputs[:5],
            }
            for input_type, r in halluc_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

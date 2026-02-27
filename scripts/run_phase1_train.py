"""Phase 1.3: Train pulse on Whisper-Tiny (Go/No-Go decision).

Trains 4 variants on 10h LibriSpeech with gap augmentation:
  A: Baseline (frozen Whisper, no injection)
  B: +Noise (random perturbation control)
  C: +Pulse (structured oscillation, fixed phase)
  D: +Pulse+Phase (state-dependent oscillation)

After training, evaluates each on gapped test set.
GO condition: Pulse (C or D) beats Baseline (A) AND Noise (B) on multi-gap WER.

Usage:
    uv run python scripts/run_phase1_train.py [--config configs/prototype.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from pulse_whisper.data.dataset import get_10h_subset_dataloader, get_dataloader
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination
from pulse_whisper.models.pulse_whisper import (
    PulseWhisperEncoder,
    Variant,
    build_variant,
    get_processor,
)
from pulse_whisper.training.config import ExperimentConfig, load_config
from pulse_whisper.training.trainer import Trainer
from pulse_whisper.analysis.alpha_analysis import extract_pulse_stats, log_pulse_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

VARIANTS_TO_TRAIN = [Variant.A, Variant.B, Variant.C, Variant.D]


def setup_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_variant(
    variant: Variant,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Train a single variant and return training history + eval results."""
    logger.info("=" * 70)
    logger.info(f"TRAINING VARIANT: {variant.name} ({variant.value})")
    logger.info("=" * 70)

    variant_dir = output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_variant(
        variant=variant,
        whisper_size=config.model.whisper_size,
        n_frequencies=config.model.n_frequencies,
        alpha_init=config.model.alpha_init,
    )

    trainable = model.trainable_param_count()
    total = model.total_param_count()
    logger.info(f"Model: {total:,} total params, {trainable:,} trainable")

    # Variant A has no trainable params — skip training
    if variant == Variant.A:
        logger.info("Variant A is frozen baseline — skipping training")
        return {"variant": variant.name, "training": None, "trainable_params": 0}

    # Load training data
    logger.info("Loading 10h training subset...")
    train_loader = get_10h_subset_dataloader(
        whisper_size=config.model.whisper_size,
        batch_size=config.training.batch_size,
        gap_augmentation=config.training.gap_augmentation,
        gap_fractions=config.training.gap_fractions,
    )
    logger.info(f"Training samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")

    # Update checkpoint dir per variant
    config.logging.checkpoint_dir = str(variant_dir / "checkpoints")

    # Train
    start_time = time.time()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
    )
    history = trainer.train()
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time / 60:.1f} minutes")

    # Log pulse stats after training
    if variant in (Variant.C, Variant.D):
        logger.info("Post-training pulse parameter stats:")
        log_pulse_stats(model)

    # Save final model
    final_path = variant_dir / "final.pt"
    pulse_state = {
        k: v for k, v in model.state_dict().items()
        if k.startswith("injected_layers")
    }
    torch.save({"model_state_dict": pulse_state}, final_path)
    logger.info(f"Model saved to {final_path}")

    return {
        "variant": variant.name,
        "training": {
            "train_loss": history["train_loss"],
            "time_minutes": train_time / 60,
        },
        "trainable_params": trainable,
    }


def evaluate_variant(
    variant: Variant,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    max_eval_batches: int | None = None,
) -> dict:
    """Evaluate a trained variant on gapped test set."""
    logger.info("=" * 70)
    logger.info(f"EVALUATING VARIANT: {variant.name} ({variant.value})")
    logger.info("=" * 70)

    variant_dir = output_dir / variant.name

    # Build model and load trained weights
    model = build_variant(
        variant=variant,
        whisper_size=config.model.whisper_size,
        n_frequencies=config.model.n_frequencies,
        alpha_init=config.model.alpha_init,
    )

    if variant != Variant.A:
        final_path = variant_dir / "final.pt"
        if final_path.exists():
            ckpt = torch.load(final_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info(f"Loaded trained weights from {final_path}")
        else:
            logger.warning(f"No trained weights found at {final_path}, using init weights")

    model = model.to(device)
    model.eval()

    processor = get_processor(config.model.whisper_size)

    # Load test data
    test_loader = get_dataloader(
        split="test-clean",
        whisper_size=config.model.whisper_size,
        batch_size=8,
        max_samples=config.eval.max_eval_samples,
    )
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Gapped evaluation
    gap_results = evaluate_all_gap_levels(
        model=model,
        dataloader=test_loader,
        processor=processor,
        device=device,
        max_batches=max_eval_batches,
    )

    logger.info(f"\n{'Gap Level':<15} {'WER':<10} {'CER':<10} {'Samples':<10}")
    logger.info("-" * 45)
    for level, result in gap_results.items():
        logger.info(f"{level:<15} {result.wer:<10.4f} {result.cer:<10.4f} {result.num_samples:<10}")

    # Hallucination test
    halluc_results = evaluate_hallucination(
        model=model,
        processor=processor,
        device=device,
        num_samples=20,
    )

    for input_type, result in halluc_results.items():
        logger.info(f"Hallucination [{input_type}]: rate={result.hallucination_rate:.2%}, "
                     f"avg_length={result.avg_output_length:.1f}")

    # Pulse stats
    pulse_stats = []
    if variant in (Variant.C, Variant.D):
        stats = extract_pulse_stats(model)
        pulse_stats = [
            {"layer": s.layer_idx, "alpha": s.alpha, "amp_mean": s.amplitude_mean,
             "omega_mean": s.omega_mean, "omega_std": s.omega_std}
            for s in stats
        ]

    return {
        "variant": variant.name,
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
        "pulse_stats": pulse_stats,
    }


def go_no_go_decision(results: dict) -> dict:
    """Analyze results for Go/No-Go decision.

    GO: Pulse (C or D) beats Baseline (A) AND Noise (B) on multi-gap WER.
    """
    logger.info("\n" + "=" * 70)
    logger.info("GO / NO-GO DECISION")
    logger.info("=" * 70)

    variant_wers = {}
    for variant_result in results["evaluations"]:
        name = variant_result["variant"]
        gap_eval = variant_result["gap_evaluation"]
        variant_wers[name] = {level: gap_eval[level]["wer"] for level in gap_eval}

    # Print comparison table
    levels = list(next(iter(variant_wers.values())).keys())
    header = f"{'Variant':<12}" + "".join(f"{l:<12}" for l in levels)
    logger.info(header)
    logger.info("-" * len(header))
    for name in ["A", "B", "C", "D"]:
        if name in variant_wers:
            row = f"{name:<12}" + "".join(f"{variant_wers[name][l]:<12.4f}" for l in levels)
            logger.info(row)

    # GO conditions
    multi_gap_key = "multi_gap"
    decision = {"go": False, "reason": ""}

    if multi_gap_key not in variant_wers.get("A", {}):
        decision["reason"] = "Missing multi_gap results"
        logger.info(f"\nDECISION: INCONCLUSIVE — {decision['reason']}")
        return decision

    a_multi = variant_wers["A"][multi_gap_key]
    b_multi = variant_wers.get("B", {}).get(multi_gap_key, float("inf"))
    c_multi = variant_wers.get("C", {}).get(multi_gap_key, float("inf"))
    d_multi = variant_wers.get("D", {}).get(multi_gap_key, float("inf"))

    best_pulse = min(c_multi, d_multi)
    best_pulse_name = "C" if c_multi <= d_multi else "D"

    logger.info(f"\nMulti-gap WER: A={a_multi:.4f}, B={b_multi:.4f}, C={c_multi:.4f}, D={d_multi:.4f}")
    logger.info(f"Best pulse variant: {best_pulse_name} (WER={best_pulse:.4f})")

    if best_pulse < a_multi and best_pulse < b_multi:
        decision["go"] = True
        delta_a = (a_multi - best_pulse) / a_multi * 100
        delta_b = (b_multi - best_pulse) / b_multi * 100
        decision["reason"] = (
            f"GO! {best_pulse_name} beats A by {delta_a:.1f}% and B by {delta_b:.1f}% "
            f"on multi-gap WER"
        )
    elif best_pulse < a_multi:
        decision["reason"] = (
            f"PARTIAL: {best_pulse_name} beats A but not B on multi-gap WER. "
            f"May need different insertion strategy."
        )
    else:
        decision["reason"] = (
            f"NO-GO: Pulse does not beat baseline on multi-gap WER. "
            f"Consider Whisper-Small or different approach."
        )

    # Check hallucination improvement
    for variant_result in results["evaluations"]:
        name = variant_result["variant"]
        halluc = variant_result.get("hallucination", {})
        silence_rate = halluc.get("silence", {}).get("hallucination_rate", 1.0)
        logger.info(f"Hallucination rate [{name}]: silence={silence_rate:.2%}")

    logger.info(f"\nDECISION: {decision['reason']}")
    return decision


def main():
    parser = argparse.ArgumentParser(description="Phase 1.3: Train pulse on Whisper-Tiny")
    parser.add_argument("--config", default="configs/prototype.yaml", help="Config file")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not set)")
    parser.add_argument("--output-dir", default="results/phase1_train", help="Output directory")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Limit eval batches")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval samples")
    parser.add_argument("--variants", default="A,B,C,D", help="Comma-separated variant names")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    args = parser.parse_args()

    device = setup_device(args.device)
    logger.info(f"Using device: {device}")

    config = load_config(args.config)
    logger.info(f"Config: {args.config}")

    # Apply overrides
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.max_epochs:
        config.training.max_epochs = args.max_epochs
    if args.max_eval_samples:
        config.eval.max_eval_samples = args.max_eval_samples

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [Variant[v.strip()] for v in args.variants.split(",")]
    logger.info(f"Variants: {[v.name for v in variants]}")

    results = {
        "experiment": "phase1_train",
        "config": {
            "whisper_size": config.model.whisper_size,
            "max_epochs": config.training.max_epochs,
            "batch_size": config.training.batch_size,
            "lr": config.training.lr,
            "gap_augmentation": config.training.gap_augmentation,
        },
        "device": str(device),
        "training": [],
        "evaluations": [],
    }

    # Phase 1: Training
    if not args.skip_training:
        logger.info("\n" + "#" * 70)
        logger.info("# PHASE 1.3: TRAINING")
        logger.info("#" * 70)

        for variant in variants:
            train_result = train_variant(variant, config, device, output_dir)
            results["training"].append(train_result)

            # Save intermediate results
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

    # Phase 2: Evaluation
    logger.info("\n" + "#" * 70)
    logger.info("# PHASE 1.3: EVALUATION")
    logger.info("#" * 70)

    for variant in variants:
        eval_result = evaluate_variant(
            variant, config, device, output_dir,
            max_eval_batches=args.max_eval_batches,
        )
        results["evaluations"].append(eval_result)

        # Save intermediate results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Go/No-Go decision
    decision = go_no_go_decision(results)
    results["decision"] = decision

    # Save final results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

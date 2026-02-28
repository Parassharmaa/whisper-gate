"""Silence-Aware Gating experiment.

Trains a lightweight bottleneck gate between Whisper encoder and decoder
to suppress non-speech frames and prevent hallucination.

Usage:
    uv run python scripts/run_silence_gate.py [--config configs/silence_gate.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from pulse_whisper.data.dataset import get_10h_subset_dataloader, get_dataloader
from pulse_whisper.data.gapped_audio import random_gap_augmentation
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination
from pulse_whisper.models.gated_whisper import build_gated_whisper
from pulse_whisper.models.pulse_whisper import build_variant, Variant, get_processor
from pulse_whisper.training.config import load_config

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


def downsample_mask(speech_mask: torch.Tensor, target_len: int) -> torch.Tensor:
    """Downsample speech mask from mel resolution to encoder output resolution.

    Whisper's encoder conv layers downsample by 2x (conv1 stride=1, conv2 stride=2).
    We use avg pooling and threshold to downsample the mask.

    Args:
        speech_mask: (batch, mel_seq_len) float mask, 1.0=speech, 0.0=silence.
        target_len: Target sequence length (encoder output length).

    Returns:
        Downsampled mask (batch, target_len).
    """
    mel_len = speech_mask.shape[1]
    if mel_len == target_len:
        return speech_mask

    # Use adaptive avg pool to downsample
    # (batch, 1, mel_len) -> (batch, 1, target_len)
    downsampled = torch.nn.functional.adaptive_avg_pool1d(
        speech_mask.unsqueeze(1), target_len
    ).squeeze(1)

    # Threshold: if >50% of source frames are speech, mark as speech
    return (downsampled > 0.5).float()


def train_gated_whisper(config, device: torch.device, output_dir: Path) -> dict:
    """Train the silence gate."""
    logger.info("=" * 70)
    logger.info("TRAINING: silence_gate")
    logger.info("=" * 70)

    variant_dir = output_dir / "silence_gate"
    variant_dir.mkdir(parents=True, exist_ok=True)

    model = build_gated_whisper(
        whisper_size=config.model.whisper_size,
        gate_hidden_dim=getattr(config.model, 'gate_hidden_dim', 32),
        gate_loss_weight=getattr(config.model, 'gate_loss_weight', 1.0),
    )
    model = model.to(device)

    trainable = model.trainable_param_count()
    total = model.total_param_count()
    logger.info(f"Model: {total:,} total params, {trainable:,} trainable (gate only)")

    # Load training data
    logger.info("Loading 10h training subset...")
    train_loader = get_10h_subset_dataloader(
        whisper_size=config.model.whisper_size,
        batch_size=config.training.batch_size,
        gap_augmentation=False,  # We handle gap augmentation manually to get masks
    )
    logger.info(f"Training samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")

    tc = config.training
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=tc.lr, weight_decay=0.01)

    total_steps = tc.max_epochs * len(train_loader)
    warmup_steps = tc.warmup_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )

    use_amp = tc.fp16 and device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    ckpt_dir = variant_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    history = {"train_loss": [], "gate_loss": []}
    global_step = 0

    for epoch in range(tc.max_epochs):
        model.train()
        total_loss = 0.0
        total_gate_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            # Apply gap augmentation WITH mask
            input_features, speech_mask = random_gap_augmentation(
                input_features,
                gap_fractions=tc.gap_fractions,
                return_mask=True,
            )

            # With some probability, replace individual samples with pure silence
            # This teaches the gate what full silence looks like
            # Cap at max 2 per batch to ensure enough speech samples for ASR loss
            batch_size = input_features.shape[0]
            silence_prob = getattr(config.model, 'silence_injection_rate', 0.1)
            n_silence = 0
            max_silence_per_batch = max(1, batch_size // 8)  # At most 12.5% of batch
            for b in range(batch_size):
                if n_silence >= max_silence_per_batch:
                    break
                if torch.rand(1).item() < silence_prob:
                    input_features[b] = -1.0  # Whisper silence value
                    speech_mask[b] = 0.0  # All silence
                    labels[b] = -100  # Ignore all tokens (empty transcript)
                    n_silence += 1

            # Downsample speech_mask to encoder output resolution
            # Whisper encoder output is input_len // 2 (conv2 stride=2)
            encoder_out_len = input_features.shape[2] // 2
            speech_mask_ds = downsample_mask(speech_mask, encoder_out_len)

            amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_features=input_features,
                    labels=labels,
                    speech_mask=speech_mask_ds,
                )
                loss = outputs["loss"]

            # Guard against NaN loss (can happen if all labels are -100)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  step {global_step + 1}: skipping NaN/Inf loss")
                optimizer.zero_grad()
                global_step += 1
                continue

            optimizer.zero_grad()
            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            global_step += 1
            if global_step <= warmup_steps:
                warmup_factor = global_step / max(1, warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = tc.lr * warmup_factor
            else:
                scheduler.step()

            total_loss += loss.item()
            gate_loss_val = outputs["gate_loss"].item() if outputs["gate_loss"] is not None else 0.0
            total_gate_loss += gate_loss_val
            num_batches += 1

            if num_batches % config.logging.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                avg_gate = total_gate_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"  step {global_step}: loss={avg_loss:.4f}, gate_loss={avg_gate:.4f}, lr={lr:.2e}")

        epoch_loss = total_loss / max(1, num_batches)
        epoch_gate = total_gate_loss / max(1, num_batches)
        history["train_loss"].append(epoch_loss)
        history["gate_loss"].append(epoch_gate)
        logger.info(f"Epoch {epoch + 1}/{tc.max_epochs}: loss={epoch_loss:.4f}, gate_loss={epoch_gate:.4f}")

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time / 60:.1f} minutes")

    # Log gate stats
    gate = model.silence_gate
    w1 = gate.gate_mlp[0].weight
    w2 = gate.gate_mlp[2].weight
    b2 = gate.gate_mlp[2].bias
    logger.info(f"Gate stats: W1 norm={w1.norm():.4f}, W2 norm={w2.norm():.4f}, bias={b2.item():.4f}")

    # Save model
    final_path = variant_dir / "final.pt"
    gate_state = {
        k: v for k, v in model.state_dict().items()
        if k.startswith("silence_gate")
    }
    torch.save({"model_state_dict": gate_state}, final_path)
    logger.info(f"Model saved to {final_path}")

    return {
        "variant": "silence_gate",
        "training": {
            "train_loss": history["train_loss"],
            "gate_loss": history["gate_loss"],
            "time_minutes": train_time / 60,
        },
        "trainable_params": trainable,
    }


def evaluate_gated_whisper(
    config,
    device: torch.device,
    output_dir: Path,
    max_eval_batches: int | None = None,
) -> dict:
    """Evaluate the trained silence gate model."""
    logger.info("=" * 70)
    logger.info("EVALUATING: silence_gate")
    logger.info("=" * 70)

    variant_dir = output_dir / "silence_gate"

    model = build_gated_whisper(
        whisper_size=config.model.whisper_size,
        gate_hidden_dim=getattr(config.model, 'gate_hidden_dim', 32),
        gate_loss_weight=getattr(config.model, 'gate_loss_weight', 1.0),
    )

    # Load trained weights
    final_path = variant_dir / "final.pt"
    if final_path.exists():
        ckpt = torch.load(final_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Loaded trained weights from {final_path}")
    else:
        logger.warning(f"No trained weights at {final_path}, using init weights")

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
        num_samples=30,
    )

    for input_type, result in halluc_results.items():
        logger.info(f"Hallucination [{input_type}]: rate={result.hallucination_rate:.2%}, "
                     f"avg_length={result.avg_output_length:.1f}")

    # Gate behavior analysis on silence
    logger.info("\nGate behavior analysis:")
    with torch.no_grad():
        # Pure silence input
        silence_mel = torch.zeros(1, 80, 3000, device=device)
        encoder_hidden = model._run_encoder(silence_mel)
        _, gate_probs = model.silence_gate(encoder_hidden)
        logger.info(f"  Silence input: gate mean={gate_probs.mean():.4f}, "
                     f"min={gate_probs.min():.4f}, max={gate_probs.max():.4f}")

        # Speech input (use first test sample)
        for batch in test_loader:
            speech_mel = batch["input_features"][:1].to(device)
            encoder_hidden = model._run_encoder(speech_mel)
            _, gate_probs = model.silence_gate(encoder_hidden)
            logger.info(f"  Speech input:  gate mean={gate_probs.mean():.4f}, "
                         f"min={gate_probs.min():.4f}, max={gate_probs.max():.4f}")
            break

    return {
        "variant": "silence_gate",
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
    parser = argparse.ArgumentParser(description="Silence-Aware Gating Experiment")
    parser.add_argument("--config", default="configs/silence_gate.yaml", help="Config file")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--output-dir", default="results/silence_gate", help="Output directory")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline eval")
    args = parser.parse_args()

    device = setup_device(args.device)
    logger.info(f"Using device: {device}")

    config = load_config(args.config)
    if args.max_eval_samples:
        config.eval.max_eval_samples = args.max_eval_samples

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "silence_gate",
        "config": {
            "whisper_size": config.model.whisper_size,
            "max_epochs": config.training.max_epochs,
            "batch_size": config.training.batch_size,
            "lr": config.training.lr,
            "gate_hidden_dim": getattr(config.model, 'gate_hidden_dim', 32),
            "gate_loss_weight": getattr(config.model, 'gate_loss_weight', 1.0),
        },
        "device": str(device),
        "training": [],
        "evaluations": [],
    }

    # Baseline evaluation (frozen Whisper, no gate)
    if not args.skip_baseline:
        logger.info("\n" + "#" * 70)
        logger.info("# BASELINE EVALUATION (Frozen Whisper)")
        logger.info("#" * 70)

        baseline_model = build_variant(Variant.A, whisper_size=config.model.whisper_size)
        baseline_model = baseline_model.to(device)
        baseline_model.eval()

        processor = get_processor(config.model.whisper_size)
        test_loader = get_dataloader(
            split="test-clean",
            whisper_size=config.model.whisper_size,
            batch_size=8,
            max_samples=config.eval.max_eval_samples,
        )

        gap_results = evaluate_all_gap_levels(
            model=baseline_model, dataloader=test_loader,
            processor=processor, device=device,
            max_batches=args.max_eval_batches,
        )
        halluc_results = evaluate_hallucination(
            model=baseline_model, processor=processor,
            device=device, num_samples=30,
        )

        baseline_eval = {
            "variant": "baseline",
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
        results["evaluations"].append(baseline_eval)

        logger.info(f"\nBaseline WER: " + ", ".join(
            f"{k}={v['wer']:.4f}" for k, v in baseline_eval["gap_evaluation"].items()
        ))

        del baseline_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Train silence gate
    if not args.skip_training:
        logger.info("\n" + "#" * 70)
        logger.info("# TRAINING: silence_gate")
        logger.info("#" * 70)

        train_result = train_gated_whisper(config, device, output_dir)
        results["training"].append(train_result)

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Evaluate
    logger.info("\n" + "#" * 70)
    logger.info("# EVALUATING: silence_gate")
    logger.info("#" * 70)

    eval_result = evaluate_gated_whisper(
        config, device, output_dir,
        max_eval_batches=args.max_eval_batches,
    )
    results["evaluations"].append(eval_result)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Variant':<25} {'Clean WER':<12} {'Multi-gap WER':<15} {'Halluc (silence)':<18}")
    logger.info("-" * 70)
    for ev in results["evaluations"]:
        clean = ev["gap_evaluation"].get("gap_0", {}).get("wer", float("nan"))
        multi = ev["gap_evaluation"].get("multi_gap", {}).get("wer", float("nan"))
        halluc = ev.get("hallucination", {}).get("silence", {}).get("hallucination_rate", float("nan"))
        logger.info(f"{ev['variant']:<25} {clean:<12.4f} {multi:<15.4f} {halluc:<18.2%}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

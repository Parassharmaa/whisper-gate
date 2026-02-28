"""Silence Gate v4: Two-stage approach.

Stage 1: Train gate as pure speech/silence classifier on frozen encoder representations.
         Only BCE loss — no ASR loss interference.
Stage 2: Plug trained gate into Whisper inference pipeline. No further training.

The key insight: "Beyond Transcription" (2025) shows Whisper's encoder already
encodes speech vs non-speech with 100% linear separability. So a simple linear
probe at the bottleneck should trivially learn the gate.

Usage:
    uv run python scripts/run_silence_gate_v4.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from pulse_whisper.data.dataset import get_10h_subset_dataloader, get_dataloader
from pulse_whisper.data.gapped_audio import random_gap_augmentation
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination
from pulse_whisper.models.gated_whisper import GatedWhisper
from pulse_whisper.models.pulse_whisper import build_variant, Variant, get_processor
from pulse_whisper.models.silence_gate import SilenceGate

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


def downsample_mask(speech_mask: torch.Tensor, target_len: int) -> torch.Tensor:
    """Downsample speech mask from mel resolution to encoder output resolution."""
    mel_len = speech_mask.shape[1]
    if mel_len == target_len:
        return speech_mask
    downsampled = torch.nn.functional.adaptive_avg_pool1d(
        speech_mask.unsqueeze(1), target_len
    ).squeeze(1)
    return (downsampled > 0.5).float()


def stage1_train_classifier(
    whisper_model_name: str,
    device: torch.device,
    output_dir: Path,
    gate_hidden_dim: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 10,
    silence_fraction: float = 0.3,
) -> SilenceGate:
    """Stage 1: Train the gate as a pure binary classifier.

    Freezes the entire Whisper model, runs encoder to get representations,
    then trains a small MLP to predict speech(1) vs silence(0) per frame.
    Only uses BCE loss — no ASR loss at all.
    """
    logger.info("=" * 70)
    logger.info("STAGE 1: Training gate as speech/silence classifier")
    logger.info("=" * 70)

    from transformers import WhisperForConditionalGeneration
    whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    whisper = whisper.to(device)
    whisper.eval()
    for param in whisper.parameters():
        param.requires_grad = False

    d_model = whisper.config.d_model
    gate = SilenceGate(d_model=d_model, hidden_dim=gate_hidden_dim)
    gate = gate.to(device)

    trainable = sum(p.numel() for p in gate.parameters())
    logger.info(f"Gate trainable params: {trainable:,}")

    # Load training data
    train_loader = get_10h_subset_dataloader(
        whisper_size="tiny",
        batch_size=16,
        gap_augmentation=False,
    )
    logger.info(f"Training samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")

    optimizer = AdamW(gate.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    start_time = time.time()
    history = {"bce_loss": [], "accuracy": []}

    # Gap fractions for creating diverse silence patterns
    gap_fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    for epoch in range(max_epochs):
        gate.train()
        total_bce = 0.0
        total_correct = 0
        total_frames = 0
        num_batches = 0

        for batch in train_loader:
            input_features = batch["input_features"].to(device)
            batch_size = input_features.shape[0]

            # Apply gap augmentation to get speech/silence labels
            input_features, speech_mask = random_gap_augmentation(
                input_features,
                gap_fractions=gap_fractions,
                return_mask=True,
            )

            # Inject pure silence samples
            n_silence = 0
            max_per_batch = max(1, int(batch_size * silence_fraction))
            for b in range(batch_size):
                if n_silence >= max_per_batch:
                    break
                if torch.rand(1).item() < silence_fraction:
                    input_features[b] = -1.0
                    speech_mask[b] = 0.0
                    n_silence += 1

            # Get encoder representations (frozen)
            with torch.no_grad():
                encoder_out = whisper.model.encoder(input_features).last_hidden_state

            # Downsample mask to encoder output resolution
            enc_len = encoder_out.shape[1]
            speech_mask_ds = downsample_mask(speech_mask, enc_len).to(device)

            # Forward through gate
            _, gate_probs = gate(encoder_out.detach())

            # BCE loss only
            bce_loss = nn.functional.binary_cross_entropy(
                gate_probs.float(), speech_mask_ds.float()
            )

            optimizer.zero_grad()
            bce_loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Accuracy
            predictions = (gate_probs > 0.5).float()
            correct = (predictions == speech_mask_ds).sum().item()
            total_frames += speech_mask_ds.numel()
            total_correct += correct

            total_bce += bce_loss.item()
            num_batches += 1

            if num_batches % 20 == 0:
                avg_bce = total_bce / num_batches
                acc = total_correct / total_frames
                logger.info(f"  step {num_batches}: bce={avg_bce:.4f}, acc={acc:.4f}")

        epoch_bce = total_bce / max(1, num_batches)
        epoch_acc = total_correct / max(1, total_frames)
        history["bce_loss"].append(epoch_bce)
        history["accuracy"].append(epoch_acc)
        logger.info(f"Epoch {epoch + 1}/{max_epochs}: bce={epoch_bce:.4f}, accuracy={epoch_acc:.4f}")

    train_time = time.time() - start_time
    logger.info(f"Stage 1 completed in {train_time / 60:.1f} minutes")
    logger.info(f"Final accuracy: {history['accuracy'][-1]:.4f}")

    # Analyze gate behavior
    gate.eval()
    with torch.no_grad():
        # Pure silence
        silence_mel = torch.zeros(1, 80, 3000, device=device)
        enc = whisper.model.encoder(silence_mel).last_hidden_state
        _, probs = gate(enc)
        logger.info(f"Gate on silence: mean={probs.mean():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")

        # Speech
        for batch in train_loader:
            speech_mel = batch["input_features"][:1].to(device)
            enc = whisper.model.encoder(speech_mel).last_hidden_state
            _, probs = gate(enc)
            logger.info(f"Gate on speech:  mean={probs.mean():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")
            break

    # Save gate
    gate_path = output_dir / "gate_classifier.pt"
    torch.save({"gate_state_dict": gate.state_dict(), "history": history}, gate_path)
    logger.info(f"Gate saved to {gate_path}")

    del whisper
    return gate, history


@torch.no_grad()
def evaluate_hallucination_hard_gate(
    model: GatedWhisper,
    processor,
    device: torch.device,
    num_samples: int = 30,
    silence_threshold: float = 0.5,
) -> dict:
    """Evaluate hallucination with hard gating enabled."""
    from pulse_whisper.eval.metrics import (
        HallucinationResult,
        compute_hallucination_rate,
        compute_hallucination_severity,
    )

    model.eval()
    n_mels, seq_len = 80, 3000
    results = {}

    for input_type, gen_fn in [
        ("silence", lambda: torch.zeros(1, n_mels, seq_len) - 1.0),
        ("white_noise", lambda: torch.randn(1, n_mels, seq_len) * 0.1),
    ]:
        all_outputs = []
        for i in range(num_samples):
            input_features = gen_fn().to(device)
            generated_ids = model.generate(
                input_features,
                hard_gate=True,
                silence_threshold=silence_threshold,
                language="en", task="transcribe", max_new_tokens=440,
            )
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            all_outputs.append(text)

        rate = compute_hallucination_rate(all_outputs)
        severity = compute_hallucination_severity(all_outputs)
        results[input_type] = HallucinationResult(
            input_type=input_type,
            hallucination_rate=rate,
            avg_output_length=severity,
            num_samples=num_samples,
            outputs=all_outputs,
        )

    return results


def stage2_evaluate(
    gate: SilenceGate,
    device: torch.device,
    output_dir: Path,
    whisper_size: str = "tiny",
    gate_hidden_dim: int = 32,
) -> dict:
    """Stage 2: Plug the trained gate into Whisper and evaluate.

    No further training — just inference with gating.
    """
    logger.info("=" * 70)
    logger.info("STAGE 2: Evaluating with trained gate")
    logger.info("=" * 70)

    from pulse_whisper.models.gated_whisper import GatedWhisper
    model = GatedWhisper(
        whisper_model_name=f"openai/whisper-{whisper_size}",
        gate_hidden_dim=gate_hidden_dim,
        gate_loss_weight=0.0,  # No gate loss during eval
    )

    # Load trained gate weights
    model.silence_gate.load_state_dict(gate.state_dict())
    model = model.to(device)
    model.eval()

    processor = get_processor(whisper_size)
    test_loader = get_dataloader(
        split="test-clean",
        whisper_size=whisper_size,
        batch_size=8,
    )
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Gapped evaluation
    gap_results = evaluate_all_gap_levels(
        model=model, dataloader=test_loader,
        processor=processor, device=device,
    )

    logger.info(f"\n{'Gap Level':<15} {'WER':<10} {'CER':<10} {'Samples':<10}")
    logger.info("-" * 45)
    for level, result in gap_results.items():
        logger.info(f"{level:<15} {result.wer:<10.4f} {result.cer:<10.4f} {result.num_samples:<10}")

    # Hallucination test — soft gating
    halluc_results = evaluate_hallucination(
        model=model, processor=processor,
        device=device, num_samples=30,
    )
    for input_type, result in halluc_results.items():
        logger.info(f"Hallucination (soft) [{input_type}]: rate={result.hallucination_rate:.2%}, "
                     f"avg_length={result.avg_output_length:.1f}")

    # Hallucination test — hard gating (the key test)
    logger.info("\nHard gating hallucination test:")
    halluc_hard_results = evaluate_hallucination_hard_gate(
        model=model, processor=processor,
        device=device, num_samples=30,
    )
    for input_type, result in halluc_hard_results.items():
        logger.info(f"Hallucination (hard) [{input_type}]: rate={result.hallucination_rate:.2%}, "
                     f"avg_length={result.avg_output_length:.1f}")

    # Detailed gate analysis
    logger.info("\nGate behavior analysis:")
    with torch.no_grad():
        silence_mel = torch.zeros(1, 80, 3000, device=device)
        encoder_hidden = model._run_encoder(silence_mel)
        _, gate_probs = model.silence_gate(encoder_hidden)
        logger.info(f"  Silence: mean={gate_probs.mean():.4f}, min={gate_probs.min():.4f}, max={gate_probs.max():.4f}")

        for batch in test_loader:
            speech_mel = batch["input_features"][:1].to(device)
            encoder_hidden = model._run_encoder(speech_mel)
            _, gate_probs = model.silence_gate(encoder_hidden)
            logger.info(f"  Speech:  mean={gate_probs.mean():.4f}, min={gate_probs.min():.4f}, max={gate_probs.max():.4f}")

            # Also test gapped speech
            from pulse_whisper.data.gapped_audio import inject_silence_gaps
            gapped_mel, gap_mask = inject_silence_gaps(speech_mel, "gap_30", seed=0)
            encoder_hidden = model._run_encoder(gapped_mel)
            _, gate_probs = model.silence_gate(encoder_hidden)
            logger.info(f"  Gapped:  mean={gate_probs.mean():.4f}, min={gate_probs.min():.4f}, max={gate_probs.max():.4f}")
            break

    return {
        "gap_evaluation": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in gap_results.items()
        },
        "hallucination_soft": {
            input_type: {
                "hallucination_rate": r.hallucination_rate,
                "avg_output_length": r.avg_output_length,
                "num_samples": r.num_samples,
            }
            for input_type, r in halluc_results.items()
        },
        "hallucination_hard": {
            input_type: {
                "hallucination_rate": r.hallucination_rate,
                "avg_output_length": r.avg_output_length,
                "num_samples": r.num_samples,
            }
            for input_type, r in halluc_hard_results.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Silence Gate v4: Two-stage approach")
    parser.add_argument("--output-dir", default="results/silence_gate_v4")
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--silence-fraction", type=float, default=0.3)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    device = setup_device()
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "silence_gate_v4_two_stage",
        "config": {
            "gate_hidden_dim": args.gate_hidden_dim,
            "lr": args.lr,
            "max_epochs": args.max_epochs,
            "silence_fraction": args.silence_fraction,
            "approach": "two-stage: BCE classifier first, then inference with gating",
        },
        "device": str(device),
        "evaluations": [],
    }

    # Baseline
    if not args.skip_baseline:
        logger.info("\n" + "#" * 70)
        logger.info("# BASELINE EVALUATION")
        logger.info("#" * 70)

        baseline = build_variant(Variant.A, whisper_size="tiny")
        baseline = baseline.to(device)
        baseline.eval()

        processor = get_processor("tiny")
        test_loader = get_dataloader(split="test-clean", whisper_size="tiny", batch_size=8)

        gap_results = evaluate_all_gap_levels(
            model=baseline, dataloader=test_loader,
            processor=processor, device=device,
        )
        halluc_results = evaluate_hallucination(
            model=baseline, processor=processor, device=device, num_samples=30,
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
        del baseline

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Stage 1: Train classifier
    gate, train_history = stage1_train_classifier(
        whisper_model_name="openai/whisper-tiny",
        device=device,
        output_dir=output_dir,
        gate_hidden_dim=args.gate_hidden_dim,
        lr=args.lr,
        max_epochs=args.max_epochs,
        silence_fraction=args.silence_fraction,
    )
    results["training"] = {
        "bce_loss": train_history["bce_loss"],
        "accuracy": train_history["accuracy"],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Stage 2: Evaluate with gating
    eval_result = stage2_evaluate(
        gate=gate,
        device=device,
        output_dir=output_dir,
        whisper_size="tiny",
        gate_hidden_dim=args.gate_hidden_dim,
    )
    eval_result["variant"] = "silence_gate_v4"
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
        # Check both old and new hallucination key names
        halluc_dict = ev.get("hallucination_soft", ev.get("hallucination", {}))
        halluc = halluc_dict.get("silence", {}).get("hallucination_rate", float("nan"))
        halluc_hard_dict = ev.get("hallucination_hard", {})
        halluc_hard = halluc_hard_dict.get("silence", {}).get("hallucination_rate", float("nan"))
        hard_str = f"{halluc_hard:.2%}" if not (halluc_hard != halluc_hard) else "N/A"
        logger.info(f"{ev['variant']:<25} {clean:<12.4f} {multi:<15.4f} {halluc:<12.2%}  {hard_str}")
    logger.info("(last column = hard gate hallucination rate)")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

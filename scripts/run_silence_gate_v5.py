"""Silence Gate v5: Two-stage with temporal smoothing.

Same two-stage approach as v4 but with TemporalSilenceGate — a 1D conv
over gate logits before sigmoid for temporally coherent gate decisions.

Tests multiple kernel sizes to find optimal smoothing window.

Usage:
    uv run python scripts/run_silence_gate_v5.py
    uv run python scripts/run_silence_gate_v5.py --kernel-sizes 3 5 7 11
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
from pulse_whisper.data.gapped_audio import random_gap_augmentation, inject_silence_gaps
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination
from pulse_whisper.eval.metrics import (
    HallucinationResult,
    compute_hallucination_rate,
    compute_hallucination_severity,
)
from pulse_whisper.models.gated_whisper import GatedWhisper
from pulse_whisper.models.pulse_whisper import get_processor
from pulse_whisper.models.silence_gate import TemporalSilenceGate

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
    mel_len = speech_mask.shape[1]
    if mel_len == target_len:
        return speech_mask
    downsampled = nn.functional.adaptive_avg_pool1d(
        speech_mask.unsqueeze(1), target_len
    ).squeeze(1)
    return (downsampled > 0.5).float()


def train_temporal_gate(
    whisper_model_name: str,
    device: torch.device,
    output_dir: Path,
    gate_hidden_dim: int = 32,
    kernel_size: int = 5,
    lr: float = 1e-3,
    max_epochs: int = 10,
    silence_fraction: float = 0.3,
) -> tuple[TemporalSilenceGate, dict]:
    """Train TemporalSilenceGate as pure BCE classifier."""
    logger.info(f"\nTraining TemporalSilenceGate (kernel_size={kernel_size})")
    logger.info("-" * 50)

    from transformers import WhisperForConditionalGeneration
    whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    whisper = whisper.to(device)
    whisper.eval()
    for param in whisper.parameters():
        param.requires_grad = False

    d_model = whisper.config.d_model
    gate = TemporalSilenceGate(
        d_model=d_model,
        hidden_dim=gate_hidden_dim,
        kernel_size=kernel_size,
    )
    gate = gate.to(device)

    trainable = sum(p.numel() for p in gate.parameters())
    logger.info(f"Gate trainable params: {trainable:,}")

    train_loader = get_10h_subset_dataloader(
        whisper_size="tiny", batch_size=16, gap_augmentation=False,
    )

    optimizer = AdamW(gate.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    gap_fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    start_time = time.time()
    history = {"bce_loss": [], "accuracy": []}

    for epoch in range(max_epochs):
        gate.train()
        total_bce = 0.0
        total_correct = 0
        total_frames = 0
        num_batches = 0

        for batch in train_loader:
            input_features = batch["input_features"].to(device)
            batch_size = input_features.shape[0]

            input_features, speech_mask = random_gap_augmentation(
                input_features, gap_fractions=gap_fractions, return_mask=True,
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

            with torch.no_grad():
                encoder_out = whisper.model.encoder(input_features).last_hidden_state

            enc_len = encoder_out.shape[1]
            speech_mask_ds = downsample_mask(speech_mask, enc_len).to(device)

            _, gate_probs = gate(encoder_out.detach())

            bce_loss = nn.functional.binary_cross_entropy(
                gate_probs.float(), speech_mask_ds.float()
            )

            optimizer.zero_grad()
            bce_loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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
    logger.info(f"Training completed in {train_time / 60:.1f} minutes")
    logger.info(f"Final accuracy: {history['accuracy'][-1]:.4f}")

    # Analyze gate behavior
    gate.eval()
    with torch.no_grad():
        silence_mel = torch.zeros(1, 80, 3000, device=device)
        enc = whisper.model.encoder(silence_mel).last_hidden_state
        _, probs = gate(enc)
        logger.info(f"Gate on silence: mean={probs.mean():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")

        for batch in train_loader:
            speech_mel = batch["input_features"][:1].to(device)
            enc = whisper.model.encoder(speech_mel).last_hidden_state
            _, probs = gate(enc)
            logger.info(f"Gate on speech:  mean={probs.mean():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")

            # Gapped speech — check boundary behavior
            gapped_mel, _ = inject_silence_gaps(speech_mel, "gap_30", seed=0)
            enc = whisper.model.encoder(gapped_mel).last_hidden_state
            _, probs = gate(enc)
            logger.info(f"Gate on gapped:  mean={probs.mean():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")
            break

    gate_path = output_dir / f"gate_temporal_k{kernel_size}.pt"
    torch.save({"gate_state_dict": gate.state_dict(), "history": history, "kernel_size": kernel_size}, gate_path)
    logger.info(f"Gate saved to {gate_path}")

    del whisper
    return gate, history


class HardGateWrapper:
    """Wraps GatedWhisper to force hard_gate=True during generate()."""

    def __init__(self, model: GatedWhisper, silence_threshold: float = 0.5):
        self.model = model
        self.silence_threshold = silence_threshold

    def generate(self, input_features, **kwargs):
        return self.model.generate(
            input_features, hard_gate=True,
            silence_threshold=self.silence_threshold, **kwargs,
        )

    def eval(self):
        self.model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.model, name)


def evaluate_variant(
    gate: TemporalSilenceGate,
    kernel_size: int,
    device: torch.device,
    processor,
    test_loader,
) -> dict:
    """Evaluate a trained temporal gate with both soft and hard gating."""
    logger.info(f"\nEvaluating kernel_size={kernel_size}")
    logger.info("=" * 50)

    model = GatedWhisper(
        whisper_model_name="openai/whisper-tiny",
        gate_hidden_dim=32,
        temporal_smoothing=True,
        temporal_kernel_size=kernel_size,
    )
    model.silence_gate.load_state_dict(gate.state_dict())
    model = model.to(device)
    model.eval()

    # Soft gate WER
    logger.info("Soft gate WER:")
    soft_gap_results = evaluate_all_gap_levels(
        model=model, dataloader=test_loader,
        processor=processor, device=device,
    )
    for level, r in soft_gap_results.items():
        logger.info(f"  {level:<15} WER={r.wer:.4f}")

    # Hard gate WER
    logger.info("Hard gate WER:")
    hard_model = HardGateWrapper(model)
    hard_gap_results = evaluate_all_gap_levels(
        model=hard_model, dataloader=test_loader,
        processor=processor, device=device,
    )
    for level, r in hard_gap_results.items():
        logger.info(f"  {level:<15} WER={r.wer:.4f}")

    # Hallucination — hard gate
    logger.info("Hard gate hallucination:")
    n_mels, seq_len = 80, 3000
    halluc_results = {}
    for input_type, gen_fn in [
        ("silence", lambda: torch.zeros(1, n_mels, seq_len) - 1.0),
        ("white_noise", lambda: torch.randn(1, n_mels, seq_len) * 0.1),
    ]:
        all_outputs = []
        for i in range(30):
            input_features = gen_fn().to(device)
            generated_ids = model.generate(
                input_features, hard_gate=True, silence_threshold=0.5,
                language="en", task="transcribe", max_new_tokens=440,
            )
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            all_outputs.append(text)

        rate = compute_hallucination_rate(all_outputs)
        severity = compute_hallucination_severity(all_outputs)
        halluc_results[input_type] = {
            "hallucination_rate": rate,
            "avg_output_length": severity,
            "num_samples": 30,
        }
        logger.info(f"  {input_type}: halluc_rate={rate:.2%}")

    return {
        "kernel_size": kernel_size,
        "trainable_params": sum(p.numel() for p in gate.parameters()),
        "soft_gate": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in soft_gap_results.items()
        },
        "hard_gate": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in hard_gap_results.items()
        },
        "hallucination_hard": halluc_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Silence Gate v5: Temporal smoothing")
    parser.add_argument("--output-dir", default="results/silence_gate_v5")
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[5, 11])
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--silence-fraction", type=float, default=0.3)
    args = parser.parse_args()

    device = setup_device()
    logger.info(f"Device: {device}")
    logger.info(f"Kernel sizes to test: {args.kernel_sizes}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "silence_gate_v5_temporal_smoothing",
        "config": {
            "gate_hidden_dim": args.gate_hidden_dim,
            "lr": args.lr,
            "max_epochs": args.max_epochs,
            "silence_fraction": args.silence_fraction,
            "kernel_sizes": args.kernel_sizes,
        },
        "device": str(device),
        "variants": [],
    }

    processor = get_processor("tiny")
    test_loader = get_dataloader(split="test-clean", whisper_size="tiny", batch_size=8)

    for kernel_size in args.kernel_sizes:
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# KERNEL SIZE = {kernel_size}")
        logger.info(f"{'#' * 70}")

        gate, history = train_temporal_gate(
            whisper_model_name="openai/whisper-tiny",
            device=device,
            output_dir=output_dir,
            gate_hidden_dim=args.gate_hidden_dim,
            kernel_size=kernel_size,
            lr=args.lr,
            max_epochs=args.max_epochs,
            silence_fraction=args.silence_fraction,
        )

        eval_result = evaluate_variant(
            gate=gate, kernel_size=kernel_size,
            device=device, processor=processor, test_loader=test_loader,
        )
        eval_result["training"] = history
        results["variants"].append(eval_result)

        # Save after each variant
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        del gate

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Temporal Smoothing Results")
    logger.info("=" * 70)

    # Compare with v4 baseline
    logger.info(f"\n{'Variant':<20} {'Params':<10} {'Clean(S)':<10} {'Clean(H)':<10} {'Multi(S)':<10} {'Multi(H)':<10} {'gap30(H)':<10} {'Halluc':<10}")
    logger.info("-" * 90)

    # v4 reference
    logger.info(f"{'v4 (no smooth)':<20} {'12,353':<10} {'0.0821':<10} {'0.0824':<10} {'0.2756':<10} {'0.2981':<10} {'0.3449':<10} {'0%':<10}")

    for v in results["variants"]:
        ks = v["kernel_size"]
        params = v["trainable_params"]
        s_clean = v["soft_gate"].get("gap_0", {}).get("wer", float("nan"))
        h_clean = v["hard_gate"].get("gap_0", {}).get("wer", float("nan"))
        s_multi = v["soft_gate"].get("multi_gap", {}).get("wer", float("nan"))
        h_multi = v["hard_gate"].get("multi_gap", {}).get("wer", float("nan"))
        h_gap30 = v["hard_gate"].get("gap_30", {}).get("wer", float("nan"))
        halluc = v["hallucination_hard"].get("silence", {}).get("hallucination_rate", float("nan"))
        logger.info(f"{'k=' + str(ks):<20} {params:<10,} {s_clean:<10.4f} {h_clean:<10.4f} {s_multi:<10.4f} {h_multi:<10.4f} {h_gap30:<10.4f} {halluc:<10.0%}")

    logger.info(f"\nAll results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

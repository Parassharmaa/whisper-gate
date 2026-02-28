"""Test silence gate on whisper-small to validate generality.

Whisper-small has d_model=768 (vs 384 for tiny), 12 encoder layers (vs 4).
The gate becomes 768→32→1 (~25K params).

Usage:
    uv run python scripts/eval_whisper_small.py
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
    compute_hallucination_rate,
    compute_hallucination_severity,
)
from pulse_whisper.models.gated_whisper import GatedWhisper
from pulse_whisper.models.silence_gate import SilenceGate
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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


def train_gate(
    whisper_model_name: str,
    device: torch.device,
    output_dir: Path,
    gate_hidden_dim: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 10,
    silence_fraction: float = 0.3,
    whisper_size: str = "small",
) -> tuple[SilenceGate, dict]:
    """Train gate classifier on whisper-small encoder."""
    logger.info("=" * 70)
    logger.info(f"Training gate on {whisper_model_name}")
    logger.info("=" * 70)

    from transformers import WhisperForConditionalGeneration
    whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    whisper = whisper.to(device)
    whisper.eval()
    for param in whisper.parameters():
        param.requires_grad = False

    d_model = whisper.config.d_model
    logger.info(f"d_model={d_model}")

    gate = SilenceGate(d_model=d_model, hidden_dim=gate_hidden_dim)
    gate = gate.to(device)

    trainable = sum(p.numel() for p in gate.parameters())
    logger.info(f"Gate trainable params: {trainable:,}")

    # GPU: use larger batch; CPU/MPS: smaller
    train_batch = 32 if device.type == "cuda" else 8
    train_loader = get_10h_subset_dataloader(
        whisper_size=whisper_size, batch_size=train_batch,
        gap_augmentation=False,
    )
    logger.info(f"Training batches: {len(train_loader)}")

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

            n_silence = 0
            max_per_batch = max(1, int(batch_size * silence_fraction))
            for b in range(batch_size):
                if n_silence >= max_per_batch:
                    break
                if torch.rand(1).item() < silence_fraction:
                    input_features[b] = -1.0
                    speech_mask[b] = 0.0
                    n_silence += 1

            with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
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
                logger.info(f"  step {num_batches}: bce={total_bce / num_batches:.4f}, acc={total_correct / total_frames:.4f}")

        epoch_bce = total_bce / max(1, num_batches)
        epoch_acc = total_correct / max(1, total_frames)
        history["bce_loss"].append(epoch_bce)
        history["accuracy"].append(epoch_acc)
        logger.info(f"Epoch {epoch + 1}/{max_epochs}: bce={epoch_bce:.4f}, accuracy={epoch_acc:.4f}")

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time / 60:.1f} minutes")

    # Analyze
    gate.eval()
    with torch.no_grad():
        silence_mel = torch.zeros(1, 80, 3000, device=device)
        enc = whisper.model.encoder(silence_mel).last_hidden_state
        _, probs = gate(enc)
        logger.info(f"Gate on silence: mean={probs.mean():.4f}")

        for batch in train_loader:
            speech_mel = batch["input_features"][:1].to(device)
            enc = whisper.model.encoder(speech_mel).last_hidden_state
            _, probs = gate(enc)
            logger.info(f"Gate on speech:  mean={probs.mean():.4f}")
            break

    gate_path = output_dir / "gate_small.pt"
    torch.save({"gate_state_dict": gate.state_dict(), "history": history}, gate_path)
    logger.info(f"Gate saved to {gate_path}")

    del whisper
    return gate, history


def main():
    device = setup_device()
    logger.info(f"Device: {device}")

    whisper_size = "small"
    whisper_model_name = f"openai/whisper-{whisper_size}"

    output_dir = Path("results/silence_gate_small")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "silence_gate_whisper_small",
        "whisper_size": whisper_size,
        "device": str(device),
    }

    # Train gate
    gate, history = train_gate(
        whisper_model_name=whisper_model_name,
        device=device,
        output_dir=output_dir,
        gate_hidden_dim=32,
        max_epochs=10,
        whisper_size=whisper_size,
    )
    results["training"] = history

    # Build GatedWhisper for evaluation
    model = GatedWhisper(
        whisper_model_name=whisper_model_name,
        gate_hidden_dim=32,
    )
    model.silence_gate.load_state_dict(gate.state_dict())
    model = model.to(device)
    model.eval()

    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    eval_batch = 16 if device.type == "cuda" else 4
    test_loader = get_dataloader(split="test-clean", whisper_size=whisper_size, batch_size=eval_batch)

    # Soft gate WER
    logger.info("\nSoft gate WER:")
    soft_results = evaluate_all_gap_levels(
        model=model, dataloader=test_loader,
        processor=processor, device=device,
    )
    for level, r in soft_results.items():
        logger.info(f"  {level:<15} WER={r.wer:.4f}")

    # Hard gate WER
    logger.info("\nHard gate WER:")

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

    hard_model = HardGateWrap(model)
    hard_results = evaluate_all_gap_levels(
        model=hard_model, dataloader=test_loader,
        processor=processor, device=device,
    )
    for level, r in hard_results.items():
        logger.info(f"  {level:<15} WER={r.wer:.4f}")

    # Hallucination — hard gate
    logger.info("\nHard gate hallucination:")
    halluc_results = {}
    n_mels, seq_len = 80, 3000
    for input_type, gen_fn in [
        ("silence", lambda: torch.zeros(1, n_mels, seq_len) - 1.0),
        ("white_noise", lambda: torch.randn(1, n_mels, seq_len) * 0.1),
    ]:
        all_outputs = []
        for i in range(30):
            feats = gen_fn().to(device)
            ids = model.generate(
                feats, hard_gate=True, silence_threshold=0.5,
                language="en", task="transcribe", max_new_tokens=440,
            )
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            all_outputs.append(text)

        rate = compute_hallucination_rate(all_outputs)
        halluc_results[input_type] = {"hallucination_rate": rate, "num_samples": 30}
        logger.info(f"  {input_type}: {rate:.2%}")

    # Also test vanilla whisper-small baseline
    logger.info("\nVanilla whisper-small baseline hallucination:")
    vanilla = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)
    vanilla.eval()
    vanilla_halluc = {}
    for input_type, gen_fn in [
        ("silence", lambda: torch.zeros(1, n_mels, seq_len) - 1.0),
        ("white_noise", lambda: torch.randn(1, n_mels, seq_len) * 0.1),
    ]:
        all_outputs = []
        for i in range(30):
            feats = gen_fn().to(device)
            ids = vanilla.generate(feats, language="en", task="transcribe", max_new_tokens=440)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            all_outputs.append(text)

        rate = compute_hallucination_rate(all_outputs)
        vanilla_halluc[input_type] = {"hallucination_rate": rate, "num_samples": 30}
        logger.info(f"  {input_type}: {rate:.2%}")
    del vanilla

    results["soft_gate"] = {
        level: {"wer": r.wer, "cer": r.cer} for level, r in soft_results.items()
    }
    results["hard_gate"] = {
        level: {"wer": r.wer, "cer": r.cer} for level, r in hard_results.items()
    }
    results["hallucination_hard_gate"] = halluc_results
    results["hallucination_vanilla"] = vanilla_halluc

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("WHISPER-SMALL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Gate params: {sum(p.numel() for p in gate.parameters()):,}")
    logger.info(f"Final accuracy: {history['accuracy'][-1]:.4f}")
    logger.info(f"\n{'Metric':<20} {'Soft Gate':<12} {'Hard Gate':<12}")
    logger.info("-" * 44)
    for level in ["gap_0", "gap_5", "gap_15", "gap_30", "multi_gap"]:
        s = soft_results[level].wer
        h = hard_results[level].wer
        logger.info(f"{level:<20} {s:<12.4f} {h:<12.4f}")

    logger.info(f"\nVanilla halluc: silence={vanilla_halluc['silence']['hallucination_rate']:.0%}, noise={vanilla_halluc['white_noise']['hallucination_rate']:.0%}")
    logger.info(f"Hard gate halluc: silence={halluc_results['silence']['hallucination_rate']:.0%}, noise={halluc_results['white_noise']['hallucination_rate']:.0%}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

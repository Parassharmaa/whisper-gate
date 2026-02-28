"""Training loop for pulse-injected Whisper.

Trains only pulse parameters while keeping Whisper frozen.
Supports AMP, gradient clipping, cosine LR with warmup, and checkpointing.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pulse_whisper.training.config import ExperimentConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for PulseWhisperEncoder."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: ExperimentConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or ExperimentConfig()
        self.device = torch.device(device)

        tc = self.config.training

        # Only optimize pulse parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

        self.optimizer = AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)

        # Cosine annealing with warmup
        total_steps = tc.max_epochs * len(train_loader)
        self.warmup_steps = tc.warmup_steps
        self.total_steps = total_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps - tc.warmup_steps)
        )

        # AMP
        self.use_amp = tc.fp16 and self.device.type in ("cuda", "mps")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and self.device.type == "cuda")

        self.global_step = 0
        self.best_val_loss = float("inf")

    def train(self) -> dict:
        """Full training loop. Returns history dict."""
        tc = self.config.training
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(tc.max_epochs):
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)
            logger.info(f"Epoch {epoch + 1}/{tc.max_epochs}: train_loss={train_loss:.4f}")

            if self.val_loader is not None:
                val_loss = self._eval_epoch()
                history["val_loss"].append(val_loss)
                logger.info(f"  val_loss={val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.config.logging.save_checkpoints:
                        self._save_checkpoint("best.pt")

            if self.config.logging.save_checkpoints:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        tc = self.config.training

        for batch in self.train_loader:
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Gap augmentation
            if tc.gap_augmentation:
                from pulse_whisper.data.gapped_audio import random_gap_augmentation
                input_features = random_gap_augmentation(
                    input_features, tc.gap_fractions
                )

            # Forward
            amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
            with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.use_amp):
                outputs = self.model(input_features=input_features, labels=labels)
                loss = outputs["loss"]

            # Backward
            self.optimizer.zero_grad()
            if self.use_amp and self.device.type == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip_norm)
                self.optimizer.step()

            # LR schedule with warmup
            self.global_step += 1
            if self.global_step <= self.warmup_steps:
                warmup_factor = self.global_step / max(1, self.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = tc.lr * warmup_factor
            else:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % self.config.logging.log_every_n_steps == 0:
                avg = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"  step {self.global_step}: loss={avg:.4f}, lr={lr:.2e}")

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def _eval_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_features=input_features, labels=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _save_checkpoint(self, filename: str) -> None:
        ckpt_dir = Path(self.config.logging.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / filename
        torch.save({
            "model_state_dict": {
                k: v for k, v in self.model.state_dict().items()
                if any(k.startswith(prefix) for prefix in ("injected_layers", "decoder_pulse", "silence_gate"))
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

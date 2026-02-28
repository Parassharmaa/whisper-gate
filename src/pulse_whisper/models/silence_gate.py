"""Silence-Aware Bottleneck Gate.

Lightweight MLP that sits between encoder and decoder, producing
frame-wise speech probability p_speech(t). Gates encoder hidden states
as h_gated = h_enc * p_speech(t) before cross-attention, suppressing
non-speech frames to prevent decoder hallucination.

Motivated by "Beyond Transcription" (2025) showing encoder representations
are linearly separable for speech vs non-speech with 100% accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SilenceGate(nn.Module):
    """Frame-wise speech/silence gate on encoder hidden states.

    Takes encoder output (batch, seq_len, d_model) and produces
    per-frame speech probability (batch, seq_len, 1), then gates
    the encoder output multiplicatively.

    Args:
        d_model: Encoder hidden dimension.
        hidden_dim: Gate MLP hidden dimension.
    """

    def __init__(self, d_model: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to pass-through (bias toward speech=1.0)
        # Set the final bias to +2.0 so sigmoid(2.0) ≈ 0.88, starting near pass-through
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.constant_(self.gate_mlp[2].bias, 2.0)

    def forward(
        self,
        encoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate encoder hidden states.

        Args:
            encoder_hidden: (batch, seq_len, d_model) from frozen encoder.

        Returns:
            Tuple of (gated_hidden, gate_probs):
                gated_hidden: (batch, seq_len, d_model) — gated encoder output.
                gate_probs: (batch, seq_len) — speech probability per frame.
        """
        # (batch, seq_len, 1)
        gate_logits = self.gate_mlp(encoder_hidden)
        gate_probs = torch.sigmoid(gate_logits)  # (batch, seq_len, 1)

        # Multiplicative gating
        gated_hidden = encoder_hidden * gate_probs

        return gated_hidden, gate_probs.squeeze(-1)  # probs: (batch, seq_len)


class TemporalSilenceGate(nn.Module):
    """Silence gate with temporal smoothing via 1D convolution.

    Adds a causal 1D convolution over gate logits before sigmoid,
    making gate decisions temporally coherent. A frame surrounded by
    silence frames is more likely classified as silence, and vice versa.

    This addresses boundary artifacts where independent per-frame
    decisions cause rapid speech/silence toggling at gap edges.

    Args:
        d_model: Encoder hidden dimension.
        hidden_dim: Gate MLP hidden dimension.
        kernel_size: Temporal smoothing window (odd number). Larger = smoother
            boundaries but less temporal precision. 5 ≈ 100ms at Whisper's
            20ms frame rate.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 32,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize MLP to pass-through
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.constant_(self.gate_mlp[2].bias, 2.0)

        # 1D conv for temporal smoothing over logits
        # Depthwise conv on 1 channel: smooths without mixing features
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        self.temporal_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # same-length output
            bias=True,
        )
        # Initialize conv to averaging (identity-like smoothing)
        nn.init.constant_(self.temporal_conv.weight, 1.0 / kernel_size)
        nn.init.zeros_(self.temporal_conv.bias)

        self.kernel_size = kernel_size

    def forward(
        self,
        encoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate encoder hidden states with temporal smoothing.

        Args:
            encoder_hidden: (batch, seq_len, d_model) from frozen encoder.

        Returns:
            Tuple of (gated_hidden, gate_probs):
                gated_hidden: (batch, seq_len, d_model) — gated encoder output.
                gate_probs: (batch, seq_len) — temporally smoothed speech probability.
        """
        # Per-frame logits: (batch, seq_len, 1)
        gate_logits = self.gate_mlp(encoder_hidden)

        # Temporal smoothing: conv over time dimension
        # Reshape: (batch, seq_len, 1) → (batch, 1, seq_len)
        logits_1d = gate_logits.transpose(1, 2)
        smoothed = self.temporal_conv(logits_1d)  # (batch, 1, seq_len)
        smoothed = smoothed.transpose(1, 2)  # (batch, seq_len, 1)

        gate_probs = torch.sigmoid(smoothed)  # (batch, seq_len, 1)

        # Multiplicative gating
        gated_hidden = encoder_hidden * gate_probs

        return gated_hidden, gate_probs.squeeze(-1)  # probs: (batch, seq_len)

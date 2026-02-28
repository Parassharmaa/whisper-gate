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

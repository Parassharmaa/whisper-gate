"""PulseModule: Oscillatory pulse injection for Transformer hidden states.

Ported from PDNA's PulseModule, adapted for Whisper encoder layers.
Core equation: h' = h + α · A · sin(ω · t + φ(h))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PulseLayer(nn.Module):
    """Learnable oscillatory pulse applied to Transformer hidden states.

    pulse(t, h) = α · A · sin(ω · t + φ(h))

    Parameters:
        A (amplitude): learned per-dimension, Gaussian init
        ω (omega): learned frequencies, log-uniform init [0.1, 10.0]
        φ(h) (phase_net): optional state-dependent phase shift
        α (alpha): mixing coefficient, initialized small
    """

    def __init__(
        self,
        hidden_size: int,
        n_frequencies: int | None = None,
        alpha_init: float = 0.01,
        use_phase_net: bool = True,
        alpha_max: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_frequencies = n_frequencies or hidden_size
        self.use_phase_net = use_phase_net
        self.alpha_max = alpha_max

        self.amplitude = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.omega = nn.Parameter(
            torch.exp(torch.linspace(math.log(0.1), math.log(10.0), hidden_size))
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        if use_phase_net:
            self.phase_net = nn.Linear(hidden_size, hidden_size)
        else:
            self.register_parameter("phase_net", None)

    def forward(self, h: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Compute pulse signal and add to hidden states.

        Args:
            h: Hidden states (batch, seq_len, hidden_size).
            time_steps: Time values (seq_len,).

        Returns:
            Modified hidden states: h + α · A · sin(ω · t + φ(h)).
        """
        t = time_steps.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

        if self.use_phase_net:
            phi = self.phase_net(h)  # (batch, seq_len, hidden_size)
            oscillation = self.amplitude * torch.sin(self.omega * t + phi)
        else:
            oscillation = self.amplitude * torch.sin(self.omega * t)

        alpha = self.alpha
        if self.alpha_max is not None:
            alpha = alpha.clamp(max=self.alpha_max)

        return h + alpha * oscillation

    def get_pulse_signal(self, h: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Return just the pulse signal (without adding to h), for analysis."""
        t = time_steps.unsqueeze(0).unsqueeze(-1)
        if self.use_phase_net:
            phi = self.phase_net(h)
            oscillation = self.amplitude * torch.sin(self.omega * t + phi)
        else:
            oscillation = self.amplitude * torch.sin(self.omega * t)
        return self.alpha * oscillation


class NoiseLayer(nn.Module):
    """Random perturbation control — matched magnitude to PulseLayer.

    Adds learned-scale random noise during training only (Variant B control).
    """

    def __init__(self, hidden_size: int, noise_scale_init: float = 0.01) -> None:
        super().__init__()
        self.noise_scale = nn.Parameter(torch.tensor(noise_scale_init))

    def forward(self, h: torch.Tensor, time_steps: torch.Tensor | None = None) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(h) * self.noise_scale
            return h + noise
        return h

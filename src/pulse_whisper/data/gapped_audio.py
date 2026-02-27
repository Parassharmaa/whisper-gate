"""Gap injection for mel spectrograms.

Creates silence gaps of varying sizes in mel spectrogram inputs
to test robustness of Whisper models to interrupted audio.
"""

from __future__ import annotations

from enum import Enum

import torch


class GapLevel(str, Enum):
    """Gap difficulty levels for gapped evaluation."""
    NONE = "gap_0"        # 0% gap (standard)
    SMALL = "gap_5"       # 5% gap
    MEDIUM = "gap_15"     # 15% gap
    LARGE = "gap_30"      # 30% gap
    MULTI = "multi_gap"   # Multiple scattered gaps (~20%)


GAP_FRACTIONS = {
    GapLevel.NONE: 0.0,
    GapLevel.SMALL: 0.05,
    GapLevel.MEDIUM: 0.15,
    GapLevel.LARGE: 0.30,
    GapLevel.MULTI: 0.20,
}


def create_gap_mask(
    seq_len: int,
    gap_level: GapLevel | str,
    batch_size: int = 1,
    seed: int | None = None,
) -> torch.Tensor:
    """Create a boolean gap mask for a mel spectrogram sequence.

    Args:
        seq_len: Number of time frames in the spectrogram.
        gap_level: Gap difficulty level.
        batch_size: Number of sequences in batch.
        seed: Random seed for reproducibility.

    Returns:
        Boolean mask (batch_size, seq_len) where True = gap (silence).
    """
    if isinstance(gap_level, str):
        gap_level = GapLevel(gap_level)

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    if gap_level == GapLevel.NONE:
        return mask

    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    if gap_level in (GapLevel.SMALL, GapLevel.MEDIUM, GapLevel.LARGE):
        gap_frac = GAP_FRACTIONS[gap_level]
        gap_len = max(1, int(seq_len * gap_frac))
        start = (seq_len - gap_len) // 2
        mask[:, start:start + gap_len] = True

    elif gap_level == GapLevel.MULTI:
        n_gaps = 4
        total_gap = int(seq_len * 0.20)
        gap_size = total_gap // n_gaps
        segment_len = seq_len // (n_gaps + 1)
        for i in range(n_gaps):
            start = segment_len * (i + 1) - gap_size // 2
            start = max(0, min(start, seq_len - gap_size))
            mask[:, start:start + gap_size] = True

    return mask


def inject_silence_gaps(
    mel_spectrogram: torch.Tensor,
    gap_level: GapLevel | str,
    seed: int | None = None,
    silence_value: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject silence gaps into a mel spectrogram.

    Args:
        mel_spectrogram: Input mel (batch, n_mels, seq_len) or (n_mels, seq_len).
        gap_level: Gap difficulty level.
        seed: Random seed for reproducibility.
        silence_value: Value to use for silence (log-mel silence ≈ -1.0 for Whisper).

    Returns:
        Tuple of (gapped_mel, gap_mask).
        gapped_mel has same shape as input.
        gap_mask is (batch, seq_len) boolean.
    """
    if isinstance(gap_level, str):
        gap_level = GapLevel(gap_level)

    squeezed = False
    if mel_spectrogram.dim() == 2:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
        squeezed = True

    batch_size, n_mels, seq_len = mel_spectrogram.shape
    gap_mask = create_gap_mask(seq_len, gap_level, batch_size, seed)

    gapped = mel_spectrogram.clone()
    # Expand mask to (batch, 1, seq_len) for broadcasting across mel bins
    expanded_mask = gap_mask.unsqueeze(1).expand_as(gapped)
    gapped[expanded_mask] = silence_value

    if squeezed:
        gapped = gapped.squeeze(0)

    return gapped, gap_mask


def random_gap_augmentation(
    mel_spectrogram: torch.Tensor,
    gap_fractions: list[float] = (0.0, 0.05, 0.15),
    silence_value: float = -1.0,
) -> torch.Tensor:
    """Randomly apply gap augmentation during training.

    Randomly selects a gap fraction and applies it.
    Used as a training-time augmentation.

    Args:
        mel_spectrogram: Input mel (batch, n_mels, seq_len).
        gap_fractions: List of possible gap fractions to sample from.
        silence_value: Value for silence regions.

    Returns:
        Augmented mel spectrogram.
    """
    idx = torch.randint(len(gap_fractions), (1,)).item()
    frac = gap_fractions[idx]

    if frac == 0.0:
        return mel_spectrogram

    batch_size, n_mels, seq_len = mel_spectrogram.shape
    gap_len = max(1, int(seq_len * frac))

    # Random start position per sample
    gapped = mel_spectrogram.clone()
    for b in range(batch_size):
        start = torch.randint(0, max(1, seq_len - gap_len), (1,)).item()
        gapped[b, :, start:start + gap_len] = silence_value

    return gapped

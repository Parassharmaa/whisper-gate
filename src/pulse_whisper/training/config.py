"""Experiment configuration: YAML -> dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    whisper_size: str = "tiny"
    freeze_whisper: bool = True
    n_frequencies: int = 64
    alpha_init: float = 0.01
    alpha_max: float | None = None  # clamp alpha to this max value (None = unconstrained)
    pulse_layers: str = "all"  # "all" or comma-separated layer indices
    decoder_pulse: bool = False  # use decoder head pulse injection instead of encoder pulse
    use_phase_net: bool = False  # state-dependent phase in pulse modules
    gate_hidden_dim: int = 32  # silence gate MLP hidden dim
    gate_loss_weight: float = 1.0  # weight for auxiliary gate BCE loss
    silence_injection_rate: float = 0.1  # fraction of training samples replaced with pure silence


@dataclass
class TrainingConfig:
    dataset: str = "librispeech-10h"
    gap_augmentation: bool = True
    gap_fractions: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.15])
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    fp16: bool = True
    seed: int = 42
    gradient_checkpointing: bool = False


@dataclass
class EvalConfig:
    gap_levels: list[str] = field(default_factory=lambda: ["gap_0", "gap_5", "gap_15", "gap_30", "multi_gap"])
    test_set: str = "test-clean"
    hallucination_test: bool = True
    max_eval_samples: int | None = None


@dataclass
class LoggingConfig:
    backend: str = "tensorboard"
    project: str = "pulse-whisper"
    log_dir: str = "runs"
    log_every_n_steps: int = 50
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load experiment config from YAML, overlaying onto dataclass defaults."""
    config = ExperimentConfig()
    if path is None:
        return config

    path = Path(path)
    if not path.exists():
        return config

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    section_map = {
        "model": config.model,
        "training": config.training,
        "eval": config.eval,
        "logging": config.logging,
    }

    for section_name, section_obj in section_map.items():
        if section_name in raw:
            for k, v in raw[section_name].items():
                if hasattr(section_obj, k):
                    setattr(section_obj, k, v)

    return config

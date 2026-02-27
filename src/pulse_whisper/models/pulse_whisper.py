"""PulseWhisperEncoder: Whisper encoder with injected pulse layers.

Wraps HuggingFace Whisper encoder, injecting PulseLayer or NoiseLayer
after each encoder layer. Supports variants A-E per the experiment plan.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class Variant(str, Enum):
    """Experiment variants A-E."""
    A = "baseline"       # Frozen Whisper (control)
    B = "noise"          # + random perturbation control
    C = "pulse"          # + structured oscillation, fixed phase
    D = "pulse_phase"    # + structured oscillation with φ(h)
    E = "adaptive_pe"    # + learned oscillatory positional encoding


class PulseWhisperEncoder(nn.Module):
    """Whisper encoder with pulse injection after each encoder layer.

    Freezes all Whisper parameters and only trains the injected pulse/noise layers.
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-tiny",
        variant: Variant | str = Variant.C,
        n_frequencies: int = 64,
        alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        if isinstance(variant, str):
            variant = Variant(variant)
        self.variant = variant
        self.whisper_model_name = whisper_model_name

        # Load full Whisper model (we need it for generation)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        self.hidden_size = self.whisper.config.d_model
        self.num_encoder_layers = self.whisper.config.encoder_layers

        # Clear max_length from generation config to avoid conflict with max_new_tokens
        if hasattr(self.whisper, "generation_config"):
            self.whisper.generation_config.max_length = None

        # Freeze all Whisper parameters
        for param in self.whisper.parameters():
            param.requires_grad = False

        # Inject layers based on variant
        self.injected_layers = nn.ModuleList()
        if variant == Variant.A:
            pass  # No injection — pure frozen baseline
        elif variant == Variant.B:
            from pulse_whisper.models.pulse_module import NoiseLayer
            for _ in range(self.num_encoder_layers):
                self.injected_layers.append(NoiseLayer(self.hidden_size, alpha_init))
        elif variant == Variant.C:
            from pulse_whisper.models.pulse_module import PulseLayer
            for _ in range(self.num_encoder_layers):
                self.injected_layers.append(
                    PulseLayer(self.hidden_size, n_frequencies, alpha_init, use_phase_net=False)
                )
        elif variant == Variant.D:
            from pulse_whisper.models.pulse_module import PulseLayer
            for _ in range(self.num_encoder_layers):
                self.injected_layers.append(
                    PulseLayer(self.hidden_size, n_frequencies, alpha_init, use_phase_net=True)
                )
        elif variant == Variant.E:
            from pulse_whisper.models.pulse_module import PulseLayer
            for _ in range(self.num_encoder_layers):
                self.injected_layers.append(
                    PulseLayer(self.hidden_size, n_frequencies, alpha_init, use_phase_net=True)
                )

    def get_encoder_with_pulse(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run encoder with pulse injection after each layer.

        Args:
            input_features: Mel spectrogram (batch, n_mels, seq_len).

        Returns:
            Encoder hidden states (batch, seq_len, hidden_size).
        """
        encoder = self.whisper.model.encoder

        # Whisper encoder embedding: conv1 -> conv2 -> positional encoding
        inputs_embeds = encoder.conv1(input_features)
        inputs_embeds = nn.functional.gelu(inputs_embeds)
        inputs_embeds = encoder.conv2(inputs_embeds)
        inputs_embeds = nn.functional.gelu(inputs_embeds)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (batch, seq_len, hidden)

        embed_pos = encoder.embed_positions.weight[:inputs_embeds.shape[1]]
        hidden_states = inputs_embeds + embed_pos
        hidden_states = encoder.layernorm_embedding(hidden_states) if hasattr(encoder, 'layernorm_embedding') else hidden_states
        hidden_states = nn.functional.dropout(hidden_states, p=encoder.dropout, training=self.training)

        seq_len = hidden_states.shape[1]
        time_steps = torch.arange(seq_len, dtype=hidden_states.dtype, device=hidden_states.device)

        # Run through encoder layers with pulse injection
        for i, layer in enumerate(encoder.layers):
            hidden_states = layer(hidden_states, attention_mask=None)[0]

            if self.variant != Variant.A and i < len(self.injected_layers):
                hidden_states = self.injected_layers[i](hidden_states, time_steps)

        hidden_states = encoder.layer_norm(hidden_states)
        return hidden_states

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
    ) -> dict:
        """Full forward pass: encoder with pulse -> Whisper decoder.

        Args:
            input_features: Mel spectrogram (batch, n_mels, seq_len).
            labels: Target token IDs for computing loss.
            decoder_input_ids: Decoder input token IDs.

        Returns:
            Dict with 'loss' and 'logits'.
        """
        encoder_hidden = self.get_encoder_with_pulse(input_features)

        # Use Whisper decoder directly
        decoder = self.whisper.model.decoder
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.whisper._shift_right(labels)

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
        )

        lm_logits = self.whisper.proj_out(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": lm_logits}

    @torch.no_grad()
    def generate(self, input_features: torch.Tensor, **generate_kwargs) -> torch.Tensor:
        """Generate text from audio using encoder with pulse injection.

        Uses Whisper's generate() but with our modified encoder output.
        """
        encoder_hidden = self.get_encoder_with_pulse(input_features)

        # Create encoder outputs object matching HF expected format
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        return self.whisper.generate(
            encoder_outputs=encoder_outputs,
            **generate_kwargs,
        )

    def pulse_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable pulse/noise parameters."""
        return list(self.injected_layers.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_variant(
    variant: Variant | str,
    whisper_size: str = "tiny",
    n_frequencies: int = 64,
    alpha_init: float = 0.01,
) -> PulseWhisperEncoder:
    """Factory to build a PulseWhisperEncoder for a given variant."""
    model_name = f"openai/whisper-{whisper_size}"
    return PulseWhisperEncoder(
        whisper_model_name=model_name,
        variant=variant,
        n_frequencies=n_frequencies,
        alpha_init=alpha_init,
    )


def get_processor(whisper_size: str = "tiny") -> WhisperProcessor:
    """Get the Whisper processor for tokenization and feature extraction."""
    return WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_size}")

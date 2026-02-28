"""GatedWhisper: Whisper with silence-aware bottleneck gate.

Frozen Whisper encoder → SilenceGate → Frozen Whisper decoder.
The gate learns to suppress encoder hidden states for non-speech frames,
preventing the decoder from hallucinating on silence/noise inputs.

Only the gate parameters are trained (~2K params for Whisper-Tiny).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

from pulse_whisper.models.silence_gate import SilenceGate


class GatedWhisper(nn.Module):
    """Whisper with silence-aware bottleneck gate.

    Freezes all Whisper parameters, trains only the SilenceGate.
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-tiny",
        gate_hidden_dim: int = 32,
        gate_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.whisper_model_name = whisper_model_name
        self.gate_loss_weight = gate_loss_weight

        # Load and freeze Whisper
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        if hasattr(self.whisper, "generation_config"):
            self.whisper.generation_config.max_length = None
        for param in self.whisper.parameters():
            param.requires_grad = False

        d_model = self.whisper.config.d_model
        self.silence_gate = SilenceGate(d_model=d_model, hidden_dim=gate_hidden_dim)

    def _run_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run frozen encoder."""
        with torch.no_grad():
            encoder_outputs = self.whisper.model.encoder(input_features)
        return encoder_outputs.last_hidden_state

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        speech_mask: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass with gated encoder output.

        Args:
            input_features: Mel spectrogram (batch, n_mels, seq_len).
            labels: Target token IDs for loss.
            decoder_input_ids: Decoder input IDs.
            speech_mask: (batch, mel_seq_len) float mask where 1.0=speech, 0.0=silence.
                Used for auxiliary gate supervision. mel_seq_len is the encoder
                output length (after conv downsampling).

        Returns:
            Dict with 'loss', 'logits', 'gate_probs', and optionally 'gate_loss'.
        """
        encoder_hidden = self._run_encoder(input_features)
        gated_hidden, gate_probs = self.silence_gate(encoder_hidden)

        # Decode with gated encoder states
        decoder = self.whisper.model.decoder
        if decoder_input_ids is None and labels is not None:
            decoder_start_id = self.whisper.config.decoder_start_token_id
            decoder_input_ids = labels.new_zeros(labels.shape)
            decoder_input_ids[:, 1:] = labels[:, :-1].clone()
            decoder_input_ids[:, 0] = decoder_start_id
            decoder_input_ids = decoder_input_ids.masked_fill(
                decoder_input_ids == -100, self.whisper.config.pad_token_id
            )

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=gated_hidden,
        )
        lm_logits = self.whisper.proj_out(decoder_outputs.last_hidden_state)

        loss = None
        gate_loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            asr_loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            # Handle case where all labels are -100 (pure silence batch)
            if torch.isnan(asr_loss):
                asr_loss = torch.tensor(0.0, device=lm_logits.device, requires_grad=True)

            loss = asr_loss

            # Auxiliary gate supervision if speech_mask provided
            if speech_mask is not None:
                # Align lengths: gate_probs is (batch, encoder_seq_len)
                enc_len = gate_probs.shape[1]
                mask_len = speech_mask.shape[1]
                if mask_len > enc_len:
                    speech_mask = speech_mask[:, :enc_len]
                elif mask_len < enc_len:
                    pad = torch.ones(
                        speech_mask.shape[0], enc_len - mask_len,
                        device=speech_mask.device, dtype=speech_mask.dtype
                    )
                    speech_mask = torch.cat([speech_mask, pad], dim=1)

                # Ensure matching dtypes (avoid MPS bfloat16 mismatch)
                gate_probs_f32 = gate_probs.float()
                speech_mask_f32 = speech_mask.float()
                gate_loss = nn.functional.binary_cross_entropy(gate_probs_f32, speech_mask_f32)
                loss = asr_loss + self.gate_loss_weight * gate_loss

        return {
            "loss": loss,
            "logits": lm_logits,
            "gate_probs": gate_probs,
            "gate_loss": gate_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        hard_gate: bool = False,
        silence_threshold: float = 0.5,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate with gated encoder output.

        Args:
            input_features: Mel spectrogram.
            hard_gate: If True, use hard gating (zero out frames below threshold)
                and short-circuit to <|endoftext|> if entire input is silence.
            silence_threshold: Gate probability below which a frame is silence.
        """
        encoder_hidden = self._run_encoder(input_features)
        _, gate_probs = self.silence_gate(encoder_hidden)

        if hard_gate:
            # Hard gating: binary mask
            gate_mask = (gate_probs > silence_threshold).float().unsqueeze(-1)
            gated_hidden = encoder_hidden * gate_mask

            # Short-circuit: if mean speech prob < threshold for any sample,
            # return <|endoftext|> immediately for that sample
            batch_speech_prob = gate_probs.mean(dim=1)  # (batch,)
            all_silence = batch_speech_prob < silence_threshold

            if all_silence.all():
                # All samples are silence — return endoftext
                eos_id = self.whisper.config.eos_token_id
                return torch.full(
                    (input_features.shape[0], 1), eos_id,
                    dtype=torch.long, device=input_features.device
                )

            # For mixed batches, still run generation but with hard-gated encoder
            encoder_outputs = BaseModelOutput(last_hidden_state=gated_hidden)
            tokens = self.whisper.generate(
                encoder_outputs=encoder_outputs,
                **generate_kwargs,
            )

            # Replace output for all-silence samples with just endoftext
            if all_silence.any():
                eos_id = self.whisper.config.eos_token_id
                for b in range(input_features.shape[0]):
                    if all_silence[b]:
                        tokens[b] = eos_id

            return tokens
        else:
            # Soft gating (original behavior)
            gated_hidden = encoder_hidden * gate_probs.unsqueeze(-1)
            encoder_outputs = BaseModelOutput(last_hidden_state=gated_hidden)
            return self.whisper.generate(
                encoder_outputs=encoder_outputs,
                **generate_kwargs,
            )

    def gate_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable gate parameters."""
        return list(self.silence_gate.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_gated_whisper(
    whisper_size: str = "tiny",
    gate_hidden_dim: int = 32,
    gate_loss_weight: float = 1.0,
) -> GatedWhisper:
    """Factory to build a GatedWhisper model."""
    model_name = f"openai/whisper-{whisper_size}"
    return GatedWhisper(
        whisper_model_name=model_name,
        gate_hidden_dim=gate_hidden_dim,
        gate_loss_weight=gate_loss_weight,
    )

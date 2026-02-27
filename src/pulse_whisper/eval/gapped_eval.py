"""Gapped evaluation: run model at each gap level, collect WER.

Core evaluation pipeline for measuring Whisper robustness to silence gaps.
"""

from __future__ import annotations

import logging

import torch
from transformers import WhisperProcessor

from pulse_whisper.data.gapped_audio import GapLevel, inject_silence_gaps
from pulse_whisper.eval.metrics import (
    EvalResult,
    HallucinationResult,
    compute_cer,
    compute_hallucination_rate,
    compute_hallucination_severity,
    compute_wer,
)

logger = logging.getLogger(__name__)


def _generate(model, input_features: torch.Tensor, language: str = "en") -> torch.Tensor:
    """Generate token IDs using the appropriate model interface."""
    gen_kwargs = {"language": language, "task": "transcribe", "max_new_tokens": 440}

    if hasattr(model, "generate"):
        return model.generate(input_features, **gen_kwargs)
    else:
        return model.whisper.generate(input_features, **gen_kwargs)


@torch.no_grad()
def evaluate_gapped(
    model,
    dataloader,
    processor: WhisperProcessor,
    gap_level: GapLevel | str,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
    language: str = "en",
) -> EvalResult:
    """Evaluate model at a specific gap level.

    Args:
        model: PulseWhisperEncoder or WhisperForConditionalGeneration.
        dataloader: DataLoader yielding dicts with 'input_features' and 'texts'.
        processor: WhisperProcessor for decoding.
        gap_level: Gap difficulty level to inject.
        device: Device to run on.
        max_batches: Limit evaluation to N batches (for speed).
        language: Language for forced decoding.

    Returns:
        EvalResult with WER, CER, and predictions.
    """
    if isinstance(gap_level, str):
        gap_level = GapLevel(gap_level)

    model.eval()
    all_predictions = []
    all_references = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_features = batch["input_features"].to(device)
        references = batch["texts"]

        # Inject gaps
        if gap_level != GapLevel.NONE:
            input_features, _ = inject_silence_gaps(
                input_features, gap_level, seed=batch_idx
            )

        generated_ids = _generate(model, input_features, language)
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        all_predictions.extend(predictions)
        all_references.extend(references)

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx + 1}: {len(all_predictions)} samples processed")

    wer = compute_wer(all_predictions, all_references)
    cer = compute_cer(all_predictions, all_references)

    return EvalResult(
        gap_level=gap_level.value,
        wer=wer,
        cer=cer,
        num_samples=len(all_predictions),
        predictions=all_predictions,
        references=all_references,
    )


@torch.no_grad()
def evaluate_all_gap_levels(
    model,
    dataloader,
    processor: WhisperProcessor,
    device: torch.device | str = "cpu",
    gap_levels: list[GapLevel] | None = None,
    max_batches: int | None = None,
    language: str = "en",
) -> dict[str, EvalResult]:
    """Run gapped evaluation at all gap levels.

    Returns:
        Dict mapping gap level name to EvalResult.
    """
    if gap_levels is None:
        gap_levels = list(GapLevel)

    results = {}
    for gap_level in gap_levels:
        logger.info(f"Evaluating at gap level: {gap_level.value}")
        result = evaluate_gapped(
            model, dataloader, processor, gap_level, device, max_batches, language
        )
        results[gap_level.value] = result
        logger.info(f"  WER={result.wer:.4f}, CER={result.cer:.4f} ({result.num_samples} samples)")

    return results


@torch.no_grad()
def evaluate_hallucination(
    model,
    processor: WhisperProcessor,
    device: torch.device | str = "cpu",
    num_samples: int = 50,
    duration_seconds: float = 30.0,
    language: str = "en",
) -> dict[str, HallucinationResult]:
    """Test hallucination on silence and noise inputs.

    Creates synthetic silence and white noise inputs, runs model,
    and measures how much text the model hallucinates.

    Returns:
        Dict mapping input type to HallucinationResult.
    """
    model.eval()
    results = {}

    # Whisper expects 30s at 16kHz -> 3000 mel frames at 80 mel bins
    n_mels = 80
    seq_len = 3000

    for input_type, gen_fn in [
        ("silence", lambda: torch.zeros(1, n_mels, seq_len) - 1.0),
        ("white_noise", lambda: torch.randn(1, n_mels, seq_len) * 0.1),
    ]:
        all_outputs = []
        for i in range(num_samples):
            input_features = gen_fn().to(device)
            generated_ids = _generate(model, input_features, language)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            all_outputs.append(text)

        rate = compute_hallucination_rate(all_outputs)
        severity = compute_hallucination_severity(all_outputs)

        results[input_type] = HallucinationResult(
            input_type=input_type,
            hallucination_rate=rate,
            avg_output_length=severity,
            num_samples=num_samples,
            outputs=all_outputs,
        )
        logger.info(
            f"Hallucination [{input_type}]: rate={rate:.2%}, "
            f"avg_length={severity:.1f} words ({num_samples} samples)"
        )

    return results

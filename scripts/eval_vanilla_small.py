"""Get vanilla whisper-small gap WER metrics for baseline comparison.

Loads test-clean directly to avoid HF downloading all clean splits.

Usage:
    uv run python scripts/eval_vanilla_small.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class TestCleanDataset(Dataset):
    """Minimal dataset that loads only test-clean split."""

    def __init__(self, whisper_size: str = "small"):
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_size}")
        # Load only test split of clean config
        self.dataset = load_dataset(
            "librispeech_asr", "clean", split="test",
            trust_remote_code=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        text = item["text"]

        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.squeeze(0)
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels,
            "text": text,
        }


def collate_fn(batch):
    input_features = torch.stack([item["input_features"] for item in batch])
    labels = [item["labels"] for item in batch]
    max_label_len = max(l.shape[0] for l in labels)
    padded_labels = torch.full((len(labels), max_label_len), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, :l.shape[0]] = l
    texts = [item["text"] for item in batch]
    return {"input_features": input_features, "labels": padded_labels, "texts": texts}


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    whisper_model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    model = model.to(device)
    model.eval()

    batch_size = 16 if device.type == "cuda" else 4
    dataset = TestCleanDataset(whisper_size="small")
    test_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Vanilla whisper-small gap WER (batch_size={batch_size}):")
    gap_results = evaluate_all_gap_levels(
        model=model, dataloader=test_loader,
        processor=processor, device=device,
    )
    for level, r in gap_results.items():
        logger.info(f"  {level:<15} WER={r.wer:.4f}")

    output_dir = Path("results/silence_gate_small")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "vanilla_whisper_small_gap_wer",
        "gap_evaluation": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in gap_results.items()
        },
    }

    with open(output_dir / "vanilla_gap_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_dir / 'vanilla_gap_results.json'}")


if __name__ == "__main__":
    main()

"""LibriSpeech data loading for Whisper evaluation and training.

Handles loading LibriSpeech splits, feature extraction via Whisper processor,
and creating dataloaders for training and evaluation.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import WhisperProcessor

from pulse_whisper.data.gapped_audio import random_gap_augmentation


class LibriSpeechDataset(Dataset):
    """Wraps HuggingFace LibriSpeech for Whisper training/eval.

    Handles feature extraction (mel spectrogram) and tokenization.
    """

    def __init__(
        self,
        split: str = "test-clean",
        whisper_size: str = "tiny",
        max_samples: int | None = None,
        gap_augmentation: bool = False,
        gap_fractions: list[float] = (0.0, 0.05, 0.15),
    ) -> None:
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_size}")
        self.gap_augmentation = gap_augmentation
        self.gap_fractions = list(gap_fractions)

        # Map split names to HF dataset config
        split_map = {
            "train-clean-100": ("train.clean.100", "clean"),
            "test-clean": ("test", "clean"),
            "test-other": ("test", "other"),
        }

        if split in split_map:
            hf_split, hf_config = split_map[split]
            self.dataset = load_dataset(
                "librispeech_asr", hf_config, split=hf_split, trust_remote_code=True
            )
        else:
            raise ValueError(f"Unknown split: {split}. Use one of {list(split_map.keys())}")

        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        text = item["text"]

        # Extract mel spectrogram features
        inputs = self.processor(
            audio, sampling_rate=sr, return_tensors="pt"
        )
        input_features = inputs.input_features.squeeze(0)  # (n_mels, seq_len)

        # Tokenize text
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels,
            "text": text,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for variable-length labels."""
    input_features = torch.stack([item["input_features"] for item in batch])

    # Pad labels to same length
    labels = [item["labels"] for item in batch]
    max_label_len = max(l.shape[0] for l in labels)
    padded_labels = torch.full((len(labels), max_label_len), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, :l.shape[0]] = l

    texts = [item["text"] for item in batch]

    return {
        "input_features": input_features,
        "labels": padded_labels,
        "texts": texts,
    }


def get_dataloader(
    split: str = "test-clean",
    whisper_size: str = "tiny",
    batch_size: int = 16,
    max_samples: int | None = None,
    gap_augmentation: bool = False,
    gap_fractions: list[float] = (0.0, 0.05, 0.15),
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader for LibriSpeech evaluation or training."""
    dataset = LibriSpeechDataset(
        split=split,
        whisper_size=whisper_size,
        max_samples=max_samples,
        gap_augmentation=gap_augmentation,
        gap_fractions=gap_fractions,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def get_10h_subset_dataloader(
    whisper_size: str = "tiny",
    batch_size: int = 16,
    gap_augmentation: bool = True,
    gap_fractions: list[float] = (0.0, 0.05, 0.15),
    num_workers: int = 0,
) -> DataLoader:
    """Get ~10h subset of train-clean-100 for prototyping.

    train-clean-100 has ~28.5k utterances (~100h).
    10h ≈ 2850 utterances.
    """
    return get_dataloader(
        split="train-clean-100",
        whisper_size=whisper_size,
        batch_size=batch_size,
        max_samples=2850,
        gap_augmentation=gap_augmentation,
        gap_fractions=gap_fractions,
        num_workers=num_workers,
        shuffle=True,
    )

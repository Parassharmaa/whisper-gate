"""Test hallucination within silence gaps of mixed speech+silence audio.

Approach:
1. Transcribe clean speech → reference text
2. Inject silence gaps → transcribe again → gapped text
3. Compare: extra words in gapped text = hallucinations from gap regions

Measures insertion rate (hallucinated words) across baseline, soft gate, and hard gate.
"""

import json
import logging
import sys
from pathlib import Path

import torch
from jiwer import process_words

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pulse_whisper.models.gated_whisper import GatedWhisper
from pulse_whisper.data.gapped_audio import inject_silence_gaps
from pulse_whisper.data.dataset import get_dataloader
from pulse_whisper.models.pulse_whisper import get_processor
from transformers import WhisperForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-6s %(message)s")
logger = logging.getLogger(__name__)


def transcribe(model, input_features, processor, hard_gate=False):
    """Transcribe a batch, return list of strings."""
    gen_kwargs = {"language": "en", "task": "transcribe", "max_new_tokens": 440}
    with torch.no_grad():
        if hard_gate and hasattr(model, "generate"):
            ids = model.generate(input_features, hard_gate=True, silence_threshold=0.5, **gen_kwargs)
        elif hasattr(model, "generate"):
            ids = model.generate(input_features, **gen_kwargs)
        else:
            ids = model.generate(input_features, **gen_kwargs)
    return processor.batch_decode(ids, skip_special_tokens=True)


def count_insertions(reference: str, hypothesis: str) -> dict:
    """Count inserted words (words in hypothesis not in reference)."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if not ref_words and not hyp_words:
        return {"insertions": 0, "ref_len": 0, "hyp_len": 0, "extra_words": []}

    if not ref_words:
        return {"insertions": len(hyp_words), "ref_len": 0, "hyp_len": len(hyp_words), "extra_words": hyp_words}

    # Use jiwer alignment to find insertions
    result = process_words([reference.strip().lower()], [hypothesis.strip().lower()])
    insertions = result.insertions

    return {
        "insertions": insertions,
        "ref_len": len(ref_words),
        "hyp_len": len(hyp_words),
    }


def evaluate_gap_hallucination(model, processor, test_loader, device, gap_level, hard_gate=False, num_batches=30):
    """Evaluate hallucination in silence gaps.

    For each sample:
    1. Transcribe clean → clean_text
    2. Inject gap → transcribe → gapped_text
    3. Count extra words (insertions) in gapped_text vs ground truth
    """
    model.eval()
    total_insertions = 0
    total_ref_words = 0
    total_samples = 0
    total_hallucinated_samples = 0
    examples = []

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= num_batches:
            break

        input_features = batch["input_features"].to(device)
        ground_truth = batch["texts"]

        # Inject gaps
        gapped_features, _ = inject_silence_gaps(input_features, gap_level, seed=batch_idx)

        # Transcribe gapped audio
        gapped_texts = transcribe(model, gapped_features, processor, hard_gate=hard_gate)

        for i in range(len(ground_truth)):
            gt = ground_truth[i]
            gapped = gapped_texts[i]

            stats = count_insertions(gt, gapped)
            total_insertions += stats["insertions"]
            total_ref_words += stats["ref_len"]
            total_samples += 1

            if stats["insertions"] > 0:
                total_hallucinated_samples += 1

            # Save first few examples
            if len(examples) < 10 and stats["insertions"] > 0:
                examples.append({
                    "ground_truth": gt,
                    "gapped_output": gapped,
                    "insertions": stats["insertions"],
                })

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{num_batches}: {total_samples} samples")

    insertion_rate = total_insertions / max(total_ref_words, 1)
    halluc_sample_rate = total_hallucinated_samples / max(total_samples, 1)

    return {
        "total_samples": total_samples,
        "total_insertions": total_insertions,
        "total_ref_words": total_ref_words,
        "insertion_rate": insertion_rate,
        "samples_with_hallucination": total_hallucinated_samples,
        "hallucination_sample_rate": halluc_sample_rate,
        "examples": examples,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    processor = get_processor("tiny")
    test_loader = get_dataloader(split="test-clean", whisper_size="tiny", batch_size=8)

    gap_levels = ["gap_5", "gap_15", "gap_30"]
    num_batches = 30  # 240 samples

    results = {}

    # --- 1. Baseline (vanilla Whisper, no gate) ---
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE (vanilla Whisper)")
    logger.info("=" * 60)
    baseline = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)
    baseline.eval()

    results["baseline"] = {}
    for gap in gap_levels:
        logger.info(f"\n  Gap level: {gap}")
        r = evaluate_gap_hallucination(baseline, processor, test_loader, device, gap, num_batches=num_batches)
        results["baseline"][gap] = r
        logger.info(f"  Insertion rate: {r['insertion_rate']:.4f} ({r['total_insertions']} insertions / {r['total_ref_words']} words)")
        logger.info(f"  Samples with hallucination: {r['hallucination_sample_rate']:.2%} ({r['samples_with_hallucination']}/{r['total_samples']})")
    del baseline

    # --- 2. Soft gate ---
    logger.info("\n" + "=" * 60)
    logger.info("SOFT GATE (v4)")
    logger.info("=" * 60)
    model = GatedWhisper(whisper_model_name="openai/whisper-tiny", gate_hidden_dim=32)
    ckpt = torch.load("results/silence_gate_v4/gate_classifier.pt", map_location="cpu")
    model.silence_gate.load_state_dict(ckpt["gate_state_dict"])
    model = model.to(device)
    model.eval()

    results["soft_gate"] = {}
    for gap in gap_levels:
        logger.info(f"\n  Gap level: {gap}")
        r = evaluate_gap_hallucination(model, processor, test_loader, device, gap, hard_gate=False, num_batches=num_batches)
        results["soft_gate"][gap] = r
        logger.info(f"  Insertion rate: {r['insertion_rate']:.4f} ({r['total_insertions']} insertions / {r['total_ref_words']} words)")
        logger.info(f"  Samples with hallucination: {r['hallucination_sample_rate']:.2%} ({r['samples_with_hallucination']}/{r['total_samples']})")

    # --- 3. Hard gate ---
    logger.info("\n" + "=" * 60)
    logger.info("HARD GATE (v4, threshold=0.5)")
    logger.info("=" * 60)

    results["hard_gate"] = {}
    for gap in gap_levels:
        logger.info(f"\n  Gap level: {gap}")
        r = evaluate_gap_hallucination(model, processor, test_loader, device, gap, hard_gate=True, num_batches=num_batches)
        results["hard_gate"][gap] = r
        logger.info(f"  Insertion rate: {r['insertion_rate']:.4f} ({r['total_insertions']} insertions / {r['total_ref_words']} words)")
        logger.info(f"  Samples with hallucination: {r['hallucination_sample_rate']:.2%} ({r['samples_with_hallucination']}/{r['total_samples']})")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Insertion Rate (hallucinated words from gaps)")
    logger.info("=" * 60)
    logger.info(f"{'Gap Level':<12} {'Baseline':<15} {'Soft Gate':<15} {'Hard Gate':<15}")
    logger.info("-" * 57)
    for gap in gap_levels:
        b = results["baseline"][gap]["insertion_rate"]
        s = results["soft_gate"][gap]["insertion_rate"]
        h = results["hard_gate"][gap]["insertion_rate"]
        logger.info(f"{gap:<12} {b:<15.4f} {s:<15.4f} {h:<15.4f}")

    logger.info(f"\n{'Gap Level':<12} {'Baseline':<15} {'Soft Gate':<15} {'Hard Gate':<15}")
    logger.info("-" * 57)
    for gap in gap_levels:
        b = results["baseline"][gap]["hallucination_sample_rate"]
        s = results["soft_gate"][gap]["hallucination_sample_rate"]
        h = results["hard_gate"][gap]["hallucination_sample_rate"]
        logger.info(f"{gap:<12} {b:<15.2%} {s:<15.2%} {h:<15.2%}")

    # Save (strip examples for JSON)
    save_results = {}
    for variant in results:
        save_results[variant] = {}
        for gap in results[variant]:
            r = results[variant][gap].copy()
            r["examples"] = r["examples"][:3]  # Keep only 3 examples
            save_results[variant][gap] = r

    out_path = Path("results/silence_gate_v4/gap_hallucination.json")
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

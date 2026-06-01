"""
End-to-end pipeline: Table Image → OCR → Bounding Box → Token Indices.

Combines ocr_utils (EasyOCR) with token_map (Qwen3-VL patch grid mapping)
to measure how many visual tokens each table cell maps to.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

from PIL import Image
from transformers import AutoProcessor

from src.ocr_utils import bb_and_text_from_table_image
from src.token_map import bb_to_token_indices, compute_token_stats


def main(image_path: str, model_name: str = "Qwen/Qwen3-VL-2B-Instruct") -> None:
    """Run the full OCR → token-index pipeline and save results to JSON."""

    # --- Step 1: OCR ---
    print(f"\n[pipeline] Running OCR on: {image_path}")
    bbs, texts = bb_and_text_from_table_image(image_path=image_path)
    print(f"[pipeline] OCR found {len(bbs)} cells.\n")

    # --- Step 2: Load processor (model not needed for token index mapping) ---
    print(f"[pipeline] Loading Qwen3-VL processor from '{model_name}'...")
    processor = AutoProcessor.from_pretrained(model_name)
    print("[pipeline] Processor ready.\n")

    # --- Step 3: Map bboxes to token indices ---
    print("[pipeline] Mapping bounding boxes to visual token indices...")
    token_indices_list = bb_to_token_indices(
        image_path=image_path,
        bbs=bbs,
        processor=processor,
    )

    # --- Step 4: Compute + print stats ---
    stats = compute_token_stats(bbs, texts, token_indices_list)

    # --- Step 5: Save to JSON ---
    output_path = Path("outputs") / "token_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # token_indices contain plain ints — safe for JSON
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[pipeline] Results saved to {output_path}")


def validate_cosyn(
    num_samples: int = 20,
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    gpu_ocr: bool = False,
) -> None:
    """
    Validate the OCR → token-index pipeline on real CoSyn-400K table images.

    Loads num_samples images from allenai/CoSyn-400K (table config), runs OCR
    and token mapping on each, and reports aggregate tokens-per-cell statistics.
    """
    from datasets import load_dataset

    print(f"\n[validate] Loading CoSyn-400K (table config) via streaming...")
    dataset = load_dataset("allenai/CoSyn-400K", "table", split="train", streaming=True)

    # Take first num_samples via streaming (no full download needed)
    samples = list(dataset.take(num_samples))
    print(f"[validate] Loaded {len(samples)} samples")

    print(f"[validate] Loading Qwen3-VL processor from '{model_name}'...")
    processor = AutoProcessor.from_pretrained(model_name)
    print("[validate] Processor ready.\n")

    all_counts = []
    total_cells = 0
    skipped = 0

    total_samples = len(samples)
    for i, sample in enumerate(samples):
        pil_image: Image.Image = sample["image"]
        print(f"\n[validate] === Image {i+1}/{total_samples} ({pil_image.size}) ===")

        # --- OCR ---
        bbs, texts = bb_and_text_from_table_image(
            pil_image=pil_image,
            gpu=gpu_ocr,
            visualize=False,
        )

        if len(bbs) == 0:
            print(f"[validate] No OCR detections — skipping")
            skipped += 1
            continue

        # --- Token mapping ---
        token_indices_list = bb_to_token_indices(
            pil_image=pil_image,
            bbs=bbs,
            processor=processor,
        )

        # --- Validate indices ---
        # Total vision tokens = h_patches * w_patches from processor output
        # We verify indices are non-negative (basic sanity check)
        for j, indices in enumerate(token_indices_list):
            if any(idx < 0 for idx in indices):
                print(f"[validate] WARNING: Negative index in cell {j}")

        # --- Collect stats ---
        counts = [len(idxs) for idxs in token_indices_list]
        all_counts.extend(counts)
        total_cells += len(bbs)

        stats = compute_token_stats(bbs, texts, token_indices_list)

    # --- Aggregate report ---
    print("\n" + "=" * 70)
    print("[validate] AGGREGATE RESULTS")
    print("=" * 70)
    print(f"  Images processed: {total_samples - skipped}")
    print(f"  Images skipped (no OCR): {skipped}")
    print(f"  Total cells: {total_cells}")

    if all_counts:
        mean_val = sum(all_counts) / len(all_counts)
        sorted_counts = sorted(all_counts)
        n = len(sorted_counts)
        median_val = (
            sorted_counts[n // 2]
            if n % 2 == 1
            else (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2
        )
        min_val = min(all_counts)
        max_val = max(all_counts)
        variance = sum((c - mean_val) ** 2 for c in all_counts) / n
        std_val = math.sqrt(variance)

        print(f"  Mean tokens/cell: {mean_val:.2f}")
        print(f"  Median:           {median_val}")
        print(f"  Min / Max:        {min_val} / {max_val}")
        print(f"  Std dev:          {std_val:.2f}")

        # Save aggregate results
        output_path = Path("outputs") / "cosyn_validation_stats.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "images_processed": total_samples - skipped,
                "images_skipped": skipped,
                "total_cells": total_cells,
                "mean_tokens": mean_val,
                "median_tokens": median_val,
                "min_tokens": min_val,
                "max_tokens": max_val,
                "std_tokens": std_val,
            }, f, indent=2)
        print(f"\n[validate] Results saved to {output_path}")
    else:
        print("  No cells detected across any images!")

    print("=" * 70)


if __name__ == "__main__":
    from src.synthetic_data import generate_sample_table_image  # local import

    parser = argparse.ArgumentParser(
        description="OCR → Qwen3-VL token index pipeline"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to table image. Defaults to a generated sample.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model identifier for Qwen3-VL.",
    )
    parser.add_argument(
        "--validate-cosyn",
        action="store_true",
        help="Run validation on CoSyn-400K images instead of a single image.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of CoSyn-400K images to validate on (default: 20).",
    )
    parser.add_argument(
        "--gpu-ocr",
        action="store_true",
        help="Use GPU for EasyOCR.",
    )
    args = parser.parse_args()

    if args.validate_cosyn:
        validate_cosyn(
            num_samples=args.num_samples,
            model_name=args.model_name,
            gpu_ocr=args.gpu_ocr,
        )
    else:
        resolved_path = args.image_path or generate_sample_table_image()
        main(image_path=resolved_path, model_name=args.model_name)

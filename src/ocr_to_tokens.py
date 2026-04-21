"""
End-to-end pipeline: Table Image → OCR → Bounding Box → Token Indices.

Combines ocr_utils (EasyOCR) with token_map (Qwen3-VL patch grid mapping)
to measure how many visual tokens each table cell maps to.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor

from src.ocr_utils import bb_and_text_from_table_image
from src.token_map import bb_to_token_indices, compute_token_stats


def main(image_path: str, model_name: str = "Qwen/Qwen3-VL-2B-Instruct") -> None:
    """Run the full OCR → token-index pipeline and save results to JSON."""

    # --- Step 1: OCR ---
    print(f"\n[pipeline] Running OCR on: {image_path}")
    bbs, texts = bb_and_text_from_table_image(image_path)
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
    args = parser.parse_args()

    resolved_path = args.image_path or generate_sample_table_image()
    main(image_path=resolved_path, model_name=args.model_name)

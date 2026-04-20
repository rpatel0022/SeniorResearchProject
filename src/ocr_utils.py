"""
OCR utilities for detecting text regions in table images.

Uses EasyOCR to extract bounding boxes and text strings from an image.
Each bounding box is returned as (x1, y1, x2, y2) in pixel coordinates.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional

import easyocr


# Module-level reader so we only initialize the heavy model once
_reader: Optional[easyocr.Reader] = None


def _get_reader(gpu: bool = False) -> easyocr.Reader:
    """Lazily create an EasyOCR reader (English)."""
    global _reader
    if _reader is None:
        print("[ocr_utils] Initializing EasyOCR reader (first call may download models)...")
        _reader = easyocr.Reader(["en"], gpu=gpu)
    return _reader


def bb_and_text_from_table_image(
    image_path: str,
    gpu: bool = False,
    min_confidence: float = 0.20,
    visualize: bool = True,
    output_dir: str = "outputs",
) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """
    Detect text regions in a table image and extract bounding boxes + OCR text.

    Args:
        image_path:      Path to the input image.
        gpu:             Whether to use GPU for OCR.
        min_confidence:  Minimum OCR confidence to keep a detection.
        visualize:       If True, save an annotated image with boxes drawn.
        output_dir:      Directory for output artifacts.

    Returns:
        bbs:   List of bounding boxes as (x1, y1, x2, y2).
        texts: List of OCR-extracted strings (same length as bbs).
    """
    # --- Load image ---
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"Could not read image at {image_path}")

    reader = _get_reader(gpu=gpu)

    # --- Run OCR ---
    # EasyOCR returns list of (bbox_polygon, text, confidence)
    # bbox_polygon is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (quadrilateral)
    raw_results = reader.readtext(image_path)
    print(f"[ocr_utils] Raw OCR detections: {len(raw_results)}")

    bbs: List[Tuple[int, int, int, int]] = []
    texts: List[str] = []

    for polygon, text, conf in raw_results:
        # Filter low confidence
        if conf < min_confidence:
            print(f"  Skipping low-confidence ({conf:.2f}): '{text}'")
            continue

        # Clean text
        cleaned = text.strip()
        if not cleaned:
            print(f"  Skipping empty text (conf={conf:.2f})")
            continue

        # Convert quadrilateral to axis-aligned bounding box
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        bbs.append((x1, y1, x2, y2))
        texts.append(cleaned)

    # --- Sanity checks ---
    assert len(bbs) == len(texts), (
        f"Mismatch: {len(bbs)} boxes vs {len(texts)} texts"
    )
    print(f"[ocr_utils] Kept {len(bbs)} detections after filtering")

    # --- Print each detection ---
    for i, (bb, txt) in enumerate(zip(bbs, texts)):
        print(f"  [{i}] box={bb}  text='{txt}'")

    # --- Visualization ---
    if visualize:
        _visualize_detections(image_path, bbs, texts, output_dir)

    return bbs, texts


def _visualize_detections(
    image_path: str,
    bbs: List[Tuple[int, int, int, int]],
    texts: List[str],
    output_dir: str,
) -> str:
    """Draw bounding boxes and labels on the image; save to output_dir."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to get a small font for labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

    colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta"]

    for i, (bb, txt) in enumerate(zip(bbs, texts)):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = bb
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{i}: {txt}"
        draw.text((x1, max(y1 - 16, 0)), label, fill=color, font=font)

    out_path = Path(output_dir) / "ocr_annotated.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    print(f"[ocr_utils] Annotated image saved to {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from synthetic_data import generate_sample_table_image

    path = sys.argv[1] if len(sys.argv) > 1 else generate_sample_table_image()
    bbs, texts = bb_and_text_from_table_image(path)
    print(f"\nTotal detections: {len(bbs)}")

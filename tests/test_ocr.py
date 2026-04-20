"""
Tests for OCR text detection and bounding-box extraction.

These tests use the synthetic table image so they are fully deterministic
and do not require any external data files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.synthetic_data import generate_sample_table_image
from src.ocr_utils import bb_and_text_from_table_image


@pytest.fixture(scope="module")
def ocr_result(tmp_path_factory):
    """Run OCR once on synthetic image and share across tests."""
    out_dir = str(tmp_path_factory.mktemp("ocr_test"))
    img_path = generate_sample_table_image(save_path=f"{out_dir}/table.png")
    bbs, texts = bb_and_text_from_table_image(
        img_path, visualize=True, output_dir=out_dir
    )
    return bbs, texts, out_dir


def test_returns_nonempty(ocr_result):
    bbs, texts, _ = ocr_result
    assert len(bbs) > 0, "OCR should detect at least one text region"
    assert len(texts) > 0, "OCR should return at least one text string"
    print(f"  Found {len(bbs)} detections")


def test_lengths_match(ocr_result):
    bbs, texts, _ = ocr_result
    assert len(bbs) == len(texts), (
        f"Bounding box count ({len(bbs)}) must equal text count ({len(texts)})"
    )


def test_bounding_boxes_valid(ocr_result):
    bbs, texts, _ = ocr_result
    for i, (x1, y1, x2, y2) in enumerate(bbs):
        assert x1 < x2, f"Box {i}: x1={x1} should be < x2={x2}"
        assert y1 < y2, f"Box {i}: y1={y1} should be < y2={y2}"
        assert x1 >= 0 and y1 >= 0, f"Box {i}: coordinates should be non-negative"
        print(f"  Box {i}: ({x1},{y1},{x2},{y2}) -> '{texts[i]}'")


def test_texts_nonempty(ocr_result):
    bbs, texts, _ = ocr_result
    for i, txt in enumerate(texts):
        assert isinstance(txt, str), f"Text {i} should be a string"
        assert len(txt.strip()) > 0, f"Text {i} should not be blank"


def test_annotated_image_saved(ocr_result):
    _, _, out_dir = ocr_result
    annotated = Path(out_dir) / "ocr_annotated.png"
    assert annotated.exists(), f"Annotated image should be saved at {annotated}"
    print(f"  Annotated image: {annotated}")


def test_known_text_detected(ocr_result):
    """
    The synthetic table contains 'Apple', 'Banana', 'Orange'.
    At least one of these should be detected.
    """
    _, texts, _ = ocr_result
    all_text = " ".join(texts).lower()
    known_words = ["apple", "banana", "orange", "item", "price"]
    found = [w for w in known_words if w in all_text]
    print(f"  Known words found: {found}")
    assert len(found) >= 1, (
        f"Expected at least one of {known_words} in OCR output, got: {texts}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

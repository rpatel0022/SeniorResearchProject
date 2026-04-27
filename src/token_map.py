"""
Token mapping: OCR bounding boxes → Qwen3-VL visual token indices.

Processes the FULL image through Qwen3-VL's processor (no cropping) and uses
patch grid math to find which visual tokens correspond to each bounding box.

Key constants match Sameen's reference implementation:
  PATCH_SIZE = 16, MERGE_SIZE = 2 → EFFECTIVE_BLOCK_SIZE = 32 px per token slot.
"""

import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Constants (from Sameen's bb_to_image_embeddings reference)
# ---------------------------------------------------------------------------
PATCH_SIZE: int = 16
MERGE_SIZE: int = 2
EFFECTIVE_BLOCK_SIZE: int = PATCH_SIZE * MERGE_SIZE  # 32 px per merged patch

VISION_START_ID: int = 151652
VISION_END_ID: int = 151653

# ---------------------------------------------------------------------------
# Module-level model / processor cache — load once, reuse everywhere
# ---------------------------------------------------------------------------
_model: Optional[Qwen3VLForConditionalGeneration] = None
_processor: Optional[AutoProcessor] = None


def load_qwen3vl(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str = "cpu",
) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    """
    Lazily load Qwen3-VL model and processor (cached at module level).

    Model is loaded in bfloat16 to reduce memory pressure.

    Args:
        model_name: HuggingFace model identifier.
        device:     Torch device string ("cpu" or "cuda").

    Returns:
        (model, processor) tuple.
    """
    global _model, _processor

    if _model is None or _processor is None:
        print(f"[token_map] Loading Qwen3-VL processor from '{model_name}'...")
        _processor = AutoProcessor.from_pretrained(model_name)

        print(f"[token_map] Loading Qwen3-VL model (bfloat16) on {device}...")
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        print("[token_map] Model and processor loaded.")

    return _model, _processor


# ---------------------------------------------------------------------------
# Patch grid helpers
# ---------------------------------------------------------------------------

def get_processed_resolution(
    image_grid_thw: torch.Tensor,
    image_index: int = 0,
) -> Tuple[int, int]:
    """
    Extract the processed image height and width (in pixels) from image_grid_thw.

    The processor may resize/pad the image; this tells us the actual pixel
    dimensions Qwen3-VL used internally so we can scale bboxes correctly.

    Args:
        image_grid_thw: Shape [num_images, 3] — columns are
                        [temporal, height_patches, width_patches].
        image_index:    Which image in the batch to query.

    Returns:
        (processed_h, processed_w) in pixels.
    """
    # image_grid_thw can be a plain tensor — index into it safely
    row = image_grid_thw[image_index]
    # row: [temporal, h_patches, w_patches]
    h_patches = int(row[1].item())
    w_patches = int(row[2].item())

    processed_h = h_patches * EFFECTIVE_BLOCK_SIZE
    processed_w = w_patches * EFFECTIVE_BLOCK_SIZE

    return processed_h, processed_w


def scale_bbox(
    bbox: Tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
    processed_w: int,
    processed_h: int,
) -> Tuple[int, int, int, int]:
    """
    Linearly scale a bbox from original image coordinates to processed dimensions.

    Args:
        bbox:        (x1, y1, x2, y2) in original pixel coords.
        orig_w:      Original image width in pixels.
        orig_h:      Original image height in pixels.
        processed_w: Width after Qwen3-VL preprocessing (pixels).
        processed_h: Height after Qwen3-VL preprocessing (pixels).

    Returns:
        (x1, y1, x2, y2) scaled to processed image space, as ints.
    """
    x1, y1, x2, y2 = bbox

    scale_x = processed_w / orig_w
    scale_y = processed_h / orig_h

    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )


def find_qwen3vl_image_tokens(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    bbox_scaled: Tuple[int, int, int, int],
    image_index: int = 0,
) -> List[int]:
    """
    Map a scaled bounding box to global visual token indices in the sequence.

    Strategy: locate the vision_start token, then walk the patch grid to find
    which (row, col) positions overlap the bbox, converting each to a sequence
    index by adding the global offset of the image token block.

    Args:
        input_ids:    1-D or 2-D (batch=1) tensor of token ids.
        image_grid_thw: Shape [num_images, 3].
        bbox_scaled:  (x1, y1, x2, y2) already in processed image coordinates.
        image_index:  Which image in the sequence (for multi-image prompts).

    Returns:
        List of global (sequence-level) token indices covered by the bbox.
    """
    # Flatten to 1-D if batched
    ids = input_ids.flatten() if input_ids.dim() == 2 else input_ids
    ids_list: List[int] = ids.tolist()

    # --- Find the image_index-th vision_start token ---
    vision_start_positions = [
        pos for pos, tok in enumerate(ids_list) if tok == VISION_START_ID
    ]
    if image_index >= len(vision_start_positions):
        raise ValueError(
            f"[token_map] image_index={image_index} but only "
            f"{len(vision_start_positions)} vision_start tokens found."
        )
    vision_start_pos = vision_start_positions[image_index]

    # --- Patch grid dimensions for this image ---
    row = image_grid_thw[image_index]
    h_patches = int(row[1].item())
    w_patches = int(row[2].item())

    x1, y1, x2, y2 = bbox_scaled

    # --- Which patch columns / rows does the bbox touch? ---
    col_start = x1 // EFFECTIVE_BLOCK_SIZE
    col_end = math.ceil(x2 / EFFECTIVE_BLOCK_SIZE)
    row_start = y1 // EFFECTIVE_BLOCK_SIZE
    row_end = math.ceil(y2 / EFFECTIVE_BLOCK_SIZE)

    # Clamp to valid patch range
    col_start = max(0, min(col_start, w_patches - 1))
    col_end = max(1, min(col_end, w_patches))
    row_start = max(0, min(row_start, h_patches - 1))
    row_end = max(1, min(row_end, h_patches))

    # Global offset: vision_start is at vision_start_pos; image tokens start
    # at vision_start_pos + 1 (the token right after the start marker).
    global_offset = vision_start_pos + 1

    token_indices: List[int] = []
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            local_idx = r * w_patches + c
            token_indices.append(global_offset + local_idx)

    return token_indices


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def bb_to_token_indices(
    image_path: Optional[str] = None,
    pil_image: Optional[Image.Image] = None,
    bbs: List[Tuple[int, int, int, int]] = [],
    processor: Optional[AutoProcessor] = None,
    device: str = "cpu",
) -> List[List[int]]:
    """
    Map OCR bounding boxes to Qwen3-VL visual token indices.

    The full image is fed through the processor unchanged; bboxes are scaled to
    the processed resolution and converted to patch-grid positions.

    Args:
        image_path: Path to the table image (provide this OR pil_image).
        pil_image:  PIL Image object (alternative to image_path).
        bbs:        List of (x1, y1, x2, y2) bboxes from OCR.
        processor:  Loaded AutoProcessor for Qwen3-VL.
        device:     Torch device string.

    Returns:
        List of token-index lists, one per bbox.
    """
    # Defer import so the module works even if qwen_vl_utils isn't installed
    # during unit-test stubs — the real pipeline will always have it.
    from qwen_vl_utils import process_vision_info  # type: ignore

    if pil_image is not None:
        img = pil_image.convert("RGB")
    elif image_path is not None:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"[token_map] Image not found: {image_path}")
        img = Image.open(image_path).convert("RGB")
    else:
        raise ValueError("Must provide either image_path or pil_image")

    orig_w, orig_h = img.size
    print(f"[token_map] Original image size: {orig_w}x{orig_h}")

    # --- Build minimal prompt following Qwen3-VL docs ---
    # Use PIL image directly in the message for both file and PIL paths
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this table."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    input_ids: torch.Tensor = inputs["input_ids"]
    image_grid_thw: torch.Tensor = inputs["image_grid_thw"]

    print(f"[token_map] input_ids shape: {input_ids.shape}")
    print(f"[token_map] image_grid_thw: {image_grid_thw}")

    # --- Get processed image dimensions ---
    processed_h, processed_w = get_processed_resolution(image_grid_thw, image_index=0)
    print(f"[token_map] Processed resolution: {processed_w}x{processed_h}")

    # --- Map each bbox ---
    all_token_indices: List[List[int]] = []
    for i, bb in enumerate(bbs):
        scaled = scale_bbox(bb, orig_w, orig_h, processed_w, processed_h)
        indices = find_qwen3vl_image_tokens(input_ids, image_grid_thw, scaled)
        all_token_indices.append(indices)
        print(f"[token_map] bb[{i}] {bb} → {len(indices)} token(s)")

    return all_token_indices


# ---------------------------------------------------------------------------
# Stats + reporting
# ---------------------------------------------------------------------------

def compute_token_stats(
    bbs: List[Tuple[int, int, int, int]],
    texts: List[str],
    token_indices_list: List[List[int]],
) -> Dict[str, Any]:
    """
    Summarise token coverage per OCR cell and across the full table.

    Args:
        bbs:               OCR bounding boxes.
        texts:             OCR text strings (same length as bbs).
        token_indices_list: Token index lists from bb_to_token_indices.

    Returns:
        Dict with keys:
          - per_cell: list of dicts with text, bbox, num_tokens, token_indices
          - mean_tokens, median_tokens, min_tokens, max_tokens, std_tokens
    """
    counts = [len(idxs) for idxs in token_indices_list]

    per_cell = [
        {
            "text": txt,
            "bbox": bb,
            "num_tokens": len(idxs),
            "token_indices": idxs,
        }
        for txt, bb, idxs in zip(texts, bbs, token_indices_list)
    ]

    n = len(counts)
    mean_val = sum(counts) / n if n else 0.0
    sorted_counts = sorted(counts)
    median_val = (
        sorted_counts[n // 2]
        if n % 2 == 1
        else (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2
        if n
        else 0.0
    )
    min_val = min(counts) if counts else 0
    max_val = max(counts) if counts else 0
    variance = sum((c - mean_val) ** 2 for c in counts) / n if n else 0.0
    std_val = math.sqrt(variance)

    print("\n[token_map] ── Token Coverage Report ─────────────────────────────")
    print(f"  Cells processed : {n}")
    print(f"  Mean tokens/cell: {mean_val:.2f}")
    print(f"  Median          : {median_val}")
    print(f"  Min / Max       : {min_val} / {max_val}")
    print(f"  Std dev         : {std_val:.2f}")
    print("[token_map] ── Per-cell breakdown ───────────────────────────────")
    for cell in per_cell:
        print(f"  '{cell['text']}'  bbox={cell['bbox']}  tokens={cell['num_tokens']}")
    print("[token_map] ──────────────────────────────────────────────────────\n")

    return {
        "per_cell": per_cell,
        "mean_tokens": mean_val,
        "median_tokens": median_val,
        "min_tokens": min_val,
        "max_tokens": max_val,
        "std_tokens": std_val,
    }

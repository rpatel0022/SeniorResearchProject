"""
Embedding utilities using CLIP.

Provides functions to encode cropped image regions and OCR text strings
into a shared embedding space via OpenAI CLIP (loaded through HuggingFace
transformers).  Because CLIP was trained on image-text pairs, cosine
similarity between image and text embeddings is meaningful out of the box.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional

from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Model loading (cached at module level)
# ---------------------------------------------------------------------------

_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None
_CLIP_NAME = "openai/clip-vit-base-patch32"


def load_clip(device: str = "cpu") -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Load the CLIP model and processor, caching after first call.

    Returns:
        (model, processor) tuple.
    """
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[embedding_utils] Loading CLIP model: {_CLIP_NAME} ...")
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_NAME)
        _clip_model = CLIPModel.from_pretrained(_CLIP_NAME)
        _clip_model = _clip_model.to(device).eval()
        print(f"[embedding_utils] CLIP loaded on {device}")
    return _clip_model, _clip_processor


# ---------------------------------------------------------------------------
# Image embeddings
# ---------------------------------------------------------------------------

def bb_to_image_embedding(
    image: Image.Image,
    bb: Tuple[int, int, int, int],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str = "cpu",
    save_crop: bool = False,
    crop_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Crop an image region defined by a bounding box and encode it with CLIP.

    Args:
        image:      PIL Image (full table image).
        bb:         Bounding box as (x1, y1, x2, y2).
        model:      CLIP model.
        processor:  CLIP processor.
        device:     'cpu' or 'cuda'.
        save_crop:  If True, save the cropped region for inspection.
        crop_path:  File path to save the crop (used if save_crop is True).

    Returns:
        Normalized embedding tensor of shape [D].
    """
    x1, y1, x2, y2 = bb

    # Defensive: clamp coordinates to image bounds
    w, h = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box after clamping: ({x1},{y1},{x2},{y2})")

    crop = image.crop((x1, y1, x2, y2))

    if save_crop and crop_path:
        Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)

    # Encode with CLIP vision encoder
    inputs = processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        # transformers v5+ returns BaseModelOutputWithPooling instead of tensor
        emb = out.pooler_output if hasattr(out, "pooler_output") else out  # [1, D]

    # L2 normalize
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0)  # [D]


def bb_to_image_embeddings(
    image: Image.Image,
    bbs: List[Tuple[int, int, int, int]],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str = "cpu",
    save_crops: bool = False,
    output_dir: str = "outputs/crops",
) -> torch.Tensor:
    """
    Encode multiple bounding-box regions into a stacked embedding tensor.

    Args:
        image:      PIL Image (full table image).
        bbs:        List of bounding boxes.
        model:      CLIP model.
        processor:  CLIP processor.
        device:     'cpu' or 'cuda'.
        save_crops: If True, save each cropped region.
        output_dir: Directory for saved crops.

    Returns:
        Tensor of shape [N, D] where N = len(bbs).
    """
    embeddings = []
    for i, bb in enumerate(bbs):
        crop_path = f"{output_dir}/crop_{i}.png" if save_crops else None
        emb = bb_to_image_embedding(
            image, bb, model, processor, device,
            save_crop=save_crops, crop_path=crop_path,
        )
        embeddings.append(emb)

    result = torch.stack(embeddings, dim=0)  # [N, D]
    print(f"[embedding_utils] Image embeddings shape: {result.shape}")
    print(f"  First embedding (first 5 values): {result[0, :5].tolist()}")
    return result


# ---------------------------------------------------------------------------
# Text embeddings
# ---------------------------------------------------------------------------

def get_text_embedding(
    text: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Encode a single text string with the CLIP text encoder.

    Args:
        text:       Input text string.
        model:      CLIP model.
        processor:  CLIP processor.
        device:     'cpu' or 'cuda'.

    Returns:
        Normalized embedding tensor of shape [D].
    """
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        # transformers v5+ returns BaseModelOutputWithPooling instead of tensor
        emb = out.pooler_output if hasattr(out, "pooler_output") else out  # [1, D]

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0)  # [D]


def get_text_embeddings(
    texts: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Encode multiple text strings into a stacked embedding tensor.

    Args:
        texts:      List of text strings.
        model:      CLIP model.
        processor:  CLIP processor.
        device:     'cpu' or 'cuda'.

    Returns:
        Tensor of shape [N, D] where N = len(texts).
    """
    embeddings = []
    for text in texts:
        emb = get_text_embedding(text, model, processor, device)
        embeddings.append(emb)

    result = torch.stack(embeddings, dim=0)  # [N, D]
    print(f"[embedding_utils] Text embeddings shape: {result.shape}")
    print(f"  First embedding (first 5 values): {result[0, :5].tolist()}")
    return result


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two sets of embeddings.

    Both a and b are assumed to be L2-normalized already.

    Args:
        a: Tensor of shape [N, D]
        b: Tensor of shape [M, D]

    Returns:
        Similarity matrix of shape [N, M].
    """
    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D tensors, got {a.shape} and {b.shape}"
    assert a.shape[1] == b.shape[1], f"Embedding dims must match: {a.shape[1]} vs {b.shape[1]}"
    return a @ b.T


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_clip(device)

    # Text embedding demo
    for text in ["Apple", "Banana", "$2", "table"]:
        emb = get_text_embedding(text, model, processor, device)
        print(f"  '{text}' -> shape {emb.shape}, first 3: {emb[:3].tolist()}")

    # Similarity demo
    e1 = get_text_embedding("Apple", model, processor, device)
    e2 = get_text_embedding("Banana", model, processor, device)
    e3 = get_text_embedding("car", model, processor, device)
    print(f"\n  cos(Apple, Banana) = {(e1 @ e2).item():.4f}")
    print(f"  cos(Apple, car)    = {(e1 @ e3).item():.4f}")

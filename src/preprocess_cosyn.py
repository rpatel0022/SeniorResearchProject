"""
preprocess_cosyn.py

Preprocess a subset of CoSyn-400K for alignment loss training:
  1. Subsample N examples (deterministic via Python random, seed=42)
  2. Run EasyOCR on each image → (bboxes, texts)
  3. Pass each text through Qwen3-VL's LM (text-only, no image)
  4. Extract hidden states from all layers, mean-pool over tokens
  5. Save per-example .pt files

One-time run. If interrupted, delete output dir and restart.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

from src.ocr_utils import bb_and_text_from_table_image


def get_image_hash(pil_image: Image.Image) -> str:
    """Compute a short SHA256 hash of image bytes for data integrity."""
    return hashlib.sha256(pil_image.tobytes()).hexdigest()[:16]


def extract_text_hidden_states(
    texts: List[str],
    processor: AutoProcessor,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Run text-only forward passes through Qwen3-VL's LM and extract
    mean-pooled hidden states from all layers.

    Args:
        texts: List of OCR text strings.
        processor: Qwen3-VL processor (used for tokenization).
        model: Qwen3-VL model (full model, we use the language_model part).
        device: Torch device.
        batch_size: Texts per forward pass.

    Returns:
        Tensor of shape [num_texts, num_layers, hidden_dim] in bf16.
    """
    all_hidden_states = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize text-only (no image)
        inputs = processor.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor: [batch, seq_len, hidden_dim]
        # Skip index 0 (embedding layer), keep layers 1..N
        hidden_states = outputs.hidden_states[1:]  # 28 layers for 2B model
        num_layers = len(hidden_states)

        # Mean-pool over non-padding tokens for each text in the batch
        attention_mask = inputs["attention_mask"]  # [batch, seq_len]
        mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]

        for j in range(len(batch_texts)):
            # Stack all layers for this text: [num_layers, seq_len, hidden_dim]
            text_layers = torch.stack(
                [hidden_states[layer][j] for layer in range(num_layers)]
            )
            # Expand mask for this text: [1, seq_len, 1]
            text_mask = mask_expanded[j].unsqueeze(0)  # [1, seq_len, 1]
            # Mean-pool: [num_layers, hidden_dim]
            pooled = (text_layers * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)
            all_hidden_states.append(pooled)

    # Stack all texts: [num_texts, num_layers, hidden_dim]
    return torch.stack(all_hidden_states)


def preprocess(
    num_samples: int = 5000,
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    output_dir: str = "data/preprocessed",
    seed: int = 42,
    gpu_ocr: bool = False,
    batch_size: int = 64,
) -> None:
    """
    Preprocess CoSyn-400K subset: OCR + text hidden state extraction.

    Processor resolution contract: uses AutoProcessor.from_pretrained(model_name)
    with default settings. Training must use the same processor defaults.
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load dataset via streaming (avoids full download) ---
    print(f"[preprocess] Loading CoSyn-400K (table config) via streaming...")
    dataset = load_dataset("allenai/CoSyn-400K", "table", split="train", streaming=True)

    # --- Deterministic subsampling with Python random ---
    # We stream and collect, skipping zero-detection examples
    print(f"[preprocess] Collecting {num_samples} samples (seed={seed})...")
    # Note: with streaming we can't shuffle by index, so we take a larger
    # pool and subsample from it. For the full 5K run, we take sequentially
    # and rely on the dataset's natural ordering.
    # For reproducibility, we process in order and skip zero-detection images.

    # --- Load model + processor ---
    print(f"[preprocess] Loading Qwen3-VL model from '{model_name}'...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(model_name)

    # Load only the language model part for text-only forward passes
    # We load the full model but only use it for text encoding
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Record processor config for resolution contract enforcement
    processor_config = {
        "model_name": model_name,
        "min_pixels": getattr(processor.image_processor, "min_pixels", None),
        "max_pixels": getattr(processor.image_processor, "max_pixels", None),
    }
    print(f"[preprocess] Processor config: {processor_config}")
    print(f"[preprocess] Device: {device}")

    # --- Process images ---
    saved = 0
    skipped = 0
    skipped_indices = []

    pbar = tqdm(total=num_samples, desc="Preprocessing")

    for stream_idx, sample in enumerate(dataset):
        if saved >= num_samples:
            break

        pil_image: Image.Image = sample["image"]

        # --- OCR ---
        bbs, texts = bb_and_text_from_table_image(
            pil_image=pil_image,
            gpu=gpu_ocr,
            visualize=False,
        )

        if len(bbs) == 0:
            skipped += 1
            skipped_indices.append(stream_idx)
            continue

        # --- Extract text hidden states ---
        text_hidden_states = extract_text_hidden_states(
            texts=texts,
            processor=processor,
            model=model,
            device=device,
            batch_size=batch_size,
        )
        # text_hidden_states shape: [num_cells, num_layers, hidden_dim]

        # --- Compute image hash for data integrity ---
        image_hash = get_image_hash(pil_image)

        # --- Save .pt file ---
        pt_data = {
            "idx": saved,
            "row_id": stream_idx,
            "bboxes": bbs,
            "texts": texts,
            "text_hidden_states": text_hidden_states.cpu(),
            "image_size": pil_image.size,  # (width, height)
            "processor_config": processor_config,
            "image_hash": image_hash,
        }

        pt_path = output_path / f"{saved:05d}.pt"
        torch.save(pt_data, pt_path)

        saved += 1
        pbar.update(1)
        pbar.set_postfix(skipped=skipped, cells=len(bbs))

    pbar.close()

    # --- Summary ---
    print(f"\n[preprocess] Done!")
    print(f"  Saved: {saved} examples")
    print(f"  Skipped (no OCR): {skipped}")
    print(f"  Output dir: {output_path}")
    if skipped_indices:
        print(f"  Skipped stream indices: {skipped_indices[:20]}{'...' if len(skipped_indices) > 20 else ''}")

    # Save metadata
    metadata = {
        "num_samples": saved,
        "skipped": skipped,
        "seed": seed,
        "model_name": model_name,
        "processor_config": processor_config,
        "skipped_indices": skipped_indices,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {output_path / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CoSyn-400K for alignment loss training"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5000,
        help="Number of examples to preprocess (default: 5000)",
    )
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model identifier for Qwen3-VL",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/preprocessed",
        help="Output directory for .pt files",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subsampling",
    )
    parser.add_argument(
        "--gpu-ocr", action="store_true",
        help="Use GPU for EasyOCR",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for text-only forward passes",
    )
    args = parser.parse_args()

    preprocess(
        num_samples=args.num_samples,
        model_name=args.model_name,
        output_dir=args.output_dir,
        seed=args.seed,
        gpu_ocr=args.gpu_ocr,
        batch_size=args.batch_size,
    )

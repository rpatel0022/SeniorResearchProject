#!/usr/bin/env python3
"""
demo.py — End-to-end demonstration of the multimodal table alignment pipeline.

Workflow:
  1. Generate (or load) a sample table image.
  2. Run OCR to extract bounding boxes and text.
  3. Visualize the detected regions.
  4. Compute CLIP image embeddings for each bounding-box crop.
  5. Compute CLIP text embeddings for each OCR string.
  6. Show the similarity matrix BEFORE training.
  7. Train lightweight projection heads to improve alignment.
  8. Show the similarity matrix AFTER training.
  9. Save all artifacts (annotated image, crops, heatmaps, loss curve).

Usage:
    python -m src.demo                          # synthetic image, defaults
    python -m src.demo --image_path my_table.png
    python -m src.demo --use_synthetic --epochs 300 --loss_type contrastive
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic_data import generate_sample_table_image
from src.ocr_utils import bb_and_text_from_table_image
from src.embedding_utils import (
    load_clip,
    bb_to_image_embeddings,
    get_text_embeddings,
    cosine_similarity_matrix,
)
from src.losses import compute_alignment_loss, retrieval_accuracy
from src.train import train_alignment


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  MULTIMODAL TABLE ALIGNMENT — DEMO")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Get a table image
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Prepare table image")
    print("=" * 60)

    if args.use_synthetic or args.image_path is None:
        image_path = generate_sample_table_image(save_path=f"{output_dir}/sample_table.png")
    else:
        image_path = args.image_path
        if not Path(image_path).exists():
            print(f"ERROR: Image not found at {image_path}")
            sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    print(f"  Image loaded: {image.size[0]}x{image.size[1]} pixels\n")

    # ------------------------------------------------------------------
    # Step 2: OCR — extract bounding boxes and text
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: OCR — detect text regions")
    print("=" * 60)

    bbs, texts = bb_and_text_from_table_image(
        image_path, visualize=True, output_dir=output_dir
    )

    if len(bbs) == 0:
        print("ERROR: OCR found no text. Cannot proceed.")
        sys.exit(1)

    print(f"\n  Detected {len(bbs)} text regions:")
    for i, (bb, txt) in enumerate(zip(bbs, texts)):
        print(f"    [{i}] '{txt}'  at {bb}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Compute CLIP embeddings
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Compute CLIP embeddings")
    print("=" * 60)

    model, processor = load_clip(device)

    # Image embeddings (one per bounding box crop)
    print("\n--- Image embeddings ---")
    img_emb = bb_to_image_embeddings(
        image, bbs, model, processor, device,
        save_crops=True, output_dir=f"{output_dir}/crops",
    )

    # Text embeddings (one per OCR string)
    print("\n--- Text embeddings ---")
    txt_emb = get_text_embeddings(texts, model, processor, device)

    # Sanity checks
    assert img_emb.shape[0] == txt_emb.shape[0], "Count mismatch between image and text embeddings!"
    assert img_emb.shape[1] == txt_emb.shape[1], "Embedding dimension mismatch!"
    print(f"\n  {img_emb.shape[0]} pairs, embedding dim = {img_emb.shape[1]}")

    # ------------------------------------------------------------------
    # Step 4: Raw similarity matrix (before training)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Raw CLIP similarity (before any training)")
    print("=" * 60)

    sim_raw = cosine_similarity_matrix(img_emb, txt_emb)
    _print_labeled_sim(sim_raw, texts)

    print("\n  Computing alignment loss (cosine mode):")
    loss_before = compute_alignment_loss(img_emb, txt_emb, mode="cosine", verbose=True)

    acc_before = retrieval_accuracy(img_emb, txt_emb)
    print(f"  Retrieval accuracy: i2t={acc_before['i2t_acc']:.2f}  t2i={acc_before['t2i_acc']:.2f}")

    # ------------------------------------------------------------------
    # Step 5: Train alignment projection heads
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 5: Train alignment ({args.epochs} epochs, loss={args.loss_type})")
    print("=" * 60)

    result = train_alignment(
        img_emb, txt_emb,
        epochs=args.epochs,
        lr=args.lr,
        loss_mode=args.loss_type,
        device=device,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Summary")
    print("=" * 60)

    diag_before = result["sim_before"].diag().mean().item()
    diag_after = result["sim_after"].diag().mean().item()

    print(f"  Matched-pair mean similarity : {diag_before:+.4f} -> {diag_after:+.4f}")
    print(f"  Training loss                : {result['losses'][0]:.4f} -> {result['losses'][-1]:.4f}")
    print(f"  Retrieval accuracy (i2t)     : {result['acc_before']['i2t_acc']:.2f} -> {result['acc_after']['i2t_acc']:.2f}")
    print(f"  Retrieval accuracy (t2i)     : {result['acc_before']['t2i_acc']:.2f} -> {result['acc_after']['t2i_acc']:.2f}")

    print(f"\n  Saved artifacts in '{output_dir}/':")
    for p in sorted(Path(output_dir).rglob("*")):
        if p.is_file():
            print(f"    {p}")

    print(f"\n{'='*60}")
    print("  DEMO COMPLETE")
    print(f"{'='*60}\n")


def _print_labeled_sim(sim: torch.Tensor, texts: list) -> None:
    """Print a similarity matrix with text labels."""
    N = sim.shape[0]
    # Header
    header = "          " + "  ".join(f"img_{j:2d}" for j in range(N))
    print(header)
    for i in range(N):
        label = texts[i][:10].ljust(10)
        row = "  ".join(f"{sim[i, j].item():+.4f}" for j in range(N))
        print(f"  {label}{row}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Table Alignment Demo"
    )
    parser.add_argument(
        "--image_path", type=str, default=None,
        help="Path to a table image (if omitted, uses synthetic)",
    )
    parser.add_argument(
        "--use_synthetic", action="store_true", default=True,
        help="Generate and use a synthetic table image",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs (default: 200)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--loss_type", type=str, default="cosine",
        choices=["cosine", "contrastive"],
        help="Loss function: 'cosine' or 'contrastive'",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory for output artifacts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

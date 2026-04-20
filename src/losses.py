"""
Alignment loss functions.

Two loss modes:
  - "cosine":       simple mean of (1 - cosine_similarity) for matched pairs
  - "contrastive":  symmetric InfoNCE / CLIP-style contrastive loss

Lower loss = better alignment between matched image-text pairs.
"""

import torch
import torch.nn.functional as F
from typing import Literal


def compute_alignment_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    mode: Literal["cosine", "contrastive"] = "cosine",
    temperature: float = 0.07,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute alignment loss between matched image-text embedding pairs.

    The i-th image embedding is assumed to correspond to the i-th text embedding.

    Args:
        image_embeddings:  Tensor of shape [N, D], L2-normalized.
        text_embeddings:   Tensor of shape [N, D], L2-normalized.
        mode:              "cosine" for simple pairwise loss,
                           "contrastive" for InfoNCE-style loss.
        temperature:       Temperature scaling for contrastive mode.
        verbose:           Print per-pair similarities and diagnostics.

    Returns:
        Scalar loss tensor (differentiable).
    """
    # --- Shape checks ---
    assert image_embeddings.dim() == 2, f"Expected 2D, got {image_embeddings.shape}"
    assert text_embeddings.dim() == 2, f"Expected 2D, got {text_embeddings.shape}"
    assert image_embeddings.shape == text_embeddings.shape, (
        f"Shape mismatch: images {image_embeddings.shape} vs texts {text_embeddings.shape}"
    )
    N, D = image_embeddings.shape

    # --- Full similarity matrix (used for diagnostics and contrastive loss) ---
    # sim[i, j] = cosine similarity between image_i and text_j
    sim_matrix = image_embeddings @ text_embeddings.T  # [N, N]

    if verbose:
        print(f"\n[losses] Similarity matrix ({N}x{N}):")
        for i in range(N):
            row_str = "  ".join(f"{sim_matrix[i, j].item():+.4f}" for j in range(N))
            print(f"  row {i}: {row_str}")
        diag = sim_matrix.diag()
        print(f"  Matched (diagonal) mean similarity: {diag.mean().item():.4f}")

    if mode == "cosine":
        # Simple pairwise loss: mean of (1 - cos_sim) for matched pairs.
        # Perfect alignment -> loss = 0.  Orthogonal -> loss = 1.
        matched_sims = sim_matrix.diag()  # [N]
        loss = (1.0 - matched_sims).mean()

        if verbose:
            print(f"[losses] Cosine loss: {loss.item():.4f}  (lower is better)")

    elif mode == "contrastive":
        # Symmetric InfoNCE (CLIP-style).
        # For each image, the correct text should have highest similarity,
        # and vice versa.  Temperature controls sharpness.
        logits = sim_matrix / temperature  # [N, N]
        labels = torch.arange(N, device=image_embeddings.device)

        # Image-to-text cross-entropy
        loss_i2t = F.cross_entropy(logits, labels)
        # Text-to-image cross-entropy
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2.0

        if verbose:
            print(f"[losses] Contrastive loss: {loss.item():.4f}  (lower is better)")
            print(f"  i2t CE: {loss_i2t.item():.4f}  |  t2i CE: {loss_t2i.item():.4f}")

    else:
        raise ValueError(f"Unknown loss mode: '{mode}'. Use 'cosine' or 'contrastive'.")

    return loss


def retrieval_accuracy(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> dict:
    """
    Compute retrieval accuracy: for each text, is the correct image top-1?

    Args:
        image_embeddings: [N, D] normalized embeddings.
        text_embeddings:  [N, D] normalized embeddings.

    Returns:
        Dict with 'i2t_acc' and 't2i_acc' (float, 0-1).
    """
    sim = image_embeddings @ text_embeddings.T  # [N, N]
    N = sim.shape[0]

    labels = torch.arange(N, device=sim.device)

    # Image-to-text: for each image row, is the argmax the correct text?
    i2t_preds = sim.argmax(dim=1)
    i2t_acc = (i2t_preds == labels).float().mean().item()

    # Text-to-image: for each text column, is the argmax the correct image?
    t2i_preds = sim.argmax(dim=0)
    t2i_acc = (t2i_preds == labels).float().mean().item()

    return {"i2t_acc": i2t_acc, "t2i_acc": t2i_acc}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Fake embeddings: 4 pairs, dim 8
    N, D = 4, 8
    img_emb = F.normalize(torch.randn(N, D), dim=-1)
    txt_emb = F.normalize(torch.randn(N, D), dim=-1)

    print("=== Cosine loss ===")
    loss_cos = compute_alignment_loss(img_emb, txt_emb, mode="cosine")

    print("\n=== Contrastive loss ===")
    loss_con = compute_alignment_loss(img_emb, txt_emb, mode="contrastive")

    print("\n=== Retrieval accuracy ===")
    acc = retrieval_accuracy(img_emb, txt_emb)
    print(f"  i2t_acc: {acc['i2t_acc']:.2f}  |  t2i_acc: {acc['t2i_acc']:.2f}")

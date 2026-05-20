"""
Alignment loss functions.

Loss modes:
  - "cosine":       simple mean of (1 - cosine_similarity) for matched pairs
  - "contrastive":  symmetric InfoNCE / CLIP-style contrastive loss
  - "lm_head_bow":  CE between LM-head logits of pooled visual reps and the
                    bag of OCR cell tokens (frozen lm_head used as a probe).

Lower loss = better alignment between matched image-text pairs.
"""

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def compute_lm_head_bow_loss(
    visual_reps: torch.Tensor,
    cell_token_ids: List[List[int]],
    lm_head: nn.Linear,
    final_norm: Optional[nn.Module] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Decode pooled visual reps through a frozen LM head and compute bag-of-words
    cross-entropy against OCR cell tokens.

    Per Sameen (2026-05-13), inspired by the "VLM needs words" finding:
    intermediate-layer visual reps are partially decodable through the LM head,
    so push them to be linguistically decodable against ground-truth OCR.

    lm_head weights are detached so gradients flow into the visual reps (and
    thus into the LM layers that produced them) but NOT into lm_head itself —
    the head stays a fixed probe; the visual stream has to move to match it.

    Args:
        visual_reps:     [num_cells, hidden_dim] pooled visual reps in fp32 or bf16.
        cell_token_ids:  per-cell list of token IDs (no special tokens).
                         Length must equal num_cells. Empty cells are skipped.
        lm_head:         model.lm_head (or equivalent Linear projecting hidden → vocab).
        final_norm:      optional final LM norm (e.g. RMSNorm) applied with detached
                         params before lm_head. None = skip the norm.
        verbose:         print per-cell NLLs.

    Returns:
        Scalar loss (mean over non-empty cells). Returns 0.0 if all cells are empty.
    """
    assert visual_reps.dim() == 2, f"Expected 2D visual_reps, got {visual_reps.shape}"
    assert len(cell_token_ids) == visual_reps.shape[0], (
        f"cell_token_ids length {len(cell_token_ids)} != num_cells {visual_reps.shape[0]}"
    )

    # Apply final norm with detached params (if provided) — same dtype as visual_reps
    h = visual_reps
    if final_norm is not None:
        # RMSNorm/LayerNorm has a `weight` param; detach to freeze.
        # Cleanest way: a functional re-apply using detached params.
        weight = final_norm.weight.detach()
        eps = getattr(final_norm, "variance_epsilon", getattr(final_norm, "eps", 1e-6))
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        in_dtype = h.dtype
        h = h.float()
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + eps)
        h = (h * weight.float()).to(in_dtype)

    # Logits via detached lm_head (functional form so weights stay frozen for this loss)
    W = lm_head.weight.detach()
    b = lm_head.bias.detach() if lm_head.bias is not None else None
    logits = F.linear(h, W, b)  # [num_cells, vocab]

    log_probs = F.log_softmax(logits.float(), dim=-1)

    cell_nlls = []
    for j, token_ids in enumerate(cell_token_ids):
        if not token_ids:
            continue
        ids = torch.tensor(token_ids, device=log_probs.device, dtype=torch.long)
        # Bag-of-words NLL: average -log p(token) across the cell's tokens
        nll = -log_probs[j, ids].mean()
        cell_nlls.append(nll)

    if not cell_nlls:
        return torch.tensor(0.0, device=visual_reps.device, requires_grad=False)

    loss = torch.stack(cell_nlls).mean()

    if verbose:
        print(f"[losses] LM-head BoW loss: {loss.item():.4f} over {len(cell_nlls)} cells")

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

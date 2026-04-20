"""
Training loop for multimodal alignment.

Strategy:
  - Freeze CLIP backbone entirely (both image and text encoders).
  - Add small trainable projection heads on top of the frozen embeddings.
  - Optimize these projection heads so that matched image-text pairs
    become closer in the projected space.

This is a lightweight approach appropriate for a demo with very few
data points.  With a real dataset you would fine-tune more aggressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Literal

from src.losses import compute_alignment_loss, retrieval_accuracy


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    A small MLP that projects embeddings from dim_in to dim_out.

    Architecture:  Linear -> ReLU -> Linear
    If dim_in == dim_out this acts as a learned nonlinear transform in
    the same space, which is enough for our alignment demo.
    """

    def __init__(self, dim_in: int, dim_hidden: int = 256, dim_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize."""
        projected = self.net(x)
        return F.normalize(projected, dim=-1)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_alignment(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    loss_mode: Literal["cosine", "contrastive"] = "cosine",
    temperature: float = 0.07,
    dim_hidden: int = 256,
    dim_out: int = 256,
    device: str = "cpu",
    output_dir: str = "outputs",
    seed: int = 42,
) -> dict:
    """
    Train projection heads to align image and text embeddings.

    Args:
        image_embeddings:  [N, D] frozen CLIP image embeddings (detached).
        text_embeddings:   [N, D] frozen CLIP text embeddings (detached).
        epochs:            Number of training iterations.
        lr:                Learning rate.
        loss_mode:         'cosine' or 'contrastive'.
        temperature:       Temperature for contrastive loss.
        dim_hidden:        Hidden dimension in projection head MLP.
        dim_out:           Output dimension of projection heads.
        device:            'cpu' or 'cuda'.
        output_dir:        Where to save plots.
        seed:              Random seed for reproducibility.

    Returns:
        Dictionary with training results:
          - 'image_proj_head': trained image projection head
          - 'text_proj_head':  trained text projection head
          - 'losses':          list of per-epoch losses
          - 'sim_before':      similarity matrix before training
          - 'sim_after':       similarity matrix after training
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, D = image_embeddings.shape
    assert text_embeddings.shape == (N, D), (
        f"Shape mismatch: images {image_embeddings.shape} vs texts {text_embeddings.shape}"
    )

    # Detach and move to device — these are frozen inputs
    img_emb = image_embeddings.detach().to(device)
    txt_emb = text_embeddings.detach().to(device)

    # --- Similarity BEFORE training ---
    sim_before = (img_emb @ txt_emb.T).cpu()
    print("\n" + "=" * 60)
    print("SIMILARITY MATRIX — BEFORE TRAINING")
    print("=" * 60)
    _print_sim_matrix(sim_before)

    acc_before = retrieval_accuracy(img_emb, txt_emb)
    print(f"  Retrieval accuracy before: i2t={acc_before['i2t_acc']:.2f}  t2i={acc_before['t2i_acc']:.2f}")

    # --- Build projection heads ---
    img_proj = ProjectionHead(D, dim_hidden, dim_out).to(device)
    txt_proj = ProjectionHead(D, dim_hidden, dim_out).to(device)

    optimizer = torch.optim.Adam(
        list(img_proj.parameters()) + list(txt_proj.parameters()),
        lr=lr,
    )

    # --- Training loop ---
    losses: List[float] = []
    print(f"\n[train] Starting training: {epochs} epochs, lr={lr}, loss={loss_mode}")
    print("-" * 60)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Project frozen embeddings through trainable heads
        img_projected = img_proj(img_emb)   # [N, dim_out], normalized
        txt_projected = txt_proj(txt_emb)   # [N, dim_out], normalized

        loss = compute_alignment_loss(
            img_projected, txt_projected,
            mode=loss_mode,
            temperature=temperature,
            verbose=False,  # too noisy every epoch
        )

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Print every 10% of epochs, plus first and last
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d}/{epochs}  loss={loss_val:.6f}")

    # --- Similarity AFTER training ---
    img_proj.eval()
    txt_proj.eval()
    with torch.no_grad():
        img_final = img_proj(img_emb)
        txt_final = txt_proj(txt_emb)

    sim_after = (img_final @ txt_final.T).cpu()
    print("\n" + "=" * 60)
    print("SIMILARITY MATRIX — AFTER TRAINING")
    print("=" * 60)
    _print_sim_matrix(sim_after)

    acc_after = retrieval_accuracy(img_final, txt_final)
    print(f"  Retrieval accuracy after:  i2t={acc_after['i2t_acc']:.2f}  t2i={acc_after['t2i_acc']:.2f}")

    # --- Improvement summary ---
    diag_before = sim_before.diag().mean().item()
    diag_after = sim_after.diag().mean().item()
    print(f"\n[train] Matched-pair mean similarity: {diag_before:.4f} -> {diag_after:.4f}")
    print(f"[train] Final loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # --- Save plots ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _save_loss_curve(losses, output_dir)
    _save_similarity_heatmap(sim_before, "Similarity BEFORE Training", f"{output_dir}/sim_before.png")
    _save_similarity_heatmap(sim_after, "Similarity AFTER Training", f"{output_dir}/sim_after.png")

    return {
        "image_proj_head": img_proj,
        "text_proj_head": txt_proj,
        "losses": losses,
        "sim_before": sim_before,
        "sim_after": sim_after,
        "acc_before": acc_before,
        "acc_after": acc_after,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_sim_matrix(sim: torch.Tensor) -> None:
    """Pretty-print a similarity matrix."""
    N = sim.shape[0]
    header = "       " + "  ".join(f"img_{j:2d}" for j in range(N))
    print(header)
    for i in range(N):
        row = "  ".join(f"{sim[i, j].item():+.4f}" for j in range(N))
        marker = " <-- matched" if N > 0 else ""
        print(f"  txt_{i}: {row}")


def _save_loss_curve(losses: List[float], output_dir: str) -> None:
    """Save a plot of training loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Alignment Training Loss")
    ax.grid(True, alpha=0.3)
    path = f"{output_dir}/training_loss.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Loss curve saved to {path}")


def _save_similarity_heatmap(
    sim: torch.Tensor, title: str, path: str
) -> None:
    """Save a heatmap of the similarity matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    N = sim.shape[0]
    im = ax.imshow(sim.numpy(), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f"img_{i}" for i in range(N)])
    ax.set_yticklabels([f"txt_{i}" for i in range(N)])
    ax.set_title(title)

    # Annotate cells with values
    for i in range(N):
        for j in range(N):
            val = sim[i, j].item()
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Heatmap saved to {path}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 5, 512
    img_emb = F.normalize(torch.randn(N, D), dim=-1)
    txt_emb = F.normalize(torch.randn(N, D), dim=-1)

    result = train_alignment(img_emb, txt_emb, epochs=100, loss_mode="cosine")
    print(f"\nLoss went from {result['losses'][0]:.4f} to {result['losses'][-1]:.4f}")

"""
Tests for alignment loss functions.

Validates that cosine and contrastive losses behave correctly:
  - scalar output
  - correct matrix shapes
  - perfect alignment => low loss
  - random alignment => higher loss
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn.functional as F

from src.losses import compute_alignment_loss, retrieval_accuracy


@pytest.fixture
def random_embeddings():
    """Generate random normalized embeddings for testing."""
    torch.manual_seed(123)
    N, D = 6, 64
    img = F.normalize(torch.randn(N, D), dim=-1)
    txt = F.normalize(torch.randn(N, D), dim=-1)
    return img, txt


@pytest.fixture
def perfect_embeddings():
    """Generate perfectly aligned embeddings (identical pairs)."""
    torch.manual_seed(456)
    N, D = 4, 64
    emb = F.normalize(torch.randn(N, D), dim=-1)
    return emb.clone(), emb.clone()


class TestCosineLoss:

    def test_is_scalar(self, random_embeddings):
        img, txt = random_embeddings
        loss = compute_alignment_loss(img, txt, mode="cosine", verbose=False)
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        print(f"  Cosine loss (random): {loss.item():.4f}")

    def test_perfect_alignment_low_loss(self, perfect_embeddings):
        img, txt = perfect_embeddings
        loss = compute_alignment_loss(img, txt, mode="cosine", verbose=False)
        assert loss.item() < 0.01, f"Perfect alignment should have ~0 loss, got {loss.item():.4f}"
        print(f"  Cosine loss (perfect): {loss.item():.6f}")

    def test_random_higher_than_perfect(self, random_embeddings, perfect_embeddings):
        img_r, txt_r = random_embeddings
        img_p, txt_p = perfect_embeddings
        loss_random = compute_alignment_loss(img_r, txt_r, mode="cosine", verbose=False)
        loss_perfect = compute_alignment_loss(img_p, txt_p, mode="cosine", verbose=False)
        assert loss_random > loss_perfect, "Random pairs should have higher loss than perfect"

    def test_is_differentiable(self, random_embeddings):
        img, txt = random_embeddings
        img = img.clone().requires_grad_(True)
        loss = compute_alignment_loss(img, txt, mode="cosine", verbose=False)
        loss.backward()
        assert img.grad is not None, "Loss should be differentiable w.r.t. image embeddings"


class TestContrastiveLoss:

    def test_is_scalar(self, random_embeddings):
        img, txt = random_embeddings
        loss = compute_alignment_loss(img, txt, mode="contrastive", verbose=False)
        assert loss.dim() == 0
        print(f"  Contrastive loss (random): {loss.item():.4f}")

    def test_perfect_alignment_low_loss(self, perfect_embeddings):
        img, txt = perfect_embeddings
        loss = compute_alignment_loss(img, txt, mode="contrastive", verbose=False)
        # With temperature=0.07 and perfect match, loss should be very low
        print(f"  Contrastive loss (perfect): {loss.item():.4f}")
        assert loss.item() < 1.0, f"Perfect alignment should have low contrastive loss"

    def test_is_differentiable(self, random_embeddings):
        img, txt = random_embeddings
        img = img.clone().requires_grad_(True)
        loss = compute_alignment_loss(img, txt, mode="contrastive", verbose=False)
        loss.backward()
        assert img.grad is not None

    def test_invalid_mode_raises(self, random_embeddings):
        img, txt = random_embeddings
        with pytest.raises(ValueError, match="Unknown loss mode"):
            compute_alignment_loss(img, txt, mode="invalid", verbose=False)


class TestShapeChecks:

    def test_mismatched_count_raises(self):
        img = F.normalize(torch.randn(3, 64), dim=-1)
        txt = F.normalize(torch.randn(4, 64), dim=-1)
        with pytest.raises(AssertionError):
            compute_alignment_loss(img, txt, verbose=False)

    def test_1d_raises(self):
        img = F.normalize(torch.randn(64), dim=-1)
        txt = F.normalize(torch.randn(64), dim=-1)
        with pytest.raises(AssertionError):
            compute_alignment_loss(img, txt, verbose=False)


class TestRetrievalAccuracy:

    def test_perfect_accuracy(self, perfect_embeddings):
        img, txt = perfect_embeddings
        acc = retrieval_accuracy(img, txt)
        assert acc["i2t_acc"] == 1.0
        assert acc["t2i_acc"] == 1.0
        print(f"  Perfect accuracy: {acc}")

    def test_returns_dict(self, random_embeddings):
        img, txt = random_embeddings
        acc = retrieval_accuracy(img, txt)
        assert "i2t_acc" in acc
        assert "t2i_acc" in acc
        assert 0.0 <= acc["i2t_acc"] <= 1.0
        assert 0.0 <= acc["t2i_acc"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

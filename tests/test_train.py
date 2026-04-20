"""
Tests for the alignment training loop.

Verifies that:
  - loss decreases during training
  - matched-pair similarities improve
  - projection heads produce correct output shapes
  - training artifacts (plots) are saved
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn.functional as F

from src.train import train_alignment, ProjectionHead


class TestProjectionHead:

    def test_output_shape(self):
        head = ProjectionHead(512, 256, 256)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"

    def test_output_normalized(self):
        head = ProjectionHead(512, 256, 256)
        x = torch.randn(4, 512)
        out = head(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_is_differentiable(self):
        head = ProjectionHead(64, 32, 32)
        x = torch.randn(2, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestTrainAlignment:

    @pytest.fixture
    def training_result(self, tmp_path):
        """Train on random embeddings and return results."""
        torch.manual_seed(42)
        N, D = 6, 512
        img = F.normalize(torch.randn(N, D), dim=-1)
        txt = F.normalize(torch.randn(N, D), dim=-1)

        result = train_alignment(
            img, txt,
            epochs=100,
            lr=1e-3,
            loss_mode="cosine",
            output_dir=str(tmp_path),
            seed=42,
        )
        return result, tmp_path

    def test_loss_decreases(self, training_result):
        result, _ = training_result
        losses = result["losses"]
        # Compare first 10% average to last 10% average
        n = len(losses)
        early = sum(losses[:n // 10]) / max(1, n // 10)
        late = sum(losses[-n // 10:]) / max(1, n // 10)
        print(f"  Early loss: {early:.4f}, Late loss: {late:.4f}")
        assert late < early, f"Loss should decrease: {early:.4f} -> {late:.4f}"

    def test_matched_similarity_improves(self, training_result):
        result, _ = training_result
        before_diag = result["sim_before"].diag().mean().item()
        after_diag = result["sim_after"].diag().mean().item()
        print(f"  Matched similarity: {before_diag:.4f} -> {after_diag:.4f}")
        assert after_diag > before_diag, (
            f"Matched similarity should improve: {before_diag:.4f} -> {after_diag:.4f}"
        )

    def test_similarity_matrices_correct_shape(self, training_result):
        result, _ = training_result
        N = 6
        assert result["sim_before"].shape == (N, N)
        assert result["sim_after"].shape == (N, N)

    def test_loss_curve_saved(self, training_result):
        _, tmp_path = training_result
        assert (tmp_path / "training_loss.png").exists()

    def test_heatmaps_saved(self, training_result):
        _, tmp_path = training_result
        assert (tmp_path / "sim_before.png").exists()
        assert (tmp_path / "sim_after.png").exists()

    def test_contrastive_mode_runs(self, tmp_path):
        """Verify contrastive loss mode also works end-to-end."""
        torch.manual_seed(99)
        N, D = 4, 512
        img = F.normalize(torch.randn(N, D), dim=-1)
        txt = F.normalize(torch.randn(N, D), dim=-1)

        result = train_alignment(
            img, txt,
            epochs=50,
            loss_mode="contrastive",
            output_dir=str(tmp_path / "contrastive"),
            seed=99,
        )
        assert len(result["losses"]) == 50
        assert result["losses"][-1] < result["losses"][0]
        print(f"  Contrastive loss: {result['losses'][0]:.4f} -> {result['losses'][-1]:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

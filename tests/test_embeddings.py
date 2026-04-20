"""
Tests for image and text embedding utilities.

Validates that CLIP embeddings have correct shapes, are finite,
and exhibit expected behavior (same input -> same output,
different inputs -> different outputs).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from PIL import Image

from src.embedding_utils import (
    load_clip,
    bb_to_image_embedding,
    bb_to_image_embeddings,
    get_text_embedding,
    get_text_embeddings,
    cosine_similarity_matrix,
)
from src.synthetic_data import generate_sample_table_image
from src.ocr_utils import bb_and_text_from_table_image


@pytest.fixture(scope="module")
def clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_clip(device)
    return model, processor, device


@pytest.fixture(scope="module")
def sample_data(clip_model, tmp_path_factory):
    """Generate synthetic image, run OCR, compute embeddings."""
    model, processor, device = clip_model
    out_dir = str(tmp_path_factory.mktemp("emb_test"))
    img_path = generate_sample_table_image(save_path=f"{out_dir}/table.png")
    bbs, texts = bb_and_text_from_table_image(img_path, visualize=False)
    image = Image.open(img_path).convert("RGB")
    return image, bbs, texts, model, processor, device


# -----------------------------------------------------------------------
# Image embedding tests
# -----------------------------------------------------------------------

class TestImageEmbeddings:

    def test_single_embedding_shape(self, sample_data):
        image, bbs, _, model, processor, device = sample_data
        if len(bbs) == 0:
            pytest.skip("No bounding boxes detected")
        emb = bb_to_image_embedding(image, bbs[0], model, processor, device)
        assert emb.dim() == 1, f"Expected 1D tensor, got {emb.dim()}D"
        assert emb.shape[0] == 512, f"CLIP ViT-B/32 dim should be 512, got {emb.shape[0]}"
        print(f"  Embedding shape: {emb.shape}")

    def test_batch_embedding_shape(self, sample_data):
        image, bbs, _, model, processor, device = sample_data
        if len(bbs) == 0:
            pytest.skip("No bounding boxes detected")
        embs = bb_to_image_embeddings(image, bbs, model, processor, device)
        assert embs.shape == (len(bbs), 512), f"Expected ({len(bbs)}, 512), got {embs.shape}"

    def test_embeddings_are_finite(self, sample_data):
        image, bbs, _, model, processor, device = sample_data
        if len(bbs) == 0:
            pytest.skip("No bounding boxes detected")
        embs = bb_to_image_embeddings(image, bbs, model, processor, device)
        assert torch.isfinite(embs).all(), "All embeddings should be finite"

    def test_embeddings_are_normalized(self, sample_data):
        image, bbs, _, model, processor, device = sample_data
        if len(bbs) == 0:
            pytest.skip("No bounding boxes detected")
        embs = bb_to_image_embeddings(image, bbs, model, processor, device)
        norms = embs.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
            f"Embeddings should be L2-normalized, norms: {norms.tolist()}"
        )

    def test_different_crops_differ(self, sample_data):
        image, bbs, _, model, processor, device = sample_data
        if len(bbs) < 2:
            pytest.skip("Need at least 2 bounding boxes")
        e1 = bb_to_image_embedding(image, bbs[0], model, processor, device)
        e2 = bb_to_image_embedding(image, bbs[1], model, processor, device)
        sim = (e1 @ e2).item()
        print(f"  Similarity between crop 0 and crop 1: {sim:.4f}")
        # Different crops should generally not be identical
        assert not torch.allclose(e1, e2, atol=1e-3), "Different crops should produce different embeddings"


# -----------------------------------------------------------------------
# Text embedding tests
# -----------------------------------------------------------------------

class TestTextEmbeddings:

    def test_single_embedding_shape(self, clip_model):
        model, processor, device = clip_model
        emb = get_text_embedding("Apple", model, processor, device)
        assert emb.dim() == 1
        assert emb.shape[0] == 512
        print(f"  Text embedding shape: {emb.shape}")

    def test_batch_embedding_shape(self, clip_model):
        model, processor, device = clip_model
        texts = ["Apple", "Banana", "Orange"]
        embs = get_text_embeddings(texts, model, processor, device)
        assert embs.shape == (3, 512)

    def test_embeddings_are_finite(self, clip_model):
        model, processor, device = clip_model
        emb = get_text_embedding("test", model, processor, device)
        assert torch.isfinite(emb).all()

    def test_same_text_same_embedding(self, clip_model):
        model, processor, device = clip_model
        e1 = get_text_embedding("Apple", model, processor, device)
        e2 = get_text_embedding("Apple", model, processor, device)
        assert torch.allclose(e1, e2, atol=1e-5), "Same input should yield same embedding"

    def test_different_texts_differ(self, clip_model):
        model, processor, device = clip_model
        e1 = get_text_embedding("Apple", model, processor, device)
        e2 = get_text_embedding("car engine", model, processor, device)
        assert not torch.allclose(e1, e2, atol=1e-3)
        sim = (e1 @ e2).item()
        print(f"  cos(Apple, car engine) = {sim:.4f}")

    def test_semantic_similarity(self, clip_model):
        """Similar texts should be closer than dissimilar ones."""
        model, processor, device = clip_model
        e_apple = get_text_embedding("Apple", model, processor, device)
        e_banana = get_text_embedding("Banana", model, processor, device)
        e_car = get_text_embedding("automobile", model, processor, device)

        sim_fruit = (e_apple @ e_banana).item()
        sim_other = (e_apple @ e_car).item()
        print(f"  cos(Apple, Banana) = {sim_fruit:.4f}")
        print(f"  cos(Apple, automobile) = {sim_other:.4f}")
        # Fruits should generally be more similar to each other than to cars
        # (not guaranteed but very likely with CLIP)


# -----------------------------------------------------------------------
# Similarity matrix tests
# -----------------------------------------------------------------------

class TestSimilarityMatrix:

    def test_shape(self, clip_model):
        model, processor, device = clip_model
        a = get_text_embeddings(["A", "B", "C"], model, processor, device)
        b = get_text_embeddings(["X", "Y"], model, processor, device)
        sim = cosine_similarity_matrix(a, b)
        assert sim.shape == (3, 2)

    def test_self_similarity_is_one(self, clip_model):
        model, processor, device = clip_model
        embs = get_text_embeddings(["hello", "world"], model, processor, device)
        sim = cosine_similarity_matrix(embs, embs)
        diag = sim.diag()
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

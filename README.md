# Table-Text Alignment Project

**Aligns OCR-extracted text from table images with corresponding visual regions inside a VLM's representation space, providing semantic anchors that improve VLM reasoning on table content.**

This is a research project under [Haz Sameen Shahgir](https://github.com/Patchwork53), building on the findings from [VLMs Need Words](https://arxiv.org/abs/2604.02486) and [LatentLens](https://arxiv.org/abs/2602.00462).

## Project Goal

VLMs route visual reasoning through language — they perform well when they can name what they see, but fail on unnamed entities (see VLMs Need Words). Table cells containing arbitrary values ("$2.47", "SKU-8827") are effectively unnamed entities. This project adds an **alignment loss** during VLM fine-tuning that pairs OCR-extracted text with corresponding image regions, giving the model explicit semantic anchors for every cell.

### Pipeline

```python
# bb_to_image_embeddings()  → [DONE] VLM hidden state extraction (reference code)
# bb_and_text_from_table_image()  → [DONE] OCR with EasyOCR
# get_text_embedding()  → [In progress] LatentLens-inspired experimentation

bbs, texts = bb_and_text_from_table_image(image)

alignment_loss = 0
for bb, text in zip(bbs, texts):
    text_embedding = get_text_embedding(text)
    image_embedding = bb_to_image_embeddings(bb)
    loss = -cosine_sim(text_embedding, image_embedding)
    alignment_loss += loss

loss = task_loss + w * alignment_loss
loss.backwards()
```

## Project Structure

```
├── src/
│   ├── ocr_utils.py          # bb_and_text_from_table_image() — EasyOCR detection
│   ├── synthetic_data.py     # Generate sample table images for testing
│   ├── embedding_utils.py    # CLIP-based embeddings (prototype/baseline)
│   ├── losses.py             # Alignment loss functions
│   ├── train.py              # Prototype training loop (CLIP-based)
│   └── demo.py               # End-to-end demo script
├── tests/                    # pytest suite (40 tests)
├── docs/
│   └── meeting_prep_apr20.md # Meeting prep notes
├── requirements.txt
└── README.md
```

### Core deliverable: `bb_and_text_from_table_image()`

Located in `src/ocr_utils.py`. Uses EasyOCR to:
- Detect text regions in a table image
- Return bounding boxes as `(x1, y1, x2, y2)` + extracted text strings
- Filter low-confidence and empty detections
- Optionally save annotated visualization

### Prototype: CLIP-based alignment demo

The files `embedding_utils.py`, `losses.py`, `train.py`, and `demo.py` implement a standalone prototype using CLIP for embeddings and trainable projection heads. This demonstrates the alignment concept but uses a different architecture than the research pipeline (which uses Qwen3-VL hidden states).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run CLIP-based prototype demo
python -m src.demo

# Run tests
pytest tests/ -v -s
```

## Key References

- [VLMs Need Words](https://arxiv.org/abs/2604.02486) — Shahgir et al. VLMs fail on unnamed visual entities; semantic anchors fix this
- [LatentLens](https://arxiv.org/abs/2602.00462) — Krojer et al. Training-free interpretability revealing that mid-layer VLM hidden states are shared semantic spaces
- [Reference code](https://github.com/Patchwork53/VLMs-Need-Words-Public/blob/main/shape_correspond/rep_qwen_squiggles.py) — `bb_to_image_embeddings` implementation using Qwen3-VL

## Status

- [x] Lit review — cell-level table alignment appears novel (closest: DoCo, CVPR 2024)
- [x] `bb_and_text_from_table_image()` — implemented and tested with EasyOCR
- [x] Understand `bb_to_image_embeddings()` from reference code
- [ ] `get_text_embedding()` — needs LatentLens-inspired experimentation
- [ ] Integration with Qwen3-VL pipeline
- [ ] Evaluation on real table datasets

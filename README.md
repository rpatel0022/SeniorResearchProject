# Multimodal Table Alignment Pipeline

**This prototype aligns OCR-extracted text from table images with the corresponding visual regions so that text and image patches representing the same content become close in embedding space.**

## What It Does

Given a table image, this pipeline:

1. Detects text regions using OCR (EasyOCR) and extracts bounding boxes + text strings.
2. Encodes each cropped image region using CLIP's vision encoder.
3. Encodes each OCR text string using CLIP's text encoder.
4. Computes an alignment score (cosine similarity) between matched image-text pairs.
5. Trains lightweight projection heads to improve alignment between matched pairs.
6. Visualizes everything: annotated OCR output, similarity heatmaps, training curves.

## Architecture Overview

```
Table Image
    │
    ├──> EasyOCR ──> Bounding Boxes + Text Strings
    │                    │                │
    │                    ▼                ▼
    │            CLIP Vision         CLIP Text
    │             Encoder             Encoder
    │                │                │
    │                ▼                ▼
    │         Image Embeddings   Text Embeddings
    │              [N, 512]        [N, 512]
    │                │                │
    │                ▼                ▼
    │         ┌─────────────────────────┐
    │         │  Trainable Projection   │
    │         │      Heads (MLP)        │
    │         └─────────────────────────┘
    │                │                │
    │                ▼                ▼
    │         Projected Image    Projected Text
    │           [N, 256]           [N, 256]
    │                │                │
    │                └──── Cosine ────┘
    │                    Similarity
    │                       │
    │                       ▼
    │               Alignment Loss
    │          (cosine or contrastive)
    │                       │
    │                  Backprop to
    │              projection heads
    └──────────────────────────────────
```

## Project Structure

```
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── synthetic_data.py     # Generate sample table images
│   ├── ocr_utils.py          # OCR detection and visualization
│   ├── embedding_utils.py    # CLIP image/text encoding
│   ├── losses.py             # Cosine and contrastive loss functions
│   ├── train.py              # Training loop with projection heads
│   └── demo.py               # End-to-end demo script
├── tests/
│   ├── test_ocr.py           # OCR function tests
│   ├── test_embeddings.py    # Embedding function tests
│   ├── test_loss.py          # Loss function tests
│   └── test_train.py         # Training loop tests
└── outputs/                  # Generated artifacts (created at runtime)
```

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Note:** The first run will download the CLIP model (~600 MB) and EasyOCR models (~100 MB). Subsequent runs use cached models.

## Running the Demo

```bash
# Basic demo with synthetic table image
python -m src.demo

# Custom options
python -m src.demo --epochs 300 --loss_type contrastive

# Use your own table image
python -m src.demo --image_path path/to/table.png

# All options
python -m src.demo --help
```

### Command-Line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--image_path` | None | Path to a table image |
| `--use_synthetic` | True | Generate a synthetic table image |
| `--epochs` | 200 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--loss_type` | cosine | Loss function: `cosine` or `contrastive` |
| `--output_dir` | outputs | Directory for saved artifacts |

## Running Tests

```bash
# Run all tests
pytest tests/ -v -s

# Run individual test files
pytest tests/test_ocr.py -v -s
pytest tests/test_embeddings.py -v -s
pytest tests/test_loss.py -v -s
pytest tests/test_train.py -v -s
```

## Expected Outputs

After running `python -m src.demo`, you should see:

### Console Output
- Number of OCR detections with bounding boxes and text
- CLIP embedding shapes and sample values
- Similarity matrix before training (text rows x image columns)
- Epoch-by-epoch training loss
- Similarity matrix after training
- Retrieval accuracy before and after

### Saved Files (in `outputs/`)
| File | Description |
|------|-------------|
| `sample_table.png` | The synthetic table image |
| `ocr_annotated.png` | Table image with bounding boxes drawn |
| `crops/crop_0.png` ... | Individual cropped text regions |
| `sim_before.png` | Heatmap of similarity matrix before training |
| `sim_after.png` | Heatmap of similarity matrix after training |
| `training_loss.png` | Loss curve over training epochs |

### What "Good" Looks Like
- **Before training:** Similarity matrix diagonal values are moderate and not clearly higher than off-diagonal.
- **After training:** Diagonal values (matched pairs) should be noticeably higher than off-diagonal values.
- **Loss curve:** Should decrease smoothly from initial value.
- **Retrieval accuracy:** Should improve (often reaching 1.0 on the small synthetic dataset).

## Limitations

This is a research prototype, not a production system. Key limitations:

- **OCR noise:** EasyOCR may miss cells, merge adjacent text, or produce errors depending on font, resolution, and layout.
- **Tiny dataset:** Training on a single image with ~10 text regions is far from realistic. The training loop demonstrates the concept but would need hundreds/thousands of examples for real generalization.
- **Projection-head-only training:** The CLIP backbone is frozen. We only train lightweight MLP heads on top, which limits how much the alignment can improve.
- **No table structure modeling:** The pipeline treats each text region independently. It does not model rows, columns, headers, or cell relationships.
- **Not a document understanding system:** This does not parse table semantics, extract key-value pairs, or understand document layout beyond OCR-level detection.
- **Cosine similarity baseline:** CLIP embeddings are already somewhat aligned for natural image-text pairs, but OCR text fragments (like "$2" or "Qty") are not typical CLIP training data.

## Suggestions for Future Improvements

- **More training data:** Use a dataset of real table images (e.g., PubTabNet, TableBank) instead of synthetic ones.
- **Fine-tune CLIP:** Unfreeze some CLIP layers for deeper adaptation.
- **Table structure awareness:** Use table detection models (e.g., DETR-based) to identify rows/columns before alignment.
- **Better OCR:** Try PaddleOCR or cloud OCR APIs for higher accuracy.
- **Cross-attention alignment:** Replace projection heads with a cross-attention mechanism between image patches and text tokens.
- **Evaluation metrics:** Add NDCG, MRR, and Recall@K for proper retrieval evaluation.
- **Data augmentation:** Apply image augmentations (rotation, noise, blur) to improve robustness.

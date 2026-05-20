# Table-Text Alignment for VLMs

**Cell-level alignment loss that pushes a VLM's visual representations of table regions toward the corresponding OCR text — giving the model semantic anchors for every cell.**

> **Headline result (run4, May 2026):** with the `lm_head_bow` auxiliary loss (frozen-LM-head probe per Sameen's "VLM Needs Words"-inspired proposal) at layer 26 of Qwen3-VL-2B-Instruct, the alignment objective **is learnable** (NLL 10.68 → 6.03, well below the random floor of 11.93) — but downstream **EM on CoSyn-400K table QA does not improve** (0.485 vs 0.495 task-only baseline, within statistical noise at n=200). The auxiliary objective is achievable but appears orthogonal to the downstream task on this dataset/model. See [Step 3](#step-3-qwen3-vl-alignment--implementation--results) for the full table and discussion.

---

## How It Works

```
Step 1 — OCR: Find every cell in the table image (using EasyOCR)
┌─────────────────────────────────────────────────┐
│  Table Image                                    │
│       │                                         │
│       ▼                                         │
│  bb_and_text_from_table_image()                 │
│       │                                         │
│       ├──► bounding boxes  (where each cell is) │
│       └──► OCR text        (what each cell says)│
└─────────────────────────────────────────────────┘

Step 2 — Embed: Convert each cell into a vector the model understands
┌─────────────────────────────────────────────────┐
│  bounding boxes ──► bb_to_image_embeddings()    │
│                     (extract what the VLM        │
│                      "sees" in that region)      │
│                                                  │
│  OCR text ────────► get_text_embedding()         │
│                     (extract what the VLM        │
│                      "understands" from the text)│
└──────────────────────────────────────────────────┘

Step 3 — Align: Train the model so the two match up
┌──────────────────────────────────────────────────┐
│  For each cell:                                  │
│    compare image embedding vs text embedding     │
│    using cosine similarity                       │
│                                                  │
│  alignment_loss = how far apart they are         │
│  total_loss = task_loss + w * alignment_loss     │
│                                                  │
│  Backpropagate → model learns to connect what    │
│  it sees in a cell with what the text says       │
└──────────────────────────────────────────────────┘
```

---

## Step 1: OCR Detection — `bb_and_text_from_table_image()`

Implemented in `src/ocr_utils.py`. Takes any table image, detects every text region using EasyOCR, and returns bounding boxes `(x1, y1, x2, y2)` paired with the OCR'd text string for each cell. This is the first piece of the pipeline — referred to as `bb_and_text_from_table_image()` in Sameen's pseudocode.

### Synthetic table

```bash
python -m src.ocr_utils
```

**Input** — a generated table with known content:

![Synthetic table](assets/synthetic_table.png)

**Output** — every cell detected, labeled with its index and extracted text:

![Annotated synthetic table](assets/synthetic_annotated.png)

11 detections. All headers (Item, Price, Qty), row labels (Apple, Banana, Orange), and values found.

### Real-world tables

Also tested on real table images downloaded from the internet to verify the function works beyond synthetic data.

```bash
python -m src.ocr_utils test_images/income_guidelines.png
```

![Income guidelines table](assets/income_guidelines.png)

![Annotated income table](assets/income_annotated.png)

27 detections — headers, row numbers, and dollar amounts all found. One OCR quirk: EasyOCR reads "$" as "S" (e.g. "$3,332" → "S3,332"). This is a known EasyOCR limitation with dollar sign characters.

```bash
python -m src.ocr_utils test_images/grade_distribution.png
```

![Grade distribution table](assets/grade_distribution.png)

![Annotated grade table](assets/grade_annotated.png)

11 detections on a clean simple table.

---

## Step 2: CLIP Prototype — Validating the Alignment Idea

### Why build this?

The real pipeline will use Qwen3-VL (a large model that requires GPU infrastructure and integration with Sameen's codebase). Before going through all of that, a quick prototype was built using [CLIP](https://openai.com/research/clip) (a smaller, easy-to-run image-text model by OpenAI) to answer a simple question: **can a model be trained to match a table cell's image with its text?** If the answer is no, there's no point setting up Qwen3-VL.

### What CLIP does here

CLIP can convert both images and text into numerical vectors (embeddings). So for each cell detected by the OCR in Step 1, two embeddings are produced:
- **Image embedding** — crop the cell region from the table image, pass it through CLIP
- **Text embedding** — take the OCR text string (e.g. "Apple"), pass it through CLIP

Now both the image and the text are represented as numbers, and cosine similarity can be used to measure how "close" they are.

### The problem

Out of the box, CLIP scores every image-text pair at roughly ~0.22. It can't tell that the image crop of "Apple" should match the text "Apple" any more than it matches "Banana." This makes sense — CLIP was trained on photos and captions, not tiny crops of table cells.

### The fix

Two small trainable layers (projection heads) were added on top of the frozen CLIP embeddings and trained with a simple rule: **the image of cell N should match the text of cell N, and not match anything else.** After ~20 rounds of training, matched pairs hit 0.999 similarity.

### Results

**Before training** — everything looks the same (~0.22), the model can't distinguish any pairs:

![Similarity before](assets/sim_before.png)

**After training** — matched pairs (the diagonal) are now clearly aligned:

![Similarity after](assets/sim_after.png)

**Training loss:**

![Training loss](assets/training_loss.png)

| Metric | Before | After |
|--------|--------|-------|
| Matched-pair similarity | 0.24 | 0.999 |
| Retrieval accuracy (image→text) | 55% | 73% |
| Retrieval accuracy (text→image) | 64% | 82% |
| Loss | 1.10 | 0.001 |

### Conclusion

The alignment concept is validated — a model **can** be trained to connect table cell images with their corresponding text. This prototype is not the final approach though. CLIP uses two separate encoders for images and text, which is why projection heads were needed to bridge them. Qwen3-VL (the real model, from Sameen's [codebase](https://github.com/Patchwork53/VLMs-Need-Words-Public/blob/main/shape_correspond/rep_qwen_squiggles.py)) is a single model where images and text already share the same internal representation space — so the alignment loss can be applied directly without needing extra projection heads.

Implemented in `src/embedding_utils.py`, `src/losses.py`, `src/train.py`, and `src/demo.py`.

---

## Step 3: Qwen3-VL Alignment — Implementation & Results

With the CLIP prototype confirming the alignment idea works in principle, the real pipeline was built on Qwen3-VL-2B-Instruct using CoSyn-400K (table split). This section documents the full setup, the diagnostic work that informed the design, and the result we landed on.

### Data preprocessing — `src/preprocess_cosyn.py`

For each CoSyn-table image:
1. Run EasyOCR → bounding boxes + OCR text per cell.
2. Pass each cell's OCR text through the Qwen3-VL **language model only** (no image) → extract the hidden state at every LM layer (28 total).
3. Save bboxes + texts + per-layer text hidden states + image hash to `data/preprocessed/{idx}.pt`.

These `.pt` files become the alignment targets at training time: for each visible cell in an image, we know which pixel region it occupies, what text it contains, and what the LM's hidden representation of that text looks like at every layer. Run once on a GPU to populate `data/preprocessed/` (1,050 images for the experiments below).

### Training setup — `src/train_qwen_cosyn.py`

Per [Sameen's design directives](https://github.com/Patchwork53/VLMs-Need-Words-Public/blob/main/shape_correspond/rep_qwen_squiggles.py):

- **Vision encoder + MLP projector are frozen**; only the LM layers are trainable.
- Training is on the task objective (next-token CE on table-QA answers) **plus** an auxiliary alignment loss applied to a single intermediate LM layer via a forward hook.
- `loss = task_loss + alignment_loss_weight * alignment_loss`

The alignment loss has two implemented formulations (`src/losses.py`):

- **`mse`** (initial spec): mean-pool visual tokens per cell at layer L, compare against the precomputed text hidden state with raw L2.
- **`lm_head_bow`** (current spec, per Sameen 2026-05-07): mean-pool visual tokens per cell at layer L, project through the **frozen** LM head, and compute per-token CE against the OCR tokens of that cell (bag-of-words within the cell). The "VLM Needs Words"-style frozen probe — push intermediate visual reps to be linguistically decodable through the unchanged LM head.

OCR bboxes are mapped to Qwen3-VL visual token indices via `src/token_map.py` using the patch grid math, so the full image is encoded once (no per-cell crops) and the per-cell visual representation is just a mean-pool of the visual tokens that fall inside that cell's bbox.

### Layer sweep — picking where to attach the auxiliary loss

The `lm_head_bow` formulation assumes intermediate LM layers encode visual reps that the frozen LM head can partially decode. To find which layer (`scripts/calibrate_alignment.py --alignment-loss-mode lm_head_bow --alignment-layer L`), the per-cell NLL was measured on 20 CoSyn-table samples (~191 cells) with the **untrained** model:

| Layer | mean NLL | min  | Δ vs random (11.93) |
|------:|---------:|-----:|---------------:|
| 8     | 14.94    | 9.08 | +3.01 |
| 12    | 13.09    | 7.43 | +1.16 |
| 16    | 12.23    | 5.68 | +0.30 |
| 20    | 12.59    | 7.84 | +0.66 |
| 24    | 11.62    | 5.94 | −0.31 |
| **26**| **10.54**| 4.55 | **−1.39** |
| 27    | 32.60    | 2.23 | +20.67 |

Random NLL is `log(151643) ≈ 11.93`. Only **layer 26** has a meaningfully below-random baseline, meaning the frozen LM head can already decode layer-26 visual reps non-trivially. Layers 8–20 start at or above random, so attaching the loss there asks the model to learn the alignment from scratch. Layer 27 is anomalous because final hiddens encode next-token predictions rather than OCR content (the model's "what comes after this visual token" pathway happens to give some tokens very low NLL on coincidental matches).

### Training runs

Each run uses 200-item held-out eval (`scripts/eval_table_metrics.py`, deterministic slice), exact-match against gold-truth answers and token-level F1.

| Run | Alignment | Weight | Layer | Notes |
|---|---|---:|---:|---|
| Base | none | — | — | Zero-shot Qwen3-VL-2B-Instruct |
| Run1 | MSE (raw L2) | 1e-5 | 16 | Original spec, ran 2026-04-27 |
| Run2 | none (task-only) | 0 | — | Baseline for the auxiliary loss, ran 2026-04-29 |
| Run4 | lm_head_bow (frozen-probe CE) | 0.04 | 26 | `--aligned_only` filter, ran 2026-05-14 |

The `--aligned_only` flag was added because only 1,050 of ~416k expanded train items have matching `.pt` data (~2.3% sparsity). Without it, ~98% of micro-batches contribute zero alignment gradient. With it, every batch contributes (~9,330 items, every batch has alignment). Val split stays unfiltered so EM is comparable across runs.

### Results

| Run | EM (200) | Token F1 (200) |
|---|---:|---:|
| Base | 0.000 | 0.051 |
| Run1 (task + MSE, w=1e-5, layer 16) | 0.500 | 0.586 |
| Run2 (task-only) | 0.495 | 0.584 |
| **Run4 (task + lm_head_bow, w=0.04, layer 26, aligned_only)** | **0.485** | **0.573** |

**The auxiliary alignment loss is *learnable* but does not *improve* downstream EM.** Run4's alignment loss dropped from 10.68 (random) → 6.03 over 2 epochs (~4.7 nats below the random floor — a real signal the LM head is decoding the visual reps better than at init), but the held-out EM is within statistical noise of both run1 (MSE-aligned) and run2 (task-only). σ at n=200 is ~0.035, so the −0.015 EM gap between run4 and run1 is well inside one standard deviation.

Why this is a meaningful negative result: the model can do CoSyn table QA via a pathway that does **not** require layer-26 visual reps to be linguistically decodable. Pushing them to be decodable is achievable but orthogonal to the downstream task on this dataset/model. The headline finding for the auxiliary loss strategy is therefore: *learnable ≠ useful*.

Two confounds worth flagging in any follow-up:

1. **Smaller train set under `--aligned_only`.** Run4 saw 9,330 unique items × 3 epochs vs Run2's ~416k × 3 epochs. A clean A/B (run5 = task-only on the same 9,330 items) would isolate "alignment hurts" vs "smaller dataset hurts."
2. **Layer choice was optimized for the *probe*, not the *task*.** Layer 26 has the strongest frozen-decoding signal, but it's also where visual reps are already mostly linguistic. An earlier layer (e.g., 20–22) — where the visual→linguistic transition is mid-flight — might be more impactful for downstream EM even though the frozen probe is noisier there.

### Engineering notes from the experiments

- Run3 (2026-05-13, single-GPU `lm_head_bow` at layer 16, w=0.01) **died silently at 12h** because stdout was block-buffered through bash redirect and the final traceback never reached disk. The fix is `PYTHONUNBUFFERED=1` + `python -u` + `tee` in the launch path (`scripts/launch_align.sh`).
- The training script's prior loss-logging code accumulated per micro-batch but divided by `log_every` (counts optimizer steps), producing values 8× the true per-forward mean at `grad_accum_steps=8`. Fixed in the same commit as the safeguards. Older run logs (pre-fix) need mental rescaling.
- Multi-GPU dataset construction reproducibly OOMs at `--num_processes ≥ 2` during the qa_pair expansion phase. Stuck to single GPU.
- The full safeguard stack (mid-epoch `accelerator.save_state` + `--resume_from` + watchdog + line-buffered logs) was validated end-to-end via a smoke run that included a deliberate SIGTERM and resume from checkpoint. Training continues at the saved step with task-loss continuity.

Implemented in:
- `src/preprocess_cosyn.py` — OCR + per-layer text hidden state extraction
- `src/train_qwen_cosyn.py` — training loop, alignment hook, mid-epoch saves, resume
- `src/token_map.py` — OCR bbox → visual token index mapping
- `src/losses.py` — `compute_lm_head_bow_loss` (and the legacy MSE / cosine / contrastive losses from the CLIP prototype)
- `scripts/calibrate_alignment.py` — weight calibration + layer sweep diagnostic
- `scripts/eval_table_metrics.py` — EM/F1 eval on the 200-item held-out slice
- `scripts/launch_align.sh` — survivable-launch wrapper
- `scripts/watchdog.sh` — process + log watchdog

---

## Setup

```bash
git clone <this-repo>
cd SeniorResearchProject
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# OCR on synthetic table
python -m src.ocr_utils

# OCR on a real image
python -m src.ocr_utils path/to/table.png

# Full prototype pipeline (OCR → CLIP embeddings → alignment training)
python -m src.demo --epochs 50

# Tests (6 OCR tests + embedding/loss/training tests)
python -m pytest tests/ -v -s
```

## Project Structure

```
src/
├── ocr_utils.py          # bb_and_text_from_table_image() — OCR function (Step 1)
├── synthetic_data.py     # Generates sample table images for testing
├── embedding_utils.py    # CLIP embedding extraction (Step 2 prototype)
├── losses.py             # Alignment loss functions (cosine/contrastive/MSE/lm_head_bow)
├── train.py              # Projection head training loop (Step 2 prototype)
├── demo.py               # End-to-end demo tying Steps 1 + 2 together
├── preprocess_cosyn.py   # CoSyn-400K preprocessing → data/preprocessed/*.pt (Step 3)
├── token_map.py          # OCR bbox → Qwen3-VL visual token indices (Step 3)
└── train_qwen_cosyn.py   # Qwen3-VL training loop with auxiliary alignment loss (Step 3)

scripts/
├── calibrate_alignment.py   # Weight calibration + layer-sweep diagnostic
├── eval_table_metrics.py    # EM / token-F1 eval on 200-item held-out slice
├── launch_align.sh          # Survivable-launch wrapper (PYTHONUNBUFFERED + nohup + tee)
└── watchdog.sh              # PID + log-staleness watchdog with forensic alerts

tests/
├── test_ocr.py           # OCR detection tests
├── test_embeddings.py    # Embedding extraction tests
├── test_loss.py          # Loss computation tests
└── test_train.py         # Training loop tests
```

## Progress

**Done:**
- [x] OCR pipeline — `bb_and_text_from_table_image()` tested on synthetic + real-world images
- [x] CLIP-based proof-of-concept — standalone prototype proving alignment training works end-to-end
- [x] OCR → Qwen3-VL token index mapping (`src/token_map.py`)
- [x] CoSyn-400K preprocessing pipeline — produces per-image bboxes, OCR text, and per-layer text hidden states (`src/preprocess_cosyn.py`)
- [x] Qwen3-VL training loop with vision+MLP frozen, single-hook alignment loss (`src/train_qwen_cosyn.py`)
- [x] Auxiliary loss formulations — raw L2 (run1), `lm_head_bow` per Sameen's May 7 proposal (run4)
- [x] Three-eval comparison (base / task-only / task+align) on 200-item held-out
- [x] Layer-sweep diagnostic — identified layer 26 as the only Qwen3-VL-2B LM layer where the frozen LM head decodes visual reps non-trivially
- [x] Run4 with `lm_head_bow` at layer 26 — alignment loss learnable (10.68 → 6.03) but EM does not improve vs task-only baseline
- [x] Engineering safeguards — mid-epoch checkpointing, `--resume_from`, generic watchdog, line-buffered logs (post run3 silent-death)

**Open questions / candidate next steps:**
- [ ] Run5 = task-only on the 9,330 aligned items — isolates the dataset-size confound in the run4 vs run2 comparison
- [ ] Sweep alignment layer with `--aligned_only` (e.g., layers 20–22) — frozen-probe NLL ≠ task-helpfulness; an earlier layer might transfer to EM even with a noisier baseline
- [ ] Expand preprocessed alignment data 1,050 → 10k images — would let us re-run without `--aligned_only` and keep full-data fidelity

## References

- [VLMs Need Words](https://arxiv.org/abs/2604.02486) — Shahgir et al. Why VLMs fail on unnamed visual entities
- [LatentLens](https://arxiv.org/abs/2602.00462) — Krojer et al. Mid-layer hidden states as shared text-image spaces
- [bb_to_image_embeddings reference code](https://github.com/Patchwork53/VLMs-Need-Words-Public/blob/main/shape_correspond/rep_qwen_squiggles.py)

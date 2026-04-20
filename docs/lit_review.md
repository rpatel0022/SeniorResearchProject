## 1. Introduction

Vision-Language Models (VLMs) have achieved impressive results across multimodal tasks, yet they systematically struggle with **fine-grained visual understanding** -- particularly in structured content like tables. This literature review surveys the landscape of table understanding in Document AI, vision-language alignment techniques, and interpretability methods that inform the **Table-Text Alignment Project**: a contrastive learning approach that forces VLMs to align their visual representations of table cell regions with the semantic meaning of the text those cells contain.

This review covers three pillars:

1. The two core papers provided by the mentor (LatentLens and VLMs Need Words)
2. The broader field of table understanding and document AI
3. Contrastive learning and text-image alignment techniques relevant to the project

---

## 2. Core Papers

### 2.1 LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs

**Authors:** Krojer, Nayak, Manas, Adlakha, Elliott, Reddy, Mosbach (MILA & McGill)
**Year:** 2026 | **arXiv:** 2602.00462
**Code:** [github.com/McGill-NLP/latentlens](https://github.com/McGill-NLP/latentlens) | `pip install latentlens`

#### Problem

Why can LLMs process visual tokens so well after just a simple MLP projection from a vision encoder? Existing interpretability methods (LogitLens, EmbeddingLens) give misleadingly low interpretability scores because they compare visual tokens to **static** vocabulary embeddings rather than **contextualized** text representations.

#### Method

LatentLens introduces a two-phase approach:

1. **Corpus Encoding:** Encode ~3M Visual Genome captions through an LLM, storing contextualized token representations at 8 selected layers. Uses reservoir sampling (20 embeddings per vocabulary token per layer) and float8 compression.
2. **Nearest Neighbor Matching:** For a given visual token at any layer, compute cosine similarity against ALL stored contextualized text representations across ALL layers, returning top-k nearest neighbors with their source sentences.

The critical insight: the reference text layer can differ from the query visual layer (**cross-layer matching**).

#### Key Results

| Method         | % Visual Tokens Interpretable |
| -------------- | ----------------------------- |
| **LatentLens** | **72%**                       |
| EmbeddingLens  | 30%                           |
| LogitLens      | 23%                           |

Evaluated across 10 VLM configurations (3 LLMs x 3 vision encoders + Qwen2-VL-7B).

#### The "Mid-Layer Leap"

Visual tokens at the input layer (layer 0) align most strongly with **mid-layer** (layers 8-16) text representations, not input-level embeddings. This explains why simple projections work for multimodal adaptation.

#### Relevance to Table-Text Alignment

- **Direct methodological relevance:** The technique of extracting contextualized hidden states and comparing via cosine similarity is exactly what the pseudocode's `get_text_embedding()` function needs to do.
- **Key insight for the project:** Using contextualized embeddings (not static word embeddings) dramatically improves alignment quality. Table cell text like "Revenue: $5M" should be embedded in context, not as isolated tokens.
- **Cross-layer matching:** The optimal text layer for comparison may not be the same layer as the visual representation -- this is an important experimental variable.

---

### 2.2 VLMs Need Words: Vision Language Models Ignore Visual Detail In Favor of Semantic Anchors

**Authors:** Shahgir, Chen, Fu, Shayegani, Abu-Ghazaleh, Kementchedjhieva, Dong (UC Riverside, MBZUAI)
**Year:** 2026 | **arXiv:** 2604.02486
**Note:** This is the mentor's paper.

#### Problem

VLMs fail on tasks requiring fine-grained visual perception, even when their internal representations contain the necessary information. This is the "hidden-in-plain-sight" gap. **Why?**

#### Core Finding: The Nameability Gap

VLMs rely on **semantic anchoring** -- they can only reason about visual entities that map to known linguistic concepts. When something is "nameable" (e.g., "bicycle pedal"), the model converts vision to language and reasons verbally. When something is "unnameable" (e.g., an abstract shape), the model hallucinates textual descriptions, degrading performance.

**Evidence:**

- Qwen3VL-2B on shapes: Known entities 54.1% accuracy vs. Unknown 29.0% (direct answer)
- But representation probing shows 100% vs. 74.2% -- **the information IS there internally**

#### Chain-of-Thought Makes It Worse for Unknown Entities

- CoT helps named entities (+20.8%)
- CoT **hurts** unnamed entities (-19.4%) -- generates hallucinated descriptions

#### Teaching Names Closes the Gap

- Unknown shapes: 29.0% → 86.0% with ordinary object names
- Even random strings: 29.0% → 62.8%

#### Task-Specific Finetuning is Even Better

- 29.0% → 99.3% in-domain; transfers to mazes (99.0%), faces (+16%), semantic correspondence (+10.7%)

#### Relevance to Table-Text Alignment

This paper is the **motivation** for the alignment project:

- Tables contain both nameable and unnameable content
- The alignment loss effectively "teaches names" to table cell regions by giving explicit text-visual correspondence
- Internal representations are sufficient -- alignment helps the model use what it already knows

---

## 3. Table Understanding in Document AI

### 3.1 Core Tasks

| Task                                  | Description                                      | Key Challenge                                     |
| ------------------------------------- | ------------------------------------------------ | ------------------------------------------------- |
| **Table Detection (TD)**              | Locating table regions in document images        | Diverse table styles, embedded in complex layouts |
| **Table Structure Recognition (TSR)** | Identifying rows, columns, cells, spanning cells | Merged cells, irregular structures                |
| **Table Content Extraction (TCR)**    | Extracting text from detected cells              | OCR errors, mixed content types                   |
| **Table Question Answering (QA)**     | Answering NL questions about table content       | Multi-hop reasoning, aggregation                  |

### 3.2 Key Models

| Model                        | Year  | Venue    | Approach                        | Notable Feature                                       |
| ---------------------------- | ----- | -------- | ------------------------------- | ----------------------------------------------------- |
| **LayoutLM**                 | 2020  | KDD      | BERT + 2D position embeddings   | First to add spatial info to language models          |
| **LayoutLMv2**               | 2021  | ACL      | Multi-modal transformer         | **Text-Image Alignment (TIA)** pre-training objective |
| **LayoutLMv3**               | 2022  | ACM MM   | ViT patches + unified masking   | **Word-Patch Alignment (WPA)** objective              |
| **TableFormer**              | 2022  | CVPR     | CNN + dual transformer decoders | Joint logical + physical structure prediction         |
| **Table Transformer (TATR)** | 2022  | CVPR     | DETR-based detection            | Released with PubTables-1M dataset                    |
| **Donut**                    | 2022  | ECCV     | End-to-end, OCR-free            | Image-in, text-out, no separate OCR                   |
| **Pix2Struct**               | 2023  | ICML     | Screenshot parsing pre-training | Purely visual language understanding                  |
| **UDOP**                     | 2023  | CVPR     | Unified vision+text+layout      | Single model for understanding + generation           |
| **mPLUG-DocOwl**             | 2023+ | EMNLP'24 | Modularized MLLM                | Unified structure learning across doc types           |
| **TAPAS**                    | 2020  | ACL      | BERT + row/col embeddings       | Weakly supervised table QA                            |

### 3.3 Key Datasets

| Dataset          | Size         | Domain             | Task       | Cell BBoxes           | Cell Text |
| ---------------- | ------------ | ------------------ | ---------- | --------------------- | --------- |
| **PubTabNet**    | 568K tables  | Scientific papers  | TSR (HTML) | Yes (non-empty cells) | Yes       |
| **PubTables-1M** | 948K tables  | Scientific papers  | TD + TSR   | Yes (all cells)       | Via OCR   |
| **FinTabNet**    | 113K tables  | Financial reports  | TSR        | Yes                   | Yes       |
| **SciTSR**       | 15K tables   | Scientific (LaTeX) | TSR        | Partial               | Yes       |
| **TableBank**    | 417K tables  | Word + LaTeX docs  | TD + TSR   | No (table-level)      | No        |
| **WTQ**          | 22K QA pairs | Wikipedia          | Table QA   | N/A                   | Yes       |

### 3.4 Evaluation Metrics

- **TEDS** (Tree-Edit-Distance-based Similarity): Standard for TSR, captures structural errors
- **GriTS** (Grid Table Similarity): Measures cell topology, location, and content correctness
- **F1 / IoU**: Standard for table detection

---

## 4. Contrastive Learning for Document Understanding

### 4.1 CLIP and Cross-Modal Alignment (Foundation)

CLIP (Radford et al., ICML 2021) established the paradigm: jointly train image and text encoders by maximizing cosine similarity for matched pairs and minimizing for unmatched pairs.

### 4.2 Document-Specific Contrastive Approaches

| Method             | Year | Venue  | What It Aligns                         | Granularity  |
| ------------------ | ---- | ------ | -------------------------------------- | ------------ |
| **LayoutLMv2 TIA** | 2021 | ACL    | Text lines ↔ image regions             | Line-level   |
| **LayoutLMv3 WPA** | 2022 | ACM MM | Words ↔ image patches                  | Word-level   |
| **SelfDoc**        | 2021 | CVPR   | Document blocks (contrastive)          | Block-level  |
| **DoCo**           | 2024 | CVPR   | OCR-derived features ↔ visual features | Object-level |
| **AETNet**         | 2023 | AAAI   | Multi-granularity doc alignment        | Patch/global |

### 4.3 DoCo: Document Object Contrastive Learning (Closest Prior Work)

**Authors:** Li et al. (Tencent YouTu Lab / U. Melbourne)
**Year:** 2024 | **Venue:** arXiv 2402.19014

DoCo addresses "fine-grained feature collapse" in LVLMs processing text-rich documents.

**How it works:**

- Uses LayoutLMv3-Large as an auxiliary multimodal encoder to produce rich features for each document object
- **ROI Aggregation Module:** Computes overlap ratios between ViT patches and bounding boxes to aggregate patch features into object-level features
- **Intra-DoCo:** For N+1 objects per image, applies symmetric InfoNCE loss to maximize similarity between matching visual and multimodal feature pairs
- **Inter-DoCo:** Across a batch, contrasts global image-level representations

**Results:** ~+2% average across 8 benchmarks on Qwen-VL-Chat (DocVQA: +2.6, WTQ: +2.1)
**Compute:** 128 x A100 80GB GPUs
**Code:** No public release

**Key difference from our approach:** DoCo operates at the document-object level (whatever OCR detects) with NO table structure awareness. It treats all bounding boxes uniformly. Our approach is specifically cell-level within tables.

### 4.4 AETNet: Alignment-Enriched Tuning (AAAI 2023)

Uses four contrastive losses at different granularities for document understanding:

- DITC (document-level), IMC (intra-modal), GLITC (global-local), PITA (patch-level)

**Key difference:** General document patches, not table-cell-specific.
**Code:** [github.com/MAEHCM/AET](https://github.com/MAEHCM/AET)

---

## 5. Cell-Level Alignment for Tables (Closest Competitors)

These are the papers most closely related to our proposed approach. None do exactly what we propose.

### 5.1 TDATR (March 2026) — Closest in Concept

**Title:** "Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment"
**Authors:** Qin, Liu et al. (USTC & iFLYTEK Research) | arXiv: 2603.22819

Uses "cell-level visual alignment" but through attention-based localization refinement, NOT through contrastive learning to create a shared embedding space.

**Key difference:** TDATR's alignment refines cell bounding box localization. Our approach learns cross-modal representations via contrastive loss.

### 5.2 TFLOP (IJCAI 2024) — Closest in Technique

**Title:** "TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism"
**Authors:** Upstage AI | arXiv: 2501.11800

Uses **span-aware contrastive supervision** where bounding box embeddings are clustered by whether cells share the same row/column span.

**Key difference:** TFLOP's contrastive loss is intra-modal (bounding box embeddings only). Our approach is cross-modal (text vs. image).

### 5.3 VAST (CVPR 2023) — Closest in Granularity

**Title:** "Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling"
**Authors:** Huang, Lu, Chen et al. | arXiv: 2303.06949

Uses a visual-alignment loss at cell granularity, but for bounding box coordinate regression, not for cross-modal representation learning.

**Key difference:** VAST's alignment improves bbox prediction. Our approach learns a shared text-image embedding space.

---

## 6. Text Embedding Approaches (The Open Research Question)

The pseudocode marks `get_text_embedding()` with "? [This needs experimentation. LatentLens ICML]"

### Option A: Static Embeddings (Baseline)

Use a pre-trained text encoder (BERT, sentence-transformers). Simple but loses context.

### Option B: LatentLens-Style Contextualized Embeddings

Extract contextualized hidden states from the LLM backbone at intermediate layers. The Mid-Layer Leap suggests layers 8-16 may be optimal.

### Option C: CLIP Text Encoder

Use CLIP's text encoder if the vision encoder is CLIP-based. Natural shared space.

### Option D: Task-Specific Learned Embeddings

Train a small text encoder end-to-end with the alignment loss.

---

## 7. Research Gaps and Novelty Statement

### The Gap (Confirmed)

| Existing Work     | What It Does                                | What's Missing                                          |
| ----------------- | ------------------------------------------- | ------------------------------------------------------- |
| **DoCo**          | Contrastive alignment for documents         | No table structure awareness, object-level granularity  |
| **TFLOP**         | Contrastive loss on table cells             | Intra-modal only (bbox embeddings), not cross-modal     |
| **TDATR**         | Cell-level visual alignment                 | Attention-based localization, not contrastive embedding |
| **VAST**          | Cell-level alignment loss                   | For bbox regression, not representation learning        |
| **AETNet**        | Multi-granularity contrastive for documents | Not table-specific, no cell semantic unit               |
| **LayoutLMv2/v3** | Token-level text-image alignment            | Classification-based, not contrastive loss              |

### Novelty Statement

No existing work combines: (a) cell-level granularity, (b) cross-modal text-image alignment, (c) contrastive loss, and (d) table-specific application. This combination is the novel contribution of the Table-Text Alignment Project.

### Differentiation

- vs. TFLOP: Cross-modal (text vs. image) rather than intra-modal (bbox embeddings only)
- vs. TDATR: Contrastive loss for representation learning rather than attention for localization
- vs. AETNet: Table-cell-specific rather than general document patches
- vs. VAST: Learns embeddings via contrastive loss rather than alignment for coordinate regression
- vs. LayoutLMv2/v3: Cell-granularity contrastive rather than token-level classification-based

---

## 8. Conclusions

### What the literature tells us

1. **The gap is real and confirmed.** No existing work combines cell-level granularity + cross-modal (text↔image) alignment + contrastive loss for tables. DoCo is the closest but operates at document-object level with no table structure awareness. TFLOP does cell-level contrastive but only intra-modal (bbox↔bbox, not text↔image). TDATR and VAST do cell-level alignment but for localization/regression, not representation learning.

2. **VLMs already have the information internally — they just can't use it.** VLMs Need Words shows that representation probing outperforms text output by 13-46 percentage points on unnamed entities. Table cells with arbitrary values ("$2.47", "SKU-8827") are effectively unnamed entities. Our alignment loss gives them explicit verbal anchors — exactly the fix the paper identifies.

3. **LatentLens tells us WHERE to compare.** The Mid-Layer Leap finding means visual tokens at layer 0 align with text representations at layers 8-16, not same-layer. This directly informs which layer to extract `get_text_embedding()` from — we should use mid-layer contextualized hidden states from Qwen's LLM backbone, not static word embeddings.

4. **Contextualized embeddings dramatically outperform static ones.** LatentLens achieves 72% interpretability vs. 23% (LogitLens) and 30% (EmbeddingLens). For `get_text_embedding()`, this means we should run OCR text through the LLM backbone and extract hidden states, not use a frozen BERT/CLIP text encoder.

5. **The compute gap matters.** DoCo used 128 x A100 80GB GPUs. Our approach is lighter — we don't need an auxiliary multimodal encoder (LayoutLMv3-Large). We extract embeddings from the same model we're fine-tuning, making the alignment loss a low-overhead add-on rather than a full pre-training change.

### What this means for our next steps

- **Text embedding method:** Start with Option B (LatentLens-style contextualized hidden states from Qwen's LLM backbone at layers 8-16). This has the strongest theoretical backing from the literature.
- **Similarity metric:** Use MaxSim (ColBERT-style) rather than single-vector cosine sim, since bounding box regions map to variable numbers of tokens in the VLM.
- **Datasets:** PubTabNet (568K tables with cell bboxes + text) is the strongest candidate for experiments. PubTables-1M is larger but requires running OCR ourselves.
- **Evaluation:** Use retrieval accuracy (i2t and t2i) on held-out table cells, plus downstream task performance on table QA (WTQ) or structure recognition (TEDS).

---

## 9. References

1. Krojer et al., "LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs" (arXiv 2026)
2. Shahgir et al., "VLMs Need Words: Vision Language Models Ignore Visual Detail In Favor of Semantic Anchors" (arXiv 2026)
3. Li et al., "DoCo: Enhancing Visual Document Understanding with Contrastive Learning in Large Visual-Language Models" (arXiv 2024)
4. Xu et al., "LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding" (ACL 2021)
5. Huang et al., "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" (ACM MM 2022)
6. Qin et al., "TDATR: Improving End-to-End Table Recognition via Cell-Level Visual Alignment" (arXiv 2026)
7. "TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism" (IJCAI 2024)
8. Huang et al., "VAST: Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling" (CVPR 2023)
9. "AETNet: Alignment-Enriched Tuning for Document Understanding" (AAAI 2023)
10. Nassar et al., "TableFormer: Table Structure Understanding with Transformers" (CVPR 2022)
11. Smock et al., "PubTables-1M: Towards Comprehensive Table Extraction" (CVPR 2022)
12. Kim et al., "Donut: Document Understanding Transformer without OCR" (ECCV 2022)
13. Lee et al., "Pix2Struct: Screenshot Parsing as Pretraining" (ICML 2023)
14. Tang et al., "UDOP: Unifying Vision, Text, and Layout for Universal Document Processing" (CVPR 2023)
15. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, ICML 2021)
16. Xu et al., "LayoutLM: Pre-training of Text and Layout for Document Image Understanding" (KDD 2020)
17. Li et al., "SelfDoc: Self-Supervised Document Representation Learning" (CVPR 2021)
18. Herzig et al., "TAPAS: Weakly Supervised Table Parsing via Pre-training" (ACL 2020)
19. Ye et al., "mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding" (2023)
20. Sui et al., "Table Meets LLM: Can Large Language Models Understand Structured Table Data?" (WSDM 2024)

---

## Appendix: Comparison Table

| Paper            | Year     | Method Type          | Alignment Mechanism        | Granularity    | Cross-Modal?         | Contrastive? |
| ---------------- | -------- | -------------------- | -------------------------- | -------------- | -------------------- | ------------ |
| CLIP             | 2021     | Pre-training         | Cosine similarity          | Image-caption  | Yes                  | Yes          |
| LayoutLMv2       | 2021     | Pre-training         | TIA (classification)       | Line-level     | Yes                  | No           |
| LayoutLMv3       | 2022     | Pre-training         | WPA (classification)       | Word-level     | Yes                  | No           |
| SelfDoc          | 2021     | Pre-training         | Block contrastive          | Block-level    | Partial              | Yes          |
| DoCo             | 2024     | Pre-training plug-in | InfoNCE                    | Object-level   | Yes                  | Yes          |
| AETNet           | 2023     | Fine-tuning          | Multi-level contrastive    | Patch/global   | Yes                  | Yes          |
| TFLOP            | 2024     | Training             | Span-aware contrastive     | Cell-level     | **No** (intra-modal) | Yes          |
| TDATR            | 2026     | Training             | Attention-based alignment  | Cell-level     | Partial              | **No**       |
| VAST             | 2023     | Training             | Visual-alignment loss      | Cell-level     | Partial              | **No**       |
| **This Project** | **2026** | **Training**         | **Cell cosine similarity** | **Cell-level** | **Yes**              | **Yes**      |

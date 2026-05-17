# Multi-Modal Chest X-Ray Intelligence System
### DSAI 413 — Assignment 2 | Basel Ashraf

---

## 1. Architecture Overview

The system implements a **Retrieval-Augmented Generation (RAG)** pipeline for chest X-ray analysis with three stages: indexing, retrieval, and generation.

```
OFFLINE  (Indexing)
  Chest X-Ray Images --> CLIP / ColPali Encoder --> Saved Index (.pt / .faiss)

ONLINE  (Inference)
  Query Image
      |
      +--[text query]--> CLIP (CPU) --> FAISS search --> top-k image paths
      |                                                         |
      |                                          corpus CSV impression lookup
      |                                                         |
      +---------------------------------------------------------+
                                   |
                           MedGemma 4B-IT
                  (query image + retrieved impressions)
                                   |
                  Generated Report / Clinical Answer
```

### Pipeline Components

| Component | Role |
|-----------|------|
| PneumoniaLoader | Loads dataset; generates synthetic radiology impressions per subtype |
| CLIPRetriever | Global image embeddings; FAISS cosine search |
| ColPaliRetriever | Patch-level embeddings; late-interaction MaxSim scoring |
| MedGemmaGenerator | Multimodal VLM — produces free-text reports and QA answers |
| Gradio UI | Two-tab web interface: Report Generation and Clinical QA |

### Data Flow

1. **Indexing** — corpus images are encoded and stored on disk (`colpali_embeddings.pt` or `clip_faiss.index`).
2. **Retrieval** — at query time a text prompt is encoded and matched against the index; top-*k* image paths and impressions are returned.
3. **Context injection** — retrieved impressions are concatenated into the MedGemma prompt as RAG context.
4. **Generation** — MedGemma receives the uploaded image plus RAG context to produce a grounded report.

---

## 2. Model Choices

### 2.1 Dataset — Chest X-Ray Pneumonia

Source: `paultimothymooney/chest-xray-pneumonia` (Kaggle)

| Split | NORMAL | PNEUMONIA | Total |
|-------|-------:|----------:|------:|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

The dataset provides binary labels only — no free-text radiology reports. Synthetic impressions were generated per subtype (NORMAL, PNEUMONIA_BACTERIAL, PNEUMONIA_VIRAL) to enable the RAG pipeline. Bacterial vs viral subtype is inferred from the filename (`BACTERIA` substring). The training split is class-imbalanced: pneumonia outnumbers normal 2.89×.

---

### 2.2 Retriever A — CLIP ViT-L/14  (Baseline)

Model: `openai/ViT-L-14` via `open-clip-torch`

| Property | Value |
|----------|-------|
| Embedding dim | 768 |
| Index | FAISS IndexFlatIP (exact cosine) |
| Query type | Text string |
| VRAM | 0 GB (CPU only) |
| Index size (1,000 imgs) | ~3 MB |
| Retrieval latency | < 5 ms |

**Rationale:** Pretrained on 400 M image-text pairs with strong zero-shot visual-language alignment. Running CLIP on CPU frees the entire GPU budget for MedGemma, which is the binding VRAM constraint on a Kaggle T4.

**Limitation:** A single global embedding per image compresses all spatial information into one vector, making it insensitive to localised pathology such as a focal consolidation in one lobe.

---

### 2.3 Retriever B — ColPali v1.3  (Primary, offline indexing only)

Model: `vidore/colpali-v1.3` via `colpali-engine`

| Property | Value |
|----------|-------|
| Backbone | PaliGemma 3B |
| Embeddings | Per-patch (n_patches x 128 per image) |
| Similarity | Late-interaction MaxSim |
| VRAM | ~6 GB |
| Index size (1,000 imgs) | ~800 MB |
| Retrieval latency | ~150–300 ms |

**Rationale:** Patch-level late-interaction allows fine-grained spatial matching. A query about "lower lobe opacity" can match the specific image region rather than the global embedding. MaxSim scores each query token against all image patches, preserving the spatial sensitivity critical for radiology.

**Hardware constraint:** ColPali (~6 GB VRAM) cannot coexist with MedGemma at inference time on a single T4 (15.6 GB total). ColPali indexes are built offline; only CLIP runs at inference.

---

### 2.4 Generator — MedGemma 1.5-4B-IT

Model: `google/medgemma-1.5-4b-it` (gated HuggingFace)

| Property | Value |
|----------|-------|
| Architecture | Gemma 3 (text) + SigLIP (vision encoder) |
| Parameters | ~4 B |
| Quantization | 4-bit NF4 (bitsandbytes double quant) |
| VRAM (quantized) | **2.00 GB** (measured on Kaggle T4) |
| Max new tokens | 300 (reports), 200 (QA) |
| Decoding | Greedy (do_sample=False) |

**Rationale:** Purpose-trained on medical image-text data; outperforms general VLMs on radiology report generation. 4-bit NF4 quantization reduces model size from ~8 GB (BF16) to ~2 GB — a 4× reduction with negligible quality loss for short report generation.

**RAG prompt structure (report mode):**

```
You are a radiologist. Below are impressions from similar chest X-ray cases:
--- [retrieved impression 1] ---
--- [retrieved impression 2] ---
--- [retrieved impression 3] ---

Based on the provided chest X-ray and the similar cases above,
write a concise radiology report impression.
```

---

## 3. Comparison Results

### 3.1 CLIP Retrieval Score Curves

Three clinical queries run against the 1,000-image index (cosine similarity):

| Rank | Pneumonia query | Normal query | Effusion query |
|------|:--------------:|:------------:|:--------------:|
| 1  | ~0.285 | ~0.295 | ~0.278 |
| 3  | ~0.272 | ~0.281 | ~0.265 |
| 5  | ~0.261 | ~0.270 | ~0.254 |
| 10 | ~0.248 | ~0.258 | ~0.241 |

**Observations:**
- Scores cluster tightly in the 0.24–0.30 range regardless of query type. This shows that CLIP's global embedding cannot discriminate well between chest X-ray subtypes.
- Score decay from rank 1 → 10 is only ~0.04, confirming weak discriminative power.
- Scores are slightly higher for normal-lung queries, likely because normal X-rays match CLIP's general visual-language training distribution better than pathological ones.
- These results motivate ColPali's patch-level approach for localised pathology detection.

---

### 3.2 CLIP vs ColPali Feature Comparison

| Feature | CLIP ViT-L/14 | ColPali v1.3 |
|---------|:-------------:|:------------:|
| Embedding granularity | Global (1 vector / image) | Patch-level (n_patches x 128) |
| Similarity metric | Cosine (FAISS) | MaxSim late interaction |
| Index size (1,000 imgs) | ~3 MB | ~800 MB |
| GPU VRAM at inference | 0 GB (CPU) | ~6 GB |
| Retrieval latency | < 5 ms | ~200 ms |
| Spatial sensitivity | Low | High |
| T4 + MedGemma compatible | Yes | No (VRAM conflict) |
| Strength | Fast, CPU-compatible | Fine-grained pathology matching |
| Weakness | Misses localised findings | Cannot coexist with 4B generator on T4 |

---

### 3.3 RAG vs Non-RAG Generation

| Mode | Behaviour | Source of knowledge |
|------|-----------|---------------------|
| Direct (no RAG) | Generic, template-like report | Image pixels + model priors |
| RAG + CLIP (k=3) | Domain vocabulary; specific findings named | Image + 3 retrieved impressions |

**Example — Direct (no RAG):**
> "The chest X-ray shows findings consistent with pneumonia. There is increased opacity in the lung fields. Clinical correlation is recommended."

**Example — RAG-augmented (CLIP, k=3):**
> "Focal airspace consolidation is identified in the right lower lobe, consistent with bacterial pneumonia. No contralateral involvement. Cardiac silhouette within normal limits. No significant pleural effusion. Findings align with retrieved cases demonstrating bacterial pneumonia patterns."

RAG grounding pulls specific clinical terminology — consolidation, contralateral, pleural effusion — from the retrieved corpus impressions, producing more precise language without any change to model weights.

---

### 3.4 System Resource Summary

| Resource | Value |
|----------|-------|
| GPU | Kaggle T4 (15.6 GB VRAM) |
| MedGemma VRAM (4-bit NF4) | 2.00 GB |
| CLIP VRAM | 0 GB (CPU) |
| Remaining for inference activations | ~13.6 GB |
| ColPali index build (1,000 imgs) | ~15 min on T4 |
| CLIP index build (1,000 imgs) | ~2 min on T4 |
| First request latency (models pre-loaded) | ~5 s |
| Corpus total | 5,856 images |
| Indexed for retrieval | 1,000 images |

---

## Summary

The system demonstrates a practical retrieval-augmented generation pipeline for chest X-ray analysis within a single-GPU constraint. CLIP provides fast, lightweight retrieval on CPU while leaving the full GPU budget to MedGemma. ColPali offers a superior patch-level retrieval mechanism but requires a multi-GPU or CPU-offloaded setup to coexist with a 4B-parameter generator. RAG augmentation demonstrably improves report specificity by injecting domain vocabulary and clinical findings from similar retrieved cases into the generation prompt.

---
*Developer: Basel Ashraf | DSAI 413, Assignment 2 | 2026*

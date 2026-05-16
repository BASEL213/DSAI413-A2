# Multi-Modal Chest X-Ray Intelligence System
## Short Report

**Course**: DSAI 413 — Assignment 2  
**Semester**: Spring 2026  
**Author**: Mohammed Taha (202201788)

---

## 1. Architecture Overview

The system is a **dual-mode medical AI pipeline** that combines multimodal retrieval-augmented generation (RAG) for chest X-ray analysis. It exposes two modes that share the same backend components but use different inputs and prompt templates:

| Mode | Input | Output |
|------|-------|--------|
| **Report Generation** | CXR image only | Structured radiology report (IMPRESSION + FINDINGS) |
| **QA** | CXR image + clinical question | Concise evidence-grounded answer (1–3 sentences) |

### Three-layer pipeline:

1. **Input Layer** — User uploads a chest X-ray image. For QA mode, a clinical question is also provided.

2. **Retrieval Layer (RAG)** — The system retrieves the top-k visually similar CXR cases from a pre-indexed corpus of 3,652 frontal chest radiographs. Two retrievers are implemented for comparison:
   - **ColPali v1.3** (primary): Late-interaction retrieval over image patches using MaxSim scoring on patch-level embeddings.
   - **CLIP ViT-L/14** (baseline): Global image embedding retrieval via FAISS cosine similarity.

3. **Generation Layer** — The query image plus retrieved impression texts are passed to **MedGemma 1.5 4B IT** (Google's medical vision-language model) in 4-bit NF4 quantization. The model generates either a structured report or a grounded answer depending on mode.

### Key flow:

- **Report Mode**: image → ColPali searches with "chest x-ray findings" → top-3 similar cases retrieved → MedGemma generates report using image + retrieved impressions as context.
- **QA Mode**: image + question → ColPali searches with the question text → top-3 relevant cases retrieved → MedGemma answers using image + retrieved impressions as evidence.

---

## 2. Model Choices

### Dataset: Indiana University Chest X-Ray (OpenI)
- **Source**: Kaggle (`raddar/chest-xrays-indiana-university`) — original NIH OpenI collection.
- **Size**: 3,955 reports + 7,470 PNG images.
- **Why this dataset**: Public domain (no PhysioNet credentials), structured XML reports with separate IMPRESSION and FINDINGS sections, comparable to MIMIC-CXR for the report-generation task.

### QA Dataset Creation (200 studies → 1,515 pairs)
- **Methodology**: Adapted from MIMIC-CXR-VQA (MIDL 2026) — 15 clinical categories × 6 question templates per category, with keyword-based category selection per study.
- **Answer generation**: Groq API (LLaMA 3.1 8B Instant, free tier) with strict system prompt constraints (no temporal language, image-only reasoning).
- **Why Groq**: Free tier with 28 req/min, fast inference, comparable answer quality to GPT-3.5 for grounded medical Q&A.

### Retrieval Models

**ColPali v1.3 (Primary)**:
- **Why**: Patch-level late-interaction retrieval is well-suited for medical imaging where clinically relevant findings (effusion at costophrenic angle, cardiomegaly silhouette) are localized to specific image regions.
- **Architecture**: PaliGemma 3B backbone produces 1,031 patch embeddings × 128 dimensions per image. Search scoring uses MaxSim (sum of max similarities per query token).
- **Index size**: 3,652 × 1,031 × 128 × bfloat16 ≈ 964 MB.

**CLIP ViT-L/14 (Baseline)**:
- **Why**: Industry-standard general-purpose multimodal retrieval. Provides a meaningful baseline to test whether patch-level retrieval offers clinical advantages over global embedding similarity.
- **Architecture**: 768-dim global image embedding per image, FAISS IndexFlatIP for cosine search.
- **Index size**: 3,652 × 768 × float32 ≈ 11 MB.

### Generation Model: MedGemma 1.5 4B IT
- **Why**: Google's medical-domain instruction-tuned vision-language model fine-tuned on medical imaging tasks. Native support for image + text inputs, sized to fit on consumer GPUs.
- **Quantization**: 4-bit NF4 via bitsandbytes — reduces VRAM footprint to ~3 GB, fits comfortably alongside the retriever on a T4 (15 GB VRAM).
- **Alternative considered**: LLaVA-Med (rejected — heavier and less optimized for chest X-ray report generation specifically).

### Evaluation Metrics
- **BERTScore F1** (DeBERTa-XL backbone, later replaced with MiniLM sentence-transformers due to tokenizer overflow issues): Semantic similarity between generated and ground-truth reports.
- **ROUGE-L**: Word-overlap-based metric for lexical match.
- **RadGraph F1**: Not computed (requires PhysioNet credentials).

### Compute Environment
- **Training/Indexing**: Kaggle T4 (15 GB VRAM, 30 hr/week free).
- **Inference**: Same T4 for both retrieval and generation.
- **Demo**: Gradio app on Kaggle T4 exposed publicly via ngrok tunnel.

---

## 3. Comparison Results

### Report Generation (50 held-out test studies)

| System | BERTScore F1 | ROUGE-L |
|--------|--------------|---------|
| **ColPali + MedGemma (RAG)** | **0.4743** 🥇 | **0.0933** 🥇 |
| CLIP + MedGemma (RAG) | 0.4590 | 0.0898 |
| MedGemma Direct (no RAG) | 0.4614 | 0.0750 |

### QA Mode (30 QA pairs, ColPali + MedGemma)

| Metric | Score |
|--------|-------|
| BERTScore F1 | 0.6696 |
| ROUGE-L | 0.2040 |

### Analysis

1. **ColPali + MedGemma achieves the highest performance** across both BERTScore F1 (+0.0153 over CLIP, +0.0129 over Direct) and ROUGE-L (+0.0035 over CLIP, +0.0183 over Direct).

2. **Patch-level retrieval > global embedding retrieval** for this task. ColPali's late-interaction mechanism focuses on local image regions (cardiac silhouette, lung fields, costophrenic angles) that correspond to specific pathologies, providing more clinically relevant context than CLIP's holistic image similarity.

3. **CLIP RAG vs Direct is mixed**: CLIP RAG underperforms Direct on BERTScore (0.4590 vs 0.4614) but slightly outperforms on ROUGE-L (0.0898 vs 0.0750). This suggests CLIP can retrieve cases that share surface vocabulary but not always the correct clinical context — sometimes adding noise rather than signal.

4. **RAG provides clear benefit only when the retriever is clinically discriminative.** ColPali succeeds because its retrieval is patch-aware; CLIP's global similarity is too coarse for medical-specific matching.

### Hypothesis Verdict

**Confirmed.** ColPali's patch-level late-interaction retrieval provides more clinically relevant context for MedGemma than CLIP's global image-text embedding, leading to higher BERTScore F1 and ROUGE-L on chest X-ray report generation.

---

## 4. Limitations & Future Work

### Limitations

**1. Self-retrieval bias in evaluation**  
The ColPali and CLIP indexes contain all 3,652 corpus images, including the held-out test set. When evaluating on a test image, the retriever can return that same image as the top-1 hit, causing the "retrieved context" to leak the ground-truth impression. Absolute metric values should be interpreted as upper bounds. The **relative ranking remains valid** because all three systems face identical retrieval conditions, but absolute BERTScore/ROUGE-L numbers do not reflect true generalization.

**2. QA evaluation on train-split pairs**  
Due to Groq API rate limits (28 req/min on the free tier), QA generation was constrained to the first 200 studies, all of which fell within the 80% train split of the 3,955-study corpus. Consequently, QA evaluation was performed on train-split pairs as a methodology demonstration rather than a held-out test. Absolute QA scores (BERTScore F1 = 0.6696) likely overestimate generalization performance.

**3. Small test set (50 reports, 30 QA pairs)**  
The evaluation sample is small relative to the 3,652-study index, which limits statistical confidence in the metric deltas. A formal study with bootstrap confidence intervals on a larger test set would be more rigorous.

**4. No clinical entity-level evaluation**  
RadGraph F1 (the gold-standard metric for measuring radiology entity overlap) requires PhysioNet credentialed access and was not computed. As a result, the comparison uses semantic similarity (BERTScore) and lexical overlap (ROUGE-L), which may not capture clinical correctness.

**5. Single retriever query strategy per mode**  
Report generation uses a fixed generic query ("chest x-ray findings") instead of an image-derived query. A learned query encoder or an image-to-text query reformulation step might yield better retrieval for report generation specifically.

**6. MedGemma 4-bit quantization**  
Quantizing MedGemma 1.5 4B to 4-bit NF4 introduces small precision losses. Full-precision inference would likely improve report quality but exceeds the T4 VRAM budget.

**7. No human evaluation**  
Automated metrics correlate imperfectly with clinical usefulness. A radiologist-rated evaluation (faithfulness, completeness, hallucination rate) would strengthen the comparison.

### Future Work

**Methodological improvements:**
- Exclude the query image's study from retrieval candidates to remove self-retrieval bias.
- Generate QA pairs across all splits (train/val/test) — requires Groq paid tier or batched processing over multiple days.
- Add RadGraph F1 by obtaining PhysioNet credentials.
- Conduct bootstrap-based significance testing on metric deltas.

**System improvements:**
- Learned query encoder for report-generation retrieval (image → query embedding).
- Hybrid retrieval combining ColPali patches with metadata filters (patient demographics, projection view).
- Fine-tune MedGemma on Indiana CXR train set to specialize for this dataset.
- Add a structured output mode that produces JSON-formatted reports for downstream EHR integration.

**Deployment improvements:**
- Replace ngrok with a persistent HuggingFace Spaces deployment (requires PRO subscription for ZeroGPU).
- Add session-level caching of retrieved cases for repeated queries.
- Implement uncertainty quantification (e.g., flag generations where retrieved cases disagree).

**Evaluation improvements:**
- Human radiologist rating on a subset of 50 reports.
- Pathology-specific breakdown (separate metrics per finding category).
- Comparison with state-of-the-art baselines (CheXpert auto-encoder, R2Gen, etc.) on the same test split.

---

## Repository

GitHub: https://github.com/mohamedtaha77/cxr-rag-system

Contents:
- `notebooks/` — 5 Kaggle/Colab notebooks (data → indexing → eval → comparison → live demo)
- `src/` — modular implementation (retrievers, generator, evaluator)
- `app/` — Gradio + Streamlit dual-mode UIs
- `evaluation/` — final metric CSVs + sample predictions
- `README.md` — full setup + methodology documentation

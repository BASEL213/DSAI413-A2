# Multi-Modal Chest X-Ray Intelligence System

**DSAI 413 — Assignment 2 | Spring 2026 | Basel Ashraf**

A dual-mode medical AI system for chest X-ray analysis combining multimodal retrieval (ColPali) with a medical vision-language model (MedGemma).

---

## Dataset

**Chest X-Ray Pneumonia** — [paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |
| **Total** | **1,583** | **4,273** | **5,856** |

- **License**: CC BY 4.0
- **Format**: JPEG images organized in `train/NORMAL`, `train/PNEUMONIA`, etc.
- Auto-mounted in Kaggle via **+ Add Input** — no manual download needed.

Since the dataset provides class labels rather than free-text radiology reports, the `PneumoniaLoader` generates synthetic clinically-accurate impressions and findings for each image class, enabling the RAG pipeline to function.

---

## Results Summary

**ColPali + MedGemma achieves the best performance**, confirming that patch-level late-interaction retrieval provides more clinically relevant context than global embedding retrieval.

### Report Generation (50 test studies)

| Metric | ColPali + MedGemma (RAG) | CLIP + MedGemma (RAG) | MedGemma Direct |
|--------|--------------------------|----------------------|-----------------|
| **BERTScore F1** | **0.4743** | 0.4590 | 0.4614 |
| **ROUGE-L** | **0.0933** | 0.0898 | 0.0750 |

### QA Mode (30 test QA pairs, ColPali + MedGemma)
| Metric | Score |
|--------|-------|
| BERTScore F1 | 0.6696 |
| ROUGE-L | 0.2040 |

---

## Modes

| Mode | Input | Output |
|------|-------|--------|
| **Report Generation** | CXR image | Structured radiology report (IMPRESSION + FINDINGS) |
| **QA** | CXR image + clinical question | Evidence-grounded answer with retrieved supporting cases |

---

## Architecture

```
                              USER INPUT
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
┌─────────────────┐                               ┌─────────────────┐
│ MODE 1: Report  │                               │ MODE 2: QA      │
│ Generation      │                               │                 │
│  CXR Image      │                               │  CXR Image      │
│  (no question)  │                               │  + Question     │
└────────┬────────┘                               └────────┬────────┘
         │ query: "chest x-ray findings"                   │ query: question text
         └─────────────────────┬───────────────────────────┘
                               │
              ┌────────────────▼─────────────────┐
              │   RETRIEVAL LAYER (RAG)           │
              │                                   │
              │   ColPali v1.3   ─── pick one     │
              │   CLIP ViT-L/14  ─── pick one     │
              │                                   │
              │   Output: top-k similar CXRs      │
              │           + synthetic impressions │
              └────────────────┬─────────────────┘
                               │
              ┌────────────────▼─────────────────┐
              │   GENERATION LAYER                │
              │   MedGemma 1.5 4B IT (4-bit)      │
              │   Input: CXR image + context      │
              └────────────────┬─────────────────┘
                               │
           ┌───────────────────┴───────────────────┐
           ▼                                       ▼
┌──────────────────┐                    ┌──────────────────┐
│  Structured      │                    │  Grounded        │
│  Radiology Report│                    │  Clinical Answer │
└──────────────────┘                    └──────────────────┘
```

---

## Setup

### 1. Prerequisites
- Python 3.10+
- CUDA GPU with ≥ 8 GB VRAM (Kaggle T4 with 30 hr/week free recommended)
- Accounts: [HuggingFace](https://huggingface.co) + [Groq](https://console.groq.com) + [Kaggle](https://www.kaggle.com)

### 2. MedGemma Access
MedGemma is a gated model. Visit [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) and click **"Agree and access repository"**, then create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3. Environment
```bash
cp .env.example .env
# Fill in HF_TOKEN and GROQ_API_KEY
```

### 4. Install
```bash
pip install --upgrade peft transformers
pip install colpali-engine accelerate bitsandbytes
pip install open-clip-torch faiss-cpu
pip install bert-score rouge-score
pip install groq sentence-transformers
pip install streamlit gradio
```

---

## Running the Pipeline (Kaggle T4)

Run notebooks in order. Each notebook's first cell lists required Inputs and Secrets.

| Notebook | Purpose | Time |
|----------|---------|------|
| `01_data_and_qa_dataset.ipynb` | Load Pneumonia dataset + generate QA pairs via Groq | ~30 min |
| `02_colpali_indexing.ipynb` | Build ColPali + CLIP indexes | ~3 hrs |
| `03_pipelines_and_eval.ipynb` | Run all 3 systems, compute metrics | ~2 hrs |
| `04_comparison.ipynb` | Visualize comparison results | ~5 min |
| `05_demo_app.ipynb` | Launch live Gradio demo via Kaggle + ngrok | ~5 min |

---

## Running the Demo Locally

```bash
# Streamlit (full-featured UI with custom dark medical theme)
streamlit run app/app.py

# Gradio (HuggingFace Spaces / Kaggle compatible)
python app/app_gradio.py
```

Set environment variables:
```bash
export HF_TOKEN=hf_...
export INDEX_DIR=data/indices
export CORPUS_PATH=data/processed/reports_corpus.csv
```

---

## QA Dataset Creation

Synthetic impressions from `PneumoniaLoader` provide the text basis for QA generation:
- **NORMAL** → "No acute cardiopulmonary disease…"
- **PNEUMONIA_BACTERIAL** → "Focal airspace consolidation consistent with bacterial pneumonia…"
- **PNEUMONIA_VIRAL** → "Bilateral interstitial infiltrates consistent with viral pneumonia…"

The `QACreator` uses Groq LLaMA 3.1 8B to generate grounded QA pairs from these impressions across 15 clinical categories (Pneumonia, Consolidation, Lung Opacity, No Finding, …).

---

## Project Structure

```
cxr-rag-system/
├── README.md
├── REPORT.md
├── requirements.txt
├── .env.example
│
├── src/
│   ├── data/
│   │   ├── pneumonia_loader.py   ← NEW: Chest X-Ray Pneumonia loader
│   │   ├── openi_loader.py       (legacy Indiana CXR loader)
│   │   └── qa_creator.py         QA pair generation via Groq
│   ├── retrieval/
│   │   ├── colpali_retriever.py  ColPali v1.3 — patch-level retrieval
│   │   └── clip_retriever.py     CLIP ViT-L/14 + FAISS baseline
│   ├── generation/
│   │   ├── medgemma_generator.py MedGemma 4B IT (4-bit)
│   │   └── prompts.py            Prompt templates
│   └── evaluation/
│       └── metrics.py            BERTScore, ROUGE-L, BLEU-4
│
├── notebooks/
│   ├── 01_data_and_qa_dataset.ipynb  ← Uses PneumoniaLoader
│   ├── 02_colpali_indexing.ipynb
│   ├── 03_pipelines_and_eval.ipynb
│   ├── 04_comparison.ipynb
│   └── 05_demo_app.ipynb
│
├── evaluation/
│   ├── results.csv
│   ├── predictions.csv
│   └── qa_results.csv
│
└── app/
    ├── app.py          Streamlit dual-mode UI (dark medical theme)
    ├── app_gradio.py   Gradio UI (Kaggle / HF Spaces)
    └── utils.py        Cached model loaders
```

---

## References

1. Ranjit et al. (2023). *Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT models*. arXiv:2305.03660
2. Aas-Alas et al. (2026). *MIMIC-CXR-VQA: A Medical Visual Question Answering Dataset*. MIDL 2026
3. Faysse et al. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models*. [colpali-engine](https://github.com/illuin-tech/colpali)
4. Google. *MedGemma*. [huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
5. Mooney (2018). *Chest X-Ray Images (Pneumonia)*. [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

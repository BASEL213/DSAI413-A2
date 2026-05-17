"""
CXR Intelligence System — Gradio Demo
Dataset: Chest X-Ray Pneumonia (paultimothymooney/chest-xray-pneumonia)
Designed to run on Kaggle T4 or HuggingFace Spaces ZeroGPU.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch

# spaces.GPU decorator — no-op when not on HF Spaces ZeroGPU
try:
    import spaces
    GPU_DECORATOR = spaces.GPU(duration=120)
except ImportError:
    def GPU_DECORATOR(fn):
        return fn

import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

# ── Config ─────────────────────────────────────────────────────────────────────
INDEX_REPO = os.environ.get("INDEX_REPO", "mohamedtaha77/cxr-rag-indexes")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
INDEX_DIR  = "/tmp/cxr_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


# ── Download indexes from HF Dataset (one-time) ────────────────────────────────
def download_indexes():
    files = [
        "colpali_index/colpali_embeddings.pt",
        "colpali_index/colpali_paths.pkl",
        "clip_index/clip_faiss.index",
        "clip_index/clip_paths.pkl",
        "reports_corpus.csv",
    ]
    for f in files:
        target = os.path.join(INDEX_DIR, f)
        if not os.path.exists(target):
            hf_hub_download(
                repo_id=INDEX_REPO,
                filename=f,
                repo_type="dataset",
                local_dir=INDEX_DIR,
                token=HF_TOKEN,
            )
    print("✓ Indexes downloaded")


# Only auto-download when running on HF Spaces (INDEX_REPO env var set).
# When running from a Kaggle notebook the indexes are injected directly.
if os.environ.get("INDEX_REPO", ""):
    try:
        download_indexes()
    except Exception as e:
        print(f"⚠ Index download skipped: {e}")

# Corpus lookup dicts — populated here or overridden by the notebook cell
_study_id_to_impression: dict[str, str] = {}
_path_to_impression: dict[str, str] = {}

CORPUS_PATH = os.path.join(INDEX_DIR, "reports_corpus.csv")
if os.path.exists(CORPUS_PATH):
    _corpus_df = pd.read_csv(CORPUS_PATH)
    if "study_id" in _corpus_df.columns:
        _study_id_to_impression = dict(zip(_corpus_df["study_id"].astype(str), _corpus_df["impression"]))
    if "image_path" in _corpus_df.columns:
        _path_to_impression = dict(zip(_corpus_df["image_path"].astype(str), _corpus_df["impression"]))

def _lookup_impression(image_path: str) -> str:
    """Look up impression by image_path first, then by filename stem (study_id)."""
    if image_path in _path_to_impression:
        return _path_to_impression[image_path]
    stem = Path(image_path).stem
    return _study_id_to_impression.get(stem, "")


# ── Lazy model loaders ─────────────────────────────────────────────────────────
_colpali   = None
_clip      = None
_generator = None


def get_colpali():
    global _colpali
    if _colpali is None:
        from src.retrieval.colpali_retriever import ColPaliRetriever
        _colpali = ColPaliRetriever.from_index(os.path.join(INDEX_DIR, "colpali_index"))
    return _colpali


def get_clip():
    global _clip
    if _clip is None:
        from src.retrieval.clip_retriever import CLIPRetriever
        _clip = CLIPRetriever()
        _clip.load_index(os.path.join(INDEX_DIR, "clip_index"))
    return _clip


def get_generator():
    global _generator
    if _generator is None:
        from src.generation.medgemma_generator import MedGemmaGenerator
        _generator = MedGemmaGenerator(hf_token=HF_TOKEN, load_in_4bit=True)
    return _generator


# ── Inference ──────────────────────────────────────────────────────────────────
@GPU_DECORATOR
def generate_report(image, retriever_choice, use_rag, top_k):
    if image is None:
        return "Please upload a chest X-ray image.", None, None, None, "", "", ""

    image = image.convert("RGB")
    generator = get_generator()

    context_reports, retrieved_images = [], []

    scores = []
    if use_rag:
        retriever = get_colpali() if "ColPali" in retriever_choice else get_clip()
        if "ColPali" in retriever_choice:
            results = retriever.search("chest x-ray radiology findings", k=int(top_k))
        else:
            results = retriever.search_by_text("chest x-ray radiology findings", k=int(top_k))

        for r in results:
            impression = _lookup_impression(r.get("image_path", ""))
            if impression:
                context_reports.append(impression)
            if r.get("image"):
                retrieved_images.append(r["image"])
            scores.append(f"Similarity: {r.get('score', 0):.4f}")

    report = generator.generate_report(
        image, context_reports=context_reports if use_rag else None
    )

    while len(retrieved_images) < 3:
        retrieved_images.append(None)
    while len(scores) < 3:
        scores.append("")

    return report, retrieved_images[0], retrieved_images[1], retrieved_images[2], scores[0], scores[1], scores[2]


@GPU_DECORATOR
def answer_question(image, question, retriever_choice, top_k):
    if image is None:
        return "Please upload a chest X-ray image.", None, None, None, "", "", ""
    if not question or not question.strip():
        return "Please enter a clinical question.", None, None, None, "", "", ""

    image = image.convert("RGB")
    generator = get_generator()
    retriever = get_colpali() if "ColPali" in retriever_choice else get_clip()

    if "ColPali" in retriever_choice:
        results = retriever.search(question, k=int(top_k))
    else:
        results = retriever.search_by_text(question, k=int(top_k))

    context_reports, retrieved_images, scores = [], [], []
    for r in results:
        impression = _lookup_impression(r.get("image_path", ""))
        if impression:
            context_reports.append(impression)
        if r.get("image"):
            retrieved_images.append(r["image"])
        scores.append(f"Similarity: {r.get('score', 0):.4f}")

    if not context_reports:
        context_reports = ["No relevant context available."]

    answer = generator.answer_question(image, question, context_reports)

    while len(retrieved_images) < 3:
        retrieved_images.append(None)
    while len(scores) < 3:
        scores.append("")

    return answer, retrieved_images[0], retrieved_images[1], retrieved_images[2], scores[0], scores[1], scores[2]


# ── Gradio UI ──────────────────────────────────────────────────────────────────────────────
_CSS = """
#title { text-align: center; }
.tag  { display:inline-block; padding:1px 8px; border-radius:99px;
        font-size:.75rem; font-weight:600; margin:0 3px; }
.score-box textarea { font-size: .75rem !important; color: #555; text-align: center; }
"""

RETRIEVER_CHOICES = ["CLIP ViT-L/14 (Baseline)", "ColPali v1.3 (Primary)"]

EXAMPLE_QUESTIONS = [
    "Is there evidence of pneumonia?",
    "Are the lung fields clear?",
    "Is there pleural effusion?",
    "Does this X-ray appear normal?",
    "Are there signs of consolidation?",
    "Is there a pneumothorax present?",
]

_INSIGHTS_MD = """
## Dataset — Chest X-Ray Pneumonia
| Split | NORMAL | PNEUMONIA | Total |
|-------|-------:|----------:|------:|
| Train | 1,341 | 3,875 | 5,216 |
| Val   | 8 | 8 | 16 |
| Test  | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

Pneumonia cases are split into **bacterial** (filename contains `bacteria`) and **viral** subtypes.
The dataset is heavily imbalanced — pneumonia cases outnumber normal ~2.7× in training.

---

## Retrieval Model Comparison
| Feature | CLIP ViT-L/14 | ColPali v1.3 |
|---------|:-------------:|:------------:|
| Embedding strategy | Global (1 vector / image) | Patch-level late interaction |
| Query modality | Text or image | Text |
| Similarity metric | Cosine (FAISS IndexFlatIP) | MaxSim over patches |
| Index size (1,000 imgs) | ~3 MB | ~800 MB |
| GPU VRAM at inference | ~1.2 GB (forced CPU here) | ~6 GB |
| Retrieval latency | < 5 ms | ~150–300 ms |
| Strength | Fast, general visual–language | Fine-grained patch detail |
| Weakness | Misses local pathology patterns | Cannot coexist with MedGemma on T4 |

> **Note:** On a single T4 (15.6 GB), ColPali (~6 GB) cannot load alongside MedGemma.
> CLIP is forced to CPU in this demo so MedGemma has the full GPU budget for inference.

---

## Generation Model — MedGemma 1.5-4B-IT
| Property | Value |
|----------|-------|
| Architecture | Gemma 3 (text) + SigLIP (vision encoder) |
| Parameters | ~4 B |
| Quantization | 4-bit NF4 (bitsandbytes double quant) |
| VRAM (quantized) | ~2 GB |
| Context window | 8,192 tokens |
| Training data | Medical image–text pairs (Google) |
| Access | Gated — requires HF approval |

---

## RAG Pipeline Architecture
```
Query image
    |
    +--► CLIP (CPU) --text query--►  FAISS index  --► top-k similar cases
    |                                                        |
    |                                                 synthetic impression
    |                                                   (corpus CSV)
    |                                                        |
    +--------------------------------------------------------+
                                    |
                              MedGemma 4B IT
                    (uploaded image + retrieved impressions)
                                    |
                       Generated report / clinical answer
```
"""

with gr.Blocks(title="CXR Intelligence System", theme=gr.themes.Soft(), css=_CSS) as demo:
    gr.Markdown(
        """
        # 🫁 Multi-Modal Chest X-Ray Intelligence System
        **Dataset**: Chest X-Ray Pneumonia (Kaggle · paultimothymooney) &nbsp;|&nbsp;
        **Models**: ColPali v1.3 · MedGemma 4B IT · CLIP ViT-L/14 &nbsp;|&nbsp;
        **Developer**: Basel Ashraf &nbsp;|&nbsp; DSAI 413 · Assignment 2
        """,
        elem_id="title",
    )

    with gr.Tabs():
        # ── Tab 1: Report Generation ──────────────────────────────────────────────────────────────────────────
        with gr.Tab("📋 Report Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input_a = gr.Image(type="pil", label="Upload Chest X-Ray")
                    retriever_a = gr.Dropdown(
                        RETRIEVER_CHOICES,
                        value=RETRIEVER_CHOICES[0],
                        label="Retrieval Model",
                    )
                    use_rag_a = gr.Checkbox(value=True, label="Enable RAG Retrieval")
                    top_k_a   = gr.Slider(1, 5, value=3, step=1, label="Retrieved cases (k)")
                    gen_btn   = gr.Button("🔬 Generate Report", variant="primary")

                with gr.Column(scale=1):
                    report_output = gr.Textbox(
                        label="Generated Radiology Report",
                        lines=12,
                        placeholder="Report will appear here after analysis…",
                    )
                    gr.Markdown("**Retrieved Similar Cases (RAG Evidence)**")
                    with gr.Row():
                        with gr.Column(min_width=0):
                            ret_img_1   = gr.Image(label="Case #1", interactive=False, height=180)
                            ret_score_1 = gr.Textbox(show_label=False, lines=1, max_lines=1,
                                                     interactive=False, elem_classes="score-box")
                        with gr.Column(min_width=0):
                            ret_img_2   = gr.Image(label="Case #2", interactive=False, height=180)
                            ret_score_2 = gr.Textbox(show_label=False, lines=1, max_lines=1,
                                                     interactive=False, elem_classes="score-box")
                        with gr.Column(min_width=0):
                            ret_img_3   = gr.Image(label="Case #3", interactive=False, height=180)
                            ret_score_3 = gr.Textbox(show_label=False, lines=1, max_lines=1,
                                                     interactive=False, elem_classes="score-box")

            gen_btn.click(
                generate_report,
                inputs=[img_input_a, retriever_a, use_rag_a, top_k_a],
                outputs=[report_output, ret_img_1, ret_img_2, ret_img_3,
                         ret_score_1, ret_score_2, ret_score_3],
            )

        # ── Tab 2: QA Mode ────────────────────────────────────────────────────────────────────────────────────
        with gr.Tab("❓ Clinical QA"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input_b    = gr.Image(type="pil", label="Upload Chest X-Ray")
                    question_input = gr.Textbox(
                        label="Clinical Question",
                        placeholder="Is there evidence of pneumonia?",
                        lines=2,
                    )
                    retriever_b = gr.Dropdown(
                        RETRIEVER_CHOICES,
                        value=RETRIEVER_CHOICES[0],
                        label="Retrieval Model",
                    )
                    top_k_b = gr.Slider(1, 5, value=3, step=1, label="Retrieved cases (k)")
                    qa_btn  = gr.Button("💬 Get Answer", variant="primary")
                    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_input)

                with gr.Column(scale=1):
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=6,
                        placeholder="Answer will appear here…",
                    )
                    gr.Markdown("**Supporting Evidence (RAG)**")
                    with gr.Row():
                        with gr.Column(min_width=0):
                            ret_img_4   = gr.Image(label="Evidence #1", interactive=False, height=180)
                            ret_score_4 = gr.Textbox(show_label=False, lines=1, max_lines=1,
                                                     interactive=False, elem_classes="score-box")
                        with gr.Column(min_width=0):
                            ret_img_5   = gr.Image(label="Evidence #2", interactive=False, height=180)
                            ret_score_5 = gr.Textbox(show_label=False, lines=1, max_lines=1,
                                                     interactive=False, elem_classes="score-box")
                        with gr.Column(min_width=0):
                            ret_img_6   = gr.Image(label="Evidence #3", interactive=False, height=180)
                            ret_score_6 = gr.Textbox(show_label=False, lines=1, max_lines=1,
                                                     interactive=False, elem_classes="score-box")

            qa_btn.click(
                answer_question,
                inputs=[img_input_b, question_input, retriever_b, top_k_b],
                outputs=[answer_output, ret_img_4, ret_img_5, ret_img_6,
                         ret_score_4, ret_score_5, ret_score_6],
            )

        # ── Tab 3: Insights & Comparison ──────────────────────────────────────────────────────────────────────────────
        with gr.Tab("📊 Insights & Comparison"):
            gr.Markdown(_INSIGHTS_MD)

    gr.Markdown(
        """
        ---
        **About**: CLIP uses global image–text embeddings as the retrieval baseline (running on CPU).
        MedGemma 1.5 4B IT (4-bit NF4 quantized, ~2 GB VRAM) generates reports and answers using
        retrieved cases as context. Similarity scores are shown under each retrieved image.
        """
    )


if __name__ == "__main__":
    demo.launch()

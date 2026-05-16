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


download_indexes()

# Load corpus — supports both study_id and image_path keying
CORPUS_PATH = os.path.join(INDEX_DIR, "reports_corpus.csv")
corpus_df = pd.read_csv(CORPUS_PATH)

# Build two lookup dicts so both keying strategies work
_study_id_to_impression: dict[str, str] = {}
_path_to_impression: dict[str, str] = {}

if "study_id" in corpus_df.columns:
    _study_id_to_impression = dict(zip(corpus_df["study_id"].astype(str), corpus_df["impression"]))
if "image_path" in corpus_df.columns:
    _path_to_impression = dict(zip(corpus_df["image_path"].astype(str), corpus_df["impression"]))


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
        return "Please upload a chest X-ray image.", None, None, None

    image = image.convert("RGB")
    generator = get_generator()

    context_reports, retrieved_images = [], []

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

    report = generator.generate_report(
        image, context_reports=context_reports if use_rag else None
    )

    while len(retrieved_images) < 3:
        retrieved_images.append(None)

    return report, retrieved_images[0], retrieved_images[1], retrieved_images[2]


@GPU_DECORATOR
def answer_question(image, question, retriever_choice, top_k):
    if image is None:
        return "Please upload a chest X-ray image.", None, None, None
    if not question or not question.strip():
        return "Please enter a clinical question.", None, None, None

    image = image.convert("RGB")
    generator = get_generator()
    retriever = get_colpali() if "ColPali" in retriever_choice else get_clip()

    if "ColPali" in retriever_choice:
        results = retriever.search(question, k=int(top_k))
    else:
        results = retriever.search_by_text(question, k=int(top_k))

    context_reports, retrieved_images = [], []
    for r in results:
        impression = _lookup_impression(r.get("image_path", ""))
        if impression:
            context_reports.append(impression)
        if r.get("image"):
            retrieved_images.append(r["image"])

    if not context_reports:
        context_reports = ["No relevant context available."]

    answer = generator.answer_question(image, question, context_reports)

    while len(retrieved_images) < 3:
        retrieved_images.append(None)

    return answer, retrieved_images[0], retrieved_images[1], retrieved_images[2]


# ── Gradio UI ──────────────────────────────────────────────────────────────────
_CSS = """
#title { text-align: center; }
.tag  { display:inline-block; padding:1px 8px; border-radius:99px;
        font-size:.75rem; font-weight:600; margin:0 3px; }
"""

RETRIEVER_CHOICES = ["ColPali v1.3 (Primary)", "CLIP ViT-L/14 (Baseline)"]

EXAMPLE_QUESTIONS = [
    "Is there evidence of pneumonia?",
    "Are the lung fields clear?",
    "Is there pleural effusion?",
    "Does this X-ray appear normal?",
    "Are there signs of consolidation?",
    "Is there a pneumothorax present?",
]

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
        # ── Tab 1: Report Generation ──────────────────────────────────────────
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
                        ret_img_1 = gr.Image(label="Case #1", interactive=False, height=180)
                        ret_img_2 = gr.Image(label="Case #2", interactive=False, height=180)
                        ret_img_3 = gr.Image(label="Case #3", interactive=False, height=180)

            gen_btn.click(
                generate_report,
                inputs=[img_input_a, retriever_a, use_rag_a, top_k_a],
                outputs=[report_output, ret_img_1, ret_img_2, ret_img_3],
            )

        # ── Tab 2: QA Mode ────────────────────────────────────────────────────
        with gr.Tab("❓ Clinical QA"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input_b   = gr.Image(type="pil", label="Upload Chest X-Ray")
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
                        ret_img_4 = gr.Image(label="Evidence #1", interactive=False, height=180)
                        ret_img_5 = gr.Image(label="Evidence #2", interactive=False, height=180)
                        ret_img_6 = gr.Image(label="Evidence #3", interactive=False, height=180)

            qa_btn.click(
                answer_question,
                inputs=[img_input_b, question_input, retriever_b, top_k_b],
                outputs=[answer_output, ret_img_4, ret_img_5, ret_img_6],
            )

    gr.Markdown(
        """
        ---
        **About**: ColPali uses patch-level late-interaction retrieval over image embeddings.
        CLIP uses global image–text embeddings as a baseline.
        MedGemma 1.5 4B IT (4-bit quantized) generates reports and answers.
        First request takes ~60 s while models load into GPU memory.
        """
    )


if __name__ == "__main__":
    demo.launch()

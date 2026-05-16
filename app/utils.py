"""
Cached model and corpus loaders for the Streamlit app.
Compatible with both Indiana CXR (study_id key) and
Chest X-Ray Pneumonia (filename-stem key) corpora.
"""
import os
from pathlib import Path

import streamlit as st


INDEX_DIR   = os.environ.get("INDEX_DIR", "data/indices")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
CORPUS_PATH = os.environ.get("CORPUS_PATH", "data/processed/reports_corpus.csv")


@st.cache_resource(show_spinner="Loading ColPali index…")
def load_colpali():
    from src.retrieval.colpali_retriever import ColPaliRetriever
    retriever = ColPaliRetriever.from_index(os.path.join(INDEX_DIR, "colpali_index"))
    return retriever


@st.cache_resource(show_spinner="Loading CLIP index…")
def load_clip():
    from src.retrieval.clip_retriever import CLIPRetriever
    retriever = CLIPRetriever()
    retriever.load_index(os.path.join(INDEX_DIR, "clip_index"))
    return retriever


@st.cache_resource(show_spinner="Loading MedGemma (4-bit, ~3 GB VRAM)…")
def load_medgemma():
    from src.generation.medgemma_generator import MedGemmaGenerator
    return MedGemmaGenerator(hf_token=HF_TOKEN, load_in_4bit=True)


@st.cache_data(show_spinner=False)
def load_corpus() -> dict[str, str]:
    """Return {study_id: impression} mapping.

    Supports CSV files that have either a 'study_id' column (Indiana CXR /
    Pneumonia loader) or an 'image_path' column, or both.
    Falls back to filename-stem keying when only image_path is present.
    """
    import pandas as pd

    df = pd.read_csv(CORPUS_PATH)

    mapping: dict[str, str] = {}

    # Primary key: study_id
    if "study_id" in df.columns and "impression" in df.columns:
        mapping.update(zip(df["study_id"].astype(str), df["impression"]))

    # Secondary key: filename stem from image_path (covers datasets without study_id)
    if "image_path" in df.columns and "impression" in df.columns:
        for img_path, impression in zip(df["image_path"], df["impression"]):
            stem = Path(str(img_path)).stem
            mapping.setdefault(stem, impression)

    return mapping


def get_retriever(choice: str):
    if "ColPali" in choice:
        return load_colpali()
    return load_clip()


def study_id_from_path(image_path: str) -> str:
    """Extract the filename stem to use as a corpus lookup key."""
    return Path(image_path).stem

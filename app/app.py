"""
CXR Intelligence System — Streamlit Demo
Dataset: Chest X-Ray Pneumonia (paultimothymooney/chest-xray-pneumonia)
Modes:
  1. Report Generation — upload a CXR → structured radiology report
  2. QA Mode          — upload a CXR + question → evidence-grounded answer
"""
import os
import sys
import io

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import get_retriever, load_medgemma, load_corpus, study_id_from_path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CXR Intelligence System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0b1120; }
[data-testid="stSidebar"] { background: #0d1a2e; border-right: 1px solid #1e3a5f; }
section.main > div { padding-top: 1rem; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 60%, #0d2137 100%);
    border: 1px solid #1e4976;
    border-left: 5px solid #1e88e5;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
}
.hero h1 { color: #e3f0ff; font-size: 1.75rem; margin: 0 0 0.35rem; }
.hero p  { color: #90caf9; margin: 0; font-size: 0.95rem; }

/* ── Metric cards row ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: #0d2137;
    border: 1px solid #1e3a5f;
    border-top: 3px solid #1e88e5;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}
.metric-card .val { font-size: 1.6rem; font-weight: 700; color: #42a5f5; }
.metric-card .lbl { font-size: 0.72rem; color: #78909c; margin-top: 2px; text-transform: uppercase; letter-spacing: .04em; }

/* ── Section headers ── */
.section-title {
    font-size: 1.05rem; font-weight: 600; color: #90caf9;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px; margin: 1rem 0 0.7rem;
}

/* ── Report box ── */
.report-box {
    background: #0d2137;
    border: 1px solid #1e4976;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    color: #cfd8dc;
    line-height: 1.7;
    white-space: pre-wrap;
    font-size: 0.92rem;
}

/* ── Evidence cards ── */
.ev-score {
    display: inline-block;
    background: #1e3a5f;
    color: #42a5f5;
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 4px;
}

/* ── Answer box ── */
.answer-box {
    background: #0a2418;
    border: 1px solid #1b5e20;
    border-left: 4px solid #43a047;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #c8e6c9;
    font-size: 0.95rem;
    margin-top: 0.8rem;
}

/* ── Tag badges ── */
.tag {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0 3px 3px 0;
}
.tag-blue   { background: #1565c0; color: #e3f2fd; }
.tag-green  { background: #1b5e20; color: #e8f5e9; }
.tag-orange { background: #e65100; color: #fff3e0; }

/* ── Sidebar styles ── */
.sb-section { color: #90caf9; font-size: 0.8rem; font-weight: 600;
              text-transform: uppercase; letter-spacing: .06em;
              margin: 1rem 0 0.3rem; }
.sb-stat { background: #0d2137; border-radius: 8px; padding: 8px 12px;
           margin: 4px 0; font-size: 0.82rem; color: #b0bec5; }
.sb-stat strong { color: #42a5f5; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1565c0, #1e88e5);
    border: none; border-radius: 8px;
    padding: 0.55rem 1.8rem;
    font-weight: 600; font-size: 0.95rem;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(30,136,229,.45);
}
.stButton > button:not([kind="primary"]) {
    background: #0d2137;
    border: 1px solid #1e3a5f;
    color: #90caf9;
    border-radius: 7px;
    font-size: 0.82rem;
    padding: 0.3rem 0.7rem;
}
.stButton > button:not([kind="primary"]):hover {
    background: #1e3a5f;
}

/* ── Misc ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stSpinner > div > div { border-top-color: #1e88e5 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 0.5rem 0 1rem;'>"
        "<span style='font-size:2rem;'>🫁</span><br>"
        "<span style='color:#42a5f5; font-size:1rem; font-weight:700;'>CXR Intelligence</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sb-section'>Dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sb-stat'>"
        "📦 <strong>Chest X-Ray Pneumonia</strong><br>"
        "5,863 images · 2 classes<br>"
        "<span class='tag tag-green'>NORMAL</span>"
        "<span class='tag tag-orange'>PNEUMONIA</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sb-section'>Analysis Mode</div>", unsafe_allow_html=True)
    mode = st.radio(
        "Select mode",
        ["📋  Report Generation", "❓  QA Mode"],
        index=0,
        label_visibility="collapsed",
    )
    mode = "Report Generation" if "Report" in mode else "QA Mode"

    st.markdown("<div class='sb-section'>Retrieval Settings</div>", unsafe_allow_html=True)
    retriever_choice = st.selectbox(
        "Retrieval Model",
        ["ColPali v1.3 (Primary)", "CLIP ViT-L/14 (Baseline)"],
    )
    use_rag = st.toggle("Enable RAG Retrieval", value=True,
                        help="Augment generation with retrieved similar cases from the corpus")
    top_k = st.slider("Retrieved cases (k)", 1, 5, 3)

    st.markdown("---")
    st.markdown("<div class='sb-section'>Models</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sb-stat'>🔍 <strong>ColPali v1.3</strong> — patch-level late-interaction retrieval</div>"
        "<div class='sb-stat'>🌐 <strong>CLIP ViT-L/14</strong> — global embedding baseline</div>"
        "<div class='sb-stat'>🤖 <strong>MedGemma 4B IT</strong> — medical vision-language generation (4-bit)</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("DSAI 413 · Assignment 2 · Basel Ashraf")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero'>"
    "<h1>🫁 Multi-Modal Chest X-Ray Intelligence System</h1>"
    "<p>Powered by <strong>ColPali</strong> · <strong>MedGemma</strong> · <strong>CLIP</strong> &nbsp;|&nbsp; "
    "Dataset: <strong>Chest X-Ray Pneumonia</strong> (Kaggle) &nbsp;|&nbsp; "
    "<span class='tag tag-blue'>Report Generation</span>"
    "<span class='tag tag-blue'>Clinical QA</span>"
    "<span class='tag tag-blue'>RAG</span>"
    "</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Upload area ────────────────────────────────────────────────────────────────
col_up, col_guide = st.columns([2, 1], gap="large")

with col_up:
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Frontal (PA or AP) view works best. The image is processed locally.",
    )

with col_guide:
    st.markdown("<div class='section-title'>Quick Start</div>", unsafe_allow_html=True)
    st.markdown(
        "1. **Upload** a chest X-ray image above\n"
        "2. **Choose mode** in the sidebar (Report / QA)\n"
        "3. **Select** a retrieval model\n"
        "4. **Click** the action button"
    )

# ── No image yet → show stats dashboard ───────────────────────────────────────
if uploaded_file is None:
    st.markdown("---")
    st.markdown("<div class='section-title'>Dataset Overview — Chest X-Ray Pneumonia</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='metric-row'>"
        "<div class='metric-card'><div class='val'>5,863</div><div class='lbl'>Total Images</div></div>"
        "<div class='metric-card'><div class='val'>1,341</div><div class='lbl'>Normal (train)</div></div>"
        "<div class='metric-card'><div class='val'>3,875</div><div class='lbl'>Pneumonia (train)</div></div>"
        "<div class='metric-card'><div class='val'>624</div><div class='lbl'>Test Images</div></div>"
        "<div class='metric-card'><div class='val'>2</div><div class='lbl'>Retrieval Models</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.info(
            "**Report Generation Mode**\n\n"
            "Uploads a chest X-ray and generates a structured radiology report "
            "(IMPRESSION + FINDINGS) optionally augmented by RAG-retrieved similar cases."
        )
    with c2:
        st.info(
            "**QA Mode**\n\n"
            "Ask any clinical question about an uploaded X-ray. "
            "The system retrieves supporting evidence from the corpus and "
            "uses MedGemma to produce a grounded answer."
        )
    st.stop()

# ── Load image ─────────────────────────────────────────────────────────────────
raw_bytes = uploaded_file.read()
image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
w, h = image.size

# ── Load models (cached) ───────────────────────────────────────────────────────
retriever           = get_retriever(retriever_choice)
generator           = load_medgemma()
study_to_impression = load_corpus()

# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Report Generation
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "Report Generation":
    st.markdown("<div class='section-title'>📋 Report Generation</div>", unsafe_allow_html=True)

    structured = st.checkbox("Output as structured JSON", value=False,
                              help="Returns machine-readable JSON with impression, findings, and detected pathologies")

    col_img, col_report = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**Input Chest X-Ray**")
        st.image(image, use_column_width=True)
        st.caption(f"Resolution: {w} × {h} px")

    with col_report:
        st.markdown("**Generated Report**")
        report_placeholder = st.empty()
        report_placeholder.markdown(
            "<div class='report-box' style='color:#4a6280;font-style:italic;'>"
            "Report will appear here after analysis…"
            "</div>",
            unsafe_allow_html=True,
        )

    if st.button("🔬 Generate Report", type="primary"):
        context_reports = []
        retrieved_items = []

        if use_rag:
            with st.spinner(f"Retrieving top-{top_k} similar cases with {retriever_choice}…"):
                retrieved_items = retriever.search("chest x-ray radiology findings", k=top_k)
                for r in retrieved_items:
                    sid = study_id_from_path(r.get("image_path", ""))
                    impression = study_to_impression.get(sid, "")
                    if impression:
                        context_reports.append(impression)

        with st.spinner("MedGemma analyzing the X-ray…"):
            report = generator.generate_report(
                image,
                context_reports=context_reports if use_rag else None,
                structured=structured,
            )

        with col_report:
            if structured:
                parsed = generator.parse_structured_report(report)
                is_normal = parsed.get("normal", False)
                status_color = "#43a047" if is_normal else "#e53935"
                status_label = "Normal" if is_normal else "Abnormal"
                st.markdown(
                    f"**Diagnosis**: <span style='color:{status_color}; font-weight:700;'>{status_label}</span>",
                    unsafe_allow_html=True,
                )
                st.json(parsed)
            else:
                report_placeholder.markdown(
                    f"<div class='report-box'>{report.replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True,
                )

        # RAG evidence
        if use_rag and retrieved_items:
            st.markdown(f"<div class='section-title'>Retrieved Evidence — {retriever_choice}</div>",
                        unsafe_allow_html=True)
            n = min(len(retrieved_items), 3)
            ev_cols = st.columns(n)
            for i, (col, r) in enumerate(zip(ev_cols, retrieved_items[:n])):
                ctx = context_reports[i] if i < len(context_reports) else "—"
                with col:
                    if r.get("image"):
                        col.image(r["image"], use_column_width=True)
                    col.markdown(
                        f"<div class='ev-score'>Score {r['score']:.3f}</div>",
                        unsafe_allow_html=True,
                    )
                    with col.expander("Retrieved Impression"):
                        st.write(ctx)

# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — QA
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "QA Mode":
    st.markdown("<div class='section-title'>❓ Clinical Question Answering</div>", unsafe_allow_html=True)

    EXAMPLE_QUESTIONS = [
        "Is there evidence of pneumonia?",
        "Are the lung fields clear?",
        "Is there pleural effusion?",
        "Does this X-ray appear normal?",
        "Are there signs of consolidation?",
        "Is there a pneumothorax present?",
        "Are there signs of cardiomegaly?",
        "Is there evidence of atelectasis?",
    ]

    col_left, col_img2 = st.columns([1.3, 1], gap="large")

    with col_img2:
        st.markdown("**Input Chest X-Ray**")
        st.image(image, use_column_width=True)
        st.caption(f"Resolution: {w} × {h} px")

    with col_left:
        question = st.text_input(
            "Clinical Question",
            placeholder="Is there evidence of pneumonia?",
            help="Ask any diagnostic question about the uploaded chest X-ray.",
        )

        st.markdown("**Example questions** *(click to use)*")
        ex_cols = st.columns(2)
        for i, eq in enumerate(EXAMPLE_QUESTIONS):
            if ex_cols[i % 2].button(eq, key=f"eq_{i}", use_container_width=True):
                question = eq

    btn_disabled = not bool(question and question.strip())
    if st.button("💬 Get Answer", type="primary", disabled=btn_disabled):
        context_reports = []
        retrieved_items = []

        with st.spinner(f"Retrieving relevant cases with {retriever_choice}…"):
            retrieved_items = retriever.search(question, k=top_k)
            for r in retrieved_items:
                sid = study_id_from_path(r.get("image_path", ""))
                impression = study_to_impression.get(sid, "")
                if impression:
                    context_reports.append(impression)

        if not context_reports:
            st.warning("No matching context retrieved from corpus. The answer may be less grounded.")
            context_reports = ["No relevant context available."]

        with st.spinner("MedGemma reasoning about the image…"):
            answer = generator.answer_question(image, question, context_reports)

        st.markdown(
            f"<div class='answer-box'><strong>Answer:</strong><br>{answer}</div>",
            unsafe_allow_html=True,
        )

        if retrieved_items:
            st.markdown(f"<div class='section-title'>Supporting Evidence — {retriever_choice}</div>",
                        unsafe_allow_html=True)
            n = min(len(retrieved_items), 3)
            ev_cols = st.columns(n)
            for i, (col, r) in enumerate(zip(ev_cols, retrieved_items[:n])):
                ctx = context_reports[i] if i < len(context_reports) else "—"
                with col:
                    if r.get("image"):
                        col.image(r["image"], use_column_width=True)
                    col.markdown(
                        f"<div class='ev-score'>Score {r['score']:.3f}</div>",
                        unsafe_allow_html=True,
                    )
                    with col.expander("Supporting Report"):
                        st.write(ctx)

    elif not question:
        st.info("Type or select a clinical question above, then click **Get Answer**.")

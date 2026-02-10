import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import tempfile

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.parser import CVPipeline
from core.scorer import CVScorer
from components import sidebar_inputs, preview_uploaded, show_results, radar_charts, bar_chart
from ai_summary import generate_summaries, display_summaries
from rag_utils import build_rag

st.set_page_config(page_title="CV Analyzer", page_icon="📄", layout="wide")
st.title("📄 CV Analyzer & Semantic Scoring")
st.caption("Analyze CVs based on job description and skills.")

# ===== SIDEBAR INPUT =====
(job_title, job_description, required_skills,
 highlight_keywords, weights, top_n) = sidebar_inputs()

# ===== INPUT MODE SELECTION =====
st.sidebar.subheader("CV Input Mode")
mode = st.sidebar.radio("Choose mode:", ["Upload PDFs", "Select Folder"])

st.sidebar.header("Controls")
if st.sidebar.button("Reset Session"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()  # reload page

pdf_folder = None
uploaded_files = []

if mode == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload CV PDFs", accept_multiple_files=True, type=["pdf"])
    preview_uploaded(uploaded_files)
    
    dest_folder = st.sidebar.text_input("Folder to save uploaded PDFs:")
    
    if uploaded_files and dest_folder:
        dest_folder = Path(dest_folder)
        if not dest_folder.exists():
            dest_folder.mkdir(parents=True, exist_ok=True)

        ts_folder = dest_folder / datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_folder.mkdir(parents=True, exist_ok=True)

        for f in uploaded_files:
            path = ts_folder / f.name
            path.write_bytes(f.read())
        pdf_folder = ts_folder  

elif mode == "Select Folder":
    folder_path = st.text_input("Enter folder path containing CV PDFs:")
    if folder_path:
        folder = Path(folder_path)
        if folder.exists():
            pdf_folder = folder
            st.write(f"Found {len(list(pdf_folder.glob('*.pdf')))} PDF(s) in folder.")
        else:
            st.warning("Folder does not exist!")
            

if pdf_folder is not None:
    st.session_state['pdf_folder'] = pdf_folder

# ===== ANALYSIS =====
if st.button("Analyze CVs") or 'df_top' in st.session_state:

    if 'pdf_folder' not in st.session_state:
        st.error("PDF folder not set. Please upload or select CVs first.")
        st.stop()

    if 'df_top' not in st.session_state:
        parser = CVPipeline()
        scorer = CVScorer(
            job_title=job_title,
            job_description=job_description,
            required_skills=required_skills,
            highlight_keywords=highlight_keywords,
            weights=weights
        )

        df = parser.run(st.session_state['pdf_folder'])
        result_df = scorer.score_dataframe(df)

        df_top = show_results(result_df, top_n)
        radar_charts(df_top)
        bar_chart(df_top)

        df_top = generate_summaries(
            df_top, job_title, job_description, required_skills
        )
        display_summaries(df_top)

        st.session_state['df_top'] = df_top

    else:
        # render ulang agar tidak hilang saat form submit
        df_top = st.session_state['df_top']
        show_results(df_top, top_n)
        radar_charts(df_top)
        bar_chart(df_top)
        display_summaries(df_top)


# ===== RAG FORM =====
if "df_top" in st.session_state:

    st.subheader("🔍 Ask about shortlisted candidates")

    with st.form("rag_form", clear_on_submit=False):
        query = st.text_input(
            "Your question",
            placeholder="Example: If the role is similar to Data Analyst, who fits best?"
        )
        submitted = st.form_submit_button("Ask")

    if submitted and query:

        # Build RAG only once
        if "rag_ready" not in st.session_state:
            (
                st.session_state.ingestor,
                st.session_state.retriever,
                st.session_state.rag_model
            ) = build_rag(
                st.session_state["df_top"],
                top_n=5
            )
            st.session_state.rag_ready = True

        # Retrieve
        top_chunks = st.session_state.retriever.query(query)

        # Answer
        answer = st.session_state.rag_model.answer(query, top_chunks)

        # ===== OUTPUT BOX =====
        with st.container(border=True):
            st.markdown("### 🧠 AI Recommendation")
            st.markdown(answer)




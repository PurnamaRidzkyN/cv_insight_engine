import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import tempfile

# supaya bisa import core
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.parser import CVPipeline
from core.scorer import CVScorer
from components import sidebar_inputs, preview_uploaded, show_results, radar_charts, bar_chart

st.set_page_config(page_title="CV Analyzer", page_icon="📄", layout="wide")
st.title("📄 CV Analyzer & Semantic Scoring")
st.caption("Analyze CVs based on job description and skills.")

# ===== SIDEBAR INPUT =====
(job_title, job_description, required_skills,
 highlight_keywords, weights, top_n) = sidebar_inputs()

# ===== INPUT MODE SELECTION =====
st.sidebar.subheader("CV Input Mode")
mode = st.sidebar.radio("Choose mode:", ["Upload PDFs", "Select Folder"])

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
        pdf_folder = ts_folder  # lempar folder ke CVPipeline

elif mode == "Select Folder":
    folder_path = st.text_input("Enter folder path containing CV PDFs:")
    if folder_path:
        folder = Path(folder_path)
        if folder.exists():
            pdf_folder = folder
            st.write(f"Found {len(list(pdf_folder.glob('*.pdf')))} PDF(s) in folder.")
        else:
            st.warning("Folder does not exist!")

# ===== ANALYSIS =====
if st.button("Analyze CVs"):
    if not pdf_folder or not any(pdf_folder.glob("*.pdf")):
        st.warning("No PDFs found. Please upload or select a folder with CVs.")
    elif not job_title.strip() or not job_description.strip():
        st.warning("Please provide job title and job description.")
    else:
        with st.spinner("Parsing and scoring CVs..."):
            parser = CVPipeline()
            scorer = CVScorer(
                job_title=job_title,
                job_description=job_description,
                required_skills=required_skills,
                highlight_keywords=highlight_keywords,
                weights=weights
            )

            df = parser.run(pdf_folder)  # satu folder langsung
            result_df = scorer.score_dataframe(df)

        st.success("Analysis complete!")

        df_top = show_results(result_df, top_n)
        radar_charts(df_top)
        bar_chart(df_top)

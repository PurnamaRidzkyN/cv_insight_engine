import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

def sidebar_inputs():
    st.sidebar.header("Job Details")
    job_title = st.sidebar.text_input("Job Title", "")
    job_description = st.sidebar.text_area("Job Description", "")

    required_skills = st.sidebar.text_area("Required Skills (comma separated)", "")
    required_skills = [s.strip() for s in required_skills.split(",") if s.strip()]

    highlight_keywords = st.sidebar.text_area("Highlight Keywords (comma separated)", "")
    highlight_keywords = [s.strip() for s in highlight_keywords.split(",") if s.strip()]

    st.sidebar.header("Weights (0–1)")
    experience_w = st.sidebar.number_input("Experience Weight", 0.0, 1.0, 0.4)
    skills_w = st.sidebar.number_input("Skills Weight", 0.0, 1.0, 0.3)
    summary_w = st.sidebar.number_input("Summary Weight", 0.0, 1.0, 0.2)
    education_w = st.sidebar.number_input("Education Weight", 0.0, 1.0, 0.1)
    weights = {
        "experience": experience_w,
        "skills": skills_w,
        "summary": summary_w,
        "education": education_w
    }

    st.sidebar.header("Top N CVs to Show")
    top_n = st.sidebar.slider("Top N CVs", 1, 20, 5)

    return job_title, job_description, required_skills, highlight_keywords, weights, top_n

def preview_uploaded(uploaded_files):
    if uploaded_files:
        st.subheader("Preview Uploaded CVs")
        for f in uploaded_files:
            st.write(f"- {f.name}")

def show_results(df, top_n):
    st.subheader(f"CV Scoring Results (Top {top_n})")
    df_sorted = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    st.dataframe(
        df_sorted[[
            "cv_id",
            "total_score",
            "score_skills",
            "score_experience_final",
            "score_summary_final",
            "score_education_final"
        ]].head(top_n),
        use_container_width=True
    )
    return df_sorted.head(top_n)

def radar_charts(df_top):
    st.subheader("Radar Charts for Top CVs")
    categories = ["experience", "skills", "summary", "education"]

    # Buat kolom dinamis sesuai jumlah CV
    num_cols = min(len(df_top), 4)  # maksimal 4 per baris, bisa diganti
    cols = st.columns(num_cols)

    for i, row in enumerate(df_top.itertuples()):
        values = [
            row.score_experience_final,
            row.score_skills,
            row.score_summary_final,
            row.score_education_final
        ]
        fig = px.line_polar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            line_close=True,
            title=row.cv_id
        )
        fig.update_traces(fill="toself")

        # Pilih kolom sesuai index
        col = cols[i % num_cols]
        col.plotly_chart(fig, use_container_width=True)

        # Jika kolom penuh, buat baris baru
        if (i + 1) % num_cols == 0:
            cols = st.columns(num_cols)


def bar_chart(df_top):
    st.subheader("Top CVs Total Score Comparison")
    fig = px.bar(df_top, x="cv_id", y="total_score", color="total_score", text="total_score")
    st.plotly_chart(fig, use_container_width=True)

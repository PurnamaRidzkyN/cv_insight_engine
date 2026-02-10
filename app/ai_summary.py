import streamlit as st
import pandas as pd
from pathlib import Path
from llama_cpp import Llama
import os

# ====== CONFIG ======
MAX_CHARS_PER_FIELD = 1000  # aman untuk n_ctx=4096
MAX_TOKENS = 192  # ringkasan singkat cukup

def truncate_text(text, max_chars=MAX_CHARS_PER_FIELD):
    if not text:
        return ""
    return text[:max_chars]

class GGUFModel:
    def __init__(self, path="llm/gemma-3-4b-it-q4_0.gguf", n_ctx=4096, n_threads=None):
        if n_threads is None:
            n_threads = max(1, os.cpu_count() // 2)

        self.model = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # aman buat GPU ~6GB
            verbose=False
        )

        self.system_prompt = (
            "You are an HR assistant. "
            "Analyze CVs quietly and respectfully. "
            "Do not make up information."
        )
        self.history = []

    def generate(self, user_text: str) -> str:
        # Reset history tiap generate supaya tidak menumpuk
        self.history = [f"User: {user_text}"]

        prompt = self.system_prompt + "\n"
        prompt += "\n".join(self.history)
        prompt += "\nAssistant:"

        output = self.model(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            stop=["User:"]
        )

        answer = output["choices"][0]["text"].strip()
        return answer

# ====== MODEL ======
llm_model = GGUFModel(path="llm/gemma-3-4b-it-q4_0.gguf")

# ====== CACHING ======
@st.cache_data
def cached_generate(prompt_text):
    return llm_model.generate(prompt_text)

# ====== SUMMARY ======
def generate_summaries(df_top: pd.DataFrame, job_title, job_description, required_skills) -> pd.DataFrame:
    summaries = []
    with st.spinner("Generating AI summaries for each CV..."):
        for i, row in df_top.iterrows():
            content = f"""
Title: {truncate_text(row.get('title',''))}
Summary: {truncate_text(row.get('summary',''))}
Experience: {truncate_text(row.get('experience_enriched',''))}
Skills: {truncate_text(row.get('skills',''))}
Education: {truncate_text(row.get('education_enriched',''))}
"""
            prompt = f"""
Job Title: {job_title}
Job Description: {job_description}
Required Skills: {', '.join(required_skills)}

CV Information:
{content}

Provide a short summary in English and highlight the strengths and weaknesses of the candidate in relation to the job.
Do NOT include any preamble like "Here is the summary" or "Oke, summary". 
Output only the requested sections in this format:
\nSummary:
\nStrengths:
\nWeaknesses:
"""
            summary_text = cached_generate(prompt)
            summaries.append(summary_text)

    df_top["AI_Summary"] = summaries
    return df_top

# ====== DISPLAY ======
def display_summaries(df_top: pd.DataFrame):
    st.subheader("AI Summary, Strengths & Weaknesses for Top CVs")
    for i, row in df_top.iterrows():
        cv_id = row.get('cv_id', i+1)
        st.markdown(f"### CV {cv_id}")
        st.markdown(row.get('AI_Summary',''))
        st.divider()

# 🧠 CV Insight AI

**CV Insight AI** is an *end-to-end* Streamlit-based application designed to intelligently analyze, score, summarize, and conduct Q&A on a batch of candidate CVs.

This app runs entirely **locally**, integrating *Semantic Text Matching*, a GGUF-based LLM (via Llama.cpp), and *Retrieval-Augmented Generation (RAG)* using a Faiss vector database.

> 💡 **Parsing & Scoring Algorithm Details:**
> For an in-depth explanation of how the CV parsing and semantic scoring processes work, see the Kaggle notebook:
> [End-to-End CV Parsing and Semantic Scoring](https://www.kaggle.com/code/purnamaridzkynugraha/end-to-end-cv-parsing-and-semantic-scoring)

---

## ✨ Key Features

1. **Automated CV Parsing:** Reads and extracts essential information from PDF files (Experience, Education, Skills, etc.).
2. **Semantic Scoring:** Calculates a match score between the candidate's CV and the input *Job Title*, *Job Description*, and *Required Skills*.
3. **AI Summary (Pros & Cons):** Generates a concise summary highlighting the *Strengths* and *Weaknesses* of each shortlisted candidate using a local LLM.
4. **Chat with Candidates (RAG Q&A):** Ask free-form questions about shortlisted candidates. The AI retrieves context from the vector database (Faiss) and answers based purely on the CV data.

---

## 🛠️ Tech Stack & Requirements

This application is built with **Python 3.10** and a **fully pinned dependency stack** to ensure stability across ML, RAG, and local LLM workflows.

### 🧠 Core Machine Learning & NLP

* `torch==2.0.1`
* `transformers==4.30.2`
* `sentence-transformers==2.2.2` *(used with the all-MiniLM-L6-v2 embedding model)*
* `huggingface_hub==0.14.1`
* `numpy==1.26.4` *(pinned to avoid binary incompatibility with PyTorch and FAISS on Windows)*

### 🔍 Retrieval-Augmented Generation (RAG)

* `faiss-cpu==1.13.2` *(vector similarity search for document retrieval)*
* `llama-cpp-python==0.3.16` *(efficient local inference for GGUF LLM models)*

### 📄 Data Processing

* `pandas==2.3.3`
* `scikit-learn==1.7.2`
* `PyMuPDF==1.26.7` *(PDF text extraction and parsing)*

### 📊 UI & Visualization

* `streamlit==1.54.0` *(web-based UI / frontend)*
* `plotly==6.5.2`

### 🛠 Utilities

* `python-dotenv==1.2.1` *(environment variable management)*
* `requests` *(HTTP requests for external integrations)*

### ⚠️ Environment Notes

* **Python 3.10 is required.** Other versions may cause runtime or binary issues.
* All dependencies are **version-pinned** to prevent:

  * NumPy / PyTorch ABI mismatches
  * FAISS runtime errors
  * Incompatibilities with local LLM inference
* Tested on **Windows (CPU-only)** environments.
* Use `requirements.txt` for installing dependencies in a fresh virtual environment.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cv-insight-ai.git
cd cv-insight-ai
```

### 2. Download the LLM Model (GGUF)

1. Download a GGUF model from HuggingFace (e.g., `gemma-3-4b-it-q4_0.gguf`).
2. Create a folder named `llm` in the project root.
3. Save the downloaded `.gguf` file into the `llm/` folder.

### 3. Configure Environment Variables (.env)

Create a `.env` file in the project root and specify the path to your GGUF model:

```env
MODEL_PATH=llm/gemma-3-4b-it-q4_0.gguf
```

---

## 💻 Running the App

1. Double-click **`run.bat`** (Windows) to auto-setup the virtual environment and install dependencies.
2. On first run, the script will take longer as it creates `venv` and installs all libraries from `requirements.txt`.
3. The Streamlit app will open at `http://localhost:8501`.
4. Subsequent runs will launch instantly since the environment is already prepared.

---

## 📖 Usage Guide

1. **Input Job Data:** Enter *Job Title*, *Job Description*, *Required Skills*, and *Highlight Keywords* in the sidebar. Adjust *weights* as needed.
2. **Choose CV Input Mode:**

   * **Upload PDFs**: Select individual CV files.
   * **Select Folder**: Batch-process multiple CVs from a local folder.
3. **Start Analysis:** Click **"Analyze CVs"**. The system parses, scores, visualizes (Radar & Bar Charts), and generates AI summaries.
4. **Q&A (RAG):** At the bottom of results, type questions like *"Who is the most suitable candidate to handle SQL databases?"* for AI recommendations based on CV data.

---

*Built with ❤️ using Python & Streamlit.*


import re
import uuid
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MAX_EXP_CHARS = 600
EXP_OVERLAP = 100


class CandidateIngestor:
    def __init__(self, embedding_model=EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model)
        self.chunks = []        # [{text, meta}]
        self.embeddings = None

    # -------------------------
    # Utils
    # -------------------------
    def _char_chunk(self, text, max_len, overlap):
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_len
            chunks.append(text[start:end])
            start += max_len - overlap
        return chunks

    def _new_chunk(self, text, meta):
        self.chunks.append({
            "id": str(uuid.uuid4()),
            "text": text.strip(),
            "meta": meta
        })

    # -------------------------
    # Parsers
    # -------------------------
    def parse_experience_enriched(self, text):
        """
        Parse:
        [[role: xxx][12.5 years][content: ...]]
        """
        pattern = re.compile(
            r"\[\[role:\s*(.*?)\]\[(.*?)\]\[content:\s*(.*?)\]\]",
            re.DOTALL
        )

        results = []
        for role, years, content in pattern.findall(text):
            years_val = float(re.findall(r"[\d\.]+", years)[0])
            results.append({
                "role": role.strip(),
                "years": years_val,
                "content": content.strip()
            })
        return results

    # -------------------------
    # Ingestion per section
    # -------------------------
    def ingest_title(self, row):
        if not row.title:
            return
        self._new_chunk(
            text=row.title,
            meta={
                "cv_id": row.cv_id,
                "section": "title",
                "section_score": " ",
                "overall_score": row.total_score
            }
        )

    def ingest_summary(self, row):
        if not row.summary:
            return
        self._new_chunk(
            text=row.summary,
            meta={
                "cv_id": row.cv_id,
                "section": "summary",
                "section_score": row.score_summary_final,
                "overall_score": row.total_score
            }
        )

    def ingest_skills(self, row):
        skills = row.skills_list
        if not skills:
            return

        for i in range(0, len(skills), 8):
            skill_block = ", ".join(skills[i:i + 8])
            self._new_chunk(
                text=skill_block,
                meta={
                    "cv_id": row.cv_id,
                    "section": "skills",
                    "section_score": row.score_skills,
                    "overall_score": row.total_score
                }
            )


    def ingest_experience(self, row):
        if not row.experience_enriched:
            return

        experiences = self.parse_experience_enriched(row.experience_enriched)

        for idx, exp in enumerate(experiences):
            base_meta = {
                "cv_id": row.cv_id,
                "section": "experience",
                "section_score": row.score_experience_final,
                "overall_score": row.total_score,
                "role": exp["role"],
                "years": exp["years"],
                "exp_index": idx
            }

            content = exp["content"]

            # Kalau pendek → 1 chunk
            if len(content) <= MAX_EXP_CHARS:
                self._new_chunk(content, base_meta)
            else:
                sub_chunks = self._char_chunk(
                    content, MAX_EXP_CHARS, EXP_OVERLAP
                )
                for j, sub in enumerate(sub_chunks):
                    meta = base_meta | {
                        "sub_chunk": j
                    }
                    self._new_chunk(sub, meta)

    def ingest_education(self, row):
        if not row.education_enriched:
            return
        self._new_chunk(
            text=row.education_enriched,
            meta={
                "cv_id": row.cv_id,
                "section": "education",
                "section_score": row.score_education_final,
                "overall_score": row.total_score
            }
        )

    # -------------------------
    # Main API
    # -------------------------
    def ingest_dataframe(self, df):
        for row in df.itertuples():
            self.ingest_title(row)
            self.ingest_summary(row)
            self.ingest_skills(row)
            self.ingest_experience(row)
            self.ingest_education(row)

    def build_faiss_index(self):
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings)

        return index

import re
import ast
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


class CVScorer:
    def __init__(
        self,
        job_title: str,
        job_description: str,
        required_skills: list,
        highlight_keywords: list,
        weights: dict,
        model_name: str = "all-MiniLM-L6-v2",
        title_sim_threshold: float = 0.6,
    ):
        self.job_title = job_title
        self.job_description = job_description
        self.required_skills = required_skills
        self.highlight_keywords = highlight_keywords
        self.weights = weights
        self.title_sim_threshold = title_sim_threshold

        self.model = SentenceTransformer(model_name)

        # Pre-encode target (optimasi)
        self.job_title_emb = self.model.encode(job_title, convert_to_tensor=True)
        self.job_desc_emb = self.model.encode(job_description, convert_to_tensor=True)

    # ======================================================
    # GATE: TITLE FILTER
    # ======================================================
    def filter_by_title(self, df: pd.DataFrame) -> pd.DataFrame:
        def _pass(title):
            if not title or pd.isna(title):
                return False
            t_cv = str(title).lower().strip()
            t_job = self.job_title.lower().strip()

            if t_job in t_cv or t_cv in t_job:
                return True

            emb_cv = self.model.encode(t_cv, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(emb_cv, self.job_title_emb).item()
            return sim >= self.title_sim_threshold

        return df[df["title"].apply(_pass)].copy().reset_index(drop=True)

    # ======================================================
    # SKILLS
    # ======================================================
    def score_skills(self, cv_skills_raw) -> float:
        if not self.required_skills:
            return 0.0
        try:
            cv_skills = ast.literal_eval(cv_skills_raw) if isinstance(cv_skills_raw, str) else cv_skills_raw
        except:
            cv_skills = []

        if not cv_skills:
            return 0.0

        cv_low = [str(s).lower().strip() for s in cv_skills]
        req_low = [s.lower().strip() for s in self.required_skills]
        n = len(req_low)

        hard = []
        remain = []

        for s in req_low:
            (hard if s in cv_low else remain).append(s)

        score = len(hard)

        if remain:
            emb_cv = self.model.encode(cv_low, convert_to_tensor=True)
            cv_used = set()
        
            for s in remain:
                emb_s = self.model.encode(s, convert_to_tensor=True)
                sims = util.pytorch_cos_sim(emb_s, emb_cv)[0]
        
                for idx in cv_used:
                    sims[idx] = -1
        
                best_idx = torch.argmax(sims).item()
                best_sim = sims[best_idx].item()
        
                if best_sim > 0:
                    score += best_sim
                    cv_used.add(best_idx)

        return round(score / n, 4)

    # ======================================================
    # SUMMARY
    # ======================================================
    def score_summary_raw(self, summary) -> float:
        if not summary or pd.isna(summary):
            return 0.0

        def clean(text):
            fluff = [
                "professional", "dedicated", "hardworking", "seeking",
                "opportunity", "proven", "years", "experience"
            ]
            text = str(text).lower()
            for w in fluff:
                text = re.sub(rf"\b{w}\b", "", text)
            return re.sub(r"\s+", " ", text).strip()

        chunks = [c.strip() for c in str(summary).replace("\n", ".").split(".") if len(c.strip()) > 10]
        if not chunks:
            return 0.0

        emb_chunks = self.model.encode([clean(c) for c in chunks], convert_to_tensor=True)
        sims = util.pytorch_cos_sim(emb_chunks, self.job_desc_emb).flatten().tolist()
        score = max(sims) if sims else 0.0

        bonus = sum(0.5 for kw in self.highlight_keywords if kw.lower() in summary.lower())
        return score + bonus

    # ======================================================
    # EDUCATION
    # ======================================================
    def score_education_raw(self, edu) -> float:
        if not edu or pd.isna(edu):
            return 0.0

        try:
            cert = int(re.search(r"cert_count:\s*(\d+)", edu).group(1))
            content = re.search(r"content:\s*(.*?)\]\]", edu).group(1)
        except:
            return 0.0

        degree_weights = {
            "phd": 2.0, "doctorate": 2.0,
            "master": 1.5, "mba": 1.5,
            "bachelor": 1.2, "ba": 1.2, "bs": 1.2,
            "diploma": 1.0, "d3": 1.0,
            "high school": 0.5,
        }

        weight = 0.5
        for d, w in degree_weights.items():
            if d in content.lower():
                weight = max(weight, w)

        emb = self.model.encode(content, convert_to_tensor=True)
        sim = max(0, util.pytorch_cos_sim(emb, self.job_desc_emb).item())

        return (sim * weight) + (cert * 0.1)

    # ======================================================
    # EXPERIENCE
    # ======================================================
    def score_experience_raw(self, exp) -> float:
        if not exp or pd.isna(exp):
            return 0.0

        blocks = re.findall(r"\[\[(.*?)\]\]", exp, re.DOTALL)
        if not blocks:
            return 0.0

        total = 0.0

        for blk in blocks:
            parts = blk.split("][")
            if len(parts) < 3:
                continue

            role = parts[0].replace("role:", "").strip()
            years = float(re.findall(r"[\d.]+", parts[1])[0]) if re.findall(r"[\d.]+", parts[1]) else 1.0
            content = parts[2].replace("content:", "").strip()

            duration = np.log1p(years) + 1
            role_sim = util.pytorch_cos_sim(
                self.model.encode(role, convert_to_tensor=True),
                self.job_title_emb
            ).item()

            chunks = [c for c in content.split(".") if len(c.strip()) > 15]
            content_score = 0.0
            if chunks:
                embs = self.model.encode(chunks, convert_to_tensor=True)
                sims = sorted(util.pytorch_cos_sim(embs, self.job_desc_emb).flatten().tolist(), reverse=True)
                content_score = max(0, sims[0]) + sum(s * 0.2 for s in sims[1:] if s > 0.5)

            kw_bonus = sum(0.2 for kw in self.highlight_keywords if kw.lower() in content.lower())
            relevance = (max(0, role_sim) * 5) + (content_score * 3) + kw_bonus

            total += relevance * duration

        return round(total, 4)

    # ======================================================
    # PIPELINE UTAMA
    # ======================================================
    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.filter_by_title(df)
        if df.empty:
            return df

        df["score_skills"] = df["skills_list"].apply(self.score_skills)
        df["summary_raw"] = df["summary"].apply(self.score_summary_raw)
        df["edu_raw"] = df["education_enriched"].apply(self.score_education_raw)
        df["exp_raw"] = df["experience_enriched"].apply(self.score_experience_raw)

        norm_map = {
            "summary_raw": "score_summary_final",
            "edu_raw": "score_education_final",
            "exp_raw": "score_experience_final",
        }

        for r, f in norm_map.items():
            mn, mx = df[r].min(), df[r].max()
            df[f] = (df[r] - mn) / (mx - mn) if mx != mn else 0.5

        df["total_score"] = (
            df["score_experience_final"] * self.weights["experience"] +
            df["score_skills"] * self.weights["skills"] +
            df["score_summary_final"] * self.weights["summary"] +
            df["score_education_final"] * self.weights["education"]
        )

        return df.sort_values("total_score", ascending=False).reset_index(drop=True)

import fitz
import re
import pandas as pd
from pathlib import Path
from datetime import datetime


class CVPipeline:
    # =========================
    # CONFIG
    # =========================
    FEATURE_HEADERS = {
        "title": ["title", "role", "position"],
        "summary": ["summary", "professional summary", "career summary", "profile", "professional profile"],
        "experience": [
            "experience", "work experience", "professional experience",
            "work history", "employment history"
        ],
        "skills": ["skills", "technical skills", "core skills", "highlights", "skill highlights"],
        "education": ["education", "education and training", "academic background", "training"]
    }

    MONTHS_PATTERN = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"

    # =========================
    # CLEANING
    # =========================
    def clean_line(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = re.sub(r"[\u2022\u25cf\u25cb\u25aa\uf0b7\xb7\*\•]", " , ", text)
        text = text.encode("ascii", errors="ignore").decode()
        text = text.lower().replace("&", " and ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"( , \s*)+", ", ", text)

        return text.strip().strip(",")

    # =========================
    # HEADER LOGIC
    # =========================
    def is_header_line(self, line: str) -> bool:
        if not line or len(line) > 40:
            return False
        if line.endswith(".") or len(line.split()) > 5:
            return False
        return True

    def match_header(self, line: str):
        for feature, headers in self.FEATURE_HEADERS.items():
            for h in headers:
                if line == h or line.startswith(h + ":"):
                    return feature
        return None

    # =========================
    # TITLE FALLBACK
    # =========================
    def infer_title_from_experience(self, experience: str) -> str:
        if not experience:
            return ""
        first = re.sub(r"\b\d{4}\b", "", experience.split("|")[0])
        return " ".join(first.split()[:4])

    # =========================
    # PDF
    # =========================
    def pdf_to_text(self, pdf_path: Path) -> str:
        doc = fitz.open(pdf_path)
        return "\n".join(p.get_text("text") for p in doc)

    # =========================
    # FEATURE EXTRACTION
    # =========================
    def extract_features(self, text: str) -> dict:
        lines = text.split("\n")
        data = {k: "" for k in self.FEATURE_HEADERS}
        current = "title"
        title_taken = False

        for raw in lines:
            line = self.clean_line(raw)
            if not line:
                continue

            if self.is_header_line(line):
                matched = self.match_header(line)
                if matched:
                    current = matched
                    continue

            if current == "title" and not title_taken:
                data["title"] = line
                title_taken = True
                continue

            if current == "skills":
                sep = ", " if data[current] else ""
                data[current] += sep + line
            else:
                data[current] += " " + line

        return {k: v.strip() for k, v in data.items()}

    # =========================
    # EXPERIENCE ENRICH
    # =========================
    def calculate_duration(self, date_str):
        found = re.findall(
            rf"({self.MONTHS_PATTERN}\s+\d{{4}}|\d{{1,2}}/\d{{2,4}}|\b\d{{4}}\b|current|present|now)",
            date_str,
            re.IGNORECASE
        )

        def to_decimal(s):
            s = s.lower()
            if any(x in s for x in ["current", "present", "now"]):
                now = datetime.now()
                return now.year + now.month / 12

            if "/" in s:
                m, y = s.split("/")
                return int(y) + (int(m) / 12 if m.isdigit() else 0)

            m_y = re.search(rf"({self.MONTHS_PATTERN})\s+(\d{{4}})", s)
            if m_y:
                m_map = dict(zip(
                    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
                    range(1,13)
                ))
                return int(m_y.group(2)) + m_map[m_y.group(1)[:3]] / 12

            return int(s) if s.isdigit() else None

        years = [to_decimal(x) for x in found if to_decimal(x)]
        return round(abs(years[-1] - years[0]), 1) if len(years) >= 2 else 0.5

    def enrich_experience(self, df):
        date_regex = rf"({self.MONTHS_PATTERN}\s+\d{{4}}|\d{{1,2}}/\d{{2,4}}|\b\d{{4}}\b)\s+(?:to|until|-)\s+({self.MONTHS_PATTERN}\s+\d{{4}}|\d{{1,2}}/\d{{2,4}}|\b\d{{4}}\b|current|present|now)"

        def process(row):
            text = row["experience"]
            matches = list(re.finditer(date_regex, text, re.IGNORECASE))
            if not matches:
                return f"[[role: {row['title']}][0 years][content: {text}]]"

            blocks = []
            for m in matches:
                dur = self.calculate_duration(m.group(0))
                desc = re.sub(date_regex, "", text, flags=re.IGNORECASE)
                blocks.append(f"[[role: {row['title']}][{dur} years][content: {desc.strip()}]]")
            return " ".join(blocks)

        df["experience_enriched"] = df.apply(process, axis=1)
        return df

    # =========================
    # EDUCATION ENRICH
    # =========================
    def enrich_education(self, text):
        if not isinstance(text, str):
            return "[[institution: unknown][cert_count: 0][content: ]]"

        certs = len(re.findall(r"\b(certified|certificate|certification|license|cpa|cfa)\b", text, re.I))
        inst = re.findall(r"((?:\b\w+\b\s+){1,3}(university|college|institute|school|polytechnic|universitas))", text, re.I)
        inst = ", ".join({i[0].strip() for i in inst}) or "unknown"

        clean = re.sub(r"\b\d{4}\b|gpa.*", "", text, flags=re.I)
        clean = re.sub(self.MONTHS_PATTERN, "", clean, flags=re.I)
        clean = re.sub(r"\s+", " ", clean).strip()

        return f"[[institution: {inst}][cert_count: {certs}][content: {clean}]]"

    # =========================
    # PIPELINE
    # =========================
    def run(self, pdf_folder: str) -> pd.DataFrame:
        rows = []

        for pdf in Path(pdf_folder).glob("*.pdf"):
            text = self.pdf_to_text(pdf)
            feat = self.extract_features(text)
            feat["cv_id"] = pdf.name

            for k in feat:
                feat[k] = self.clean_line(feat[k])

            if not feat["title"]:
                feat["title"] = self.infer_title_from_experience(feat["experience"])

            feat["skills_list"] = [s.strip() for s in feat["skills"].split(",") if s.strip()]
            rows.append(feat)

        df = pd.DataFrame(rows)

        df = self.enrich_experience(df)
        df["education_enriched"] = df["education"].apply(self.enrich_education)

        return df

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalLLM:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

    def generate_summary(self, cv_text: str, job_title: str, job_description: str):
        """
        Kembalikan dict: summary, pros, cons
        """
        prompt = f"""
You are an HR assistant. Given the following CV and job description, provide:
1. A concise summary of the candidate.
2. Pros (strengths) relative to the job.
3. Cons (weaknesses or missing skills).

Job Title: {job_title}
Job Description: {job_description}
CV Content:
{cv_text}

Return in JSON format like:
{{"summary": "...", "pros": ["...", "..."], "cons": ["...", "..."]}}
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=300)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Coba parse JSON, fallback text jika gagal
        import json
        try:
            import re
            json_str = re.search(r"\{.*\}", output_text, re.DOTALL).group()
            return json.loads(json_str)
        except:
            return {"summary": output_text.strip(), "pros": [], "cons": []}

from llama_cpp import Llama

class RAGModel:
    def __init__(self, model_path, n_ctx=4096, n_threads=None):
        import os
        if n_threads is None:
            n_threads = max(1, os.cpu_count() // 2)

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False
        )

    def build_context(self, chunks):
        by_cv = {}

        for c in chunks:
            m = c["meta"]
            cv_id = m["cv_id"]
            by_cv.setdefault(cv_id, []).append(c)

        sections = []
        for cv_id, cv_chunks in by_cv.items():
            header = f"=== Candidate: {cv_id} | Overall Score: {cv_chunks[0]['meta']['overall_score']:.2f} ==="
            body = []

            for c in cv_chunks:
                m = c["meta"]
                body.append(
                    f"- Section: {m['section']} "
                    f"(section_score={m['section_score']:.2f})\n"
                    f"{c['text']}"
                )

            sections.append(header + "\n" + "\n".join(body))

        return "\n\n".join(sections)


    def answer(self, question, chunks, max_tokens=256):
        context = self.build_context(chunks)
        prompt = (
            "You are a professional HR analyst.\n"
            "Answer ONLY based on the provided CV context.\n\n"

            "If the question asks WHY or COMPARE:\n"
            "- Provide reasoning and comparison.\n"
            "- Use experience, skills, and role relevance.\n\n"

            "If the question asks WHO:\n"
            "- Identify the candidate and explain briefly.\n\n"

            "Context:\n"
            f"{context}\n\n"

            "Question:\n"
            f"{question}\n\n"

            "Answer:"
        )


        output = self.model(prompt, max_tokens=max_tokens, temperature=0)
        return output["choices"][0]["text"].strip()

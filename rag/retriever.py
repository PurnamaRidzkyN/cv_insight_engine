import faiss
import numpy as np
class Retriever:
    def __init__(self, index, chunks, embedder, top_k=5):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.top_k = top_k

    def query(self, query_text):
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)

        # ambil lebih banyak dulu
        _, idxs = self.index.search(q_emb, self.top_k * 3)

        seen_cv = set()
        results = []

        for i in idxs[0]:
            chunk = self.chunks[i]
            cv_id = chunk["meta"]["cv_id"]

            if cv_id not in seen_cv:
                results.append(chunk)
                seen_cv.add(cv_id)

            if len(results) >= self.top_k:
                break

        return results


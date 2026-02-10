from rag.ingest import CandidateIngestor
from rag.retriever import Retriever
from rag.rag_qa import RAGModel

def build_rag(df_top, top_n):
    ingestor = CandidateIngestor()
    ingestor.ingest_dataframe(df_top)

    index = ingestor.build_faiss_index()

    retriever = Retriever(
        index=index,
        chunks=ingestor.chunks,
        embedder=ingestor.embedder,
        top_k=top_n
    )

    rag_model = RAGModel(
        model_path="llm/gemma-3-4b-it-q4_0.gguf"
    )

    return ingestor, retriever, rag_model


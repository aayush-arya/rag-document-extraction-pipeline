"""In-memory hybrid index: dense vectors + BM25 over the same chunks.

Exact cosine search over a normalised numpy matrix gives the same results as a
flat FAISS index, with no extra dependency and no pickle files.  (For corpora of
hundreds of thousands of chunks, swap ``dense_scores`` for FAISS/pgvector.)
"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document

from .bm25 import BM25
from .embeddings import get_embeddings


class HybridVectorStore:
    def __init__(self, chunks: list[Document], embeddings, batch_size: int = 64):
        if not chunks:
            raise ValueError("Cannot build a vector store from zero chunks.")
        self.chunks = chunks
        self.embeddings = embeddings
        texts = [chunk.page_content for chunk in chunks]

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(embeddings.embed_documents(texts[start:start + batch_size]))
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.where(norms == 0, 1.0, norms)
        self.bm25 = BM25(texts)

    def __len__(self) -> int:
        return len(self.chunks)

    def dense_scores(self, query: str) -> np.ndarray:
        vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        return self.matrix @ (vector / norm)

    def bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.scores(query)


def create_vectorstore(chunks: list[Document]) -> HybridVectorStore:
    return HybridVectorStore(chunks, get_embeddings())

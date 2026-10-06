"""Hybrid retriever.

One generic "extract everything" query retrieving k=5 chunks cannot find a
specific fact (the "GGN / Gurgaon" note, "214 vs 219").  This retriever:

1. runs several rankers over the same chunks
     - dense   : embedding cosine similarity (meaning)
     - lexical : BM25 (exact words, IDs, numbers)
     - exact   : literal match of identifiers found in the query
                 (R-1841, 77B, GGN, 214 ...) weighted by rarity
2. fuses the rankings with Reciprocal Rank Fusion (no score calibration needed)
3. supports metadata filters (``where``), multi-query fusion
   (``invoke_many``), and structural lookups (``by_section``, ``by_table``,
   ``neighbors``) so callers can pull a whole table or section on demand.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Optional

import numpy as np
from langchain_core.documents import Document

_ID_PATTERNS = [
    re.compile(r"\b[A-Za-z]{1,8}-\d[\w-]*\b"),        # R-1841, R-1841-A, EX-0105
    re.compile(r"\b\d+[A-Za-z]\b"),                   # 77B, 7A
    re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?"),    # 214, 4.8%, 48,219
    re.compile(r"\b[A-Z]{2,8}\b"),                    # GGN, GB, GiB-like caps
]
_IGNORED_CAPS = {"AND", "THE", "FOR", "ALL", "NOT", "ANY", "ID", "OR", "IS", "OF", "TO", "IN"}

Where = Optional[object]  # dict | callable(metadata) -> bool | None


def extract_identifiers(query: str) -> list[str]:
    found: list[str] = []
    for quoted in re.findall(r"\"([^\"]{2,60})\"|'([^']{2,60})'", query):
        found.extend(q for q in quoted if q)
    for pattern in _ID_PATTERNS:
        for match in pattern.findall(query):
            token = match.strip(" ,.")
            if token and token.upper() not in _IGNORED_CAPS and len(token) >= 2:
                found.append(token)
    seen, unique = set(), []
    for token in found:
        if token.lower() not in seen:
            seen.add(token.lower())
            unique.append(token)
    return unique


class HybridRetriever:
    def __init__(self, store, k: int = 8, candidates: int = 40, rrf_k: int = 60,
                 dense_weight: float = 1.0, lexical_weight: float = 1.0,
                 exact_weight: float = 1.0):
        self.store = store
        self.k = k
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.weights = {"dense": dense_weight, "lexical": lexical_weight, "exact": exact_weight}
        self._lowered = [c.page_content.lower() for c in store.chunks]

    # ------------------------------------------------------------- filters
    def _mask(self, where: Where) -> np.ndarray:
        chunks = self.store.chunks
        if where is None:
            return np.ones(len(chunks), dtype=bool)
        if callable(where):
            return np.array([bool(where(c.metadata)) for c in chunks])
        mask = np.ones(len(chunks), dtype=bool)
        for key, wanted in where.items():
            allowed = set(wanted) if isinstance(wanted, (list, tuple, set)) else {wanted}
            mask &= np.array([c.metadata.get(key) in allowed for c in chunks])
        return mask

    # ------------------------------------------------------------- rankers
    @staticmethod
    def _top(scores: np.ndarray, mask: np.ndarray, n: int, floor: float = 0.0) -> list[int]:
        masked = np.where(mask, scores, -np.inf)
        order = np.argsort(-masked, kind="stable")[:n]
        return [int(i) for i in order if masked[i] > floor]

    def _exact_ranking(self, query: str, mask: np.ndarray) -> list[int]:
        identifiers = extract_identifiers(query)
        if not identifiers:
            return []
        n = len(self._lowered)
        scores = np.zeros(n, dtype=np.float32)
        for token in identifiers:
            pattern = re.compile(r"(?<![a-z0-9])" + re.escape(token.lower()) + r"(?![a-z0-9])")
            hits = [i for i, text in enumerate(self._lowered) if pattern.search(text)]
            if not hits:
                continue
            weight = math.log(1 + n / (1 + len(hits)))      # rarer identifier => bigger boost
            for i in hits:
                scores[i] += weight
        return self._top(scores, mask, self.candidates)

    def _rankings(self, query: str, mask: np.ndarray) -> list[tuple[str, list[int]]]:
        return [
            ("dense", self._top(self.store.dense_scores(query), mask, self.candidates, floor=-1.0)),
            ("lexical", self._top(self.store.bm25_scores(query), mask, self.candidates)),
            ("exact", self._exact_ranking(query, mask)),
        ]

    # --------------------------------------------------------------- search
    def _fuse(self, queries: list[str], where: Where) -> list[tuple[int, float]]:
        mask = self._mask(where)
        fused: dict[int, float] = {}
        for query in queries:
            for name, ranking in self._rankings(query, mask):
                for rank, index in enumerate(ranking):
                    fused[index] = fused.get(index, 0.0) + self.weights[name] / (self.rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

    def _as_documents(self, ranked: list[tuple[int, float]], k: int) -> list[Document]:
        docs = []
        for index, score in ranked[:k]:
            chunk = self.store.chunks[index]
            docs.append(Document(page_content=chunk.page_content,
                                 metadata={**chunk.metadata, "retrieval_score": round(score, 5)}))
        return docs

    def invoke(self, query: str, k: Optional[int] = None, where: Where = None) -> list[Document]:
        return self._as_documents(self._fuse([query], where), k or self.k)

    def invoke_many(self, queries: list[str], k: Optional[int] = None,
                    where: Where = None) -> list[Document]:
        """Fuse the rankings of several phrasings of the same information need."""
        return self._as_documents(self._fuse(list(queries), where), k or self.k)

    # ----------------------------------------------------------- structure
    def by_section(self, pattern: str, where: Where = None) -> list[Document]:
        regex = re.compile(pattern, re.I)
        mask = self._mask(where)
        return [c for i, c in enumerate(self.store.chunks)
                if mask[i] and c.metadata.get("section") and regex.search(c.metadata["section"])]

    def by_table(self, table_id: str) -> list[Document]:
        return [c for c in self.store.chunks if c.metadata.get("table_id") == table_id]

    def neighbors(self, doc: Document, window: int = 1) -> list[Document]:
        index = doc.metadata.get("chunk_index")
        if index is None:
            return []
        chunks = self.store.chunks
        lo, hi = max(0, index - window), min(len(chunks), index + window + 1)
        return [chunks[i] for i in range(lo, hi) if i != index]


def get_retriever(vectorstore, k: int = 8) -> HybridRetriever:
    return HybridRetriever(vectorstore, k=k)

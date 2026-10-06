"""Small dependency-free BM25 (Okapi) index.

The tokenizer is built for IDs and numbers: ``R-1841-A`` is indexed as the
whole token AND as its parts (``r``, ``1841``, ``a``), ``4.8%`` as ``4.8``,
so both exact and partial identifier queries hit.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_SPLIT = re.compile(r"[-_./]")
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "be", "by", "with", "as", "at", "it", "this", "that", "from", "all",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        tokens.append(token)
        if _SPLIT.search(token):
            tokens.extend(part for part in _SPLIT.split(token) if part)
    return [t for t in tokens if t not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(texts)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = np.zeros(self.n, dtype=np.float32)
        for index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths[index] = sum(counts.values())
            for term, freq in counts.items():
                self.postings[term].append((index, freq))
        self.avg_len = float(self.lengths.mean()) if self.n else 0.0
        self.idf = {
            term: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, freq in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / (self.avg_len or 1.0))
                scores[index] += idf * freq * (self.k1 + 1) / (freq + norm)
        return scores

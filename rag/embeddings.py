"""Embedding model + token accounting.

Why this file matters
---------------------
``all-MiniLM-L6-v2`` silently truncates every input at 256 word-pieces.  Table
text full of IDs and numbers (``R-1001 | 03/04/26 | North-01 ...``) costs 3-5
word-pieces per cell, so a 1,200-character chunk can lose its last half *before
it is embedded*, and the embedding never "sees" those rows.

Fix (three parts)
  1. default model = ``BAAI/bge-small-en-v1.5`` (512-token window, same speed
     class, noticeably better retrieval); override with ``EMBEDDING_MODEL``
  2. chunks are sized in *tokens of the actual embedding model* (see
     ``preprocessing/chunker.py``) so nothing can ever be truncated
  3. exact IDs / numbers are matched lexically (BM25 + exact match) in
     ``rag/retriever.py``, so retrieval does not depend on the embedding alone

Environment
  EMBEDDING_MODEL       HF model id (default BAAI/bge-small-en-v1.5)
  EMBEDDING_MAX_TOKENS  override the model's input window
  EMBEDDING_BACKEND     "hf" (default) or "hash" (offline, for tests/CI only)
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_BGE_QUERY = "Represent this sentence for searching relevant passages: "

MODEL_SPECS = {
    "all-minilm-l6-v2": {"max_tokens": 256, "query_prefix": "", "doc_prefix": ""},
    "all-minilm-l12-v2": {"max_tokens": 256, "query_prefix": "", "doc_prefix": ""},
    "bge-small-en-v1.5": {"max_tokens": 512, "query_prefix": _BGE_QUERY, "doc_prefix": ""},
    "bge-base-en-v1.5": {"max_tokens": 512, "query_prefix": _BGE_QUERY, "doc_prefix": ""},
    "bge-large-en-v1.5": {"max_tokens": 512, "query_prefix": _BGE_QUERY, "doc_prefix": ""},
    "e5-small-v2": {"max_tokens": 512, "query_prefix": "query: ", "doc_prefix": "passage: "},
    "e5-base-v2": {"max_tokens": 512, "query_prefix": "query: ", "doc_prefix": "passage: "},
}
UNKNOWN_SPEC = {"max_tokens": 256, "query_prefix": "", "doc_prefix": ""}


def get_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)


def get_spec() -> dict:
    name = get_model_name().lower()
    for key, spec in MODEL_SPECS.items():
        if key in name:
            return dict(spec)
    return dict(UNKNOWN_SPEC)


def get_max_tokens() -> int:
    override = os.getenv("EMBEDDING_MAX_TOKENS")
    return int(override) if override else get_spec()["max_tokens"]


# ---------------------------------------------------------------- tokens ----

def _heuristic_tokens(text: str) -> int:
    """Conservative word-piece estimate (IDs and numbers split into many pieces)."""
    return sum(max(1, math.ceil(len(w) / 3.2))
               for w in re.findall(r"[A-Za-z]+|\d+|[^\w\s]", text))


@lru_cache(maxsize=1)
def get_token_counter():
    """text -> token count, using the embedding model's own tokenizer."""
    if os.getenv("EMBEDDING_BACKEND", "hf").lower() == "hash":
        return _heuristic_tokens
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(get_model_name())
        tokenizer.model_max_length = 10 ** 9      # silence "sequence too long" warnings

        def count(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        return count
    except Exception:                             # offline / transformers missing
        return _heuristic_tokens


# ------------------------------------------------------------ embeddings ----

class HashingEmbeddings:
    """Deterministic offline embeddings (hashed word + char n-grams).

    Only meant for tests / CI where the HF model cannot be downloaded.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.lower())
        grams = words + [w[i:i + 4] for w in words for i in range(max(1, len(w) - 3))]
        for gram in grams:
            digest = hashlib.md5(gram.encode()).digest()
            vec[int.from_bytes(digest[:4], "little") % self.dim] += 1.0 if digest[4] & 1 else -1.0
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class Embedder:
    """Adds the model-specific query/passage prefixes (bge, e5) around any backend."""

    def __init__(self, inner, spec: dict):
        self.inner, self.spec = inner, spec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = self.spec["doc_prefix"]
        return self.inner.embed_documents([prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.spec["query_prefix"] + text)


@lru_cache(maxsize=1)
def get_embeddings() -> Embedder:
    spec = get_spec()
    if os.getenv("EMBEDDING_BACKEND", "hf").lower() == "hash":
        return Embedder(HashingEmbeddings(), {"query_prefix": "", "doc_prefix": ""})

    from langchain_huggingface import HuggingFaceEmbeddings

    inner = HuggingFaceEmbeddings(
        model_name=get_model_name(),
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    return Embedder(inner, spec)

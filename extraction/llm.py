"""One place that builds the Gemini chat model (used by extraction, RAG chain, vision OCR)."""

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"     # same default as before; override with GEMINI_MODEL


@lru_cache(maxsize=1)
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        temperature=0,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

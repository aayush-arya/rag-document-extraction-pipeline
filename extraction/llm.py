"""Configurable chat model construction with quota-triggered provider fallback."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"


def is_quota_exhausted(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status = status or getattr(response, "status_code", None)
    code = getattr(error, "code", "")
    message = f"{code} {error}".upper()
    return status == 429 or "RESOURCE_EXHAUSTED" in message or "429" in message


def _provider_settings(prefix: str = "LLM") -> tuple[str, str, str | None, str | None]:
    provider = os.getenv(f"{prefix}_PROVIDER", "gemini" if prefix == "LLM" else "").lower()
    model = os.getenv(f"{prefix}_MODEL") or (
        os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        if provider in {"gemini", "google", "google_genai"} else ""
    )
    base_url = os.getenv(f"{prefix}_BASE_URL")
    api_key = os.getenv(f"{prefix}_API_KEY")
    return provider, model, base_url, api_key


def _build_chat_model(provider: str, model: str, base_url=None, api_key=None):
    if provider in {"gemini", "google", "google_genai"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            temperature=0,
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        )

    if provider in {"openai", "openai-compatible", "openrouter"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise ImportError(
                "Install langchain-openai to use an OpenAI-compatible LLM provider."
            ) from error
        default_urls = {
            "openrouter": "https://openrouter.ai/api/v1",
        }
        default_keys = {
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
        }
        return ChatOpenAI(
            model=model,
            temperature=0,
            api_key=api_key or default_keys.get(provider) or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or default_urls.get(provider),
        )

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as error:
            raise ImportError("Install langchain-groq to use Groq.") from error
        return ChatGroq(
            model=model,
            temperature=0,
            api_key=api_key or os.getenv("GROQ_API_KEY"),
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:
            raise ImportError("Install langchain-ollama to use Ollama.") from error
        return ChatOllama(
            model=model,
            temperature=0,
            base_url=base_url or os.getenv("OLLAMA_BASE_URL"),
        )

    raise ValueError(
        f"Unsupported LLM provider {provider!r}; use gemini, openai-compatible, "
        "openrouter, groq, or ollama."
    )


class _FallbackRunnable:
    def __init__(self, primary, fallback_factory):
        self.primary = primary
        self.fallback_factory = fallback_factory

    def invoke(self, *args, **kwargs):
        try:
            return self.primary.invoke(*args, **kwargs)
        except Exception as error:
            if not is_quota_exhausted(error) or self.fallback_factory is None:
                raise
            logger.warning("Primary LLM quota exhausted; trying configured fallback provider.")
            return self.fallback_factory().invoke(*args, **kwargs)


class _FallbackChatModel:
    def __init__(self, primary, fallback_factory=None):
        self.primary = primary
        self.fallback_factory = fallback_factory

    def invoke(self, *args, **kwargs):
        return _FallbackRunnable(self.primary, self.fallback_factory).invoke(*args, **kwargs)

    def with_structured_output(self, schema):
        primary = self.primary.with_structured_output(schema)
        fallback = (
            (lambda: self.fallback_factory().with_structured_output(schema))
            if self.fallback_factory else None
        )
        return _FallbackRunnable(primary, fallback)


@lru_cache(maxsize=1)
def get_llm():
    provider, model, base_url, api_key = _provider_settings("LLM")
    if provider in {"gemini", "google", "google_genai"} and not model:
        model = DEFAULT_MODEL
    primary = _build_chat_model(provider, model, base_url, api_key)

    fallback_provider, fallback_model, fallback_url, fallback_key = _provider_settings(
        "LLM_FALLBACK"
    )
    fallback_factory = None
    if fallback_provider:
        fallback_factory = lambda: _build_chat_model(
            fallback_provider, fallback_model, fallback_url, fallback_key
        )
    return _FallbackChatModel(primary, fallback_factory)


def llm_cache_identity() -> str:
    provider, model, base_url, _ = _provider_settings("LLM")
    fallback_provider, fallback_model, fallback_url, _ = _provider_settings("LLM_FALLBACK")
    return "|".join((provider, model, base_url or "",
                     fallback_provider, fallback_model, fallback_url or ""))

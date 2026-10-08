"""Small local JSON cache for repeatable OCR and LLM operations."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _enabled(namespace: str) -> bool:
    specific = os.getenv(f"{namespace.upper()}_CACHE_ENABLED")
    value = specific if specific is not None else os.getenv("PIPELINE_CACHE_ENABLED", "1")
    return value.lower() not in {"0", "false", "no", "off"}


def _directory(namespace: str) -> Path:
    specific = os.getenv(f"{namespace.upper()}_CACHE_DIR")
    root = Path(specific) if specific else Path(os.getenv("PIPELINE_CACHE_DIR", "data/cache"))
    return root / namespace.lower()


def cache_get(namespace: str, key: str):
    if not _enabled(namespace):
        return None
    path = _directory(namespace) / f"{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read %s cache entry %s: %s", namespace, key, error)
        return None


def cache_put(namespace: str, key: str, value) -> None:
    if not _enabled(namespace):
        return
    directory = _directory(namespace)
    temporary = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, suffix=".tmp",
                prefix=f"{key}-", delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, directory / f"{key}.json")
    except (OSError, TypeError, ValueError) as error:
        logger.warning("Could not write %s cache entry %s: %s", namespace, key, error)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove incomplete cache file %s", temporary)

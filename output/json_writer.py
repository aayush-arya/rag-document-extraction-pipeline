"""Streaming JSON serialization for complete extraction results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _object_items(value):
    if isinstance(value, BaseModel):
        return ((name, getattr(value, name)) for name in type(value).model_fields)
    return value.items()


def _write_value(file, value: Any, level: int = 0) -> None:
    indent = "  " * level
    child_indent = "  " * (level + 1)

    if isinstance(value, BaseModel) or isinstance(value, dict):
        items = _object_items(value)
        file.write("{")
        first = True
        for key, item in items:
            if first:
                file.write("\n")
                first = False
            else:
                file.write(",\n")
            file.write(child_indent)
            file.write(json.dumps(str(key), ensure_ascii=False))
            file.write(": ")
            _write_value(file, item, level + 1)
        if not first:
            file.write("\n" + indent)
        file.write("}")
        return

    if isinstance(value, (list, tuple)):
        file.write("[")
        for index, item in enumerate(value):
            file.write("\n" if index == 0 else ",\n")
            file.write(child_indent)
            _write_value(file, item, level + 1)
        if value:
            file.write("\n" + indent)
        file.write("]")
        return

    encoder = json.JSONEncoder(ensure_ascii=False, default=str, allow_nan=True)
    for piece in encoder.iterencode(value):
        file.write(piece)


def write_json(extracted_data, output_path) -> str:
    """Write every field and row without materializing a second full document."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        _write_value(file, extracted_data)
        file.write("\n")
    return str(path)

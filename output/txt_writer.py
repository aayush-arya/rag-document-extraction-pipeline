"""Backward-compatible imports for report and JSON output writers."""

from .json_writer import write_json
from .report_writer import write_txt

__all__ = ["write_json", "write_txt"]

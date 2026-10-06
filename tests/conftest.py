import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EMBEDDING_BACKEND", "hash")      # offline: no model download in tests
os.environ.setdefault("OUTPUT_DIR", str(ROOT / "data" / "output"))

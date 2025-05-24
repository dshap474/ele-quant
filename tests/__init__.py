import sys
from pathlib import Path

# Ensure the package in src/ is importable when running tests without Poetry
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

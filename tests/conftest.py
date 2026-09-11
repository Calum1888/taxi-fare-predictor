from pathlib import Path
import sys

# Allows tests to import modules whether pytest is run from the repository root
# with source files in ./src, or directly from the src directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(PROJECT_ROOT))

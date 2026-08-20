"""Make the repository root and the training package importable for these tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lst_ai
sys.path.insert(0, str(Path(__file__).resolve().parent))       # lst_training

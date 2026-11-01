"""General configuration file for the project."""

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

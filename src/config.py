"""Shared paths and constants for all phases."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEO = ROOT / "ps3_geocoder"

# Resolve shared data directory dynamically
if (ROOT / "shared").exists():
    SHARED = ROOT / "shared"
elif "CN_SHARED_PATH" in os.environ:
    SHARED = Path(os.environ["CN_SHARED_PATH"])
elif (ROOT.parent / "shared").exists():
    SHARED = ROOT.parent / "shared"
else:
    candidates = list(ROOT.parent.glob("*shared*"))
    SHARED = candidates[0] if candidates else ROOT / "shared"

OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# Phase 1 thresholds
POI_FUZZY_THRESHOLD = 82.0
LOCALITY_THRESHOLD = 75.0

"""Shared paths and constants for all phases."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEO = ROOT / "ps3_geocoder"
SHARED = ROOT / "shared" if (ROOT / "shared").exists() else Path(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared")
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# Phase 1 thresholds (blueprint 3.4 / 3.6)
POI_FUZZY_THRESHOLD = 82.0
LOCALITY_THRESHOLD = 75.0

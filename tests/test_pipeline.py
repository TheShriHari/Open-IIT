"""Automated Unit & Integration Test Suite for PS3 Geocoder Pipeline.

Tests enforce the frozen technical safeguards:
1. test_generic_fraud_detection: Generic detector catches FA009's fake-spot clusters without knowing FA009 or its coordinates.
2. test_genuine_agent_integrity: 0 false positive cluster fraud on genuine field agents.
3. test_dwell_centroid_error_reduction: Canonical dwell coordinates improve accuracy over raw check-ins.
4. test_negative_evidence_search_quality: Quality scoring differentiates short vs thorough searches.
5. test_out_address_completeness: All 3,117 addresses exist, exactly 237 OUT addresses tagged CANNOT_GEOCODE.
6. test_evaluation_split_isolation: Test split accounts excluded from training anchors (zero leakage).
7. test_canonical_18_column_schema: Verifies all 18 columns in geocoder_output.csv.
"""

import math
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import GEO, OUT, SHARED
from fraud_detection import detect_fraudulent_visits
from dwell_kinematics import extract_dwell_coordinates, apply_dwell_corrections
from negative_evidence import compute_search_quality, compute_repulsion_penalty
from cross_account import build_street_anchors
from calibration import compute_conformal_quantile


class TestGeocoderPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visits = pd.read_csv(SHARED / "field_visits.csv")
        cls.gps = pd.read_csv(GEO / "visit_gps_points.csv")
        cls.addr = pd.read_csv(SHARED / "addresses.csv")
        cls.splits = pd.read_csv(SHARED / "splits.csv")
        cls.surveyed = pd.read_csv(GEO / "surveyed_addresses.csv")

    def test_generic_fraud_detection(self):
        """Test generic detector identifies FA009's fixed-spot check-in clusters without hardcoded coordinates."""
        clean, fraud = detect_fraudulent_visits(self.visits, self.gps)
        
        # Must detect over 200 fraudulent visits from agent FA009
        fa9_fraud = fraud[fraud["agent_id"] == "FA009"]
        self.assertGreaterEqual(len(fa9_fraud), 220, "Generic fraud detector failed to catch FA009 clusters")
        
        # Must catch both photo reuse and cluster spoofing
        self.assertTrue((fa9_fraud["fraud_type"] == "fixed_location_cluster_spoof").any())
        self.assertTrue((fa9_fraud["fraud_type"] == "photo_reuse_fraud").any())

    def test_genuine_agent_integrity(self):
        """Test generic detector generates zero false positive cluster fraud on genuine agents."""
        clean, fraud = detect_fraudulent_visits(self.visits, self.gps)
        cluster_spoof_agents = set(fraud[fraud["fraud_type"] == "fixed_location_cluster_spoof"]["agent_id"])
        
        # Only FA009 should have fixed-location cluster spoofing
        self.assertEqual(cluster_spoof_agents, {"FA009"}, f"Unexpected agents flagged for cluster spoofing: {cluster_spoof_agents}")

    def test_dwell_centroid_error_reduction(self):
        """Test canonical dwell coordinates reduce doorstep error on surveyed positive visits."""
        clean, _ = detect_fraudulent_visits(self.visits, self.gps)
        can_v, dwell_audit = apply_dwell_corrections(clean, self.gps)
        
        # Dwell audit must contain extracted clusters
        self.assertGreater(len(dwell_audit), 1000)
        
        # Evaluate on surveyed visits with positive outcomes
        m = can_v.merge(self.surveyed, on="address_id").merge(dwell_audit[["visit_id", "original_checkin_x", "original_checkin_y"]], on="visit_id")
        m["err_orig"] = np.hypot(m["original_checkin_x"] - m["surveyed_x"], m["original_checkin_y"] - m["surveyed_y"])
        m["err_dwell"] = np.hypot(m["checkin_x"] - m["surveyed_x"], m["checkin_y"] - m["surveyed_y"])
        
        for outcome in ["locked_premises", "met_borrower", "met_family"]:
            sub = m[m["outcome"] == outcome]
            if len(sub) >= 5:
                self.assertLess(
                    sub["err_dwell"].median(),
                    sub["err_orig"].median(),
                    f"Dwell correction failed to reduce median error for outcome {outcome}"
                )

    def test_negative_evidence_search_quality(self):
        """Test search quality score differentiates low-effort vs thorough searches."""
        # Simulated superficial 5-point, 20-second search
        short_trail = pd.DataFrame({
            "point_ts": [
                "2026-04-01 10:00:00", "2026-04-01 10:00:05", "2026-04-01 10:00:10",
                "2026-04-01 10:00:15", "2026-04-01 10:00:20"
            ],
            "x": [100.0, 101.0, 102.0, 101.0, 100.0],
            "y": [200.0, 201.0, 200.0, 201.0, 200.0],
            "accuracy_m": [25.0, 25.0, 25.0, 25.0, 25.0]
        })
        
        # Simulated thorough 60-point, 600-second search covering 200m
        thorough_trail = pd.DataFrame({
            "point_ts": pd.date_range("2026-04-01 10:00:00", periods=60, freq="10s").astype(str),
            "x": np.linspace(100.0, 300.0, 60),
            "y": np.linspace(200.0, 350.0, 60),
            "accuracy_m": [10.0] * 60
        })
        
        q_short = compute_search_quality(short_trail)
        q_thorough = compute_search_quality(thorough_trail)
        
        self.assertLess(q_short, 0.35, f"Short search scored too high: {q_short}")
        self.assertGreater(q_thorough, 0.70, f"Thorough search scored too low: {q_thorough}")
        self.assertGreater(q_thorough, q_short * 2.0, "Thorough search should score over 2x short search")

    def test_out_address_completeness(self):
        """Test output/geocoder_output.csv has exactly 3,117 rows with 237 OUT addresses tagged CANNOT_GEOCODE."""
        out_path = OUT / "geocoder_output.csv"
        if not out_path.exists():
            from pipeline import run_pipeline
            run_pipeline()
            
        df = pd.read_csv(out_path)
        self.assertEqual(len(df), len(self.addr), f"Expected {len(self.addr)} rows, found {len(df)}")
        
        out_rows = df[df["town_id"] == "OUT"]
        self.assertEqual(len(out_rows), 237, f"Expected 237 OUT addresses, found {len(out_rows)}")
        self.assertTrue((out_rows["action"] == "CANNOT_GEOCODE").all(), "All OUT addresses must be CANNOT_GEOCODE")
        self.assertTrue((out_rows["can_geocode"] == False).all(), "All OUT addresses must have can_geocode=False")
        self.assertTrue(out_rows["px"].isna().all(), "OUT addresses must have px=NaN")
        self.assertTrue(out_rows["py"].isna().all(), "OUT addresses must have py=NaN")

    def test_evaluation_split_isolation(self):
        """Test ground-truth test accounts are strictly isolated from cross-account anchors."""
        test_accs = set(self.splits[self.splits["split"] == "test"]["account_id"])
        test_addrs = set(self.addr[self.addr["account_id"].isin(test_accs)]["address_id"])
        
        # Build dummy master dataframe
        dummy_master = pd.DataFrame({
            "address_id": list(test_addrs),
            "town_id": ["TW01"] * len(test_addrs),
            "clean_street": ["MG Road"] * len(test_addrs),
            "clean_locality": ["Central"] * len(test_addrs),
            "clean_pincode": ["560001"] * len(test_addrs),
            "px": [100.0] * len(test_addrs),
            "py": [200.0] * len(test_addrs),
            "tier": ["visits_agree"] * len(test_addrs),
        })
        
        # When excluded, no anchors should be formed from these test addresses
        loc_a, gen_a = build_street_anchors(dummy_master, excluded_address_ids=test_addrs)
        self.assertEqual(len(loc_a), 0, "Test addresses leaked into locality anchor map")
        self.assertEqual(len(gen_a), 0, "Test addresses leaked into general anchor map")

    def test_canonical_18_column_schema(self):
        """Test output/geocoder_output.csv conforms to the exact 18 canonical audit columns."""
        out_path = OUT / "geocoder_output.csv"
        self.assertTrue(out_path.exists())
        df = pd.read_csv(out_path)
        
        expected_cols = [
            "address_id",
            "account_id",
            "town_id",
            "px",
            "py",
            "R90_meters",
            "tier",
            "action",
            "can_geocode",
            "landmark_hint",
            "reason",
            "visit_evidence_count",
            "reliable_visit_count",
            "best_evidence_source",
            "fraud_flag",
            "negative_evidence_score",
            "cross_account_support",
            "remark_confidence",
        ]
        self.assertEqual(list(df.columns), expected_cols)


if __name__ == "__main__":
    unittest.main()

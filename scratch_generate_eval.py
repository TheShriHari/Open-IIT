import sys
sys.path.append("src")
import pandas as pd
import numpy as np

from config import GEO, OUT, SHARED

# Load truth and baseline
truth = pd.read_csv(GEO / "surveyed_addresses.csv")
base = pd.read_csv(GEO / "baseline_geocodes.csv")
audit = pd.read_csv(OUT / "remark_extracted_corrections.csv")
mp = pd.read_csv(OUT / "master_pins.csv")
radii = pd.read_csv(OUT / "calibration_radii.csv")
splits = pd.read_csv(SHARED / "splits.csv")

# 1. Remark stats
n_total = len(audit)
n_corr = (audit.correction_confidence >= 0.7).sum()
n_dir = audit.detected_relation.notna().sum()
n_lm = audit.detected_landmark.notna().sum()
mean_conf_all = audit.final_remark_confidence.mean()
cues = audit[audit.remark_status != "UNRESOLVED"]
mean_conf_cues = cues.final_remark_confidence.mean()
n_unres = (audit.remark_status == "UNRESOLVED").sum()
n_conflict = (audit.remark_status == "CONFLICTING").sum()

print("=== REMARK PARSER EVALUATION ===")
print(f"Total remarks processed: {n_total}")
print(f"Detected correction:     {n_corr} ({n_corr/n_total*100:.2f}%)")
print(f"Recognized direction:    {n_dir} ({n_dir/n_total*100:.2f}%)")
print(f"Recognized landmark:     {n_lm} ({n_lm/n_total*100:.2f}%)")
print(f"Average confidence (all):  {mean_conf_all:.3f}")
print(f"Average confidence (cues): {mean_conf_cues:.3f}")
print(f"Marked UNRESOLVED:       {n_unres} ({n_unres/n_total*100:.2f}%)")
print(f"Marked CONFLICTING:      {n_conflict} ({n_conflict/n_total*100:.2f}%)")
print("\nRemark Status Breakdown:")
print(audit.remark_status.value_counts())

# 2. 10 Real Examples from the dataset
print("\n=== 10 REAL DATASET EXAMPLES ===")
sample_ids = [
    "VS000003", "VS000006", "VS000009", "VS000011", "VS000017",
    "VS000028", "VS000035", "VS000007", "VS000140", "VS000001"
]
samples = audit[audit.visit_id.isin(sample_ids)].copy()
for i, r in enumerate(samples.itertuples(), 1):
    print(f"\nExample {i}: Visit {r.visit_id} (Address: {r.address_id})")
    print(f"  Raw Remark: {r.remark_raw}")
    print(f"  Interpreted Meaning: Correction={r.correction_detected}, Relation={r.detected_relation}, Landmark={r.detected_landmark}")
    print(f"  Fuzzy Confidences: Lexical={r.token_match_score:.2f}, Correction={r.correction_confidence:.2f}, Direction={r.direction_confidence:.2f}, Landmark={r.landmark_confidence:.2f} -> Final={r.final_remark_confidence:.3f}")
    print(f"  Status: {r.remark_status}")
    print(f"  Reason: {r.reason}")

# 3. Geocoder Comparison on 100 Surveyed Addresses
d = truth.merge(mp, on="address_id").merge(base, on="address_id").merge(splits, on="account_id")
d["err_model"] = np.hypot(d.px - d.surveyed_x, d.py - d.surveyed_y)
d["err_base"] = np.hypot(d.geocoder_x - d.surveyed_x, d.geocoder_y - d.surveyed_y)

print("\n=== GEOCODER METRICS ON 100 SURVEYED ADDRESSES ===")
print("Baseline Geocoder:")
print(f"  Median error: {d.err_base.median():.1f}m, P90 error: {d.err_base.quantile(0.9):.1f}m")
print("Current Geocoder (with Fuzzy Remarks):")
print(f"  Median error: {d.err_model.median():.1f}m, P90 error: {d.err_model.quantile(0.9):.1f}m")

print("\nPer-tier breakdown (Current Geocoder + Remarks):")
g = d.groupby("tier").agg(
    count=("err_model", "size"),
    model_med=("err_model", "median"),
    model_p90=("err_model", lambda s: s.quantile(0.9)),
    base_med=("err_base", "median")
).round(1)
print(g)

# Promoted out of baseline_coarse
st = pd.read_csv(OUT / "structured_addresses.csv")
print("\nAddress tier counts in full dataset (2880 master pins):")
print(mp.tier.value_counts())

import sys
sys.path.append("src")
import pandas as pd
import numpy as np

from config import GEO, OUT, SHARED
from scratch_batch_test import parse_remark_full

truth = pd.read_csv(GEO / "surveyed_addresses.csv")
visits = pd.read_csv(SHARED / "field_visits.csv")
addr = pd.read_csv(SHARED / "addresses.csv")
base = pd.read_csv(GEO / "baseline_geocodes.csv")
pois = pd.read_csv(GEO / "landmarks_poi.csv")

ACCEPT_OUTCOMES = ["locked_premises", "met_borrower", "met_family", "neighbour_says_shifted"]
MAX_GPS_ACC = 30.0
FAKE_PHOTO_MIN_ACCOUNTS = 3
AGREE_SPREAD_M = 100.0

def weighted_median(vals, weights):
    v, w = np.asarray(vals, float), np.asarray(weights, float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return v[np.searchsorted(c, 0.5 * c[-1])]

# Filter visits as in Phase 2
v = visits[visits.outcome.isin(ACCEPT_OUTCOMES) & (visits.gps_accuracy_m <= MAX_GPS_ACC)].copy()
n_acc = visits.groupby("photo_hash").account_id.nunique()
v = v[~v.photo_hash.isin(n_acc[n_acc >= FAKE_PHOTO_MIN_ACCOUNTS].index)].copy()
v = v.merge(addr[["address_id", "town_id"]], on="address_id", how="left")

# Parse remarks
parsed_rows = []
for row in v.itertuples(index=False):
    p = parse_remark_full(row.remark, row.town_id, pois, row.checkin_x, row.checkin_y)
    parsed_rows.append(p)
df_p = pd.DataFrame(parsed_rows)
v["remark_confidence"] = df_p["remark_confidence"].values
v["remark_status"] = df_p["remark_status"].values
v["correction_detected"] = df_p["correction_detected"].values

# Baseline weight
v["w_base"] = 1.0 / v.gps_accuracy_m.clip(lower=1.0) ** 2

# Remark multiplier
rem_mult = np.ones(len(v))
for i, status in enumerate(v.remark_status):
    c = v.remark_confidence.iloc[i]
    if status == "STRONG_CORRECTION":
        rem_mult[i] = 1.0 + 2.0 * c
    elif status == "USEFUL_SPATIAL_CUE":
        rem_mult[i] = 1.0 + 1.0 * c
    elif status == "CONFLICTING":
        rem_mult[i] = 0.2
    else:
        rem_mult[i] = 1.0
v["w_fused"] = v["w_base"] * rem_mult

# Compute pins under baseline vs fused
rows_base = []
rows_fused = []
for aid, g in v.groupby("address_id"):
    bx, by = weighted_median(g.checkin_x, g.w_base), weighted_median(g.checkin_y, g.w_base)
    fx, fy = weighted_median(g.checkin_x, g.w_fused), weighted_median(g.checkin_y, g.w_fused)
    n = len(g)
    spread_b = float(np.hypot(g.checkin_x - bx, g.checkin_y - by).max()) if n >= 2 else 0.0
    spread_f = float(np.hypot(g.checkin_x - fx, g.checkin_y - fy).max()) if n >= 2 else 0.0
    tier_b = "visit_1" if n == 1 else "visits_agree" if spread_b <= AGREE_SPREAD_M else "visits_disagree"
    tier_f = "visit_1" if n == 1 else "visits_agree" if spread_f <= AGREE_SPREAD_M else "visits_disagree"
    max_rem_conf = float(g.remark_confidence.max())
    rows_base.append((aid, bx, by, tier_b, n))
    rows_fused.append((aid, fx, fy, tier_f, n, max_rem_conf))

df_b = pd.DataFrame(rows_base, columns=["address_id", "bx", "by", "tier_b", "n_b"])
df_f = pd.DataFrame(rows_fused, columns=["address_id", "fx", "fy", "tier_f", "n_f", "remark_conf"])

# Evaluate on surveyed
tb = truth.merge(df_b, on="address_id")
tf = truth.merge(df_f, on="address_id")
tb["err"] = np.hypot(tb.bx - tb.surveyed_x, tb.by - tb.surveyed_y)
tf["err"] = np.hypot(tf.fx - tf.surveyed_x, tf.fy - tf.surveyed_y)

print(f"Evaluated on {len(tb)} visit-backed surveyed addresses:")
print(f"Baseline weights: median error = {tb.err.median():.2f}m, p90 = {tb.err.quantile(0.9):.2f}m")
print(f"Fused weights:    median error = {tf.err.median():.2f}m, p90 = {tf.err.quantile(0.9):.2f}m")

diff = tf.err - tb.err
print(f"Addresses improved: {(diff < -0.1).sum()}, unchanged: {(diff.abs() <= 0.1).sum()}, worsened: {(diff > 0.1).sum()}")
if (diff < -0.1).sum() > 0:
    print("Improved addresses:")
    print(tf[diff < -0.1][["address_id", "err", "tier_f", "remark_conf"]])

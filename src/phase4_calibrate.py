"""Phase 4 - Split conformal R90 calibration per tier (blueprint section 6).

Dev radii (train+validation) are used only to test coverage on the held-out test rows.
Final radii use all 100 surveyed rows. Output: output/calibration_radii.csv
"""
import math

import numpy as np
import pandas as pd

from config import GEO, OUT, SHARED

ALPHA = 0.10
MIN_N = 9
# blueprint 6.4 benchmark values, used only if a tier still has < MIN_N rows
BENCHMARK = {"visits_agree": 48, "visit_1": 77, "baseline_street": 162,
             "landmark": 358, "visits_disagree": 388, "baseline_coarse": 3710}


def conformal_quantile(errors, alpha=ALPHA):
    e = np.sort(np.asarray(errors, float))
    k = math.ceil((len(e) + 1) * (1 - alpha))
    return e[k - 1] if k <= len(e) else np.nan


def main():
    mp = pd.read_csv(OUT / "master_pins.csv")
    truth = pd.read_csv(GEO / "surveyed_addresses.csv")
    splits = pd.read_csv(SHARED / "splits.csv")
    d = truth.merge(mp, on="address_id").merge(splits, on="account_id")
    d["err"] = np.hypot(d.px - d.surveyed_x, d.py - d.surveyed_y)
    dev, test = d[d.split != "test"], d[d.split == "test"]

    rows = []
    for t in BENCHMARK:
        r_dev = conformal_quantile(dev.err[dev.tier == t])
        r_all = conformal_quantile(d.err[d.tier == t])
        src = "conformal"
        if np.isnan(r_all):
            r_all, src = BENCHMARK[t], "benchmark"
        rows.append((t, (dev.tier == t).sum(), r_dev, (d.tier == t).sum(), r_all, src))
    R = pd.DataFrame(rows, columns=["tier", "n_dev", "R90_dev", "n_all", "R90_final", "source"])
    R.to_csv(OUT / "calibration_radii.csv", index=False)
    print(R.round(1).to_string(index=False))

    # held-out coverage with dev radii
    t = test.merge(R[["tier", "R90_dev"]], on="tier")
    fin = t[t.R90_dev.notna()]
    print("\ntest coverage: %.2f (%d of %d finite-radius rows)" % (
        (fin.err <= fin.R90_dev).mean(), (fin.err <= fin.R90_dev).sum(), len(fin)))
    print("test rows outside radius:")
    print(fin[fin.err > fin.R90_dev][["address_id", "tier", "err", "R90_dev"]].round(1).to_string(index=False))

    # leave-one-out coverage per tier on all 100 rows
    print("\nleave-one-out coverage:")
    for tier, g in d.groupby("tier"):
        if len(g) - 1 < MIN_N:
            print(f"  {tier:16s} n={len(g)}  (too few)")
            continue
        hits = [e <= conformal_quantile(g.err.drop(i)) for i, e in g.err.items()]
        print(f"  {tier:16s} n={len(g):3d}  {np.mean(hits):.2f}")


if __name__ == "__main__":
    main()

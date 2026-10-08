"""Fold-Safe Cross-Account Street-Level Place Resolution.

Connects addresses on the same street/locality across borrower accounts:
1. Compiles empirical street anchors (median coordinates of visited accounts in visits_agree or visit_1).
2. Prevents data leakage by strictly excluding ground-truth evaluation accounts when compiling anchors.
3. Promotes unvisited baseline_coarse accounts sharing the exact street and locality to 'cross_account_street'.
4. Replaces synthetic '+10m along x-axis' door geometry with empirical street centroids.

Outputs: output/cross_account_promotions.csv
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from config import OUT


def build_street_anchors(
    master_df: pd.DataFrame,
    excluded_address_ids: Optional[Set[str]] = None,
    max_street_spread_m: float = 120.0,
) -> Tuple[Dict[Tuple, Tuple[float, float, int]], Dict[Tuple, Tuple[float, float, int]]]:
    """Compiles empirical street anchors from verified visit-pinned accounts.
    
    Args:
        master_df: DataFrame containing address_id, town_id, clean_pincode, clean_locality, clean_street, px, py, tier.
        excluded_address_ids: Address IDs to exclude (ensures fold-safety / zero evaluation leakage).
        max_street_spread_m: Maximum acceptable spatial spread to confirm unambiguous street geometry.
    
    Returns:
        (locality_anchor_map, general_street_anchor_map)
        Key: (town_id, pincode, locality, street) -> (median_x, median_y, count)
    """
    excluded = excluded_address_ids or set()

    # Eligible ground-truth anchors: verified physical visits, non-empty street
    anchors = master_df[
        master_df["tier"].isin(["visits_agree", "visit_1"])
        & (~master_df["address_id"].isin(excluded))
        & master_df["clean_street"].notna()
        & (master_df["clean_street"] != "")
    ].copy()

    anchors["pincode_str"] = anchors["clean_pincode"].fillna("").astype(str).str.replace(".0", "", regex=False)
    anchors["locality_str"] = anchors["clean_locality"].fillna("")

    # 1. Locality-specific anchors: (town_id, pincode, locality, street)
    loc_anchors = {}
    for (tid, pcode, loc, st), g in anchors[anchors["locality_str"] != ""].groupby(
        ["town_id", "pincode_str", "locality_str", "clean_street"]
    ):
        mx, my = float(g["px"].median()), float(g["py"].median())
        spread = float(np.hypot(g["px"] - mx, g["py"] - my).max()) if len(g) >= 2 else 0.0
        if spread <= max_street_spread_m:
            loc_anchors[(tid, pcode, loc, st)] = (round(mx, 1), round(my, 1), len(g))

    # 2. General street anchors (when locality is unpopulated or matches street across town):
    gen_anchors = {}
    for (tid, pcode, st), g in anchors.groupby(["town_id", "pincode_str", "clean_street"]):
        mx, my = float(g["px"].median()), float(g["py"].median())
        spread = float(np.hypot(g["px"] - mx, g["py"] - my).max()) if len(g) >= 2 else 0.0
        if spread <= max_street_spread_m:
            gen_anchors[(tid, pcode, st)] = (round(mx, 1), round(my, 1), len(g))

    return loc_anchors, gen_anchors


def resolve_cross_account_streets(
    master_df: pd.DataFrame,
    loc_anchors: Dict[Tuple, Tuple[float, float, int]],
    gen_anchors: Dict[Tuple, Tuple[float, float, int]],
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Promotes eligible unvisited accounts to empirical street anchors.
    
    Returns:
        updated_master: Master DataFrame with promoted cross-account pins.
        promotions_audit: Audit DataFrame saved to output/cross_account_promotions.csv.
    """
    m = master_df.copy()
    promotions = []

    m["pincode_str"] = m["clean_pincode"].fillna("").astype(str).str.replace(".0", "", regex=False)
    m["locality_str"] = m["clean_locality"].fillna("")

    for idx, r in m.iterrows():
        # Only unvisited coarse addresses with recognized street name
        if r["tier"] in ("baseline_coarse", "visits_disagree") and r["clean_street"]:
            tid = r["town_id"]
            pcode = r["pincode_str"]
            loc = r["locality_str"]
            st = r["clean_street"]

            loc_key = (tid, pcode, loc, st)
            gen_key = (tid, pcode, st)

            anchor = None
            source_type = None

            if loc_key in loc_anchors:
                anchor = loc_anchors[loc_key]
                source_type = "locality_street_anchor"
            elif (not loc) and (gen_key in gen_anchors):
                anchor = gen_anchors[gen_key]
                source_type = "general_street_anchor"

            if anchor is not None:
                ax, ay, a_count = anchor
                old_t = r["tier"]
                old_x, old_y = r["px"], r["py"]

                # Gating:
                # baseline_coarse has 0 visits -> promote to street anchor
                # visits_disagree has visits -> only promote if anchor is concordant (<= 150m)
                should_promote = False
                if old_t == "baseline_coarse":
                    should_promote = True
                elif old_t == "visits_disagree":
                    dist = float(np.hypot(ax - old_x, ay - old_y))
                    if dist <= 150.0:
                        should_promote = True

                if should_promote:
                    m.at[idx, "px"] = ax
                    m.at[idx, "py"] = ay
                    m.at[idx, "tier"] = "cross_account_street"

                    promotions.append({
                        "address_id": r["address_id"],
                        "account_id": r.get("account_id", ""),
                        "town_id": tid,
                        "street_name": st,
                        "old_tier": old_t,
                        "old_px": round(old_x, 1) if pd.notna(old_x) else np.nan,
                        "old_py": round(old_y, 1) if pd.notna(old_y) else np.nan,
                        "new_px": ax,
                        "new_py": ay,
                        "new_tier": "cross_account_street",
                        "anchor_support_accounts": a_count,
                        "anchor_type": source_type,
                    })

    promo_df = pd.DataFrame(promotions)
    out_file = output_path or (OUT / "cross_account_promotions.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    promo_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"Cross-Account Resolution: Promoted {len(promo_df)} addresses to 'cross_account_street' anchor.")
    print(f"Saved promotions audit to {out_file}")

    return m, promo_df

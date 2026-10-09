"""Step 1: load every CSV and join them into two master tables.

addr_df  - one row per address (3,117): text, town, lender format, split, baseline pin.
visit_df - one row per field visit (5,578): check-in, outcome, remark, photo, plus a
           summary of the GPS trail (duration, path length, straightness, trail end, longest
           stationary dwell), and the visit position pos_x/pos_y (pos_source): the outcome's
           preferred source from POS_PREFERENCE, else dwell -> stationary tail (trailing steps
           < 15 m, >= 2 points) -> check-in. checkin_x/y stay the raw check-in.

surveyed_addresses.csv (the answer key) is deliberately NOT loaded here.
Only evaluate.py reads it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
N_ADDRESSES = 3117
TAIL_MIN_POINTS = 2
# stationary-dwell detector (stationary_dwell)
DWELL_MAX_SPEED = 0.4      # m/s
DWELL_MAX_ACC = 20         # m, GPS point accuracy
DWELL_MIN_S = 120
DWELL_MAX_SPREAD_M = 15
# preferred visit position per outcome, chosen on train visits with both a dwell and a tail, vs the
# leave-one-out median check-in of other reliable visits (median / p90 m, tail vs dwell):
#   met_family 15.7/70.3 vs 15.3/68.1, neighbour_says_shifted 14.4/38.9 vs 14.4/38.8 -> dwell
#   met_borrower 17.6/185.6 vs 18.0/187.3, locked_premises 12.6/51.5 vs 11.6/52.4,
#   cash_collected 10.6/144.6 vs 10.7/145.3 -> tail (medians within 1 m, lower p90);
#   no_such_person: only 14 visits with both -> tail. Missing -> dwell, tail, check-in in that order.
POS_PREFERENCE = {"met_family": "dwell", "neighbour_says_shifted": "dwell"}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def map_tables():
    """The static map: towns, localities and landmarks."""
    return read("towns.csv"), read("localities.csv"), read("landmarks_poi.csv")


def load_addresses() -> pd.DataFrame:
    addr = read("addresses.csv")
    acc = read("accounts.csv")[["account_id", "lender_id", "income_type", "preferred_language"]]
    lenders = read("lenders.csv")[["lender_id", "kyc_address_format"]]
    splits = read("splits.csv")
    base = read("baseline_geocodes.csv").rename(
        columns={"geocoder_x": "base_x", "geocoder_y": "base_y", "precision": "base_precision"}
    )
    df = (
        addr.merge(acc, on="account_id", how="left")
        .merge(lenders, on="lender_id", how="left")
        .merge(splits, on="account_id", how="left")
        .merge(base, on="address_id", how="left")
    )
    assert len(df) == len(addr), "join created duplicate address rows"
    return df


def summarise_trails() -> pd.DataFrame:
    """Squeeze each visit's GPS breadcrumb trail into a few numbers."""
    g = read("visit_gps_points.csv").sort_values(["visit_id", "seq"])
    g["ts"] = pd.to_datetime(g["point_ts"])
    g["step_m"] = np.hypot(g.groupby("visit_id").x.diff(), g.groupby("visit_id").y.diff()).fillna(0)
    agg = g.groupby("visit_id").agg(
        trail_points=("seq", "size"),
        trail_start=("ts", "min"),
        trail_end_ts=("ts", "max"),
        path_m=("step_m", "sum"),
        x0=("x", "first"),
        y0=("y", "first"),
        trail_end_x=("x", "last"),
        trail_end_y=("y", "last"),
    )
    agg["trail_minutes"] = (agg.trail_end_ts - agg.trail_start).dt.total_seconds() / 60
    direct = np.hypot(agg.trail_end_x - agg.x0, agg.trail_end_y - agg.y0)
    # 1.0 = rode straight to the spot, near 0 = wandered / searched
    agg["straightness"] = (direct / agg.path_m.clip(lower=1)).clip(upper=1)

    # time spent standing still at the end of the trail (steps < 15 m per ~30 s)
    def end_dwell(v):
        moving = (v.step_m.values >= 15)[::-1]
        still = np.argmax(moving) if moving.any() else len(moving)
        return still * 30

    agg["trail_end_still_s"] = g.groupby("visit_id")[["step_m"]].apply(end_dwell)

    # stationary tail: the trailing run of steps < 15 m, plus the point where that run starts
    moving = (g.step_m >= 15).astype(int)
    after_last_move = moving[::-1].groupby(g.visit_id[::-1]).cumsum()[::-1] == 0
    starts_run = after_last_move.groupby(g.visit_id).shift(-1, fill_value=False) & ~after_last_move
    tail = g[after_last_move | starts_run]
    agg = agg.join(tail.groupby("visit_id").agg(tail_n=("x", "size"), tail_x=("x", "mean"), tail_y=("y", "mean")))
    agg = agg.join(stationary_dwell(g).set_index("visit_id"))
    return agg.drop(columns=["trail_start", "trail_end_ts", "x0", "y0"]).reset_index()


def stationary_dwell(g: pd.DataFrame) -> pd.DataFrame:
    """Longest stop in each trail: visit_id, dwell_x, dwell_y, trail_dwell_s (visits without one are absent).

    A point is stationary when its speed (step / time gap; first point 0) is < DWELL_MAX_SPEED and
    accuracy_m <= DWELL_MAX_ACC. Consecutive stationary points form a run; a run counts when it has
    >= 2 points, lasts >= DWELL_MIN_S and stays within DWELL_MAX_SPREAD_M of its centroid.
    Position = accuracy-weighted mean (1 / max(acc, 1)^2) of the longest run (earliest on ties).
    """
    g = g.sort_values(["visit_id", "seq"]).copy()
    ts = pd.to_datetime(g["point_ts"])
    by = g.groupby("visit_id")
    step = np.hypot(by.x.diff(), by.y.diff())
    gap = ts.groupby(g.visit_id).diff().dt.total_seconds().clip(lower=1)
    speed = (step / gap).fillna(0)
    g["ts_s"] = (ts - ts.min()).dt.total_seconds()
    still = (speed < DWELL_MAX_SPEED) & (g.accuracy_m <= DWELL_MAX_ACC)
    new_run = (still != still.groupby(g.visit_id).shift(fill_value=False)) | (g.visit_id != g.visit_id.shift())
    g["run"] = new_run.cumsum()
    s = g[still].copy()
    s["w"] = 1 / s.accuracy_m.clip(lower=1) ** 2
    s["wx"], s["wy"] = s.w * s.x, s.w * s.y
    s["cx"] = s.groupby("run").x.transform("mean")
    s["cy"] = s.groupby("run").y.transform("mean")
    s["spread"] = np.hypot(s.x - s.cx, s.y - s.cy)
    runs = s.groupby("run").agg(visit_id=("visit_id", "first"), n=("x", "size"), t0=("ts_s", "min"),
                                t1=("ts_s", "max"), spread=("spread", "max"),
                                w=("w", "sum"), wx=("wx", "sum"), wy=("wy", "sum"))
    runs["trail_dwell_s"] = runs.t1 - runs.t0
    runs = runs[(runs.n >= 2) & (runs.trail_dwell_s >= DWELL_MIN_S) & (runs.spread <= DWELL_MAX_SPREAD_M)]
    runs = runs.sort_values(["visit_id", "trail_dwell_s", "t0"], ascending=[True, False, True])
    best = runs.drop_duplicates("visit_id")
    return pd.DataFrame({"visit_id": best.visit_id.values, "dwell_x": (best.wx / best.w).values,
                         "dwell_y": (best.wy / best.w).values, "trail_dwell_s": best.trail_dwell_s.values})


def check_pins(pins: pd.DataFrame, name: str) -> None:
    """One row per address, every address present."""
    dup = pins.address_id[pins.address_id.duplicated()].unique()
    assert len(dup) == 0, f"{name}: {len(dup)} duplicate address_ids, e.g. {list(dup[:5])}"
    assert len(pins) == N_ADDRESSES, f"{name}: {len(pins)} rows, expected {N_ADDRESSES}"


def load_visits(addr_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if addr_df is None:
        addr_df = load_addresses()
    v = read("field_visits.csv")
    v = v.merge(addr_df[["address_id", "town_id", "split"]], on="address_id", how="left")
    v = v.merge(summarise_trails(), on="visit_id", how="left")
    v["checkin_vs_trail_end_m"] = np.hypot(v.checkin_x - v.trail_end_x, v.checkin_y - v.trail_end_y)
    # visit position: the outcome's preferred source, falling back dwell -> trail tail (>= 2 points) -> check-in
    has_tail = v.tail_n.fillna(0) >= TAIL_MIN_POINTS
    cand = {"dwell": (v.dwell_x.notna(), v.dwell_x, v.dwell_y),
            "trail_tail": (has_tail, v.tail_x, v.tail_y),
            "checkin": (pd.Series(True, index=v.index), v.checkin_x, v.checkin_y)}
    pref = v.outcome.map(POS_PREFERENCE).fillna("trail_tail")
    v["pos_source"] = None
    for first in cand:   # each outcome tries its preferred source, then the default order
        order = [first] + [c for c in cand if c != first]
        rows = pref == first
        for c in order:
            ok, _, _ = cand[c]
            v.loc[rows & ok & v.pos_source.isna(), "pos_source"] = c
    v["pos_x"] = np.select([v.pos_source == c for c in cand], [cand[c][1] for c in cand])
    v["pos_y"] = np.select([v.pos_source == c for c in cand], [cand[c][2] for c in cand])
    return v


def sanity_report(addr: pd.DataFrame, visits: pd.DataFrame) -> dict:
    out = {
        "addresses": len(addr),
        "visits": len(visits),
        "visits_with_unknown_address": int(visits.town_id.isna().sum()),
        "visits_without_trail": int(visits.trail_points.isna().sum()),
        "addresses_without_baseline": int(addr.base_x.isna().sum()),
        "of_which_OUT": int((addr.base_x.isna() & (addr.town_id == "OUT")).sum()),
        "addresses_with_visits": int(addr.address_id.isin(visits.address_id).sum()),
        "split_accounts": addr.drop_duplicates("account_id").split.value_counts().to_dict(),
    }
    return out


if __name__ == "__main__":
    a = load_addresses()
    v = load_visits(a)
    for k, val in sanity_report(a, v).items():
        print(f"{k:30s} {val}")

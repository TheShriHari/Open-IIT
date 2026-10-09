"""Steps 6-7: turn every field visit into weighted evidence, and flag fake visits.

Each visit gets a role (its position pos_x/pos_y is the trail's stationary dwell, stationary-tail
centroid or check-in, chosen per outcome; see load.py):
  positive - the agent stood at (or very near) the door: the position is evidence of the pin
  negative - address_not_traceable: the agent searched here and failed; never a pin
and an uncertainty sigma (metres):
  sigma = sqrt(base[outcome]^2 + gps_accuracy^2 + (K_TRAIL * checkin_vs_trail_end)^2)
          x remark multiplier (met_borrower at shop / dukaan / angadi / work)
          x short-dwell multiplier x agent reliability factor

Integrity (no agent IDs anywhere, everything is a rule on the data):
  flag_photo   - the photo_hash is reused on >= 3 different accounts
  flag_hotspot - a 50 m grid cell where one agent checked in far more often than that
                 agent's normal (>= max(10, 5 x the agent's p95 cell count)), plus the
                 8 neighbouring cells of that same agent (hotspots straddle cell edges)
Agent reliability factor - median distance of the agent's reliable check-ins to the
  consensus of OTHER agents at the same address, divided by the overall median,
  clipped to [1, 4]. Train split only; the agent's own flagged visits are excluded.

All tuning uses TRAIN-split addresses only. Stand-in truth for a visit is the
leave-one-out median of the other unflagged reliable visits at the same address.
The answer key (surveyed_addresses.csv) is never used here.
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load import load_addresses, load_visits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RELIABLE = ["met_family", "locked_premises", "neighbour_says_shifted"]
POSITIVE = RELIABLE + ["met_borrower", "cash_collected", "no_such_person"]
NEGATIVE = ["address_not_traceable"]
AWAY_FROM_HOME = re.compile(r"dukaan|dukan|angadi|shop|work|office|kelsa|kaam", re.I)

PHOTO_MIN_ACCOUNTS = 3
CELL_M = 50
HOTSPOT_MIN, HOTSPOT_MULT = 10, 5
AGENT_MIN_N, AGENT_CLIP = 20, (1.0, 4.0)

# fitted by tune() on train split; frozen here so the pipeline is deterministic
PARAMS = {
    # Step 6 kept these check-in-era bases. Retuning on trail-tail positions drove four bases to the
    # 5 m grid floor (5/5/5/5/60/10); single visits then won cluster ties, and the train holdout
    # got worse (p75 64 -> 74 m, >100 m 35 -> 40 of 203). With these bases: p75 53 m, 33 > 100 m.
    "base": {"met_family": 30, "locked_premises": 30, "neighbour_says_shifted": 15,
             "met_borrower": 50, "cash_collected": 65, "no_such_person": 45},
    "k_trail": 0.5,
    "away_mult": 1.0,      # shop/work remark did not make met_borrower worse on train
    "dwell_min_s": 0,      # short dwell did not make check-ins worse on train
    "dwell_mult": 1.5,
}
PARAMS_START = {"base": {"met_family": 18, "locked_premises": 18, "neighbour_says_shifted": 18,
                         "met_borrower": 40, "cash_collected": 60, "no_such_person": 40},
                "k_trail": 0.5, "away_mult": 1.5, "dwell_min_s": 60, "dwell_mult": 1.5}


# ------------------------------------------------------------------ integrity
def integrity_flags(v: pd.DataFrame) -> pd.DataFrame:
    v = v.copy()
    n_acc = v.groupby("photo_hash").account_id.nunique()
    v["flag_photo"] = v.photo_hash.isin(n_acc[n_acc >= PHOTO_MIN_ACCOUNTS].index)

    v["cx"] = np.floor(v.checkin_x / CELL_M).astype(int)
    v["cy"] = np.floor(v.checkin_y / CELL_M).astype(int)
    cells = v.groupby(["agent_id", "town_id", "cx", "cy"]).size().rename("n").reset_index()
    p95 = cells.groupby("agent_id").n.quantile(0.95).rename("p95")
    cells = cells.join(p95, on="agent_id")
    hot = cells[cells.n >= np.maximum(HOTSPOT_MIN, HOTSPOT_MULT * cells.p95)]
    # hotspot cell plus its 8 neighbours, same agent and town only
    ring = pd.DataFrame(
        [(r.agent_id, r.town_id, r.cx + dx, r.cy + dy)
         for r in hot.itertuples() for dx, dy in itertools.product((-1, 0, 1), repeat=2)],
        columns=["agent_id", "town_id", "cx", "cy"]).drop_duplicates()
    ring["flag_hotspot"] = True
    v = v.merge(ring, on=["agent_id", "town_id", "cx", "cy"], how="left")
    v["flag_hotspot"] = v.flag_hotspot.fillna(False).astype(bool)
    v["flag_fake"] = v.flag_photo | v.flag_hotspot
    return v.drop(columns=["cx", "cy"])


# ------------------------------------------------------------ stand-in truth
def _loo_consensus(v: pd.DataFrame, other_agents_only: bool, min_others: int = 2) -> pd.DataFrame:
    """For each train visit: median of the OTHER unflagged reliable visits at its address."""
    ref = v[(v.split == "train") & v.outcome.isin(RELIABLE) & ~v.flag_fake]
    ref_by_addr = {a: g for a, g in ref.groupby("address_id")}
    rows = []
    for r in v[v.split == "train"].itertuples():
        g = ref_by_addr.get(r.address_id)
        if g is None:
            continue
        g = g[g.visit_id != r.visit_id]
        if other_agents_only:
            g = g[g.agent_id != r.agent_id]
        if len(g) < min_others:
            continue
        rows.append((r.visit_id, g.pos_x.median(), g.pos_y.median()))
    t = pd.DataFrame(rows, columns=["visit_id", "tx", "ty"])
    out = v.merge(t, on="visit_id")
    out["err"] = np.hypot(out.pos_x - out.tx, out.pos_y - out.ty)
    return out


def agent_factors(v: pd.DataFrame) -> pd.Series:
    d = _loo_consensus(v, other_agents_only=True)
    d = d[d.outcome.isin(RELIABLE) & ~d.flag_fake]   # fakes are already dropped; measure the rest
    overall = d.err.median()
    per = d.groupby("agent_id").err.agg(["median", "size"])
    f = (per["median"] / overall).clip(*AGENT_CLIP)
    f[per["size"] < AGENT_MIN_N] = 1.0
    return f.reindex(v.agent_id.unique()).fillna(1.0).rename("agent_factor")


# ------------------------------------------------------------------- sigma
def visit_sigma(v: pd.DataFrame, p: dict = PARAMS) -> pd.Series:
    base = v.outcome.map(p["base"]).astype(float)
    s = np.sqrt(base ** 2 + v.gps_accuracy_m ** 2 + (p["k_trail"] * v.checkin_vs_trail_end_m) ** 2)
    away = (v.outcome == "met_borrower") & v.remark.fillna("").str.contains(AWAY_FROM_HOME)
    s = s * np.where(away, p["away_mult"], 1.0)
    s = s * np.where(v.dwell_s < p["dwell_min_s"], p["dwell_mult"], 1.0)
    return s * v.agent_factor


def _loglik(err, sigma, eps=0.1, far_m=2000.0):
    """Rayleigh (good visit) + uniform-in-disc (stray visit) mixture."""
    ray = err / sigma ** 2 * np.exp(-err ** 2 / (2 * sigma ** 2))
    stray = 2 * err / far_m ** 2 * (err <= far_m) + 1e-12
    return np.log((1 - eps) * ray + eps * stray).sum()


def tune(v: pd.DataFrame) -> dict:
    d = _loo_consensus(v, other_agents_only=False)
    d = d[d.outcome.isin(POSITIVE) & ~d.flag_fake & (d.err > 0)]
    best = (-np.inf, None)
    base_grid = np.arange(5, 151, 5)
    for k, am, dmin, dm in itertools.product([0, 0.5, 1.0], [1.0, 1.5, 2.0], [0, 60, 120], [1.5, 2.0]):
        p = {"k_trail": k, "away_mult": am, "dwell_min_s": dmin, "dwell_mult": dm, "base": {}}
        total = 0.0
        for o, g in d.groupby("outcome"):   # each outcome's base only affects its own visits
            lls = [_loglik(g.err.values, visit_sigma(g, {**p, "base": {o: b}}).values) for b in base_grid]
            i = int(np.argmax(lls))
            p["base"][o] = int(base_grid[i])
            total += lls[i]
        if total > best[0]:
            best = (total, p)
    print(f"tuned on {len(d)} train visits, loglik {best[0]:.1f}")
    return best[1]


# -------------------------------------------------------------------- build
def build_evidence(v: pd.DataFrame | None = None, p: dict = PARAMS) -> pd.DataFrame:
    if v is None:
        v = load_visits(load_addresses())
    v = integrity_flags(v)
    v = v.join(agent_factors(v), on="agent_id")
    v["role"] = np.select([v.outcome.isin(POSITIVE), v.outcome.isin(NEGATIVE)],
                          ["positive", "negative"], "ignore")
    v["sigma"] = np.where(v.role == "positive", visit_sigma(v.assign(
        outcome=v.outcome.where(v.role == "positive", "met_family")), p), np.nan)
    return v


def flag_table(v: pd.DataFrame) -> pd.DataFrame:
    t = v.groupby("agent_id").agg(
        visits=("visit_id", "size"), photo=("flag_photo", "sum"), hotspot=("flag_hotspot", "sum"),
        hotspot_only=("flag_hotspot", lambda s: int((s & ~v.loc[s.index, "flag_photo"]).sum())),
        flagged=("flag_fake", "sum"), agent_factor=("agent_factor", "first"))
    t["pct_flagged"] = (100 * t.flagged / t.visits).round(1)
    t["agent_factor"] = t.agent_factor.round(2)
    return t


if __name__ == "__main__":
    v = integrity_flags(load_visits(load_addresses()))
    v = v.join(agent_factors(v), on="agent_id")
    if "--tune" in sys.argv:
        new = tune(v)
        flat = lambda p: {**{f"base.{k}": b for k, b in p["base"].items()}, **{k: x for k, x in p.items() if k != "base"}}
        print(pd.DataFrame({"start": flat(PARAMS_START), "frozen": flat(PARAMS), "tuned": flat(new)}).to_string())
    ev = build_evidence()
    print(flag_table(ev).to_string())
    print(f"\nflagged total {int(ev.flag_fake.sum())}  (photo {int(ev.flag_photo.sum())}, "
          f"hotspot not caught by photo {int((ev.flag_hotspot & ~ev.flag_photo).sum())})")
    print(ev[ev.role == "positive"].groupby("outcome").sigma.median().round(1).to_string())
    cols = ["visit_id", "address_id", "agent_id", "town_id", "split", "outcome", "checkin_x", "checkin_y",
            "pos_x", "pos_y", "pos_source", "tail_n", "role", "sigma", "agent_factor", "flag_photo", "flag_hotspot", "flag_fake"]
    ev[cols].to_csv(ROOT / "outputs" / "visit_evidence.csv", index=False)

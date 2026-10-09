"""Step 9: fuse the text prior with visit evidence into one pin per address.

Per address:
  1. prior = text pin + prior_sigma (outputs/pins_text_prior.csv); OUT -> no pin.
     Pass 2 only: the neighbour street-key clue (neighbours.py) is merged into the prior
     by inverse variance, so it stays one point next to the address's own visits
  2. positive, non-fake visits, weight 1/sigma^2 (sigma from evidence.py)
  3. agent cap: no single agent may hold more than 50% of the visit weight
  4. visits are grouped by single linkage at 100 m; if there are several groups, keep the
     one with the largest total weight, the rest are outliers (tier = visits_disagree).
     If the top two are within 20% in weight, the one nearer the remark-correction POI
     (remarks.py) wins, else the one nearer the prior pin; sigma >= half the gap between them
  5. robust centre: weighted median, drop visits > 3 sigma from it, twice
  6. negative evidence: every not-traceable check-in within prior_sigma of the prior pin
     (someone searched there and failed) widens prior_sigma by NEG_FACTOR (max x4)
  7. pin = weighted median of the kept visits plus the prior as one more point
     sigma = max(1/sqrt(sum w), spread of the kept visits). If a neighbour clue was merged
     into the prior, its weight here is capped at NB_PRIOR_CAP x the kept visits' weight.
     A remark-correction POI is one more point, weight 1/rc_sigma^2, also capped at
     NB_PRIOR_CAP x the kept visits' weight

NEG_FACTOR is tuned on TRAIN addresses. A fusion holdout on train addresses (half the
reliable visits hidden as stand-in truth) is printed for prior-only vs fused.
The answer key (surveyed_addresses.csv) is never used here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evidence import RELIABLE, build_evidence  # noqa: E402
from load import check_pins  # noqa: E402
from neighbours import combine, neighbour_clues, street_table  # noqa: E402
from remarks import all_corrections, remark_clues, remark_hints  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AGENT_CAP = 0.5
CLUSTER_M = 100
TRIM_SIGMAS = 3
TIE_RATIO = 0.8         # second cluster within 20% of the top cluster's weight = a tie
NEG_FACTOR = 1.47       # tuned by tune_negative() on train; see README
NEG_MAX = 4.0
NB_PRIOR_CAP = 0.5      # with a neighbour clue, prior weight <= 0.5 x own kept visits


def weighted_median(vals, w):
    o = np.argsort(vals)
    c = np.cumsum(w[o])
    return float(vals[o][np.searchsorted(c, 0.5 * c[-1])])


def _centre(x, y, w):
    return weighted_median(x, w), weighted_median(y, w)


def cap_agents(w, agents):
    """Scale down an agent holding > AGENT_CAP of the weight to exactly AGENT_CAP."""
    w = w.copy()
    if len(set(agents)) < 2:
        return w
    for a in set(agents):
        m = agents == a
        rest = w[~m].sum()
        if w[m].sum() > AGENT_CAP * (w[m].sum() + rest):
            w[m] *= rest * AGENT_CAP / (1 - AGENT_CAP) / w[m].sum()
    return w


def _clusters(x, y):
    """Single-linkage groups at CLUSTER_M."""
    n, lab = len(x), -np.ones(len(x), int)
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], k
        while stack:
            j = stack.pop()
            for nb in np.where((d[j] <= CLUSTER_M) & (lab < 0))[0]:
                lab[nb] = k
                stack.append(nb)
        k += 1
    return lab


def fuse_one(px, py, ps, pos: pd.DataFrame, neg: pd.DataFrame, neg_factor=NEG_FACTOR, prior_cap=None,
             rc=None):
    """rc = (x, y, sigma) of a remark-correction POI, or None."""
    # 6. negative evidence widens the prior
    n_neg = int((np.hypot(neg.pos_x - px, neg.pos_y - py) <= ps).sum()) if len(neg) else 0
    ps = ps * min(neg_factor ** n_neg, NEG_MAX)
    if pos.empty:
        return px, py, ps, 0, "prior_only", 0.0, n_neg

    x, y = pos.pos_x.values, pos.pos_y.values
    s, agents = pos.sigma.values, pos.agent_id.values
    w = cap_agents(1 / s ** 2, agents)

    # 4. two (or more) clusters -> keep the heaviest; near-tie -> nearest the prior pin
    lab = _clusters(x, y)
    disagree = lab.max() > 0
    tie_gap = 0.0
    if disagree:
        ks = range(lab.max() + 1)
        cents = np.array([_centre(x[lab == k], y[lab == k], w[lab == k]) for k in ks])
        cw = np.array([w[lab == k].sum() for k in ks])
        top2 = np.argsort(-cw)[:2]
        best = int(top2[0])
        if cw[top2[1]] >= TIE_RATIO * cw[top2[0]]:
            ref_x, ref_y = (rc[0], rc[1]) if rc is not None else (px, py)
            d_ref = np.hypot(cents[top2, 0] - ref_x, cents[top2, 1] - ref_y)
            best = int(top2[np.argmin(d_ref)])
            tie_gap = float(np.hypot(*(cents[top2[0]] - cents[top2[1]])))
        keep = lab == best
    else:
        keep = np.ones(len(x), bool)

    # 5. robust centre + 3-sigma trim
    for _ in range(2):
        cx, cy = _centre(x[keep], y[keep], w[keep])
        d = np.hypot(x - cx, y - cy)
        new = keep & (d <= TRIM_SIGMAS * s)
        if not new.any():
            new = keep & (d == d[keep].min())
        keep = new

    xk, yk, sk = x[keep], y[keep], s[keep]
    wk = cap_agents(1 / sk ** 2, agents[keep])
    cx, cy = _centre(xk, yk, wk)
    spread = float(np.hypot(xk - cx, yk - cy).max())

    # 7. final fusion with the prior as one more point
    wp = 1 / ps ** 2
    if prior_cap is not None:    # own visits must dominate a neighbour-merged prior
        wp = min(wp, prior_cap * wk.sum())
    xs, ys, ws = np.r_[xk, px], np.r_[yk, py], np.r_[wk, wp]
    if rc is not None:           # remark POI: one more point, never more than a capped share
        xs, ys = np.r_[xs, rc[0]], np.r_[ys, rc[1]]
        ws = np.r_[ws, min(1 / rc[2] ** 2, NB_PRIOR_CAP * wk.sum())]
    fx, fy = weighted_median(xs, ws), weighted_median(ys, ws)
    sigma = max(1 / np.sqrt(ws.sum()), float(np.median(np.hypot(xk - fx, yk - fy))), tie_gap / 2)
    n = int(keep.sum())
    tier = ("visits_disagree" if disagree or (n >= 2 and spread > CLUSTER_M)
            else "visit_1" if n == 1 else "visits_agree")
    return fx, fy, sigma, n, tier, float(wk.sum() / (wk.sum() + wp)), n_neg


def fuse_all(prior: pd.DataFrame, ev: pd.DataFrame, neg_factor=NEG_FACTOR,
             nb: pd.DataFrame | None = None, rc: pd.DataFrame | None = None) -> pd.DataFrame:
    """nb = neighbour clues (address_id, nb_x, nb_y, nb_sigma, nb_key, nb_n), None in pass 1.
    rc = remark-correction clues (address_id, rc_x, rc_y, rc_sigma, ...) from remarks.py."""
    rc_by = {} if rc is None else {a: (x, y, s) for a, x, y, s in
                                   zip(rc.address_id, rc.rc_x, rc.rc_y, rc.rc_sigma)}
    ok = ~ev.flag_fake
    pos_by = {a: g for a, g in ev[ok & (ev.role == "positive")].groupby("address_id")}
    neg_by = {a: g for a, g in ev[ok & (ev.role == "negative")].groupby("address_id")}
    empty = ev.iloc[:0]
    nb_by = {} if nb is None else nb.set_index("address_id").to_dict("index")
    rows = []
    for r in prior.itertuples(index=False):
        if pd.isna(r.pin_x):
            rows.append(dict(address_id=r.address_id, pin_x=np.nan, pin_y=np.nan, sigma=np.nan,
                             n_good_visits=0, tier="out_of_territory", source=r.source, n_neg_near_prior=0,
                             nb_key=None, nb_n=0))
            continue
        px, py, ps, src = r.pin_x, r.pin_y, r.prior_sigma, r.source
        c = nb_by.get(r.address_id)
        if c is not None:
            px, py, ps = combine(px, py, ps, c["nb_x"], c["nb_y"], c["nb_sigma"])
            if c["nb_sigma"] < r.prior_sigma:      # the clue holds > 50% of the prior weight
                src = "neighbours"
        fx, fy, sg, n, tier, vshare, n_neg = fuse_one(
            px, py, ps, pos_by.get(r.address_id, empty), neg_by.get(r.address_id, empty), neg_factor,
            NB_PRIOR_CAP if c is not None else None, rc_by.get(r.address_id))
        rows.append(dict(address_id=r.address_id, pin_x=round(fx, 1), pin_y=round(fy, 1),
                         sigma=round(sg, 1), n_good_visits=n, tier=tier,
                         source="visits" if vshare > 0.5 else src, n_neg_near_prior=n_neg,
                         nb_key=c["nb_key"] if c else None, nb_n=c["nb_n"] if c else 0))
    return pd.DataFrame(rows)


def build_pins(prior: pd.DataFrame, ev: pd.DataFrame, streets: pd.DataFrame, corr: pd.DataFrame):
    """Full fusion: pass 1 (prior + visits + remark clues), neighbour clues from pass-1 visit
    pins, pass 2 with the clues. Returns (pass1, pins, nb, rc)."""
    rc = remark_clues(ev, corr=corr)
    pass1 = fuse_all(prior, ev, rc=rc)
    nb = neighbour_clues(streets, pass1[pass1.source == "visits"])
    pins = fuse_all(prior, ev, nb=nb, rc=rc)
    return pass1, pins, nb, rc


# ------------------------------------------------------------------ tuning
def _train_truth(ev: pd.DataFrame, min_n=2) -> pd.DataFrame:
    rel = ev[(ev.split == "train") & ev.outcome.isin(RELIABLE) & ~ev.flag_fake]
    t = rel.groupby("address_id").agg(tx=("pos_x", "median"), ty=("pos_y", "median"),
                                      k=("visit_id", "size"))
    return t[t.k >= min_n]


def tune_negative(prior: pd.DataFrame, ev: pd.DataFrame) -> float:
    """How much worse is the prior when not-traceable visits sit on top of it? (train only)"""
    t = _train_truth(ev).join(prior.set_index("address_id")[["pin_x", "pin_y", "prior_sigma"]])
    neg = ev[(ev.role == "negative") & ~ev.flag_fake].merge(
        t.reset_index()[["address_id", "pin_x", "pin_y", "prior_sigma"]], on="address_id")
    neg["near"] = np.hypot(neg.pos_x - neg.pin_x, neg.pos_y - neg.pin_y) <= neg.prior_sigma
    t["n_neg"] = neg.groupby("address_id").near.sum().reindex(t.index).fillna(0)
    t["err"] = np.hypot(t.pin_x - t.tx, t.pin_y - t.ty)
    print("prior error on train by # not-traceable check-ins near the prior pin:")
    print(t.groupby(t.n_neg.clip(upper=3)).err.agg(["size", "median"]).round(0).to_string())
    base, hit = t[t.n_neg == 0].err.median(), t[t.n_neg >= 1]
    if hit.empty or base <= 0:
        return 1.0
    f = (hit.err.median() / base) ** (1 / hit.n_neg.mean())
    return float(np.clip(f, 1.0, 2.0))


def holdout(prior: pd.DataFrame, ev: pd.DataFrame, neg_factor=NEG_FACTOR,
            corr: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train addresses with >=4 reliable visits: every other reliable visit (by visit_id)
    is hidden as stand-in truth; fuse prior + the remaining visits; measure error.
    corr = resolved remark corrections; only those from non-hidden visits are used."""
    rel = ev[(ev.split == "train") & ev.outcome.isin(RELIABLE) & ~ev.flag_fake].sort_values("visit_id")
    rel = rel[rel.groupby("address_id").visit_id.transform("size") >= 4]
    hidden = rel[rel.groupby("address_id").cumcount() % 2 == 0]
    truth = hidden.groupby("address_id")[["pos_x", "pos_y"]].median()
    ev_used = ev[ev.address_id.isin(truth.index) & ~ev.visit_id.isin(hidden.visit_id)]
    p = prior[prior.address_id.isin(truth.index)]
    fused = fuse_all(p, ev_used, neg_factor).set_index("address_id").join(truth)
    p = p.set_index("address_id").join(truth)
    out = pd.DataFrame({
        "prior_only": np.hypot(p.pin_x - p.pos_x, p.pin_y - p.pos_y),
        "fused": np.hypot(fused.pin_x - fused.pos_x, fused.pin_y - fused.pos_y)})
    if corr is not None:
        rc = remark_clues(ev_used, corr=corr[corr.visit_id.isin(ev_used.visit_id)])
        f2 = fuse_all(prior[prior.address_id.isin(truth.index)], ev_used, neg_factor, rc=rc)
        f2 = f2.set_index("address_id").join(truth)
        out["fused_remarks"] = np.hypot(f2.pin_x - f2.pos_x, f2.pin_y - f2.pos_y)
        out["has_remark"] = out.index.isin(rc.address_id)
    return out


if __name__ == "__main__":
    prior = pd.read_csv(ROOT / "outputs" / "pins_text_prior.csv")
    ev = build_evidence()
    nf = tune_negative(prior, ev)
    print(f"tuned NEG_FACTOR = {nf:.2f}  (frozen: {NEG_FACTOR})\n")
    corr = all_corrections(ev)
    h = holdout(prior, ev, corr=corr)
    print(f"train holdout ({len(h)} addresses, stand-in truth = hidden reliable visits):")
    cols = ["prior_only", "fused", "fused_remarks"]
    print(h[cols].describe(percentiles=[0.5, 0.9]).loc[["50%", "90%"]].round(0).to_string())
    print(f"  with a remark clue ({int(h.has_remark.sum())}):")
    print(h[h.has_remark][cols].describe(percentiles=[0.5, 0.9]).loc[["mean", "50%", "90%"]]
          .round(0).to_string(), "\n")
    pass1, pins, nb, rc = build_pins(prior, ev, street_table(), corr)
    no_rc = fuse_all(prior, ev, nb=nb)
    d_rc = np.hypot(pins.pin_x - no_rc.pin_x, pins.pin_y - no_rc.pin_y)
    print(f"remark clues: {len(rc)} addresses; pin shift from the remark clue by tier (m):")
    print(d_rc[pins.address_id.isin(rc.address_id)].groupby(pins.tier).describe(
        percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]].round(0).to_string())
    print(f"pins moved > 50 m (tie broken the other way): {int((d_rc > 50).sum())}\n")
    pins = pins.merge(remark_hints(corr), on="address_id", how="left", validate="one_to_one")
    check_pins(prior, "pins_text_prior")
    check_pins(pins, "pins_fused")
    pins.drop(columns="n_neg_near_prior").to_csv(ROOT / "outputs" / "pins_fused.csv", index=False)
    print(pins.tier.value_counts().to_string(), "\n")
    print(f"neighbour clues: {len(nb)} addresses, by key:\n" + nb.nb_key.value_counts().to_string(), "\n")
    print("source, pass 1 (rows) -> pass 2 (columns):")
    print(pd.crosstab(pass1.source, pins.source, margins=True).to_string(), "\n")
    moved = np.hypot(pins.pin_x - pass1.pin_x, pins.pin_y - pass1.pin_y)
    print("pin shift pass 1 -> pass 2 (m) by pass-1 source:")
    print(moved.groupby(pass1.source).describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]]
          .round(0).to_string())

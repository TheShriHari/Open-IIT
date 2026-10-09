"""Step 10: neighbour evidence from street keys.

Houses on the same street sit close together. For every address we look for OTHER
addresses in the same town + locality that share its most specific street key and have a
visit-based pin from fusion pass 1 (source == visits). The clue is the median of those pins.

Street keys, most specific first (the first key with >= 1 neighbour pin is used):
  building    same building name
  cross_main  same cross_no AND main_no   (T1 style)
  block_road  same block AND road_no      (T3 style)
  block       same block
  gali        same gali_no                (T2 style)

sigma_k(n) = sqrt(floor_k^2 + spread_k^2 / n), with n = number of neighbour pins.
floor = how far apart houses on one street are, spread = noise that averages out with more
neighbours. Fitted on TRAIN addresses that have their own visit pin: leave-one-out clue
(self excluded) vs their own pin, sigma ~= median error / 1.18 per n-bin, least squares
on 1/n. `python src/neighbours.py` refits and prints; the values are frozen in NB_SIGMA.
The answer key (surveyed_addresses.csv) is never used here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load import load_addresses  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KEYS = [
    ("building", ["building"]),
    ("cross_main", ["cross_no", "main_no"]),
    ("block_road", ["block", "road_no"]),
    ("block", ["block"]),
    ("gali", ["gali_no"]),
]
RAYLEIGH_MEDIAN = 1.18
MIN_FIT_ROWS = 15
MAX_SIGMA = 300         # a key no better than the locality centroid is not used
N_BINS = [(1, 1), (2, 2), (3, 4), (5, 10 ** 6)]
NB_SIGMA = {            # (floor, spread) in metres, fitted by fit_sigma() on train.
    "cross_main": (54, 0),  # building: LOO median 376 m (> prior), dropped by MAX_SIGMA
    "block_road": (77, 0),
    "block": (112, 104),
    "gali": (161, 196),
}


def street_table() -> pd.DataFrame:
    """address_id, town_id, split, locality_id + the street-key fields, as strings."""
    with open(ROOT / "outputs" / "parsed_addresses.json", encoding="utf-8") as f:
        p = pd.DataFrame(json.load(f))
    cols = ["locality_id", "building", "cross_no", "main_no", "block", "road_no", "gali_no"]
    p = p.reindex(columns=["address_id"] + cols)
    a = load_addresses()[["address_id", "town_id", "split"]]
    return a.merge(p, on="address_id", how="left", validate="one_to_one")


def sigma_of(key: str, n, table=NB_SIGMA):
    floor, spread = table[key]
    return np.sqrt(floor ** 2 + spread ** 2 / np.asarray(n, float))


def neighbour_clues(streets: pd.DataFrame, anchors: pd.DataFrame, table=NB_SIGMA) -> pd.DataFrame:
    """One clue per address that has a neighbour anchor: address_id, nb_x, nb_y, nb_sigma,
    nb_key, nb_n. anchors = address_id, pin_x, pin_y of visit-based pass-1 pins.
    An address never uses its own anchor. Keys missing from `table` are skipped."""
    st = streets.merge(anchors[["address_id", "pin_x", "pin_y"]], on="address_id", how="left",
                       validate="one_to_one")
    st = st[st.town_id != "OUT"]
    done, rows = set(), []
    for key, cols in KEYS:
        if key not in table:
            continue
        full = ["town_id", "locality_id"] + cols
        for _, g in st[st[full].notna().all(axis=1)].groupby(full):
            anc = g[g.pin_x.notna()]
            if anc.empty:
                continue
            for aid in g.address_id:
                if aid in done:
                    continue
                o = anc[anc.address_id != aid]
                if o.empty:
                    continue
                done.add(aid)
                rows.append(dict(address_id=aid, nb_x=float(o.pin_x.median()), nb_y=float(o.pin_y.median()),
                                 nb_key=key, nb_n=len(o)))
    out = pd.DataFrame(rows, columns=["address_id", "nb_x", "nb_y", "nb_key", "nb_n"])
    out["nb_sigma"] = [float(sigma_of(k, n, table)) if k in table else np.nan
                       for k, n in zip(out.nb_key, out.nb_n)]
    return out


# ------------------------------------------------------------------ tuning
def loo_table(streets: pd.DataFrame, pass1: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Train addresses with their own visit pin: leave-one-out clue vs own pin."""
    anchors = pass1[pass1.source == "visits"]
    all_keys = {k: (1.0, 0.0) for k, _ in KEYS}
    c = neighbour_clues(streets, anchors, all_keys).drop(columns="nb_sigma")
    train = set(streets.address_id[streets.split == "train"])
    t = c[c.address_id.isin(train)].merge(anchors[["address_id", "pin_x", "pin_y"]], on="address_id")
    t["err"] = np.hypot(t.nb_x - t.pin_x, t.nb_y - t.pin_y)
    pr = prior.set_index("address_id").loc[t.address_id]
    t["prior_err"] = np.hypot(pr.pin_x.values - t.pin_x, pr.pin_y.values - t.pin_y)
    t["prior_x"], t["prior_y"], t["prior_sigma"] = pr.pin_x.values, pr.pin_y.values, pr.prior_sigma.values
    return t


def combine(px, py, ps, nx, ny, ns):
    """Inverse-variance merge of the text prior and the neighbour clue (as in prior.py)."""
    wp, wn = 1 / ps ** 2, 1 / ns ** 2
    return (wp * px + wn * nx) / (wp + wn), (wp * py + wn * ny) / (wp + wn), 1 / np.sqrt(wp + wn)


def fit_sigma(t: pd.DataFrame) -> dict:
    table = {}
    for key, _ in KEYS:
        g = t[t.nb_key == key]
        if len(g) == 0:
            continue
        if len(g) < MIN_FIT_ROWS:
            floor, spread = g.err.median() / RAYLEIGH_MEDIAN, 0.0
        else:
            xs, ys, ws = [], [], []
            for lo, hi in N_BINS:
                b = g[(g.nb_n >= lo) & (g.nb_n <= hi)]
                if len(b) >= 3:
                    xs.append(1 / b.nb_n.mean())
                    ys.append((b.err.median() / RAYLEIGH_MEDIAN) ** 2)
                    ws.append(len(b))
            xs, ys, ws = map(np.array, (xs, ys, ws))
            if len(xs) >= 2:
                A = np.c_[np.ones_like(xs), xs] * np.sqrt(ws)[:, None]
                a, b = np.linalg.lstsq(A, ys * np.sqrt(ws), rcond=None)[0]
            else:
                a, b = (ys[0] if len(ys) else (g.err.median() / RAYLEIGH_MEDIAN) ** 2), 0.0
            if b < 0:     # no shrinkage with n -> one sigma
                a, b = (g.err.median() / RAYLEIGH_MEDIAN) ** 2, 0.0
            if a < 0:     # pure 1/n -> fit through the origin
                a, b = 0.0, float(np.dot(ws * xs, ys) / np.dot(ws * xs, xs))
            floor, spread = np.sqrt(a), np.sqrt(b)
        if sigma_of(key, 1, {key: (floor, spread)}) > MAX_SIGMA and floor > MAX_SIGMA:
            continue
        table[key] = (round(float(floor)), round(float(spread)))
    return table


def report(t: pd.DataFrame, table: dict) -> None:
    print("leave-one-out on train (clue vs own visit pin; prior_err = text prior on same rows):")
    s = t.groupby("nb_key", sort=False).agg(
        rows=("err", "size"), median_n=("nb_n", "median"), median_err=("err", "median"),
        p90_err=("err", lambda e: e.quantile(0.9)), prior_median=("prior_err", "median"))
    print(s.reindex([k for k, _ in KEYS]).dropna(how="all").round(0).to_string())
    t = t.assign(nbin=pd.cut(t.nb_n, [0, 1, 2, 4, 10 ** 6], labels=["1", "2", "3-4", "5+"]))
    print("\nmedian error by key and # neighbours:")
    print(t.pivot_table(index="nb_key", columns="nbin", values="err", aggfunc="median",
                        observed=False).round(0).to_string())
    t = t[t.nb_key.isin(table)]
    sg = np.array([sigma_of(k, n, table) for k, n in zip(t.nb_key, t.nb_n)])
    cx, cy, _ = combine(t.prior_x, t.prior_y, t.prior_sigma, t.nb_x, t.nb_y, sg)
    t = t.assign(comb_err=np.hypot(cx - t.pin_x, cy - t.pin_y))
    print("\nprior vs prior+neighbour (median error, train LOO):")
    print(t.groupby("nb_key")[["prior_err", "comb_err"]].median().round(0).to_string())
    print(f"all: {t.prior_err.median():.0f} -> {t.comb_err.median():.0f} m")
    print(f"\ncoverage of 2.15*sigma (should be ~0.90 for a 2-D normal): {(t.err <= 2.15 * sg).mean():.2f}")


if __name__ == "__main__":
    from evidence import build_evidence
    from fuse import fuse_all

    prior = pd.read_csv(ROOT / "outputs" / "pins_text_prior.csv")
    pass1 = fuse_all(prior, build_evidence())
    t = loo_table(street_table(), pass1, prior)
    fitted = fit_sigma(t)
    report(t, fitted)
    print("\nfitted NB_SIGMA (floor, spread):", fitted)
    print("frozen NB_SIGMA:                 ", NB_SIGMA)

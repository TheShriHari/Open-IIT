"""Step 16: learning over time. Replays the pipeline as if run on earlier dates.

For a cutoff date d, only visits with visit_date < d are used, through the same code path:
  build_evidence (integrity flags + agent factors recomputed on the visits seen so far)
  -> build_pins (pass 1, neighbour clues, pass 2, remark clues)
  -> apply_q with the FROZEN q from calibrate.py (outputs/calibration_q.json); nothing is re-tuned.

1. Weekly: cutoffs first visit date + 7k, the last one day after the last visit (= all visits).
   Median / p90 error and R90 coverage on the 100 surveyed addresses (evaluate.address_errors)
   and the action mix over all addresses -> outputs/replay_weekly.csv + replay_weekly.png
2. Not-traceable visits (non-fake): the pin and R90 the system had the day before the visit.
   wrong_place  - DIRECT_VISIT / VISIT_WITH_HINT pin and the check-in > R90 from it: the agent
                  searched where our pin would not have sent them
   phone_first  - VERIFY_FIRST or CANNOT_GEOCODE (OUT): the visit would have been a phone check
   in_circle    - the agent searched inside our R90: not avoidable
   pct_avoidable = wrong_place + phone_first is an UPPER BOUND: it shows the agent was not sent
   where our pin points, not that our pin would have found the house.
   -> outputs/replay_not_traceable.csv (per town + ALL)
3. Later found: the subset of those visits whose address was found AFTER the failed visit.
   The final pin comes from visits (source == "visits") and at least one non-fake positive
   visit dated after the failed one checked in within the final R90 of the final pin.
   The final pin is taken as the confirmed location. Per visit:
   (a) search_dist_m - how close the agent ever got to it (check-in + every GPS trail point)
   (b) daybefore_dist_m / within_r90 - our pin of the day before vs the confirmed location
   pin_would_lead - a visit action whose R90 held the house while the agent never came within
                    that R90 of it: our pin would have led there and the agent's search did not
   and the baseline geocoder pin for comparison
   -> outputs/replay_later_found.csv (per visit) + replay_later_found_summary.csv
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import address_errors  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from calibrate import ACTIONS, PHONE_FIRST, Q_PATH, apply_q  # noqa: E402
from evidence import build_evidence  # noqa: E402
from fuse import build_pins, fuse_all  # noqa: E402
from load import check_pins, load_addresses, load_visits, read  # noqa: E402
from neighbours import street_table  # noqa: E402
from remarks import all_corrections  # noqa: E402

OUT = ROOT / "outputs"


class Replay:
    def __init__(self):
        addr = load_addresses()
        self.visits = load_visits(addr)
        self.prior = pd.read_csv(OUT / "pins_text_prior.csv")
        self.streets = street_table()
        self.qs = json.load(open(Q_PATH))
        self.ev_all = build_evidence(self.visits)
        # parse remark corrections once for every positive visit; each cutoff keeps the
        # ones from its own non-fake positive visits (= all_corrections on the subset)
        self.corr_all = all_corrections(self.ev_all.assign(flag_fake=False))
        self.cache = {}

    def state_at(self, cutoff: str) -> pd.DataFrame:
        """Final pins (pin, sigma, radius_90, action, source) from visits before cutoff."""
        if cutoff in self.cache:
            return self.cache[cutoff]
        sub = self.visits[self.visits.visit_date < cutoff]
        if sub.empty:
            pins = fuse_all(self.prior, self.ev_all.iloc[:0])
        else:
            ev = build_evidence(sub.reset_index(drop=True))
            ok = ev[(ev.role == "positive") & ~ev.flag_fake].visit_id
            _, pins, _, _ = build_pins(self.prior, ev, self.streets,
                                       self.corr_all[self.corr_all.visit_id.isin(ok)])
        pins = apply_q(pins, self.qs)
        check_pins(pins, f"replay {cutoff}")
        self.cache[cutoff] = pins
        return pins


def score(pins: pd.DataFrame) -> dict:
    e = address_errors(pins).merge(pins[["address_id", "radius_90"]], on="address_id")
    return dict(median_m=round(e.err.median()), p90_m=round(e.err.quantile(0.9)),
                coverage_90=round(float((e.err <= e.radius_90).mean()), 2))


def weekly(rp: Replay) -> pd.DataFrame:
    first = pd.Timestamp(rp.visits.visit_date.min())
    end = pd.Timestamp(rp.visits.visit_date.max()) + pd.Timedelta(days=1)
    cutoffs = list(pd.date_range(first, end, freq="7D"))
    if cutoffs[-1] < end:
        cutoffs.append(end)
    rows = []
    for k, c in enumerate(cutoffs):
        c = c.strftime("%Y-%m-%d")
        pins = rp.state_at(c)
        mix = pins.action.value_counts().reindex(ACTIONS, fill_value=0)
        rows.append(dict(week=k, cutoff=c, n_visits=int((rp.visits.visit_date < c).sum()),
                         **score(pins), n_visit_pins=int((pins.source == "visits").sum()),
                         **mix.to_dict()))
        print(f"  week {k:2d} < {c}: {rows[-1]}", flush=True)
    return pd.DataFrame(rows)


def not_traceable(rp: Replay) -> tuple[pd.DataFrame, int]:
    nt = rp.ev_all[rp.ev_all.outcome == "address_not_traceable"]
    n_fake = int(nt.flag_fake.sum())
    nt = nt[~nt.flag_fake]
    parts = []
    for d, g in nt.groupby("visit_date"):
        p = rp.state_at(d)[["address_id", "pin_x", "pin_y", "radius_90", "action"]]
        parts.append(g[["visit_id", "address_id", "town_id", "pos_x", "pos_y"]].merge(
            p, on="address_id", validate="many_to_one"))
    v = pd.concat(parts, ignore_index=True)
    dist = np.hypot(v.pos_x - v.pin_x, v.pos_y - v.pin_y)
    visit = v.action.isin(["DIRECT_VISIT", "VISIT_WITH_HINT"])
    v["cls"] = np.select([visit & (dist > v.radius_90), v.action.isin(PHONE_FIRST)],
                         ["wrong_place", "phone_first"], "in_circle")

    def table(g):
        n = len(g)
        c = g.cls.value_counts()
        return pd.Series({"n_not_traceable": n,
                          **{k: int(c.get(k, 0)) for k in ["wrong_place", "phone_first", "in_circle"]},
                          "pct_wrong_place": round(100 * c.get("wrong_place", 0) / n, 1),
                          "pct_phone_first": round(100 * c.get("phone_first", 0) / n, 1),
                          "pct_avoidable_upper_bound": round(100 * (c.get("wrong_place", 0) + c.get("phone_first", 0)) / n, 1)})

    t = pd.concat([v.groupby("town_id").apply(table, include_groups=False),
                   table(v).to_frame("ALL").T])
    t.index.name = "town_id"
    return t.reset_index(), n_fake


def later_found(rp: Replay) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(OUT / "pins_final.csv")
    final = final[final.source == "visits"][["address_id", "pin_x", "pin_y", "radius_90"]].rename(
        columns={"pin_x": "fx", "pin_y": "fy", "radius_90": "fr90"})
    ev = rp.ev_all.merge(final, on="address_id")
    good = ev[(ev.role == "positive") & ~ev.flag_fake]
    good = good[np.hypot(good.pos_x - good.fx, good.pos_y - good.fy) <= good.fr90]
    last_hit = good.groupby("address_id").visit_date.max().rename("last_hit")

    nt = ev[(ev.outcome == "address_not_traceable") & ~ev.flag_fake].merge(last_hit, on="address_id")
    nt = nt[nt.last_hit > nt.visit_date]
    parts = []
    for d, g in nt.groupby("visit_date"):
        p = rp.state_at(d)[["address_id", "pin_x", "pin_y", "radius_90", "action"]]
        parts.append(g.merge(p, on="address_id", validate="many_to_one"))
    v = pd.concat(parts, ignore_index=True)

    trail = read("visit_gps_points.csv")[["visit_id", "x", "y"]].merge(v[["visit_id", "fx", "fy"]],
                                                                         on="visit_id")
    trail["d"] = np.hypot(trail.x - trail.fx, trail.y - trail.fy)
    v["checkin_dist_m"] = np.hypot(v.checkin_x - v.fx, v.checkin_y - v.fy)
    v["search_dist_m"] = np.fmin(v.checkin_dist_m, v.visit_id.map(trail.groupby("visit_id").d.min()))
    v["daybefore_dist_m"] = np.hypot(v.pin_x - v.fx, v.pin_y - v.fy)
    v["within_r90"] = v.daybefore_dist_m <= v.radius_90
    # our pin (a visit action) had the house inside its R90, but the agent never got that close
    v["pin_would_lead"] = v.action.isin(["DIRECT_VISIT", "VISIT_WITH_HINT"]) & v.within_r90 &         (v.search_dist_m > v.radius_90)
    base = load_addresses().set_index("address_id")
    v["baseline_dist_m"] = np.hypot(v.address_id.map(base.base_x) - v.fx, v.address_id.map(base.base_y) - v.fy)
    v = v.rename(columns={"action": "daybefore_action", "radius_90": "daybefore_r90"})
    cols = ["visit_id", "address_id", "town_id", "visit_date", "last_hit", "checkin_dist_m", "search_dist_m",
            "baseline_dist_m", "daybefore_action", "daybefore_dist_m", "daybefore_r90", "within_r90", "pin_would_lead"]
    v = v[cols].round(1)

    def table(g):
        pinned = g[~g.daybefore_action.isin(PHONE_FIRST)]
        return pd.Series({
            "n": len(g),
            "search_median_m": round(g.search_dist_m.median()), "search_p90_m": round(g.search_dist_m.quantile(.9)),
            "baseline_median_m": round(g.baseline_dist_m.median()),
            "daybefore_median_m": round(g.daybefore_dist_m.median()),
            "daybefore_p90_m": round(g.daybefore_dist_m.quantile(.9)),
            "pct_daybefore_within_r90": round(100 * g.within_r90.mean(), 1),
            "n_verify_first": int(g.daybefore_action.isin(PHONE_FIRST).sum()),
            "n_pinned": len(pinned),
            "pinned_daybefore_median_m": round(pinned.daybefore_dist_m.median()) if len(pinned) else np.nan,
            "pct_pinned_within_r90": round(100 * pinned.within_r90.mean(), 1) if len(pinned) else np.nan,
            "n_pin_would_lead": int(g.pin_would_lead.sum()),
            "pct_pin_would_lead": round(100 * g.pin_would_lead.mean(), 1),
        })

    t = pd.concat([v.groupby("town_id").apply(table, include_groups=False), table(v).to_frame("ALL").T])
    t.index.name = "town_id"
    return v, t.reset_index()


def plot(w: pd.DataFrame, base: dict, path: Path):
    blue, orange, aqua, grey = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted)
    x = w.week.values
    for col, c, name in [("p90_m", orange, "p90"), ("median_m", blue, "median")]:
        a1.plot(x, w[col], color=c, lw=2, marker="o", ms=4, label=f"{name} error")
        a1.axhline(base[col], color=c, lw=1.2, ls="--", label=f"baseline geocoder {name}")
        a1.annotate(f"{name} {w[col].iloc[-1]:.0f} m", (x[-1], w[col].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=ink, fontsize=9)
        a1.annotate(f"baseline {name} {base[col]:.0f} m", (0, base[col]), xytext=(0, 4),
                    textcoords="offset points", color=muted, fontsize=8)
    a1.set_ylabel("error on 100 surveyed addresses (m)", color=ink)
    a1.set_ylim(0, base["p90_m"] * 1.2)     # headroom so the legend clears the baseline line
    a1.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    a1.set_title("Pin error as visits accumulate (only visits before each week)", color=ink, loc="left")

    ys = [w[a].values for a in ACTIONS]
    a2.stackplot(x, ys, colors=[blue, orange, aqua, grey], labels=[a.replace("_", " ").lower() for a in ACTIONS],
                 edgecolor="white", linewidth=1.5)
    cum = np.cumsum(ys, axis=0)
    for i, a in enumerate(ACTIONS):
        mid = cum[i][-1] - ys[i][-1] / 2
        a2.annotate(f"{ys[i][-1]:,}", (x[-1], mid), xytext=(6, 0), textcoords="offset points",
                    va="center", color=ink, fontsize=9)
    a2.set_ylabel("addresses", color=ink)
    a2.set_xlabel("week (cutoff = first visit date + 7 x week)", color=ink)
    a2.set_title("Action mix (all 3,117 addresses)", color=ink, loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper left", ncol=4, bbox_to_anchor=(0, -0.15))
    a2.set_xlim(0, x[-1] + 1)
    a2.set_xticks(x)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


def main():
    rp = Replay()
    base_pins = load_addresses()[["address_id", "base_x", "base_y"]].rename(
        columns={"base_x": "pin_x", "base_y": "pin_y"})
    e = address_errors(base_pins).err
    base = dict(median_m=round(e.median()), p90_m=round(e.quantile(0.9)))
    print(f"baseline geocoder: {base}")

    print("weekly replay:")
    w = weekly(rp)
    final = pd.read_csv(OUT / "pins_final.csv")
    last = rp.state_at(w.cutoff.iloc[-1]).merge(final, on="address_id", suffixes=("", "_f"))
    d = np.hypot(last.pin_x - last.pin_x_f, last.pin_y - last.pin_y_f)
    assert (d.fillna(0) < 0.5).all() and (last.action == last.action_f).all(), \
        f"last replay week differs from pins_final: {int((d > 0.5).sum())} pins"
    w.to_csv(OUT / "replay_weekly.csv", index=False)
    plot(w, base, OUT / "replay_weekly.png")

    print("\nnot-traceable visits, judged with the pins of the day before:")
    t, n_fake = not_traceable(rp)
    t.to_csv(OUT / "replay_not_traceable.csv", index=False)
    print(f"(excluded {n_fake} not-traceable visits flagged fake)")
    print(t.to_string(index=False))
    print("pct_avoidable_upper_bound: the agent was not sent where our pin points; "
          "it does not show our pin would have found the house")

    print("\nnot-traceable visits at addresses found later (confirmed location = final visit pin):")
    lf, lt = later_found(rp)
    lf.to_csv(OUT / "replay_later_found.csv", index=False)
    lt.to_csv(OUT / "replay_later_found_summary.csv", index=False)
    print(lt.to_string(index=False))
    print("by day-before action:")
    print(lf.groupby("daybefore_action").agg(n=("visit_id", "size"), daybefore_median_m=("daybefore_dist_m", "median"),
                                             pct_within_r90=("within_r90", "mean")).round(2).to_string())
    print("\n" + w.to_string(index=False))
    print("\nwrote replay_weekly.csv, replay_weekly.png, replay_not_traceable.csv, replay_later_found*.csv")


if __name__ == "__main__":
    main()

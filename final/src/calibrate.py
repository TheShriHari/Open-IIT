"""Step 12: calibrated 90% radius (R90) and field action for every pin.

  1. score = error / sigma on the surveyed addresses (errors come from
     evaluate.address_errors; this file never reads the answer key)
  2. split conformal: q = ceil((n+1) * 0.9)-th smallest score on train+validation,
     radius_90 = q * sigma. OUT -> no radius
  3. variant A = one q for all pins; variant B = one q for visit pins, one for text-only pins.
     B is used if A's train+validation coverage is uneven across the two groups
     (a group outside [0.80, 0.97] or the groups more than 0.10 apart). Decided before test.
  4. action: radius_90 <= 100 DIRECT_VISIT, <= 500 VISIT_WITH_HINT, else VERIFY_FIRST;
     no pin (OUT) -> CANNOT_GEOCODE

Reports test coverage (15 rows, noisy) and 5-fold CV coverage over all 100 (q refit per fold).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import address_errors  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from load import check_pins  # noqa: E402

LEVEL = 0.9
CV_FOLDS = 5
CV_SEED = 0
EVEN_BAND = (0.80, 0.97)
EVEN_GAP = 0.10
DIRECT_M = 100
HINT_M = 500
Q_PATH = ROOT / "outputs" / "calibration_q.json"
ACTIONS = ["DIRECT_VISIT", "VISIT_WITH_HINT", "VERIFY_FIRST", "CANNOT_GEOCODE"]
PHONE_FIRST = ["VERIFY_FIRST", "CANNOT_GEOCODE"]   # no field visit: tele-dialer first


def conformal_q(scores, level=LEVEL) -> float:
    s = np.sort(np.asarray(scores, float))
    k = int(np.ceil((len(s) + 1) * level))
    if k > len(s):
        print(f"  warning: n={len(s)} too small for level {level}; using the max score")
        k = len(s)
    return float(s[k - 1])


def group_of(source: pd.Series) -> pd.Series:
    return np.where(source == "visits", "visits", "text")


def fit(cal: pd.DataFrame, variant: str) -> dict:
    """q per group. Variant A: same q for both groups."""
    if variant == "A":
        q = conformal_q(cal.score)
        return {"visits": q, "text": q}
    return {g: conformal_q(cal[cal.group == g].score) for g in ("visits", "text")}


def radius(df: pd.DataFrame, qs: dict) -> pd.Series:
    return df.sigma * df.group.map(qs)


def coverage_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    t = df.groupby(by).agg(n=("covered", "size"), coverage=("covered", "mean"),
                           median_radius_m=("radius_90", "median"))
    t.loc["all"] = [len(df), df.covered.mean(), df.radius_90.median()]
    return t.astype({"n": int}).round({"coverage": 2, "median_radius_m": 0})


def cross_val(sv: pd.DataFrame, variant: str) -> pd.DataFrame:
    """5-fold CV over all surveyed addresses; q refit on the other folds."""
    rng = np.random.default_rng(CV_SEED)
    fold = np.empty(len(sv), int)
    fold[rng.permutation(len(sv))] = np.arange(len(sv)) % CV_FOLDS
    out = []
    for k in range(CV_FOLDS):
        qs = fit(sv[fold != k], variant)
        part = sv[fold == k].copy()
        part["radius_90"] = radius(part, qs)
        part["fold"] = k
        out.append(part)
    cv = pd.concat(out)
    cv["covered"] = cv.err <= cv.radius_90
    return cv


def is_uneven(cov_by_group: pd.Series) -> bool:
    c = cov_by_group.drop("all")
    return bool(((c < EVEN_BAND[0]) | (c > EVEN_BAND[1])).any() or c.max() - c.min() > EVEN_GAP)


def action(r90: pd.Series) -> pd.Series:
    return pd.Series(np.select([r90 <= DIRECT_M, r90 <= HINT_M], ["DIRECT_VISIT", "VISIT_WITH_HINT"],
                               "VERIFY_FIRST"), index=r90.index)


def apply_q(pins: pd.DataFrame, qs: dict) -> pd.DataFrame:
    """group, radius_90 = q * sigma (NaN for OUT) and action for every pin."""
    pins = pins.assign(group=group_of(pins.source))
    pins["radius_90"] = radius(pins, qs).round(0)
    pins["action"] = action(pins.radius_90)
    pins.loc[pins.pin_x.isna(), "action"] = "CANNOT_GEOCODE"
    return pins


def report(sv: pd.DataFrame, variant: str) -> tuple[dict, pd.Series]:
    cal = sv[sv.split != "test"]
    qs = fit(cal, variant)
    sv = sv.assign(radius_90=radius(sv, qs))
    sv["covered"] = sv.err <= sv.radius_90
    cal, test = sv[sv.split != "test"], sv[sv.split == "test"]
    print(f"\n===== variant {variant}: q = " + ", ".join(f"{g} {q:.2f}" for g, q in qs.items()))
    tv_group = coverage_table(cal, "group")
    print("train+validation (in-sample) by group:\n" + tv_group.to_string())
    print("train+validation by source:\n" + coverage_table(cal, "source").to_string())
    print(f"test coverage: {test.covered.mean():.2f} (n={len(test)}, noisy), "
          f"median R90 {test.radius_90.median():.0f} m")
    cv = cross_val(sv.drop(columns=["radius_90", "covered"]), variant)
    print(f"5-fold CV coverage: {cv.covered.mean():.2f}  per fold: "
          + " ".join(f"{c:.2f}" for c in cv.groupby("fold").covered.mean()))
    print("CV by group:\n" + coverage_table(cv, "group").to_string())
    print("CV by source:\n" + coverage_table(cv, "source").to_string())
    return qs, tv_group.coverage


if __name__ == "__main__":
    pins = pd.read_csv(ROOT / "outputs" / "pins_fused.csv")
    check_pins(pins, "pins_fused")
    pins["group"] = group_of(pins.source)
    sv = address_errors(pins).merge(pins[["address_id", "sigma", "source", "group"]], on="address_id",
                                    validate="one_to_one")
    sv["score"] = sv.err / sv.sigma

    qs_a, cov_a = report(sv, "A")
    qs_b, _ = report(sv, "B")
    uneven = is_uneven(cov_a)
    print(f"\nvariant A train+validation coverage by group: {cov_a.drop('all').to_dict()} -> "
          + ("uneven, using B (one q per group)" if uneven else "even enough, using A (single q)"))
    qs = qs_b if uneven else qs_a
    with open(Q_PATH, "w") as f:      # frozen q for replay.py
        json.dump(qs, f, indent=1)

    pins = apply_q(pins, qs)
    print("\naction counts (all addresses):")
    print(pins.action.value_counts().to_string())
    print("\naction by source:")
    print(pd.crosstab(pins.source, pins.action, margins=True).to_string())
    check_pins(pins, "pins_final")
    pins.drop(columns="group").to_csv(ROOT / "outputs" / "pins_final.csv", index=False)

"""Final evaluation: baseline geocoder vs final pins (outputs/geocodes.csv) on the 100 surveyed addresses.

Errors come from evaluate.address_errors; this file never reads the answer key.
Pins never use the answer key, so their errors on every split are honest held-out numbers.
Only R90 is fitted on surveyed errors (q on train+validation), so coverage is reported two ways:
  - per split with the frozen q from calibrate.py (test: 15 rows, scored once)
  - 5-fold CV over all 100 (q refit on the other folds, calibrate.cross_val)
-> outputs/final_results.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import address_errors  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from calibrate import Q_PATH, cross_val, group_of  # noqa: E402
from load import check_pins, load_addresses  # noqa: E402

OUT = ROOT / "outputs"


def row(g: pd.DataFrame) -> dict:
    r = dict(n=len(g), median_m=round(g.err.median()), p90_m=round(g.err.quantile(0.9)),
             within_100m=round((g.err <= 100).mean(), 2), within_300m=round((g.err <= 300).mean(), 2))
    if "radius_90" in g:
        r |= dict(r90_coverage=round((g.err <= g.radius_90).mean(), 2), median_r90_m=round(g.radius_90.median()))
    if "cv_radius_90" in g:
        r["cv_r90_coverage"] = round((g.err <= g.cv_radius_90).mean(), 2)
    return r


def table(df: pd.DataFrame, version: str) -> pd.DataFrame:
    rows = [dict(version=version, by="all", group="all", **row(df))]
    for by in ("split", "source"):
        if by in df:
            rows += [dict(version=version, by=by, group=k, **row(g)) for k, g in df.groupby(by)]
    return pd.DataFrame(rows)


def main():
    base = load_addresses()[["address_id", "base_x", "base_y"]].rename(columns={"base_x": "pin_x", "base_y": "pin_y"})
    geo = pd.read_csv(OUT / "geocodes.csv")
    check_pins(geo, "geocodes")
    fused = pd.read_csv(OUT / "pins_final.csv")[["address_id", "sigma"]]

    b = address_errors(base).merge(geo[["address_id", "source"]], on="address_id")  # source = final pin's
    f = address_errors(geo).merge(geo[["address_id", "radius_90", "source"]], on="address_id")
    f = f.merge(fused, on="address_id", validate="one_to_one")
    f["group"] = group_of(f.source)
    f["score"] = f.err / f.sigma
    qs = json.load(open(Q_PATH))
    variant = "A" if qs["visits"] == qs["text"] else "B"      # the variant calibrate.py chose
    cv = cross_val(f[["address_id", "split", "err", "sigma", "group", "score"]], variant)
    f["cv_radius_90"] = f.address_id.map(cv.set_index("address_id").radius_90)

    res = pd.concat([table(b, "baseline geocoder"), table(f, "final")], ignore_index=True)
    res.to_csv(OUT / "final_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print(f"\ntest split: {int((f.split == 'test').sum())} addresses only - its p90 is set by 1-2 rows")
    print("r90_coverage uses the frozen q (fit on train+validation); cv_r90_coverage refits q per fold")
    print("wrote outputs/final_results.csv")


if __name__ == "__main__":
    main()

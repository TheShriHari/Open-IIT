"""The scorer. The ONLY file allowed to read surveyed_addresses.csv (the answer key).

Usage:
    python evaluate.py outputs/pins.csv  "label for scoreboard"

pins.csv needs: address_id, pin_x, pin_y   (optional: radius_90, source)
Prints median / p90 error, % within 100 m and 300 m, radius coverage, by split and
overall, and appends one row per split to outputs/scoreboard.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def answer_key() -> pd.DataFrame:
    truth = pd.read_csv(DATA / "surveyed_addresses.csv")
    addr = pd.read_csv(DATA / "addresses.csv")[["address_id", "account_id", "town_id"]]
    splits = pd.read_csv(DATA / "splits.csv")
    return truth.merge(addr, on="address_id").merge(splits, on="account_id")


def score_table(pins: pd.DataFrame) -> pd.DataFrame:
    dup = pins.address_id[pins.address_id.duplicated()].unique()
    assert len(dup) == 0, f"pins have {len(dup)} duplicate address_ids, e.g. {list(dup[:5])}"
    df = answer_key().merge(pins, on="address_id", how="left", validate="one_to_one")
    missing = df.pin_x.isna().sum()
    if missing:
        raise ValueError(f"{missing} surveyed addresses have no pin")
    df["err"] = np.hypot(df.pin_x - df.surveyed_x, df.pin_y - df.surveyed_y)
    if "radius_90" in df:
        df["covered"] = df.err <= df.radius_90
    return df


def address_errors(pins: pd.DataFrame) -> pd.DataFrame:
    """Per-address error for the surveyed addresses: address_id, split, err.
    No truth coordinates leave this file."""
    return score_table(pins)[["address_id", "split", "err"]]


def summarise(df: pd.DataFrame, by: str | None = "split") -> pd.DataFrame:
    def f(g):
        row = {
            "n": len(g),
            "median_m": round(g.err.median()),
            "p90_m": round(g.err.quantile(0.9)),
            "within_100m": round((g.err <= 100).mean(), 2),
            "within_300m": round((g.err <= 300).mean(), 2),
        }
        if "covered" in g:
            row["coverage_90"] = round(g.covered.mean(), 2)
            row["median_radius_m"] = round(g.radius_90.median())
        return pd.Series(row)

    parts = [f(df).rename("all")]
    if by:
        parts += [f(g).rename(k) for k, g in df.groupby(by)]
    return pd.DataFrame(parts)


def main(pins_path: str, label: str):
    pins = pd.read_csv(pins_path)
    df = score_table(pins)
    table = summarise(df)
    print(f"\n=== {label} ===")
    print(table.to_string())
    if "source" in df:
        print("\nby pin source:")
        print(summarise(df, by=None if df.source.nunique() == 1 else "source").to_string())
    sb = ROOT / "outputs" / "scoreboard.csv"
    rows = table.reset_index().rename(columns={"index": "split"})
    rows.insert(0, "version", label)
    if sb.exists():  # rewrite so new columns (e.g. coverage_90) don't misalign old rows
        rows = pd.concat([pd.read_csv(sb), rows], ignore_index=True)
    rows.to_csv(sb, index=False)
    return table


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).stem)

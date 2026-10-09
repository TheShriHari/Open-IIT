"""Step 5: the text-only first guess ("prior") for every address.

Three possible clues, each with an uncertainty sigma (metres):
  1. LANDMARK  - the map POI of the named type closest to the address's locality
                 (only trusted if within LANDMARK_MAX_M of that locality: beyond that the
                 named landmark is simply missing from our map and we'd pick the wrong one)
  2. LOCALITY  - the locality centroid (or the average of the pincode's localities
                 if the locality name was not found)
  3. BASELINE  - the commercial geocoder pin, trusted according to its precision label
The pin is the inverse-variance weighted average (weight = 1/sigma^2) of the clues present,
and the combined sigma = 1/sqrt(sum of weights).

The sigma values were measured on TRAIN-split addresses only, using the median check-in of
reliable visits as a stand-in for the true location (sigma ~= median error / 1.18).
The answer key (surveyed_addresses.csv) is never used here.
"""
import numpy as np
import pandas as pd

LANDMARK_MAX_M = 550
SIGMA = {
    "landmark": 150,
    "locality": 300,          # locality centroid when the locality name is known
    "pincode_area": 1200,     # mean of centroids sharing the pincode
    "base_rooftop": 50,
    "base_street": 100,
    "base_locality": 300,
    "base_pincode": 1200,
}
TEMPLE_TYPES = ["ganesha_temple", "hanuman_temple"]


def _area_centroids(r, loc):
    if pd.notna(r.locality_id):
        return loc[loc.locality_id == r.locality_id], "locality"
    if pd.notna(r.pincode):
        c = loc[(loc.town_id == r.town_id) & (loc.pincode == r.pincode)]
        if len(c):
            return c, "pincode_area"
    return loc[loc.town_id == r.town_id], "pincode_area"


def _resolve_landmark(r, area, lm):
    """Pick the POI matching any named landmark that lies closest to the address's area."""
    best = None
    for item in r.landmarks or []:
        types = TEMPLE_TYPES if item["type"] == "temple_any" else [item["type"]]
        c = lm[(lm.town_id == r.town_id) & lm.landmark_type.isin(types)]
        if c.empty:
            continue
        d = np.sqrt((c.x.values[:, None] - area.centroid_x.values[None]) ** 2
                    + (c.y.values[:, None] - area.centroid_y.values[None]) ** 2).min(axis=1)
        i = int(d.argmin())
        if d[i] <= LANDMARK_MAX_M and (best is None or d[i] < best["dist"]):
            best = {"poi_id": c.poi_id.iloc[i], "type": c.landmark_type.iloc[i],
                    "x": c.x.iloc[i], "y": c.y.iloc[i], "dist": d[i],
                    "relation": item.get("relation")}
    return best


def text_prior(addr_parsed: pd.DataFrame, loc: pd.DataFrame, lm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in addr_parsed.itertuples(index=False):
        if r.town_id == "OUT":
            rows.append({"address_id": r.address_id, "pin_x": np.nan, "pin_y": np.nan,
                         "prior_sigma": np.nan, "source": "out_of_territory"})
            continue
        area, area_kind = _area_centroids(r, loc)
        clues = [(area.centroid_x.mean(), area.centroid_y.mean(), SIGMA[area_kind], area_kind)]
        poi = _resolve_landmark(r, area, lm)
        if poi:
            clues.append((poi["x"], poi["y"], SIGMA["landmark"], "landmark"))
        if pd.notna(r.base_x):
            clues.append((r.base_x, r.base_y, SIGMA["base_" + r.base_precision], "base_" + r.base_precision))
        w = np.array([1 / c[2] ** 2 for c in clues])
        x = float(np.dot(w, [c[0] for c in clues]) / w.sum())
        y = float(np.dot(w, [c[1] for c in clues]) / w.sum())
        strongest = clues[int(w.argmax())][3]
        rows.append({
            "address_id": r.address_id, "pin_x": round(x, 1), "pin_y": round(y, 1),
            "prior_sigma": round(1 / np.sqrt(w.sum()), 1),
            "source": "landmark" if poi else ("baseline_" + r.base_precision if strongest.startswith("base_r") or strongest.startswith("base_s") else area_kind),
            "landmark_poi": poi["poi_id"] if poi else None,
            "landmark_type": poi["type"] if poi else None,
            "landmark_relation": poi["relation"] if poi else None,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from load import check_pins, load_addresses, map_tables
    from textparse import parse_all

    root = Path(__file__).resolve().parent.parent
    a = load_addresses()
    _, loc, lm = map_tables()
    parsed = parse_all(a, loc)
    d = a.merge(parsed.drop(columns=["locality_name"]), on="address_id")
    pins = text_prior(d, loc, lm)
    check_pins(pins, "pins_text_prior")
    pins.to_csv(root / "outputs" / "pins_text_prior.csv", index=False)
    print(pins.source.value_counts())

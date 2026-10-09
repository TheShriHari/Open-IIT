"""Steps 14-15: landmark directions + one output file per consumer.

Reads outputs/pins_final.csv (Step 12) and writes:
  geocodes.csv               field app + address records (pin, R90, action, directions)
  geocodes_offline.json      same rows grouped by town, for the offline field app
  planner_actions.csv        action + one-line reason + dispatch channel per address
  ps2_location_confidence.csv  high/medium/low confidence + hard_to_find flag + dispatch fields
  dashboard_metrics.csv      per town and locality: action mix, median R90, not-traceable rate

Directions: nearest map POI (any type) within 200 m of the pin, with distance and compass
direction from the POI to the pin, worded in the town's address style; then the agents'
remark hint, then visual door cues from remarks ("Look for: blue gate"), then either an
uncertainty warning (R90 > 500 m: phone first, no search radius) or the search radius.
Pins are not changed. surveyed_addresses.csv is not read.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import ACTIONS, PHONE_FIRST
from load import check_pins, load_addresses, map_tables, read
from remarks import door_cues

OUT = Path(__file__).resolve().parent.parent / "outputs"
POI_RADIUS = 200          # m: nearest landmark searched within this distance of the pin
NT_RADIUS = 200           # m: not-traceable visits counted within this distance of the prior
UNCERTAIN_R90 = 500       # m: above this, phone verification first

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_WORDS = {
    "karnataka": ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"],
    "hindi": ["uttar", "uttar-purab", "purab", "dakshin-purab", "dakshin", "dakshin-paschim", "paschim", "uttar-paschim"],
    "metro": ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"],
}

# landmark names per town style; metro uses the POI's own map name
LANDMARK_NAMES = {
    "karnataka": {
        "bus_stop": "Bus Stop", "church": "Church", "community_hall": "Samudaya Bhavana",
        "ganesha_temple": "Ganapathi Gudi", "govt_school": "Sarkari Shale",
        "hanuman_temple": "Anjaneya Gudi", "masjid": "Masjid", "medical_store": "Medical Store",
        "milk_dairy": "Haalina Dairy", "park": "Park", "petrol_bunk": "Petrol Bunk",
        "post_office": "Post Office", "ration_shop": "Ration Angadi", "water_tank": "Neerina Tank",
    },
    "hindi": {
        "bus_stop": "Bus Stop", "church": "Girja Ghar", "community_hall": "Samudayik Bhavan",
        "ganesha_temple": "Ganesh Mandir", "govt_school": "Sarkari School",
        "hanuman_temple": "Hanuman Mandir", "masjid": "Masjid", "medical_store": "Dawai ki Dukaan",
        "milk_dairy": "Doodh Dairy", "park": "Park", "petrol_bunk": "Petrol Pump",
        "post_office": "Daakghar", "ration_shop": "Ration ki Dukaan", "water_tank": "Paani ki Tanki",
    },
}

RELATION_WORDS = {
    "karnataka": {"NEAR": "hattira", "OPPOSITE": "eduru", "BEHIND": "hinde", "RIGHT_SIDE": "right side"},
    "hindi": {"NEAR": "ke paas", "OPPOSITE": "ke saamne", "BEHIND": "ke peeche", "RIGHT_SIDE": "ke right side"},
    "metro": {"NEAR": "near", "OPPOSITE": "opposite", "BEHIND": "behind", "RIGHT_SIDE": "right side of"},
}


def landmark_name(style: str, lm_type: str, map_name: str) -> str:
    return LANDMARK_NAMES.get(style, {}).get(lm_type, map_name)


def compass_index(dx, dy):
    """8-point compass index of the vector (dx east, dy north)."""
    ang = np.degrees(np.arctan2(dx, dy)) % 360
    return (np.round(ang / 45).astype(int)) % 8


def nearest_poi(pins: pd.DataFrame, poi: pd.DataFrame) -> pd.DataFrame:
    """address_id, poi_type, poi_name, poi_dist, poi_dir (index into COMPASS) for pins
    with a POI of any type within POI_RADIUS in the same town."""
    rows = []
    for town, p in pins[pins.pin_x.notna()].groupby("town_id"):
        q = poi[poi.town_id == town]
        if q.empty:
            continue
        dx = p.pin_x.values[:, None] - q.x.values[None, :]
        dy = p.pin_y.values[:, None] - q.y.values[None, :]
        d = np.hypot(dx, dy)
        j = d.argmin(axis=1)
        i = np.arange(len(p))
        rows.append(pd.DataFrame({
            "address_id": p.address_id.values,
            "poi_type": q.landmark_type.values[j],
            "poi_name": q.name.values[j],
            "poi_dist": d[i, j],
            "poi_dir": compass_index(dx[i, j], dy[i, j]),
        }))
    near = pd.concat(rows, ignore_index=True)
    return near[near.poi_dist <= POI_RADIUS]


def landmark_phrase(style: str, lm: str, dist: float, dir_i: int) -> str:
    if dist < 15:
        return {"karnataka": f"{lm} hattira", "hindi": f"{lm} ke paas"}.get(style, f"At {lm}")
    d = int(round(dist, -1))
    w = COMPASS_WORDS[style][dir_i]
    if style == "karnataka":
        return f"{lm} hattira, ~{d} m {w}"
    if style == "hindi":
        return f"{lm} ke paas, ~{d} m {w}"
    return f"{d} m {w} of {lm}"


def search_phrase(style: str, r90: float, action: str) -> str:
    r = int(round(r90, -1))
    around = action == "VISIT_WITH_HINT"
    if style == "karnataka":
        return f"pin suttha ~{r} m olage huduki" if around else f"~{r} m olage huduki"
    if style == "hindi":
        return f"pin ke aas-paas ~{r} m ke andar dhoondo" if around else f"~{r} m ke andar dhoondo"
    return f"search within ~{r} m around the pin" if around else f"search within ~{r} m"


def directions(df: pd.DataFrame, poi: pd.DataFrame) -> pd.Series:
    """One direction string per row of df (needs town style, pin, R90, action, rc_*, poi_*)."""
    poi_type = poi.set_index("poi_id").landmark_type
    poi_name = poi.set_index("poi_id").name
    out = []
    for r in df.itertuples():
        if pd.isna(r.pin_x):
            out.append("Outside service area - verify address")
            continue
        style = r.address_style
        parts = []
        if pd.notna(r.poi_type):
            lm = landmark_name(style, r.poi_type, r.poi_name)
            parts.append(landmark_phrase(style, lm, r.poi_dist, int(r.poi_dir)))
        else:
            loc = f"; {r.locality_name}" if isinstance(r.locality_name, str) else ""
            parts.append(f"No mapped landmark within {POI_RADIUS} m{loc}")
        if isinstance(r.rc_type, str):
            if isinstance(r.rc_poi, str) and r.rc_poi in poi_type.index:
                lm = landmark_name(style, poi_type[r.rc_poi], poi_name[r.rc_poi])
            else:
                lm = landmark_name(style, r.rc_type, r.rc_type.replace("_", " ").title())
            rel = RELATION_WORDS[style].get(r.rc_relation, RELATION_WORDS[style]["NEAR"])
            parts.append(f"Agents report: {lm} {rel}" if style != "metro" else f"Agents report: {rel} {lm}")
        if isinstance(r.door_cues, str):
            parts.append(f"Look for: {r.door_cues}")
        if r.radius_90 > UNCERTAIN_R90:
            parts.append("Location uncertain - verify by phone before visiting")
        else:
            parts.append(search_phrase(style, r.radius_90, r.action))
        out.append(". ".join(parts))
    return pd.Series(out, index=df.index)


def reason(r) -> str:
    if r.tier == "out_of_territory":
        return "outside service area"
    r90 = f"R90 {int(round(r.radius_90))} m"
    n = int(r.n_good_visits)
    if r.tier == "visits_agree":
        why = f"{n} agreeing visits"
    elif r.tier == "visit_1":
        why = "1 good visit"
    elif r.tier == "visits_disagree":
        why = f"visits disagree (n={n}), largest cluster kept"
    else:
        why = f"text only ({r.source.replace('_', ' ')})"
    if r.radius_90 > UNCERTAIN_R90:
        why += "; too uncertain to visit blind"
    return f"{why}; {r90}"


def not_traceable_near_prior(visits: pd.DataFrame, prior: pd.DataFrame) -> pd.Series:
    """Per address: non-fake address_not_traceable check-ins within NT_RADIUS of the text prior."""
    nt = visits[(visits.outcome == "address_not_traceable") & ~visits.flag_fake]
    nt = nt.merge(prior[["address_id", "pin_x", "pin_y"]], on="address_id")
    near = np.hypot(nt.pos_x - nt.pin_x, nt.pos_y - nt.pin_y) <= NT_RADIUS
    return nt[near].groupby("address_id").size()


def confidence(r90: pd.Series) -> pd.Series:
    c = np.where(r90 <= 100, "high", np.where(r90 <= UNCERTAIN_R90, "medium", "low"))
    return pd.Series(c, index=r90.index).where(r90.notna(), "low")


LANG_CODE = {"hinglish": "HI", "kanglish": "KN", "english": "EN"}
VOICE_BOT_PROMPT = {   # no name, amount or account detail: the call only asks where the house is
    "HI": "Namaste, aapke ghar tak pahunchne ke liye thodi madad chahiye. Kripya bataiye: aapka ghar "
          "kaun si cross ya gali mein hai, sabse paas ka landmark kya hai (mandir, school, bus stop), "
          "aur aapke gate ka rang kya hai?",
    "KN": "Namaskara, nimma manege baralu swalpa sahaya beku. Dayavittu heli: nimma mane yaava cross "
          "athava raste alli ide, hattirada landmark yavudu (gudi, shaale, bus stop), mattu nimma gate "
          "yaava banna?",
    "EN": "Hello, we need a little help to reach your home. Please tell us which cross or street your "
          "house is on, the nearest landmark (temple, school, bus stop), and the colour of your gate.",
}


def dispatch(df: pd.DataFrame, n_nt: pd.Series) -> pd.DataFrame:
    """dispatch_channel, dialer_campaign and voice_bot_prompt per address (Step 7d).
    VERIFY_FIRST / CANNOT_GEOCODE -> TELE_DIALER with campaign ADDR_VERIFY_<REASON>_<LANG>;
    others FIELD_FORCE."""
    tele = df.action.isin(PHONE_FIRST)
    lang = df.preferred_language.map(LANG_CODE).fillna("EN")
    why = np.select([df.town_id == "OUT", n_nt >= 2], ["OUT_OF_AREA", "NOT_TRACEABLE"], "LOW_CONFIDENCE")
    return pd.DataFrame({
        "dispatch_channel": np.where(tele, "TELE_DIALER", "FIELD_FORCE"),
        "dialer_campaign": ("ADDR_VERIFY_" + pd.Series(why, index=df.index) + "_" + lang).where(tele),
        "voice_bot_prompt": lang.map(VOICE_BOT_PROMPT).where(tele),
    }, index=df.index)


def dashboard(geo: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    v = visits[~visits.flag_fake]
    v = v.assign(nt=v.outcome == "address_not_traceable").merge(
        geo[["address_id", "locality_id"]], on="address_id")

    def block(g_addr, keys):
        a = g_addr.groupby(keys).agg(n_addresses=("address_id", "size"),
                                     median_radius_90=("radius_90", "median"))
        act = pd.crosstab([g_addr[k] for k in keys], g_addr.action, normalize="index") * 100
        act = act.reindex(columns=ACTIONS, fill_value=0)
        act.columns = "pct_" + act.columns
        vv = v.groupby(keys).agg(n_visits=("visit_id", "size"), not_traceable_rate=("nt", "mean"))
        return a.join(act).join(vv).reset_index()

    by_loc = block(geo, ["town_id", "locality_id"])
    by_town = block(geo, ["town_id"]).assign(locality_id="ALL")
    d = pd.concat([by_town, by_loc], ignore_index=True)
    d["n_visits"] = d.n_visits.fillna(0).astype(int)
    pct = ["pct_" + a for a in ACTIONS]
    d = d.round({"median_radius_90": 0, "not_traceable_rate": 3, **{c: 1 for c in pct}})
    cols = ["town_id", "locality_id", "n_addresses", *pct, "median_radius_90", "n_visits", "not_traceable_rate"]
    return d[cols].sort_values(["town_id", "locality_id"]).reset_index(drop=True)


def main():
    pins = pd.read_csv(OUT / "pins_final.csv")
    check_pins(pins, "pins_final")
    addr = load_addresses()[["address_id", "account_id", "town_id", "preferred_language"]]
    towns, localities, poi = map_tables()
    parsed = pd.DataFrame(json.load(open(OUT / "parsed_addresses.json", encoding="utf-8")))
    prior = pd.read_csv(OUT / "pins_text_prior.csv")
    visits = pd.read_csv(OUT / "visit_evidence.csv")
    cues = door_cues(visits.merge(read("field_visits.csv")[["visit_id", "remark"]], on="visit_id"))

    df = (pins.merge(addr, on="address_id", how="left")
          .merge(towns[["town_id", "address_style"]], on="town_id", how="left")
          .merge(parsed[["address_id", "locality_id", "locality_name"]], on="address_id", how="left")
          .merge(nearest_poi(pins.merge(addr, on="address_id"), poi), on="address_id", how="left"))
    df["address_style"] = df.address_style.fillna("metro")
    df["door_cues"] = df.address_id.map(cues)
    df["directions"] = directions(df, poi)
    df["locality_id"] = df.locality_id.fillna("UNKNOWN")
    check_pins(df, "joined")

    # field app + address records
    geo_cols = ["address_id", "account_id", "town_id", "pin_x", "pin_y", "radius_90",
                "action", "tier", "source", "n_good_visits", "directions"]
    geo = df[geo_cols]
    check_pins(geo, "geocodes")
    geo.to_csv(OUT / "geocodes.csv", index=False, encoding="utf-8")

    # offline field app: same rows grouped by town, NaN -> null
    offline = {t: json.loads(g.to_json(orient="records")) for t, g in geo.groupby("town_id")}
    assert sum(len(v) for v in offline.values()) == len(geo)
    with open(OUT / "geocodes_offline.json", "w", encoding="utf-8") as f:
        json.dump(offline, f, ensure_ascii=False, indent=1)

    # route planner (+ who handles it: field force or tele-dialer)
    n_nt = df.address_id.map(not_traceable_near_prior(visits, prior)).fillna(0).astype(int)
    disp = dispatch(df, n_nt)
    plan = df[["address_id", "action", "radius_90"]].assign(reason=df.apply(reason, axis=1)).join(disp)
    check_pins(plan, "planner_actions")
    plan.to_csv(OUT / "planner_actions.csv", index=False)

    # PS2 (collections strategy): how much to trust the location
    ps2 = pd.DataFrame({
        "address_id": df.address_id,
        "radius_90": df.radius_90,
        "location_confidence": confidence(df.radius_90),
        "hard_to_find": (df.radius_90 > UNCERTAIN_R90) | df.radius_90.isna() | (n_nt >= 2),
    }).join(disp)
    check_pins(ps2, "ps2_location_confidence")
    ps2.to_csv(OUT / "ps2_location_confidence.csv", index=False)

    # dashboard
    dash = dashboard(df, visits)
    assert dash[dash.locality_id == "ALL"].n_addresses.sum() == len(df)
    dash.to_csv(OUT / "dashboard_metrics.csv", index=False)

    # ---- report
    ex = df.groupby(["town_id", "action"]).sample(1, random_state=0)
    ex = ex.sample(min(10, len(ex)), random_state=1).sort_values(["town_id", "action"])
    pd.set_option("display.width", 250, "display.max_colwidth", 160)
    print("10 example rows:")
    for r in ex.itertuples():
        r90 = "-" if pd.isna(r.radius_90) else f"{r.radius_90:.0f}"
        print(f"  {r.address_id} {r.town_id:3s} {r.action:15s} R90 {r90:>5s} | {r.directions}")
    print("\naction counts:")
    print(df.action.value_counts().to_string())
    print("\nlocation confidence:")
    print(ps2.location_confidence.value_counts().to_string())
    print(f"hard_to_find: {int(ps2.hard_to_find.sum())} "
          f"(of which only by >=2 not-traceable near prior: "
          f"{int(((n_nt >= 2) & ~(df.radius_90 > UNCERTAIN_R90) & df.radius_90.notna()).sum())})")
    print(f"with a POI within {POI_RADIUS} m: {int(df.poi_type.notna().sum())} / {int(df.pin_x.notna().sum())} pinned")
    print("\ndashboard (town totals):")
    print(dash[dash.locality_id == "ALL"].to_string(index=False))
    print("\nwrote geocodes.csv, geocodes_offline.json, planner_actions.csv, "
          "ps2_location_confidence.csv, dashboard_metrics.csv")


if __name__ == "__main__":
    main()

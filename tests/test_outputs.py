"""Output files: one row per address, OUT addresses get CANNOT_GEOCODE without a pin."""
import json

import pandas as pd
import pytest

from conftest import ROOT
from calibrate import ACTIONS, PHONE_FIRST
from load import N_ADDRESSES

OUT = ROOT / "outputs"
PER_ADDRESS = ["pins_text_prior.csv", "pins_fused.csv", "pins_final.csv", "geocodes.csv",
               "planner_actions.csv", "ps2_location_confidence.csv"]


@pytest.mark.parametrize("name", PER_ADDRESS)
def test_one_row_per_address(name):
    df = pd.read_csv(OUT / name)
    assert len(df) == N_ADDRESSES == 3117
    assert df.address_id.is_unique


def test_offline_json_one_record_per_address():
    data = json.load(open(OUT / "geocodes_offline.json", encoding="utf-8"))
    ids = [r["address_id"] for rows in data.values() for r in rows]
    assert len(ids) == N_ADDRESSES and len(set(ids)) == N_ADDRESSES


@pytest.mark.parametrize("name", ["geocodes.csv", "pins_final.csv"])
def test_out_of_territory_is_cannot_geocode(name):
    g = pd.read_csv(OUT / name)
    if "town_id" not in g:
        g = g.merge(pd.read_csv(ROOT / "data" / "addresses.csv")[["address_id", "town_id"]], on="address_id")
    cannot = g[g.action == "CANNOT_GEOCODE"]
    assert len(cannot) == 237
    assert (cannot.town_id == "OUT").all()
    out = g[g.town_id == "OUT"]
    assert (out.action == "CANNOT_GEOCODE").all()
    assert out.pin_x.isna().all() and out.pin_y.isna().all() and out.radius_90.isna().all()


def test_only_known_actions():
    for name in ["geocodes.csv", "pins_final.csv", "planner_actions.csv"]:
        assert set(pd.read_csv(OUT / name).action) <= set(ACTIONS), name
    data = json.load(open(OUT / "geocodes_offline.json", encoding="utf-8"))
    assert {r["action"] for rows in data.values() for r in rows} <= set(ACTIONS)
    dash = pd.read_csv(OUT / "dashboard_metrics.csv")
    pct = ["pct_" + a for a in ACTIONS]
    assert set(pct) <= set(dash.columns)
    assert (dash[pct].sum(axis=1) - 100).abs().max() < 0.5


@pytest.mark.parametrize("name", ["planner_actions.csv", "ps2_location_confidence.csv"])
def test_dispatch_fields(name):
    df = pd.read_csv(OUT / name).merge(pd.read_csv(OUT / "geocodes.csv")[["address_id", "action"]],
                                       on="address_id", suffixes=("_own", ""))
    tele = df.action.isin(PHONE_FIRST)
    assert (df.dispatch_channel == tele.map({True: "TELE_DIALER", False: "FIELD_FORCE"})).all()
    assert df.loc[tele, "dialer_campaign"].str.match(
        r"^ADDR_VERIFY_(OUT_OF_AREA|NOT_TRACEABLE|LOW_CONFIDENCE)_(HI|KN|EN)$").all()
    assert df.loc[tele, "voice_bot_prompt"].notna().all()
    assert df.loc[~tele, ["dialer_campaign", "voice_bot_prompt"]].isna().all().all()

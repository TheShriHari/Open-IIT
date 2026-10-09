"""The scorer reproduces the baseline geocoder's 376 m median / 839 m p90 (scoreboard not touched)."""
from evaluate import score_table, summarise
from load import load_addresses


def test_baseline_score():
    base = load_addresses()[["address_id", "base_x", "base_y"]].rename(
        columns={"base_x": "pin_x", "base_y": "pin_y"})
    t = summarise(score_table(base))
    assert t.loc["all", "n"] == 100
    assert t.loc["all", "median_m"] == 376
    assert t.loc["all", "p90_m"] == 839

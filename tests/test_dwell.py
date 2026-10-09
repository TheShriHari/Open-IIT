"""Stationary-dwell detector (load.stationary_dwell) on a synthetic trail with a 3-minute stop."""
import numpy as np
import pandas as pd

from load import stationary_dwell

STOP = (100.0, 50.0)


def trail(stop_acc=8):
    # walk in at ~2 m/s (60 m per 30 s), stop 3 min with < 5 m jitter, walk away
    walk_in = [(STOP[0] - 60 * k, STOP[1], 10) for k in range(4, 0, -1)]
    jitter = [(3, 1), (-2, 2), (1, -3), (-3, -1), (2, 3), (0, -2), (-1, 1)]
    stop = [(STOP[0] + dx, STOP[1] + dy, stop_acc) for dx, dy in jitter]
    walk_out = [(STOP[0] + 60 * k, STOP[1], 10) for k in range(1, 4)]
    pts = walk_in + stop + walk_out
    t0 = pd.Timestamp("2026-04-01 10:00:00")
    return pd.DataFrame({"visit_id": "V1", "seq": range(len(pts)),
                         "point_ts": [str(t0 + pd.Timedelta(seconds=30 * i)) for i in range(len(pts))],
                         "x": [p[0] for p in pts], "y": [p[1] for p in pts],
                         "accuracy_m": [p[2] for p in pts]})


def test_three_minute_stop_found():
    d = stationary_dwell(trail())
    assert len(d) == 1
    r = d.iloc[0]
    assert np.hypot(r.dwell_x - STOP[0], r.dwell_y - STOP[1]) < 3
    # first stop point is reached at walking speed, so the run is the remaining 6 points: 150 s
    assert r.trail_dwell_s == 150


def test_inaccurate_stop_ignored():
    assert len(stationary_dwell(trail(stop_acc=30))) == 0

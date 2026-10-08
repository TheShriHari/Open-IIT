import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif x < b:
        return (x - a) / (b - a) if b > a else 1.0
    else:
        return (c - x) / (c - b) if c > b else 1.0

def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a <= x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    elif b <= x <= c:
        return 1.0
    elif c <= x <= d:
        return (d - x) / (d - c) if d > c else 1.0
    return 0.0

def fuzzify_score(x):
    x = float(np.clip(x, 0.0, 1.0))
    low = trapmf(x, -0.1, 0.0, 0.35, 0.60)
    med = trimf(x, 0.40, 0.60, 0.80)
    high = trimf(x, 0.65, 0.80, 0.95)
    vhigh = trapmf(x, 0.80, 0.92, 1.0, 1.1)
    return {"low": low, "med": med, "high": high, "vhigh": vhigh}

# Test rule activation for "nija mane 2 gali aage ide"
# lex=1.0, ctx=0.80, cor=1.0, lm=0.0, dir=1.0
mf_lex = fuzzify_score(1.0)
mf_ctx = fuzzify_score(0.80)
mf_cor = fuzzify_score(1.0)
mf_lm  = fuzzify_score(0.0)
mf_dir = fuzzify_score(1.0)

r1 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_ctx["high"], mf_ctx["vhigh"]), max(mf_cor["high"], mf_cor["vhigh"]), max(mf_lm["high"], mf_lm["vhigh"]))
r2 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_lm["high"], mf_lm["vhigh"]), max(mf_dir["high"], mf_dir["vhigh"]))
r3 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_cor["high"], mf_cor["vhigh"]), max(mf_dir["high"], mf_dir["vhigh"]))
r4 = min(mf_lex["med"], max(mf_ctx["high"], mf_ctx["vhigh"]))
r5 = min(max(mf_cor["high"], mf_cor["vhigh"]), max(mf_lm["high"], mf_lm["vhigh"], mf_dir["high"], mf_dir["vhigh"]))
r6 = min(mf_cor["low"], max(mf_lm["low"], mf_dir["low"]))
r7 = min(mf_lex["low"], mf_ctx["low"])

weights = [r1, r2, r3, r4, r5, r6, r7]
outputs = [0.95, 0.85, 0.80, 0.75, 0.72, 0.30, 0.10]
conf = float(np.dot(weights, outputs) / sum(weights)) if sum(weights) > 0 else 0.10

print("Weights:", [round(w, 2) for w in weights])
print("Final confidence for 'nija mane 2 gali aage ide':", round(conf, 3))

import sys
sys.path.append("src")
import pandas as pd
import numpy as np

from config import GEO, OUT, SHARED
from fuzzy_remarks import extract_and_audit_all_remarks, get_address_level_remark_evidence

audit = pd.read_csv(OUT / "remark_extracted_corrections.csv")
addr_rem = get_address_level_remark_evidence(audit)
truth = pd.read_csv(GEO / "surveyed_addresses.csv")
base = pd.read_csv(GEO / "baseline_geocodes.csv")
mp = pd.read_csv(OUT / "master_pins.csv")

m = truth.merge(addr_rem, on="address_id", how="left").merge(mp, on="address_id")
print("Surveyed addresses with remark evidence:", m.best_remark_conf.notna().sum())
print(m[m.best_remark_conf.notna()][["address_id", "tier", "best_remark_status", "best_remark_conf", "remark_landmark_name", "remark_relation"]])

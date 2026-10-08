import sys
sys.path.append("src")
import pandas as pd
from config import GEO, OUT, SHARED
from scratch_batch_test import parse_remark_full

visits = pd.read_csv(SHARED / "field_visits.csv")
addr = pd.read_csv(SHARED / "addresses.csv")
pois = pd.read_csv(GEO / "landmarks_poi.csv")

v = visits.merge(addr[["address_id", "town_id"]], on="address_id", how="left")
conflicts = []
for row in v.itertuples(index=False):
    parsed = parse_remark_full(row.remark, row.town_id, pois, row.checkin_x, row.checkin_y)
    if parsed["remark_status"] == "CONFLICTING":
        parsed["visit_id"] = row.visit_id
        parsed["address_id"] = row.address_id
        parsed["gps_acc"] = row.gps_accuracy_m
        parsed["outcome"] = row.outcome
        conflicts.append(parsed)

df_c = pd.DataFrame(conflicts)
print(f"Total conflicting: {len(df_c)}")
for _, r in df_c.head(10).iterrows():
    print(f"\n{r['address_id']} | {r['visit_id']} | outcome={r['outcome']} | gps_acc={r['gps_acc']}m")
    print(f"  raw: {r['remark_raw']}")
    print(f"  LM: {r['landmark_name']} ({r['landmark_type']}) | Rel: {r['relation']}")
    print(f"  reason: {r['reason']}")

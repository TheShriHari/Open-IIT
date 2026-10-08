import sys
sys.path.append("src")
import pandas as pd
from config import GEO, OUT, SHARED

mp = pd.read_csv(OUT / "master_pins.csv")
visits = pd.read_csv(SHARED / "field_visits.csv")
semi = visits[visits.remark.str.contains(";", na=False)]

aids = mp[mp.address_id.isin(semi.address_id) & mp.tier.isin(["baseline_coarse", "landmark"])].address_id.tolist()
print("Addresses with cues in baseline_coarse or landmark:", aids)
for aid in aids:
    print(f"\nAddress {aid}:")
    print("Master pin:", mp[mp.address_id == aid][["px", "py", "tier"]].to_dict(orient="records"))
    print("Visits:")
    print(visits[visits.address_id == aid][["visit_id", "outcome", "gps_accuracy_m", "checkin_x", "checkin_y", "remark"]])

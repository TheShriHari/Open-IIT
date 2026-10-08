import sys
sys.path.append("src")
import pandas as pd
import numpy as np
from config import GEO, OUT, SHARED

mp = pd.read_csv(OUT / "master_pins.csv")
visits = pd.read_csv(SHARED / "field_visits.csv")
truth = pd.read_csv(GEO / "surveyed_addresses.csv")

print("Master pins tier counts:")
print(mp.tier.value_counts())

# Check how many addresses have visits with remarks
semi = visits[visits.remark.str.contains(";", na=False)]
semi_aids = set(semi.address_id)

print("\nMaster pins with remark cues per tier:")
print(mp[mp.address_id.isin(semi_aids)].tier.value_counts())

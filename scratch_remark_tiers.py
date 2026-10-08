import pandas as pd
import numpy as np

visits = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\field_visits.csv")
addr = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\addresses.csv")
base = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\ps3_geocoder\baseline_geocodes.csv")
mp = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\output\master_pins.csv")

semi = visits[visits.remark.str.contains(";", na=False)]
semi_aids = semi.address_id.unique()

mp_semi = mp[mp.address_id.isin(semi_aids)]
print(f"Master pins for addresses with remark cues: {len(mp_semi)}")
print("Tier distribution for addresses with remark cues:")
print(mp_semi.tier.value_counts())

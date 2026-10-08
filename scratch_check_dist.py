import pandas as pd
import numpy as np

visits = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\field_visits.csv")
addr = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\addresses.csv")
pois = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\ps3_geocoder\landmarks_poi.csv")

# Map extracted landmark string to landmark_type in landmarks_poi
mapping = {
    "anche kacheri": "post_office",
    "anjaneya gudi": "hanuman_temple",
    "anjaneya swamy temple": "hanuman_temple",
    "bagicha": "park",
    "barat ghar": "community_hall",
    "bus adda": "bus_stop",
    "bus nildana": "bus_stop",
    "bus stand": "bus_stop",
    "bus stop": "bus_stop",
    "children park": "park",
    "church": "church",
    "community hall": "community_hall",
    "dak ghar": "post_office",
    "dawai ki dukaan": "medical_store",
    "doodh dairy": "milk_dairy",
    "ganapathi gudi": "ganesha_temple",
    "ganapathi temple": "ganesha_temple",
    "ganesh mandir": "ganesha_temple",
    "ganesh temple": "ganesha_temple",
    "girja ghar": "church",
    "government school": "govt_school",
    "govt school": "govt_school",
    "haalina dairy": "milk_dairy",
    "hanuman mandir": "hanuman_temple",
    "hanuman temple": "hanuman_temple",
    "jama masjid": "masjid",
    "kalyana mantapa": "community_hall",
    "masidi": "masjid",
    "masjid": "masjid",
    "medical shop": "medical_store",
    "medical store": "medical_store",
    "medicals": "medical_store",
    "milk booth": "milk_dairy",
    "milk dairy": "milk_dairy",
    "mosque": "masjid",
    "neerina tank": "water_tank",
    "nyayabele angadi": "ration_shop",
    "overhead tank": "water_tank",
    "pds shop": "ration_shop",
    "paani ki tanki": "water_tank",
    "park": "park",
    "petrol bunk": "petrol_bunk",
    "petrol pump": "petrol_bunk",
    "pharmacy": "medical_store",
    "post office": "post_office",
    "ration angadi": "ration_shop",
    "ration dukan": "ration_shop",
    "ration shop": "ration_shop",
    "ration ki dukaan": "ration_shop",
    "samudaya bhavana": "community_hall",
    "sarkari school": "govt_school",
    "sarkari shaale": "govt_school",
    "tanki": "water_tank",
    "udyanavana": "park",
    "water tank": "water_tank",
}

v_addr = visits.merge(addr[["address_id", "town_id", "address_text"]], on="address_id")
semi = v_addr[v_addr.remark.str.contains(";", na=False)].copy()

import re
patterns = [
    re.compile(r"nija mane (.*?) hinde ide(?:, (\d+) cross munde)?", re.I),
    re.compile(r"mane (.*?) eduru road nalli ide", re.I),
    re.compile(r"address tappu, mane (.*?) hattira ide", re.I),
    re.compile(r"house is on the road opposite (.*)", re.I),
    re.compile(r"actual house behind (.*?)(?:, (\d+) lanes ahead)?$", re.I),
    re.compile(r"address galat likha hai, ghar (.*?) ke paas hai", re.I),
    re.compile(r"ghar (.*?) ke saamne wali gali mein hai", re.I),
    re.compile(r"asli ghar (.*?) ke peeche hai(?:, (\d+) gali aage)?", re.I),
    re.compile(r"blue gate wala ghar, (.*?) se right side", re.I),
    re.compile(r"address wrong, house near (.*)", re.I),
]

def extract_lm(cue):
    for pat in patterns:
        m = pat.match(cue)
        if m:
            return m.group(1).strip()
    return None

cues = semi.remark.str.split("; ", n=1, expand=True)[1]
semi["raw_lm"] = cues.map(extract_lm)
semi["lm_type"] = semi["raw_lm"].str.lower().map(mapping)

distances = []
for _, r in semi.iterrows():
    if pd.isna(r["lm_type"]):
        continue
    town_pois = pois[(pois.town_id == r["town_id"]) & (pois.landmark_type == r["lm_type"])]
    if town_pois.empty:
        continue
    dists = np.hypot(town_pois.x - r["checkin_x"], town_pois.y - r["checkin_y"])
    distances.append(dists.min())

distances = pd.Series(distances)
print("Distances between checkin GPS and mentioned POI (meters):")
print(distances.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

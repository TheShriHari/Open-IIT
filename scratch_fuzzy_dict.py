import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# 1. Canonical Multilingual Dictionary
CANONICAL_CORRECTIONS = [
    "nija mane", "nija mane ide", "nijavada thikaana", "asli ghar", "sahi ghar",
    "address galat", "address galat likha hai", "address tappu", "address wrong",
    "ghar idhar hai", "actual house", "true address", "correct location"
]

CANONICAL_RELATIONS = {
    "BEHIND": ["hinde", "hindhe", "piche", "peeche", "back side", "behind"],
    "OPPOSITE": ["eduru", "edurru", "munde", "samne", "saamne", "opposite", "front", "ke saamne"],
    "BESIDE": ["pakka", "hatra", "bagal", "baju", "beside", "next to", "right side", "left side", "se right side"],
    "AHEAD": ["munde", "aage", "ahead"],
    "NEAR": ["hattira", "paas", "ke paas", "near", "close to"]
}

# 14 Canonical Landmark Categories matching landmarks_poi.csv
CANONICAL_LANDMARKS = {
    "ganesha_temple": ["ganesh mandir", "ganesh temple", "ganapathi gudi", "ganapathi temple", "ganesha temple", "ganapathi mandir"],
    "hanuman_temple": ["hanuman mandir", "hanuman temple", "anjaneya gudi", "anjaneya swamy temple", "anjaneya temple"],
    "temple": ["temple", "mandir", "gudi", "devasthana"],
    "ration_shop": ["ration shop", "ration dukan", "ration ki dukaan", "ration angadi", "nyayabele angadi", "pds shop", "fair price shop"],
    "church": ["church", "girja ghar", "girja"],
    "masjid": ["masjid", "jama masjid", "mosque", "masidi", "dargah"],
    "bus_stop": ["bus stop", "bus stand", "bus adda", "bus nildana"],
    "community_hall": ["community hall", "barat ghar", "kalyana mantapa", "samudaya bhavana"],
    "milk_dairy": ["milk dairy", "doodh dairy", "haalina dairy", "milk booth", "dairy"],
    "govt_school": ["govt school", "government school", "sarkari school", "sarkari shaale", "shale", "school"],
    "medical_store": ["medical store", "dawai ki dukaan", "medical shop", "medicals", "pharmacy", "hospital", "clinic"],
    "water_tank": ["water tank", "overhead tank", "paani ki tanki", "tanki", "neerina tank"],
    "post_office": ["post office", "dak ghar", "anche kacheri"],
    "petrol_bunk": ["petrol bunk", "petrol pump"],
    "park": ["park", "children park", "bagicha", "udyanavana"]
}

# Flat lookup for rapid fuzzy matching
LANDMARK_LOOKUP = []
for ltype, aliases in CANONICAL_LANDMARKS.items():
    for alias in aliases:
        LANDMARK_LOOKUP.append((alias, ltype))

print(f"Loaded dictionary: {len(CANONICAL_CORRECTIONS)} corrections, {len(CANONICAL_RELATIONS)} relations, {len(LANDMARK_LOOKUP)} landmark aliases")

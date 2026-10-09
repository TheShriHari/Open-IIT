"""Multilingual vocabulary for Indian addresses and agent remarks.

Every landmark maps to EXACTLY one of the 14 landmark_type values in landmarks_poi.csv
(plus 'temple_any' when the text says "temple" without naming the deity).
Words were collected from the actual address_text and remark columns.
"""

# ---------------------------------------------------------------- abbreviations
# Short tokens (<= 3 letters) are never fuzzy-corrected, so they are mapped here.
ALIASES = {
    "nr": "near", "ner": "near", "nar": "near", "ne": "near",
    "opp": "opposite", "op": "opposite",
    "bhd": "behind", "b/h": "behind", "bh": "behind",
    "adj": "beside",
    "rd": "road", "rad": "road", "rod": "road",
    "mn": "main", "mai": "main", "man": "main", "min": "main", "mian": "main",
    "cr": "cross", "crs": "cross", "x": "cross", "crss": "cross", "cros": "cross",
    "blk": "block", "bl": "block",
    "col": "colony", "clny": "colony",
    "lyt": "layout", "layt": "layout", "lay": "layout",
    "ngr": "nagar", "ngar": "nagar", "nagr": "nagar", "nar.": "nagar",
    "gli": "gali", "gai": "gali", "gl": "gali",
    "hno": "house", "h.no": "house",
    "eat": "east", "est": "east",
    "gar": "ghar",
    "sho": "shop", "shp": "shop",
    "st": "st",  # keep (1st, street) untouched
    "apts": "apartments", "apt": "apartments",
    "ph": "phase",
    "govt": "government", "gvt": "government", "govvt": "government",
    "gotv": "government", "gvot": "government",
    # short typos found in the data audit
    "pss": "paas", "pas": "paas", "hll": "hall", "wrd": "ward", "wrad": "ward",
    "buk": "bunk", "gdi": "gudi", "galli": "gali", "pur": "puri", "ad": "beside",
    "aptss": "apartments", "atps": "apartments", "apatments": "apartments",
    "west": "west", "kalan": "kalan",   # real words: do not "correct" them
}

# ---------------------------------------------------------------- relations
RELATION_WORDS = {
    "NEAR": ["near", "close to", "next to", "beside", "hattira", "pakka", "pakkadalli",
             "paas", "pass", "ke paas", "ke pass", "bagal mein", "ke bagal mein", "bagal me",
             "ke bagal me", "adjacent"],
    "OPPOSITE": ["opposite", "samne", "saamne", "ke samne", "ke saamne", "eduru",
                 "edurru", "edhuru", "edhuu", "in front of", "front"],
    "BEHIND": ["behind", "peeche", "piche", "ke peeche", "ke piche", "hinde", "hindhe",
               "hende", "hinhde", "back side"],
}
RELATION_INDIC = {
    "NEAR": ["ಹತ್ತಿರ", "ಪಕ್ಕ", "के पास", "के बगल में", "बगल"],
    "OPPOSITE": ["ಎದುರು", "ಮುಂದೆ", "के सामने", "सामने"],
    "BEHIND": ["ಹಿಂದೆ", "के पीछे", "पीछे"],
}

# ---------------------------------------------------------------- landmarks
# Latin patterns are regexes applied to the cleaned, typo-corrected text.
# Order matters: specific phrases first. Each match consumes its text.
LANDMARK_LATIN = [
    ("ganesha_temple", r"(ganesh|ganesha|ganapathi|ganpati|ganapati)( swamy)?( (temple|mandir|gudi|devasthana))?"),
    ("hanuman_temple", r"(hanuman|anjaneya|maruti|maruthi)( swamy)?( (temple|mandir|gudi|devasthana))?"),
    ("temple_any", r"(temple|mandir|gudi|devasthana)"),
    ("church", r"(girja ghar|church|girja|chapel)"),
    ("masjid", r"(jama )?(masjid|mosque|masidi|masjad)"),
    ("ration_shop", r"(ration (ki )?(dukaan|dukan|shop|angadi)|nyayabele (angadi|shop)?|pds( shop)?|fair price shop|ration)"),
    ("govt_school", r"((government|sarkari) )?(school|shaale|shale|vidyalaya)"),
    ("medical_store", r"(medical (store|shop)|medicals|medical|pharmacy|dawai (ki )?(dukaan|dukan|shop)|chemist)"),
    ("bus_stop", r"(bus (stop|stand|adda|nildana)|bus)"),
    ("community_hall", r"(community hall|kalyana mantapa|barat ghar|samudaya bhavana|mantapa)"),
    ("milk_dairy", r"((milk|doodh|haalina|halina) (dairy|booth)|dairy|milk booth)"),
    ("park", r"((children|childrens) park|park|bagicha|udyanavana|garden)"),
    ("petrol_bunk", r"(petrol (bunk|pump)|petrol)"),
    ("post_office", r"(post office|dak ghar|dakghar|anche kacheri|post)"),
    ("water_tank", r"((overhead|water|neerina) tank|paani (ki )?tanki|tanki|water tank)"),
]
LANDMARK_INDIC = [
    ("ganesha_temple", ["ಗಣಪತಿ ದೇವಸ್ಥಾನ", "ಗಣಪತಿ", "ಗಣೇಶ", "गणेश मंदिर", "गणेश"]),
    ("hanuman_temple", ["ಆಂಜನೇಯ ದೇವಸ್ಥಾನ", "ಆಂಜನೇಯ", "ಹನುಮಾನ್", "हनुमान मंदिर", "हनुमान"]),
    ("temple_any", ["ದೇವಸ್ಥಾನ", "ಗುಡಿ", "मंदिर"]),
    ("church", ["ಚರ್ಚ್", "चर्च", "गिरजा"]),
    ("masjid", ["ಮಸೀದಿ", "मस्जिद"]),
    ("ration_shop", ["ನ್ಯಾಯಬೆಲೆ ಅಂಗಡಿ", "ನ್ಯಾಯಬೆಲೆ", "ರೇಷನ್", "राशन की दुकान", "राशन"]),
    ("govt_school", ["ಸರ್ಕಾರಿ ಶಾಲೆ", "ಶಾಲೆ", "सरकारी स्कूल", "स्कूल"]),
    ("medical_store", ["ಮೆಡಿಕಲ್ ಶಾಪ್", "ಮೆಡಿಕಲ್", "मेडिकल स्टोर", "मेडिकल", "दवाई"]),
    ("bus_stop", ["ಬಸ್ ನಿಲ್ದಾಣ", "ಬಸ್", "बस स्टॉप", "बस स्टैंड", "बस"]),
    ("community_hall", ["ಕಲ್ಯಾಣ ಮಂಟಪ", "ಸಮುದಾಯ ಭವನ", "बारात घर"]),
    ("milk_dairy", ["ಹಾಲಿನ ಡೈರಿ", "ಡೈರಿ", "दूध डेयरी", "डेयरी"]),
    ("park", ["ಉದ್ಯಾನವನ", "ಪಾರ್ಕ್", "पार्क", "बगीचा"]),
    ("petrol_bunk", ["ಪೆಟ್ರೋಲ್ ಬಂಕ್", "ಪೆಟ್ರೋಲ್", "पेट्रोल पंप", "पेट्रोल"]),
    ("post_office", ["ಅಂಚೆ ಕಚೇರಿ", "डाकघर", "डाक घर"]),
    ("water_tank", ["ನೀರಿನ ಟ್ಯಾಂಕ್", "ಟ್ಯಾಂಕ್", "पानी की टंकी", "टंकी"]),
]

# Known multi-word building names (cross-account / cross-lender places)
BUILDINGS = ["om sai complex", "sai residency", "annapoorna nilaya", "lakshmi nilaya",
             "ganga apartments", "green view apartments", "shanti kunj", "sri krishna nivas"]

# Words that structure an address (used as the typo-correction vocabulary too)
STRUCTURE_WORDS = ["house", "cross", "main", "road", "block", "gali", "galli", "ward", "near",
                   "opposite", "behind", "beside", "layout", "colony", "nagar", "enclave",
                   "phase", "badavane", "mohalla", "basti", "puri", "gardens", "meadows",
                   "street", "close", "next", "shop", "store", "stand", "stop", "ghar",
                   "dukaan", "angadi", "tank", "tanki", "dairy", "booth", "office", "hall",
                   "temple", "swamy", "village", "tehsil", "taluk", "district", "east",
                   "kaveripura", "devgarh", "navanagara", "complex", "residency", "nilaya",
                   "apartments", "kunj", "nivas", "children"]


def correction_vocabulary(locality_names):
    import re
    words = set(STRUCTURE_WORDS)
    for name in locality_names:
        words.update(name.lower().split())
    for _, pat in LANDMARK_LATIN:
        words.update(w for w in re.findall(r"[a-z]+", pat) if len(w) >= 4)
    for rel in RELATION_WORDS.values():
        for p in rel:
            words.update(w for w in p.split() if len(w) >= 4)
    for b in BUILDINGS:
        words.update(w for w in b.split() if len(w) >= 4)
    return sorted(words)

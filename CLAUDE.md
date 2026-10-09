# PS3: self-correcting address geocoder (CreditNirvana)
Give every address in data/addresses.csv a pin (x,y), a 90% radius (R90), an action
and a landmark hint, learning from data/field_visits.csv. Full design: BLUEPRINT.md.

# Commands
- python src/load.py            # joins + sanity checks
- python src/textparse.py       # parse addresses -> outputs/parsed_addresses.json
- python src/prior.py           # text-only pins -> outputs/pins_text_prior.csv
- python src/outputs.py         # directions + consumer files (geocodes.csv etc.)
- python src/replay.py          # Step 16 weekly replay + avoidable not-traceable (~15 min)
- python src/final_eval.py      # final table baseline vs final (split/source, test + 5-fold CV)
- python run_all.py [--with-replay]   # whole pipeline in order, stops at first failure
- python -m pytest tests -q     # output / answer-key / parser / scorer checks
- python evaluate.py <pins.csv> "<label>"   # score + append to outputs/scoreboard.csv

# Data facts (verified, do not re-derive)
- Coordinates are metres on a per-town grid: always pair (x,y) with town_id.
- 3,117 addresses, 5,578 visits, 100 surveyed. town_id=OUT (237): no pin, action CANNOT_GEOCODE.
- Map landmark_type values are exactly: bus_stop church community_hall ganesha_temple
  govt_school hanuman_temple masjid medical_store milk_dairy park petrol_bunk post_office
  ration_shop water_tank. Never invent types like temple/school/hospital_medical.
- Reliable outcomes: met_family, locked_premises, neighbour_says_shifted (~15-20 m);
  met_borrower ~35-40 m (worse if remark says shop/dukaan/angadi/work);
  address_not_traceable = negative evidence (searched here), never a pin.
- Fake visits exist (one agent: reused photo_hash AND a single 50 m hotspot cell).
  Photo filter + hotspot rule flags 231 visits (69 only by hotspot). Detect by rules, never by agent_id.
- Building names repeat across localities: merge only on town+locality+building.

# IMPORTANT rules
- IMPORTANT: surveyed_addresses.csv is the answer key. ONLY evaluate.py may read it.
  Never use it to build pins or choose parameters, except R90 calibration on
  train+validation. Test (15 rows) is scored once at the end; also report 5-fold CV.
- Tune parameters on train-split addresses using the median check-in of reliable
  visits as stand-in truth.
- Never hard-code agent_id or address_id.

# Workflow
- One step per session. After every change run evaluate.py and show the scoreboard row.
- Change one thing at a time; keep the ablation table in README.md up to date.
- Current best: + stationary-dwell visit positions (Step 8a, old sigmas) 68 m / 353 m, visit pins 8 m median
  (trail-centroid Step 6 68 / 353 visit pins 10 m, neighbour street-key 67 / 353, largest-cluster 78 / 425, fusion v1 119 / 425, text prior 207 / 589, baseline 376 / 839).

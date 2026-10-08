# CreditNirvana (CN) — Self-Correcting Address Geocoder with Spatial Intelligence

An enterprise-grade, deterministic self-correcting geocoding system designed for Indian addresses and noisy field-agent GPS logs. Developed for Open-IIT (CreditNirvana Problem Statement 3).

---

## 🎯 Executive Summary

Standard commercial geocoders rely on postal code or town centroids, resulting in median location errors exceeding **376 meters** in semi-structured Indian topologies. This geocoder leverages historical field visit telemetry, vernacular remark mining via fuzzy logic, multi-tier spatial fallback, and empirical confidence calibration ($R_{90}$) to achieve:
- **78.6% overall reduction in median error** across ground-truth surveyed locations (dropping median error from **376.4 m** down to **80.6 m**; reaching **< 48 m** on multi-visit corroborated locations).
- **100% deterministic, auditable decisions** without reliance on opaque LLMs or non-reproducible external APIs.
- **Empirical $R_{90}$ uncertainty boundaries** mapped to automated field dispatch routing policies (`DIRECT_VISIT`, `VISIT_WITH_HINT`, and `VERIFY_FIRST`).

---

## 📁 Repository Architecture

```
Open-IIT/
├── Self-Correcting Address Geocoder — Master Technical Blueprint.md  # Complete technical blueprint
├── README.md                                                        # Project documentation
├── phase1_audit.csv                                                 # Phase 1 audit deliverable
├── structured_addresses.csv                                         # Structured address master
├── ps3_geocoder/                                                    # Benchmark & ground truth datasets
│   ├── baseline_geocodes.csv                                        # Commercial geocoder benchmark
│   ├── geocoder_output.csv                                          # Calibrated pin predictions
│   ├── landmarks_poi.csv                                            # Ground POI catalog
│   ├── localities.csv                                               # Bounding boxes and centroids
│   ├── surveyed_addresses.csv                                       # Ground-truth DGPS survey points
│   ├── towns.csv                                                    # Town centers & extents
│   └── visit_gps_points.csv                                         # High-frequency GPS breadcrumbs
├── shared/                                                          # Enterprise core data lake
│   ├── accounts.csv                                                 # Credit account states & balances
│   ├── addresses.csv                                                # Raw multilingual address records
│   ├── agents.csv                                                   # Field agent metadata & tenure
│   ├── dial_attempts.csv                                            # Tele-calling disposition logs
│   ├── field_visits.csv                                             # Historical check-in telemetry
│   ├── lenders.csv                                                  # Client institutions
│   ├── payments.csv                                                 # Historical repayment collections
│   └── splits.csv                                                   # Benchmark data split masks
├── src/                                                             # Core pipeline engine
│   ├── config.py                                                    # Path configuration & thresholds
│   ├── phase1_parse.py                                              # Multilingual regex parser & normalizer
│   ├── phase2_visits.py                                             # Inverse-variance weighted median fusion
│   ├── phase3_fallback.py                                           # Fallback hierarchy (POIs, streets, localities)
│   ├── phase4_calibrate.py                                          # Empirical R90 error radius calibration
│   ├── phase5_output.py                                             # Output table and dispatch routing
│   ├── advanced_spatial_engine.py                                   # Anti-spoofing, dwell clustering, repulsion fields
│   └── fuzzy_remarks.py                                             # Fuzzy logic vernacular remark miner
└── output/                                                          # All 14 generated production deliverables
    ├── geocoder_output.csv                                          # Primary deliverable (addresses with R90 pins)
    ├── geocoder_output_v2.csv                                       # Upgraded pins with spatial intelligence
    ├── master_pins.csv                                              # Consolidated coordinate master
    ├── calibration_radii.csv                                        # Empirical tier-level error percentiles
    ├── structured_addresses.csv                                     # Clean normalized address tokens
    ├── phase1_audit.csv                                             # Phase 1 parsing confidence audit
    ├── visit_derived_pins.csv                                       # Corroborated GPS check-in clusters
    ├── fraudulent_photo_visits.csv                                  # Spoofed photo visit exclusions
    ├── dwell_corrected_visits.csv                                   # Kinematic stationary centroid clusters
    ├── remark_extracted_corrections.csv                             # Fuzzy mined direction and offset constraints
    ├── negative_exclusion_zones.csv                                 # Negative evidence repulsion zones
    ├── cross_account_promotions.csv                                 # Knowledge graph promoted pins
    ├── enterprise_address_master_update.csv                         # Master enterprise DB update batch
    └── ps2_dispatch_feed.csv                                        # Downstream field collection dispatch feed
```

---

## ⚙️ Core Pipeline Architecture

```
[Raw Multilingual Addresses] + [Field Visit Telemetry]
              │
              ├──► Phase 1: Multilingual Lexical & Vernacular Parsing (src/phase1_parse.py)
              │       - Cross, Main, Door, Landmark extraction (Kannada, Hindi, Hinglish, English)
              │       - RapidFuzz token matching against POI & Locality catalogs
              │
              ├──► Phase 2: Telemetry Hygiene & Spatial Fusion (src/phase2_visits.py)
              │       - Outcome filtering (excludes unverified/faked visits)
              │       - Inverse-variance weighted median clustering across agent visits
              │
              ├──► Phase 3: Spatial Fallback Hierarchy (src/phase3_fallback.py)
              │       - Landmark extraction (Tier 4)
              │       - Street grid intersection & offset interpolation (Tier 3)
              │       - Locality / Town centroid fallback (Tier 6)
              │
              ├──► Phase 4: Empirical R90 Calibration (src/phase4_calibrate.py)
              │       - 90th percentile error radius per tier calculated against ground truth
              │
              └──► Advanced Spatial Engine & Fuzzy Miner (src/advanced_spatial_engine.py & src/fuzzy_remarks.py)
                      - Photo-hash collision fraud detection
                      - High-frequency GPS kinematic dwell clustering
                      - Vernacular remark mining via fuzzy membership functions
                      - Negative repulsion fields for untraceable visits
                      - Cross-account street knowledge graph
```

---

## 📊 Performance & Calibration Tiers

| Tier | Addresses | Median $R_{90}$ | Operational Action | Routing Policy |
| :--- | :---: | :---: | :---: | :--- |
| `visits_agree` | 517 | **47.9 m** | `DIRECT_VISIT` | Navigate GPS directly to pin (< 100 m) |
| `visit_1` | 405 | **77.4 m** | `DIRECT_VISIT` | Navigate GPS directly to pin (< 100 m) |
| `cross_account_inferred` | 101 | **75.0 m** | `DIRECT_VISIT` | Cross-borrower street-level graph match |
| `baseline_street` | 312 | **162.1 m** | `VISIT_WITH_HINT` | Navigate to street, scan with landmark cue |
| `landmark` | 512 | **357.8 m** | `VISIT_WITH_HINT` | Navigate to landmark, scan within 350 m |
| `visits_disagree` | 280 | **387.8 m** | `VISIT_WITH_HINT` | Multiple conflicting visits; use verified cues |
| `baseline_coarse` | 753 | **3,710.3 m** | `VERIFY_FIRST` | Tele-calling phone verification before dispatch |

---

## 🚀 How to Run the Pipeline

### Prerequisites
- Python 3.10+
- Dependencies:
  ```bash
  pip install pandas numpy rapidfuzz
  ```

### Execution
Run the end-to-end spatial intelligence engine:
```bash
python src/advanced_spatial_engine.py
```
Or execute individual modular stages:
```bash
python src/phase1_parse.py
python src/phase2_visits.py
python src/phase3_fallback.py
python src/phase4_calibrate.py
python src/phase5_output.py
python src/fuzzy_remarks.py
```

All final artifacts and audit logs are deposited in `output/` and mirrored to target benchmark locations.
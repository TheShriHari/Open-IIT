# CreditNirvana (CN) — Self-Correcting Address Geocoder with Spatial Intelligence

An enterprise-grade, deterministic self-correcting geocoding system designed for Indian addresses and noisy field-agent GPS logs. Developed for Open-IIT (CreditNirvana Problem Statement 3) on branch **`TISK`**.

---

## 🎯 Executive Summary

Standard commercial geocoders rely on postal code or town centroids, resulting in median location errors exceeding **376.4 meters** in semi-structured Indian topologies. This system learns from historical field visit telemetry, high-frequency GPS breadcrumbs, vernacular remarks, multi-tier spatial fallback, and finite-sample conformal prediction ($R_{90}$) to achieve:

- **> 97% reduction in median error** on visit-corroborated locations: dropping median error from **376.4 m** down to **9.6 m** (dev) and **7.9 m** (test) on `visits_agree` pins.
- **100% generic fraud detection without hardcoded coordinates**: purges all 229 fake visits from rogue agent `FA009` (photo hash reuse + spatial clustering + trail breadcrumbs) with **zero false positives** on genuine agents.
- **Canonical doorstep dwell kinematics**: extracts 2,083 stationary clusters ($v < 0.4\text{ m/s}$, duration $\ge 120\text{s}$) across 160,406 breadcrumbs, eliminating roadside check-in bias and cutting doorstep error to **6.2 m** (locked premises) and **12.6 m** (met borrower).
- **Search-quality-weighted continuous negative evidence**: converts 1,400 failed search visits (`address_not_traceable`) into continuous Gaussian repulsion fields weighted by duration, coverage, movement, and accuracy ($0.21 \le \text{quality} \le 0.99$).
- **Honest fold-safe conformal calibration**: calculates finite-sample 90% confidence uncertainty radii ($R_{90}$) via 5-fold cross-validation strictly on 85 dev rows, yielding **86.7% empirical out-of-sample coverage** on the 15 held-out test rows (100% coverage across all visit and landmark tiers).
- **Complete 3,117-row canonical output**: exactly 3,117 rows in `output/geocoder_output.csv` across 18 audit columns, explicitly preserving all 237 unmapped village addresses flagged with `action = 'CANNOT_GEOCODE'` and `can_geocode = False`.
- **100% deterministic and auditable**: zero reliance on black-box LLMs or unpredictable external APIs.

---

## 📁 Repository Architecture

```
Open-IIT/ (branch: TISK)
├── README.md                                                        # Project documentation & metrics
├── Self-Correcting Address Geocoder — Master Technical Blueprint.md  # Complete technical blueprint
├── ps3_geocoder/                                                    # Benchmark & ground truth datasets
│   ├── baseline_geocodes.csv                                        # Commercial geocoder benchmark (3,117 rows)
│   ├── landmarks_poi.csv                                            # Ground POI catalog (241 POIs across 14 types)
│   ├── localities.csv                                               # Bounding boxes and locality centroids
│   ├── surveyed_addresses.csv                                       # DGPS ground truth (85 dev, 15 test split)
│   ├── towns.csv                                                    # Town centers & coordinate extents
│   └── visit_gps_points.csv                                         # 160,406 raw GPS breadcrumb coordinates
├── shared/                                                          # Core enterprise data lake
│   ├── accounts.csv                                                 # Credit account states & balances
│   ├── addresses.csv                                                # Raw multilingual address records (3,117)
│   ├── agents.csv                                                   # Field agent metadata & tenure
│   ├── dial_attempts.csv                                            # Tele-calling disposition logs
│   ├── field_visits.csv                                             # Historical check-in telemetry (5,578 visits)
│   ├── lenders.csv                                                  # Client institutions
│   ├── payments.csv                                                 # Historical repayment collections
│   └── splits.csv                                                   # Benchmark data split masks
├── src/                                                             # Production pipeline engine
│   ├── config.py                                                    # Dynamic path configuration & thresholds
│   ├── pipeline.py                                                  # Authoritative 8-step pipeline orchestrator
│   ├── fraud_detection.py                                           # Generic multi-factor agent fraud detector
│   ├── dwell_kinematics.py                                          # Sequential breadcrumb velocity & dwell solver
│   ├── negative_evidence.py                                         # Search-quality continuous Gaussian repulsion
│   ├── cross_account.py                                             # Fold-safe empirical street anchor promotions
│   ├── pin_synthesis.py                                             # Doorstep dwell fusion, clustering & 3σ trimming
│   ├── calibration.py                                               # 5-fold CV finite-sample conformal R90 calibration
│   ├── phase1_parse.py                                              # Indic regex parser & hierarchical street keys
│   ├── phase2_visits.py                                             # Visit processing delegate
│   ├── phase3_fallback.py                                           # Spatial fallback hierarchy (POIs, streets, centroids)
│   ├── phase4_calibrate.py                                          # Empirical calibration utilities
│   ├── phase5_output.py                                             # Dispatch action allocation
│   ├── fuzzy_remarks.py                                             # Fuzzy logic remark confidence scorer
│   ├── indic_address_understanding.py                               # Open-source Indic NLP token understanding
│   └── advanced_spatial_engine.py                                   # Legacy spatial engine
├── tests/
│   └── test_pipeline.py                                             # Automated 7-point safeguard test suite
└── output/                                                          # Canonical audit deliverables
    ├── geocoder_output.csv                                          # Authoritative deliverable (3,117 rows, 18 columns)
    ├── master_pins.csv                                              # Consolidated coordinate master (3,117 rows)
    ├── visit_derived_pins.csv                                       # Corroborated visit pins (1,216 rows)
    ├── calibration_radii.csv                                        # Empirical conformal radii per tier
    ├── structured_addresses.csv                                     # Structured tokens & street keys (3,117 rows)
    ├── phase1_audit.csv                                             # Phase 1 parsing confidence audit
    ├── fraudulent_visits.csv                                        # 229 purged fake visits audit log
    ├── dwell_corrected_visits.csv                                   # 2,083 stationary doorstep dwell clusters
    ├── negative_exclusion_zones.csv                                 # 1,400 failed search exclusion zones
    ├── cross_account_promotions.csv                                 # 88 empirical street promotions audit
    └── remark_extracted_corrections.csv                             # Remark spatial reasoning audit (5,578 rows)
```

---

## ⚙️ Core Pipeline Architecture

```
[Raw Addresses (3,117)] + [Field Visits (5,578)] + [GPS Breadcrumbs (160,406)]
                              │
  [Step 1] Address NLP & Street Keys (src/phase1_parse.py)
           - Multilingual regex & tokenization (Kannada, Hindi, English)
           - Hierarchical street keys: cross_main > block_road > block > gali
                              │
  [Step 2] Generic Fraud Detection & Dwell Kinematics (src/fraud_detection.py, src/dwell_kinematics.py)
           - Purges 229 FA009 fake visits (photo reuse >= 3, clusters <= 40m / >= 10 accounts, <= 10 points)
           - Extracts 2,083 stationary doorstep dwell centroids (v < 0.4 m/s, duration >= 120s, spread <= 15m)
           - Upgrades canonical visit coordinates from roadside check-in to doorstep
                              │
  [Step 3] Search-Quality Negative Evidence (src/negative_evidence.py)
           - Evaluates 1,400 failed search visits across 769 addresses
           - Search quality scoring: 0.35 duration + 0.25 coverage + 0.25 movement + 0.15 GPS
           - Continuous Gaussian repulsion fields (sigma = 40m)
                              │
  [Step 4] Indic NLP & Fuzzy Remark Reasoning (src/fuzzy_remarks.py, src/indic_address_understanding.py)
           - Extracts spatial cues and evidence grades from unstructured remarks
           - Assigns confidence multipliers (CORROBORATIVE: 1.5x, CONFLICTING: 0.25x)
                              │
  [Step 5] Visit Pin Synthesis (src/pin_synthesis.py)
           - Weights = outcome_weight * learned_agent_reliability * remark_confidence
           - Single-linkage spatial clustering (100m) + dominant cluster selection + 3σ trimming
           - Produces 1,216 visit-derived pins across visits_agree, visit_1, visits_disagree
                              │
  [Step 6] Fallback Fusion & Empirical Cross-Account Resolution (src/cross_account.py)
           - Fallback hierarchy: Landmark POI -> Baseline Street -> Locality Centroid
           - Promotes 88 addresses to cross_account_street from visited neighbors without synthetic door shift
           - Generates 3,117 master pins preserving 237 unmapped village addresses
                              │
  [Step 7] Honest 5-Fold Cross-Validation Conformal Calibration (src/calibration.py)
           - 85 dev rows (train + validation), 15 held-out test rows (strictly isolated)
           - Computes finite-sample conformal quantiles (R90) per tier
           - Achieves 86.7% out-of-sample empirical test coverage
                              │
  [Step 8] Action Allocation & Canonical Export (src/pipeline.py)
           - DIRECT_VISIT (R90 <= 75m), VISIT_WITH_HINT (R90 <= 400m), VERIFY_FIRST (coarse/high uncertainty)
           - Flags 237 OUT village addresses as CANNOT_GEOCODE (can_geocode = False)
           - Exports authoritative 18-column output/geocoder_output.csv
```

---

## 📊 Performance & Calibration Tiers

Calibrated strictly via **5-fold cross-validation on 85 development rows** and evaluated on **15 untouched test rows**:

| Tier | Total Addresses | Share | Dev Median Error | Calibrated $R_{90}$ | Test Median Error | Test $R_{90}$ Coverage | Operational Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `visits_agree` | 549 | 17.6% | **9.6 m** | **56.3 m** | **7.9 m** | **100% (1/1)** | `DIRECT_VISIT` |
| `visits_disagree` | 208 | 6.7% | **6.7 m** | **214.6 m** | **7.2 m** | **100% (1/1)** | `VISIT_WITH_HINT` |
| `visit_1` | 448 | 14.4% | **30.5 m** | **387.2 m** | **16.3 m** | **100% (3/3)** | `VISIT_WITH_HINT` |
| `cross_account_street` | 88 | 2.8% | — | **150.0 m** | — | — | `VISIT_WITH_HINT` |
| `baseline_street` | 311 | 10.0% | **72.0 m** | **162.1 m** | **40.0 m** | **100% (1/1)** | `VISIT_WITH_HINT` |
| `landmark` | 541 | 17.4% | **163.7 m** | **1,430.4 m** | **197.0 m** | **100% (4/4)** | `VISIT_WITH_HINT` |
| `baseline_coarse` | 735 | 23.6% | **464.5 m** | **2,201.7 m** | **626.9 m** | **60% (3/5)** | `VERIFY_FIRST` |
| `unmapped_village` | 237 | 7.6% | — | **NaN** | — | — | `CANNOT_GEOCODE` |
| **Total / Overall** | **3,117** | **100.0%** | — | — | — | **86.7% (13/15)** | — |

### Action Breakdown in Final Output:
- **`VERIFY_FIRST`**: 1,276 addresses (40.9%) — telephony pre-verification required before field dispatch.
- **`VISIT_WITH_HINT`**: 1,055 addresses (33.8%) — dispatch agent with vernacular landmark cues and street bounds.
- **`DIRECT_VISIT`**: 549 addresses (17.6%) — high-confidence doorstep GPS navigation ($R_{90} \le 56.3\text{ m}$).
- **`CANNOT_GEOCODE`**: 237 addresses (7.6%) — explicit handling for out-of-service/unmapped village addresses.

---

## 🔬 Benchmark Comparison (Surveyed Ground Truth)

| Metric | Commercial Baseline | Our Self-Correcting Geocoder | Improvement |
| :--- | :---: | :---: | :---: |
| **`visits_agree` Median Error** | 376.4 m | **7.9 m (test) / 9.6 m (dev)** | **97.9% reduction** |
| **`visit_1` Median Error** | 376.4 m | **16.3 m (test) / 30.5 m (dev)** | **95.7% reduction** |
| **Doorstep Dwell (Locked Premises)** | 15.5 m | **6.2 m** | **60.0% reduction** |
| **Doorstep Dwell (Met Borrower)** | 33.5 m | **12.6 m** | **62.4% reduction** |
| **Doorstep Dwell (Cash Collected)** | 42.1 m | **8.5 m** | **79.8% reduction** |
| **Overall Surveyed P90 Error** | 839.0 m | **~350 m** | **58.3% reduction** |
| **Test Set Conformal Coverage** | — | **86.7%** (13 / 15 rows) | Finite-sample valid |
| **FA009 Fraudulent Visits Caught** | 162 (photo-only) | **229 (all fixed clusters + photos)** | **+67 fake visits purged** |
| **Genuine Agent False Positives** | — | **0 (0.0%)** | Zero collateral damage |

---

## 🚀 How to Run the Pipeline

### Prerequisites
- Python 3.10+
- Install dependencies:
  ```bash
  pip install pandas numpy rapidfuzz
  ```

### 1. Run the Complete Production Pipeline
Execute the full unified 8-step pipeline from start to finish:
```bash
python -m src.pipeline
```
This generates the authoritative 18-column deliverable `output/geocoder_output.csv` (3,117 rows) along with all intermediate audit tables.

### 2. Run the Automated Test Suite
Run the 7 regression safeguards and integrity tests:
```bash
python -m unittest discover tests -v
```
All 7 tests verify:
1. `test_canonical_18_column_schema`: Authoritative output adheres to 18 audit columns.
2. `test_out_address_completeness`: Output contains exactly 3,117 rows with 237 `OUT` addresses tagged `CANNOT_GEOCODE`.
3. `test_generic_fraud_detection`: Generic detector purges FA009's fake check-ins without hardcoded coordinates.
4. `test_genuine_agent_integrity`: Zero false positive cluster detections on genuine field agents.
5. `test_dwell_centroid_error_reduction`: Canonical dwell centroids reduce doorstep error on positive visits.
6. `test_negative_evidence_search_quality`: Search quality metric properly weights movement, coverage, and duration.
7. `test_evaluation_split_isolation`: The 15 held-out test accounts are strictly isolated from training/anchors.
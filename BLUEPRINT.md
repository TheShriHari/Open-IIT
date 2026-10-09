# Self-Correcting Address Geocoder — Master Technical Blueprint

Oct 8, 2026 · @SH

## 1. Project Overview & Problem Statement

This system converts unstructured, multilingual Indian residential addresses and noisy historical field-agent GPS logs into calibrated doorstep coordinates with honest uncertainty bounds — replacing a commercial geocoder whose median error is 363 m.

### The Operational Problem

Indian addresses in field collections are descriptive directions, not postal codes:

> *"6th Cross, 5th Main, near church, Kuvempu Layt, Kaveripura — 560102"*

Commercial geocoders drop pins at town or pincode centroids, producing errors of 350 m to 3 km. Field agents wander, waste ₹150–400 per trip, and mark recoverable accounts as `address_not_traceable`. Two failure modes cost money:

- **False untraceable:** Borrower is home; agent cannot find the house → lost recovery opportunity.
- **Wasted dispatch:** Agent travels to a wrong pin 2 km away → ₹300+ burned per visit.

### The Core Innovation

Field agents have already been to many of these addresses. Their GPS check-ins are sitting unused in visit logs. This system treats those check-ins as **noisy but real evidence** of where the house is, filters the noise, fuses the evidence, and quantifies how confident the resulting pin is.

Five things this system does that the baseline geocoder does not:

1. **Parses multilingual addresses** — Kannada script, Devanagari, English, Hinglish, Kanglish — into structured spatial tokens without a translation API.
2. **Filters visit noise** — removes drive-bys, fraud, and off-site check-ins before they corrupt coordinates.
3. **Fuses evidence** via an inverse-variance weighted median that is robust to outlier GPS pings.
4. **Falls back gracefully** through a four-tier hierarchy when visit data is absent.
5. **Outputs a calibrated R₉₀ radius** — a 90% confidence circle around the predicted pin — so field agents know exactly how precisely to trust the location.

### Self-Correction Loop

Every new field visit becomes additional calibration evidence:

```
Address → Prediction → Agent visits → New GPS / outcome / remarks
       → Evidence added → Better future prediction
```

The system improves over time without retraining a model — it accumulates real-world evidence.

### Dataset Scale (Current)

| File | Rows | Role |
| --- | --- | --- |
| `addresses.csv` | 3,118 | Master address records |
| `field_visits.csv` | 5,579 | Historical agent check-ins |
| `visit_gps_points.csv` | 160,407 | Breadcrumb GPS trail per visit |
| `baseline_geocodes.csv` | 2,881 | Commercial geocoder benchmark |
| `surveyed_addresses.csv` | 100 | Ground-truth answer key (eval only) |
| `landmarks_poi.csv` | 241 | POI database with (x, y) coordinates |
| `localities.csv` | 37 | Locality centroids per town |
| `towns.csv` | 3 | Town metadata |
| `splits.csv` | 2,401 | Train / validation / test labels |

## 2. Dataset Schema & Coordinate System

### Coordinate System

All spatial data is pre-projected into **flat Euclidean meters (x, y)**. This is not latitude/longitude — it is a local Cartesian plane where 1 unit = 1 metre. All distance calculations are:

```latex
\text{dist}(A, B) = \sqrt{(x_A - x_B)^2 + (y_A - y_B)^2}
```

No Haversine formula, no spherical trigonometry. This works because the towns covered are small enough that the Earth's curvature introduces negligible error at these scales (< 5 km across).

### CSV File Reference

#### `addresses.csv` — Master Address Records

| Column | Type | Description |
| --- | --- | --- |
| `address_id` | String | Unique address key (e.g. `AD000001`) |
| `account_id` | String | Borrower account key (e.g. `AC000001`) |
| `town_id` | String | `T1`, `T2`, `T3`, or `OUT` (out-of-coverage rural) |
| `address_text` | String | Raw unstructured address (multilingual) |

#### `field_visits.csv` — Historical Agent Check-ins

| Column | Type | Description |
| --- | --- | --- |
| `visit_id` | String | Unique visit key |
| `address_id` | String | FK → `addresses.csv` |
| `agent_id` | String | FK → `agents.csv` |
| `outcome` | String | Visit result (see outcome table below) |
| `checkin_x` | Float | Agent GPS x at check-in (metres) |
| `checkin_y` | Float | Agent GPS y at check-in (metres) |
| `gps_accuracy_m` | Float | Device-reported GPS accuracy (metres) |
| `photo_hash` | String | Perceptual hash of check-in photo |

**Outcome classification (empirically validated):**

| Outcome | Median Error to True Door | Decision |
| --- | --- | --- |
| `met_family` | 16 m | ✅ KEEP |
| `neighbour_says_shifted` | 20 m | ✅ KEEP |
| `locked_premises` | 20 m | ✅ KEEP |
| `met_borrower` | 34 m | ✅ KEEP |
| `address_not_traceable` | 1,603 m | ❌ DROP — agent was searching |

#### `baseline_geocodes.csv` — Commercial Geocoder Output

| Column | Type | Description |
| --- | --- | --- |
| `address_id` | String | FK → `addresses.csv` |
| `geocoder_x` | Float | Baseline pin x (metres) |
| `geocoder_y` | Float | Baseline pin y (metres) |
| `precision` | String | `rooftop`, `street`, `locality`, `pincode` |

**Precision hierarchy matters for the fallback tier logic:**

- `rooftop` / `street` → strong enough to keep as a fallback pin
- `locality` → too coarse; triggers landmark POI search
- `pincode` → very coarse; median error 1,376 m

#### `landmarks_poi.csv` — Points of Interest Database

| Column | Type | Description |
| --- | --- | --- |
| `town_id` | String | Town scope |
| `landmark_type` | String | Canonical category (e.g. `temple`, `school`) |
| `name` | String | Specific POI name (e.g. `Hanuman Temple`) |
| `x` | Float | POI centroid x (metres) |
| `y` | Float | POI centroid y (metres) |

241 POIs across 3 towns. Used both for address parsing (entity linking) and for landmark fallback geocoding.

#### `localities.csv` — Locality Reference

| Column | Type | Description |
| --- | --- | --- |
| `town_id` | String | Town scope |
| `locality_name` | String | Canonical locality name |
| `pincode` | String | Associated pincode |
| `centroid_x` | Float | Locality centroid x |
| `centroid_y` | Float | Locality centroid y |

37 locality records. Pincodes are used to pre-filter candidates before fuzzy matching.

#### `surveyed_addresses.csv` — Ground Truth (Evaluation Only)

| Column | Type | Description |
| --- | --- | --- |
| `address_id` | String | FK → `addresses.csv` |
| `surveyed_x` | Float | True doorstep x (metres) |
| `surveyed_y` | Float | True doorstep y (metres) |

100 rows. **Never used during prediction** — only for measuring error after the fact. The `splits.csv` file divides these 100 into `train` (for conformal calibration, 66 rows), `validation` (19 rows), and `test` (15 rows, held out until final evaluation).

### Data Flow Diagram

```
addresses.csv ─────────────────────────────────────────────────────► Phase 1 (parsing)
       │
       └─► field_visits.csv ────────────────────────────────────────► Phase 2 (GPS filter)
                   │
       baseline_geocodes.csv ──────────────────────────────────────► Phase 3 (fusion)
                   │
       landmarks_poi.csv + localities.csv ──────────────────────────► Phase 3 (fallback)
                   │
       surveyed_addresses.csv + splits.csv ─────────────────────────► Phase 4 (calibration)
                   │
                   ▼
       output/geocoder_output.csv ◄──────── final deliverable
```

## 3. Phase 1 — Address Parsing & Structuring

Phase 1 converts a raw multilingual address string into a structured set of atomic fields. It is entirely deterministic — no ML model is involved here, only compiled regex and string distance functions.

### 3.1 Pincode Extraction (Vectorized)

A 6-digit Indian pincode always starts with a non-zero digit:

```python
addr["pincode"] = addr["clean_text_lower"].str.extract(r"\b([1-9][0-9]{5})\b")[0]
```

This runs across the entire dataframe as a single Pandas vectorized operation — no loop.

### 3.2 Door / House Number Extraction

Two-stage extraction to avoid false positives on survey numbers, account IDs, or street ordinals:

**Stage A — Explicit keyword prefix (anywhere in string):**

```
Pattern: \b(?:no\.?|#|h\.?no\.?|house\s*(?:no\.?)?|flat\s*(?:no\.?)?|plot\s*(?:no\.?)?)\s*([a-z0-9/-]+)
```

Matches: `"H.No 45"`, `"Flat No. 12B"`, `"Plot 7/3"`, `"# 98"`. The capture group extracts only the number token, not the keyword.

**Stage B — Bare number anchored to the first 35 characters only:**

```
Pattern: ^\s*([a-z0-9]+[/-][a-z0-9]+|\d{1,4}[a-z]?)\b
```

Additional guards: length ≤ 5 characters, and value must not end in `st`, `nd`, `rd`, `th` (which would indicate a street ordinal like `"2nd"`).

**Why 35 characters?** Indian addresses almost always open with the house number. Restricting the anchor prevents matching pincode digits or account numbers appearing later in the string.

### 3.3 Street / Cross / Road Extraction

```
Pattern: \b(\d+(?:st|nd|rd|th)?\s*(?:cross|main|road|rd|x|gali|lane)|gali\s*(?:no\.?)?\s*\d+|block\s*[a-z0-9]+|[a-z]\s*block)\b
```

Matches: `"2nd Cross"`, `"5th Main"`, `"Gali No. 4"`, `"B Block"`. Deduplication via `dict.fromkeys()` preserves first-seen order.

### 3.4 Landmark Extraction — Two-Tier Hybrid

#### Tier 1: Multi-Script Regex Lexicon

13 canonical landmark categories with patterns covering English, Kannada script, Kannada transliteration, and Devanagari/Hindi:

| Category | Sample patterns |
| --- | --- |
| `temple` | `temple`, `mandir`, `gudi`, `ಗುಡಿ`, `ದೇವಸ್ಥಾನ`, `मंदिर`, `हनुमान` |
| `school` | `school`, `vidyalaya`, `shale`, `ಶಾಲೆ`, `स्कूल` |
| `bus_stop` | `bus stop`, `bus stand`, `ಬಸ್ ನಿಲ್ದಾಣ`, `बस स्टैंड` |
| `hospital_medical` | `hospital`, `clinic`, `aspatre`, `ಆಸ್ಪತ್ರೆ`, `अस्पताल` |
| `ration_shop` | `ration`, `pds`, `fair price`, `ರೇಷನ್`, `राशन` |
| `masjid` | `masjid`, `mosque`, `dargah`, `ಮಸೀದಿ`, `मस्जिद` |

All 13 patterns are pre-compiled at module load time (`re.compile(pat, re.IGNORECASE)`) — never compiled inside a loop.

Regex runs in O(P × L) where P = number of patterns and L = address string length. For 13 patterns and typical 100-character addresses this is effectively instantaneous.

#### Tier 2: RapidFuzz Token-Set POI Fallback

Applied **only to rows where Tier 1 found no landmark** — typically 40–60% of the dataset.

**Why `token_set_ratio` and not `JaroWinkler`?**

`JaroWinkler` is a character-level full-string comparison. Comparing a 2-gram span `"primary school"` against a full POI name `"Government Higher Primary School"` produces a low score despite a genuine match. `token_set_ratio` tokenizes both strings and computes the ratio on the intersection + remainder — correctly matching subsets:

```python
fuzz.token_set_ratio("primary school", "Government Higher Primary School") → 100
JaroWinkler.similarity("primary school", "Government Higher Primary School") → 0.71
```

Implementation uses `rapidfuzz.process.extractOne` against a pre-grouped town-level candidate list:

```python
poi_lookup_by_town = {
    tid: {"records": [...], "names": [name.lower() for ...]}
    for tid, grp in lm.groupby("town_id")
}
```

Threshold: **82.0**. Tuning note — if you see `hospital_medical` matching addresses that just say `"medical road"` (a street name), raise this to 85–88.

### 3.5 Spatial Relation Extraction & Disambiguation

Three canonical relations with multilingual patterns:

| Relation | Sample keywords |
| --- | --- |
| `NEAR` | `near`, `nr`, `pakka`, `hattira`, `ಹತ್ತಿರ`, `के पास` |
| `OPPOSITE` | `opp`, `opposite`, `eduru`, `ಎದುರು`, `के सामने` |
| `BEHIND` | `behind`, `hinde`, `ke peeche`, `ಹಿಂದೆ`, `के पीछे` |

**Disambiguation when multiple relations appear:** Instead of taking the first match (which depends on dict insertion order), the code computes the character-position distance between each relation match and the detected landmark match, and selects the relation that is **physically closest to the landmark token** in the string:

```python
lm_mid = (lm_span[0] + lm_span[1]) / 2.0
for regex, rel_label in COMPILED_RELATIONS:
    for match in regex.finditer(text):
        dist = abs(((match.start() + match.end()) / 2.0) - lm_mid)
        if dist < min_dist:
            min_dist, best_rel = dist, rel_label
```

This correctly handles: `"behind market, near temple"` → picks `NEAR` for the temple landmark.

### 3.6 Locality Resolution

Two-stage candidate pruning before fuzzy match:

1. **Pincode filter:** If a pincode was extracted, restrict candidates to `localities.csv` rows where `pincode` matches — reduces the candidate pool from all 37 localities to typically 3–8.
2. **Partial ratio match:** `fuzz.partial_ratio` from rapidfuzz correctly finds a short locality name (`"Jayanagar"`) embedded in a long address string. Threshold: **75.0**.

**Why `partial_ratio` and not `JaroWinkler` for locality?**

`JaroWinkler` is designed for full-string comparison of similar-length strings. Comparing `"jayanagar"` (9 chars) against a full address string `"h.no 45, near hanuman temple, jayanagar, bangalore 560041"` (57 chars) produces a meaningless score. `partial_ratio` slides the shorter string across the longer and returns the maximum window score — exactly the right metric for substring presence.

### 3.7 Output Schema

Phase 1 produces `output/structured_addresses.csv`:

| Column | What it holds |
| --- | --- |
| `door_no` | Extracted house/plot/flat number |
| `street_info` | Cross/Main/Road/Gali references |
| `landmark_type` | Canonical category (`temple`, `school`, etc.) |
| `spatial_relation` | `NEAR` / `OPPOSITE` / `BEHIND` (or empty) |
| `landmark_source` | `regex_lexicon` or `poi_token_set` |
| `landmark_match_score` | 100.0 for regex; fuzzy score (82–100) for Tier 2 |
| `locality_name` | Matched canonical locality name |
| `pincode` | 6-digit pincode (or NaN) |

It also produces `output/phase1_audit.csv` — every row where at least one of landmark, locality, or pincode is missing — for manual inspection.

### 3.8 Performance Notes

- Pincode, door, and street fields are vectorized over the full dataframe before any per-row function runs.
- Tier 1 landmark extraction uses `apply()` but over pre-compiled regex — this is fast.
- Tier 2 POI matching uses `apply()` but runs only on the missed-landmark subset (typically 40–60% of rows).
- Locality matching uses `apply()` on the full frame. At 100k+ rows, consider grouping by `(town_id, pincode)` and running `process.cdist` in batch.

## 4. Phase 2 — Visit Ingestion, GPS Filtering & Weighted Median

Phase 2 takes raw `field_visits.csv` records and produces a single reliable doorstep pin per address, with a tier label describing how much to trust it.

### 4.1 Visit Acceptance Filter

Only four outcomes are accepted as doorstep evidence (empirically validated on ground-truth surveyed data):

```python
ACCEPT_OUTCOMES = ["locked_premises", "met_borrower", "met_family", "neighbour_says_shifted"]
```

`address_not_traceable` is explicitly dropped. Empirical validation shows these visits have a median error of **1,603 m** from the true doorstep — the agent was actively searching, not standing at the house.

### 4.2 GPS Accuracy Filter

```python
v = v[v["gps_accuracy_m"] <= 30.0]
```

Cutoff: **30 metres**. Tightening to 15 m worsened the 90th-percentile error on the dev set (from 245 m to 274 m) because inverse-variance weighting already down-weights noisy readings — hard filtering at 15 m throws away valid but slightly imprecise fixes.

**No dwell-time filter is applied.** Analysis showed every accepted visit already had ≥ 60 seconds dwell time, and longer dwell times did not correlate with better accuracy (18.6 m error for 120–300 s vs. 29.2 m for > 300 s).

### 4.3 Photo Anti-Spoofing Filter

Field agents discovered they could photograph one location and reuse the same photo for multiple accounts. The `photo_hash` column stores a perceptual hash of the check-in photo.

The theoretical approach (referenced in the architecture doc) uses dHash Hamming distance:

```latex
D_H(h_1, h_2) = \sum_{k=1}^{64} (h_{1,k} \oplus h_{2,k})
```

In practice, the empirical finding was simpler: exact duplicate `photo_hash` values appeared across up to **31 distinct accounts** under one hash. The implemented filter:

```python
photo_counts = visits.groupby("photo_hash")["account_id"].nunique()
suspicious_photos = photo_counts[photo_counts >= 3].index
v = v[~v["photo_hash"].isin(suspicious_photos)]
```

Threshold: **≥ 3 distinct accounts** sharing a photo hash → all visits with that hash are dropped. A hash appearing for 2 accounts could be two different borrowers living in the same building — a genuine coincidence. At 3+, fraud is the overwhelmingly likely explanation.

### 4.4 Inverse-Variance Precision Weighting

Each accepted visit is assigned a weight inversely proportional to the squared GPS accuracy:

```latex
w_i = \frac{1}{\max(\text{gps\_accuracy\_m}_i,\ 1.0)^2}
```

The `max(..., 1.0)` floor prevents division-by-zero for theoretically perfect GPS readings (accuracy reported as 0 m by some devices).

**Intuition:** A visit logged with 5 m GPS accuracy gets weight 1/25 = 0.04. A visit logged with 30 m accuracy gets weight 1/900 ≈ 0.001. The first visit has 40× more influence on the final pin. This is the standard inverse-variance weighting used in precision-weighted averaging, adapted to spatial coordinates.

### 4.5 2D Weighted Median (L₁ Aggregation)

The doorstep pin is the **weighted median** of the accepted visit coordinates, solved independently per axis:

```latex
p_x^* = \arg\min_x \sum_{i=1}^{N} w_i |x_i - x|
```

```latex
p_y^* = \arg\min_y \sum_{i=1}^{N} w_i |y_i - y|
```

The solution satisfies the weighted median condition:

```latex
\sum_{i:\, x_i \le p_x^*} w_i \ge \frac{1}{2}\sum_{i=1}^N w_i \quad \text{and} \quad \sum_{i:\, x_i \ge p_x^*} w_i \ge \frac{1}{2}\sum_{i=1}^N w_i
```

Implementation:

```python
def weighted_median(vals, weights):
    val_arr, w_arr = np.asarray(vals, float), np.asarray(weights, float)
    order = np.argsort(val_arr)
    v_s, w_s = val_arr[order], w_arr[order]
    c = np.cumsum(w_s)
    return v_s[np.searchsorted(c, 0.5 * c[-1])]
```

**Why weighted median and not weighted mean?**

The weighted mean minimizes squared error (L₂ loss) and is strongly influenced by outliers. If one agent visited a tea stall 200 m from the house, the mean shifts toward that outlier. The weighted median minimizes L₁ loss and has a **breakdown point of 50%** — up to half the total weight can be outliers without moving the estimate away from the true cluster.

Numerical example:

| Visit | x (m) | w |
| --- | --- | --- |
| 1 (locked\_premises, 5m GPS) | 1400 | 0.040 |
| 2 (met\_borrower, 8m GPS) | 1405 | 0.016 |
| 3 (met\_family, 10m GPS) | 1398 | 0.010 |
| 4 (met\_borrower, 25m GPS, tea stall) | 1580 | 0.002 |

Weighted mean: ≈ 1404 m (pulled toward outlier) Weighted median: 1400 m (correctly anchored to the cluster)

### 4.6 Consensus Tier Classification

After computing the weighted median pin, classify the visit set:

```python
n_v = len(grp)
spread = np.max(np.hypot(grp["checkin_x"] - wx, grp["checkin_y"] - wy)) if n_v >= 2 else 0.0

tier = (
    "visit_1" if n_v == 1
    else "visits_agree" if spread <= 100.0
    else "visits_disagree"
)
```

| Tier | Condition | Empirical R₉₀ |
| --- | --- | --- |
| `visits_agree` | ≥ 2 visits, max spread ≤ 100 m | 48 m |
| `visit_1` | Exactly 1 accepted visit | 77 m |
| `visits_disagree` | ≥ 2 visits, max spread > 100 m | 388 m |

`visits_disagree` has a much larger uncertainty radius because contradictory visits suggest either a genuine address ambiguity (the address matches two buildings) or one fraudulent/off-site check-in that survived all filters. The landmark hint becomes especially important for these cases.

### 4.7 Output

Phase 2 produces `output/visit_derived_pins.csv`:

| Column | Description |
| --- | --- |
| `address_id` | FK to master addresses |
| `visit_px` | Weighted median x (metres) |
| `visit_py` | Weighted median y (metres) |
| `n_good_visits` | Count of accepted visits after all filters |
| `visit_tier` | `visit_1`, `visits_agree`, or `visits_disagree` |

## 5. Phase 3 — Spatial Fallback Hierarchy

Phase 3 selects the best available pin for each address by walking a four-tier evidence hierarchy. The key design principle: never discard good evidence just because better evidence is unavailable.

### 5.1 The Four Tiers

```
Tier 1: Visit pin (visits_agree / visit_1 / visits_disagree)
     ↓ if no accepted visits
Tier 2: Baseline street / rooftop pin
     ↓ if baseline precision is locality or pincode
Tier 3: Landmark POI pin (within 800 m of locality centroid)
     ↓ if no matching POI within 800 m
Tier 4: Coarse baseline pin (locality centroid)
```

### 5.2 Tier 1 — Visit Pin

If `visit_derived_pins.csv` has a row for this `address_id`, use it. The tier label (`visits_agree`, `visit_1`, `visits_disagree`) is carried forward directly from Phase 2.

### 5.3 Tier 2 — Baseline Street / Rooftop

If no visits exist and `baseline_geocodes.precision` is `street` or `rooftop`, use the commercial geocoder output directly.

Empirical justification: Street-precision baseline pins have a median error of only **109 m** on the dev set — much better than any landmark fallback (which starts at 358 m R90). Keeping good baseline pins avoids replacing a decent street-level geocode with a rougher POI guess.

### 5.4 Tier 3 — Landmark POI Geocoding

Triggered when: baseline precision is `locality` (or worse) AND a `landmark_type` was extracted in Phase 1.

**Step 1:** Determine the locality centroid (cx, cy). Use the `(town_id, pincode)` lookup first; fall back to the town centroid if pincode was not extracted.

**Step 2:** Filter `landmarks_poi.csv` to POIs in the same town matching the extracted `landmark_type`.

**Step 3:** Find the nearest matching POI:

```latex
\text{best POI} = \arg\min_{j} \sqrt{(x_j - c_x)^2 + (y_j - c_y)^2}
```

**Step 4:** Accept if distance ≤ **800 m**. Reject and fall to Tier 4 otherwise.

**Why 800 m?** This is the maximum plausible radius for a borrower to describe a landmark as their reference point. A temple 1.5 km away is not "near the temple". Empirically, 800 m eliminated spurious cross-locality matches in the three covered towns.

**Important labeling:** The landmark tier pin is placed at the POI itself, not at the house. The house is *near* the POI by some unknown offset. This is why the landmark hint explicitly reads `"pin is a guess near this landmark"` rather than the confident `"Near X (N m from pin)"` used for visit/street tiers.

### 5.5 Tier 4 — Coarse Baseline

When no higher tier applies, retain the commercial geocoder's locality-level pin as-is. This is deliberately the lowest-confidence tier — R90 = **3,710 m** — and always routes to `VERIFY_FIRST`.

### 5.6 Landmark Navigational Hints

For every final pin (regardless of tier), the system finds the nearest POI within **200 m** of the pin and generates a navigational hint for the field agent:

```python
d_p = np.hypot(t_pois["x"].values - px, t_pois["y"].values - py)
b_idx = np.argmin(d_p)
if d_p[b_idx] <= HINT_MAX_DIST:  # 200 m
    p_name = t_pois.iloc[b_idx]["name"]
    hint = (
        f"pin is a guess near this landmark: {p_name}"
        if tier == "landmark"
        else f"Near {p_name} (about {d_p[b_idx]:.0f} m from pin)"
    )
```

**The 200 m cap (down from 500 m in earlier designs):** A hint that says "Near Government School" when the school is 450 m away misleads more agents than it helps. At 200 m, the agent can see the landmark from the predicted doorstep.

### 5.7 Output

Phase 3 produces `output/master_pins.csv`:

| Column | Description |
| --- | --- |
| `address_id` | Master key |
| `account_id` | Borrower key |
| `town_id` | Town scope |
| `px` | Final pin x (metres) |
| `py` | Final pin y (metres) |
| `tier` | Evidence tier selected |
| `n_good_visits` | Visit count (0 if visit tier not used) |
| `landmark_hint` | Navigational cue string (empty if no POI within 200 m) |

## 6. Phase 4 — Conformal Confidence Radius Calibration

Phase 4 answers the question: *given a predicted pin, how large a circle must a field agent search to find the actual house with 90% probability?* The answer is the R90 radius, computed using split conformal prediction.

### 6.1 Why Conformal Prediction?

Naive approaches like saying "visit pins are accurate to ±50 m" are guesses. Conformal prediction provides a **formal finite-sample guarantee**: if the calibration set and production data are exchangeable (drawn from the same distribution), the R90 radius contains the true doorstep with at least 90% empirical probability.

No distributional assumptions are required — no Gaussian error model, no parametric family. The guarantee comes from order statistics on real prediction errors.

### 6.2 The Non-Conformity Score

For each address in the calibration set with known ground-truth coordinates (surveyed\_x, surveyed\_y), compute the 2D Euclidean error of the predicted pin:

```latex
E_i = \sqrt{(p_{x,i} - y_{x,i})^2 + (p_{y,i} - y_{y,i})^2}
```

This is a scalar residual in metres — how far off the prediction was for that address.

### 6.3 Stratification by Evidence Tier

Calibration errors are computed **separately per tier** because the error distributions are fundamentally different:

- `visits_agree`: tight cluster, small errors
- `visit_1`: one data point, moderate errors
- `baseline_street`: commercial geocoder quality, moderate errors
- `landmark`: POI-level approximation, larger errors
- `visits_disagree`: contradictory evidence, larger errors
- `baseline_coarse`: locality centroid, very large errors

Mixing tiers before computing quantiles would produce a single R90 that is simultaneously too large for high-confidence cases and too small for low-confidence cases.

### 6.4 The Conformal Quantile Formula

For a tier with n calibration residuals E(1) ≤ E(2) ≤ … ≤ E(n), sorted ascending, the index for 90% coverage (alpha = 0.10) is:

```latex
k = \left\lceil (n + 1)(1 - \alpha) \right\rceil = \left\lceil 0.90 \times (n + 1) \right\rceil
```

The R90 radius is the k-th order statistic:

```latex
R_{90} = E_{(k)}
```

Implementation:

```python
def conformal_quantile(errors, coverage=0.90):
    n = len(errors)
    idx = math.ceil((n + 1) * coverage)
    return np.sort(errors)[idx - 1] if idx <= n else np.nan
```

The `(n+1)` correction (not just n) is the finite-sample adjustment from the conformal prediction literature. It ensures the coverage guarantee holds even for small n — without it, the coverage is slightly below 90% by construction.

**Minimum sample requirement:** Tiers with fewer than 9 calibration samples cannot compute a valid finite quantile (ceil(10 × 0.9) = 9, so you need at least 9 calibration points). For sparse tiers, fall back to the empirically measured benchmark values:

| Tier | Empirical R90 | n in dev set |
| --- | --- | --- |
| `visits_agree` | 48 m | — |
| `visit_1` | 77 m | — |
| `baseline_street` | 162 m | — |
| `landmark` | 358 m | — |
| `visits_disagree` | 388 m | — |
| `baseline_coarse` | 3,710 m | — |

### 6.5 Formal Guarantee Statement

Under exchangeability between the calibration set (dev split, 85 rows) and production data:

```latex
\mathbb{P}\left(E_{\text{new}} \le R_{90}\right) \ge 1 - \alpha = 0.90
```

**Honest limitation:** The test set (15 rows) achieved empirical coverage of 79% (11/14 finite-tier rows). This is below the 90% target. With only 15 test rows, a 79% coverage result is statistically consistent with the true coverage being 90% — the confidence interval on 11/14 overlaps \[0.90\]. But it is not proof of 90% coverage. More surveyed ground-truth addresses are the single highest-value data collection investment for this project.

### 6.6 How R90 Is Used

R90 is not just a statistic — it is the operational routing signal. It determines what the field agent does before leaving the office:

- R90 ≤ 100 m → `DIRECT_VISIT` (agent navigates to pin; house is within GPS walking error)
- R90 ≤ 500 m → `VISIT_WITH_HINT` (agent navigates to pin and uses landmark cue to search nearby)
- R90 > 500 m → `VERIFY_FIRST` (hold visit; phone-verify cross-street before dispatch, saving ₹300+ per wasted trip)

## 7. Phase 5 — Operational Dispatch & Final Output

Phase 5 assembles the final deliverable and converts the R90 radius into an actionable field instruction.

### 7.1 Action Routing Logic

```python
conditions = [final_export["R90_meters"] <= 100.0, final_export["R90_meters"] <= 500.0]
final_export["action"] = np.select(conditions, ["DIRECT_VISIT", "VISIT_WITH_HINT"], default="VERIFY_FIRST")
```

| Action | R90 Range | What the agent does |
| --- | --- | --- |
| `DIRECT_VISIT` | ≤ 100 m | Navigate GPS to pin; house is within walking distance of the pin |
| `VISIT_WITH_HINT` | 101–500 m | Navigate to pin, use landmark hint to scan within the circle |
| `VERIFY_FIRST` | > 500 m | Phone-verify cross-street or house number before dispatch |

**Cost justification for `VERIFY_FIRST`:** A field visit costs ₹150–400 in travel. A tele-calling verification costs < ₹10. For an R90 of 3,710 m (baseline\_coarse tier), the agent would need to search a \~11.5 km² area — guaranteeing a wasted trip without prior verification.

### 7.2 Final Output Schema (`geocoder_output.csv`)

| Column | Type | Example |
| --- | --- | --- |
| `address_id` | String | `AD000001` |
| `account_id` | String | `AC000001` |
| `town_id` | String | `T1` |
| `px` | Float | `1399.2` |
| `py` | Float | `-150.1` |
| `tier` | String | `visits_agree` |
| `R90_meters` | Float | `48.0` |
| `action` | String | `DIRECT_VISIT` |
| `landmark_hint` | String | `Near Ganesh Temple (about 42 m from pin)` |

### 7.3 Self-Correction Loop

This is the long-term value of the architecture. Every new field visit is a new data point:

```
New visit outcome → accepted if outcome in ACCEPT_OUTCOMES
                 → filtered if gps_accuracy > 30 m
                 → filtered if photo_hash appears on >= 3 accounts
                 → added to visit pool for that address_id
                 → weighted median recomputed
                 → R90 radius shrinks (more evidence = more confidence)
                 → action may upgrade: VERIFY_FIRST → VISIT_WITH_HINT → DIRECT_VISIT
```

No model retraining is required. The system accumulates evidence and re-runs the deterministic fusion pipeline. As more addresses receive real field visits, the proportion of `baseline_coarse` tier addresses shrinks automatically.

### 7.4 Population Routing Breakdown (Projected)

Based on the tier distribution across 2,880 geocoded addresses:

| Tier | Approx. % of addresses | Expected Action |
| --- | --- | --- |
| `visits_agree` | \~15% | `DIRECT_VISIT` |
| `visit_1` | \~20% | `DIRECT_VISIT` |
| `baseline_street` | \~5% | `VISIT_WITH_HINT` |
| `landmark` | \~15% | `VISIT_WITH_HINT` |
| `visits_disagree` | \~10% | `VISIT_WITH_HINT` |
| `baseline_coarse` | \~35% | `VERIFY_FIRST` |

The 35% in `baseline_coarse` represents the primary improvement opportunity — each successful verified visit for one of these addresses permanently upgrades it.

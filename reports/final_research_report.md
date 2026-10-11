# Final Research Report: Urban Taxi Mobility Pattern Discovery

**Project**: Urban Taxi Mobility Pattern Discovery using Microsoft T-Drive GPS Trajectories  
**Domain**: Spatial Data Mining, Sequential Pattern Mining, Urban Transportation Analytics  
**Date**: October 2026  
**Repository**: [https://github.com/PetaSivaNandhanReddy/urban-taxi-mobility-pattern-discovery](https://github.com/PetaSivaNandhanReddy/urban-taxi-mobility-pattern-discovery)  

---

## 1. Abstract

Discovering recurring spatial-temporal mobility patterns from raw vehicle GPS traces is a foundational problem in intelligent transportation analytics. Raw GPS observations are continuous, irregular, and noisy. This research project presents a reproducible end-to-end analytical pipeline applied to the Microsoft Research T-Drive trajectory dataset (Beijing, 2008). 

The pipeline implements:
1. Multi-stage quality-controlled trajectory cleaning and calendar-day trajectory reconstruction;
2. Equi-rectangular spatial discretization (500 m grid) and Haversine density-based spatial clustering (DBSCAN);
3. Sequential pattern mining using bounded PrefixSpan on calendar-date-delimited taxi-day sequences;
4. Multi-criteria composite pattern ranking and spatial hotspot corridor integration;
5. Methodologically rigorous model evaluation based on a strictly uniform **ordered cell-pair evaluation unit** `(origin, destination)`;
6. An objective synthetic benchmark with known planted patterns evaluated against **realized support** in the finite sample;
7. Empirical parameter sensitivity (support thresholds 1%, 2%, 5% and grid sizes 300 m, 500 m, 800 m);
8. Deterministic nested sample-size scaling across 99, 249, and 497 active taxis.

On the baseline sample of 99 active taxis (630 taxi-day sequences, 116,913 GPS observations), PrefixSpan (min_support = 5%, max_length = 4) discovered 22 frequent sequential patterns. Evaluated against an empirical reference set of 35 observed recurring transitions, PrefixSpan achieved a Pair Precision of **0.5500**, Pair Recall of **0.3143**, and Pair F1 of **0.4000**, outperforming direct adjacent-transition mining (Recall: 0.0571, F1: 0.1081) and order-blind co-occurrence (F1: 0.1762). On the objective synthetic benchmark, PrefixSpan achieved **1.0000 Recall** across all realized frequent ground-truth pairs.

---

## 2. Problem Statement & Research Objectives

Urban transportation networks require empirical data to discover recurring travel routes, identify activity hotspots, and detect rare or anomalous mobility transitions. Raw GPS traces present several analytical hurdles:
- **Spatial Sparsity**: Taxis traverse continuous 2D space, requiring discretization into a countable vocabulary for sequence mining.
- **Midnight Boundary Artifice**: Grouping sequences across multiple days can create artificial transitions between a taxi's last evening ping and its first morning ping.
- **Evaluation Circularity**: Urban mobility pattern discovery is unsupervised; there is no independent ground-truth database of "true" routes. Past evaluations in the literature often suffered from unit-mixing bugs (comparing whole patterns against transition pairs) or trivial True-Negative inflation (reporting >99% classification accuracy on sparse networks).

### Research Objectives:
1. Construct an end-to-end reproducible pipeline from raw text logs to pattern discovery without discarding legitimate physical movement.
2. Formulate a mathematically defensible evaluation protocol using a single uniform unit: the **ordered cell pair** `(origin_cell, destination_cell)`.
3. Construct a transparent, configurable empirical reference benchmark while clearly acknowledging its limitations.
4. Establish an objective synthetic benchmark to test PrefixSpan's ability to recover planted sequences with controlled noise and calculate realized support from the actual finite sample.
5. Quantify parameter sensitivity across support thresholds and spatial grid resolutions, and benchmark fleet-size scaling.

---

## 3. Dataset Description & Preprocessing Pipeline

### 3.1 Microsoft Research T-Drive Dataset
- **Format**: Individual per-taxi text logs (`1.txt` ... `10357.txt`), comma-separated, without header rows: `[taxi_id, timestamp, longitude, latitude]`.
- **Temporal Coverage**: One week, 2008-02-02 to 2008-02-08 (Beijing Local Time).
- **Geographic Bounding Box**: Beijing urban expanse (Latitude: 39.4°N to 41.6°N, Longitude: 115.4°E to 117.5°E).

### 3.2 Preprocessing Pipeline & Data Provenance
Data cleaning is tracked stage-by-stage using a transparent `CleaningReport` ([`src/preprocessing.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/src/preprocessing.py)):
1. **Taxi ID Verification**: Stripped whitespace, checked non-empty strings.
2. **Timestamp Parsing**: Dual-pass datetime conversion using format `%Y-%m-%d %H:%M:%S` with fallback parsing.
3. **Physical Coordinate Validity**: Removed invalid coordinates outside physical Earth limits (lat $\in [-90, 90]$, lon $\in [-180, 180]$) and null-fix sentinels `(0, 0)`.
4. **Out-of-Region Flagging**: Beijing bounding box checked. Points outside are flagged, not silently deleted (332 points flagged in baseline sample).
5. **Exact Duplicate Removal**: Dropped duplicate `(taxi_id, timestamp, lat, lon)` records.

#### Baseline Data Cleaning Summary Table
Saved to [`outputs/tables/cleaning_summary.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/tables/cleaning_summary.csv):

| Cleaning Stage | Records Remaining | Records Removed |
| :--- | :---: | :---: |
| Original records loaded | 129,545 | 0 |
| After taxi_id cleaning | 129,545 | 0 |
| After timestamp cleaning | 129,545 | 0 |
| After coordinate validity cleaning | 128,800 | 745 |
| After out-of-region flagging (flagged, not dropped) | 128,800 | 0 |
| After duplicate record removal | **116,913** | 11,887 |
| **Final Clean Records** | **116,913** | — |

---

## 4. Methodology: Discretization, Clustering & Sequence Mining

### 4.1 Equi-Rectangular Spatial Discretization
Continuous GPS coordinates are discretized into uniform square grid cells ([`src/spatial.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/src/spatial.py)). At Beijing's mean latitude ($\phi_0 \approx 39.9^\circ\text{N}$):
$$\Delta\text{lat} = \frac{\text{cell\_size\_m}}{111,320}, \quad \Delta\text{lon} = \frac{\text{cell\_size\_m}}{111,320 \times \cos(\phi_0)}$$
- Grid resolution: **500 m** (default).
- Discretized occupied vocabulary: **5,413 unique cells** across Beijing.

### 4.2 Haversine DBSCAN Hotspot Clustering
Density-Based Spatial Clustering of Applications with Noise (DBSCAN) identifies geographic activity hotspots ([`src/clustering.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/src/clustering.py)):
- Great-circle distance using Haversine metric (preventing Euclidean lat/lon distortion away from the equator).
- Parameters selected via k-distance elbow curve: $\epsilon = 300\text{ m}$, $\text{min\_samples} = 15$.
- Result: **237 spatial clusters** (hotspots) identified; cluster ID `-1` explicitly isolated as background noise.

### 4.3 Calendar-Day Sequence Construction
Trajectories are grouped by `taxi_id::date` ([`src/sequential.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/src/sequential.py)):
- Dividing by calendar date prevents artificial cross-day jumps between consecutive shifts.
- Consecutive duplicate cells (dwelling at traffic lights or passenger queues) are compressed: $[A, A, A, B, B] \rightarrow [A, B]$.
- Sequences of length $< 2$ are excluded.
- Total sequences built: **630 taxi-day sequences**.

### 4.4 PrefixSpan Sequential Pattern Mining
PrefixSpan mines frequent ordered subsequences:
- Minimum support: **5.0%** of taxi-day sequences.
- Bounded maximum pattern length: **4 cells** (preventing combinatorial explosion).
- Conversion to integer count uses exact ceiling:
$$\text{min\_support\_count} = \max\Big(2, \; \lceil\text{min\_support} \times n_{\text{sequences}}\rceil\Big) = \lceil 0.05 \times 630 \rceil = 32\text{ taxi-days}$$
- Discovered patterns: **22 frequent sequential patterns**.

---

## 5. Evaluation Redesign: Unified Ordered-Pair Framework

### 5.1 Common Evaluation Unit: Ordered Cell Pairs
Past evaluation code mixed whole-pattern false positives with pair-level true positives. We enforce a strictly uniform evaluation unit across all models:
$$\text{Unit} = (c_{\text{origin}}, \; c_{\text{destination}})$$
1. **Directional Separation**: $(A, B) \neq (B, A)$.
2. **Subsequence Pair Extraction (PrefixSpan)**: A sequence $(c_1, c_2, \dots, c_k)$ implies all distinct ordered pairs $(c_i, c_j)$ for $i < j$.
3. **Contiguous Pair Extraction (Direct Baseline)**: Mined contiguous sequences imply strictly adjacent pairs $(c_i, c_{i+1})$.
4. **Deduplication**: Pairs are accumulated into a Python `set`. A pair appearing in multiple patterns is credited at most once.

### 5.2 Empirical Reference Benchmark Construction
Because unannotated GPS data has no external ground truth, an empirical reference set is constructed from physically observed adjacent transitions ([`src/pattern_evaluation.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/src/pattern_evaluation.py)):
- A transition qualifies if: $\text{count} \ge 15$ occurrences across $\ge 2$ distinct taxis.
- Reference size: **35 reference transitions**.

> [!IMPORTANT]
> **Academic Caveats**:
> 1. This reference represents observed historical mobility agreement, NOT external ground truth.
> 2. Because it is built from adjacent transitions, direct contiguous mining structurally achieves 100% precision on recovered transitions.

### 5.3 Corrected Metric Formulas & Zero-Denominator Policy
$$\text{TP} = |\text{Predicted Pairs} \cap \text{Reference Pairs}|$$
$$\text{FP} = |\text{Predicted Pairs} \setminus \text{Reference Pairs}|$$
$$\text{FN} = |\text{Reference Pairs} \setminus \text{Predicted Pairs}|$$
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Zero-Denominator Policy**: When $\text{TP} + \text{FP} = 0$ (no predictions) or $\text{TP} + \text{FN} = 0$ (empty reference), precision/recall/F1 return a documented default of `0.0` or `None`.

### 5.4 Bounded-Universe Accuracy Audit
In a finite active universe of $N = 49$ cells in play, total pairs $= 49 \times 48 = 2,352$:
- $\text{TP} = 11, \; \text{FP} = 9, \; \text{FN} = 24, \; \text{TN} = 2,308$.
- Raw Accuracy $= \frac{11 + 2308}{2352} = 98.60\%$.
- **TN Dominance Ratio**: $98.13\%$.
Because True Negatives vastly dominate spatial trajectory graphs, reporting ~98.6% accuracy without context is misleading. Precision, Recall, and F1 over active transitions are the proper primary metrics.

---

## 6. Real-Data Baseline Comparisons

Results saved to [`outputs/results/evaluation_method_comparison.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/evaluation_method_comparison.csv):

| Mining Method | Evaluation Unit | Predicted Pairs | Reference Pairs | TP | FP | FN | Precision | Recall | F1 Score | Runtime (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PrefixSpan (Gapped Sequential)** | Distinct Ordered Pairs | 20 | 35 | **11** | 9 | 24 | **0.5500** | **0.3143** | **0.4000** | 0.523 |
| **Direct Adjacent Transition** | Distinct Ordered Pairs | 2 | 35 | 2 | 0 | 33 | 1.0000 | 0.0571 | 0.1081 | 0.055 |
| **Unordered Co-occurrence (Symmetric)** | Distinct Ordered Pairs | 158 | 35 | 17 | 141 | 18 | 0.1076 | 0.4857 | 0.1762 | 2.420 |

### Order-Agnostic Co-occurrence Baseline
Saved to [`outputs/results/evaluation_cooccurrence_order_agnostic.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/evaluation_cooccurrence_order_agnostic.csv):

| Mining Method | Evaluation Unit | Predicted Pairs | Reference Pairs | TP | FP | FN | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unordered Co-occurrence** | Undirected Cell Pairs $\{u, v\}$ | 158 | 30 | 21 | 137 | 9 | 0.1329 | 0.7000 | 0.2234 |

### Method Comparison Discussion:
1. **PrefixSpan vs Direct Transitions**: Direct contiguous mining finds only 2 frequent transitions at 5% support, missing gapped multi-hop sequential corridors (Recall: 5.71%). PrefixSpan recovers 11 reference transitions (Recall: 31.43%) with an F1 score of **0.4000** (nearly 4x higher than direct mining).
2. **Order-Blind Co-occurrence**: Co-occurrence achieves higher recall (48.57% directed, 70.0% undirected) but produces massive false positives (141 false directed pairs), yielding poor precision (10.76%). Directional sequential mining is essential for filtering out coincidental spatial co-occurrence.

---

## 7. Objective Synthetic Benchmark Results

The synthetic benchmark ([`src/synthetic_benchmark.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/src/synthetic_benchmark.py)) generates 350 controlled taxi-day sequences (50 taxis $\times$ 7 days, seed=42) containing 5 deliberately planted patterns with controlled noise and gapped insertions.

### 7.1 Realized Support Table
Saved to [`outputs/results/synthetic_benchmark_planted_table.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/synthetic_benchmark_planted_table.csv):

| Planted Pattern | Length | Target Prob | Realized Matches | Realized Support | Min Count Req (10%) | Realized Frequent? | PrefixSpan Recovered? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `(CORRIDOR_A1, CORRIDOR_A2, CORRIDOR_A3)` | 3 | 0.35 | 120 | 34.29% | 35 | **True** | **True** |
| `(CORRIDOR_B1, CORRIDOR_B2)` | 2 | 0.25 | 84 | 24.00% | 35 | **True** | **True** |
| `(CORRIDOR_A2, SHARED_S1, CORRIDOR_B2)` | 3 | 0.20 | 74 | 21.14% | 35 | **True** | **True** |
| `(CORRIDOR_C1, C2, C3, C4)` | 4 | 0.15 | 47 | 13.43% | 35 | **True** | **True** |
| `(RARE_D1, RARE_D2)` | 2 | 0.03 | 16 | 4.57% | 35 | **False** | **False (Excluded)** |

### 7.2 Synthetic Pair-Level and Pattern-Level Recovery
Saved to [`outputs/results/synthetic_benchmark_pair_level.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/synthetic_benchmark_pair_level.csv) and [`outputs/results/synthetic_benchmark_pattern_level.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/synthetic_benchmark_pattern_level.csv):

| Evaluation Level | Method | Predicted | Reference | TP | FP | FN | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pair-Level** | **PrefixSpan (Gapped)** | 58 | 13 | **13** | 45 | **0** | 0.2241 | **1.0000** | **0.3662** |
| Pair-Level | Direct Adjacent | 8 | 13 | 8 | 0 | 5 | 1.0000 | 0.6154 | 0.7619 |
| Pair-Level | Co-occurrence (Symmetric) | 558 | 13 | 13 | 545 | 0 | 0.0233 | 1.0000 | 0.0455 |
| **Pattern-Level** | **PrefixSpan (Gapped)** | 68 | 4 | **4** | 64 | **0** | 0.0588 | **1.0000** | **0.1111** |
| Pattern-Level | Direct Adjacent | 10 | 4 | 3 | 7 | 1 | 0.3000 | 0.7500 | 0.4286 |

### Synthetic Benchmark Key Takeaways:
1. **100% Recovery of Frequent Planted Patterns**: PrefixSpan recovered 100% of realized frequent planted patterns and ordered pairs ($\text{Recall} = 1.0000$).
2. **Proper Exclusion of Rare Pattern**: `(RARE_D1, RARE_D2)` had a realized support of 4.57% (16 sequences), below the 10% threshold. It was correctly excluded from ground truth, confirming the realized-support logic.
3. **Handling of Overlapping & Gapped Patterns**: PrefixSpan successfully separated the shared-cell pattern `(A2, SHARED_S1, B2)` despite shared cells with Corridors A and B and intermittent background noise.

---

## 8. Parameter Sensitivity & Stability Analysis

### 8.1 Support Threshold Sensitivity (1%, 2%, 5%)
Saved to [`outputs/results/evaluation_support_thresholds.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/evaluation_support_thresholds.csv):

| Min Support | Min Count Req | Discovered Patterns | Runtime (s) | Avg Length | Ref Precision | Ref Recall | Ref F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0%** | 7 | 39,785 | 13.228 | 2.27 | 0.0011 | **0.9429** | 0.0022 |
| **2.0%** | 13 | 3,426 | 1.886 | 2.05 | 0.0094 | 0.8571 | 0.0186 |
| **5.0%** | 32 | **22** | **0.212** | 2.00 | **0.5500** | 0.3143 | **0.4000** |

### 8.2 Jaccard Pattern Stability
Saved to [`outputs/results/evaluation_pattern_stability.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/evaluation_pattern_stability.csv):

| Set A | Set B | $n_A$ | $n_B$ | Intersection $|A \cap B|$ | Union $|A \cup B|$ | Jaccard Similarity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1.0% | 2.0% | 39,785 | 3,426 | 3,426 | 39,785 | **0.0861** |
| 2.0% | 5.0% | 3,426 | 22 | 22 | 3,426 | **0.0064** |
| 1.0% | 5.0% | 39,785 | 22 | 22 | 39,785 | **0.0006** |

**Discussion**:
- Discovered pattern sets demonstrate strict monotonic subset inclusion: $\mathcal{P}_{5\%} \subset \mathcal{P}_{2\%} \subset \mathcal{P}_{1\%}$.
- Low Jaccard similarity reflects the exponential expansion of candidate patterns at lower thresholds, not algorithm instability. At 5% support, PrefixSpan achieves high precision (55%) on robust core transitions; at 1%, it maximizes recall (94.29%) at the cost of high candidate volume.

### 8.3 Spatial Grid Resolution Sensitivity (300 m, 500 m, 800 m)
Saved to [`outputs/results/evaluation_grid_sizes.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/evaluation_grid_sizes.csv):

| Grid Cell Size | Occupied Cells | Avg Pts / Cell | Sequences | Min Count Req (5%) | Patterns at 5% | Total Runtime (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **300 m** | 8,925 | 13.10 | 627 | 32 | 1 | 0.413 |
| **500 m** | 5,413 | 21.60 | 630 | 32 | **22** | 0.578 |
| **800 m** | 3,322 | 35.19 | 625 | 32 | **550** | 1.307 |

**Discussion**:
- A 300 m grid fragments trajectories across 8,925 cells; only 1 pattern meets the 5% support threshold.
- An 800 m grid aggregates wider geographic areas into 3,322 cells, producing 550 frequent patterns.
- Direct Jaccard comparison across different grid sizes is geographically meaningless because cell identifiers represent entirely different spatial bounding boxes.

---

## 9. Nested T-Drive Sample-Size Scaling (99 -> 250 -> 500 Taxis)

To verify pipeline scalability and evaluate pattern persistence, deterministic nested samples were drawn from the 10,357 raw T-Drive logs (seed=42) ([`src/experiment_runner.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/src/experiment_runner.py)).

### 9.1 Sample Expansion Comparison Table
Saved to [`outputs/results/sample_expansion_comparison.csv`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/sample_expansion_comparison.csv):

| Sample Tier | Requested Files | Active Taxis Processed | Cleaned GPS Points | Taxi-Day Sequences | Min Support Count (5%) | Discovered Patterns | Reference Transitions | Pair F1 Score | Total Pipeline Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 100 | **99** | 116,913 | 630 | 32 | **22** | 35 | **0.4000** | 2.47 |
| **Medium** | 250 | **249** | 371,399 | 1,592 | 80 | **46** | 111 | 0.0632 | 8.39 |
| **Large** | 500 | **497** | 833,316 | 3,185 | 160 | **43** | 204 | 0.0205 | 23.96 |

### 9.2 Pattern Persistence Analysis
Saved to [`outputs/results/sample_expansion_persistence.json`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/outputs/results/sample_expansion_persistence.json):
- In a 99-taxi fleet (630 sequences), 5% support requires 32 taxi-days.
- In a 497-taxi fleet (3,185 sequences), 5% support requires **160 taxi-days**.
- Requiring 160 distinct taxi-days across an expanded fleet represents a substantially stricter absolute threshold. Consequently, baseline patterns that recurred across 32-40 taxi-days fall below 5% support in the expanded sample. Expanding the sample tests robustness to additional fleet data in the same period; it demonstrates that relative percentage support naturally becomes more stringent as fleet size grows.

---

## 10. Mobility Dashboard Overview

The interactive Streamlit dashboard ([`dashboard/app.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/dashboard/app.py)) provides dual operating modes:

1. **Administrator Mode**:
   - **Overview**: High-level KPIs, active taxis, PrefixSpan pair-level precision/recall/F1 metrics.
   - **Beijing Mobility Map**: Folium interactive map showing 237 DBSCAN hotspot clusters and GPS heatmap density.
   - **Temporal Intelligence**: Hourly, weekday/weekend, and time-segment mobility distributions.
   - **Sequential Patterns**: Top PrefixSpan patterns with composite ranking scores.
   - **Mobility Corridors**: Displays zero qualifying corridors gracefully (`most_recurring_corridor = N/A`).
   - **Taxi Allocation**: Exploratory heuristic allocation distributing vehicles across hotspots by historical point density.
   - **Rare Mobility**: Identifies transitions occurring on $\le 2$ taxi-days across the sample.
   - **Methodology & Evaluation**: Comprehensive 5-tab academic evaluation interface displaying real-data comparisons, synthetic benchmark recovery, sensitivity tables, and sample scaling.
   - **Export**: One-click download of all analytical CSVs.

2. **Customer Mode**:
   - Activity search by day type, time of day, and specific hotspot lookup.
   - Early departure guidance based on historical traffic activity.

3. **Validation**: Verified with 4 automated Streamlit `AppTest` integration tests ([`tests/test_dashboard.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/tests/test_dashboard.py)).

---

## 11. Automated Verification & Test Results

A test suite of 19 automated tests was developed and executed ([`tests/test_evaluation.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/tests/test_evaluation.py), [`tests/test_dashboard.py`](file:///c:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/tests/test_dashboard.py)):

```
test_different_methods_identical_units .................... PASS
test_duplicate_pairs_across_multiple_patterns ............. PASS
test_gapped_pairs_in_prefixspan ........................... PASS
test_no_predictions ....................................... PASS
test_no_reference_pairs ................................... PASS
test_partial_recovery ..................................... PASS
test_perfect_recovery ..................................... PASS
test_repeated_cell_patterns ............................... PASS
test_reversed_ordered_pairs ............................... PASS
test_zero_denominator_handling ............................ PASS
test_below_threshold_pattern_not_ground_truth ............. PASS
test_generator_reproducibility ............................ PASS
test_no_duplicate_counting_in_synthetic_evaluation ........ PASS
test_planted_frequent_pattern_recovery .................... PASS
test_realized_support_calculation ......................... PASS
test_admin_mode_overview_render ........................... PASS
test_customer_mode_render ................................. PASS
test_methodology_and_evaluation_section ................... PASS
test_mobility_corridors_zero_state ........................ PASS

Ran 19 tests in 6.55s — ALL 19 TESTS PASSED (100% SUCCESS)
```

---

## 12. Methodological Limitations & Academic Defense Statements

For academic defense and reporting integrity, the following limitations are explicitly declared:
1. **Empirical Reference Set**: Real-world evaluation uses observed adjacent transitions meeting frequency rules. It is an agreement metric, not external truth.
2. **Structural Bias Toward Direct Mining**: Because reference transitions are adjacent pairs, direct contiguous mining structurally achieves 100% precision on recovered transitions.
3. **Synthetic Benchmark Scope**: Synthetic benchmark metrics evaluate recovery of the known synthetic construction, not real-world predictive accuracy.
4. **GPS Observations vs Passenger Demand**: GPS pings describe historical taxi vehicle movements. They do not measure unmet passenger travel demand or road congestion.
5. **Temporal & Geographic Bounds**: Findings are bounded to Beijing in February 2008 and do not claim generalization to other cities or seasons.
6. **Zero-Corridor Validity**: Finding zero corridors connecting distinct hotspots is an expected, scientifically valid descriptive result under current support thresholds.
7. **Heuristic Ranking Weights**: The 6-factor composite score uses equal weighting ($1/6$ each) as a descriptive heuristic.

---

## 13. Commands to Reproduce All Results

```bash
# 1. Run Automated Unit and Dashboard Integration Tests (19 tests)
.\.venv\Scripts\python.exe -m unittest discover -s tests

# 2. Run Unified Experiment Runner (Phases 2, 3, 4, 5)
.\.venv\Scripts\python.exe src\experiment_runner.py

# 3. Execute Final Notebook 05 End-to-End
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\05_model_evaluation.ipynb

# 4. Launch Streamlit Mobility Dashboard
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```

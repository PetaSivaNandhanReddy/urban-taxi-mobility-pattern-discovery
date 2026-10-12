# Urban Taxi Mobility Pattern Discovery for Smart Transportation Planning

**Final-Year M.Tech Business Analytics Research Project**  
Sequential and spatial data mining on urban taxi GPS trajectories, using the Microsoft Research **T-Drive** dataset (Beijing, 2008) as a generic benchmark.

> **Status: All Phases Complete (Phases 1–5).**  
> - **Phase 1**: Ingestion, schema detection, multi-stage cleaning, trajectory reconstruction, and exploratory spatial data analysis.
> - **Phase 2**: Equi-rectangular spatial grid discretization (500m), Haversine DBSCAN hotspot clustering, and parameter tuning.
> - **Phase 3**: PrefixSpan sequential pattern mining on taxi-day sequences, contiguous n-gram baseline, temporal segmentation, and Streamlit dashboard.
> - **Phase 4**: Multi-criteria composite pattern ranking, spatial hotspot-to-hotspot corridor integration, occurrence-based rare mobility analysis, and transportation planning insights.
> - **Phase 5**: Rigorous model evaluation redesign (pair-level TP/FP/FN, precision/recall/F1, objective synthetic planted benchmark, parameter sensitivity, and temporal validation).

---

## 1. Research Objectives & Problem Statement

Urban transportation systems require data-driven intelligence to understand recurring traffic corridors, demand hotspots, and mobility patterns. Raw GPS traces are continuous, noisy, and unorganized. 

This project develops an end-to-end analytical pipeline to:
1. **Clean and reconstruct trajectories** without aggressively discarding valid physical movements.
2. **Discretize continuous coordinates** into uniform, deterministic spatial cells (500m) to form a countable vocabulary for sequence mining.
3. **Discover frequent sequential movement patterns** using bounded PrefixSpan on calendar-date-delimited taxi-day sequences.
4. **Identify spatial taxi hotspots** using density-based spatial clustering (DBSCAN) with great-circle Haversine distance.
5. **Rank discovered patterns** using a 6-factor composite evidence score (support, taxi coverage, temporal consistency, spatial density, pattern length, and date recurrence).
6. **Evaluate pattern recovery objectively** using both pair-level empirical metrics and a controlled synthetic benchmark with planted ground-truth sequences.
7. **Deliver an interactive analytics dashboard** for transportation planners and fleet dispatchers.

---

## 2. Dataset: Microsoft Research T-Drive

- **Source**: Microsoft Research T-Drive trajectory dataset sample (Beijing, 2008).
- **Format**: Per-taxi text files (`1.txt`, `2.txt`, ...) containing comma-separated `(taxi_id, timestamp, longitude, latitude)`.
- **Benchmark Sample Size**:
  - Raw / cleaned observations: **116,913** GPS points
  - Active taxis: **99** taxis
  - Calendar taxi-day sequences: **630** sequences (7 calendar days, 2008-02-02 to 2008-02-08)
  - Occupied grid cells (500m): **5,413** cells
  - Non-noise DBSCAN clusters: **237** clusters ($\text{eps}=300\text{m}, \text{min\_samples}=15$)
  - Frequent PrefixSpan patterns ($\ge 5\%$ support): **22** patterns

> **Domain Limitation Notice**: Taxi GPS point volume represents observed historical vehicle mobility in a sampled fleet. It is a proxy for urban movement and should not be equated directly with real-time passenger demand or overall traffic congestion.

---

## 3. Project Directory Structure

```
urban-taxi-mobility/
├── .gitignore                               # Excludes venvs, raw logs, caches, large maps
├── config.py                                # Root configuration
├── requirements.txt                         # Pin-compatible project dependencies
├── README.md                                # Authoritative project documentation
├── README(p1).md                            # Historical Phase 1 documentation archive
├── dashboard/
│   ├── app.py                               # Interactive Streamlit Mobility Dashboard
│   └── requirements.txt                     # Dashboard dependencies
├── data/
│   ├── raw/                                 # Raw T-Drive .txt files (taxi_log_2008_by_id/)
│   ├── processed/                           # Cleaned, clustered, and labeled trajectory data
│   └── samples/                             # Sample extracts
├── notebooks/
│   ├── 01_data_exploration.ipynb            # Phase 1: Ingestion, schema validation, cleaning, EDA
│   ├── 02_spatial_analysis.ipynb            # Phase 2: Spatial grid & Haversine DBSCAN clustering
│   ├── 03_sequential_pattern_mining.ipynb   # Phase 3: PrefixSpan mining on taxi-day sequences
│   ├── 04_pattern_ranking_integration.ipynb # Phase 4: Composite ranking, corridors, anomalies, KPIs
│   └── 05_model_evaluation.ipynb            # Phase 5: Redesigned pair-level evaluation & baselines
├── outputs/
│   ├── figures/                             # High-resolution charts and visual summaries
│   ├── maps/                                # Folium interactive HTML maps
│   ├── results/                             # Mined patterns, research metrics, corridor tables
│   └── tables/                              # 18 summary and KPI tables
├── src/
│   ├── anomaly.py                           # Rare / unusual transition identification
│   ├── clustering.py                        # Haversine DBSCAN & k-distance elbow analysis
│   ├── config.py                            # Central configuration module
│   ├── data_loader.py                       # Recursive raw data loader with schema detection & metadata
│   ├── departure_optimizer.py               # Historical departure-time window evaluation & correlation reporting
│   ├── evaluation.py                        # Grid, DBSCAN & support threshold sensitivity
│   ├── experiment_runner.py                 # Automated reproducible pipeline runner
│   ├── integration.py                       # Sequential patterns + DBSCAN hotspot corridor mapping
│   ├── pattern_evaluation.py                # Pair-level & pattern-level evaluation and metrics
│   ├── pattern_ranking.py                   # Enriched sequence building & composite scoring
│   ├── planning.py                          # Transportation planning insights & KPI generation
│   ├── preprocessing.py                     # Multi-stage cleaning pipeline with CleaningReport
│   ├── recommendation.py                    # Evidence-based alternative-route comparison & recommendation
│   ├── route_candidates.py                  # Historical candidate route discovery & travel-time distributions
│   ├── sequential.py                        # Sequence building & PrefixSpan / n-gram mining
│   ├── spatial.py                           # Equi-rectangular spatial grid, coordinate decoding & path continuity
│   ├── synthetic_benchmark.py               # Objective synthetic benchmark with planted ground truth
│   ├── temporal.py                          # Hourly, segment, date & normalized day-type activity profiling
│   └── trajectory.py                        # Trajectory reconstruction & dwell-based trip/session segmentation
└── tests/
    ├── test_dashboard.py                    # Streamlit dashboard AppTest integration tests
    ├── test_evaluation.py                   # Automated test suite for evaluation methodology
    ├── test_ingestion_and_cleaning.py       # Ingestion, cleaning, coordinate validity & sampling tests
    ├── test_route_and_departure.py          # Route candidates, recommendation & departure optimization tests
    ├── test_sequential.py                   # Calendar sequences, duplicate compression & PrefixSpan tests
    └── test_spatial_temporal.py             # Spatial grid, decoding, continuity & normalized temporal tests
```

---

## 4. Installation & Setup

### 4.1 Prerequisites
- Python 3.10 to 3.12 installed on Windows, macOS, or Linux.
- VS Code or preferred IDE.

### 4.2 Setup Commands (Windows PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/PetaSivaNandhanReddy/urban-taxi-mobility-pattern-discovery.git
cd urban-taxi-mobility-pattern-discovery

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Execution Order: Running the Research Pipeline

To reproduce the complete research findings from raw data to evaluation:

1. **Phase 1: Exploratory Data Analysis & Cleaning**
   ```powershell
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```
   *Outputs cleaned sample and trajectory diagnostics.*

2. **Phase 2: Spatial Discretization & DBSCAN Hotspots**
   ```powershell
   jupyter notebook notebooks/02_spatial_analysis.ipynb
   ```
   *Outputs `clustered_sample.csv`, `spatial_grid_summary.csv`, and Folium hotspot maps.*

3. **Phase 3: Sequential Pattern Mining (PrefixSpan)**
   ```powershell
   jupyter notebook notebooks/03_sequential_pattern_mining.ipynb
   ```
   *Mines 22 frequent patterns ($\ge 5\%$ support) on 630 taxi-day sequences.*

4. **Phase 4: Pattern Ranking & Corridor Integration**
   ```powershell
   jupyter notebook notebooks/04_pattern_ranking_integration.ipynb
   ```
   *Ranks patterns by 6-factor composite score, maps hotspot corridors, identifies rare transitions, and generates transportation planning insights.*

5. **Phase 5: Redesigned Model Evaluation & Baselines**
   ```powershell
   jupyter notebook notebooks/05_model_evaluation.ipynb
   ```
   *Runs pair-level evaluation against empirical reference, executes synthetic planted benchmark, and conducts parameter sensitivity.*

---

## 6. Launching the Interactive Dashboard

Launch the Streamlit visualization dashboard:

```powershell
streamlit run dashboard/app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

**Features**:
- **Overview KPIs**: Total taxis, GPS observations, active clusters, noise ratio.
- **Spatial Hotspots Tab**: Interactive Folium map with DBSCAN cluster centroids and point density.
- **Temporal Patterns Tab**: Activity distribution across morning, afternoon, evening, and night segments.
- **Sequential Patterns Tab**: PrefixSpan ranked patterns with composite scores and support metrics.
- **Mobility Corridors Tab**: Transparent corridor reporting (explaining why 0 multi-cluster corridors formed at 5% support).
- **Taxi Allocation (Exploratory Scenario)**: Proportional vehicle allocation across hotspots.
- **Rare Mobility Tab**: Occurrence-based unusual movement transition inspector.

---

## 7. Model Evaluation Methodology & Benchmark Results

### 7.1 Methodological Redesign: Consistent Pair-Level Units
To prevent mixing incompatible prediction units (e.g. counting reference pairs as TP but whole patterns as FP), all empirical comparisons are evaluated at the **Ordered-Transition-Pair Level**:
- **Unit**: Distinct ordered cell pair $(o, d)$.
- **$TP$**: Number of reference pairs recovered by discovered patterns.
- **$FP$**: Number of discovered pairs that are not in the reference set.
- **$FN$**: Number of reference pairs not recovered.
- **$\text{Precision} = \frac{TP}{TP + FP}$**, **$\text{Recall} = \frac{TP}{TP + FN}$**, **$F1 = \frac{2 \cdot P \cdot R}{P + R}$**.

### 7.2 Real-Data Empirical Reference Comparison (T-Drive Sample)

Evaluated strictly at the **ordered cell-pair level** $(o, d)$:

| Method | Predicted Pairs | Reference Pairs | TP | FP | FN | Pair Precision | Pair Recall | Pair F1 | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PrefixSpan (Gapped Sequential)** | 20 | 35 | **11** | 9 | 24 | **0.5500** | **0.3143** | **0.4000** | 0.523s |
| **Direct Contiguous Transition** | 2 | 35 | 2 | 0 | 33 | **1.0000** | 0.0571 | 0.1081 | 0.055s |
| **Unordered Co-occurrence (Symmetric)** | 158 | 35 | 17 | 141 | 18 | 0.1076 | 0.4857 | 0.1762 | 2.420s |

*Key Findings*:
- **PrefixSpan achieves the highest balanced $F1$ score ($0.4000$)** by discovering ordered transitions with moderate precision ($0.5500$) and recall ($0.3143$).
- **Direct Contiguous mining** achieves 100% precision on recovered pairs but severe under-recall ($0.0571$) because it misses transitions when intervening cells exist.
- **Unordered Co-occurrence** suffers from 141 false positive directed pairs because it cannot establish travel direction (F1: $0.1762$). Under an order-agnostic comparison (undirected pairs), it achieves $0.1329$ precision and $0.7000$ recall (F1: $0.2234$).

### 7.3 Objective Synthetic Planted Benchmark Results

Tested against 5 deliberately planted patterns with controlled noise in 350 generated taxi-days, evaluated against **realized support** in the finite sample:

| Evaluation Level | Method | Predicted | Reference | TP | FP | FN | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pair-Level** | **PrefixSpan (Gapped)** | 58 | 13 | **13** | 45 | **0** | 0.2241 | **1.0000** | **0.3662** |
| Pair-Level | Direct Adjacent | 8 | 13 | 8 | 0 | 5 | 1.0000 | 0.6154 | 0.7619 |
| Pair-Level | Co-occurrence (Symmetric) | 558 | 13 | 13 | 545 | 0 | 0.0233 | 1.0000 | 0.0455 |
| **Pattern-Level** | **PrefixSpan (Gapped)** | 68 | 4 | **4** | 64 | **0** | 0.0588 | **1.0000** | **0.1111** |
| Pattern-Level | Direct Adjacent | 10 | 4 | 3 | 7 | 1 | 0.3000 | 0.7500 | 0.4286 |

*Key Synthetic Insights*:
- **PrefixSpan achieves 100% recall** on planted corridors even when noise is injected between corridor cells.
- **Direct Contiguous transitions miss 38.5% of planted transitions** ($Recall = 0.6154$) whenever noise intervenes.
- Rare pattern `(RARE_D1, RARE_D2)` with 4.57% realized support was correctly excluded from required ground truth under the 10% threshold.

### 7.4 Nested T-Drive Sample-Size Scaling (99 → 250 → 500 Taxis)

| Sample Tier | Active Taxis | Clean GPS Points | Taxi-Days | Min Support Count (5%) | Discovered Patterns | Reference Transitions | Pair F1 | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 99 | 116,913 | 630 | 32 | 22 | 35 | **0.4000** | 2.47s |
| **Medium** | 249 | 371,399 | 1,592 | 80 | 46 | 111 | 0.0632 | 8.39s |
| **Large** | 497 | 833,316 | 3,185 | 160 | 43 | 204 | 0.0205 | 23.96s |

*Key Scaling Takeaway*: A constant relative support threshold (5%) increases in absolute stringency as the fleet grows (requiring 160 distinct taxi-days for 497 taxis vs 32 for 99 taxis).

### 7.5 Historical Route Candidate Discovery & Departure-Time Analytics

Phase 5 introduces an evidence-based routing layer that segments trajectories into discrete trips/sessions based on physical dwell periods ($\ge 300\text{s}$) and signal gaps ($\ge 1800\text{s}$), discovering candidate paths without fabricating passenger status:
- **Trajectory Segmentation**: 40,827 discrete trip segments identified in the 99-taxi baseline.
- **Chronological Train/Holdout Split**: Discovery on Feb 2-6, 2008 (89,391 points) vs Holdout Evaluation on Feb 7-8, 2008 (27,522 points), strictly preventing future observation leakage.
- **Route Recommendation Coverage**: Across 20 popular OD cell pairs evaluated, **9 pairs (45.0% coverage)** possessed multiple supported continuous paths enabling comparative recommendation. The remaining 11 pairs exhibited a single observed path or fell below the minimum evidence threshold ($\ge 3$ trips).
- **Chronological Holdout Validation**: Evaluating recommended route durations against actual observed journeys on unseen future days (Feb 7-8) yielded a Mean Absolute Error of **44.4 seconds** (Median Absolute Error: **0.0 seconds**) across 21 test journeys.
- **Departure-Time Window Evaluation**: Evaluated travel-time variations across $\pm 5, 10, 15$-minute windows. Windows with insufficient sample sizes ($< 2$ trips) return an explicit `INSUFFICIENT_DATA` flag.
- **Methodological Guardrails**: Differences reflect observational historical associations in the February 2008 sample, explicitly distinguished from causal improvements or live traffic navigation.

---

## 8. Running Automated Unit & Dashboard Tests

Run the complete 45-test automated test suite across all 6 test modules:

```powershell
python -m unittest discover -s tests
```

Expected output:
```
Ran 45 tests in 5.1s — ALL 45 TESTS PASSED (100% SUCCESS)
```

Test modules covered:
1. `tests/test_ingestion_and_cleaning.py`: Malformed records, duplicates, invalid coordinates, timestamp sorting, deterministic sampling.
2. `tests/test_spatial_temporal.py`: 500m grid discretization, centroid coordinate decoding, path continuity, normalized temporal activity.
3. `tests/test_sequential.py`: Sequence building, duplicate compression, PrefixSpan gapped mining, contiguous n-grams, traceability.
4. `tests/test_route_and_departure.py`: Route discovery, recommendation logic, continuity checks, departure windows, insufficient-evidence guards.
5. `tests/test_evaluation.py`: Pair-level evaluation, confusion counts, synthetic benchmark recovery, realized support.
6. `tests/test_dashboard.py`: Streamlit AppTest integration tests across Administrator and Customer modes.

To run the complete automated research experiment pipeline end-to-end:
```powershell
python src/experiment_runner.py
```


To launch the Streamlit Mobility Intelligence Dashboard:
```powershell
streamlit run dashboard/app.py
```

---

## 9. Academic Presentation Summary

### Five Strongest Defensible Findings
1. **Taxi-Day Sequence Delimitation**: Grouping trajectories by `(taxi_id, calendar_date)` prevents artificial boundary transitions between consecutive operating shifts.
2. **Gapped Sequential Mining Value**: PrefixSpan successfully recovers recurring movement patterns with intervening noise where contiguous n-gram methods fail.
3. **Evidence-Based Spatial Tuning**: Haversine distance DBSCAN at $\text{eps}=300\text{m}$ effectively isolates 237 urban activity hotspots without planar Euclidean angular distortion.
4. **Transparent Zero-Corridor Reporting**: Explaining that 0 multi-cluster corridors formed at 5% support because 500m adjacent frequent moves occur within dense clusters, rather than between separated distant hotspots.
5. **Multi-Criteria Composite Ranking**: Balancing support with taxi breadth, temporal concentration, and date recurrence provides richer prioritization than raw frequency alone.

### Three Main Limitations
1. **Temporal Horizon**: Single-week dataset sample limits observation of seasonal and long-term infrastructure shifts.
2. **Fleet Sampling Bias**: Taxi traces represent commercial passenger transport behavior, not general private or public transit mobility.
3. **Empirical Reference Bias**: Real-world evaluation reference is adjacency-based, structurally rewarding contiguous baselines over gapped sequential patterns.

---

## 10. Authors & License

- **Author**: Peta Siva Nandhan Reddy
- **Degree**: Final-Year M.Tech in Business Analytics
- **Dataset**: Microsoft Research T-Drive Trajectory Data Sample (Beijing, 2008)
- **License**: Academic Research Use Only

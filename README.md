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
│   ├── data_loader.py                       # Recursive raw data loader with schema detection
│   ├── evaluation.py                        # Grid, DBSCAN & support threshold sensitivity
│   ├── integration.py                       # Sequential patterns + DBSCAN hotspot corridor mapping
│   ├── pattern_evaluation.py                # Pair-level & pattern-level evaluation and metrics
│   ├── pattern_ranking.py                   # Enriched sequence building & composite scoring
│   ├── planning.py                          # Transportation planning insights & KPI generation
│   ├── preprocessing.py                     # Multi-stage cleaning pipeline with CleaningReport
│   ├── sequential.py                        # Sequence building & PrefixSpan / n-gram mining
│   ├── spatial.py                           # Equi-rectangular spatial grid & compression
│   ├── synthetic_benchmark.py               # Objective synthetic benchmark with planted ground truth
│   ├── temporal.py                          # Hourly & time-segment activity profiling
│   └── trajectory.py                        # Trajectory reconstruction & speed jump diagnostics
└── tests/
    └── test_evaluation.py                   # Automated test suite for evaluation methodology
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

| Method | Discovered Pairs | Reference Pairs | TP | FP | FN | Pair Precision | Pair Recall | Pair F1 | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PrefixSpan (Gapped Sequential)** | 22 | 35 | **11** | 11 | 24 | **0.5000** | **0.3143** | **0.3860** | 0.638s |
| **Direct Contiguous Transition** | 2 | 35 | 2 | 0 | 33 | **1.0000** | 0.0571 | 0.1081 | 0.317s |
| **Unordered Co-occurrence** | 158 | 35 | 25 | 133 | 10 | 0.1582 | **0.7143** | 0.2591 | 3.350s |

*Key Findings*:
- PrefixSpan achieves the highest balanced $F1$ score ($0.3860$) by discovering ordered transitions with moderate precision and recall.
- Direct Adjacent mining achieves perfect precision ($1.0000$) but severe under-recall ($0.0571$) because it misses transitions when intervening cells exist.
- Unordered Co-occurrence achieves high recall ($0.7143$) but poor precision ($0.1582$) due to 133 false positive pairs caused by lack of directional order.

### 7.3 Objective Synthetic Planted Benchmark Results

In a controlled environment with 350 synthetic taxi-days and planted frequent corridors with inserted noise and gaps:

| Method | Evaluation Level | TP | FP | FN | Precision | Recall | F1 Score | Runtime (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PrefixSpan** | Pair-Level (Ordered Transitions) | **10** | 10 | **0** | 0.5000 | **1.0000** | **0.6667** | 0.027s |
| **Direct Contiguous** | Pair-Level (Ordered Transitions) | 6 | **0** | 4 | **1.0000** | 0.6000 | 0.7500 | 0.015s |
| **Unordered Co-occurrence** | Pair-Level (Ordered Transitions) | 10 | 522 | 0 | 0.0188 | 1.0000 | 0.0369 | 0.013s |

*Key Synthetic Insights*:
- **PrefixSpan achieves 100% recall** on planted corridors even when noise is injected between corridor cells.
- **Direct Contiguous transitions miss 40% of planted transitions** ($Recall = 0.60$) whenever non-systematic noise intervenes.

---

## 8. Running Automated Unit Tests

Run the complete test suite:

```powershell
python tests/test_evaluation.py
python src/synthetic_benchmark.py
```

Expected output:
```
Running automated test suite for evaluation methodology...
  [PASS] test_multiple_reference_recovery
  [PASS] test_no_reference_recovery_and_zero_tp
  [PASS] test_repeated_cells_in_pattern
  [PASS] test_duplicate_prediction_handling
  [PASS] test_empty_predictions
  [PASS] test_empty_reference
  [PASS] test_synthetic_planted_benchmark_execution

ALL 7 AUTOMATED EVALUATION TESTS PASSED SUCCESSFULLY!
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

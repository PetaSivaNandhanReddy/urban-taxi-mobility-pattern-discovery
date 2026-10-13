# Scientific Software & Methodological Audit: Urban Taxi Mobility Pattern Discovery

**Repository:** `https://github.com/PetaSivaNandhanReddy/urban-taxi-mobility-pattern-discovery`
**Working Branch:** `feature/taxi-mobility-analytics`
**Latest Audited Commit:** `99910b12d2f497e407ef9965c7e03a659efa8926`
**Auditor Role:** Research Software Auditor, Experimental Reproducibility Reviewer & Scientific Asset Preparation Assistant
**Target Paper:** 8–10 Page IEEE Conference Research Paper (*Urban Taxi Mobility Pattern Discovery Using Sequential and Spatial Data Mining*)
**Audit Date:** October 13, 2026

---

## Executive Summary

This audit provides an exhaustive software, mathematical, and empirical verification of the **Urban Taxi Mobility Pattern Discovery** codebase. The repository processes the Microsoft Research T-Drive trajectory dataset (Beijing, China; February 2–8, 2008) to uncover recurring spatiotemporal transit corridors, analyze spatial density hotspots, and evaluate unsupervised sequential mining algorithms against empirical and synthetic benchmarks.

All 51 automated unit and integration tests currently pass (`51 passed, 0 failed` in `pytest`). The full raw T-Drive fleet archive (10,357 individual taxi log files) is verified locally on disk, and reproducible pipeline runs have been executed across nested sample fleet sizes (99, 249, and 497 active taxis).

Crucially, this audit resolves previous documentation ambiguities, enforces strict mathematical definitions across all evaluation routines, and establishes full traceability from raw logs to IEEE-ready figures, tables, and manifests.

---

## 1. Implementation Inventory & Functional Scope

The repository comprises 21 Python source modules, a Streamlit interactive dashboard, 5 comprehensive test suites, and 4 Jupyter analytical notebooks. A detailed method-by-method audit is tabulated in [`implementation_inventory.csv`](../audit/implementation_inventory.csv).

### Core Pipeline Components
1. **Data Ingestion (`src/data_loader.py`):**
   - Implements chunked streaming and taxi-by-taxi file parsing.
   - Automatically detects available raw logs in `data/raw/taxi_log_2008_by_id/`.
   - Verified on 100 sample files (129,545 raw fixes) and validated up to 500 files (907,479 fixes).

2. **Data Preprocessing & Validation (`src/preprocessing.py`, `src/trajectory.py`):**
   - Implements multi-stage validation: integer taxi identifiers, ISO timestamp parsing, coordinate boundary pruning, out-of-region bounding box tagging, duplicate timestamp suppression, and kinematic speed anomaly detection.
   - Preserves non-destructive filtering where appropriate.

3. **Spatial Discretization & Clustering (`src/spatial.py`, `src/clustering.py`):**
   - Projects geographic coordinates into equirectangular planar grid cells (default resolution $R = 500\text{ m}$).
   - Computes pairwise Haversine distance matrices in radians ($R_{\text{earth}} = 6,371,008.8\text{ m}$).
   - Executes density-based spatial clustering (DBSCAN) with calibrated parameters $\epsilon = 300\text{ m}$ and $\text{min\_samples} = 15$.

4. **Sequential Trajectory Mining (`src/sequential.py`, `src/temporal.py`):**
   - Partitions trajectories at calendar-day midnight boundaries (`taxi_id::date`), producing 630 daily driver sequences.
   - Implements run-length compression to collapse immediate dwell fixes in identical cells ($A \to A \to B \implies A \to B$).
   - Provides two distinct mining engines:
     - **Direct Contiguous $n$-gram Mining:** Extracts strictly adjacent 1-hop physical transitions ($n=2$).
     - **PrefixSpan Subsequence Mining:** Recursively mines frequent gapped subsequences up to depth $L_{\max} = 4$.

5. **Multi-Factor Pattern Ranking (`src/pattern_ranking.py`):**
   - Formulates a composite mobility persistence score:
     $$S_{\text{composite}} = 0.4 \cdot S_{\text{norm\_support}} + 0.3 \cdot S_{\text{taxi\_spread}} + 0.3 \cdot S_{\text{day\_spread}}$$
   - Filters out high-frequency bursts confined to single drivers or isolated days.

6. **Unified Evaluation & Benchmarking (`src/pattern_evaluation.py`, `src/optimization_study.py`, `src/synthetic_benchmark.py`):**
   - Projects all pattern formats into a common evaluation unit: deduplicated directed transition pairs $(u \to v)$ with $u \neq v$.
   - Computes exact True Positives ($TP$), False Positives ($FP$), False Negatives ($FN$), Precision ($P$), Recall ($R$), and F1-score ($F1$).
   - Enforces chronological train/validation/test isolation.
   - Evaluates planted ground-truth recovery on controlled synthetic benchmarks.

---

## 2. Dataset & Preprocessing Audit

### 2.1 Dataset Provenance & Spatial Bounds
- **Source:** Microsoft Research T-Drive trajectory dataset.
- **Coverage Period:** February 2, 2008, 00:00:00 to February 8, 2008, 23:59:59 (7 consecutive days).
- **Timezone:** Local Beijing Time (UTC+8).
- **Geographic Bounding Box:** $[39.40^\circ\text{ N}, 41.60^\circ\text{ N}]$ latitude, $[115.70^\circ\text{ E}, 117.40^\circ\text{ E}]$ longitude.
- **Fleet Availability:** 10,357 raw vehicle logs present in local storage.

### 2.2 Preprocessing Filtering Stages (100 Sample Taxis)
Table I in [`table1_dataset_preprocessing.csv`](../tables/table1_dataset_preprocessing.csv) establishes the exact record filtering attrition:

| Preprocessing Stage | Records Remaining | Records Removed | Purge Ratio | Active Taxis | Operational Rule |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw Logs Ingestion** | 129,545 | 0 | 0.00% | 100 | Ingested from raw CSV text logs |
| **Taxi ID & Timestamp Check** | 129,545 | 0 | 0.00% | 100 | Format verified, chronologically sorted |
| **Coordinate Validity Purging** | 128,800 | 745 | 0.58% | 99 | Purged null `(0.0, 0.0)` fixes and out-of-range lat/lon |
| **Geographic Boundary Flagging** | 128,800 | 0 | 0.00% | 99 | 332 fixes outside Beijing bbox flagged (`in_region=False`); **retained** |
| **Duplicate Timestamp Purging** | 116,913 | 11,887 | 9.18% | 99 | Purged concurrent identical timestamps per taxi |
| **Final Analytical Trajectory Base** | **116,913** | **12,632 (total)** | **9.75%** | **99** | Partitioned into 630 calendar-day sequences |

#### Important Dataset Findings & Gotchas:
1. **The 99 vs. 100 Taxi Count:** Requesting 100 taxi files yields exactly 99 active taxis in the clean trajectory dataset. Investigation revealed that Taxi ID `4375` contained exclusively 311 null GPS coordinates `(0.0, 0.0)`. During coordinate validation, all 311 records were purged, correctly reducing the vehicle count to 99.
2. **Non-Destructive Bounding Box Filtering:** Rather than discarding GPS fixes outside Beijing's municipal boundary, the pipeline marks them with boolean attribute `in_region=False`. In the 100-taxi baseline, 332 records fell outside the bounding box and were safely retained without data corruption.
3. **Temporal Sampling Granularity:** Sampling intervals vary from 30 seconds to several minutes, with median interval $t_{\text{median}} = 300\text{ seconds}$ (5 minutes). Consecutive same-cell dwell points occur during driver wait states or traffic congestion; collapsing these consecutive duplicates via run-length compression is essential to avoid self-loop noise ($A \to A$).

---

## 3. Spatial & Sequential Algorithm Audit

### 3.1 Spatial Grid Discretization
- **Planar Coordinate Projection:** Equirectangular projection maps $(\text{lat}, \text{lon})$ to row/column grid indices.
- **Resolution Sensitivity:**
  - $300\text{ m}$: Yields 8,925 active cells; average 13.1 points/cell.
  - $500\text{ m}$ (Selected Production): Yields 5,413 active cells; average 21.6 points/cell. Provides balanced aggregation without spatial over-fragmentation.
  - $800\text{ m}$: Yields 3,322 active cells; average 35.2 points/cell. Leads to excessive agglomeration of distinct arterial corridors.

### 3.2 Haversine DBSCAN Hotspot Clustering
- **Distance Metric:** Exact Haversine metric using radians:
  $$d_{\text{rad}}(\mathbf{x}_i, \mathbf{x}_j) = 2 \arcsin \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos \phi_i \cos \phi_j \sin^2\left(\frac{\Delta \lambda}{2}\right)}$$
  $$\epsilon_{\text{rad}} = \frac{300.0\text{ m}}{6,371,008.8\text{ m}} \approx 4.7088 \times 10^{-5}\text{ radians}$$
- **Parameter Calibration:** Sensitivity analysis across $\epsilon \in [150, 800]\text{ m}$ and $\text{min\_samples} \in [5, 25]$ (Table IV) shows that $\epsilon = 300\text{ m}, \text{min\_samples} = 15$ forms an optimal elbow point, isolating 237 core hotspot clusters while separating diffuse transit noise.

### 3.3 Sequential Mining Mechanics
- **Sequence Demarcation:** Continuous multi-day GPS streams are partitioned at midnight into daily driver trajectories (`taxi_id::date`), generating 630 sequences across 99 taxis over 7 days.
- **PrefixSpan Subsequence Mining:**
  - Recursively mines frequent subsequences in prefix-projected databases.
  - At default $s_{\min} = 5.0\%$ ($N_{\text{seq}} \ge 32$) and $L_{\max} = 4$, PrefixSpan discovers 22 patterns.
  - Permits intermediate hops ($A \to X \to B$ contains subsequence $A \to B$).
- **Direct Contiguous Mining:**
  - Extracts strictly adjacent 1-hop bigrams ($A \to B$).
  - At $s_{\min} = 5.0\%$, only 2 patterns meet the support threshold.
  - Tuning $s_{\min}$ down to $2.0\%$ uncovers 33 frequent adjacent transitions.

---

## 4. Evaluation Methodology Audit

### 4.1 Common Evaluation Unit & Directionality
A primary source of methodological error in sequential pattern evaluation is comparing patterns of variable length ($k \ge 2$) against transition baselines without unit alignment.

The codebase strictly enforces **pair-level decomposition**:
1. Any discovered sequence $\langle s_1, s_2, \dots, s_k \rangle$ is projected into ordered, directed pairs:
   $$P_{\text{pairs}} = \{ (s_i, s_{i+m}) \mid 1 \le i < i+m \le k, \, s_i \neq s_{i+m} \}$$
2. Self-transitions ($u = v$) are excluded.
3. Multiple occurrences of the same directed pair within or across patterns are deduplicated.
4. Directionality is preserved: $(u, v) \neq (v, u)$.

### 4.2 Mathematical Metrics Formulation
All metrics are computed directly from the confusion matrix counts:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F1 = \frac{2 \cdot P \cdot R}{P + R}$$
Zero denominators are safely guarded: if $(TP + FP) = 0$ or $(TP + FN) = 0$, the function returns $0.0$.

### 4.3 Nature of the Empirical Reference Benchmark (Critical Caveat)
In real-world unsupervised trajectory mining, independent human ground truth for urban flows rarely exists. The repository constructs an **Empirical Reference Set** consisting of the 35 most frequent adjacent cell transitions observing support $s \ge 1.5\%$ across the sample fleet.

> **Methodological Note for Research Paper:**
> The empirical reference reflects **historical trajectory agreement**, NOT independently verified ground truth.
> Because the empirical reference is composed exclusively of *adjacent* 1-hop transitions, it structurally favors direct contiguous mining over PrefixSpan. When PrefixSpan discovers valid arterial corridors that involve multi-hop hops (e.g., $A \to C$ across a highway), these are counted as False Positives ($FP$) against the 1-hop reference.
> The paper must clearly state this limitation and present the synthetic benchmark to establish objective ground-truth recovery.

### 4.4 Objective Synthetic Benchmark Validation
To provide an uncompromised evaluation independent of empirical reference biases, `src/synthetic_benchmark.py` implements a controlled synthetic generator (Seed = 42, 350 sequences):
- 5 known ground-truth mobility corridors (13 directed transitions) are planted at controlled supports (8% to 15%) alongside realistic background noise.
- **Planted Recovery Results:** PrefixSpan achieves **100% recall** ($13 / 13$ planted transitions) across all evaluated support thresholds.
- Support tuning from 10% baseline to 13% tuned threshold increases precision from $0.2241$ to $0.5652$ and $F1$ from $0.3662$ to $0.7222$ (+97.2% relative gain) by eliminating background noise fragments while maintaining 100% ground-truth recovery.

### 4.5 Chronological Holdout Validation & Temporal Scope
To prevent temporal data leakage, the repository partitions the 7-day dataset chronologically:
- **Training Period:** Feb 2–5, 2008 (364 sequences) — used for pattern discovery.
- **Validation Period:** Feb 6, 2008 (90 sequences, 39 reference transitions) — used strictly for parameter tuning.
- **Untouched Test Period:** Feb 7–8, 2008 (176 sequences, 22 reference transitions) — evaluated without parameter re-tuning.

#### Temporal Generalization Caveats:
- Pattern recurrence drops on untouched future days: Direct Contiguous peak F1 decreases from $0.1975$ on validation day to $0.0937$ on the test period; PrefixSpan F1 decreases from $0.0760$ to $0.0325$.
- Because the dataset spans a single calendar week, **no claims of seasonal, monthly, or annual generalizability can be made**. The findings represent localized weekly mobility patterns.

---

## 5. Verification of Numerical Results

All reported numerical claims in existing reports have been cross-checked against source CSVs and reproduced scripts. Full provenance is documented in [`results_provenance.csv`](../audit/results_provenance.csv).

| Metric / Experiment | Reported Claim | Source Output CSV | Verification Status | Discrepancy / Auditor Notes |
| :--- | :--- | :--- | :---: | :--- |
| Raw Observations | 129,545 records | `cleaning_summary.csv` | **VERIFIED** | Exact match across 100 raw taxi files |
| Active Taxis | 99 taxis | `cleaning_summary.csv` | **VERIFIED** | 100 requested; Taxi 4375 dropped due to 311 null fixes |
| Clean Observations | 116,913 records | `cleaning_summary.csv` | **VERIFIED** | Exact match (9.75% cumulative filter rate) |
| Occupied Grid Cells | 5,413 cells | `spatial_grid_summary.csv` | **VERIFIED** | At 500m planar equirectangular grid |
| DBSCAN Hotspots | 237 clusters | `spatial_clusters.csv` | **VERIFIED** | Full pipeline ($\epsilon=300\text{ m}, \text{min}=15$); 52 in sampled table |
| PrefixSpan Patterns (5%) | 22 patterns | `prefixspan_patterns.csv` | **VERIFIED** | Max pattern length = 4 |
| Empirical Ref Transitions | 35 pairs | `evaluation_reference_patterns.csv` | **VERIFIED** | Adjacent transitions with support $\ge 1.5\%$ |
| Direct Baseline (5%) | $P=1.0, R=0.057, F1=0.108$ | `baseline_vs_best_comparison.csv` | **VERIFIED** | $TP=2, FP=0, FN=33$ |
| Direct Exploratory (2%) | $P=0.818, R=0.771, F1=0.794$ | `baseline_vs_best_comparison.csv` | **VERIFIED** | $TP=27, FP=6, FN=8$; +634.6% relative F1 gain |
| PrefixSpan Baseline (5%) | $P=0.550, R=0.314, F1=0.400$ | `baseline_vs_best_comparison.csv` | **VERIFIED** | $TP=11, FP=9, FN=24$ |
| PrefixSpan Top-5 Composite | $P=0.750, R=0.086, F1=0.154$ | `baseline_vs_best_comparison.csv` | **VERIFIED** | Precision gains +36.4% via composite ranking |
| Synthetic Planted Baseline | $P=0.224, R=1.000, F1=0.366$ | `synthetic_benchmark_tuning.csv` | **VERIFIED** | 100% recall of 13 planted pairs |
| Synthetic Planted Tuned | $P=0.565, R=1.000, F1=0.722$ | `synthetic_benchmark_tuning.csv` | **VERIFIED** | 100% recall; +97.2% relative F1 gain |
| Pipeline Scaling (500 fleet) | 8.19 s runtime, 833k clean pts | `sample_expansion_comparison.csv` | **VERIFIED** | Highly efficient sub-10-second processing |

---

## 6. Identified Contradictions & Audit Resolutions

1. **"100 Taxis Analyzed" in Early Documentation:**
   - *Contradiction:* Early drafts stated "100 taxis analyzed", but the clean analytical dataset contains exactly 99 taxis.
   - *Resolution:* Clarified that 100 files were ingested, but Taxi 4375 was purged during coordinate cleaning due to null `(0.0, 0.0)` fixes.
2. **Empirical Reference Described as "Ground Truth":**
   - *Contradiction:* Initial report language occasionally referred to the empirical reference as "ground truth".
   - *Resolution:* Standardized to "Empirical Reference Benchmark" and formally distinguished from the synthetic ground-truth benchmark.
3. **PrefixSpan False Positives vs. Adjacent Reference:**
   - *Contradiction:* PrefixSpan appeared to have low precision ($0.55$) compared to Direct Contiguous mining ($1.00$).
   - *Resolution:* Demonstrated mathematically that multi-hop corridors mined by PrefixSpan are penalized by the 1-hop adjacent reference, while the synthetic benchmark proves PrefixSpan achieves $100\%$ true corridor recovery.

---

## 7. Conclusion & Readiness

The repository code, test suites, and empirical tables are mathematically sound, fully reproducible, and rigorously traceable. The planned report assets (Tables I–VIII, Figures 1–14, and reproducibility manifests) are ready for inclusion in the IEEE research paper draft.

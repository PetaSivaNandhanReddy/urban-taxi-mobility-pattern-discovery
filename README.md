# Urban Taxi Mobility Pattern Discovery for Smart Transportation Planning

**Final-year M.Tech Business Analytics research project.**
Sequential and spatial data mining on urban taxi GPS trajectories, using
the Microsoft Research **T-Drive** dataset as a generic benchmark
(methodology designed to later accept an Indian taxi dataset as well).

> **Status: Phase 3 complete.** Phase 1 (loading, cleaning, trajectory
> reconstruction, EDA), Phase 2 (spatial grid, DBSCAN clustering,
> PrefixSpan mining, temporal analysis, corridor integration), and Phase 3
> (interactive Streamlit dashboard) are all done. See "Current Progress"
> below for what each phase covers.

---

## 1. Project Overview

This project builds an end-to-end mobility analytics pipeline that takes
raw taxi GPS trajectories and produces, across all planned phases:

1. Clean taxi trajectories
2. Spatial zones/grid representation
3. Frequent sequential movement patterns
4. Spatial taxi hotspots
5. Temporal mobility patterns
6. Integrated sequential + spatial mobility corridors
7. Transportation-planning insights
8. An interactive Streamlit dashboard

The dataset used to build and validate the system is T-Drive (Beijing).
**This is treated purely as a benchmark** — nothing in the code assumes
Beijing- or China-specific logic beyond a configurable bounding-box sanity
check (see `src/config.py`), so the pipeline can later accept an Indian
taxi GPS dataset with only configuration changes.

---

## 2. Dataset — Microsoft Research T-Drive

- **Source (official):** Microsoft Research T-Drive project page —
  search "Microsoft Research T-Drive trajectory data sample" or check
  `https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/`
  for the current official download links. Microsoft has changed hosting
  for this dataset before, so **verify the link resolves before relying on it.**
- **Content:** ~10,357 taxis, ~15 million GPS points, one week of Beijing
  taxi trajectories.
- **Format:** the official release is **one `.txt` file per taxi**
  (e.g. `1.txt`, `2.txt`, ... `10357.txt`), comma-separated, no header,
  with columns `taxi_id, timestamp, longitude, latitude`. This is
  documented in the T-Drive readme, but this project's loader **verifies
  it against your actual files rather than assuming it blindly**
  (`src/data_loader.py::detect_schema`).
- **Secondary mirror (fallback only):** if the official source is
  inconvenient, the dataset is also commonly mirrored on Kaggle under
  searches like "T-Drive taxi trajectory Beijing" — verify the mirror's
  contents match the official schema before trusting it, since mirrors can
  repackage the data differently.

### 2.1 Where to place the files

1. Download the archive (likely named something like
   `taxi_log_2008_by_id.zip` or similar — **confirm the actual filename
   from your download**, do not assume).
2. Extract it. However it extracts (directly into a folder of `.txt`
   files, or into a nested subfolder), place the **top-level extracted
   folder** under:
   ```
   urban-taxi-mobility/data/raw/
   ```
   The loader searches **recursively** under `data/raw/` for files named
   as plain integers (`1.txt`, `2.txt`, ...), so it does not matter
   whether your files sit directly in `data/raw/` or in
   `data/raw/taxi_log_2008_by_id/`.
3. Verify the extraction:
   ```
   python src/data_loader.py
   ```
   This prints how many taxi files were found, the detected schema, and a
   preview of loaded rows. If it raises `DataLoadError`, the message tells
   you exactly what's missing.

**Raw data must remain untouched** — nothing in this codebase writes to
`data/raw/`. Samples and cleaned data are written to `data/samples/` and
`data/processed/` instead.

---

## 3. Installation

### 3.1 What you must do manually

See **"WHAT I MUST DO MANUALLY"** at the end of this document for the
exact commands (Python install check, virtual environment, package
installation, dataset download).

### 3.2 Opening the project in VS Code

1. Open VS Code.
2. `File > Open Folder...` → select the `urban-taxi-mobility` folder.
3. Open a terminal in VS Code (`` Ctrl+` ``) — it opens already inside the
   project folder.
4. Create and activate a virtual environment, then install dependencies
   (exact commands below in "What I Must Do Manually").
5. Install the Jupyter extension for VS Code if prompted (needed to run
   `notebooks/01_data_exploration.ipynb` inside VS Code). When opening the
   notebook, select the project's virtual environment as the kernel.

---

## 4. Folder Structure

```
urban-taxi-mobility/
├── data/
│   ├── raw/           # Extracted T-Drive files go here (untouched by code)
│   ├── processed/     # Cleaned/labeled/clustered data (generated)
│   └── samples/       # Development-sized samples (generated)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Phase 1: EDA
│   ├── 02_spatial_analysis.ipynb   # Phase 2: grid, DBSCAN, hotspots, temporal
│   └── 03_sequential_mining.ipynb  # Phase 2: PrefixSpan, corridors, metrics
├── src/
│   ├── config.py         # All paths + parameters (Phase 1 & 2) — edit here
│   ├── data_loader.py    # Flexible T-Drive loader + schema detection
│   ├── preprocessing.py  # Cleaning pipeline with stage-by-stage reporting
│   ├── trajectory.py     # Per-taxi trajectory reconstruction + diagnostics
│   ├── spatial.py        # Grid discretization + consecutive-dup compression
│   ├── sequential.py     # Sequence building + PrefixSpan pattern mining
│   ├── clustering.py     # DBSCAN spatial clustering (haversine distance)
│   ├── temporal.py       # Hour / segment / weekday-weekend activity analysis
│   └── integration.py    # Sequential + spatial corridor identification
├── outputs/
│   ├── figures/    # EDA + Phase 2 charts (PNG)
│   ├── maps/       # Interactive Folium maps (HTML)
│   ├── tables/     # Summary tables (CSV) — grids, clusters, activity
│   └── results/    # Phase 2 mining/metrics results (CSV)
├── dashboard/
│   └── app.py      # Phase 3: interactive Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. Data Format (as standardized by this project)

After loading (`src/data_loader.py`), all data uses these standardized
column names regardless of the raw file's exact header/order:

| Column      | Type              | Description                          |
|-------------|-------------------|---------------------------------------|
| `taxi_id`   | string            | Unique taxi identifier                |
| `timestamp` | string → datetime | Parsed in `preprocessing.py`          |
| `lon`       | float             | Longitude                             |
| `lat`       | float             | Latitude                              |

After trajectory reconstruction (`src/trajectory.py`), these derived
columns are added: `time_diff_sec`, `distance_m`, `speed_kmh`,
`is_suspicious_jump`.

---

## 6. Preprocessing Approach

`src/preprocessing.py` runs a staged pipeline, reporting record counts
removed at every stage (never silent):

1. Remove missing/empty `taxi_id`
2. Parse timestamps; remove rows that fail to parse
3. Remove missing/non-numeric coordinates
4. Remove physically impossible coordinates (outside ±90/±180, or exactly
   `(0, 0)` — a common GPS null-fix sentinel)
5. **Flag** (not delete) points outside the expected Beijing bounding box —
   pass `drop_out_of_region=True` to actually remove them after review
6. Remove exact duplicate `(taxi_id, timestamp, lat, lon)` rows

The full audit trail is saved to `outputs/tables/cleaning_summary.csv`
each time the notebook runs.

---

## 6a. Spatial Discretization (`src/spatial.py`)

Sequential pattern mining needs discrete, repeatable *items*. Raw
(lat, lon) pairs are effectively unique to many decimal places, so no
"item" could ever recur and support counting would be meaningless. This
project overlays a configurable square grid (`config.GRID_CELL_SIZE_METERS`,
default 500m) on the data's bounding box and assigns each GPS point a
`spatial_cell_id` (e.g. `CELL_00001`), numbered deterministically over
only the cells that actually contain data. Cell size is converted from
metres to degrees at the data's median latitude, since a degree of
longitude is not a constant physical distance away from the equator.

Consecutive repeats of the same cell (a taxi idling within one cell) are
compressed to a single visit; non-consecutive repeats (returning to a
cell later) are preserved — e.g. `A,A,A,B,B,C` → `A,B,C`, but `A,B,A`
stays `A,B,A`.

## 6b. DBSCAN Spatial Clustering (`src/clustering.py`)

Hotspot detection uses scikit-learn's DBSCAN with **haversine distance**
(great-circle distance), not raw Euclidean distance on lat/lon — Euclidean
distance on angular coordinates distorts real-world distances away from
the equator. `eps` and `min_samples` are chosen from evidence rather than
guessed: a k-distance elbow plot (`k = min_samples`, look for where the
sorted k-th-nearest-neighbour distance curve bends sharply) plus a
parameter comparison grid (`clustering.compare_dbscan_parameters()`)
reporting cluster count and noise fraction across candidate values — both
computed in `02_spatial_analysis.ipynb`. Defaults
(`config.DBSCAN_EPS_METERS = 300`, `config.DBSCAN_MIN_SAMPLES = 15`) are
starting points; re-run the comparison on your own sample size, since the
right elbow shifts with point density.

## 6c. PrefixSpan Sequential Pattern Mining (`src/sequential.py`)

Uses the `prefixspan` PyPI package (support-counting verified during
development against a manual brute-force subsequence count on a small
fixture before being trusted in the pipeline). Minimum support is
expressed as a **fraction of sequences** (`config.DEFAULT_MIN_SUPPORT`,
default 0.05) and converted internally to an absolute count. Sequences
are built per-taxi from compressed cell visits, with optional
segmentation by time-of-day (`morning`/`afternoon`/`evening`/`night`,
boundaries in `config.TIME_SEGMENTS`) or weekday/weekend
(`sequential.build_taxi_sequences(df, segment_by=...)`).

## 6d. Temporal Segmentation (`src/temporal.py`)

Named time-of-day segments and weekday/weekend labels are derived from
each point's timestamp (`sequential.add_temporal_labels()`) and
summarized by hour, segment, and day type — used both for the temporal
activity charts in notebook 02 and as the segmentation keys for mining in
notebook 03.

## 6e. Sequential + Spatial Integration (`src/integration.py`)

Three concepts are kept explicitly distinct:

1. **Spatial hotspot** — a dense DBSCAN cluster. Says nothing about
   movement or order.
2. **Frequent sequence** — a recurring chain of grid cells (from
   PrefixSpan). Says nothing about whether those cells are dense.
3. **Recurring spatial mobility corridor** — a frequent sequence whose
   cells map onto **2 or more distinct, non-noise DBSCAN clusters** —
   i.e. it actually connects real hotspots, not just an arbitrary path
   through low-density space. Only this subset is called a "corridor."

Each grid cell is mapped to its dominant cluster (the most common
`cluster_id` among the raw points that fall in that cell), and every
mined pattern is translated into a cluster-id sequence to check how many
distinct real clusters it touches. **No causal claim is made** — a
corridor is a descriptive, statistical recurrence, not an explanation of
why taxis move that way.

---

## 7. How to Run Phase 1

```bash
# from the project root, with your virtual environment activated
python src/data_loader.py        # sanity check: confirms files are found + schema
python src/preprocessing.py      # sanity check: runs cleaning on a tiny synthetic fixture
python src/trajectory.py         # sanity check: runs trajectory reconstruction on a tiny synthetic fixture
```

Then open and run all cells in `notebooks/01_data_exploration.ipynb`
(via VS Code's Jupyter integration, or `jupyter notebook` from the
terminal). It will:

- detect and report the actual raw file schema
- load a configurable sample (`config.SAMPLE_TAXI_COUNT`, default 100 taxis)
- clean it and save the audit trail
- reconstruct trajectories and save diagnostics
- produce 11 EDA charts under `outputs/figures/`
- produce an interactive map under `outputs/maps/`

To change the sample size (e.g. to run on all taxis for a final pass),
edit `SAMPLE_TAXI_COUNT` in `src/config.py`.

---

## 7a. How to Run Phase 2

```bash
# from the project root, with your virtual environment activated
python src/spatial.py       # sanity check: grid assignment + compression on a tiny synthetic fixture
python src/sequential.py    # sanity check: sequence building + PrefixSpan mining on a tiny synthetic fixture
python src/clustering.py    # sanity check: DBSCAN on a tiny two-cluster synthetic fixture
python src/temporal.py      # sanity check: hour/segment/day-type activity summaries on a tiny synthetic fixture
python src/integration.py   # sanity check: corridor identification on a tiny synthetic fixture
```

Then run, **in order**:

1. `notebooks/02_spatial_analysis.ipynb` — grid assignment, DBSCAN
   clustering (with k-distance elbow + parameter comparison), hotspot
   maps, and temporal activity analysis. Produces
   `data/processed/labeled_sample.csv` (cells + clusters + temporal
   labels combined) and `outputs/tables/cluster_summary.csv`, both
   required by notebook 03.
2. `notebooks/03_sequential_mining.ipynb` — sequence building, PrefixSpan
   mining, pattern analysis, the sequential+spatial corridor integration,
   and the research metrics summary. Reads the two files notebook 02
   produces — running notebook 03 without first running notebook 02 will
   raise a clear `FileNotFoundError` telling you to run 02 first.

To change grid resolution, DBSCAN parameters, or the minimum support
threshold, edit the corresponding values in `src/config.py` (`GRID_CELL_SIZE_METERS`,
`DBSCAN_EPS_METERS`, `DBSCAN_MIN_SAMPLES`, `DEFAULT_MIN_SUPPORT`).

---

## 7b. How to Run Phase 3 (Dashboard)

```bash
# from the project root, with your virtual environment activated
streamlit run dashboard/app.py
```

This opens the dashboard in your browser (usually `http://localhost:8501`).
It reads `data/processed/labeled_sample.csv`, produced by
`notebooks/02_spatial_analysis.ipynb` — **run Phase 2 first**. If that
file is missing, the dashboard shows a clear on-screen message telling
you which notebooks to run, instead of crashing.

**What the dashboard does:**
- Sidebar filters (day type, time segment) and live parameter sliders
  (DBSCAN `eps`/`min_samples`, PrefixSpan minimum support) that re-run
  clustering and pattern mining on the currently filtered subset — so
  numbers shown here can differ from the notebooks' saved CSVs (which use
  the full sample and `config.py`'s fixed defaults); that's expected.
- **Spatial Hotspots** tab — interactive cluster map + cluster size table/chart
- **Temporal Patterns** tab — hourly, segment, and weekday/weekend activity charts
- **Sequential Patterns** tab — top patterns chart, length distribution,
  full patterns table
- **Mobility Corridors** tab — corridor table + a map of the single
  highest-support corridor's route between real clusters
- **About** tab — methodology notes and the hotspot/sequence/corridor
  distinction, restated for a reader of the dashboard itself

---

## 8. Current Progress

**Completed (Phase 1):**
- Project structure, configuration, `.gitignore`
- Dataset acquisition instructions
- Schema-detecting, flexible T-Drive loader with configurable sampling
- Staged, reported data cleaning pipeline
- Trajectory reconstruction with time/distance/speed diagnostics and
  suspicious-jump flagging
- Full EDA notebook (11+ analyses) with saved figures
- Initial interactive map (sampled points + example trajectories)

**Completed (Phase 2):**
- Configurable grid-based spatial discretization with deterministic
  `spatial_cell_id` assignment, and consecutive-duplicate compression
- Per-taxi sequence construction with optional time-segment / weekday-weekend
  segmentation
- PrefixSpan sequential pattern mining (support-counting verified against
  a manual brute-force check), with pattern-length and support-distribution
  analysis, and pattern mining broken out by time segment and day type
- DBSCAN spatial clustering using haversine distance, with an
  evidence-based eps/min_samples selection procedure (k-distance elbow +
  parameter comparison grid), noise-point identification, cluster size
  table, and interactive hotspot/heatmap visualizations
- Temporal mobility analysis by hour, named segment, and weekday/weekend
- Sequential + spatial integration distinguishing hotspots, frequent
  sequences, and genuine hotspot-to-hotspot mobility corridors (no
  causal claims)
- Research metrics summary and a minimum-support sensitivity comparison
  (0.01 / 0.02 / 0.05)
- Two new notebooks (`02_spatial_analysis.ipynb`,
  `03_sequential_mining.ipynb`) and updated documentation

**Completed (Phase 3):**
- Interactive Streamlit dashboard (`dashboard/app.py`) with live filters
  (day type, time segment) and live-recomputed DBSCAN/PrefixSpan
  parameters
- Five dashboard sections: Spatial Hotspots, Temporal Patterns,
  Sequential Patterns, Mobility Corridors, and About
- Clean on-screen guidance (not a crash) when Phase 2 hasn't been run yet
- Verified with Streamlit's `AppTest` framework: default load, each
  filter, each slider, an empty-filter-result edge case, and the
  missing-data guard clause all confirmed exception-free; also
  smoke-tested with a real `streamlit run` server (HTTP 200)

**Not yet started (future work — do not implement until explicitly requested):**
- Deployment/packaging of the dashboard beyond local `streamlit run`
- Validation against a second (e.g. Indian) taxi dataset

---


## WHAT I MUST DO MANUALLY

These are the only actions you personally need to perform. Exact commands
are given for Windows (PowerShell, since you're in VS Code on Windows).

### 1. Confirm Python is installed
```powershell
python --version
```
If this fails, install Python 3.10+ from https://www.python.org/downloads/
(check "Add python.exe to PATH" during install).

### 2. Confirm VS Code is installed
Download from https://code.visualstudio.com/ if not already installed.
Install the **Python** and **Jupyter** extensions from the VS Code
Extensions panel.

### 3. Create the project folder and open it in VS Code
```powershell
mkdir urban-taxi-mobility
cd urban-taxi-mobility
code .
```
(All the files Claude created should be placed into this folder — see
"WHAT CLAUDE CREATED" below.)

### 4. Create and activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, run PowerShell as Administrator
once and execute `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`,
then retry activation.

### 5. Install dependencies
```powershell
pip install -r requirements.txt
```

### 6. Download the T-Drive dataset
- Go to the official Microsoft Research T-Drive page (search
  "Microsoft Research T-Drive trajectory data sample" — verify the link
  before downloading, hosting has changed over time).
- Download the release archive.
- Note the exact filename you receive — it may not match older
  documentation.

### 7. Extract the archive
- Right-click the downloaded `.zip` → "Extract All..."
- Move the extracted folder (or its contents) into:
  `urban-taxi-mobility\data\raw\`

### 8. Verify the dataset loads correctly
```powershell
python src\data_loader.py
```
Confirm it prints a file count and a preview of loaded rows without
raising `DataLoadError`.

### 9. Run the Phase 1 notebook
Open `notebooks/01_data_exploration.ipynb` in VS Code, select the `venv`
kernel when prompted, and run all cells (`Run All`).

### 10. Add the Phase 2 files
Copy the new/updated files listed under "WHAT CLAUDE CREATED (Phase 2)"
below into your existing `urban-taxi-mobility` folder — `src/config.py`,
`src/spatial.py`, `src/sequential.py`, `src/clustering.py`,
`src/temporal.py`, `src/integration.py`, the two new notebooks, and the
updated `README.md` / `requirements.txt`. This does not touch your
downloaded data or Phase 1 outputs.

### 11. Install the one new Phase 2 dependency
```powershell
pip install -r requirements.txt
```
(re-running this picks up the newly added `prefixspan` package; everything
else was already installed for Phase 1.)

### 12. Run the Phase 2 notebooks, in order
Open `notebooks/02_spatial_analysis.ipynb`, select the `venv` kernel, and
`Run All`. Then open `notebooks/03_sequential_mining.ipynb` and `Run All`
— it depends on files notebook 02 produces, so it must run second.

### 13. Add the Phase 3 dashboard file
Copy `dashboard/app.py` (listed under "WHAT CLAUDE CREATED (Phase 3)"
below) into your existing `urban-taxi-mobility/dashboard/` folder.

### 14. Install the two new Phase 3 dependencies
```powershell
pip install -r requirements.txt
```
(picks up the newly added `streamlit` and `streamlit-folium` packages).

### 15. Launch the dashboard
```powershell
streamlit run dashboard\app.py
```
This opens your default browser to the dashboard automatically (usually
`http://localhost:8501`). Press `Ctrl+C` in the terminal to stop it.

---

## WHAT CLAUDE CREATED (Phase 2)

```
urban-taxi-mobility/
├── src/config.py           # UPDATED — added Phase 2 grid/mining/DBSCAN parameters
├── src/spatial.py          # NEW — grid discretization + consecutive-dup compression
├── src/sequential.py       # NEW — sequence building + PrefixSpan mining
├── src/clustering.py       # NEW — DBSCAN haversine clustering + parameter selection
├── src/temporal.py         # NEW — hour/segment/weekday-weekend activity analysis
├── src/integration.py      # NEW — sequential + spatial corridor identification
├── notebooks/02_spatial_analysis.ipynb    # NEW
├── notebooks/03_sequential_mining.ipynb   # NEW
├── requirements.txt        # UPDATED — added prefixspan
├── README.md                # UPDATED — Phase 2 methodology + this section
└── outputs/results/          (created empty; populated when you run notebook 03)
```

All five Phase 2 modules were tested end-to-end against a synthetic
multi-taxi fixture with injected hotspot/corridor structure during
development — grid assignment, non-consecutive-repeat preservation,
PrefixSpan support counts (cross-checked against a manual brute-force
subsequence count), DBSCAN cluster/noise recovery, temporal labeling, and
corridor identification (correctly recovering the 3 injected hotspots and
the exact injected A→B→C→A route) all verified to run without errors —
that test fixture is **not** included in your copy of the project.

---

## WHAT CLAUDE CREATED (Phase 3)

```
urban-taxi-mobility/
├── dashboard/app.py      # NEW — interactive Streamlit dashboard
├── requirements.txt      # UPDATED — added streamlit, streamlit-folium
└── README.md              # UPDATED — Phase 3 how-to-run + this section
```

`dashboard/app.py` was tested with Streamlit's official `AppTest`
framework (headless, no browser needed) against a synthetic multi-taxi
fixture run through the real Phase 1 + 2 pipeline — confirmed
exception-free on: initial load, the day-type filter, the time-segment
filter, all three parameter sliders (DBSCAN eps, DBSCAN min_samples,
minimum support), a narrow combined-filter edge case, and the
missing-`labeled_sample.csv` guard clause (shows a clean on-screen
message instead of crashing). Also smoke-tested by actually launching
`streamlit run` and confirming the server responds (HTTP 200). That test
fixture and its generated CSVs are **not** included in your copy of the
project.

---

## PHASE 2 COMPLETION CHECK

Run these in order (Phase 1's dataset must already be in `data/raw/`):

1. **Module sanity checks** (each runs against a small built-in synthetic
   fixture — no real data needed for these five):
   ```powershell
   python src\spatial.py
   python src\sequential.py
   python src\clustering.py
   python src\temporal.py
   python src\integration.py
   ```
   ✅ Pass: each prints its fixture's input/output tables with no exception.

2. **Notebook 02 — spatial analysis**
   Open and "Run All" on `notebooks/02_spatial_analysis.ipynb`.
   ✅ Pass:
   - No cell raises an exception
   - `outputs/figures/` gains 5 new PNGs (`07_...` through `11_...`)
   - `outputs/maps/activity_heatmap.html` and
     `outputs/maps/hotspot_clusters.html` exist and open in a browser
   - `outputs/tables/spatial_grid_summary.csv`,
     `outputs/tables/cluster_summary.csv`, and
     `outputs/tables/dbscan_parameter_comparison.csv` exist
   - `data/processed/clustered_sample.csv` and
     `data/processed/labeled_sample.csv` exist and are non-empty

3. **Notebook 03 — sequential mining**
   Open and "Run All" on `notebooks/03_sequential_mining.ipynb` (after
   notebook 02).
   ✅ Pass:
   - No cell raises an exception (a `FileNotFoundError` here means
     notebook 02 wasn't run first)
   - `outputs/figures/` gains 4 more PNGs (`12_...` through `15_...`)
   - `outputs/results/frequent_patterns.csv`,
     `outputs/results/mobility_corridors.csv`, and
     `outputs/results/phase2_research_metrics.csv` exist and are non-empty
     (an empty `mobility_corridors.csv` is possible and not necessarily a
     bug — it means no mined pattern touched 2+ distinct clusters at the
     current min_support/DBSCAN settings; try lowering
     `config.DEFAULT_MIN_SUPPORT` or revisiting the DBSCAN parameters)

4. **Spot-check the numbers make sense**
   - `n_spatial_clusters` in `phase2_research_metrics.csv` is a small,
     plausible number (not 0, not one cluster per point)
   - The top rows of `mobility_corridors.csv` involve cluster paths that
     make geographic sense when checked against `cluster_summary.csv`'s
     `center_lat`/`center_lon`
   - Pattern counts in the support-threshold comparison table increase as
     `min_support` decreases (0.05 → 0.02 → 0.01), as expected

---

## PHASE 3 COMPLETION CHECK

Requires `data/processed/labeled_sample.csv` to already exist (run
notebook 02 first).

1. **Launch check**
   ```powershell
   streamlit run dashboard\app.py
   ```
   ✅ Pass: terminal shows a `Local URL: http://localhost:8501` line and
   your browser opens the dashboard with no red error box.
   ❌ Fail (missing-data message shown instead): run
   `02_spatial_analysis.ipynb` first.

2. **Overview metrics**
   ✅ Pass: the five metric tiles at the top (Taxis in view, GPS points in
   view, Spatial clusters, Noise points, Date range) all show non-zero,
   plausible values matching your configured sample size.

3. **Each tab loads without an error banner**
   Click through all five tabs — Spatial Hotspots, Temporal Patterns,
   Sequential Patterns, Mobility Corridors, About.
   ✅ Pass: each renders its map/charts/tables; Streamlit shows a red
   error box only if something is genuinely wrong.

4. **Filters and sliders actually change the results**
   - Switch "Day type" to Weekend, or "Time segment" to a specific
     segment — the metric tiles and charts should update.
   - Move the `eps (metres)` or `min_samples` sliders — the hotspot map
     and cluster count should change.
   - Move the "Minimum support (%)" slider down — the Sequential Patterns
     and Mobility Corridors tabs should generally show *more* patterns.
   ✅ Pass: no red error box appears after any of these changes, even if
   a narrow filter combination legitimately produces the "no data" info
   message (that's a correct outcome, not a bug).

---

**All three phases of the original project scope are now complete.**
If you'd like further work — validating against a second (e.g. Indian)
taxi dataset, refining any phase, or preparing this for submission —
just ask.

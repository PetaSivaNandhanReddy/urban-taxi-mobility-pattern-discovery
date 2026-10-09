"""
config.py
=========
Central configuration for the Urban Taxi Mobility Pattern Discovery project.

Everything that might need to change between machines, development runs,
and full runs lives here — no other module should hardcode a path or a
sampling parameter. Edit the values in this file to change project-wide
behaviour.

All paths are built with pathlib and are relative to the project root,
so the project works identically on Windows, macOS, and Linux.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------
# This file lives at <project_root>/src/config.py, so the project root is
# one level up from this file's parent directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# DATA DIRECTORIES
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"

# Where the T-Drive taxi log files live after extraction. T-Drive's
# "release" archive extracts into a folder usually named `taxi_log_2008_by_id`
# containing one .txt file per taxi (e.g. 1.txt, 2.txt, ...). We point at
# RAW_DATA_DIR itself and let the loader search recursively, so it does not
# matter whether you extract directly into data/raw/ or into a subfolder —
# see PART 2 of the README for exact extraction instructions.
TDRIVE_RAW_DIR = RAW_DATA_DIR

# ---------------------------------------------------------------------------
# OUTPUT DIRECTORIES
# ---------------------------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MAPS_DIR = OUTPUTS_DIR / "maps"
TABLES_DIR = OUTPUTS_DIR / "tables"
RESULTS_DIR = OUTPUTS_DIR / "results"

# ---------------------------------------------------------------------------
# SAMPLING STRATEGY (PART 5)
# ---------------------------------------------------------------------------
# During development, do NOT load all ~10,357 taxis / ~15M points.
# Change SAMPLE_TAXI_COUNT to None to load every taxi file found.
SAMPLE_TAXI_COUNT = 100          # e.g. 100, 500, or None for "all taxis"
SAMPLE_ROW_COUNT = None          # optional cap on total rows after loading; None = no cap
RANDOM_SEED = 42                 # for reproducible sampling

# ---------------------------------------------------------------------------
# T-DRIVE SCHEMA (verified against the official release format)
# ---------------------------------------------------------------------------
# The official T-Drive per-taxi .txt files are comma-separated, no header,
# with four columns: taxi_id, timestamp, longitude, latitude.
# The loader in src/data_loader.py DETECTS this rather than assuming it
# blindly — these are documented defaults, used as a fallback only if
# detection is inconclusive. See PART 3 / notebooks/01_data_exploration.ipynb
# for the actual detected schema on your downloaded files.
TDRIVE_EXPECTED_COLUMNS = ["taxi_id", "timestamp", "longitude", "latitude"]
TDRIVE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Beijing's approximate bounding box, used only as a sanity-check range for
# "impossible coordinate" detection in PART 6 — NOT used to silently drop
# points. Points outside this box are flagged and reported, not deleted
# outright (see src/preprocessing.py).
BEIJING_LAT_RANGE = (39.4, 41.6)
BEIJING_LON_RANGE = (115.4, 117.5)

# ---------------------------------------------------------------------------
# STANDARDIZED COLUMN NAMES (used across all modules from this point on)
# ---------------------------------------------------------------------------
COL_TAXI_ID = "taxi_id"
COL_TIMESTAMP = "timestamp"
COL_LAT = "lat"
COL_LON = "lon"


# ---------------------------------------------------------------------------
# PHASE 2 — SPATIAL GRID (src/spatial.py)
# ---------------------------------------------------------------------------
# Side length of each square spatial cell, in metres. This is the primary
# knob controlling spatial discretization resolution: smaller cells give
# finer-grained (but sparser, harder-to-mine) sequences; larger cells give
# coarser (but denser, easier-to-mine) sequences. 500m is a reasonable
# starting point for urban taxi mobility — roughly a 5-6 minute walk.
GRID_CELL_SIZE_METERS = 500.0

# ---------------------------------------------------------------------------
# PHASE 2 — SEQUENTIAL PATTERN MINING (src/sequential.py)
# ---------------------------------------------------------------------------
# Minimum support as a fraction of sequences (0-1). PrefixSpan support is
# computed as an absolute count internally; this fraction is converted to
# a count based on the number of sequences being mined at call time.
DEFAULT_MIN_SUPPORT = 0.05
MAX_PATTERN_LENGTH = 4

# Support thresholds compared side-by-side in the sensitivity analysis
# (Part 10). Chosen to span "strict" to "lenient" while staying
# computationally feasible on a development-sized sample.
SUPPORT_THRESHOLDS_TO_COMPARE = [0.01, 0.02, 0.05]

# Time-of-day segment boundaries (24h clock, hour of day, start-inclusive).
TIME_SEGMENTS = {
    "morning": (6, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 6),  # wraps past midnight
}

# ---------------------------------------------------------------------------
# PHASE 2 — SPATIAL CLUSTERING (src/clustering.py)
# ---------------------------------------------------------------------------
# DBSCAN parameters. eps is in METRES (converted internally to radians for
# sklearn's haversine metric) rather than raw lat/lon degrees, because a
# degree of longitude is not a constant physical distance except at the
# equator — using Euclidean distance directly on raw lat/lon would distort
# clusters away from the equator. See src/clustering.py for the eps/
# min_samples selection procedure (k-distance elbow plot) rather than an
# arbitrary guess.
DBSCAN_EPS_METERS = 300.0
DBSCAN_MIN_SAMPLES = 15
DBSCAN_EPS_CANDIDATES_METERS = [150.0, 300.0, 500.0, 800.0]
DBSCAN_MIN_SAMPLES_CANDIDATES = [5, 10, 15, 25]

# ---------------------------------------------------------------------------
# PHASE 4 — PATTERN RANKING (src/pattern_ranking.py)
# ---------------------------------------------------------------------------
# Weights for the composite mobility-pattern score. Must sum to 1.0 — each
# weight is applied to a component already normalized to [0, 1]. See
# src/pattern_ranking.py module docstring for exactly how each component
# is computed and why. Equal weighting is the default (no component is
# asserted to matter more than another) — override here if your research
# question justifies weighting one component more heavily, and say so in
# the write-up.
PATTERN_SCORE_WEIGHTS = {
    "support": 1 / 6,
    "n_taxis": 1 / 6,
    "temporal_consistency": 1 / 6,
    "spatial_density": 1 / 6,
    "pattern_length": 1 / 6,
    "recurrence": 1 / 6,
}

# ---------------------------------------------------------------------------
# PHASE 4 — ANOMALY / UNUSUAL PATTERN ANALYSIS (src/anomaly.py)
# ---------------------------------------------------------------------------
# A cell-to-cell transition is flagged as "unusual" if its share of all
# observed transitions falls at or below this percentile — i.e. it is
# among the rarest transitions actually observed, not merely "not in the
# frequent-pattern set" (which would flag the vast majority of legitimate
# but simply infrequent movements).
#
# NOTE: on real T-Drive-scale samples, many distinct transitions occur
# only once, so a percentile cut can flag a much larger share of distinct
# transitions than the percentile number itself suggests (ties at the
# bottom of the count distribution). The percentile rule is kept here for
# sensitivity reporting, but the PRIMARY rare-mobility definition used by
# anomaly.py is the more transparent occurrence-count rule below.
UNUSUAL_TRANSITION_PERCENTILE = 5.0

# Primary, occurrence-based definition of a "rare mobility transition":
# a cell-to-cell transition observed in at most this many DISTINCT
# taxi-day sequences. This is easier to defend/interpret than a
# percentile cut ("this route was seen on 2 taxi-days or fewer in the
# entire sample") and does not silently balloon when many transitions
# are tied at count 1. See anomaly.py module docstring.
RARE_TRANSITION_MAX_TAXI_DAYS = 2

# ---------------------------------------------------------------------------
# PHASE 4 — RESEARCH EVALUATION (src/evaluation.py)
# ---------------------------------------------------------------------------
# Grid cell sizes compared in the Part 9 research evaluation, alongside the
# existing DBSCAN and support-threshold candidate lists above.
GRID_SIZES_TO_COMPARE_METERS = [300.0, 500.0, 800.0]


def ensure_output_dirs() -> None:
    """Create all output directories if they do not already exist.

    Safe to call repeatedly; does nothing if directories are already present.
    """
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLES_DATA_DIR,
        FIGURES_DIR,
        MAPS_DIR,
        TABLES_DIR,
        RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

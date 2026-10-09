"""
clustering.py
=============
DBSCAN spatial clustering of taxi GPS activity — Phase 2, Parts 6-7.

WHY HAVERSINE, NOT EUCLIDEAN ON RAW LAT/LON
---------------------------------------------
Latitude and longitude are angular coordinates on a sphere, not a flat
plane. A degree of longitude covers less physical distance the further you
get from the equator (it shrinks by cos(latitude)), so naively running
Euclidean-distance DBSCAN on raw (lat, lon) pairs distorts real-world
distances and produces clusters that do not correspond to genuine spatial
proximity. This module uses scikit-learn's DBSCAN with `metric='haversine'`,
which computes great-circle distance correctly, at the cost of requiring
coordinates in RADIANS and an `eps` expressed in radians rather than
metres — see `meters_to_radians()` below for that conversion.

CHOOSING eps AND min_samples (NOT ARBITRARY)
-----------------------------------------------
Rather than picking eps/min_samples out of thin air, this module:
  1. Provides `k_distance_plot_data()` — computes the distance from every
     point to its k-th nearest neighbour (k = min_samples), sorted
     descending. The "elbow" in this curve is the standard heuristic
     (Ester et al. 1996) for a reasonable eps at that min_samples: points
     below the elbow are dense-region distances, points above are noise
     jumps.
  2. Provides `compare_dbscan_parameters()` — actually runs DBSCAN across
     the candidate eps/min_samples grid in config.py and reports how many
     clusters and how much noise each combination produces, so a value can
     be chosen by evidence rather than guesswork. config.py's
     DBSCAN_EPS_METERS / DBSCAN_MIN_SAMPLES defaults were themselves chosen
     this way against a development sample — re-run the comparison on your
     own data/sample size, since the right elbow shifts with density.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from config import (
    COL_LAT,
    COL_LON,
    DBSCAN_EPS_CANDIDATES_METERS,
    DBSCAN_EPS_METERS,
    DBSCAN_MIN_SAMPLES,
    DBSCAN_MIN_SAMPLES_CANDIDATES,
)

EARTH_RADIUS_M = 6_371_000  # mean Earth radius in metres, consistent with trajectory.py

COL_CLUSTER_ID = "cluster_id"


def meters_to_radians(meters: float) -> float:
    """Convert a great-circle distance in metres to radians (for haversine eps)."""
    return meters / EARTH_RADIUS_M


def k_distance_plot_data(df: pd.DataFrame, k: int) -> np.ndarray:
    """Compute each point's distance (metres) to its k-th nearest neighbour.

    Returns the distances sorted DESCENDING (the conventional orientation
    for a k-distance/elbow plot). Plot this in the notebook and look for
    the elbow — that distance is a reasonable eps candidate for this k
    (used as min_samples in DBSCAN).
    """
    coords_rad = np.radians(df[[COL_LAT, COL_LON]].values)
    nn = NearestNeighbors(n_neighbors=k, metric="haversine")
    nn.fit(coords_rad)
    distances_rad, _ = nn.kneighbors(coords_rad)

    k_distances_rad = distances_rad[:, -1]  # distance to the k-th neighbour
    k_distances_m = k_distances_rad * EARTH_RADIUS_M

    return np.sort(k_distances_m)[::-1]


def run_dbscan(
    df: pd.DataFrame,
    eps_meters: float = DBSCAN_EPS_METERS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> pd.DataFrame:
    """Run DBSCAN with haversine distance and assign a cluster_id per row.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain COL_LAT, COL_LON.
    eps_meters : float
        Maximum great-circle distance (metres) for two points to be
        considered neighbours.
    min_samples : int
        Minimum points required to form a dense region.

    Returns
    -------
    pd.DataFrame
        Original columns preserved, plus `cluster_id` (int). Noise points
        (DBSCAN label -1) are preserved with cluster_id == -1, NOT dropped
        — Part 6 explicitly requires identifying noise points, not
        discarding them silently.
    """
    df = df.copy()
    coords_rad = np.radians(df[[COL_LAT, COL_LON]].values)
    eps_rad = meters_to_radians(eps_meters)

    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    df[COL_CLUSTER_ID] = model.fit_predict(coords_rad)

    return df


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each cluster: point count, geographic center, taxi count.

    Cluster -1 (noise) is included as its own row so its size is visible,
    but is clearly labeled.
    """
    summary = (
        df.groupby(COL_CLUSTER_ID)
        .agg(
            n_points=(COL_CLUSTER_ID, "count"),
            center_lat=(COL_LAT, "mean"),
            center_lon=(COL_LON, "mean"),
        )
        .reset_index()
        .sort_values("n_points", ascending=False)
        .reset_index(drop=True)
    )
    summary["is_noise"] = summary[COL_CLUSTER_ID] == -1
    return summary


def compare_dbscan_parameters(
    df: pd.DataFrame,
    eps_candidates_m: list[float] = DBSCAN_EPS_CANDIDATES_METERS,
    min_samples_candidates: list[int] = DBSCAN_MIN_SAMPLES_CANDIDATES,
) -> pd.DataFrame:
    """Run DBSCAN across a grid of (eps, min_samples) candidates and report
    cluster count / noise fraction for each, so a choice can be justified
    by evidence rather than picked arbitrarily.

    NOTE: this evaluates every combination in the two candidate lists —
    on a full (non-sample) dataset this can be slow; run on a development
    sample first.
    """
    coords_rad = np.radians(df[[COL_LAT, COL_LON]].values)
    rows = []

    for eps_m in eps_candidates_m:
        eps_rad = meters_to_radians(eps_m)
        for min_samples in min_samples_candidates:
            model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
            labels = model.fit_predict(coords_rad)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            rows.append(
                {
                    "eps_meters": eps_m,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise_points": n_noise,
                    "pct_noise": 100 * n_noise / len(labels),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, not real T-Drive data): two tight
    # clusters of points plus one isolated noise point.
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=[39.900, 116.400], scale=0.0005, size=(20, 2))
    cluster_b = rng.normal(loc=[39.950, 116.450], scale=0.0005, size=(20, 2))
    noise_point = np.array([[39.700, 116.200]])
    coords = np.vstack([cluster_a, cluster_b, noise_point])

    test_fixture = pd.DataFrame({COL_LAT: coords[:, 0], COL_LON: coords[:, 1]})

    print("Running DBSCAN on a synthetic TEST FIXTURE (two clusters + 1 noise point):\n")
    result = run_dbscan(test_fixture, eps_meters=200.0, min_samples=5)
    print(cluster_summary(result))

    print("\nParameter comparison grid:")
    comparison = compare_dbscan_parameters(
        test_fixture, eps_candidates_m=[100, 200, 500], min_samples_candidates=[3, 5, 10]
    )
    print(comparison)

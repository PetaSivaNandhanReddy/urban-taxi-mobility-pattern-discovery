"""
test_route_and_departure.py
===========================
Automated unit tests for Phase 5 Historical Route and Departure-Time Analytics:
- Historical route candidate discovery (route_candidates.py)
- Evidence thresholds and path continuity verification
- Alternative-route comparison and recommendation (recommendation.py)
- Departure-time window evaluation and insufficient-data guards (departure_optimizer.py)
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
import departure_optimizer as do
import recommendation as rc
import route_candidates as rcan
import spatial as sp
import trajectory as tr


class TestRouteAndDepartureAnalytics(unittest.TestCase):
    """Test suite verifying route candidates, recommendations, and departure optimization."""

    def setUp(self):
        # Synthetic trajectory with multiple trips between CELL_A and CELL_C:
        # Route 1: A -> B -> C (takes 600 seconds, 3 trips)
        # Route 2: A -> D -> C (takes 900 seconds, 3 trips)
        # Route 3: A -> E (doesn't reach C)
        trips = []

        # Route 1 trips (A -> B -> C): 10 minutes (600s)
        for i in range(1, 4):
            t_base = pd.Timestamp(f"2008-02-04 08:{i*10:02d}:00")
            trips.extend([
                {
                    config.COL_TAXI_ID: str(i),
                    config.COL_TIMESTAMP: t_base,
                    "spatial_cell_id": "CELL_A",
                    "trip_id": f"trip_r1_{i}",
                    config.COL_LAT: 39.900,
                    config.COL_LON: 116.400,
                },
                {
                    config.COL_TAXI_ID: str(i),
                    config.COL_TIMESTAMP: t_base + pd.Timedelta(minutes=5),
                    "spatial_cell_id": "CELL_B",
                    "trip_id": f"trip_r1_{i}",
                    config.COL_LAT: 39.904,
                    config.COL_LON: 116.405,
                },
                {
                    config.COL_TAXI_ID: str(i),
                    config.COL_TIMESTAMP: t_base + pd.Timedelta(minutes=10),
                    "spatial_cell_id": "CELL_C",
                    "trip_id": f"trip_r1_{i}",
                    config.COL_LAT: 39.908,
                    config.COL_LON: 116.410,
                },
            ])

        # Route 2 trips (A -> D -> C): 15 minutes (900s)
        for i in range(4, 7):
            t_base = pd.Timestamp(f"2008-02-04 08:{i*5:02d}:00")
            trips.extend([
                {
                    config.COL_TAXI_ID: str(i),
                    config.COL_TIMESTAMP: t_base,
                    "spatial_cell_id": "CELL_A",
                    "trip_id": f"trip_r2_{i}",
                    config.COL_LAT: 39.900,
                    config.COL_LON: 116.400,
                },
                {
                    config.COL_TAXI_ID: str(i),
                    config.COL_TIMESTAMP: t_base + pd.Timedelta(minutes=8),
                    "spatial_cell_id": "CELL_D",
                    "trip_id": f"trip_r2_{i}",
                    config.COL_LAT: 39.904,
                    config.COL_LON: 116.408,
                },
                {
                    config.COL_TAXI_ID: str(i),
                    config.COL_TIMESTAMP: t_base + pd.Timedelta(minutes=15),
                    "spatial_cell_id": "CELL_C",
                    "trip_id": f"trip_r2_{i}",
                    config.COL_LAT: 39.908,
                    config.COL_LON: 116.410,
                },
            ])


        self.segmented_fixture = pd.DataFrame(trips)
        self.cell_lookup = sp.build_cell_lookup(self.segmented_fixture)

    def test_discover_route_candidates_sufficient_evidence(self):
        """Should discover both candidate routes with correct trip counts and medians."""
        result = rcan.discover_route_candidates(
            self.segmented_fixture,
            origin_cell="CELL_A",
            destination_cell="CELL_C",
            min_evidence_trips=2,
            cell_lookup=self.cell_lookup,
        )
        self.assertEqual(result["status"], "SUFFICIENT_EVIDENCE")
        candidates = result["candidate_routes"]
        self.assertEqual(len(candidates), 2)

        # Fastest route (A -> B -> C) has 600s median duration
        fastest = candidates.iloc[0]
        self.assertEqual(fastest["cell_path"], ("CELL_A", "CELL_B", "CELL_C"))
        self.assertEqual(fastest["median_duration_sec"], 600.0)
        self.assertEqual(fastest["n_trips"], 3)

        # Alternative route (A -> D -> C) has 900s median duration
        second = candidates.iloc[1]
        self.assertEqual(second["cell_path"], ("CELL_A", "CELL_D", "CELL_C"))
        self.assertEqual(second["median_duration_sec"], 900.0)
        self.assertEqual(second["n_trips"], 3)

    def test_discover_route_candidates_insufficient_evidence(self):
        """Should return INSUFFICIENT_EVIDENCE when observations fail threshold."""
        result = rcan.discover_route_candidates(
            self.segmented_fixture,
            origin_cell="CELL_A",
            destination_cell="CELL_C",
            min_evidence_trips=10,  # Requiring 10 trips (only 3 exist per route)
            cell_lookup=self.cell_lookup,
        )
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(len(result["candidate_routes"]), 0)

    def test_compare_route_alternatives_recommendation(self):
        """Should compare competing alternatives and recommend the faster route."""
        cand_result = rcan.discover_route_candidates(
            self.segmented_fixture,
            origin_cell="CELL_A",
            destination_cell="CELL_C",
            min_evidence_trips=2,
            cell_lookup=self.cell_lookup,
        )
        rec = rc.compare_route_alternatives(cand_result, min_evidence_trips=2)

        self.assertTrue(rec["has_recommendation"])
        self.assertEqual(rec["status"], "RECOMMENDED")
        self.assertEqual(rec["recommended_route"]["cell_path"], ("CELL_A", "CELL_B", "CELL_C"))
        # 900s - 600s = 300s duration advantage
        self.assertEqual(rec["duration_advantage_sec"], 300.0)
        self.assertAlmostEqual(rec["duration_advantage_pct"], 33.3, places=1)
        self.assertTrue(len(rec["limitations"]) > 0)

    def test_compare_route_alternatives_single_route_only(self):
        """When only one route meets evidence threshold, should return SINGLE_ROUTE_ONLY."""
        # Filter fixture to only route 1
        r1_only = self.segmented_fixture[self.segmented_fixture["trip_id"].str.startswith("trip_r1_")].copy()
        cand_result = rcan.discover_route_candidates(
            r1_only,
            origin_cell="CELL_A",
            destination_cell="CELL_C",
            min_evidence_trips=2,
            cell_lookup=self.cell_lookup,
        )
        rec = rc.compare_route_alternatives(cand_result, min_evidence_trips=2)
        self.assertFalse(rec["has_recommendation"])
        self.assertEqual(rec["status"], "SINGLE_ROUTE_ONLY")
        self.assertEqual(rec["duration_advantage_sec"], 0.0)

    def test_departure_optimizer_window_evaluation(self):
        """Should evaluate departure windows and report observed differences."""
        cand_result = rcan.discover_route_candidates(
            self.segmented_fixture,
            origin_cell="CELL_A",
            destination_cell="CELL_C",
            min_evidence_trips=1,
            cell_lookup=self.cell_lookup,
        )
        journeys = cand_result["raw_journeys"]

        # Target 08:30: trips exist around 08:10-08:30
        dep_eval = do.evaluate_departure_windows(
            journeys,
            target_departure_time="2008-02-04 08:25:00",
            window_shifts_minutes=[-10, 0, 10],
            half_width_minutes=15,
            min_trips_per_window=2,
        )
        self.assertEqual(dep_eval["status"], "SUFFICIENT_DATA")
        self.assertTrue("windows_summary" in dep_eval)
        self.assertIsNotNone(dep_eval["best_historical_window"])
        self.assertTrue(len(dep_eval["limitations"]) > 0)

    def test_departure_optimizer_insufficient_data(self):
        """Should guard against low sample sizes with INSUFFICIENT_DATA status."""
        empty_journeys = pd.DataFrame()
        dep_eval = do.evaluate_departure_windows(
            empty_journeys,
            target_departure_time="2008-02-04 08:30:00",
            min_trips_per_window=3,
        )
        self.assertEqual(dep_eval["status"], "INSUFFICIENT_DATA")


if __name__ == "__main__":
    unittest.main()

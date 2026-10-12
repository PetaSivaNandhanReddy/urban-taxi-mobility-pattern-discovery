"""
test_spatial_temporal.py
========================
Automated unit tests for Phase 2 Spatial and Temporal Analysis:
- Spatial discretization and deterministic cell labeling
- Cell coordinate decoding and bounding box lookup
- Adjacency and path continuity verification
- Temporal activity breakdown: by hour, segment, date, and normalized day-type
- DBSCAN parameter comparison validation
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import clustering as cl
import config
import sequential as sq
import spatial as sp
import temporal as tm


class TestSpatialAndTemporalAnalysis(unittest.TestCase):
    """Test suite verifying spatial discretization, clustering, and temporal analysis."""

    def setUp(self):
        # 4 points along a line ~500m apart, plus 1 point 50km away (disconnected)
        self.points_df = pd.DataFrame({
            config.COL_TAXI_ID: ["1", "1", "1", "1", "2"],
            config.COL_TIMESTAMP: pd.to_datetime([
                "2008-02-02 08:00:00",  # Saturday
                "2008-02-02 08:05:00",
                "2008-02-04 14:00:00",  # Monday
                "2008-02-04 14:05:00",
                "2008-02-05 22:00:00",  # Tuesday
            ]),
            config.COL_LON: [116.400, 116.405, 116.410, 116.415, 117.000],
            config.COL_LAT: [39.900, 39.904, 39.908, 39.912, 40.500],
        })

    def test_assign_spatial_cells_deterministic(self):
        """Cell assignments must be deterministic given identical inputs."""
        cell_df1 = sp.assign_spatial_cells(self.points_df, cell_size_meters=500.0)
        cell_df2 = sp.assign_spatial_cells(self.points_df, cell_size_meters=500.0)
        self.assertTrue(sp.COL_CELL_ID in cell_df1.columns)
        self.assertEqual(cell_df1[sp.COL_CELL_ID].tolist(), cell_df2[sp.COL_CELL_ID].tolist())

    def test_cell_lookup_and_decode(self):
        """Cell lookup must accurately return centroid coordinates."""
        cell_df = sp.assign_spatial_cells(self.points_df, cell_size_meters=500.0)
        lookup = sp.build_cell_lookup(cell_df)
        first_cell = cell_df.iloc[0][sp.COL_CELL_ID]
        coords = sp.decode_cell(first_cell, lookup)
        # First cell contains point 0 (39.900, 116.400) and point 1 (39.904, 116.405) -> mean lat is 39.902
        self.assertAlmostEqual(coords["centroid_lat"], 39.902, places=3)
        self.assertAlmostEqual(coords["centroid_lon"], 116.4025, places=3)


    def test_cell_adjacency_and_continuity(self):
        """Path continuity must detect continuous paths and reject disconnected jumps."""
        cell_df = sp.assign_spatial_cells(self.points_df, cell_size_meters=500.0)
        lookup = sp.build_cell_lookup(cell_df)
        cells = cell_df[sp.COL_CELL_ID].tolist()

        # First 4 cells are adjacent (< 1000m apart)
        is_cont, dists = sp.verify_path_continuity(cells[:4], lookup, max_gap_m=1000.0)
        self.assertTrue(is_cont)
        self.assertEqual(len(dists), 3)

        # Sequence jumping from cell 1 to cell 5 (50km away) must fail continuity
        is_cont_jump, _ = sp.verify_path_continuity([cells[0], cells[4]], lookup, max_gap_m=1000.0)
        self.assertFalse(is_cont_jump)

    def test_temporal_activity_by_date(self):
        """Date-level summary must report points, active taxis, and day types."""
        date_summary = tm.activity_by_date(self.points_df)
        self.assertEqual(len(date_summary), 3)  # 3 distinct dates: Feb 2, Feb 4, Feb 5
        self.assertTrue("day_name" in date_summary.columns)
        self.assertTrue("day_type" in date_summary.columns)
        self.assertEqual(date_summary.iloc[0]["day_type"], "weekend")
        self.assertEqual(date_summary.iloc[1]["day_type"], "weekday")

    def test_normalized_day_type_activity(self):
        """Activity by day-type must normalize points per observed calendar day."""
        labeled = sq.add_temporal_labels(self.points_df)
        day_type_summary = tm.activity_by_day_type(labeled, normalize=True)
        self.assertTrue("points_per_day" in day_type_summary.columns)
        self.assertTrue("n_days_observed" in day_type_summary.columns)

        weekend_row = day_type_summary[day_type_summary["day_type"] == "weekend"].iloc[0]
        weekday_row = day_type_summary[day_type_summary["day_type"] == "weekday"].iloc[0]
        # Feb 2 (1 weekend day, 2 points -> 2.0 pts/day)
        self.assertEqual(weekend_row["n_days_observed"], 1)
        self.assertEqual(weekend_row["points_per_day"], 2.0)
        # Feb 4 and 5 (2 weekdays, 3 points -> 1.5 pts/day)
        self.assertEqual(weekday_row["n_days_observed"], 2)
        self.assertEqual(weekday_row["points_per_day"], 1.5)

    def test_dbscan_parameter_comparison_schema(self):
        """DBSCAN parameter grid comparison must return expected columns."""
        comp_df = cl.compare_dbscan_parameters(
            self.points_df, eps_candidates_m=[500.0], min_samples_candidates=[2]
        )
        expected_cols = {"eps_meters", "min_samples", "n_clusters", "n_noise_points", "pct_noise"}
        self.assertTrue(expected_cols.issubset(set(comp_df.columns)))


if __name__ == "__main__":
    unittest.main()

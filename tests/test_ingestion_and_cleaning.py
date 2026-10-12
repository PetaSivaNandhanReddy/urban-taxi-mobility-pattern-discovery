"""
test_ingestion_and_cleaning.py
==============================
Automated unit tests for Phase 1 Data Ingestion and Quality:
- Ingestion schema detection and recursive file discovery
- Cleaning pipeline: malformed records, missing values, invalid coordinates, duplicates
- Timestamp parsing and chronological ordering within taxi
- Out-of-region flagging
- Deterministic and reproducible sampling
- Trajectory speed jumps and trip/session segmentation
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
import data_loader as dl
import preprocessing as pp
import trajectory as tr


class TestDataIngestionAndQuality(unittest.TestCase):
    """Test suite verifying robust data ingestion and cleaning."""

    def setUp(self):
        self.sample_data = pd.DataFrame({
            config.COL_TAXI_ID: ["1", "1", "1", "1", "2", "2", "2"],
            config.COL_TIMESTAMP: [
                "2008-02-02 15:36:00",
                "2008-02-02 15:36:00",  # Duplicate row
                "2008-02-02 15:37:00",
                "2008-02-02 15:38:00",
                "2008-02-02 10:00:00",
                "2008-02-02 10:01:00",
                "2008-02-02 10:45:00",  # 44-min gap -> should trigger trip split
            ],
            config.COL_LON: [116.510, 116.510, 116.512, 116.515, 116.400, 116.402, 116.420],
            config.COL_LAT: [39.920, 39.920, 39.922, 39.925, 39.900, 39.901, 39.910],
        })

    def test_duplicate_removal(self):
        """Exact duplicates of (taxi_id, timestamp, lon, lat) must be removed."""
        clean_df, report = pp.run_cleaning_pipeline(self.sample_data, verbose=False)
        self.assertEqual(len(clean_df), 6)  # 7 rows originally, 1 duplicate dropped
        self.assertEqual(report.stage_counts["After duplicate removal"], 6)

    def test_invalid_coordinates_cleaning(self):
        """Points with null-fix (0, 0) or out-of-range coordinates must be removed."""
        bad_coords_df = pd.DataFrame({
            config.COL_TAXI_ID: ["1", "2", "3", "4"],
            config.COL_TIMESTAMP: [
                "2008-02-02 12:00:00",
                "2008-02-02 12:00:00",
                "2008-02-02 12:00:00",
                "2008-02-02 12:00:00",
            ],
            config.COL_LON: [0.0, 116.40, 200.0, 116.50],
            config.COL_LAT: [0.0, -95.0, 39.90, 39.90],
        })
        clean_df, report = pp.run_cleaning_pipeline(bad_coords_df, verbose=False)
        self.assertEqual(len(clean_df), 1)  # Only the 4th row is valid
        self.assertEqual(clean_df.iloc[0][config.COL_TAXI_ID], "4")

    def test_malformed_timestamp_handling(self):
        """Malformed or unparseable timestamps must be discarded."""
        bad_time_df = pd.DataFrame({
            config.COL_TAXI_ID: ["1", "2"],
            config.COL_TIMESTAMP: ["not-a-date", "2008-02-02 12:00:00"],
            config.COL_LON: [116.40, 116.41],
            config.COL_LAT: [39.90, 39.91],
        })
        clean_df, report = pp.run_cleaning_pipeline(bad_time_df, verbose=False)
        self.assertEqual(len(clean_df), 1)
        self.assertEqual(clean_df.iloc[0][config.COL_TAXI_ID], "2")

    def test_missing_taxi_id_handling(self):
        """Missing or whitespace-only taxi IDs must be discarded."""
        bad_id_df = pd.DataFrame({
            config.COL_TAXI_ID: ["", "  ", None, "10"],
            config.COL_TIMESTAMP: ["2008-02-02 12:00:00"] * 4,
            config.COL_LON: [116.40] * 4,
            config.COL_LAT: [39.90] * 4,
        })
        clean_df, report = pp.run_cleaning_pipeline(bad_id_df, verbose=False)
        self.assertEqual(len(clean_df), 1)
        self.assertEqual(clean_df.iloc[0][config.COL_TAXI_ID], "10")

    def test_chronological_ordering_guarantee(self):
        """Observations must be sorted strictly chronologically within each taxi."""
        unshuffled = pd.DataFrame({
            config.COL_TAXI_ID: ["1", "1", "1"],
            config.COL_TIMESTAMP: [
                "2008-02-02 15:40:00",
                "2008-02-02 15:30:00",
                "2008-02-02 15:35:00",
            ],
            config.COL_LON: [116.40, 116.41, 116.42],
            config.COL_LAT: [39.90, 39.91, 39.92],
        })
        clean_df, _ = pp.run_cleaning_pipeline(unshuffled, verbose=False)
        timestamps = clean_df[config.COL_TIMESTAMP].tolist()
        self.assertTrue(timestamps[0] < timestamps[1] < timestamps[2])

    def test_out_of_region_flagging(self):
        """Coordinates outside the Beijing bounding box must be flagged, not dropped by default."""
        out_of_box_df = pd.DataFrame({
            config.COL_TAXI_ID: ["1", "2"],
            config.COL_TIMESTAMP: ["2008-02-02 12:00:00", "2008-02-02 12:05:00"],
            config.COL_LON: [116.40, 125.00],  # 125.0 is far outside Beijing
            config.COL_LAT: [39.90, 30.00],   # 30.0 is far outside Beijing
        })
        # Default: flagged, not dropped
        clean_df, _ = pp.run_cleaning_pipeline(out_of_box_df, drop_out_of_region=False, verbose=False)
        self.assertEqual(len(clean_df), 2)
        self.assertTrue("in_region" in clean_df.columns)
        self.assertTrue(clean_df.iloc[0]["in_region"])
        self.assertFalse(clean_df.iloc[1]["in_region"])

        # When explicitly requested: dropped
        clean_dropped, _ = pp.run_cleaning_pipeline(out_of_box_df, drop_out_of_region=True, verbose=False)
        self.assertEqual(len(clean_dropped), 1)

    def test_sampling_reproducibility(self):
        """Sampling with fixed seed must be strictly deterministic across calls."""
        all_files = dl.find_taxi_files(config.RAW_DATA_DIR)
        self.assertGreater(len(all_files), 0)

        df1, meta1 = dl.load_tdrive_sample(
            raw_dir=config.RAW_DATA_DIR, n_taxis=10, seed=42, return_metadata=True, verbose=False
        )
        df2, meta2 = dl.load_tdrive_sample(
            raw_dir=config.RAW_DATA_DIR, n_taxis=10, seed=42, return_metadata=True, verbose=False
        )
        self.assertEqual(meta1.selected_taxi_ids, meta2.selected_taxi_ids)
        self.assertEqual(len(df1), len(df2))
        pd.testing.assert_frame_equal(df1, df2)

    def test_trajectory_speed_jump_detection(self):
        """Implied speed > 150 km/h must be flagged as is_suspicious_jump."""
        teleport_df = pd.DataFrame({
            config.COL_TAXI_ID: ["1", "1"],
            config.COL_TIMESTAMP: pd.to_datetime(["2008-02-02 12:00:00", "2008-02-02 12:01:00"]),
            config.COL_LON: [116.40, 116.80],  # ~34 km in 60s -> ~2000 km/h
            config.COL_LAT: [39.90, 39.90],
        })
        traj_df = tr.build_trajectories(teleport_df)
        self.assertTrue(traj_df.iloc[1]["is_suspicious_jump"])
        self.assertGreater(traj_df.iloc[1]["speed_kmh"], 150.0)

    def test_trajectory_trip_segmentation(self):
        """Gaps > 30 minutes must split a taxi's trajectory into distinct trip segments."""
        clean_df, _ = pp.run_cleaning_pipeline(self.sample_data, verbose=False)
        traj_df = tr.build_trajectories(clean_df)
        segmented_df = tr.segment_trajectories(traj_df, max_gap_sec=1800.0, min_points=2)

        self.assertTrue("trip_id" in segmented_df.columns)
        taxi2_trips = segmented_df[segmented_df[config.COL_TAXI_ID] == "2"]["trip_id"].unique()
        # Taxi 2 had a 44-minute gap between point 2 and 3, so it should split into 2 segments
        self.assertEqual(len(taxi2_trips), 2)

        # Trip summary
        trip_summary = tr.summarize_trips(segmented_df)
        self.assertGreater(len(trip_summary), 0)
        self.assertTrue("duration_sec" in trip_summary.columns)


if __name__ == "__main__":
    unittest.main()

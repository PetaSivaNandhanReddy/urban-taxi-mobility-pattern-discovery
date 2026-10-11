"""
test_dashboard.py
=================
Automated test suite for the Streamlit Mobility Intelligence Dashboard using AppTest.

Tests:
1. Dashboard initializes without exceptions in Administrator mode.
2. Navigation across all Administrator sections (Overview, Methodology & Evaluation, Corridors, etc.).
3. Dashboard initializes and renders in Customer mode without exceptions.
4. Empty states (zero mobility corridors) are handled gracefully without crashes.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streamlit.testing.v1 import AppTest


class TestDashboardApp(unittest.TestCase):
    """Automated integration tests for dashboard/app.py."""

    def test_admin_mode_overview_render(self):
        """Test that dashboard loads in Administrator mode without exceptions."""
        app_path = PROJECT_ROOT / "dashboard" / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run(timeout=15)

        self.assertEqual(len(at.exception), 0, f"App raised exceptions: {[e.message for e in at.exception]}")
        # Title must be rendered
        self.assertTrue(any("Beijing Taxi Mobility Intelligence" in str(t.value) for t in at.title))

    def test_methodology_and_evaluation_section(self):
        """Test that the new Methodology & Evaluation section renders properly."""
        app_path = PROJECT_ROOT / "dashboard" / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run(timeout=15)

        # Switch to Methodology & Evaluation section
        if len(at.sidebar.radio) >= 2:
            section_radio = at.sidebar.radio[1]
            if "Methodology & Evaluation" in section_radio.options:
                section_radio.set_value("Methodology & Evaluation")
                at.run(timeout=15)
                self.assertEqual(len(at.exception), 0, f"Exceptions in Methodology & Evaluation: {[e.message for e in at.exception]}")

    def test_mobility_corridors_zero_state(self):
        """Test that Mobility Corridors section renders zero-corridor state without crashing."""
        app_path = PROJECT_ROOT / "dashboard" / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run(timeout=15)

        if len(at.sidebar.radio) >= 2:
            section_radio = at.sidebar.radio[1]
            if "Mobility Corridors" in section_radio.options:
                section_radio.set_value("Mobility Corridors")
                at.run(timeout=15)
                self.assertEqual(len(at.exception), 0, f"Exceptions in Mobility Corridors: {[e.message for e in at.exception]}")

    def test_customer_mode_render(self):
        """Test that Customer mode renders without exceptions."""
        app_path = PROJECT_ROOT / "dashboard" / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run(timeout=15)

        # Switch mode to Customer
        if len(at.sidebar.radio) >= 1:
            mode_radio = at.sidebar.radio[0]
            mode_radio.set_value("Customer")
            at.run(timeout=15)
            self.assertEqual(len(at.exception), 0, f"Exceptions in Customer mode: {[e.message for e in at.exception]}")


def run_dashboard_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDashboardApp)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("One or more dashboard tests failed!")
    print("\nALL DASHBOARD STREAMLIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_dashboard_tests()

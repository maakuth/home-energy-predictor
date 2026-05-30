"""
Pytest configuration for test isolation.

This module ensures that tests run in an isolated environment without
affecting production files (hepo.db, optimization_plan.json, future_predictions.json, etc).
"""

import os
import tempfile
import shutil
import pytest


@pytest.fixture(autouse=True)
def isolated_test_env(monkeypatch, tmp_path):
    """
    Automatically provides test isolation for all tests.
    
    - Creates a temporary directory for the test
    - Sets environment variables to use test-specific paths
    - Restores original environment after the test
    
    This ensures tests don't overwrite production files even when run
    in the same working directory.
    """
    # Create a test-specific temporary directory
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir(exist_ok=True)
    
    # Set environment variables to point to test directories/files
    monkeypatch.setenv("TEST_DB_PATH", str(test_dir / "test_hepo.db"))
    monkeypatch.setenv("TEST_PREDICTIONS_FILE", str(test_dir / "future_predictions.json"))
    monkeypatch.setenv("TEST_PLAN_FILE", str(test_dir / "optimization_plan.json"))
    monkeypatch.setenv("TEST_SARIMA_FILE", str(test_dir / "sarimax_predictions.json"))
    
    # Also set for backward compatibility with code that might check these
    monkeypatch.setenv("DB_PATH", str(test_dir / "test_hepo.db"))
    
    # Yield control to the test
    yield {
        "test_dir": test_dir,
        "db_path": str(test_dir / "test_hepo.db"),
        "predictions_file": str(test_dir / "future_predictions.json"),
        "plan_file": str(test_dir / "optimization_plan.json"),
        "sarima_file": str(test_dir / "sarimax_predictions.json"),
    }
    
    # Cleanup happens automatically when tmp_path is garbage collected

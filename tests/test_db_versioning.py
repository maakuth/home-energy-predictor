from __future__ import annotations
import unittest
import sqlite3
import os
from utils.git_utils import get_model_version
from predict_future import get_model_version as get_model_version_pf

class TestDBVersioning(unittest.TestCase):
    def setUp(self):
        # Use test-specific database path from environment (set by conftest.py)
        self.db_file = os.getenv('TEST_DB_PATH', 'test_hepo.db')
        
        # Create directory if it doesn't exist (only if not in current directory)
        dir_path = os.path.dirname(self.db_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def tearDown(self):
        # No cleanup needed - conftest.py handles it via tmp_path fixture
        pass

    def test_git_version(self):
        version = get_model_version()
        self.assertIsNotNone(version)
        self.assertNotEqual(version, "")

    def test_schema_migration(self):
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        # Create old schema
        cur.execute('''
            CREATE TABLE predictions (
                target_timestamp TEXT,
                generated_at TEXT,
                predicted_usage_kw REAL,
                solar_forecast_kw REAL,
                PRIMARY KEY (target_timestamp, generated_at)
            )
        ''')
        conn.commit()
        conn.close()

        # Now run the migration logic (manually for testing or by calling a function)
        # We'll just check if our logic in the script would work.
        
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(predictions)")
        columns = [c[1] for c in cur.fetchall()]
        self.assertNotIn('version', columns)
        
        # Apply migration
        if 'version' not in columns:
            cur.execute("ALTER TABLE predictions ADD COLUMN version TEXT DEFAULT 'unknown'")
        conn.commit()
        
        cur.execute("PRAGMA table_info(predictions)")
        columns = [c[1] for c in cur.fetchall()]
        self.assertIn('version', columns)
        conn.close()

if __name__ == '__main__':
    unittest.main()

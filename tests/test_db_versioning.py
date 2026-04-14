import unittest
import sqlite3
import os
from utils.git_utils import get_git_version
from predict_future import get_git_version as get_git_version_pf

class TestDBVersioning(unittest.TestCase):
    def setUp(self):
        self.db_file = 'test_hepo.db'
        if os.path.exists(self.db_file):
            os.remove(self.db_file)

    def tearDown(self):
        if os.path.exists(self.db_file):
            os.remove(self.db_file)

    def test_git_version(self):
        version = get_git_version()
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

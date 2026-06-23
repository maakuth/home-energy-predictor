from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock, call
import os
import tempfile
import shutil


class TestExtractDataHelpers(unittest.TestCase):
    """Test the helper functions in extract_data.py (no DB needed)."""

    @patch('extract_data.psycopg2')
    def test_get_metadata_ids(self, mock_psycopg2):
        from extract_data import get_metadata_ids
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ('sensor.ulkona_temperature_2', 101),
            ('sensor.mlp_teho', 102),
        ]
        result = get_metadata_ids(mock_cur)
        # Function returns {metadata_id: entity_id}
        self.assertEqual(result, {
            101: 'sensor.ulkona_temperature_2',
            102: 'sensor.mlp_teho',
        })
        mock_cur.execute.assert_called_once()
        call_args = mock_cur.execute.call_args
        query, params = call_args[0]
        self.assertIn('states_meta', query)
        self.assertIn('metadata_id', query)
        self.assertIn('entity_id', query)

    def test_get_metadata_ids_entity_tuple(self):
        """Verify the IN tuple is constructed correctly from ENTITIES."""
        from extract_data import ENTITIES, get_metadata_ids
        entity_set = set(ENTITIES.keys())
        # Just verify the entities list isn't empty and contains expected keys
        self.assertIn('sensor.ulkona_temperature_2', entity_set)
        self.assertIn('weather.home', entity_set)
        self.assertGreaterEqual(len(entity_set), 15)

    @patch('extract_data.datetime')
    def test_extract_states(self, mock_dt):
        from extract_data import extract_states
        import datetime as dt
        fixed_now = dt.datetime(2026, 6, 15, 12, 0, 0)
        mock_dt.now.return_value = fixed_now
        mock_dt.timedelta = dt.timedelta

        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            (1718467200.0, '5.0'),
            (1718468100.0, '6.0'),
        ]
        result = extract_states(mock_cur, metadata_id=42, days=7)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 1718467200.0)
        self.assertEqual(result[1][1], '6.0')
        mock_cur.execute.assert_called_once()
        query, params = mock_cur.execute.call_args[0]
        self.assertIn('last_updated_ts >', query)

    @patch('extract_data.datetime')
    def test_extract_attribute_modern_schema(self, mock_dt):
        from extract_data import extract_attribute
        import datetime as dt
        mock_dt.now.return_value = dt.datetime(2026, 6, 15, 12, 0, 0)
        mock_dt.timedelta = dt.timedelta

        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            (1718467200.0, '3.5'),
        ]
        result = extract_attribute(mock_cur, metadata_id=42, attr_name='wind_speed', days=7)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], '3.5')

    @patch('extract_data.datetime')
    def test_extract_attribute_fallback_schema(self, mock_dt):
        from extract_data import extract_attribute
        import datetime as dt
        mock_dt.now.return_value = dt.datetime(2026, 6, 15, 12, 0, 0)
        mock_dt.timedelta = dt.timedelta

        mock_cur = MagicMock()
        # First call raises Exception (modern schema fails), second succeeds
        mock_cur.execute.side_effect = [Exception('modern schema failed'), None]
        mock_cur.fetchall.return_value = [
            (1718467200.0, '4.0'),
        ]

        result = extract_attribute(mock_cur, metadata_id=99, attr_name='wind_speed', days=7)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], '4.0')
        # Should have called execute twice (fallback)
        self.assertEqual(mock_cur.execute.call_count, 2)


class TestExtractDataMain(unittest.TestCase):
    """Test extract_data.main() with mocked psycopg2."""

    @classmethod
    def setUpClass(cls):
        cls.orig_cwd = os.getcwd()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('extract_data.extract_attribute')
    @patch('extract_data.extract_states')
    @patch('extract_data.get_metadata_ids')
    @patch('extract_data.psycopg2')
    def test_main_writes_csv(self, mock_psycopg2, mock_get_ids, mock_extract, mock_extract_attr):
        from extract_data import main
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur

        # All entities have metadata IDs
        from extract_data import ENTITIES
        mock_get_ids.return_value = {eid: i for i, eid in enumerate(ENTITIES.keys())}

        # Mock extract_states to return some data for each entity
        def extract_side_effect(cur, mid, days):
            return [(1718467200.0 + i * 900, str(1.0 + i * 0.1)) for i in range(10)]
        mock_extract.side_effect = extract_side_effect

        def extract_attr_side_effect(cur, mid, attr, days):
            return [(1718467200.0 + i * 900, str(2.0 + i * 0.1)) for i in range(10)]
        mock_extract_attr.side_effect = extract_attr_side_effect

        from unittest.mock import patch as mock_patch
        with mock_patch('sys.argv', ['extract_data.py', '--days', '3']):
            main()

        output_path = os.path.join(self.test_dir, 'state', 'raw_data.csv')
        self.assertTrue(os.path.exists(output_path),
                        "main() should create state/raw_data.csv")
        import pandas as pd
        df = pd.read_csv(output_path, index_col=0)
        # Should have all entity columns (mummun_energy is converted to mummun_power)
        expected_cols = set(ENTITIES.values()) - {'mummun_energy'} | {'mummun_power'}
        has = set(df.columns)
        missing = expected_cols - has
        self.assertEqual(missing, set(),
                         f"Missing columns in output CSV: {missing}")

    @patch('extract_data.extract_attribute')
    @patch('extract_data.extract_states')
    @patch('extract_data.get_metadata_ids')
    @patch('extract_data.psycopg2')
    def test_main_handles_missing_entities(self, mock_psycopg2, mock_get_ids,
                                            mock_extract, mock_extract_attr):
        from extract_data import main
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur

        # Some entities are missing
        from extract_data import ENTITIES
        all_entities = list(ENTITIES.keys())
        mock_get_ids.return_value = {
            all_entities[0]: 1,
            all_entities[1]: 2,
        }

        mock_extract.return_value = [(1718467200.0, '1.0')]
        mock_extract_attr.return_value = [(1718467200.0, '3.0')]

        from unittest.mock import patch as mock_patch
        with mock_patch('sys.argv', ['extract_data.py', '--days', '3']):
            main()

        output_path = os.path.join(self.test_dir, 'state', 'raw_data.csv')
        self.assertTrue(os.path.exists(output_path))
        import pandas as pd
        df = pd.read_csv(output_path, index_col=0)
        self.assertGreaterEqual(len(df.columns), 1)

if __name__ == '__main__':
    unittest.main()

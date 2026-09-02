from pathlib import Path


def test_weekly_pipeline_uses_dedicated_training_data_files():
    script = Path('run_weekly.sh').read_text()

    assert 'state/raw_data_weekly.csv' in script
    assert 'state/processed_data_weekly.csv' in script
    assert 'extract_data.py --days 730 --output state/raw_data_weekly.csv' in script
    assert 'process_data.py --input state/raw_data_weekly.csv --output state/processed_data_weekly.csv' in script
    assert 'train_model.py --data-path state/processed_data_weekly.csv' in script
    assert 'train_sarima.py --days 30 --data-path state/processed_data_weekly.csv' in script

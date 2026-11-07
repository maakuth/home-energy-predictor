# Test selections

- Routine iteration: `venv/bin/python3 -m pytest -k 'not slow'`
- Full regression suite: `venv/bin/python3 -m pytest`
- Explicit production-sized SARIMA cold-fit diagnostic:
  `venv/bin/python3 -m pytest tests/test_sarimax.py -m cold_fit_benchmark --run-cold-fit-benchmark --durations=0`

The cold-fit diagnostic is skipped unless the explicit flag is supplied, even
when selecting `-m slow`. It took 7,360 seconds in the November 7 validation run.
It tests recovery of an easy synthetic seasonal signal, not relative quality on
household data. Its JUnit properties report SARIMA, historical-mean and
repeat-yesterday MAEs; repeat-yesterday is essentially perfect on this fixture.

Routine coverage instead uses a real statsmodels fit with eight-slot seasons,
80 training samples and 16 held-out samples. This reduces the state dimension
from 194 to 18. It checks optimizer convergence, non-fallback prediction,
timestamp alignment, finite outputs and confidence-interval ordering, with a
100-iteration fit limit. A separate mocked-constructor test verifies that
production still requests `(1,1,1) × (1,1,0,96)`. Neither test changes production
model settings or claims to establish real-world forecast superiority.

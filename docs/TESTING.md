# Test selections

- Routine iteration: use the parallel command below with `-k 'not slow'`.
- Full regression suite: use the parallel command below without `-k 'not slow'`.
- Explicit production-sized SARIMA cold-fit diagnostic:
  `venv/bin/python3 -m pytest tests/test_sarimax.py -m cold_fit_benchmark --run-cold-fit-benchmark --durations=0`

## Parallel execution

Use parallel execution by default for routine and full regression runs. Install
the dependencies in `requirements.txt` to enable pytest-xdist. To run the
routine selection in four worker processes:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  venv/bin/python3 -m pytest -n 4 --dist worksteal -k 'not slow' --durations=25
```

The thread limits reduce CPU oversubscription from numerical libraries; they
apply only to this command. Adjust the worker count to suit available CPU and
memory, especially on a machine running the live pipeline. Remove `-k 'not slow'`
for the full regression suite; the cold-fit diagnostic remains opt-in. Omit
`-n` to run serially when debugging a failure or comparing against a baseline.
Pytest's reported durations help identify expensive tests.

For reference, one November 2026 run of the routine selection took 98 seconds
serially, 60 seconds with two workers and 37 seconds with four workers. These
are single-run measurements, not guaranteed speeds; the parallel commands used
the thread limits above.

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

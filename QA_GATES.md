# QA Gates

## Local execution

Run the lightweight onboarding UX gate suite locally:

```bash
. .venv/bin/activate  # optional if using virtualenv
python scripts/ux_quality_gates.py
python scripts/ux_quality_gates.py --json
```

Run the onboarding smoke test explicitly:

```bash
. .venv/bin/activate
python -m unittest tests.test_onboarding_smoke_e2e
```

Run the UX gate assertion tests directly:

```bash
. .venv/bin/activate
python -m unittest tests.test_ux_quality_gates
```

## CI workflow

Use the same commands in CI (or equivalent pytest invocation in your project pipeline) to keep behavior deterministic:

```bash
python scripts/ux_quality_gates.py
python -m unittest tests.test_onboarding_smoke_e2e tests.test_ux_quality_gates
```

Both paths are fully deterministic and do not perform network calls.

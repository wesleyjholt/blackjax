# Local BlackJAX SMC Layout

## Dependency

- Primary dependency: local editable BlackJAX checkout
  - path: `/Users/holtw/Documents/mydocs/software/blackjax`
  - install: `pip install -e /Users/holtw/Documents/mydocs/software/blackjax`

Use this checkout as the source of truth for SMC behavior.

## Key Modules

- `blackjax/smc/base.py`
  - generic SMC state, info, weighting, resampling decision, mutation flow
- `blackjax/smc/from_mcmc.py`
  - adapts MCMC kernels into particle mutation kernels
- `blackjax/smc/tempered.py`
  - tempered SMC state and fixed-temperature-step logic
- `blackjax/smc/adaptive_tempered.py`
  - adaptive temperature selection from ESS targets
- `blackjax/smc/ess.py`
  - ESS utilities and ESS-root solver logic
- `blackjax/smc/resampling.py`
  - concrete resamplers and higher-level resampling strategies
- `blackjax/smc/waste_free.py`
  - waste-free mutation strategy

## Useful Tests

- `tests/smc/test_smc.py`
- `tests/smc/test_tempered_smc.py`
- `tests/smc/test_smc_ess.py`
- `tests/smc/test_waste_free_smc.py`
- `tests/smc/test_kernel_compatibility.py`

## Practical Notes

- Treat JAX compatibility as a hard constraint.
- Prefer reusable policy abstractions over algorithm-specific branching.
- If SMC behavior changes, update tests, docstrings, and at least one example in the local checkout.

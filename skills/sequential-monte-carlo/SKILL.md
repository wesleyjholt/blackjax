---
name: sequential-monte-carlo
description: Build, debug, extend, and explain sequential Monte Carlo workflows in JAX, especially BlackJAX tempered SMC, adaptive tempering, ESS-based resampling, mutation kernels, and particle approximations. Use when implementing or reviewing SMC algorithms, adding new SMC strategies, tracing how ESS drives tempering or resampling, writing SMC tests/examples, or working in the local BlackJAX checkout at /Users/holtw/Documents/mydocs/software/blackjax.
---

# Sequential Monte Carlo

Use the local BlackJAX checkout as the default implementation target for code and examples.

## Dependencies

- Depend on the local BlackJAX checkout at `/Users/holtw/Documents/mydocs/software/blackjax`.
- Install it in editable mode when code needs to run against this exact version:
  `pip install -e /Users/holtw/Documents/mydocs/software/blackjax`
- Prefer the SMC code in this checkout over any released package documentation when behavior differs.

## Workflow

1. Read the local BlackJAX SMC entry points before proposing changes.
2. Identify whether the task concerns:
   - generic SMC flow;
   - tempered/adaptive tempered SMC;
   - ESS computation or temperature adaptation;
   - resampling policy;
   - mutation/update strategy;
   - tests or examples.
3. Trace the actual runtime path through the relevant local files before editing.
4. Preserve JAX compatibility and jittability. Use `jax.lax` control flow for traced branching.
5. Keep abstractions reusable. Prefer generic resampling-decision or mutation-policy hooks over special-casing one algorithm.
6. Validate changes with targeted tests in the local checkout.

## Read These Files First

- For generic SMC control flow: read `blackjax/smc/base.py`.
- For MCMC-backed particle mutation: read `blackjax/smc/from_mcmc.py`.
- For tempered SMC: read `blackjax/smc/tempered.py`.
- For adaptive tempering: read `blackjax/smc/adaptive_tempered.py`.
- For ESS logic: read `blackjax/smc/ess.py`.
- For resampling schemes and strategies: read `blackjax/smc/resampling.py`.
- For regression coverage: read `tests/smc/test_smc.py` and `tests/smc/test_tempered_smc.py`.

Read `references/blackjax-smc-layout.md` when you need the local module map and dependency notes.

## Implementation Rules

- Keep `target_ess` semantics separate from any resampling threshold semantics.
- When adding a new resampling policy, make it composable from the shared SMC layer rather than embedding it in one algorithm.
- When resampling is conditional, document and test:
  - how ESS is computed;
  - when resampling is triggered;
  - what happens to weights when resampling is skipped;
  - what diagnostics are exposed in step info.
- Update docstrings and at least one example when behavior changes materially.
- Prefer targeted tests over broad end-to-end runs first.

## Validation

- Run focused SMC tests first, then widen only if needed.
- Good defaults:
  - `pytest tests/smc/test_smc.py -q`
  - `pytest tests/smc/test_tempered_smc.py -q`
  - `pytest tests/smc/test_smc_ess.py -q`
  - `python -m compileall blackjax/smc`

## References

- Local BlackJAX SMC map and dependency notes: `references/blackjax-smc-layout.md`

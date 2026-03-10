# Adaptive Tempered Resampling Strategy Contract

## Data Models

Implemented additions:
- `SMCInfo` may need to continue exposing ancestor indices even when resampling is skipped.
- Adaptive tempered SMC state should remain `TemperedSMCState(particles, weights, tempering_param)`.
- `SMCInfo` should expose whether resampling occurred and the ESS value used for the decision.

## API Signatures

Adaptive tempered kernel shape:

```python
build_kernel(
    logprior_fn,
    loglikelihood_fn,
    mcmc_step_fn,
    mcmc_init_fn,
    resampling_fn,
    target_ess,
    root_solver=solver.dichotomy,
    resampling_strategy=None,
    ...
) -> Callable
```

Adaptive tempered top-level API shape:

```python
as_top_level_api(
    logprior_fn,
    loglikelihood_fn,
    mcmc_step_fn,
    mcmc_init_fn,
    mcmc_parameters,
    resampling_fn,
    target_ess,
    root_solver=solver.dichotomy,
    num_mcmc_steps=10,
    resampling_threshold=0.9,
    resampling_strategy=None,
    ...
) -> SamplingAlgorithm
```

If `resampling_strategy` is provided, it overrides the ESS-threshold default.

## Module Boundaries

- `blackjax/smc/ess.py`
  - ESS helpers and any new utility needed to evaluate resampling conditions.
- `blackjax/smc/base.py`
  - General SMC step logic if conditional resampling is implemented generically.
- `blackjax/smc/from_mcmc.py`
  - Bridge between tempered SMC and the generic SMC step if new hooks are needed.
- `blackjax/smc/tempered.py`
  - Tempered kernel integration point if conditional resampling is implemented there.
- `blackjax/smc/adaptive_tempered.py`
  - Adaptive schedule API and wiring for the new strategy.
- `tests/smc/test_tempered_smc.py`
  - Behavioral coverage for adaptive tempering and conditional resampling.
- `tests/smc/test_smc_ess.py`
  - ESS helper coverage if new ESS utilities are introduced.

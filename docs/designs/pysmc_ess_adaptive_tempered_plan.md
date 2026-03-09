# Plan: Add `pysmc`-Style ESS Handling to BlackJAX Adaptive Tempered SMC

## Goal

Add an optional mode to BlackJAX's adaptive tempered SMC that matches the way
`pysmc` uses ESS:

- ESS selects the next tempering increment relative to the current ESS.
- ESS also decides whether resampling happens at that step.
- Resampling is conditional, not automatic.

This plan is scoped to the adaptive tempered SMC path first. It preserves the
current BlackJAX behavior by default.

## Why This Is Not Just a New Threshold

The current BlackJAX adaptive tempered SMC and `pysmc` use ESS at different
points in the transition.

Current BlackJAX adaptive tempered SMC:

1. Choose `delta` from an ESS target.
2. Resample unconditionally using the current weights.
3. Run MCMC targeting the current tempered posterior.
4. Reweight by the likelihood increment.

`pysmc`:

1. Choose the next `gamma` from an ESS target.
2. Reweight to that new `gamma`.
3. Compute ESS of the reweighted particles.
4. Resample only if ESS is below a threshold.
5. Run MCMC targeting the new tempered posterior.

Because the resampling decision in `pysmc` is based on the ESS after
reweighting, we cannot reproduce this by only adding a new threshold parameter
to `blackjax.smc.base.step`. The step order has to change for the new mode.

## Variable Mapping Between `pysmc` and BlackJAX

These variables play the same role even though the names differ:

- `pysmc.gamma` == BlackJAX `tempering_param`
- `pysmc.ess_reduction` == desired ratio between next ESS and current ESS
- `pysmc.ess_threshold` == resampling threshold as a fraction of `N`
- `pysmc.num_mcmc` == BlackJAX `num_mcmc_steps`

Important difference:

- Current BlackJAX `target_ess` means target ESS as a fraction of `N`.
- `pysmc.ess_reduction` means target ESS as a fraction of the current ESS.

Those are not interchangeable when weights are no longer uniform between steps.

## Proposed User-Facing Behavior

Add an optional `pysmc`-style ESS mode to `blackjax.adaptive_tempered_smc`.

Recommended API shape:

```python
blackjax.adaptive_tempered_smc(
    logprior_fn,
    loglikelihood_fn,
    mcmc_step_fn,
    mcmc_init_fn,
    mcmc_parameters,
    resampling_fn,
    target_ess=0.5,
    ess_reduction=None,
    resampling_threshold=None,
    root_solver=solver.dichotomy,
    num_mcmc_steps=10,
)
```

Semantics:

- `target_ess` keeps its current meaning and behavior.
- `ess_reduction=None` means use the current BlackJAX ESS adaptation.
- `ess_reduction in (0, 1)` enables `pysmc`-style adaptive tempering.
- `resampling_threshold=None` means keep current BlackJAX always-resample
  behavior.
- `resampling_threshold in (0, 1)` enables conditional resampling based on the
  ESS after reweighting.

Validation rules:

- Allow the current API unchanged.
- If `ess_reduction is not None`, reject simultaneous use of a non-default
  `target_ess`.
- If `resampling_threshold is not None`, it must be in `(0, 1)`.

Rationale:

- This keeps backward compatibility.
- It avoids overloading `target_ess` to mean two different things.
- It lets users enable the two `pysmc` behaviors independently during rollout.

## Recommended Behavioral Spec

### Existing Mode

Keep the current behavior as-is:

- solve `delta` from an absolute ESS target `target_ess * N`
- resample every step
- MCMC update happens before the new weights are attached

### New `pysmc`-Style Mode

At each SMC step:

1. Compute `delta` or the next `lambda_next` so that
   `ESS(new_weights) = ess_reduction * ESS(current_weights)`.
2. Form the new unnormalized log-weights:
   `log_w_new_unnorm = log_w_current + delta * loglikelihood(particles)`.
3. Normalize those weights and compute `ESS_new`.
4. If `ESS_new < resampling_threshold * N`, resample and reset weights to
   uniform.
5. Run the MCMC rejuvenation kernel targeting
   `logprior + lambda_next * loglikelihood`.
6. Return the moved particles and the post-step weights.

Post-step weights:

- if resampling happened: uniform weights
- if resampling did not happen: normalized reweighted weights

This matches the `pysmc` transition structure.

## Core Design Changes

### 1. Generalize ESS Solving in `blackjax/smc/ess.py`

Current `ess_solver` assumes the candidate ESS depends only on a fresh
incremental weight term. That works for the existing always-resample path,
because the previous step effectively starts from equally weighted particles.

For the new mode, the solver must account for carried-over weights.

Recommended new helper shape:

```python
def ess_solver(
    current_log_weights,
    log_weight_increment_fn,
    target_log_ess,
    max_delta,
    root_solver,
): ...
```

Where:

- `current_log_weights` are normalized log-weights for the current state
- `log_weight_increment_fn(delta)` returns the incremental log-weight term for
  that candidate `delta`
- `target_log_ess` is either:
  - `log(N * target_ess)` for the current BlackJAX mode
  - `log_ess(current_log_weights) + log(ess_reduction)` for the new mode

This refactor also removes the current sign ambiguity in the helper by making
the solver operate directly on log-weight increments rather than on a vaguely
named `logdensity_fn`.

### 2. Add a Reweight-Then-Resample Tempered Step

Current `blackjax.smc.base.step` hardcodes:

1. resample
2. update
3. reweight

The new mode needs:

1. reweight
2. maybe resample
3. update at the new temperature

Recommended approach:

- Do not force both orders through a single complicated `base.step`.
- Add a new internal helper specifically for tempered SMC with conditional
  resampling.

Suggested internal flow:

```python
def conditional_tempered_step(...):
    current_log_w = safe_log(state.weights)
    log_increment = delta * batched_loglikelihood(state.particles)
    log_w_unnorm = current_log_w + log_increment
    log_norm = logsumexp(log_w_unnorm)
    log_w = log_w_unnorm - log_norm
    ess = ess_from_log_weights(log_w)

    if ess < threshold * N:
        ancestors = resample_fn(...)
        particles = particles[ancestors]
        weights = uniform
        did_resample = True
    else:
        ancestors = arange(N)
        particles = state.particles
        weights = exp(log_w)
        did_resample = False

    particles = mcmc_update(particles, target=lambda_next)
    return new_state, info
```

### 3. Target the New Tempered Posterior During MCMC

This is a required semantic change for the new mode.

Current BlackJAX tempered SMC runs MCMC with

`logprior + lambda_t * loglikelihood`

before attaching the `lambda_{t+1} - lambda_t` weight increment.

The new mode must run MCMC with

`logprior + lambda_{t+1} * loglikelihood`

because reweighting has already moved the particle system to the new target
before rejuvenation, exactly as in `pysmc`.

### 4. Preserve Existing Low-Level Reuse

Avoid duplicating the MCMC parameter splitting and update strategy logic in
`from_mcmc.py`.

Recommended refactor:

- extract a helper that builds the particle update function from:
  - `mcmc_step_fn`
  - `mcmc_init_fn`
  - `mcmc_parameters`
  - `num_mcmc_steps`
  - target log density
- reuse that helper in both:
  - the current always-resample tempered path
  - the new conditional-resampling tempered path

## State and Info Contract

The public state can stay unchanged:

- `TemperedSMCState.particles`
- `TemperedSMCState.weights`
- `TemperedSMCState.tempering_param`

Recommended `SMCInfo` additions for the new mode:

- `ess`: ESS after reweighting, before optional resampling
- `did_resample`: boolean flag

Keep `ancestors` present in both modes:

- if resampling happens: actual resampling indices
- if not: `jnp.arange(num_particles)`

This avoids breaking downstream code that expects an ancestor array.

## Normalizing Constant Accounting

The current always-resample path computes the step contribution to the
normalizing constant assuming the particles were equally weighted before the
increment.

That is not correct once weights can persist across steps.

For the new mode, the incremental log normalizing constant should be:

```python
log_Z_increment = logsumexp(log_w_current + log_increment)
```

because `log_w_current` is already normalized.

This matches the `pysmc` behavior, where the new weights are formed from the
existing normalized weights and the incremental factor.

## Testing Plan

### Unit Tests

Add tests for `blackjax/smc/ess.py`:

- relative ESS solver hits `ess_reduction * current_ess`
- solver works with non-uniform incoming weights
- solver falls back to `max_delta` when the full step still satisfies the target

Add tests for the new conditional tempered transition:

- if `ESS_new >= threshold * N`, no resampling occurs
- if `ESS_new < threshold * N`, resampling occurs
- weights are uniform after resampling
- weights are preserved after the MCMC move when resampling does not occur
- MCMC uses `lambda_next`, not `lambda_current`

### Regression Tests

Add parity-style tests against the intended semantics:

- with `resampling_threshold=None`, behavior matches current BlackJAX
- with `ess_reduction` and `resampling_threshold` set, the transition follows
  the `pysmc` ordering
- higher `ess_reduction` produces smaller temperature jumps and more iterations

### Integration Tests

Extend `tests/smc/test_tempered_smc.py` with:

- a conditional-resampling adaptive tempered SMC test
- a test that resampling is not triggered at every iteration in the new mode
- a normalizing-constant test in the new mode

## Rollout Plan

### Phase 1

Implement the new ESS solver support and the new conditional tempered step
internally, but keep it behind optional arguments on
`adaptive_tempered.build_kernel` and `adaptive_tempered.as_top_level_api`.

### Phase 2

Document the new mode in `docs/` with one example showing:

- current BlackJAX adaptive ESS behavior
- `pysmc`-style relative ESS behavior
- conditional resampling behavior

### Phase 3

Optionally consider whether the same conditional-resampling flow should also be
exposed in the fixed-schedule `tempered_smc` API.

This should be a follow-up decision, not part of the initial feature.

## Risks

### Semantics Drift

Trying to retrofit the new behavior into the existing `base.step` may produce a
hybrid transition that is neither the current BlackJAX algorithm nor the
`pysmc` one. The new mode should have an explicitly separate internal control
flow.

### Ambiguous ESS Parameters

Using one parameter name for both "fraction of `N`" and "fraction of current
ESS" will make the API error-prone. The new mode should use a distinct argument
name, preferably `ess_reduction`.

### Normalizing Constant Bugs

If the current `log(N)` correction is reused in the new mode, the evidence
estimate will be wrong when resampling is skipped. This needs a dedicated test.

### `SMCInfo` Compatibility

Some downstream code may assume that `ancestors` always corresponds to an actual
resampling event. Returning the identity indices plus `did_resample=False` is
the least disruptive option.

## Recommended Initial Scope

Implement only this combination in the first PR:

- `adaptive_tempered_smc`
- optional `ess_reduction`
- optional `resampling_threshold`
- no changes to the default behavior

That is the smallest change that can truthfully support the `pysmc` way of
using ESS in BlackJAX.

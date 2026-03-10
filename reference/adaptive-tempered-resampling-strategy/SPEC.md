# Adaptive Tempered Resampling Strategy

## Scope

Add an adaptive tempered SMC strategy to BlackJAX that:
- preserves JAX compatibility and jittability;
- keeps the existing adaptive temperature selection behavior available;
- supports a second ESS-based policy that decides whether resampling occurs after reweighting;
- exposes the behavior through the adaptive tempered SMC API without introducing non-JAX dependencies.

Out of scope:
- changing unrelated SMC kernels;
- adding non-jittable Python-side control flow to the sampling path;
- introducing MPI, database, or mutable sampler state features from external libraries.

## Functional Requirements

1. The adaptive tempered SMC implementation must continue to compute the next tempering increment from an ESS target.
2. The implementation must support an additional ESS threshold that determines whether resampling is triggered after weights are updated.
3. The implementation must support skipping resampling when the post-weighting ESS remains above the threshold.
4. When resampling is skipped, particle weights must remain normalized and represent the current weighted approximation after the mutation step.
5. When resampling occurs, the resulting particle weights must be reset consistently with BlackJAX conventions.
6. The feature must integrate with the existing MCMC update path and remain compatible with existing JAX-based kernels.
7. The API must make the distinction between:
   - ESS used to pick the next tempering increment;
   - ESS threshold used to trigger resampling.
8. Existing behavior should remain accessible for callers that rely on unconditional resampling through an explicit resampling strategy.
9. The default adaptive tempered SMC behavior should use conditional ESS-threshold resampling with a conservative default threshold.

## Non-Functional Requirements

1. The implementation must be JAX compatible and safe to `jit`.
2. Control flow should be expressed with JAX primitives where runtime branching depends on traced values.
3. Added tests must cover both resampling and no-resampling branches.
4. The implementation should minimize API breakage except where the user explicitly requested a new default policy.

## Constraints

1. Use only JAX-based tools and libraries already consistent with BlackJAX.
2. Do not reference the external project by name in BlackJAX source code.
3. Do not introduce non-JAX arrays or SciPy/Numpy root-finding into runtime code paths.
4. Any deviation from JAX-compatible code would require explicit user permission.

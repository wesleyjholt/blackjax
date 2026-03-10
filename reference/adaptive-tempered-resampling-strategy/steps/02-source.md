# Step 2: Write Source Code

1. Identify the narrowest integration point that can support conditional resampling without duplicating the tempered SMC implementation.
2. Add any ESS utility required to evaluate the post-weighting ESS from log weights or normalized weights.
3. Implement conditional resampling with JAX control flow so traced execution remains valid under `jit`.
4. Preserve the current unconditional-resampling path for existing callers.
5. Wire the new configuration through the adaptive tempered API.
6. Keep all runtime logic in JAX arrays and JAX transformations.
7. Avoid changing unrelated public APIs unless required by the feature.

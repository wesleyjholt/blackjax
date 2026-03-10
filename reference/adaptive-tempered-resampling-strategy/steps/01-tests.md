# Step 1: Write Tests

1. Add a test that verifies the adaptive temperature increment still solves for the requested ESS target.
2. Add a test that sets a high resampling threshold and confirms resampling occurs after reweighting.
3. Add a test that sets a low resampling threshold and confirms resampling is skipped.
4. Add a test that verifies weights remain normalized when resampling is skipped.
5. Add a test that verifies weights reset to uniform when resampling occurs.
6. Add a test that exercises the new behavior under `jax.jit`.
7. Ensure any new info fields are asserted explicitly.

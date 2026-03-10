# Step 3: Tie Up Loose Ends

1. Check that ancestor semantics are well-defined when resampling is skipped.
2. Check that the normalizing-constant estimate remains consistent with the new control flow.
3. Check compatibility with waste-free and inner-kernel tuning integrations if they rely on current SMC info semantics.
4. Verify no existing tests assume unconditional resampling unless that behavior remains the default.

---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Use Conditional Resampling in Adaptive Tempered SMC

Adaptive tempered SMC in BlackJAX now separates two ESS-based decisions:

- `target_ess` chooses the next tempering increment;
- `resampling_threshold` decides whether the reweighted particles should be resampled before the MCMC mutation step.

If resampling is skipped, the particle system keeps its normalized non-uniform weights. The step info reports both the ESS used for the decision and whether resampling happened.

```{code-cell}
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import blackjax
import blackjax.smc.resampling as smc_resampling
```

```{code-cell}
num_particles = 128
rng_key = jax.random.key(0)

def logprior_fn(x):
    return stats.norm.logpdf(x, 0.0, 1.0)

def loglikelihood_fn(x):
    return stats.norm.logpdf(1.5, x, 0.5)

initial_particles = jax.random.normal(rng_key, (num_particles,))

hmc = blackjax.hmc.build_kernel()
hmc_init = blackjax.hmc.init
hmc_parameters = blackjax.smc.extend_params(
    {
        "step_size": 1e-1,
        "inverse_mass_matrix": jnp.ones((1,)),
        "num_integration_steps": 5,
    }
)
```

## Default conditional resampling

By default, adaptive tempered SMC uses a conservative ESS threshold of `0.9`.

```{code-cell}
tempering = blackjax.adaptive_tempered_smc(
    logprior_fn,
    loglikelihood_fn,
    hmc,
    hmc_init,
    hmc_parameters,
    smc_resampling.systematic,
    target_ess=0.8,
    num_mcmc_steps=2,
)

state = tempering.init(initial_particles)
rng_key, step_key = jax.random.split(rng_key)
state, info = tempering.step(step_key, state)

print("tempering parameter:", state.tempering_param)
print("ESS before resampling:", float(info.ess))
print("resampled:", bool(info.resampled))
```

## Customizing the resampling policy

You can change the threshold while keeping the built-in ESS rule:

```{code-cell}
tempering = blackjax.adaptive_tempered_smc(
    logprior_fn,
    loglikelihood_fn,
    hmc,
    hmc_init,
    hmc_parameters,
    smc_resampling.systematic,
    target_ess=0.8,
    num_mcmc_steps=2,
    resampling_threshold=0.5,
)
```

Or provide an explicit strategy. For example, this recovers unconditional resampling:

```{code-cell}
tempering = blackjax.adaptive_tempered_smc(
    logprior_fn,
    loglikelihood_fn,
    hmc,
    hmc_init,
    hmc_parameters,
    smc_resampling.systematic,
    target_ess=0.8,
    num_mcmc_steps=2,
    resampling_strategy=smc_resampling.always(smc_resampling.systematic),
)
```

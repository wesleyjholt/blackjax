# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, NamedTuple, Optional, Protocol

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import blackjax.smc.resampling as resampling
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey


class SMCState(NamedTuple):
    """State of the SMC sampler.

    Parameters
    ----------
    particles: ArrayTree | ArrayLikeTree
        Particles representing samples from the target distribution. Each leaf
        represents a variable from the posterior, being an array of size
        `(n_particles, ...)`.
    weights: Array
        Normalized weights for each particle, shape (n_particles,).
    update_parameters: ArrayTree
        Parameters passed to the update function.

    Examples
    --------
    Three particles with different posterior structures:
        - Single univariate posterior:
            [ Array([[1.], [1.2], [3.4]]) ]
        - Single bivariate  posterior:
            [ Array([[1,2], [3,4], [5,6]]) ]
        - Two variables, each univariate:
            [ Array([[1.], [1.2], [3.4]]),
            Array([[50.], [51], [55]]) ]
        - Two variables, first one bivariate, second one 4-variate:
            [ Array([[1., 2.], [1.2, 0.5], [3.4, 50]]),
            Array([[50., 51., 52., 51], [51., 52., 52. ,54.], [55., 60, 60, 70]]) ]
    """

    particles: ArrayTree | ArrayLikeTree
    weights: Array
    update_parameters: ArrayTree


class SMCInfo(NamedTuple):
    """Additional information on the tempered SMC step.

    Parameters
    ----------
    ancestors: Array
        The index of the particles proposed by the MCMC pass that were selected
        by the resampling step.
    log_likelihood_increment: float | Array
        The log-likelihood increment due to the current step of the SMC algorithm.
    update_info: NamedTuple
        Additional information returned by the update function.
    ess: float | Array
        Effective sample size after reweighting and before any optional resampling.
    resampled: bool | Array
        Whether the resampling strategy selected a resampling move.
    """

    ancestors: Array
    log_likelihood_increment: float | Array
    update_info: NamedTuple
    ess: float | Array
    resampled: bool | Array


def init(particles: ArrayLikeTree, init_update_params: ArrayTree) -> SMCState:
    """Initialize the SMC state.

    Parameters
    ----------
    particles: ArrayLikeTree
        Initial particles, typically sampled from the prior.
    init_update_params: ArrayTree
        Initial parameters for the update function.

    Returns
    -------
    SMCState
        Initial state with uniform weights.
    """
    # Infer the number of particles from the size of the leading dimension of
    # the first leaf of the inputted PyTree.
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    weights = jnp.ones(num_particles) / num_particles
    return SMCState(particles, weights, init_update_params)


def step(
    rng_key: PRNGKey,
    state: SMCState,
    update_fn: Callable,
    weight_fn: Callable,
    resample_fn: Callable,
    num_resampled: Optional[int] = None,
    resampling_strategy: Optional[Callable] = None,
) -> tuple[SMCState, SMCInfo]:
    """General SMC sampling step.

    `update_fn` here corresponds to the Markov kernel $M_{t+1}$, and `weight_fn`
    corresponds to the potential function $G_t$. The step first reweights the
    current particles, then delegates the resampling decision to
    ``resampling_strategy`` and finally mutates the selected particles with
    ``update_fn``.

    The `update_fn` and `weight_fn` functions must be batched by the caller either
    using `jax.vmap` or `jax.pmap`.

    In Feynman-Kac terms, the algorithm goes roughly as follows:

    .. code::

        G_t: weight_fn
        R_t: resampling_strategy
        M_t: update_fn
        weights_t = normalize(log(w_{t-1}) + G_t(x_{t-1}))
        idx = R_t(weights_t)
        x_t = x_tm1[idx]
        x_{t+1} = M_t(x_t)
        weights_{t+1} = uniform if resampled else weights_t

    Parameters
    ----------
    rng_key: PRNGKey
        Key used to generate pseudo-random numbers.
    state: SMCState
        Current state of the SMC sampler: particles and their respective weights.
    update_fn: Callable
        Function that takes an array of keys and particles and returns
        new particles.
    weight_fn: Callable
        Function that assigns a weight to the particles.
    resample_fn: Callable
        Particle resampling function. This is used by the default strategy and by
        any custom strategy that chooses to draw new ancestors.
    num_resampled: int, optional
        The number of particles to resample. This can be used to implement
        Waste-Free SMC :cite:p:`dau2020waste`, in which case we resample a number
        :math:`M<N` of particles, and the update function is in charge of returning
        :math:`N` samples.
    resampling_strategy: Callable, optional
        Callable with signature
        ``(rng_key, weights, num_samples) -> ResamplingDecision``.
        If ``None``, the step always resamples using ``resample_fn``. Strategies
        can implement conditional resampling based on ESS or other diagnostics.

    Returns
    -------
    new_state: SMCState
        The new SMCState containing updated particles and weights.
    info: SMCInfo
        An `SMCInfo` object that contains extra information about the SMC
        transition, including the ESS before optional resampling and whether the
        strategy triggered resampling.

    """
    updating_key, resampling_key = jax.random.split(rng_key, 2)

    num_particles = state.weights.shape[0]

    if num_resampled is None:
        num_resampled = num_particles

    if resampling_strategy is None:
        resampling_strategy = resampling.always(resample_fn)

    log_weights = jnp.log(state.weights) + weight_fn(state.particles)
    logsum_weights = logsumexp(log_weights)
    normalizing_constant = logsum_weights
    weights = jnp.exp(log_weights - logsum_weights)
    decision = resampling_strategy(resampling_key, weights, num_resampled)

    particles = jax.tree.map(lambda x: x[decision.ancestors], state.particles)
    keys = jax.random.split(updating_key, num_resampled)
    particles, update_info = update_fn(keys, particles, state.update_parameters)

    num_output_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    new_weights = jax.lax.cond(
        decision.resampled,
        lambda _: jnp.ones(num_output_particles, dtype=weights.dtype) / num_output_particles,
        lambda _: weights,
        operand=None,
    )

    return SMCState(particles, new_weights, state.update_parameters), SMCInfo(
        decision.ancestors,
        normalizing_constant,
        update_info,
        decision.ess,
        decision.resampled,
    )


def extend_params(params: Array) -> Array:
    """Extend parameters to be used for all particles in SMC.

    Given a dictionary of params, repeats them for every single particle. The
    expected usage is in cases where the aim is to repeat the same parameters for
    all chains within SMC.

    Parameters
    ----------
    params: Array
        Parameters to extend for all particles.

    Returns
    -------
    Array
        Extended parameters with an additional dimension for particles.
    """

    return jax.tree.map(lambda x: jnp.asarray(x)[None, ...], params)


def update_and_take_last(
    mcmc_init_fn: Callable,
    tempered_logposterior_fn: Callable,
    shared_mcmc_step_fn: Callable,
    num_mcmc_steps: int,
    n_particles: int | Array,
) -> tuple[Callable, int | Array]:
    """Create an MCMC update strategy that runs multiple steps and keeps the last.

    Given N particles, runs num_mcmc_steps of a kernel starting at each particle, and
    returns the last values, wasting the previous num_mcmc_steps-1 samples per chain.

    Parameters
    ----------
    mcmc_init_fn: Callable
        Function that initializes an MCMC state from a position.
    tempered_logposterior_fn: Callable
        Tempered log-posterior probability density function.
    shared_mcmc_step_fn: Callable
        MCMC step function.
    num_mcmc_steps: int
        Number of MCMC steps to run for each particle.
    n_particles: int | Array
        Number of particles.

    Returns
    -------
    mcmc_kernel: Callable
        A vectorized MCMC kernel function.
    n_particles: int | Array
        Number of particles (returned unchanged).
    """

    # simply typing protocal for type checker
    class MCMCStateProtocol(Protocol):
        position: Any

    def mcmc_kernel(
        rng_key: PRNGKey,
        position: ArrayTree,
        step_parameters: dict[str, float | Array],
    ) -> tuple[MCMCStateProtocol, SMCInfo]:
        state = mcmc_init_fn(position, tempered_logposterior_fn)

        def body_fn(
            state: MCMCStateProtocol,
            rng_key: PRNGKey,
        ) -> tuple[MCMCStateProtocol, SMCInfo]:
            new_state, info = shared_mcmc_step_fn(
                rng_key,
                state,
                tempered_logposterior_fn,
                **step_parameters,
            )
            return new_state, info

        keys = jax.random.split(rng_key, num_mcmc_steps)
        last_state, info = jax.lax.scan(body_fn, state, keys)
        return last_state.position, info

    return jax.vmap(mcmc_kernel), n_particles

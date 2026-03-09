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
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import blackjax.smc.base as base
import blackjax.smc.ess as ess
import blackjax.smc.from_mcmc as smc_from_mcmc
import blackjax.smc.solver as solver
import blackjax.smc.tempered as tempered
from blackjax.base import SamplingAlgorithm
from blackjax.smc.base import update_and_take_last
from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = ["build_kernel", "init", "as_top_level_api"]


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    target_ess: Optional[float],
    root_solver: Callable = solver.dichotomy,
    **extra_parameters: dict[str, Any],
) -> Callable:
    """Build a Tempered SMC step using an adaptive schedule.

    Parameters
    ----------
    logprior_fn: Callable
        Log prior probability function.
    loglikelihood_fn: Callable
        Log likelihood function.
    mcmc_step_fn: Callable
        Function that creates MCMC step from log-probability density function.
    mcmc_init_fn: Callable
        A function that creates a new mcmc state from a position and a
        log-probability density function.
    resampling_fn: Callable
        Resampling function (from blackjax.smc.resampling).
    target_ess: float | Array
        Target effective sample size (ESS) to determine the next tempering
        parameter.
    root_solver: Callable, optional
        The solver used to adaptively compute the temperature given a target number
        of effective samples. By default, blackjax.smc.solver.dichotomy.
    **extra_parameters : dict[str, Any]
        Additional parameters to pass to tempered.build_kernel.

    Returns
    -------
    kernel: Callable
        A callable that takes a rng_key, a TemperedSMCState, num_mcmc_steps,
        and mcmc_parameters, and returns a new TemperedSMCState along with
        information about the transition.

    """

    ess_reduction = extra_parameters.pop("ess_reduction", None)
    resampling_threshold = extra_parameters.pop("resampling_threshold", None)
    update_strategy = extra_parameters.get("update_strategy", update_and_take_last)
    update_particles_fn = extra_parameters.get("update_particles_fn", None)

    if ess_reduction is not None and not 0.0 < ess_reduction < 1.0:
        raise ValueError("ess_reduction must be in (0, 1).")
    if resampling_threshold is not None and not 0.0 < resampling_threshold < 1.0:
        raise ValueError("resampling_threshold must be in (0, 1).")
    if ess_reduction is None and target_ess is None:
        raise ValueError("target_ess must be provided when ess_reduction is None.")

    use_pysmc_ess = ess_reduction is not None or resampling_threshold is not None

    if use_pysmc_ess:
        if update_particles_fn is not None:
            raise ValueError(
                "update_particles_fn is not supported with conditional resampling."
            )
        if update_strategy is not update_and_take_last:
            raise ValueError(
                "Conditional resampling currently supports only update_and_take_last."
            )

    def compute_delta(
        state: tempered.TemperedSMCState, current_log_weights: Optional[Array] = None
    ) -> float | Array:
        tempering_param = state.tempering_param
        max_delta = 1 - tempering_param
        batched_loglikelihood = jax.vmap(loglikelihood_fn)

        if ess_reduction is None:
            delta = ess.ess_solver(
                batched_loglikelihood,
                state.particles,
                target_ess,
                max_delta,
                root_solver,
                current_log_weights=current_log_weights,
            )
        else:
            if current_log_weights is None:
                current_log_weights = jnp.where(
                    state.weights > 0, jnp.log(state.weights), -jnp.inf
                )
            delta = ess.ess_solver_by_ratio(
                batched_loglikelihood,
                state.particles,
                ess_reduction,
                max_delta,
                root_solver,
                current_log_weights,
            )
        delta = jnp.clip(delta, 0.0, max_delta)

        return delta

    tempered_kernel = None
    if not use_pysmc_ess:
        tempered_kernel = tempered.build_kernel(
            logprior_fn,
            loglikelihood_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            resampling_fn,
            **extra_parameters,  # type: ignore
        )

    def kernel(
        rng_key: PRNGKey,
        state: tempered.TemperedSMCState,
        num_mcmc_steps: int | Array,
        mcmc_parameters: dict,
    ) -> tuple[tempered.TemperedSMCState, base.SMCInfo]:
        if not use_pysmc_ess:
            delta = compute_delta(state)
            tempering_param = delta + state.tempering_param
            assert tempered_kernel is not None
            return tempered_kernel(
                rng_key, state, num_mcmc_steps, tempering_param, mcmc_parameters
            )

        resampling_key, updating_key = jax.random.split(rng_key, 2)
        num_particles = state.weights.shape[0]
        current_log_weights = jnp.where(state.weights > 0, jnp.log(state.weights), -jnp.inf)
        delta = compute_delta(state, current_log_weights)
        tempering_param = delta + state.tempering_param

        batched_loglikelihood = jax.vmap(loglikelihood_fn)
        log_weight_increment = delta * batched_loglikelihood(state.particles)
        log_weights = current_log_weights + log_weight_increment
        logsum_weights = logsumexp(log_weights)
        normalized_log_weights = log_weights - logsum_weights
        weights = jnp.exp(normalized_log_weights)
        ess_value = ess.ess(normalized_log_weights)
        normalizing_constant = logsum_weights

        def tempered_logposterior_fn(position: ArrayLikeTree) -> float:
            logprior = logprior_fn(position)
            tempered_loglikelihood = tempering_param * loglikelihood_fn(position)
            return logprior + tempered_loglikelihood

        update_fn, num_resampled, unshared_mcmc_parameters = (
            smc_from_mcmc.build_mcmc_update_fn(
                mcmc_parameters,
                mcmc_step_fn,
                mcmc_init_fn,
                tempered_logposterior_fn,
                num_mcmc_steps,
                num_particles,
                update_strategy,
            )
        )

        if num_resampled != num_particles:
            raise ValueError(
                "Conditional resampling currently requires num_resampled == num_particles."
            )

        should_resample = (
            jnp.asarray(True)
            if resampling_threshold is None
            else ess_value < resampling_threshold * num_particles
        )

        def resample_fn(_: None) -> tuple[ArrayLikeTree, dict, Array, Array, Array]:
            ancestors = resampling_fn(resampling_key, weights, num_particles)
            particles = jax.tree.map(lambda x: x[ancestors], state.particles)
            parameters = jax.tree.map(lambda x: x[ancestors], unshared_mcmc_parameters)
            new_weights = jnp.ones(num_particles) / num_particles
            return particles, parameters, ancestors, new_weights, jnp.asarray(True)

        def skip_resample(
            _: None,
        ) -> tuple[ArrayLikeTree, dict, Array, Array, Array]:
            ancestors = jnp.arange(num_particles)
            return (
                state.particles,
                unshared_mcmc_parameters,
                ancestors,
                weights,
                jnp.asarray(False),
            )

        particles, update_parameters, ancestors, new_weights, did_resample = (
            jax.lax.cond(should_resample, resample_fn, skip_resample, operand=None)
        )
        keys = jax.random.split(updating_key, num_particles)
        particles, update_info = update_fn(keys, particles, update_parameters)

        tempered_state = tempered.TemperedSMCState(
            particles,
            new_weights,
            tempering_param,
        )
        info = base.SMCInfo(
            ancestors,
            normalizing_constant,
            update_info,
            ess_value,
            did_resample,
        )
        return tempered_state, info

    return kernel


init = tempered.init


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    resampling_fn: Callable,
    target_ess: Optional[float] = None,
    root_solver: Callable = solver.dichotomy,
    num_mcmc_steps: int = 10,
    **extra_parameters: dict[str, Any],
) -> SamplingAlgorithm:
    """Implements the user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logprior_fn: Callable
        The log-prior function of the model we wish to draw samples from.
    loglikelihood_fn: Callable
        The log-likelihood function of the model we wish to draw samples from.
    mcmc_step_fn: Callable
        The MCMC step function used to update the particles.
    mcmc_init_fn: Callable
        The MCMC init function used to build a MCMC state from a particle position.
    mcmc_parameters: dict
        The parameters of the MCMC step function. Parameters with leading dimension
        length of 1 are shared amongst the particles.
    resampling_fn: Callable
        The function used to resample the particles.
    target_ess: float | Array
        Target effective sample size (ESS) to determine the next tempering
        parameter.
    root_solver: Callable, optional
        The solver used to adaptively compute the temperature given a target number
        of effective samples. By default, blackjax.smc.solver.dichotomy.
    num_mcmc_steps: int, optional
        The number of times the MCMC kernel is applied to the particles per step,
        by default 10.
    **extra_parameters: dict [str, Any]
        Additional parameters to pass to the kernel. Two optional parameters
        enable pysmc-style ESS handling:
        - ess_reduction: target ESS ratio relative to the current ESS.
        - resampling_threshold: only resample when ESS falls below this
          fraction of the number of particles.

    Returns
    -------
    SamplingAlgorithm
        A ``SamplingAlgorithm`` instance with init and step methods.

    """
    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        target_ess,
        root_solver,
        **extra_parameters,
    )

    def init_fn(
        position: ArrayLikeTree, rng_key: Optional[PRNGKey] = None
    ) -> tempered.TemperedSMCState:
        del rng_key
        return init(position)

    def step_fn(
        rng_key: PRNGKey, state: tempered.TemperedSMCState
    ) -> tuple[tempered.TemperedSMCState, base.SMCInfo]:
        return kernel(
            rng_key,
            state,
            num_mcmc_steps,
            mcmc_parameters,
        )

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]

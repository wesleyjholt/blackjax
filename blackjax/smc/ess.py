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
"""All things related to SMC effective sample size"""
from typing import Callable, Optional

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from blackjax.types import Array, ArrayLikeTree


def ess(log_weights: Array) -> float | Array:
    """Compute the effective sample size.

    Parameters
    ----------
    log_weights: Array
        Log-weights of the sample, shape (n_particles,).

    Returns
    -------
    ess: float | Array
        The effective sample size.
    """
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Array) -> float | Array:
    """Compute the logarithm of the effective sample size.

    Parameters
    ----------
    log_weights: Array
        Log-weights of the sample, shape (n_particles,).

    Returns
    -------
    log_ess: float | Array
        The logarithm of the effective sample size.
    """
    return 2 * logsumexp(log_weights) - logsumexp(2 * log_weights)


def _normalized_log_weights(weights: Array) -> float | Array:
    """Safely convert normalized weights to log-weights."""
    return jnp.where(weights > 0, jnp.log(weights), -jnp.inf)


def _ess_solver(
    current_log_weights: Array,
    log_weight_increment: Array,
    target_log_ess: float | Array,
    max_delta: float | Array,
    root_solver: Callable,
) -> float | Array:
    current_log_weights = jnp.ravel(current_log_weights)
    log_weight_increment = jnp.ravel(log_weight_increment)

    def fun_to_solve(delta: float | Array) -> Array:
        log_weights = current_log_weights + jnp.nan_to_num(delta * log_weight_increment)
        ess_val = log_ess(log_weights)
        return ess_val - target_log_ess

    return root_solver(fun_to_solve, 0.0, max_delta)


def ess_solver(
    logdensity_fn: Callable,
    particles: ArrayLikeTree,
    target_ess: float | Array,
    max_delta: float | Array,
    root_solver: Callable,
    current_log_weights: Optional[Array] = None,
) -> float | Array:
    """ESS solver for computing the next increment of SMC tempering.

    Parameters
    ----------
    logdensity_fn: Callable
        The log probability function we wish to sample from.
    particles: ArrayLikeTree
        Current particles of the tempered SMC algorithm.
    target_ess: float | Array
        Target effective sample size (ESS) for the next increment of SMC tempering.
    max_delta: float | Array
        Maximum acceptable delta increment.
    root_solver: Callable
        A solver to find the root of a function. Signature is
        root_solver(fun, min_delta, max_delta). Use e.g. dichotomy from
        blackjax.smc.solver.
    current_log_weights: Array, optional
        Current normalized log-weights. If omitted, all particles are assumed
        to have equal weights.

    Returns
    -------
    delta: float | Array
        The increment that solves for the target ESS.

    """
    log_weight_increment = jnp.ravel(logdensity_fn(particles))
    n_particles = log_weight_increment.shape[0]
    if current_log_weights is None:
        current_log_weights = -jnp.log(n_particles) * jnp.ones(n_particles)
    else:
        current_log_weights = jnp.ravel(current_log_weights)
    target_val = jnp.log(n_particles * target_ess)
    return _ess_solver(
        current_log_weights,
        log_weight_increment,
        target_val,
        max_delta,
        root_solver,
    )


def ess_solver_by_ratio(
    logdensity_fn: Callable,
    particles: ArrayLikeTree,
    ess_reduction: float | Array,
    max_delta: float | Array,
    root_solver: Callable,
    current_log_weights: Array,
) -> float | Array:
    """Solve for the next increment using an ESS target relative to the current ESS."""
    log_weight_increment = jnp.ravel(logdensity_fn(particles))
    current_log_weights = jnp.ravel(current_log_weights)
    target_log_ess = log_ess(current_log_weights) + jnp.log(ess_reduction)
    return _ess_solver(
        current_log_weights,
        log_weight_increment,
        target_log_ess,
        max_delta,
        root_solver,
    )

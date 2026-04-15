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
from typing import Callable

import jax
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


def ess_solver(
    logdensity_fn: Callable,
    particles: ArrayLikeTree,
    target_ess: float | Array,
    max_delta: float | Array,
    root_solver: Callable,
    current_log_weights: Array | None = None,
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

    Returns
    -------
    delta: float | Array
        The increment that solves for the target ESS.

    """
    logprob = logdensity_fn(particles)
    n_particles = logprob.shape[0]
    target_val = jnp.log(n_particles * target_ess)
    if current_log_weights is None:
        current_log_weights = jnp.zeros_like(logprob)
    current_ess = log_ess(current_log_weights)

    def fun_to_solve(delta: float | Array) -> Array:
        log_weights = current_log_weights + jnp.nan_to_num(-delta * logprob)
        ess_val = log_ess(log_weights)

        return ess_val - target_val

    def if_already_below_target(_: None) -> float | Array:
        # Cumulative weight drift has put ``current_ess`` at or below the
        # caller's ``target_val``. Because reweighting only decreases ESS, the
        # original target is unreachable. The naive fallback is to return
        # ``max_delta``, but that is the *worst* possible choice in adaptive
        # tempered SMC: it teleports the tempering parameter to 1.0 in a
        # single step, leaving the inner MCMC kernel calibrated for the old
        # intermediate target and producing near-zero acceptance at the
        # final step (see ``test_ess_solver_does_not_teleport_when_drift_below_target``
        # for a regression that fails on the old code).
        #
        # Instead, retry the dichotomy against a relaxed target that preserves
        # ``0.99 * current_ess`` — still a small, valid tempering step, avoids
        # both the ``max_delta`` blowup and the dichotomy NaN, and keeps SMC
        # making monotone progress.
        relaxed_target = current_ess + jnp.log(0.99)

        def fun_to_solve_relaxed(delta: float | Array) -> Array:
            log_weights = current_log_weights + jnp.nan_to_num(-delta * logprob)
            return log_ess(log_weights) - relaxed_target

        return root_solver(fun_to_solve_relaxed, 0.0, max_delta)

    estimated_delta = jax.lax.cond(
        current_ess <= target_val,
        if_already_below_target,
        lambda _: root_solver(fun_to_solve, 0.0, max_delta),
        operand=None,
    )
    return estimated_delta

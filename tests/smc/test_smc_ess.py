"""Test the ess function"""
import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.scipy.stats.multivariate_normal import logpdf as multivariate_logpdf
from jax.scipy.stats.norm import logpdf as univariate_logpdf

import blackjax.smc.ess as ess
import blackjax.smc.solver as solver


class SMCEffectiveSampleSizeTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_ess(self):
        # All particles have zero weight but one
        weights = jnp.array([-jnp.inf, -jnp.inf, 0, -jnp.inf])
        ess_val = self.variant(ess.ess)(weights)
        assert ess_val == 1.0

        weights = jnp.ones(12)
        ess_val = self.variant(ess.ess)(weights)
        assert ess_val == 12

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([0.2, 0.95])
    def test_ess_solver(self, target_ess):
        num_particles = 1000
        potential_fn = lambda pytree: -univariate_logpdf(pytree, scale=0.1)
        potential = jax.vmap(lambda x: potential_fn(x), in_axes=[0])
        particles = np.random.normal(0, 1, size=(num_particles, 1))
        self.ess_solver_test_case(potential, particles, target_ess, num_particles, 1.0)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([0.2, 0.95])
    def test_ess_solver_multivariate(self, target_ess):
        """
        Posterior with more than one variable. Let's assume we want to
        sample from P(x) x ~ N(mean, cov) x in R^{2}
        """
        num_particles = 1000
        mean = jnp.zeros((1, 2))
        cov = jnp.diag(jnp.array([1, 1]))
        _logdensity_fn = lambda pytree: multivariate_logpdf(pytree, mean=mean, cov=cov)
        potential = jax.vmap(_logdensity_fn, in_axes=[0], out_axes=0)
        particles = np.random.multivariate_normal(
            mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_particles
        )
        self.ess_solver_test_case(potential, particles, target_ess, num_particles, 10.0)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([0.2, 0.95])
    def test_ess_solver_posterior_signature(self, target_ess):
        """
        Posterior with more than one variable. Let's assume we want to
        sample from P(x,y) x ~ N(mean, cov) y ~ N(mean, cov)
        """
        num_particles = 1000
        mean = jnp.zeros((1, 2))
        cov = jnp.diag(jnp.array([1, 1]))

        def _logdensity_fn(pytree):
            return multivariate_logpdf(
                pytree[0], mean=mean, cov=cov
            ) + multivariate_logpdf(pytree[1], mean=mean, cov=cov)

        potential = jax.vmap(_logdensity_fn, in_axes=[0], out_axes=0)
        particles = [
            np.random.multivariate_normal(
                mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_particles
            ),
            np.random.multivariate_normal(
                mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_particles
            ),
        ]
        self.ess_solver_test_case(potential, particles, target_ess, num_particles, 10.0)

    def ess_solver_test_case(self, potential, particles, target_ess, N, max_delta):
        ess_solver_fn = functools.partial(
            ess.ess_solver,
            potential,
            target_ess=target_ess,
            max_delta=max_delta,
            root_solver=solver.dichotomy,
        )

        delta = self.variant(ess_solver_fn)(particles)
        assert delta > 0

        ess_val = ess.ess(-delta * potential(particles))
        np.testing.assert_allclose(ess_val, target_ess * N, atol=1e-1, rtol=1e-2)

    @chex.all_variants(with_pmap=False)
    def test_ess_solver_with_existing_weights(self):
        num_particles = 1000
        potential = jax.vmap(lambda x: jnp.square(x))
        particles = np.random.normal(0, 1, size=(num_particles,))
        current_weights = jax.nn.log_softmax(jnp.linspace(-2.0, 2.0, num_particles))
        target_ess = 0.4

        delta = self.variant(
            functools.partial(
                ess.ess_solver,
                potential,
                target_ess=target_ess,
                max_delta=1.0,
                root_solver=solver.dichotomy,
                current_log_weights=current_weights,
            )
        )(particles)

        ess_val = ess.ess(current_weights - delta * potential(particles))
        np.testing.assert_allclose(
            ess_val, target_ess * num_particles, atol=1e-1, rtol=1e-2
        )

    @chex.all_variants(with_pmap=False)
    def test_adaptive_tempered_compute_delta_sign_consistency(self):
        """Regression for the sign mismatch between ``ess_solver`` and ``tempered``.

        ``blackjax.smc.ess.ess_solver`` expects a *potential* argument
        (negative log density) so that its internal
        ``log_weights = current_log_weights + (-delta * logprob)`` matches the
        actual reweighting performed by ``blackjax.smc.tempered``
        (``log_weights_fn = delta * loglikelihood_fn``, see tempered.py). The
        old code in ``blackjax/smc/adaptive_tempered.py`` passed the raw
        ``loglikelihood_fn`` — inverting the sign.

        This test runs the patched ``adaptive_tempered.compute_delta`` path
        (via ``blackjax.adaptive_tempered_smc``) on a left-skewed likelihood
        and asserts that after applying the returned delta with the same
        reweighting formula ``tempered.py`` uses, the resulting ESS matches
        the caller's ``target_ess * N`` within tolerance. Before the sign fix
        the two disagreed, so the solver returned a delta too large for the
        real reweighting.
        """
        import blackjax
        import blackjax.smc.resampling as resampling_mod

        num_particles = 2000
        target_ess = 0.75

        # Left-skewed loglik: mean near -10, long negative tail, max near 0.
        def logprior_fn(x):
            return jnp.sum(-0.5 * x * x - 0.5 * jnp.log(2 * jnp.pi))

        def loglikelihood_fn(x):
            return -5.0 * jnp.sum((x - 0.5) ** 2)

        rng = jax.random.key(202604)
        rng, init_key = jax.random.split(rng)
        init_particles = jax.random.normal(init_key, shape=(num_particles, 3))

        # Identity MCMC kernel: we only care about the reweighting step. Use
        # a single RMH mutation with a tiny proposal so mutation is a no-op
        # and ESS is determined entirely by reweighting.
        import blackjax.mcmc.random_walk as rw

        rmh_kernel = rw.build_additive_step()

        def mcmc_step_fn(rng_key, state, logdensity_fn, proposal_chol):
            return rmh_kernel(
                rng_key, state, logdensity_fn, rw.normal(proposal_chol)
            )

        tempering = blackjax.adaptive_tempered_smc(
            logprior_fn,
            loglikelihood_fn,
            mcmc_step_fn,
            rw.init,
            {"proposal_chol": 1e-8 * jnp.eye(3)[None, ...]},
            resampling_mod.systematic,
            target_ess=target_ess,
            num_mcmc_steps=1,
            resampling_threshold=0.0,  # disable resampling: isolate reweight
        )
        state = tempering.init(init_particles)

        # Snapshot pre-step weights and compute what ESS *should* be after
        # reweighting by the chosen delta using tempered.py's +delta * loglik
        # formula.
        log_w_before = jnp.log(state.weights)
        loglik_at_particles = jax.vmap(loglikelihood_fn)(state.particles)

        rng, step_key = jax.random.split(rng)
        new_state, _info = self.variant(tempering.step)(step_key, state)

        delta = float(new_state.tempering_param - state.tempering_param)
        assert delta > 0.0, f"expected positive delta, got {delta}"

        # Apply tempered.py's reweighting formula to pre-step weights.
        log_w_after_via_tempered = log_w_before + delta * loglik_at_particles
        ess_after_via_tempered = float(np.exp(ess.log_ess(log_w_after_via_tempered)))

        target_val = target_ess * num_particles
        np.testing.assert_allclose(
            ess_after_via_tempered, target_val, rtol=0.05, atol=10.0,
        )

    @chex.all_variants(with_pmap=False)
    def test_ess_solver_does_not_teleport_when_drift_below_target(self):
        """Regression for the ``if_already_below_target`` short-circuit.

        Construct a particle + weight configuration where ``current_ess`` is
        just barely below ``target_val * N``. On the original code this
        triggered the short-circuit and returned ``max_delta``, which in
        adaptive tempered SMC teleports the tempering parameter to 1.0 in a
        single step. The patched code retries against a relaxed target
        (0.99 * current_ess), so the returned delta must be strictly less
        than ``max_delta`` and the resulting ESS must stay close to
        ``current_ess`` (no further than 2% below).
        """
        num_particles = 1000
        # Small positive potential so reweighting drops ESS as delta grows.
        # The sign here matches the existing test convention
        # (potential = -logdensity).
        potential = jax.vmap(lambda x: jnp.square(x))
        particles = np.linspace(-1.0, 1.0, num_particles)

        # Build current_log_weights such that current_ess is *just* below the
        # target we are about to ask for. Approach: start uniform, apply a
        # tiny initial reweighting with the same potential to seed mild
        # non-uniformity, then tune the target_ess to land just above the
        # resulting current_ess.
        seed_delta = 1e-3
        current_log_weights = -seed_delta * potential(particles)
        current_ess_val = float(np.exp(ess.log_ess(current_log_weights)))

        # Target ESS is set such that ``target_val`` exceeds ``current_ess``
        # by a hair — firmly inside the ``current_ess <= target_val`` branch.
        target_ess_fraction = min((current_ess_val + 0.5) / num_particles, 0.9999)
        max_delta = 1.0

        delta = self.variant(
            functools.partial(
                ess.ess_solver,
                potential,
                target_ess=target_ess_fraction,
                max_delta=max_delta,
                root_solver=solver.dichotomy,
                current_log_weights=current_log_weights,
            )
        )(particles)

        # 1. The solver must NOT return max_delta.
        assert float(delta) < max_delta - 1e-6, (
            f"ess_solver returned delta={float(delta)} which is at max_delta={max_delta} — "
            "the short-circuit is still teleporting."
        )

        # 2. Applying the delta should drop ESS by at most ~2% of current.
        #    Use the same reweighting convention the solver uses internally
        #    (log_weights = current - delta * potential).
        new_log_weights = current_log_weights - float(delta) * potential(particles)
        new_ess_val = float(np.exp(ess.log_ess(new_log_weights)))
        assert new_ess_val >= 0.97 * current_ess_val, (
            f"after retry with relaxed target, ESS dropped too far: "
            f"{new_ess_val:.2f} < 0.97 * {current_ess_val:.2f}"
        )
        # And the delta must be positive (we did make some progress).
        assert float(delta) > 0.0, (
            f"ess_solver returned delta={float(delta)}; expected positive progress."
        )


if __name__ == "__main__":
    absltest.main()

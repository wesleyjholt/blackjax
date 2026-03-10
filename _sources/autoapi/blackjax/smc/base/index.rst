blackjax.smc.base
=================

.. py:module:: blackjax.smc.base


Classes
-------

.. autoapisummary::

   blackjax.smc.base.SMCState
   blackjax.smc.base.SMCInfo


Functions
---------

.. autoapisummary::

   blackjax.smc.base.init
   blackjax.smc.base.step
   blackjax.smc.base.extend_params
   blackjax.smc.base.update_and_take_last


Module Contents
---------------

.. py:class:: SMCState



   State of the SMC sampler.

   :param particles: Particles representing samples from the target distribution. Each leaf
                     represents a variable from the posterior, being an array of size
                     `(n_particles, ...)`.
   :type particles: ArrayTree | ArrayLikeTree
   :param weights: Normalized weights for each particle, shape (n_particles,).
   :type weights: Array
   :param update_parameters: Parameters passed to the update function.
   :type update_parameters: ArrayTree

   .. rubric:: Examples

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


   .. py:attribute:: particles
      :type:  blackjax.types.ArrayTree | blackjax.types.ArrayLikeTree


   .. py:attribute:: weights
      :type:  blackjax.types.Array


   .. py:attribute:: update_parameters
      :type:  blackjax.types.ArrayTree


.. py:class:: SMCInfo



   Additional information on the tempered SMC step.

   :param ancestors: The index of the particles proposed by the MCMC pass that were selected
                     by the resampling step.
   :type ancestors: Array
   :param log_likelihood_increment: The log-likelihood increment due to the current step of the SMC algorithm.
   :type log_likelihood_increment: float | Array
   :param update_info: Additional information returned by the update function.
   :type update_info: NamedTuple
   :param ess: Effective sample size after reweighting and before any optional resampling.
   :type ess: float | Array
   :param resampled: Whether the resampling strategy selected a resampling move.
   :type resampled: bool | Array


   .. py:attribute:: ancestors
      :type:  blackjax.types.Array


   .. py:attribute:: log_likelihood_increment
      :type:  float | blackjax.types.Array


   .. py:attribute:: update_info
      :type:  NamedTuple


   .. py:attribute:: ess
      :type:  float | blackjax.types.Array


   .. py:attribute:: resampled
      :type:  bool | blackjax.types.Array


.. py:function:: init(particles: blackjax.types.ArrayLikeTree, init_update_params: blackjax.types.ArrayTree) -> SMCState

   Initialize the SMC state.

   :param particles: Initial particles, typically sampled from the prior.
   :type particles: ArrayLikeTree
   :param init_update_params: Initial parameters for the update function.
   :type init_update_params: ArrayTree

   :returns: Initial state with uniform weights.
   :rtype: SMCState


.. py:function:: step(rng_key: blackjax.types.PRNGKey, state: SMCState, update_fn: Callable, weight_fn: Callable, resample_fn: Callable, num_resampled: Optional[int] = None, resampling_strategy: Optional[Callable] = None) -> tuple[SMCState, SMCInfo]

   General SMC sampling step.

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

   :param rng_key: Key used to generate pseudo-random numbers.
   :type rng_key: PRNGKey
   :param state: Current state of the SMC sampler: particles and their respective weights.
   :type state: SMCState
   :param update_fn: Function that takes an array of keys and particles and returns
                     new particles.
   :type update_fn: Callable
   :param weight_fn: Function that assigns a weight to the particles.
   :type weight_fn: Callable
   :param resample_fn: Particle resampling function. This is used by the default strategy and by
                       any custom strategy that chooses to draw new ancestors.
   :type resample_fn: Callable
   :param num_resampled: The number of particles to resample. This can be used to implement
                         Waste-Free SMC :cite:p:`dau2020waste`, in which case we resample a number
                         :math:`M<N` of particles, and the update function is in charge of returning
                         :math:`N` samples.
   :type num_resampled: int, optional
   :param resampling_strategy: Callable with signature
                               ``(rng_key, weights, num_samples) -> ResamplingDecision``.
                               If ``None``, the step always resamples using ``resample_fn``. Strategies
                               can implement conditional resampling based on ESS or other diagnostics.
   :type resampling_strategy: Callable, optional

   :returns: * **new_state** (*SMCState*) -- The new SMCState containing updated particles and weights.
             * **info** (*SMCInfo*) -- An `SMCInfo` object that contains extra information about the SMC
               transition, including the ESS before optional resampling and whether the
               strategy triggered resampling.


.. py:function:: extend_params(params: blackjax.types.Array) -> blackjax.types.Array

   Extend parameters to be used for all particles in SMC.

   Given a dictionary of params, repeats them for every single particle. The
   expected usage is in cases where the aim is to repeat the same parameters for
   all chains within SMC.

   :param params: Parameters to extend for all particles.
   :type params: Array

   :returns: Extended parameters with an additional dimension for particles.
   :rtype: Array


.. py:function:: update_and_take_last(mcmc_init_fn: Callable, tempered_logposterior_fn: Callable, shared_mcmc_step_fn: Callable, num_mcmc_steps: int, n_particles: int | blackjax.types.Array) -> tuple[Callable, int | blackjax.types.Array]

   Create an MCMC update strategy that runs multiple steps and keeps the last.

   Given N particles, runs num_mcmc_steps of a kernel starting at each particle, and
   returns the last values, wasting the previous num_mcmc_steps-1 samples per chain.

   :param mcmc_init_fn: Function that initializes an MCMC state from a position.
   :type mcmc_init_fn: Callable
   :param tempered_logposterior_fn: Tempered log-posterior probability density function.
   :type tempered_logposterior_fn: Callable
   :param shared_mcmc_step_fn: MCMC step function.
   :type shared_mcmc_step_fn: Callable
   :param num_mcmc_steps: Number of MCMC steps to run for each particle.
   :type num_mcmc_steps: int
   :param n_particles: Number of particles.
   :type n_particles: int | Array

   :returns: * **mcmc_kernel** (*Callable*) -- A vectorized MCMC kernel function.
             * **n_particles** (*int | Array*) -- Number of particles (returned unchanged).



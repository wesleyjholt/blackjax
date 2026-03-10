blackjax.smc.adaptive_tempered
==============================

.. py:module:: blackjax.smc.adaptive_tempered


Attributes
----------

.. autoapisummary::

   blackjax.smc.adaptive_tempered.init


Functions
---------

.. autoapisummary::

   blackjax.smc.adaptive_tempered.build_kernel
   blackjax.smc.adaptive_tempered.as_top_level_api


Module Contents
---------------

.. py:function:: build_kernel(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, target_ess: float, root_solver: Callable = solver.dichotomy, resampling_strategy: Optional[Callable] = None, **extra_parameters: dict[str, Any]) -> Callable

   Build a Tempered SMC step using an adaptive schedule.

   :param logprior_fn: Log prior probability function.
   :type logprior_fn: Callable
   :param loglikelihood_fn: Log likelihood function.
   :type loglikelihood_fn: Callable
   :param mcmc_step_fn: Function that creates MCMC step from log-probability density function.
   :type mcmc_step_fn: Callable
   :param mcmc_init_fn: A function that creates a new mcmc state from a position and a
                        log-probability density function.
   :type mcmc_init_fn: Callable
   :param resampling_fn: Resampling function (from blackjax.smc.resampling).
   :type resampling_fn: Callable
   :param target_ess: Target effective sample size (ESS) to determine the next tempering
                      parameter. This keeps the existing BlackJAX semantics: the solver seeks
                      the next tempering increment such that the ESS after reweighting matches
                      ``target_ess * num_particles`` whenever possible.
   :type target_ess: float | Array
   :param root_solver: The solver used to adaptively compute the temperature given a target number
                       of effective samples. By default, blackjax.smc.solver.dichotomy.
   :type root_solver: Callable, optional
   :param resampling_strategy: Optional policy controlling whether particles are resampled after
                               reweighting at the new temperature. If omitted, callers of
                               ``build_kernel`` get the unconditional-resampling behavior from the base
                               tempered kernel.
   :type resampling_strategy: Callable, optional
   :param \*\*extra_parameters: Additional parameters to pass to tempered.build_kernel.
   :type \*\*extra_parameters: dict[str, Any]

   :returns: **kernel** -- A callable that takes a rng_key, a TemperedSMCState, num_mcmc_steps,
             and mcmc_parameters, and returns a new TemperedSMCState along with
             information about the transition.
   :rtype: Callable


.. py:data:: init

.. py:function:: as_top_level_api(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, mcmc_parameters: dict, resampling_fn: Callable, target_ess: float, root_solver: Callable = solver.dichotomy, num_mcmc_steps: int = 10, resampling_threshold: float | blackjax.types.Array = 0.9, resampling_strategy: Optional[Callable] = None, **extra_parameters: dict[str, Any]) -> blackjax.base.SamplingAlgorithm

   Implements the user interface for the Adaptive Tempered SMC kernel.

   :param logprior_fn: The log-prior function of the model we wish to draw samples from.
   :type logprior_fn: Callable
   :param loglikelihood_fn: The log-likelihood function of the model we wish to draw samples from.
   :type loglikelihood_fn: Callable
   :param mcmc_step_fn: The MCMC step function used to update the particles.
   :type mcmc_step_fn: Callable
   :param mcmc_init_fn: The MCMC init function used to build a MCMC state from a particle position.
   :type mcmc_init_fn: Callable
   :param mcmc_parameters: The parameters of the MCMC step function. Parameters with leading dimension
                           length of 1 are shared amongst the particles.
   :type mcmc_parameters: dict
   :param resampling_fn: The function used to resample the particles.
   :type resampling_fn: Callable
   :param target_ess: Target effective sample size (ESS) to determine the next tempering
                      parameter.
   :type target_ess: float | Array
   :param root_solver: The solver used to adaptively compute the temperature given a target number
                       of effective samples. By default, blackjax.smc.solver.dichotomy.
   :type root_solver: Callable, optional
   :param num_mcmc_steps: The number of times the MCMC kernel is applied to the particles per step,
                          by default 10.
   :type num_mcmc_steps: int, optional
   :param resampling_threshold: ESS threshold used to decide whether resampling is triggered after
                                reweighting. The threshold is interpreted as a fraction of the number of
                                particles, and defaults to ``0.9``.
   :type resampling_threshold: float | Array, optional
   :param resampling_strategy: Optional custom resampling-decision policy. If provided, it overrides
                               ``resampling_threshold`` and receives normalized weights for the
                               reweighted particle system.
   :type resampling_strategy: Callable, optional
   :param \*\*extra_parameters: Additional parameters to pass to the kernel.
   :type \*\*extra_parameters: dict [str, Any]

   :returns: A ``SamplingAlgorithm`` instance with init and step methods. The returned
             step info contains the pre-resampling ESS and a boolean flag indicating
             whether resampling happened.
   :rtype: SamplingAlgorithm



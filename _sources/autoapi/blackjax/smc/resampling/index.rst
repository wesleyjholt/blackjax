blackjax.smc.resampling
=======================

.. py:module:: blackjax.smc.resampling

.. autoapi-nested-parse::

   All things resampling.



Classes
-------

.. autoapisummary::

   blackjax.smc.resampling.ResamplingDecision


Functions
---------

.. autoapisummary::

   blackjax.smc.resampling.systematic
   blackjax.smc.resampling.stratified
   blackjax.smc.resampling.multinomial
   blackjax.smc.resampling.residual
   blackjax.smc.resampling.always
   blackjax.smc.resampling.ess_threshold


Module Contents
---------------

.. py:class:: ResamplingDecision



   Decision returned by a resampling strategy.

   :param ancestors: Indices of the parent particles selected for mutation.
   :param resampled: Boolean flag indicating whether the strategy performed a true resampling
                     move. When this is ``False``, ``ancestors`` is typically the identity map.
   :param ess: Effective sample size of the normalized weights that were used to make
               the decision.


   .. py:attribute:: ancestors
      :type:  blackjax.types.Array


   .. py:attribute:: resampled
      :type:  blackjax.types.Array


   .. py:attribute:: ess
      :type:  float | blackjax.types.Array


.. py:function:: systematic(rng_key: blackjax.types.PRNGKey, weights: blackjax.types.Array, num_samples: int) -> blackjax.types.Array

.. py:function:: stratified(rng_key: blackjax.types.PRNGKey, weights: blackjax.types.Array, num_samples: int) -> blackjax.types.Array

.. py:function:: multinomial(rng_key: blackjax.types.PRNGKey, weights: blackjax.types.Array, num_samples: int) -> blackjax.types.Array

.. py:function:: residual(rng_key: blackjax.types.PRNGKey, weights: blackjax.types.Array, num_samples: int) -> blackjax.types.Array

.. py:function:: always(resampling_fn: Callable) -> Callable

   Build a strategy that always resamples.


.. py:function:: ess_threshold(threshold: float | blackjax.types.Array, resampling_fn: Callable) -> Callable

   Build a strategy that resamples when ESS drops below a threshold.

   The threshold is interpreted as a fraction of the total number of particles.
   If ``num_samples`` differs from the number of particles, resampling is forced.
   This keeps the strategy compatible with update schemes that require drawing a
   strict subset of parents.

   The returned strategy reports the ESS used for the decision and returns the
   identity ancestry when resampling is skipped.



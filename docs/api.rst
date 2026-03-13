.. _api:

Stingray API
************

Library of Time Series Methods For Astronomical X-ray Data.

Base Class
==========
Most `stingray`` classes are subclasses of a single class, :class:`stingray.StingrayObject`, which
implements most of the I/O functionality and common behavior (e.g. strategies to combine data and
make operations such as the sum, difference, or negation). This class is not intended to be
instantiated directly, but rather to be used as a base class for other classes. Any class wanting
to inherit from :class:`stingray.StingrayObject` should define a ``main_array_attr`` attribute, which
defines the name of the attribute that will be used to store the "independent variable" main data array.
For example, for all time series-like objects, the main array is the time array, while for the
periodograms the main array is the frequency array.
All arrays sharing the length (not the shape: they might be multi-dimensional!) of the main array are called
"array attributes" and are accessible through the ``array_attrs`` method.
When applying a mask or any other selection to a :class:`stingray.StingrayObject`,
all array attributes are filtered in the same way. Some array-like attributes might have the same length
by chance, in this case the user or the developer should add these to the ``not_array_attr`` attribute.
For example, :class:`stingray.StingrayTimeseries` has ``gti`` among the not_array_attrs, since it is an
array but not related 1-to-1 to the main array, even if in some cases it might happen to have the same numbers
of elements of the main array, which is ``time``.

StingrayObject
--------------

.. autoclass:: stingray.StingrayObject
   :members:

----

Data Classes
============

These classes define basic functionality related to common data types and typical methods
that apply to these data types, including basic read/write functionality. Currently
implemented are :class:`stingray.StingrayTimeseries`, :class:`stingray.Lightcurve` and
:class:`stingray.events.EventList`.

All time series-like data classes inherit from :class:`stingray.StingrayTimeseries`, which
implements most of the common functionality. The main data array is stored in the ``time``
attribute.
Good Time Intervals (GTIs) are stored in the ``gti`` attribute, which is a list of 2-tuples or 2-lists
containing the start and stop times of each GTI. The ``gti`` attribute is not an array attribute, since
it is not related 1-to-1 to the main array, even if in some cases it might happen to have the same number
of elements of the main array. It is by default added to the ``not_array_attr`` attribute.


StingrayTimeseries
------------------

.. autoclass:: stingray.StingrayTimeseries
   :members:

----

Lightcurve
----------

.. autoclass:: stingray.Lightcurve
   :members:

----

EventList
---------

.. autoclass:: stingray.events.EventList
   :members:

----


Fourier Products
================

These classes implement commonly used Fourier analysis products, most importantly :class:`Crossspectrum` and
:class:`Powerspectrum`, along with the variants for averaged cross/power spectra.

Crossspectrum
-------------

.. autoclass:: stingray.Crossspectrum
   :members:

----

Coherence
---------

Convenience function to compute the coherence between two :class:`stingray.Lightcurve`
objects.

.. autofunction:: stingray.coherence

----

Powerspectrum
-------------

.. autoclass:: stingray.Powerspectrum
   :members:
   :private-members:
   :inherited-members:

----

AveragedCrossspectrum
---------------------

.. autoclass:: stingray.AveragedCrossspectrum
   :members:
   :inherited-members:

----


AveragedPowerspectrum
---------------------

.. autoclass:: stingray.AveragedPowerspectrum
   :members:
   :inherited-members:

----

Dynamical Powerspectrum
-----------------------

.. autoclass:: stingray.DynamicalPowerspectrum
   :members:
   :inherited-members:

----

CrossCorrelation
----------------

.. autoclass:: stingray.CrossCorrelation
   :members:

----

AutoCorrelation
---------------

.. autoclass:: stingray.AutoCorrelation
   :members:
   :inherited-members:

----

Dead-Time Corrections
---------------------

.. automodule:: stingray.deadtime.fad
   :members:
   :imported-members:

.. automodule:: stingray.deadtime.model
   :members:
   :imported-members:

----


Higher-Order Fourier and Spectral Timing Products
=================================================

These classes implement higher-order Fourier analysis products (e.g. :class:`Bispectrum`) and
Spectral Timing related methods taking advantage of both temporal and spectral information in
modern data sets.

Bispectrum
----------

.. autoclass:: stingray.bispectrum.Bispectrum
   :members:

----


Covariancespectrum
------------------

.. autoclass:: stingray.Covariancespectrum
   :members:

----

AveragedCovariancespectrum
--------------------------

.. autoclass:: stingray.AveragedCovariancespectrum
   :members:
   :inherited-members:

----

VarEnergySpectrum
------------------
Abstract base class for spectral timing products including
both variability and spectral information.

.. autoclass:: stingray.varenergyspectrum.VarEnergySpectrum
   :members:

----

RmsEnergySpectrum
-----------------

.. autoclass:: stingray.varenergyspectrum.RmsEnergySpectrum
   :members:
   :inherited-members:

----

LagEnergySpectrum
-----------------

.. autoclass:: stingray.varenergyspectrum.LagEnergySpectrum
   :members:
   :inherited-members:

----

ExcessVarianceSpectrum
----------------------

.. autoclass:: stingray.varenergyspectrum.ExcessVarianceSpectrum
   :members:
   :inherited-members:

----

CountSpectrum
----------------------

.. autoclass:: stingray.varenergyspectrum.CountSpectrum
   :members:
   :inherited-members:

----

Utilities
=========

Commonly used utility functionality, including Good Time Interval operations and input/output
helper methods.

Base Fourier methods
--------------------

.. automodule:: stingray.fourier
   :members:
   :imported-members:

Statistical Functions
---------------------

.. automodule:: stingray.stats
   :members:
   :imported-members:

GTI Functionality
-----------------
.. automodule:: stingray.gti
   :members:
   :imported-members:

I/O Functionality
-----------------

.. automodule:: stingray.io
   :members:

Mission-specific I/O
--------------------

.. automodule:: stingray.mission_support.missions
   :members:

.. automodule:: stingray.mission_support.rxte
   :members:


Other Utility Functions
-----------------------

.. automodule:: stingray.utils
   :members:
   :imported-members:

Modeling
========

This subpackage defines classes and functions related to parametric modelling of various types of
data sets. Currently, most functionality is focused on modelling Fourier products (especially
power spectra and averaged power spectra), but rudimentary functionality exists for modelling
e.g. light curves.


.. _loglikelihoods:

Log-Likelihood Classes
----------------------

These classes define basic log-likelihoods for modelling time series and power spectra.
:class:`stingray.modeling.LogLikelihood` is an abstract base class, i.e. a template for creating
user-defined log-likelihoods and should not be instantiated itself. Based on this base class
are several definitions for a :class:`stingray.modeling.GaussianLogLikelihood`, appropriate for
data with normally distributed uncertainties, a :class:`stingray.modeling.PoissonLogLikelihood`
appropriate for photon counting data, and a :class:`stingray.modeling.PSDLogLikelihood`
appropriate for (averaged) power spectra.

.. autoclass:: stingray.modeling.LogLikelihood
   :members:
   :inherited-members:

.. autoclass:: stingray.modeling.GaussianLogLikelihood
   :members:
   :inherited-members:

.. autoclass:: stingray.modeling.PoissonLogLikelihood
   :members:
   :inherited-members:

.. autoclass:: stingray.modeling.PSDLogLikelihood
   :members:
   :inherited-members:

.. autoclass:: stingray.modeling.LaplaceLogLikelihood
   :members:
   :inherited-members:

----

Posterior Classes
-----------------

These classes define basic posteriors for parametric modelling of time series and power spectra, based on
the log-likelihood classes defined in :ref:`loglikelihoods`. :class:`stingray.modeling.Posterior` is an
abstract base class laying out a basic template for defining posteriors. As with the log-likelihood classes
above, several posterior classes are defined for a variety of data types.

Note that priors are **not** pre-defined in these classes, since they are problem dependent and should be
set by the user. The convenience function :func:`stingray.modeling.set_logprior` can be useful to help set
priors for these posterior classes.

.. autoclass:: stingray.modeling.Posterior
   :members:
   :inherited-members:

.. autoclass:: stingray.modeling.GaussianPosterior
   :members:
   :inherited-members:

.. autoclass:: stingray.modeling.PoissonPosterior
   :members:
   :inherited-members:

.. autoclass:: stingray.modeling.PSDPosterior
   :members:
   :inherited-members:

.. autoclass:: stingray.modeling.LaplacePosterior
   :members:
   :inherited-members:

----

Parameter Estimation Classes
----------------------------

These classes implement functionality related to parameter estimation. They define basic ``fit`` and
``sample`` methods using ``scipy.optimize`` and ``emcee``, respectively, for optimization and Markov Chain Monte
Carlo sampling. :class:`stingray.modeling.PSDParEst` implements some more advanced functionality for modelling
power spectra, including both frequentist and Bayesian searches for (quasi-)periodic signals.

.. autoclass:: stingray.modeling.ParameterEstimation
   :members:

.. autoclass:: stingray.modeling.PSDParEst
   :members:
   :inherited-members:

----

Auxiliary Classes
-----------------

These are helper classes instantiated by :class:`stingray.modeling.ParameterEstimation` and its subclasses to
organize the results of model fitting and sampling in a more meaningful, easily accessible way.

.. autoclass:: stingray.modeling.OptimizationResults
   :members:
   :private-members:

.. autoclass:: stingray.modeling.SamplingResults
   :members:
   :private-members:

----

Convenience Functions
---------------------

These functions are designed to help the user perform common tasks related to modelling and parameter
estimation. In particular, the function :func:`stingray.modeling.set_logprior` is designed to
help users set priors in their :class:`stingray.modeling.Posterior` subclass objects.

.. autofunction:: stingray.modeling.set_logprior

.. automodule:: stingray.modeling.scripts
   :members:
   :imported-members:

----

Pulsar
======

This submodule broadly defines functionality related to (X-ray) pulsar data analysis, especially
periodicity searches.

.. automodule:: stingray.pulse
   :members:
   :imported-members:

Simulator
=========

This submodule defines extensive functionality related to simulating spectral-timing data sets,
including transfer and window functions, simulating light curves from power spectra for a range
of stochastic processes.


.. autoclass:: stingray.simulator.simulator.Simulator
   :members:
   :undoc-members:

Exceptions
==========

Some basic Stingray-related errors and exceptions.

.. autoclass:: stingray.exceptions.StingrayError
   :members:
   :undoc-members:

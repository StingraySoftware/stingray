import abc
import warnings

import numpy as np
from collections.abc import Iterable

np.seterr("warn")

from scipy.special import gammaln as scipy_gammaln

try:
    from astropy.modeling.fitting import fitter_to_model_params
except ImportError:
    from astropy.modeling.fitting import _fitter_to_model_params as fitter_to_model_params

from astropy.modeling import models

from stingray.utils import assign_if_not_finite


# TODO: Add checks and balances to code

# from stingray.modeling.parametricmodels import logmin

__all__ = [
    "set_logprior",
    "Posterior",
    "PSDPosterior",
    "LogLikelihood",
    "PoissonLogLikelihood",
    "PSDLogLikelihood",
    "GaussianLogLikelihood",
    "LaplaceLogLikelihood",
    "PoissonPosterior",
    "GaussianPosterior",
    "LaplacePosterior",
    "PriorUndefinedError",
    "LikelihoodUndefinedError",
]

logmin = -10000000000000000.0


class PriorUndefinedError(Exception):
    pass


class LikelihoodUndefinedError(Exception):
    pass


class IncorrectParameterError(Exception):
    pass


def set_logprior(lpost, priors):
    """
    This function constructs the ``logprior`` method required to successfully
    use a :class:`Posterior` object.

    All instances of class :class:`Posterior` and its subclasses require to implement a
    ``logprior`` methods. However, priors are strongly problem-dependent and
    therefore usually user-defined.

    This function allows for setting the ``logprior`` method on any instance
    of class :class:`Posterior` efficiently by allowing the user to pass a
    dictionary of priors and an instance of class :class:`Posterior`.

    Parameters
    ----------
    lpost : :class:`Posterior` object
        An instance of class :class:`Posterior` or any of its subclasses

    priors : dict
        A dictionary containing the prior definitions. Keys are parameter
        names as defined by the ``astropy.models.FittableModel`` instance supplied
        to the ``model`` parameter in :class:`Posterior`. Items are functions
        that take a parameter as input and return the log-prior probability
        of that parameter.

    Returns
    -------
        logprior : function
            The function definition for the prior

    Examples
    --------
    Make a light curve and power spectrum

    >>> photon_arrivals = np.sort(np.random.uniform(0,1000, size=10000))
    >>> lc = Lightcurve.make_lightcurve(photon_arrivals, dt=1.0)
    >>> ps = Powerspectrum(lc, norm="frac")

    Define the model

    >>> pl = models.PowerLaw1D()
    >>> pl.x_0.fixed = True

    Instantiate the posterior:

    >>> lpost = PSDPosterior(ps.freq, ps.power, pl, m=ps.m)

    Define the priors:

    >>> p_alpha = lambda alpha: ((-1. <= alpha) & (alpha <= 5.))
    >>> p_amplitude = lambda amplitude: ((-10 <= np.log(amplitude)) &
    ...                                 ((np.log(amplitude) <= 10.0)))
    >>> priors = {"alpha":p_alpha, "amplitude":p_amplitude}

    Set the logprior method in the lpost object:

    >>> lpost.logprior = set_logprior(lpost, priors)
    """

    # get the number of free parameters in the model
    # free_params = [p for p in lpost.model.param_names if not
    #                getattr(lpost.model, p).fixed]
    free_params = [key for key, l in lpost.model.fixed.items() if not l]

    # define the logprior
    def logprior(t0, neg=False):
        """
        The logarithm of the prior distribution for the
        model defined in self.model.

        Parameters
        ----------
        t0 : {list | numpy.ndarray}
            The list with parameters for the model

        Returns
        -------
        logp : float
            The logarithm of the prior distribution for the model and
            parameters given.
        """

        if len(t0) != len(free_params):
            raise IncorrectParameterError(
                "The number of parameters passed into "
                "the prior does not match the number "
                "of parameters in the model."
            )

        logp = 0.0  # initialize log-prior
        ii = 0  # counter for the variable parameter

        # loop through all parameter names, but only compute
        # prior for those that are not fixed
        # Note: need to do it this way to preserve order of parameters
        # correctly!
        for pname in lpost.model.param_names:
            if not lpost.model.fixed[pname]:
                with warnings.catch_warnings(record=True) as out:
                    logp += np.log(priors[pname](t0[ii]))
                    if len(out) > 0:
                        if isinstance(out[0].message, RuntimeWarning):
                            logp = np.nan

                ii += 1

        logp = assign_if_not_finite(logp, logmin)

        if neg:
            return -logp
        else:
            return logp

    return logprior


class LogLikelihood(object, metaclass=abc.ABCMeta):
    """

    Abstract Base Class defining the structure of a :class:`LogLikelihood` object.
    This class cannot be called itself, since each statistical distribution
    has its own definition for the likelihood, which should occur in subclasses.

    Parameters
    ----------
    x : iterable
        x-coordinate of the data. Could be multi-dimensional.

    y : iterable
        y-coordinate of the data. Could be multi-dimensional.

    model : an ``astropy.modeling.FittableModel`` instance
        Your model

    kwargs :
        keyword arguments specific to the individual sub-classes. For
        details, see the respective docstrings for each subclass

    """

    def __init__(self, x, y, model, **kwargs):
        self.x = x
        self.y = y

        self.model = model

    @abc.abstractmethod
    def evaluate(self, parameters):
        """
        This is where you define your log-likelihood. Do this, but do it in a subclass!

        """
        pass

    def __call__(self, parameters, neg=False):
        return self.evaluate(parameters, neg)


class GaussianLogLikelihood(LogLikelihood):
    """
    Likelihood for data with Gaussian uncertainties.
    Astronomers also call this likelihood *Chi-Squared*, but be aware
    that this has *nothing* to do with the likelihood based on the
    Chi-square distribution, which is also defined as in of
    :class:`PSDLogLikelihood` in this module!

    Use this class here whenever your data has Gaussian uncertainties.

    Parameters
    ----------
    x : iterable
        x-coordinate of the data

    y : iterable
        y-coordinte of the data

    yerr : iterable
        the uncertainty on the data, as standard deviation

    model : an ``astropy.modeling.FittableModel`` instance
        The model to use in the likelihood.

    Attributes
    ----------
    x : iterable
        x-coordinate of the data

    y : iterable
        y-coordinte of the data

    yerr : iterable
        the uncertainty on the data, as standard deviation

    model : an Astropy Model instance
        The model to use in the likelihood.

    npar : int
        The number of free parameters in the model
    """

    def __init__(self, x, y, yerr, model):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model

        self.npar = 0
        for pname in self.model.param_names:
            if not self.model.fixed[pname]:
                self.npar += 1

    def evaluate(self, pars, neg=False):
        """
        Evaluate the Gaussian log-likelihood for a given set of parameters.

        Parameters
        ----------
        pars : numpy.ndarray
            An array of parameters at which to evaluate the model
            and subsequently the log-likelihood. Note that the
            length of this array must match the free parameters in
            ``model``, i.e. ``npar``

        neg : bool, optional, default ``False``
            If ``True``, return the *negative* log-likelihood, i.e.
            ``-loglike``, rather than ``loglike``. This is useful e.g.
            for optimization routines, which generally minimize
            functions.

        Returns
        -------
        loglike : float
            The log(likelihood) value for the data and model.

        """
        if np.size(pars) != self.npar:
            raise IncorrectParameterError("Input parameters must" + " match model parameters!")

        fitter_to_model_params(self.model, pars)

        mean_model = self.model(self.x)

        loglike = np.sum(
            -0.5 * np.log(2.0 * np.pi)
            - np.log(self.yerr)
            - (self.y - mean_model) ** 2 / (2.0 * self.yerr**2)
        )

        loglike = assign_if_not_finite(loglike, logmin)

        if neg:
            return -loglike
        else:
            return loglike


class PoissonLogLikelihood(LogLikelihood):
    """
    Likelihood for data with uncertainties following a Poisson distribution.
    This is useful e.g. for (binned) photon count data.

    Parameters
    ----------
    x : iterable
        x-coordinate of the data

    y : iterable
        y-coordinte of the data

    model : an ``astropy.modeling.FittableModel`` instance
        The model to use in the likelihood.

    Attributes
    ----------
    x : iterable
        x-coordinate of the data

    y : iterable
        y-coordinte of the data

    yerr : iterable
        the uncertainty on the data, as standard deviation

    model : an ``astropy.modeling.FittableModel`` instance
        The model to use in the likelihood.

    npar : int
        The number of free parameters in the model
    """

    def __init__(self, x, y, model):
        self.x = x
        self.y = y
        self.model = model
        self.npar = 0
        for pname in self.model.param_names:
            if not self.model.fixed[pname]:
                self.npar += 1

    def evaluate(self, pars, neg=False):
        """
        Evaluate the log-likelihood for a given set of parameters.

        Parameters
        ----------
        pars : numpy.ndarray
            An array of parameters at which to evaluate the model
            and subsequently the log-likelihood. Note that the
            length of this array must match the free parameters in
            ``model``, i.e. ``npar``

        neg : bool, optional, default ``False``
            If ``True``, return the *negative* log-likelihood, i.e.
            ``-loglike``, rather than ``loglike``. This is useful e.g.
            for optimization routines, which generally minimize
            functions.

        Returns
        -------
        loglike : float
            The log(likelihood) value for the data and model.

        """
        if np.size(pars) != self.npar:
            raise IncorrectParameterError("Input parameters must" + " match model parameters!")

        fitter_to_model_params(self.model, pars)

        mean_model = self.model(self.x)

        loglike = np.sum(-mean_model + self.y * np.log(mean_model) - scipy_gammaln(self.y + 1.0))

        loglike = assign_if_not_finite(loglike, logmin)

        if neg:
            return -loglike
        else:
            return loglike


class PSDLogLikelihood(LogLikelihood):
    """
    A likelihood based on the Chi-square distribution, appropriate for modelling
    (averaged) power spectra. Note that this is *not* the same as the statistic
    astronomers commonly call *Chi-Square*, which is a fit statistic derived from
    the Gaussian log-likelihood, defined elsewhere in this module.

    Parameters
    ----------
    freq : iterable
        Array with frequencies

    power : iterable
        Array with (averaged/singular) powers corresponding to the
        frequencies in ``freq``

    model : an ``astropy.modeling.FittableModel`` instance
        The model to use in the likelihood.

    m : int
        1/2 of the degrees of freedom

    Attributes
    ----------
    x : iterable
        x-coordinate of the data

    y : iterable
        y-coordinte of the data

    yerr : iterable
        the uncertainty on the data, as standard deviation

    model : an ``astropy.modeling.FittableModel`` instance
        The model to use in the likelihood.

    npar : int
        The number of free parameters in the model
    """

    def __init__(self, freq, power, model, m=1):
        LogLikelihood.__init__(self, freq, power, model)

        self.m = m
        self.npar = 0
        for pname in self.model.param_names:
            if not self.model.fixed[pname] and not self.model.tied[pname]:
                self.npar += 1

    def evaluate(self, pars, neg=False):
        """
        Evaluate the log-likelihood for a given set of parameters.

        Parameters
        ----------
        pars : numpy.ndarray
            An array of parameters at which to evaluate the model
            and subsequently the log-likelihood. Note that the
            length of this array must match the free parameters in
            ``model``, i.e. ``npar``

        neg : bool, optional, default ``False``
            If ``True``, return the *negative* log-likelihood, i.e.
            ``-loglike``, rather than ``loglike``. This is useful e.g.
            for optimization routines, which generally minimize
            functions.

        Returns
        -------
        loglike : float
            The log(likelihood) value for the data and model.

        """
        if np.size(pars) != self.npar:
            raise IncorrectParameterError("Input parameters must" + " match model parameters!")

        fitter_to_model_params(self.model, pars)

        mean_model = self.model(self.x)

        with warnings.catch_warnings(record=True) as out:
            if not isinstance(self.m, Iterable) and self.m == 1:
                loglike = -np.sum(np.log(mean_model)) - np.sum(self.y / mean_model)

            else:
                dof = 2.0 * self.m
                loglike = -(
                    np.sum(dof * np.log(mean_model))
                    + np.sum(dof * self.y / mean_model)
                    + np.sum(dof * (2.0 / dof - 1.0) * np.log(self.y))
                )

        loglike = assign_if_not_finite(loglike, logmin)

        if neg:
            return -loglike
        else:
            return loglike


class LaplaceLogLikelihood(LogLikelihood):
    """
    A Laplace likelihood for the cospectrum.

    Parameters
    ----------
    x : iterable
        Array with independent variable

    y : iterable
        Array with dependent variable

    model : an ``astropy.modeling.FittableModel`` instance
        The model to use in the likelihood.

    yerr : iterable
        Array with the uncertainties on ``y``, in standard deviation

    Attributes
    ----------
    x : iterable
        x-coordinate of the data

    y : iterable
        y-coordinte of the data

    yerr : iterable
        the uncertainty on the data, as standard deviation

    model : an ``astropy.modeling.FittableModel`` instance
        The model to use in the likelihood.

    npar : int
        The number of free parameters in the model
    """

    def __init__(self, x, y, yerr, model):
        LogLikelihood.__init__(self, x, y, model)
        self.yerr = yerr

        self.npar = 0
        for pname in self.model.param_names:
            if not self.model.fixed[pname] and not self.model.tied[pname]:
                self.npar += 1

    def evaluate(self, pars, neg=False):
        """
        Evaluate the log-likelihood for a given set of parameters.

        Parameters
        ----------
        pars : numpy.ndarray
            An array of parameters at which to evaluate the model
            and subsequently the log-likelihood. Note that the
            length of this array must match the free parameters in
            ``model``, i.e. ``npar``

        neg : bool, optional, default ``False``
            If ``True``, return the *negative* log-likelihood, i.e.
            ``-loglike``, rather than ``loglike``. This is useful e.g.
            for optimization routines, which generally minimize
            functions.

        Returns
        -------
        loglike : float
            The log(likelihood) value for the data and model.
        """

        if np.size(pars) != self.npar:
            raise IncorrectParameterError("Input parameters must" + " match model parameters!")

        fitter_to_model_params(self.model, pars)

        mean_model = self.model(self.x)

        with warnings.catch_warnings(record=True) as out:
            loglike = np.sum(-np.log(2.0 * self.yerr) - (np.abs(self.y - mean_model) / self.yerr))

        loglike = assign_if_not_finite(loglike, logmin)

        if neg:
            return -loglike
        else:
            return loglike


class Posterior(object):
    """
    Define a :class:`Posterior` object.

    The :class:`Posterior` describes the Bayesian probability distribution of
    a set of parameters :math:`\\theta` given some observed data :math:`D` and
    some prior assumptions :math:`I`.

    It is defined as

    .. math::

        p(\\theta | D, I) = p(D | \\theta, I) p(\\theta | I)/p(D| I)

    where :math:`p(D | \\theta, I)` describes the likelihood, i.e. the
    sampling distribution of the data and the (parametric) model, and
    :math:`p(\\theta | I)` describes the prior distribution, i.e. our information
    about the parameters :math:`\\theta` before we gathered the data.
    The marginal likelihood :math:`p(D| I)` describes the probability of
    observing the data given the model assumptions, integrated over the
    space of all parameters.

    Parameters
    ----------
    x : iterable
        The abscissa or independent variable of the data. This could
        in principle be a multi-dimensional array.

    y : iterable
        The ordinate or dependent variable of the data.

    model : ``astropy.modeling.models`` instance
        The parametric model supposed to represent the data. For details
        see the ``astropy.modeling`` documentation

    kwargs :
        keyword arguments related to the subclasses of :class:`Posterior`. For
        details, see the documentation of the individual subclasses

    References
    ----------
    * Sivia, D. S., and J. Skilling. "Data Analysis: \
        A Bayesian Tutorial. 2006."
    * Gelman, Andrew, et al. Bayesian data analysis. Vol. 2. Boca Raton, \
        FL, USA: Chapman & Hall/CRC, 2014.
    * von Toussaint, Udo. "Bayesian inference in physics." \
        Reviews of Modern Physics 83.3 (2011): 943.
    * Hogg, David W. "Probability Calculus for inference". \
        arxiv: 1205.4446

    """

    def __init__(self, x, y, model, **kwargs):
        self.x = x
        self.y = y

        self.model = model

        self.npar = 0
        for pname in self.model.param_names:
            if not self.model.fixed[pname]:
                self.npar += 1

    def logposterior(self, t0, neg=False):
        """
        Definition of the log-posterior.
        Requires methods ``loglikelihood`` and ``logprior`` to both
        be defined.

        Note that ``loglikelihood`` is set in the subclass of :class:`Posterior`
        appropriate for your problem at hand, as is ``logprior``.

        Parameters
        ----------
        t0 : numpy.ndarray
            An array of parameters at which to evaluate the model
            and subsequently the log-posterior. Note that the
            length of this array must match the free parameters in
            ``model``, i.e. ``npar``

        neg : bool, optional, default ``False``
            If ``True``, return the *negative* log-posterior, i.e.
            ``-lpost``, rather than ``lpost``. This is useful e.g.
            for optimization routines, which generally minimize
            functions.

        Returns
        -------
        lpost : float
            The value of the log-posterior for the given parameters ``t0``
        """

        if not hasattr(self, "logprior"):
            raise PriorUndefinedError(
                "There is no prior implemented. " + "Cannot calculate posterior!"
            )

        if not hasattr(self, "loglikelihood"):
            raise LikelihoodUndefinedError(
                "There is no likelihood implemented. " + "Cannot calculate posterior!"
            )

        logpr = self.logprior(t0)
        loglike = self.loglikelihood(t0)

        if np.isclose(logpr, logmin):
            lpost = logmin
        else:
            lpost = logpr + loglike

        if neg is True:
            return -lpost
        else:
            return lpost

    def __call__(self, t0, neg=False):
        return self.logposterior(t0, neg=neg)


class PSDPosterior(Posterior):
    """
    :class:`Posterior` distribution for power spectra.
    Uses an exponential distribution for the errors in the likelihood,
    or a :math:`\\chi^2` distribution with :math:`2M` degrees of freedom, where
    :math:`M` is the number of frequency bins or power spectra averaged in each bin.


    Parameters
    ----------
    ps : {:class:`stingray.Powerspectrum` | :class:`stingray.AveragedPowerspectrum`} instance
        the :class:`stingray.Powerspectrum` object containing the data

    model : instance of any subclass of ``astropy.modeling.FittableModel``
        The model for the power spectrum.

    priors : dict of form ``{"parameter name": function}``, optional
        A dictionary with the definitions for the prior probabilities.
        For each parameter in ``model``, there must be a prior defined with
        a key of the exact same name as stored in ``model.param_names``.
        The item for each key is a function definition defining the prior
        (e.g. a lambda function or a ``scipy.stats.distribution.pdf``.
        If ``priors = None``, then no prior is set. This means priors need
        to be added by hand using the :func:`set_logprior` function defined in
        this module. Note that it is impossible to call a :class:`Posterior` object
        itself or the ``self.logposterior`` method without defining a prior.

    m : int, default ``1``
        The number of averaged periodograms or frequency bins in ``ps``.
        Useful for binned/averaged periodograms, since the value of
        m will change the likelihood function!

    Attributes
    ----------
    ps : {:class:`stingray.Powerspectrum` | :class:`stingray.AveragedPowerspectrum`} instance
        the :class:`stingray.Powerspectrum` object containing the data

    x : numpy.ndarray
        The independent variable (list of frequencies) stored in ``ps.freq``

    y : numpy.ndarray
        The dependent variable (list of powers) stored in ``ps.power``

    model : instance of any subclass of ``astropy.modeling.FittableModel``
        The model for the power spectrum.

    """

    def __init__(self, freq, power, model, priors=None, m=1):
        self.loglikelihood = PSDLogLikelihood(freq, power, model, m=m)

        self.m = m
        Posterior.__init__(self, freq, power, model)

        if not priors is None:
            self.logprior = set_logprior(self, priors)


class PoissonPosterior(Posterior):
    """
    :class:`Posterior` for Poisson light curve data. Primary intended use is for
    modelling X-ray light curves, but alternative uses are conceivable.

    Parameters
    ----------
    x : numpy.ndarray
        The independent variable (e.g. time stamps of a light curve)

    y : numpy.ndarray
        The dependent variable (e.g. counts per bin of a light curve)

    model : instance of any subclass of ``astropy.modeling.FittableModel``
        The model for the power spectrum.

    priors : dict of form ``{"parameter name": function}``, optional
        A dictionary with the definitions for the prior probabilities.
        For each parameter in ``model``, there must be a prior defined with
        a key of the exact same name as stored in ``model.param_names``.
        The item for each key is a function definition defining the prior
        (e.g. a lambda function or a ``scipy.stats.distribution.pdf``.
        If ``priors = None``, then no prior is set. This means priors need
        to be added by hand using the :func:`set_logprior` function defined in
        this module. Note that it is impossible to call a :class:`Posterior` object
        itself or the ``self.logposterior`` method without defining a prior.

    Attributes
    ----------
    x : numpy.ndarray
        The independent variable (list of frequencies) stored in ps.freq

    y : numpy.ndarray
        The dependent variable (list of powers) stored in ps.power

    model : instance of any subclass of ``astropy.modeling.FittableModel``
        The model for the power spectrum.

    """

    def __init__(self, x, y, model, priors=None):
        self.x = x
        self.y = y

        self.loglikelihood = PoissonLogLikelihood(self.x, self.y, model)

        Posterior.__init__(self, self.x, self.y, model)

        if not priors is None:
            self.logprior = set_logprior(self, priors)


class GaussianPosterior(Posterior):
    """
    A general class for two-dimensional data following a Gaussian
    sampling distribution.

    Parameters
    ----------
    x : numpy.ndarray
        independent variable

    y : numpy.ndarray
        dependent variable

    yerr : numpy.ndarray
        measurement uncertainties for y

    model : instance of any subclass of ``astropy.modeling.FittableModel``
        The model for the power spectrum.

    priors : dict of form ``{"parameter name": function}``, optional
        A dictionary with the definitions for the prior probabilities.
        For each parameter in ``model``, there must be a prior defined with
        a key of the exact same name as stored in ``model.param_names``.
        The item for each key is a function definition defining the prior
        (e.g. a lambda function or a ``scipy.stats.distribution.pdf``.
        If ``priors = None``, then no prior is set. This means priors need
        to be added by hand using the :func:`set_logprior` function defined in
        this module. Note that it is impossible to call a :class:`Posterior` object
        itself or the ``self.logposterior`` method without defining a prior.

    """

    def __init__(self, x, y, yerr, model, priors=None):
        self.loglikelihood = GaussianLogLikelihood(x, y, yerr, model)

        Posterior.__init__(self, x, y, model)

        self.yerr = yerr

        if not priors is None:
            self.logprior = set_logprior(self, priors)


class LaplacePosterior(Posterior):
    """
    A general class for two-dimensional data following a Gaussian
    sampling distribution.

    Parameters
    ----------
    x : numpy.ndarray
        independent variable

    y : numpy.ndarray
        dependent variable

    yerr : numpy.ndarray
        measurement uncertainties for y, in standard deviation

    model : instance of any subclass of ``astropy.modeling.FittableModel``
        The model for the power spectrum.

    priors : dict of form ``{"parameter name": function}``, optional
        A dictionary with the definitions for the prior probabilities.
        For each parameter in ``model``, there must be a prior defined with
        a key of the exact same name as stored in ``model.param_names``.
        The item for each key is a function definition defining the prior
        (e.g. a lambda function or a ``scipy.stats.distribution.pdf``.
        If ``priors = None``, then no prior is set. This means priors need
        to be added by hand using the :func:`set_logprior` function defined in
        this module. Note that it is impossible to call a :class:`Posterior` object
        itself or the ``self.logposterior`` method without defining a prior.

    """

    def __init__(self, x, y, yerr, model, priors=None):
        self.loglikelihood = LaplaceLogLikelihood(x, y, yerr, model)

        Posterior.__init__(self, x, y, model)

        self.yerr = yerr

        if not priors is None:
            self.logprior = set_logprior(self, priors)

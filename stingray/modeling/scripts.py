import numpy as np
from astropy.modeling import models

from stingray.modeling import PSDParEst, PSDPosterior, PSDLogLikelihood
from stingray.modeling import GaussianPosterior, GaussianLogLikelihood

__all__ = ["fit_powerspectrum", "fit_crossspectrum", "fit_lorentzians"]


def fit_powerspectrum(
    ps, model, starting_pars=None, max_post=False, priors=None, fitmethod="L-BFGS-B"
):
    """
    Fit a number of Lorentzians to a power spectrum, possibly including white
    noise. Each Lorentzian has three parameters (amplitude, centroid position,
    full-width at half maximum), plus one extra parameter if the white noise
    level should be fit as well. Priors for each parameter can be included in
    case `max_post = True`, in which case the function will attempt a
    Maximum-A-Posteriori fit. Priors must be specified as a dictionary with one
    entry for each parameter.
    The parameter names are `(amplitude_i, x_0_i, fwhm_i)` for each `i` out of
    a total of `N` Lorentzians. The white noise level has a parameter
    `amplitude_(N+1)`. For example, a model with two Lorentzians and a
    white noise level would have parameters:
    [amplitude_0, x_0_0, fwhm_0, amplitude_1, x_0_1, fwhm_1, amplitude_2].

    Parameters
    ----------
    ps : Powerspectrum
        A Powerspectrum object with the data to be fit

    model: astropy.modeling.models class instance
        The parametric model supposed to represent the data. For details
        see the astropy.modeling documentation

    starting_pars : iterable, optional, default None
        The list of starting guesses for the optimizer. If it is not provided,
        then default parameters are taken from `model`. See explanation above
        for ordering of parameters in this list.

    fit_whitenoise : bool, optional, default True
        If True, the code will attempt to fit a white noise level along with
        the Lorentzians. Be sure to include a starting parameter for the
        optimizer in `starting_pars`!

    max_post : bool, optional, default False
        If True, perform a Maximum-A-Posteriori fit of the data rather than a
        Maximum Likelihood fit. Note that this requires priors to be specified,
        otherwise this will cause an exception!

    priors : {dict | None}, optional, default None
        Dictionary with priors for the MAP fit. This should be of the form
        {"parameter name": probability distribution, ...}

    fitmethod : string, optional, default "L-BFGS-B"
        Specifies an optimization algorithm to use. Supply any valid option for
        `scipy.optimize.minimize`.

    Returns
    -------
    parest : PSDParEst object
        A PSDParEst object for further analysis

    res : OptimizationResults object
        The OptimizationResults object storing useful results and quantities
        relating to the fit

    Examples
    --------

    We start by making an example power spectrum with three Lorentzians
    >>> m = 1
    >>> nfreq = 100000
    >>> freq = np.linspace(1, 1000, nfreq)

    >>> np.random.seed(100)  # set the seed for the random number generator
    >>> noise = np.random.exponential(size=nfreq)

    >>> model = models.PowerLaw1D() + models.Const1D()
    >>> model.x_0_0.fixed = True

    >>> alpha_0 = 2.0
    >>> amplitude_0 = 100.0
    >>> amplitude_1 = 2.0

    >>> model.alpha_0 = alpha_0
    >>> model.amplitude_0 = amplitude_0
    >>> model.amplitude_1 = amplitude_1

    >>> p = model(freq)
    >>> power = noise * p

    >>> ps = Powerspectrum()
    >>> ps.freq = freq
    >>> ps.power = power
    >>> ps.m = m
    >>> ps.df = freq[1] - freq[0]
    >>> ps.norm = "leahy"

    Now we have to guess starting parameters. For each Lorentzian, we have
    amplitude, centroid position and fwhm, and this pattern repeats for each
    Lorentzian in the fit. The white noise level is the last parameter.
    >>> t0 = [80, 1.5, 2.5]

    Let's also make a model to test:
    >>> model_to_test = models.PowerLaw1D() + models.Const1D()
    >>> model_to_test.amplitude_1.fixed = True

    We're ready for doing the fit:
    >>> parest, res = fit_powerspectrum(ps, model_to_test, t0)

    `res` contains a whole array of useful information about the fit, for
    example the parameters at the optimum:
    >>> p_opt = res.p_opt

    """
    if not (isinstance(starting_pars, np.ndarray) or isinstance(starting_pars, list)):
        starting_pars = model.parameters

    if priors:
        lpost = PSDPosterior(ps.freq, ps.power, model, priors=priors, m=ps.m)
    else:
        lpost = PSDLogLikelihood(ps.freq, ps.power, model, m=ps.m)

    parest = PSDParEst(ps, fitmethod=fitmethod, max_post=max_post)
    res = parest.fit(lpost, starting_pars, neg=True)

    return parest, res


def fit_crossspectrum(
    cs, model, starting_pars=None, max_post=False, priors=None, fitmethod="L-BFGS-B"
):
    """
    Fit a number of Lorentzians to a cross spectrum, possibly including white
    noise. Each Lorentzian has three parameters (amplitude, centroid position,
    full-width at half maximum), plus one extra parameter if the white noise
    level should be fit as well. Priors for each parameter can be included in
    case `max_post = True`, in which case the function will attempt a
    Maximum-A-Posteriori fit. Priors must be specified as a dictionary with one
    entry for each parameter.
    The parameter names are `(amplitude_i, x_0_i, fwhm_i)` for each `i` out of
    a total of `N` Lorentzians. The white noise level has a parameter
    `amplitude_(N+1)`. For example, a model with two Lorentzians and a
    white noise level would have parameters:
    [amplitude_0, x_0_0, fwhm_0, amplitude_1, x_0_1, fwhm_1, amplitude_2].

    Parameters
    ----------
    cs : Crossspectrum
        A Crossspectrum object with the data to be fit

    model: astropy.modeling.models class instance
        The parametric model supposed to represent the data. For details
        see the astropy.modeling documentation

    starting_pars : iterable, optional, default None
        The list of starting guesses for the optimizer. If it is not provided,
        then default parameters are taken from `model`. See explanation above
        for ordering of parameters in this list.

    max_post : bool, optional, default False
        If True, perform a Maximum-A-Posteriori fit of the data rather than a
        Maximum Likelihood fit. Note that this requires priors to be specified,
        otherwise this will cause an exception!

    priors : {dict | None}, optional, default None
        Dictionary with priors for the MAP fit. This should be of the form
        {"parameter name": probability distribution, ...}

    fitmethod : string, optional, default "L-BFGS-B"
        Specifies an optimization algorithm to use. Supply any valid option for
        `scipy.optimize.minimize`.

    Returns
    -------
    parest : PSDParEst object
        A PSDParEst object for further analysis

    res : OptimizationResults object
        The OptimizationResults object storing useful results and quantities
        relating to the fit
    """
    if not (isinstance(starting_pars, np.ndarray) or isinstance(starting_pars, list)):
        starting_pars = model.parameters
    if priors:
        lgauss = GaussianPosterior(cs.freq, np.abs(cs.power), cs.power_err, model, priors)
    else:
        lgauss = GaussianLogLikelihood(cs.freq, np.abs(cs.power), model=model, yerr=cs.power_err)

    parest = PSDParEst(cs, fitmethod=fitmethod, max_post=max_post)
    res = parest.fit(lgauss, starting_pars, neg=True)

    return parest, res


def fit_lorentzians(
    ps, nlor, starting_pars, fit_whitenoise=True, max_post=False, priors=None, fitmethod="L-BFGS-B"
):
    """
    Fit a number of Lorentzians to a power spectrum, possibly including white
    noise. Each Lorentzian has three parameters (amplitude, centroid position,
    full-width at half maximum), plus one extra parameter if the white noise
    level should be fit as well. Priors for each parameter can be included in
    case `max_post = True`, in which case the function will attempt a
    Maximum-A-Posteriori fit. Priors must be specified as a dictionary with one
    entry for each parameter.
    The parameter names are `(amplitude_i, x_0_i, fwhm_i)` for each `i` out of
    a total of `N` Lorentzians. The white noise level has a parameter
    `amplitude_(N+1)`. For example, a model with two Lorentzians and a
    white noise level would have parameters:
    [amplitude_0, x_0_0, fwhm_0, amplitude_1, x_0_1, fwhm_1, amplitude_2].

    Parameters
    ----------
    ps : Powerspectrum
        A Powerspectrum object with the data to be fit

    nlor : int
        The number of Lorentzians to fit

    starting_pars : iterable
        The list of starting guesses for the optimizer. If it is not provided,
        then default parameters are taken from `model`. See explanation above
        for ordering of parameters in this list.

    fit_whitenoise : bool, optional, default True
        If True, the code will attempt to fit a white noise level along with
        the Lorentzians. Be sure to include a starting parameter for the
        optimizer in `starting_pars`!

    max_post : bool, optional, default False
        If True, perform a Maximum-A-Posteriori fit of the data rather than a
        Maximum Likelihood fit. Note that this requires priors to be specified,
        otherwise this will cause an exception!

    priors : {dict | None}, optional, default None
        Dictionary with priors for the MAP fit. This should be of the form
        {"parameter name": probability distribution, ...}

    fitmethod : string, optional, default "L-BFGS-B"
        Specifies an optimization algorithm to use. Supply any valid option for
        `scipy.optimize.minimize`.

    Returns
    -------
    parest : PSDParEst object
        A PSDParEst object for further analysis

    res : OptimizationResults object
        The OptimizationResults object storing useful results and quantities
        relating to the fit

    Examples
    --------

    We start by making an example power spectrum with three Lorentzians
    >>> np.random.seed(400)
    >>> nlor = 3

    >>> x_0_0 = 0.5
    >>> x_0_1 = 2.0
    >>> x_0_2 = 7.5

    >>> amplitude_0 = 150.0
    >>> amplitude_1 = 50.0
    >>> amplitude_2 = 15.0

    >>> fwhm_0 = 0.1
    >>> fwhm_1 = 1.0
    >>> fwhm_2 = 0.5

    We will also include a white noise level:
    >>> whitenoise = 2.0

    >>> model = (models.Lorentz1D(amplitude_0, x_0_0, fwhm_0) +
    ...          models.Lorentz1D(amplitude_1, x_0_1, fwhm_1) +
    ...          models.Lorentz1D(amplitude_2, x_0_2, fwhm_2) +
    ...          models.Const1D(whitenoise))

    >>> freq = np.linspace(0.01, 10.0, 1000)
    >>> p = model(freq)
    >>> noise = np.random.exponential(size=len(freq))

    >>> power = p*noise
    >>> ps = Powerspectrum()
    >>> ps.freq = freq
    >>> ps.power = power
    >>> ps.df = ps.freq[1] - ps.freq[0]
    >>> ps.m = 1

    Now we have to guess starting parameters. For each Lorentzian, we have
    amplitude, centroid position and fwhm, and this pattern repeats for each
    Lorentzian in the fit. The white noise level is the last parameter.
    >>> t0 = [150, 0.4, 0.2, 50, 2.3, 0.6, 20, 8.0, 0.4, 2.1]

    We're ready for doing the fit:
    >>> parest, res = fit_lorentzians(ps, nlor, t0)

    `res` contains a whole array of useful information about the fit, for
    example the parameters at the optimum:
    >>> p_opt = res.p_opt

    """

    model = models.Lorentz1D()

    if nlor > 1:
        for i in range(nlor - 1):
            model += models.Lorentz1D()

    if fit_whitenoise:
        model += models.Const1D()

    return fit_powerspectrum(
        ps, model, starting_pars, max_post=max_post, priors=priors, fitmethod=fitmethod
    )

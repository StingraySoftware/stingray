import numpy as np
import scipy.stats
import copy

import pytest
from astropy.modeling import models
from scipy.special import gammaln as scipy_gammaln

from stingray import Lightcurve, Powerspectrum
from stingray.modeling import (
    Posterior,
    PSDPosterior,
    PoissonPosterior,
    GaussianPosterior,
    LaplacePosterior,
)

from stingray.modeling import set_logprior
from stingray.modeling.posterior import logmin
from stingray.modeling.posterior import IncorrectParameterError
from stingray.modeling.posterior import LogLikelihood

np.random.seed(20150907)


class TestMeta(object):
    def test_use_loglikelihood_class_directly(self):
        with pytest.raises(TypeError):
            a = LogLikelihood(1, 2, models.Lorentz1D)

    def test_inherit_loglikelihood_improperly(self):
        class a(LogLikelihood):
            def __init__(self, *args, **kwargs):
                LogLikelihood.__init__(self, *args, **kwargs)

        with pytest.raises(TypeError):
            a(1, 2, models.Lorentz1D)

    def test_inherit_loglikelihood_properly(self):
        class a(LogLikelihood):
            def __init__(self, *args, **kwargs):
                LogLikelihood.__init__(self, *args, **kwargs)

            def evaluate(self, parameters):
                pass

        a(1, 2, models.Lorentz1D)


class TestSetPrior(object):
    @classmethod
    def setup_class(cls):
        photon_arrivals = np.sort(np.random.uniform(0, 1000, size=10000))
        cls.lc = Lightcurve.make_lightcurve(photon_arrivals, dt=1.0)
        cls.ps = Powerspectrum(cls.lc, norm="frac")
        pl = models.PowerLaw1D()
        pl.x_0.fixed = True

        cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power, pl, m=cls.ps.m)

    def test_set_prior_runs(self):
        p_alpha = lambda alpha: ((-1.0 <= alpha) & (alpha <= 5.0)) / 6.0
        p_amplitude = lambda amplitude: (
            ((-10 <= np.log(amplitude)) & (np.log(amplitude) <= 10.0)) / 20.0
        )

        priors = {"alpha": p_alpha, "amplitude": p_amplitude}
        self.lpost.logprior = set_logprior(self.lpost, priors)

    def test_prior_executes_correctly(self):
        p_alpha = lambda alpha: ((-1.0 <= alpha) & (alpha <= 5.0)) / 6.0
        p_amplitude = lambda amplitude: (
            ((-10 <= np.log(amplitude)) & (np.log(amplitude) <= 10.0)) / 20.0
        )

        priors = {"alpha": p_alpha, "amplitude": p_amplitude}
        self.lpost.logprior = set_logprior(self.lpost, priors)
        true_logprior = np.log(1.0 / 6.0) + np.log(1.0 / 20.0)
        assert self.lpost.logprior([np.exp(0.0), np.exp(0.0)]) == true_logprior

    def test_prior_returns_logmin_outside_prior_range(self):
        p_alpha = lambda alpha: ((-1.0 <= alpha) & (alpha <= 5.0)) / 6.0
        p_amplitude = lambda amplitude: (
            ((-10 <= np.log(amplitude)) & (np.log(amplitude) <= 10.0)) / 20.0
        )

        priors = {"alpha": p_alpha, "amplitude": p_amplitude}
        self.lpost.logprior = set_logprior(self.lpost, priors)
        assert self.lpost.logprior([-2.0, np.exp(11.0)]) == logmin


class PosteriorClassDummy(Posterior):
    """
    This is a test class that tests the basic functionality of the
    Posterior superclass.
    """

    def __init__(self, x, y, model):
        Posterior.__init__(self, x, y, model)

    def loglikelihood(self, t0, neg=False):
        loglike = 1.0
        return loglike

    def logprior(self, t0):
        lp = 2.0
        return lp


class TestPosterior(object):
    @classmethod
    def setup_class(cls):
        cls.x = np.arange(100)
        cls.y = np.ones(cls.x.shape[0])
        cls.model = models.Const1D()
        cls.p = PosteriorClassDummy(cls.x, cls.y, cls.model)
        p_alpha = lambda alpha: ((-1.0 <= alpha) & (alpha <= 5.0)) / 6.0

        priors = {"amplitude": p_alpha}
        cls.p.logprior = set_logprior(cls.p, priors)

    def test_inputs(self):
        assert np.allclose(self.p.x, self.x)
        assert np.allclose(self.p.y, self.y)
        assert isinstance(self.p.model, models.Const1D)

    def test_call_method_positive(self):
        t0 = [1]
        post = self.p(t0, neg=False)
        assert post == 1.0 + np.log(1.0 / 6.0)

    def test_call_method_negative(self):
        t0 = [1]
        post = self.p(t0, neg=True)
        assert post == -(1.0 + np.log(1.0 / 6.0))


class TestPSDPosterior(object):
    @classmethod
    def setup_class(cls):
        m = 1
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = np.random.exponential(size=nfreq)
        power = noise * 2.0

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(
            amplitude
        )

        cls.priors = {"amplitude": p_amplitude}

    def test_logprior_fails_without_prior(self):
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)

        with pytest.raises(AttributeError):
            lpost.logprior([1])

    def test_making_posterior(self):
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        assert lpost.x.all() == self.ps.freq.all()
        assert lpost.y.all() == self.ps.power.all()

    def test_correct_number_of_parameters(self):
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        with pytest.raises(IncorrectParameterError):
            lpost([2, 3])

    def test_logprior(self):
        t0 = [2.0]

        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        lp_test = lpost.logprior(t0)
        lp = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        assert lp == lp_test

    def test_loglikelihood(self):
        t0 = [2.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.ps.freq)

        loglike = -np.sum(np.log(mean_model)) - np.sum(self.ps.power / mean_model)

        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=False)

        assert np.isclose(loglike, loglike_test)

    def test_negative_loglikelihood(self):
        t0 = [2.0]
        self.model.amplitude = t0[0]
        m = self.model(self.ps.freq[1:])
        loglike = np.sum(self.ps.power[1:] / m + np.log(m))

        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=True)

        assert np.isclose(loglike, loglike_test)

    def test_posterior(self):
        t0 = [2.0]
        self.model.amplitude = t0[0]

        m = self.model(self.ps.freq[1:])
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=False)

        loglike = -np.sum(self.ps.power[1:] / m + np.log(m))
        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = loglike + logprior

        assert np.isclose(post_test, post, atol=1.0e-10)

    def test_negative_posterior(self):
        t0 = [2.0]
        self.model.amplitude = t0[0]

        m = self.model(self.ps.freq[1:])
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=True)

        loglike = -np.sum(self.ps.power[1:] / m + np.log(m))
        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = -loglike - logprior

        assert np.isclose(post_test, post, atol=1.0e-10)


class TestPoissonPosterior(object):
    @classmethod
    def setup_class(cls):
        nx = 1000000
        cls.x = np.arange(nx)
        cls.countrate = 10.0
        cls.y = np.random.poisson(cls.countrate, size=cls.x.shape[0])

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: scipy.stats.norm(
            loc=cls.countrate, scale=cls.countrate
        ).pdf(amplitude)

        cls.priors = {"amplitude": p_amplitude}

    def test_logprior_fails_without_prior(self):
        lpost = PoissonPosterior(self.x, self.y, self.model)

        with pytest.raises(AttributeError):
            lpost.logprior([10])

    def test_making_posterior(self):
        lpost = PoissonPosterior(self.x, self.y, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        assert lpost.x.all() == self.x.all()
        assert lpost.y.all() == self.y.all()

    def test_correct_number_of_parameters(self):
        lpost = PoissonPosterior(self.x, self.y, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        with pytest.raises(IncorrectParameterError):
            lpost([2, 3])

    def test_logprior(self):
        t0 = [10.0]

        lpost = PoissonPosterior(self.x, self.y, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        lp_test = lpost.logprior(t0)
        lp = np.log(scipy.stats.norm(self.countrate, self.countrate).pdf(t0))
        assert lp == lp_test

    def test_loglikelihood(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        loglike = np.sum(-mean_model + self.y * np.log(mean_model) - scipy_gammaln(self.y + 1))

        lpost = PoissonPosterior(self.x, self.y, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=False)

        assert np.isclose(loglike, loglike_test)

    def test_negative_loglikelihood(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        loglike = -np.sum(-mean_model + self.y * np.log(mean_model) - scipy_gammaln(self.y + 1))

        lpost = PoissonPosterior(self.x, self.y, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=True)

        assert np.isclose(loglike, loglike_test)

    def test_posterior(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        lpost = PoissonPosterior(self.x, self.y, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=False)

        loglike = np.sum(-mean_model + self.y * np.log(mean_model) - scipy_gammaln(self.y + 1))
        logprior = np.log(scipy.stats.norm(self.countrate, self.countrate).pdf(t0))

        post = loglike + logprior

        assert np.isclose(post_test, post, atol=1.0e-10)

    def test_negative_posterior(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        lpost = PoissonPosterior(self.x, self.y, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=True)

        loglike = np.sum(-mean_model + self.y * np.log(mean_model) - scipy_gammaln(self.y + 1))
        logprior = np.log(scipy.stats.norm(self.countrate, self.countrate).pdf(t0))

        post = -loglike - logprior

        assert np.isclose(post_test, post, atol=1.0e-10)

    def test_counts_are_nan(self):
        y = np.nan * np.ones(self.x.shape[0])

        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        lpost = PoissonPosterior(self.x, y, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        assert np.isclose(lpost(t0), logmin, 1e-5)


class TestGaussianPosterior(object):
    @classmethod
    def setup_class(cls):
        nx = 1000000
        cls.x = np.arange(nx)
        cls.countrate = 10.0
        cls.cerr = 2.0
        cls.y = np.random.normal(cls.countrate, cls.cerr, size=cls.x.shape[0])
        cls.yerr = np.ones_like(cls.y) * cls.cerr

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: scipy.stats.norm(loc=cls.countrate, scale=cls.cerr).pdf(
            amplitude
        )

        cls.priors = {"amplitude": p_amplitude}

    def test_logprior_fails_without_prior(self):
        lpost = GaussianPosterior(self.x, self.y, self.yerr, self.model)

        with pytest.raises(AttributeError):
            lpost.logprior([10])

    def test_making_posterior(self):
        lpost = GaussianPosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        assert lpost.x.all() == self.x.all()
        assert lpost.y.all() == self.y.all()

    def test_correct_number_of_parameters(self):
        lpost = GaussianPosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        with pytest.raises(IncorrectParameterError):
            lpost([2, 3])

    def test_logprior(self):
        t0 = [10.0]

        lpost = GaussianPosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        lp_test = lpost.logprior(t0)
        lp = np.log(scipy.stats.norm(self.countrate, self.cerr).pdf(t0))
        assert lp == lp_test

    def test_loglikelihood(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        loglike = np.sum(
            -0.5 * np.log(2.0 * np.pi)
            - np.log(self.yerr)
            - 0.5 * ((self.y - mean_model) / self.yerr) ** 2.0
        )

        lpost = GaussianPosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=False)

        assert np.isclose(loglike, loglike_test)

    def test_negative_loglikelihood(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        loglike = -np.sum(
            -0.5 * np.log(2.0 * np.pi)
            - np.log(self.yerr)
            - 0.5 * ((self.y - mean_model) / self.yerr) ** 2.0
        )

        lpost = GaussianPosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=True)

        assert np.isclose(loglike, loglike_test)

    def test_posterior(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        lpost = GaussianPosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=False)

        loglike = np.sum(
            -0.5 * np.log(2.0 * np.pi)
            - np.log(self.yerr)
            - 0.5 * ((self.y - mean_model) / self.yerr) ** 2.0
        )
        logprior = np.log(scipy.stats.norm(self.countrate, self.cerr).pdf(t0))

        post = loglike + logprior

        assert np.isclose(post_test, post, atol=1.0e-10)

    def test_negative_posterior(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        lpost = GaussianPosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=True)

        loglike = np.sum(
            -0.5 * np.log(2.0 * np.pi)
            - np.log(self.yerr)
            - 0.5 * ((self.y - mean_model) / self.yerr) ** 2.0
        )
        logprior = np.log(scipy.stats.norm(self.countrate, self.cerr).pdf(t0))

        post = -loglike - logprior

        assert np.isclose(post_test, post, atol=1.0e-10)

    def test_counts_are_nan(self):
        y = np.nan * np.ones(self.x.shape[0])

        t0 = [10.0]
        self.model.amplitude = t0[0]

        lpost = GaussianPosterior(self.x, y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        assert np.isclose(lpost(t0), logmin, 1e-5)


class TestLaplacePosterior(object):
    @classmethod
    def setup_class(cls):
        nx = 1000000
        cls.x = np.arange(nx)
        cls.countrate = 10.0
        cls.cerr = 2.0
        cls.y = np.random.normal(cls.countrate, cls.cerr, size=cls.x.shape[0])
        cls.yerr = np.ones_like(cls.y) * cls.cerr

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: scipy.stats.norm(loc=cls.countrate, scale=cls.cerr).pdf(
            amplitude
        )

        cls.priors = {"amplitude": p_amplitude}

    def test_logprior_fails_without_prior(self):
        lpost = LaplacePosterior(self.x, self.y, self.yerr, self.model)

        with pytest.raises(AttributeError):
            lpost.logprior([10])

    def test_making_posterior(self):
        lpost = LaplacePosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        assert lpost.x.all() == self.x.all()
        assert lpost.y.all() == self.y.all()

    def test_correct_number_of_parameters(self):
        lpost = LaplacePosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        with pytest.raises(IncorrectParameterError):
            lpost([2, 3])

    def test_logprior(self):
        t0 = [10.0]

        lpost = LaplacePosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        lp_test = lpost.logprior(t0)
        lp = np.log(scipy.stats.norm(self.countrate, self.cerr).pdf(t0))
        assert lp == lp_test

    def test_loglikelihood(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        loglike = np.sum(-np.log(2.0 * self.yerr) - np.abs(self.y - mean_model) / self.yerr)

        lpost = LaplacePosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=False)

        assert np.isclose(loglike, loglike_test)

    def test_negative_loglikelihood(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        loglike = -np.sum(-np.log(2.0 * self.yerr) - np.abs(self.y - mean_model) / self.yerr)

        lpost = LaplacePosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=True)

        assert np.isclose(loglike, loglike_test)

    def test_posterior(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        lpost = LaplacePosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=False)

        loglike = np.sum(-np.log(2.0 * self.yerr) - np.abs(self.y - mean_model) / self.yerr)

        logprior = np.log(scipy.stats.norm(self.countrate, self.cerr).pdf(t0))

        post = loglike + logprior

        assert np.isclose(post_test, post, atol=1.0e-10)

    def test_negative_posterior(self):
        t0 = [10.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.x)

        lpost = LaplacePosterior(self.x, self.y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=True)

        loglike = np.sum(-np.log(2.0 * self.yerr) - np.abs(self.y - mean_model) / self.yerr)

        logprior = np.log(scipy.stats.norm(self.countrate, self.cerr).pdf(t0))

        post = -loglike - logprior

        assert np.isclose(post_test, post, atol=1.0e-10)

    def test_counts_are_nan(self):
        y = np.nan * np.ones(self.x.shape[0])

        t0 = [10.0]
        self.model.amplitude = t0[0]

        lpost = LaplacePosterior(self.x, y, self.yerr, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        assert np.isclose(lpost(t0), logmin, 1e-5)


class TestPerPosteriorAveragedPeriodogram(object):
    @classmethod
    def setup_class(cls):
        cls.m = 10
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = scipy.stats.chi2(2.0 * cls.m).rvs(size=nfreq) / float(cls.m)
        power = noise

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = cls.m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(
            amplitude
        )

        cls.priors = {"amplitude": p_amplitude}

    def test_logprior_fails_without_prior(self):
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)

        with pytest.raises(AttributeError):
            lpost.logprior([1])

    def test_making_posterior(self):
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        assert lpost.x.all() == self.ps.freq.all()
        assert lpost.y.all() == self.ps.power.all()

    def test_correct_number_of_parameters(self):
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        with pytest.raises(IncorrectParameterError):
            lpost([2, 3])

    def test_logprior(self):
        t0 = [2.0]

        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        lp_test = lpost.logprior(t0)
        lp = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        assert lp == lp_test

    def test_loglikelihood(self):
        t0 = [2.0]
        self.model.amplitude = t0[0]
        mean_model = self.model(self.ps.freq)

        loglike = (
            -2.0
            * self.m
            * (
                np.sum(np.log(mean_model))
                + np.sum(self.ps.power / mean_model)
                + np.sum((2.0 / (2.0 * self.m) - 1.0) * np.log(self.ps.power))
            )
        )

        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=False)

        assert np.isclose(loglike, loglike_test)

    def test_negative_loglikelihood(self):
        t0 = [2.0]
        self.model.amplitude = t0[0]

        mean_model = self.model(self.ps.freq)

        loglike = (
            2.0
            * self.m
            * (
                np.sum(np.log(mean_model))
                + np.sum(self.ps.power / mean_model)
                + np.sum((2.0 / (2.0 * self.m) - 1.0) * np.log(self.ps.power))
            )
        )

        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        loglike_test = lpost.loglikelihood(t0, neg=True)

        assert np.isclose(loglike, loglike_test)

    def test_posterior(self):
        t0 = [2.0]
        self.model.amplitude = t0[0]

        mean_model = self.model(self.ps.freq)
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=False)

        loglike = (
            -2.0
            * self.m
            * (
                np.sum(np.log(mean_model))
                + np.sum(self.ps.power / mean_model)
                + np.sum((2.0 / (2.0 * self.m) - 1.0) * np.log(self.ps.power))
            )
        )

        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = loglike + logprior

        assert np.isclose(post_test, post, atol=1.0e-10)

    def test_negative_posterior(self):
        t0 = [2.0]
        self.model.amplitude = t0[0]

        mean_model = self.model(self.ps.freq)
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        lpost.logprior = set_logprior(lpost, self.priors)

        post_test = lpost(t0, neg=True)

        loglike = (
            -2.0
            * self.m
            * (
                np.sum(np.log(mean_model))
                + np.sum(self.ps.power / mean_model)
                + np.sum((2.0 / (2.0 * self.m) - 1.0) * np.log(self.ps.power))
            )
        )

        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = -loglike - logprior

        assert np.isclose(post_test, post, atol=1.0e-10)

    def test_counts_are_nan(self):
        y = np.nan * np.ones_like(self.ps.freq)

        ps_nan = copy.copy(self.ps)
        ps_nan.power = np.nan * np.ones_like(self.ps.freq)

        t0 = [2.0]
        self.model.amplitude = t0[0]
        m = self.model(self.ps.freq[1:])
        lpost = PSDPosterior(ps_nan.freq, ps_nan.power, self.model)
        lpost.logprior = set_logprior(lpost, self.priors)

        assert np.isclose(lpost(t0), logmin, 1e-5)

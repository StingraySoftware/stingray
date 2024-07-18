import numpy as np
import scipy.stats
import os
import warnings
import logging

import pytest
from astropy.modeling import models

from stingray import Powerspectrum, AveragedPowerspectrum
from stingray.modeling import ParameterEstimation, PSDParEst, OptimizationResults, SamplingResults
from stingray.modeling import PSDPosterior, set_logprior, PSDLogLikelihood, LogLikelihood
from stingray.modeling.posterior import fitter_to_model_params
from stingray.loggingconfig import CustomFormatter, setup_logger

try:
    from statsmodels.tools.numdiff import approx_hess

    comp_hessian = True
except ImportError:
    comp_hessian = False

try:
    import emcee

    can_sample = True
except ImportError:
    can_sample = False

import matplotlib.pyplot as plt

logger = setup_logger()

pytestmark = pytest.mark.slow


class LogLikelihoodDummy(LogLikelihood):
    def __init__(self, x, y, model):
        LogLikelihood.__init__(self, x, y, model)

    def evaluate(self, parse, neg=False):
        return np.nan


class OptimizationResultsSubclassDummy(OptimizationResults):
    def __init__(self, lpost, res, neg, log=None):
        if log is None:
            self.log = logging.getLogger("Fitting summary")
            self.log.setLevel(logging.DEBUG)
            if not self.log.handlers:
                ch = logging.StreamHandler()
                formatter = CustomFormatter()
                ch.setFormatter(formatter)
                ch.setLevel(logging.DEBUG)
                self.log.addHandler(ch)

        self.neg = neg
        if res is not None:
            self.result = res.fun
            self.p_opt = res.x
        else:
            self.result = None
            self.p_opt = None
        self.model = lpost.model


class TestParameterEstimation(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(100)
        m = 1
        nfreq = 100
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
        cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power, cls.model, m=cls.ps.m)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

    def test_par_est_initializes(self):
        pe = ParameterEstimation()

    def test_parest_stores_max_post_correctly(self):
        """
        Make sure the keyword for Maximum A Posteriori fits is stored correctly
        as a default.
        """
        pe = ParameterEstimation()

        assert pe.max_post is True, "max_post should be set to True as a default."

    def test_object_works_with_loglikelihood_object(self):
        llike = PSDLogLikelihood(self.ps.freq, self.ps.power, self.model, m=self.ps.m)
        pe = ParameterEstimation()
        res = pe.fit(llike, [2.0])
        assert isinstance(res, OptimizationResults), "res must be of " "type OptimizationResults"

    def test_fit_fails_when_object_is_not_posterior_or_likelihood(self):
        x = np.ones(10)
        y = np.ones(10)
        pe = ParameterEstimation()
        with pytest.raises(TypeError):
            res = pe.fit(x, y)

    def test_fit_fails_without_lpost_or_t0(self):
        pe = ParameterEstimation()
        with pytest.raises(TypeError):
            res = pe.fit()

    def test_fit_fails_without_t0(self):
        pe = ParameterEstimation()
        with pytest.raises(TypeError):
            res = pe.fit(np.ones(10))

    def test_fit_fails_with_incorrect_number_of_parameters(self):
        pe = ParameterEstimation()
        t0 = [1, 2]
        with pytest.raises(ValueError):
            res = pe.fit(self.lpost, t0)

    def test_fit_method_works_with_correct_parameter(self):
        pe = ParameterEstimation()
        t0 = [2.0]
        res = pe.fit(self.lpost, t0)

    def test_fit_method_fails_with_too_many_tries(self):
        lpost = LogLikelihoodDummy(self.ps.freq, self.ps.power, self.model)
        pe = ParameterEstimation()
        t0 = [2.0]

        with pytest.raises(Exception):
            res = pe.fit(lpost, t0, neg=True)

    def test_compute_lrt_fails_when_garbage_goes_in(self):
        pe = ParameterEstimation()
        t0 = [2.0]

        with pytest.raises(TypeError):
            pe.compute_lrt(self.lpost, t0, None, t0)

        with pytest.raises(ValueError):
            pe.compute_lrt(self.lpost, t0[:-1], self.lpost, t0)

    def test_compute_lrt_sets_max_post_to_false(self):
        t0 = [2.0]
        pe = ParameterEstimation(max_post=True)

        assert pe.max_post is True
        delta_deviance, opt1, opt2 = pe.compute_lrt(self.lpost, t0, self.lpost, t0)

        assert pe.max_post is False
        assert delta_deviance < 1e-7

    @pytest.mark.skipif("not can_sample")
    def test_sampler_runs(self):
        pe = ParameterEstimation()
        if os.path.exists("test_corner.pdf"):
            os.unlink("test_corner.pdf")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sample_res = pe.sample(
                self.lpost, [2.0], nwalkers=50, niter=10, burnin=50, print_results=True, plot=True
            )

        assert os.path.exists("test_corner.pdf")
        assert sample_res.acceptance > 0.25
        assert isinstance(sample_res, SamplingResults)

    # TODO: Fix pooling with the current setup of logprior
    #    @pytest.mark.skipif("not can_sample")
    #    def test_sampler_pooling(self):
    #        pe = ParameterEstimation()
    #        if os.path.exists("test_corner.pdf"):
    #            os.unlink("test_corner.pdf")
    #        with pytest.warns(RuntimeWarning):
    #            sample_res = pe.sample(self.lpost, [2.0], nwalkers=50, niter=10,
    #                                   burnin=50, print_results=True, plot=True,
    #                                   pool=True)

    @pytest.mark.skipif("can_sample")
    def test_sample_raises_error_without_emcee(self):
        pe = ParameterEstimation()

        with pytest.raises(ImportError):
            sample_res = pe.sample(self.lpost, [2.0])

    def test_simulate_lrt_fails_in_superclass(self):
        pe = ParameterEstimation()
        with pytest.raises(NotImplementedError):
            pe.simulate_lrts(None, None, None, None, None)


class TestOptimizationResults(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(1000)
        m = 1
        nfreq = 100
        freq = np.arange(nfreq)
        noise = np.random.exponential(size=nfreq)
        power = noise * 2.0

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.n = freq.shape[0]
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(
            amplitude
        )

        cls.priors = {"amplitude": p_amplitude}
        cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power, cls.model, m=cls.ps.m)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

        cls.fitmethod = "powell"
        cls.max_post = True
        cls.t0 = np.array([2.0])
        cls.neg = True

        cls.opt = scipy.optimize.minimize(
            cls.lpost, cls.t0, method=cls.fitmethod, args=cls.neg, tol=1.0e-10
        )

        cls.opt.x = np.atleast_1d(cls.opt.x)
        cls.optres = OptimizationResultsSubclassDummy(cls.lpost, cls.opt, neg=True)

    def test_object_initializes_correctly(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)
        assert hasattr(res, "p_opt")
        assert hasattr(res, "result")
        assert hasattr(res, "deviance")
        assert hasattr(res, "aic")
        assert hasattr(res, "bic")
        assert hasattr(res, "model")
        assert isinstance(res.model, models.Const1D)
        assert res.p_opt == self.opt.x, "res.p_opt must be the same as opt.x!"
        assert np.isclose(res.p_opt[0], 2.0, atol=0.1, rtol=0.1)
        assert res.model == self.lpost.model
        assert res.result == self.opt.fun

        mean_model = np.ones_like(self.lpost.x) * self.opt.x[0]
        assert np.allclose(res.mfit, mean_model), (
            "res.model should be exactly " "the model for the data."
        )

    def test_compute_criteria_works_correctly(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

        test_aic = res.result + 2.0 * res.p_opt.shape[0]
        test_bic = res.result + res.p_opt.shape[0] * np.log(self.lpost.x.shape[0])
        test_deviance = -2 * self.lpost.loglikelihood(res.p_opt, neg=False)

        assert np.isclose(res.aic, test_aic, atol=0.1, rtol=0.1)
        assert np.isclose(res.bic, test_bic, atol=0.1, rtol=0.1)
        assert np.isclose(res.deviance, test_deviance, atol=0.1, rtol=0.1)

    def test_merit_calculated_correctly(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

        test_merit = np.sum(((self.ps.power - 2.0) / 2.0) ** 2.0)
        assert np.isclose(res.merit, test_merit, rtol=0.2)

    def test_compute_statistics_computes_mfit(self):
        assert hasattr(self.optres, "mfit") is False
        self.optres._compute_statistics(self.lpost)

        assert hasattr(self.optres, "mfit")

    def test_compute_model(self):
        self.optres._compute_model(self.lpost)

        assert hasattr(self.optres, "mfit"), (
            "OptimizationResult object should have mfit " "attribute at this point!"
        )

        fitter_to_model_params(self.model, self.opt.x)
        mfit_test = self.model(self.lpost.x)

        assert np.allclose(self.optres.mfit, mfit_test)

    def test_compute_statistics_computes_all_statistics(self):
        self.optres._compute_statistics(self.lpost)

        assert hasattr(self.optres, "merit")
        assert hasattr(self.optres, "dof")
        assert hasattr(self.optres, "sexp")
        assert hasattr(self.optres, "ssd")
        assert hasattr(self.optres, "sobs")

        test_merit = np.sum(((self.ps.power - 2.0) / 2.0) ** 2.0)
        test_dof = self.ps.n - self.lpost.npar
        test_sexp = 2.0 * self.lpost.x.shape[0] * len(self.optres.p_opt)
        test_ssd = np.sqrt(2.0 * test_sexp)
        test_sobs = np.sum(self.ps.power - self.optres.p_opt[0])

        assert np.isclose(test_merit, self.optres.merit, rtol=0.2)
        assert test_dof == self.optres.dof
        assert test_sexp == self.optres.sexp
        assert test_ssd == self.optres.ssd
        assert np.isclose(test_sobs, self.optres.sobs, atol=0.01, rtol=0.01)

    def test_compute_criteria_returns_correct_attributes(self):
        self.optres._compute_criteria(self.lpost)

        assert hasattr(self.optres, "aic")
        assert hasattr(self.optres, "bic")
        assert hasattr(self.optres, "deviance")

        npar = self.optres.p_opt.shape[0]

        test_aic = self.optres.result + 2.0 * npar
        test_bic = self.optres.result + npar * np.log(self.ps.freq.shape[0])
        test_deviance = -2 * self.lpost.loglikelihood(self.optres.p_opt, neg=False)

        assert np.isclose(test_aic, self.optres.aic)
        assert np.isclose(test_bic, self.optres.bic)
        assert np.isclose(test_deviance, self.optres.deviance)

    def test_compute_covariance_with_hess_inverse(self):
        self.optres._compute_covariance(self.lpost, self.opt)

        assert np.allclose(self.optres.cov, np.asanyarray(self.opt.hess_inv))
        assert np.allclose(self.optres.err, np.sqrt(np.diag(self.opt.hess_inv)))

    @pytest.mark.skipif("comp_hessian")
    def test_compute_covariance_without_comp_hessian(self):
        self.optres._compute_covariance(self.lpost, None)
        assert self.optres.cov is None
        assert self.optres.err is None

    @pytest.mark.skipif("not comp_hessian")
    def test_compute_covariance_with_hess_inverse(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt, neg=True)

        optres._compute_covariance(self.lpost, self.opt)

        if comp_hessian:
            phess = approx_hess(self.opt.x, self.lpost)
            hess_inv = np.linalg.inv(phess)

            assert np.allclose(optres.cov, hess_inv)
            assert np.allclose(optres.err, np.sqrt(np.diag(np.abs(hess_inv))))

    def test_print_summary_works(self, logger, caplog):
        self.optres._compute_covariance(self.lpost, None)
        self.optres.print_summary(self.lpost)

        assert "Parameter amplitude" in caplog.text
        assert "Fitting statistics" in caplog.text
        assert "number of data points" in caplog.text
        assert "Deviance [-2 log L] D =" in caplog.text
        assert "The Akaike Information Criterion of " "the model is" in caplog.text
        assert "The Bayesian Information Criterion of " "the model is" in caplog.text
        assert "The figure-of-merit function for this model" in caplog.text
        assert "Summed Residuals S =" in caplog.text
        assert "Expected S" in caplog.text
        assert "merit function" in caplog.text


if can_sample:

    class SamplingResultsDummy(SamplingResults):
        def __init__(self, sampler, ci_min=0.05, ci_max=0.95, log=None):
            if log is None:
                self.log = logging.getLogger("Fitting summary")
                self.log.setLevel(logging.DEBUG)
                if not self.log.handlers:
                    ch = logging.StreamHandler()
                    ch.setLevel(logging.DEBUG)
                    self.log.addHandler(ch)

            # store all the samples
            self.samples = sampler.get_chain(flat=True)

            chain_ndims = sampler.get_chain().shape
            self.nwalkers = float(chain_ndims[0])
            self.niter = float(chain_ndims[1])

            # store number of dimensions
            self.ndim = chain_ndims[2]

            # compute and store acceptance fraction
            self.acceptance = np.nanmean(sampler.acceptance_fraction)
            self.L = self.acceptance * self.samples.shape[0]

    class TestSamplingResults(object):
        @classmethod
        def setup_class(cls):
            m = 1
            nfreq = 100
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
            cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power, cls.model, m=cls.ps.m)
            cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

            cls.fitmethod = "BFGS"
            cls.max_post = True
            cls.t0 = [2.0]
            cls.neg = True

            pe = ParameterEstimation()
            res = pe.fit(cls.lpost, cls.t0)

            cls.nwalkers = 50
            cls.niter = 100

            np.random.seed(200)
            p0 = np.array(
                [np.random.multivariate_normal(res.p_opt, res.cov) for i in range(cls.nwalkers)]
            )

            cls.sampler = emcee.EnsembleSampler(
                cls.nwalkers, len(res.p_opt), cls.lpost, args=[False]
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                _, _, _ = cls.sampler.run_mcmc(p0, cls.niter)

        def test_can_sample_is_true(self):
            assert can_sample

        def test_sample_results_object_initializes(self):
            s = SamplingResults(self.sampler)

            assert s.samples.shape[0] == self.nwalkers * self.niter
            assert s.acceptance > 0.25
            assert np.isclose(s.L, s.acceptance * self.nwalkers * self.niter)

        def test_check_convergence_works(self):
            s = SamplingResultsDummy(self.sampler)
            s._check_convergence(self.sampler)
            assert hasattr(s, "rhat")

            rhat_test = 0.038688
            assert np.isclose(rhat_test, s.rhat[0], atol=0.02, rtol=0.1)

            s._infer()
            assert hasattr(s, "mean")
            assert hasattr(s, "std")
            assert hasattr(s, "ci")

            test_mean = 2.0
            test_std = 0.2

            assert np.isclose(test_mean, s.mean[0], rtol=0.1)
            assert np.isclose(test_std, s.std[0], atol=0.01, rtol=0.01)
            assert s.ci.size == 2

        def test_infer_computes_correct_values(self):
            s = SamplingResults(self.sampler)


@pytest.fixture()
def logger():
    logger = logging.getLogger("Some.Logger")
    logger.setLevel(logging.INFO)

    return logger


class TestPSDParEst(object):
    @classmethod
    def setup_class(cls):
        m = 1
        nfreq = 100
        freq = np.linspace(0, 10.0, nfreq + 1)[1:]

        rng = np.random.RandomState(100)  # set the seed for the random number generator
        noise = rng.exponential(size=nfreq)

        cls.model = models.Lorentz1D() + models.Const1D()

        cls.x_0_0 = 2.0
        cls.fwhm_0 = 0.05
        cls.amplitude_0 = 1000.0

        cls.amplitude_1 = 2.0
        cls.model.x_0_0 = cls.x_0_0
        cls.model.fwhm_0 = cls.fwhm_0
        cls.model.amplitude_0 = cls.amplitude_0
        cls.model.amplitude_1 = cls.amplitude_1

        p = cls.model(freq)

        np.random.seed(400)
        power = noise * p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0
        cls.a2_mean, cls.a2_var = 100.0, 10.0

        p_amplitude_1 = lambda amplitude: scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(
            amplitude
        )

        p_x_0_0 = lambda alpha: scipy.stats.uniform(0.0, 5.0).pdf(alpha)

        p_fwhm_0 = lambda alpha: scipy.stats.uniform(0.0, 0.5).pdf(alpha)

        p_amplitude_0 = lambda amplitude: scipy.stats.norm(loc=cls.a2_mean, scale=cls.a2_var).pdf(
            amplitude
        )

        cls.priors = {
            "amplitude_1": p_amplitude_1,
            "amplitude_0": p_amplitude_0,
            "x_0_0": p_x_0_0,
            "fwhm_0": p_fwhm_0,
        }

        cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power, cls.model, m=cls.ps.m)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

        cls.fitmethod = "powell"
        cls.max_post = True
        cls.t0 = [cls.x_0_0, cls.fwhm_0, cls.amplitude_0, cls.amplitude_1]
        cls.neg = True

    @pytest.mark.parametrize("rebin", [0, 0.01])
    def test_fitting_with_ties_and_bounds(self, capsys, rebin):
        double_f = lambda model: model.x_0_0 * 2
        model = self.model.copy()
        model += models.Lorentz1D(
            amplitude=model.amplitude_0, x_0=model.x_0_0 * 2, fwhm=model.fwhm_0
        )
        model.x_0_0 = self.model.x_0_0
        model.amplitude_0 = self.model.amplitude_0
        model.amplitude_1 = self.model.amplitude_1
        model.fwhm_0 = self.model.fwhm_0
        model.x_0_2.tied = double_f
        model.fwhm_0.bounds = [0, 10]
        model.amplitude_0.fixed = True

        p = model(self.ps.freq)

        noise = np.random.exponential(size=len(p))
        power = noise * p

        ps = Powerspectrum()
        ps.freq = self.ps.freq
        ps.power = power
        ps.m = self.ps.m
        ps.df = self.ps.df
        ps.norm = "leahy"

        if rebin != 0:
            ps = ps.rebin_log(rebin)

        pe = PSDParEst(ps, fitmethod="TNC")
        llike = PSDLogLikelihood(ps.freq, ps.power, model)

        true_pars = [
            self.x_0_0,
            self.fwhm_0,
            self.amplitude_1,
            model.amplitude_2.value,
            model.fwhm_2.value,
        ]

        res = pe.fit(llike, true_pars, neg=True)

        compare_pars = [
            self.x_0_0,
            self.fwhm_0,
            self.amplitude_1,
            model.amplitude_2.value,
            model.fwhm_2.value,
        ]

        assert np.allclose(compare_pars, res.p_opt, rtol=0.5)

    def test_par_est_initializes(self):
        pe = PSDParEst(self.ps)
        assert pe.max_post is True, "max_post should be set to True as a default."

    def test_fit_fails_when_object_is_not_posterior_or_likelihood(self):
        x = np.ones(10)
        y = np.ones(10)
        pe = PSDParEst(self.ps)
        with pytest.raises(TypeError):
            res = pe.fit(x, y)

    def test_fit_fails_without_lpost_or_t0(self):
        pe = PSDParEst(self.ps)
        with pytest.raises(TypeError):
            res = pe.fit()

    def test_fit_fails_without_t0(self):
        pe = PSDParEst(self.ps)
        with pytest.raises(TypeError):
            res = pe.fit(np.ones(10))

    def test_fit_fails_with_incorrect_number_of_parameters(self):
        pe = PSDParEst(self.ps)
        t0 = [1, 2]
        with pytest.raises(ValueError):
            res = pe.fit(self.lpost, t0)

    def test_fit_method_works_with_correct_parameter(self):
        pe = PSDParEst(self.ps)
        lpost = PSDPosterior(self.ps.freq, self.ps.power, self.model, self.priors, m=self.ps.m)
        t0 = [2.0, 1, 1, 1]
        res = pe.fit(lpost, t0)
        assert isinstance(res, OptimizationResults), "res must be of type " "OptimizationResults"

        pe.plotfits(res, save_plot=True)

        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

        pe.plotfits(res, save_plot=True, log=True)
        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

        pe.plotfits(res, res2=res, save_plot=True)
        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

        pe.plotfits(res, res2=res, log=True, save_plot=True)
        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

    def test_compute_lrt_fails_when_garbage_goes_in(self):
        pe = PSDParEst(self.ps)
        t0 = [2.0, 1, 1, 1]

        with pytest.raises(TypeError):
            pe.compute_lrt(self.lpost, t0, None, t0)

        with pytest.raises(ValueError):
            pe.compute_lrt(self.lpost, t0[:-1], self.lpost, t0)

    def test_compute_lrt_works(self):
        t0 = [2.0, 1, 1, 1]
        pe = PSDParEst(self.ps, max_post=True)

        assert pe.max_post is True
        delta_deviance, _, _ = pe.compute_lrt(self.lpost, t0, self.lpost, t0)

        assert pe.max_post is False
        assert np.absolute(delta_deviance) < 1.5e-4

    def test_simulate_lrts_works(self):
        m = 1
        nfreq = 100
        freq = np.linspace(1, 10, nfreq)
        rng = np.random.RandomState(100)
        noise = rng.exponential(size=nfreq)
        model = models.Const1D()
        model.amplitude = 2.0
        p = model(freq)
        power = noise * p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        loglike = PSDLogLikelihood(ps.freq, ps.power, model, m=1)

        s_all = np.atleast_2d(np.ones(5) * 2.0).T

        model2 = models.PowerLaw1D() + models.Const1D()
        model2.x_0_0.fixed = True
        loglike2 = PSDLogLikelihood(ps.freq, ps.power, model2, 1)

        pe = PSDParEst(ps)

        lrt_obs, res1, res2 = pe.compute_lrt(loglike, [2.0], loglike2, [2.0, 1.0, 2.0], neg=True)
        lrt_sim = pe.simulate_lrts(s_all, loglike, [2.0], loglike2, [2.0, 1.0, 2.0], seed=100)

        assert (lrt_obs > 0.4) and (lrt_obs < 0.6)
        assert np.all(lrt_sim < 10.0) and np.all(lrt_sim > 0.01)

    def test_compute_lrt_fails_with_wrong_input(self):
        pe = PSDParEst(self.ps)
        with pytest.raises(AssertionError):
            lrt_sim = pe.simulate_lrts(
                np.arange(5), self.lpost, [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]
            )

    def test_generate_model_data(self):
        pe = PSDParEst(self.ps)

        m = self.model
        fitter_to_model_params(m, self.t0)

        model = m(self.ps.freq)

        pe_model = pe._generate_model(
            self.lpost, [self.x_0_0, self.fwhm_0, self.amplitude_0, self.amplitude_1]
        )

        assert np.allclose(model, pe_model)

    def generate_data_rng_object_works(self):
        pe = PSDParEst(self.ps)

        sim_data1 = pe._generate_data(
            self.lpost, [self.x_0_0, self.fwhm_0, self.amplitude_0, self.amplitude_1], seed=1
        )
        sim_data2 = pe._generate_data(
            self.lpost, [self.x_0_0, self.fwhm_0, self.amplitude_0, self.amplitude_1], seed=1
        )

        assert np.allclose(sim_data1.power, sim_data2.power)

    def test_generate_data_produces_correct_distribution(self):
        model = models.Const1D()

        model.amplitude = 2.0

        p = model(self.ps.freq)

        seed = 100
        rng = np.random.RandomState(seed)

        noise = rng.exponential(size=len(p))
        power = noise * p

        ps = Powerspectrum()
        ps.freq = self.ps.freq
        ps.power = power
        ps.m = 1
        ps.df = self.ps.freq[1] - self.ps.freq[0]
        ps.norm = "leahy"

        lpost = PSDLogLikelihood(ps.freq, ps.power, model, m=1)

        pe = PSDParEst(ps)

        rng2 = np.random.RandomState(seed)
        sim_data = pe._generate_data(lpost, [2.0], rng2)

        assert np.allclose(ps.power, sim_data.power)

    def test_generate_model_breaks_with_wrong_input(self):
        pe = PSDParEst(self.ps)

        with pytest.raises(AssertionError):
            pe_model = pe._generate_model([1, 2, 3, 4], [1, 2, 3, 4])

    def test_generate_model_breaks_for_wrong_number_of_parameters(self):
        pe = PSDParEst(self.ps)

        with pytest.raises(AssertionError):
            pe_model = pe._generate_model(self.lpost, [1, 2, 3])

    def test_pvalue_calculated_correctly(self):
        a = [1, 1, 1, 2]
        obs_val = 1.5

        pe = PSDParEst(self.ps)
        pval = pe._compute_pvalue(obs_val, a)

        assert np.isclose(pval, 1.0 / len(a))

    def test_calibrate_lrt_fails_without_lpost_objects(self):
        pe = PSDParEst(self.ps)

        with pytest.raises(TypeError):
            pval = pe.calibrate_lrt(self.lpost, [1, 2, 3, 4], np.arange(10), np.arange(4))

    def test_calibrate_lrt_fails_with_wrong_parameters(self):
        pe = PSDParEst(self.ps)

        with pytest.raises(ValueError):
            pval = pe.calibrate_lrt(self.lpost, [1, 2, 3, 4], self.lpost, [1, 2, 3])

    def test_calibrate_lrt_works_as_expected(self):
        m = 1
        df = 0.01
        freq = np.arange(df, 5 + df, df)
        nfreq = freq.size
        rng = np.random.RandomState(100)
        noise = rng.exponential(size=nfreq)
        model = models.Const1D()
        model.amplitude = 2.0
        p = model(freq)
        power = noise * p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = df
        ps.norm = "leahy"

        loglike = PSDLogLikelihood(ps.freq, ps.power, model, m=1)

        s_all = np.atleast_2d(np.ones(10) * 2.0).T

        model2 = models.PowerLaw1D() + models.Const1D()
        model2.x_0_0.fixed = True
        loglike2 = PSDLogLikelihood(ps.freq, ps.power, model2, m=1)

        pe = PSDParEst(ps)

        pval = pe.calibrate_lrt(
            loglike,
            [2.0],
            loglike2,
            [2.0, 1.0, 2.0],
            sample=s_all,
            max_post=False,
            nsim=5,
            seed=100,
        )

        assert pval > 0.001

    @pytest.mark.skipif("not can_sample")
    def test_calibrate_lrt_works_with_sampling(self):
        m = 1
        nfreq = 100
        freq = np.linspace(1, 10, nfreq)
        rng = np.random.RandomState(100)
        noise = rng.exponential(size=nfreq)
        model = models.Const1D()
        model.amplitude = 2.0
        p = model(freq)
        power = noise * p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        lpost = PSDPosterior(ps.freq, ps.power, model, m=1)

        p_amplitude_1 = lambda amplitude: scipy.stats.norm(loc=2.0, scale=1.0).pdf(amplitude)

        p_alpha_0 = lambda alpha: scipy.stats.uniform(0.0, 5.0).pdf(alpha)

        p_amplitude_0 = lambda amplitude: scipy.stats.norm(loc=self.a2_mean, scale=self.a2_var).pdf(
            amplitude
        )

        priors = {"amplitude": p_amplitude_1}

        priors2 = {"amplitude_1": p_amplitude_1, "amplitude_0": p_amplitude_0, "alpha_0": p_alpha_0}

        lpost.logprior = set_logprior(lpost, priors)

        model2 = models.PowerLaw1D() + models.Const1D()
        model2.x_0_0.fixed = True
        lpost2 = PSDPosterior(ps.freq, ps.power, model2, 1)
        lpost2.logprior = set_logprior(lpost2, priors2)

        pe = PSDParEst(ps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pval = pe.calibrate_lrt(
                lpost,
                [2.0],
                lpost2,
                [2.0, 1.0, 2.0],
                sample=None,
                max_post=True,
                nsim=10,
                nwalkers=10,
                burnin=10,
                niter=10,
                seed=100,
            )

        assert pval > 0.001

    def test_find_highest_outlier_works_as_expected(self):
        mp_ind = 5
        max_power = 1000.0

        ps = Powerspectrum()
        ps.freq = np.arange(10)
        ps.power = np.ones_like(ps.freq)
        ps.power[mp_ind] = max_power
        ps.m = 1
        ps.df = ps.freq[1] - ps.freq[0]
        ps.norm = "leahy"

        pe = PSDParEst(ps)

        max_x, max_ind = pe._find_outlier(ps.freq, ps.power, max_power)

        assert np.isclose(max_x, ps.freq[mp_ind])
        assert max_ind == mp_ind

    def test_compute_highest_outlier_works(self):
        mp_ind = 5
        max_power = 1000.0

        ps = Powerspectrum()
        ps.freq = np.arange(10)
        ps.power = np.ones_like(ps.freq)
        ps.power[mp_ind] = max_power
        ps.m = 1
        ps.df = ps.freq[1] - ps.freq[0]
        ps.norm = "leahy"

        model = models.Const1D()
        p_amplitude = lambda amplitude: scipy.stats.norm(loc=1.0, scale=1.0).pdf(amplitude)

        priors = {"amplitude": p_amplitude}

        lpost = PSDPosterior(ps.freq, ps.power, model, 1)
        lpost.logprior = set_logprior(lpost, priors)

        pe = PSDParEst(ps)

        res = pe.fit(lpost, [1.0])

        res.mfit = np.ones_like(ps.freq)

        max_y, max_x, max_ind = pe._compute_highest_outlier(lpost, res)

        assert np.isclose(max_y[0], 2 * max_power)
        assert np.isclose(max_x[0], ps.freq[mp_ind])
        assert max_ind == mp_ind

    def test_simulate_highest_outlier_works(self):
        m = 1
        nfreq = 100
        seed = 100
        freq = np.linspace(1, 10, nfreq)
        rng = np.random.RandomState(seed)
        noise = rng.exponential(size=nfreq)
        model = models.Const1D()
        model.amplitude = 2.0
        p = model(freq)
        power = noise * p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        nsim = 5

        loglike = PSDLogLikelihood(ps.freq, ps.power, model, m=1)

        s_all = np.atleast_2d(np.ones(nsim) * 2.0).T

        pe = PSDParEst(ps)

        maxpow_sim = pe.simulate_highest_outlier(s_all, loglike, [2.0], max_post=False, seed=seed)

        assert maxpow_sim.shape[0] == nsim
        assert np.all(maxpow_sim > 9.00) and np.all(maxpow_sim < 31.0)

    def test_calibrate_highest_outlier_works(self):
        m = 1
        nfreq = 100
        seed = 100
        freq = np.linspace(1, 10, nfreq)
        rng = np.random.RandomState(seed)
        noise = rng.exponential(size=nfreq)
        model = models.Const1D()
        model.amplitude = 2.0
        p = model(freq)
        power = noise * p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        nsim = 5

        loglike = PSDLogLikelihood(ps.freq, ps.power, model, m=1)

        s_all = np.atleast_2d(np.ones(nsim) * 2.0).T

        pe = PSDParEst(ps)

        pval = pe.calibrate_highest_outlier(loglike, [2.0], sample=s_all, max_post=False, seed=seed)

        assert pval > 0.001

    @pytest.mark.skipif("not can_sample")
    def test_calibrate_highest_outlier_works_with_sampling(self):
        m = 1
        nfreq = 100
        seed = 100
        freq = np.linspace(1, 10, nfreq)
        rng = np.random.RandomState(seed)
        noise = rng.exponential(size=nfreq)
        model = models.Const1D()
        model.amplitude = 2.0
        p = model(freq)
        power = noise * p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        nsim = 5

        lpost = PSDPosterior(ps.freq, ps.power, model, m=1)
        p_amplitude = lambda amplitude: scipy.stats.norm(loc=1.0, scale=1.0).pdf(amplitude)

        priors = {"amplitude": p_amplitude}
        lpost.logprior = set_logprior(lpost, priors)

        pe = PSDParEst(ps)

        pval = pe.calibrate_highest_outlier(
            lpost,
            [2.0],
            sample=None,
            max_post=True,
            seed=seed,
            nsim=nsim,
            niter=10,
            nwalkers=20,
            burnin=10,
        )

        assert pval > 0.001

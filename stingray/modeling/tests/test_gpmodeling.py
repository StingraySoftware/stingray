import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from stingray.modeling.gpmodeling import (
    get_kernel,
    get_mean,
    get_gp_params,
    _psd_model,
    get_psd_and_approx,
    run_prior_checks,
    _get_coefficients_approximation,
    get_prior,
    get_log_likelihood,
    GPResult,
    get_priors_samples,
)
from stingray import Lightcurve

try:
    import jax
    import jax.numpy as jnp
    from jax import random

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pytest.skip(allow_module_level=True)

_HAS_TINYGP = True
_HAS_TFP = True
_HAS_JAXNS = True

try:
    import tinygp
    from tinygp import GaussianProcess, kernels
except ImportError:
    _HAS_TINYGP = False


try:
    import tensorflow_probability.substrates.jax as tfp

    tfpd = tfp.distributions
except ImportError:
    _HAS_TFP = False

try:
    import jaxns
    from jaxns import ExactNestedSampler, TerminationCondition, Prior, Model
except ImportError:
    _HAS_JAXNS = False


def clear_all_figs():
    fign = plt.get_fignums()
    for fig in fign:
        plt.close(fig)


# @pytest.mark.xfail
@pytest.mark.skipif(not _HAS_TINYGP, reason="tinygp not installed")
class Testget_kernel(object):
    def setup_class(self):
        self.x = np.linspace(0, 1, 5)
        self.kernel_params = {"arn": 1.0, "aqpo": 1.0, "crn": 1.0, "cqpo": 1.0, "freq": 1.0}

    def test_get_kernel_qpo_plus_rn(self):
        kernel_qpo_plus_rn = kernels.quasisep.Exp(
            scale=1 / 1, sigma=(1) ** 0.5
        ) + kernels.quasisep.Celerite(
            a=1,
            b=0.0,
            c=1,
            d=2 * jnp.pi * 1,
        )
        kernel_qpo_plus_rn_test = get_kernel("QPO_plus_RN", self.kernel_params)
        assert (
            kernel_qpo_plus_rn(self.x, jnp.array([0.0]))
            == kernel_qpo_plus_rn_test(self.x, jnp.array([0.0]))
        ).all()

    def test_get_kernel_rn(self):
        kernel_rn = kernels.quasisep.Exp(scale=1 / 1, sigma=(1) ** 0.5)
        kernel_rn_test = get_kernel("RN", self.kernel_params)
        assert (
            kernel_rn(self.x, jnp.array([0.0])) == kernel_rn_test(self.x, jnp.array([0.0]))
        ).all()

    def test_get_kernel_qpo(self):
        kernel_qpo = kernels.quasisep.Celerite(
            a=1,
            b=0.0,
            c=1,
            d=2 * jnp.pi * 1,
        )
        kernel_qpo_test = get_kernel("QPO", self.kernel_params)
        assert (
            kernel_qpo(self.x, jnp.array([0.0])) == kernel_qpo_test(self.x, jnp.array([0.0]))
        ).all()

    def test_get_kernel_powL(self):

        t = jnp.arange(0, 100.05, 0.05)
        f_min, f_max = 1 / (t[-1] - t[0]) / 20, 1 / (2 * (t[1] - t[0])) * 20
        n_approx_components = 20
        alpha_1, f_1, alpha_2, variance = 0.3, 0.05, 2.9, 0.2

        kernel_params = {
            "alpha_1": alpha_1,
            "f_bend": f_1,
            "alpha_2": alpha_2,
            "variance": variance,
        }

        spectral_points = jnp.geomspace(f_min, f_max, 20)
        spectral_matrix = 1 / (
            1 + jnp.power(jnp.atleast_2d(spectral_points).T / spectral_points, 4)
        )
        psd_values = _psd_model("PowL", kernel_params)(spectral_points)
        psd_values /= psd_values[0]
        spectral_coefficients = jnp.linalg.solve(spectral_matrix, psd_values)
        amplitudes = (
            spectral_coefficients
            * spectral_points
            * kernel_params["variance"]
            / jnp.sum(spectral_coefficients * spectral_points)
        )
        kernel = amplitudes[0] * kernels.quasisep.SHO(
            quality=1 / jnp.sqrt(2), omega=2 * jnp.pi * spectral_points[0]
        )
        for j in range(1, n_approx_components):
            kernel += amplitudes[j] * kernels.quasisep.SHO(
                quality=1 / jnp.sqrt(2), omega=2 * jnp.pi * spectral_points[j]
            )

        kernel_PowL_test = get_kernel(
            "PowL",
            kernel_params,
            f_min=1 / (t[-1] - t[0]),
            f_max=0.5 / (t[1] - t[0]),
            n_approx_components=n_approx_components,
        )
        assert (
            kernel(self.x, jnp.array([0.0])) == kernel_PowL_test(self.x, jnp.array([0.0]))
        ).all()

    def test_value_error(self):
        with pytest.raises(ValueError, match="Kernel type not implemented"):
            get_kernel("periodic", self.kernel_params)


class Testget_mean(object):
    def setup_class(self):
        self.t = np.linspace(0, 5, 10)
        self.mean_params = {
            "A": jnp.array([3.0, 4.0]),
            "t0": jnp.array([0.2, 0.7]),
            "sig": jnp.array([0.2, 0.1]),
        }
        self.skew_mean_params = {
            "A": jnp.array([3.0, 4.0]),
            "t0": jnp.array([0.2, 0.7]),
            "sig1": jnp.array([0.2, 0.1]),
            "sig2": jnp.array([0.3, 0.4]),
        }
        self.fred_mean_params = {
            "A": jnp.array([3.0, 4.0]),
            "t0": jnp.array([0.2, 0.7]),
            "phi": jnp.array([4.0, 5.0]),
            "delta": jnp.array([0.3, 0.4]),
        }

    def test_get_mean_gaussian(self):
        result_gaussian = 3 * jnp.exp(-((self.t - 0.2) ** 2) / (2 * (0.2**2))) + 4 * jnp.exp(
            -((self.t - 0.7) ** 2) / (2 * (0.1**2))
        )
        assert (get_mean("gaussian", self.mean_params)(self.t) == result_gaussian).all()

    def test_get_mean_exponential(self):
        result_exponential = 3 * jnp.exp(-jnp.abs(self.t - 0.2) / (2 * (0.2**2))) + 4 * jnp.exp(
            -jnp.abs(self.t - 0.7) / (2 * (0.1**2))
        )
        assert (get_mean("exponential", self.mean_params)(self.t) == result_exponential).all()

    def test_get_mean_constant(self):
        result_constant = 3 * jnp.ones_like(self.t)
        const_param_dict = {"A": jnp.array([3.0])}
        assert (get_mean("constant", const_param_dict)(self.t) == result_constant).all()

    def test_get_mean_skew_gaussian(self):
        result_skew_gaussian = 3.0 * jnp.where(
            self.t > 0.2,
            jnp.exp(-((self.t - 0.2) ** 2) / (2 * (0.3**2))),
            jnp.exp(-((self.t - 0.2) ** 2) / (2 * (0.2**2))),
        ) + 4.0 * jnp.where(
            self.t > 0.7,
            jnp.exp(-((self.t - 0.7) ** 2) / (2 * (0.4**2))),
            jnp.exp(-((self.t - 0.7) ** 2) / (2 * (0.1**2))),
        )
        assert (
            get_mean("skew_gaussian", self.skew_mean_params)(self.t) == result_skew_gaussian
        ).all()

    def test_get_mean_skew_exponential(self):
        result_skew_exponential = 3.0 * jnp.where(
            self.t > 0.2,
            jnp.exp(-jnp.abs(self.t - 0.2) / (2 * (0.3**2))),
            jnp.exp(-jnp.abs(self.t - 0.2) / (2 * (0.2**2))),
        ) + 4.0 * jnp.where(
            self.t > 0.7,
            jnp.exp(-jnp.abs(self.t - 0.7) / (2 * (0.4**2))),
            jnp.exp(-jnp.abs(self.t - 0.7) / (2 * (0.1**2))),
        )
        assert (
            get_mean("skew_exponential", self.skew_mean_params)(self.t) == result_skew_exponential
        ).all()

    def test_get_mean_fred(self):
        result_fred = 3.0 * jnp.exp(-4.0 * ((self.t + 0.3) / 0.2 + 0.2 / (self.t + 0.3))) * jnp.exp(
            2 * 4.0
        ) + 4.0 * jnp.exp(-5.0 * ((self.t + 0.4) / 0.7 + 0.7 / (self.t + 0.4))) * jnp.exp(2 * 5.0)
        assert (get_mean("fred", self.fred_mean_params)(self.t) == result_fred).all()

    def test_value_error(self):
        with pytest.raises(ValueError, match="Mean type not implemented"):
            get_mean("polynomial", self.mean_params)


class Testget_gp_params(object):
    def setup_class(self):
        pass

    def test_get_gp_params_rn(self):
        assert get_gp_params("RN", "gaussian") == ["log_arn", "log_crn", "log_A", "t0", "log_sig"]
        assert get_gp_params("RN", "constant") == ["log_arn", "log_crn", "log_A"]
        assert get_gp_params("RN", "skew_gaussian") == [
            "log_arn",
            "log_crn",
            "log_A",
            "t0",
            "log_sig1",
            "log_sig2",
        ]
        assert get_gp_params("RN", "skew_exponential") == [
            "log_arn",
            "log_crn",
            "log_A",
            "t0",
            "log_sig1",
            "log_sig2",
        ]
        assert get_gp_params("RN", "exponential") == [
            "log_arn",
            "log_crn",
            "log_A",
            "t0",
            "log_sig",
        ]
        assert get_gp_params("RN", "fred") == [
            "log_arn",
            "log_crn",
            "log_A",
            "t0",
            "delta",
            "phi",
        ]

    def test_get_gp_params_qpo_plus_rn(self):
        assert get_gp_params("QPO_plus_RN", "gaussian") == [
            "log_arn",
            "log_crn",
            "log_aqpo",
            "log_cqpo",
            "log_freq",
            "log_A",
            "t0",
            "log_sig",
        ]
        with pytest.raises(ValueError, match="Mean type not implemented"):
            get_gp_params("QPO_plus_RN", "notimplemented")

        with pytest.raises(ValueError, match="Kernel type not implemented"):
            get_gp_params("notimplemented", "gaussian")

    def test_get_qpo(self):
        assert get_gp_params("QPO", "gaussian") == [
            "log_aqpo",
            "log_cqpo",
            "log_freq",
            "log_A",
            "t0",
            "log_sig",
        ]

    def test_get_params_PowL(self):
        assert get_gp_params("PowL", "constant") == [
            "alpha_1",
            "log_f_bend",
            "alpha_2",
            "variance",
            "log_A",
        ]
        assert get_gp_params("PowL", "constant", scale_errors=True) == [
            "alpha_1",
            "log_f_bend",
            "alpha_2",
            "variance",
            "log_A",
            "scale_err",
        ]

    def test_get_params_DoubPowL(self):
        assert get_gp_params("DoubPowL", "constant") == [
            "alpha_1",
            "log_f_bend_1",
            "alpha_2",
            "log_f_bend_2",
            "alpha_3",
            "variance",
            "log_A",
        ]
        assert get_gp_params("DoubPowL", "constant", scale_errors=True) == [
            "alpha_1",
            "log_f_bend_1",
            "alpha_2",
            "log_f_bend_2",
            "alpha_3",
            "variance",
            "log_A",
            "scale_err",
        ]


@pytest.mark.skipif(
    not (_HAS_TINYGP and _HAS_TFP and _HAS_JAXNS), reason="tinygp, tfp or jaxns not installed"
)
class TestPSDapprox(object):
    def setup_class(self):
        self.t = np.linspace(0, 1, 10)
        self.y = np.array([2.4, 3.5, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2])
        self.yerr = np.array([0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2])

        self.kernel_type = "PowL"
        self.mean_type = "constant"
        self.kernel_params = get_gp_params(
            self.kernel_type, self.mean_type, scale_errors=True, log_transform=True
        )
        self.loglike = get_log_likelihood(
            self.kernel_params, self.kernel_type, self.mean_type, self.t, self.y, self.yerr
        )

        min_f_1, max_f_1 = 5.5e-3, 0.5
        muL = -0.2

        self.priors = [
            tfpd.Uniform(low=0.0, high=1.25),
            tfpd.Uniform(low=jnp.log(min_f_1), high=jnp.log(max_f_1)),
            tfpd.Uniform(low=1.5, high=4),
            tfpd.LogNormal(muL, 1.0),
            tfpd.Normal(0.0, 2),
            tfpd.Gamma(jnp.array(2.0), jnp.array(2.0)),  # was not working without jnp.array, why?
            tfpd.Uniform(low=jnp.log(1e-6), high=jnp.log(0.99 * 3)),
        ]

    def test_get_prior_samples(self):

        loglike = get_log_likelihood(
            self.kernel_params, self.kernel_type, self.mean_type, self.t, self.y, self.yerr
        )

        # prior_dict = dict(zip(self.kernel_params, priors))
        # prior_model = get_prior(self.kernel_params, prior_dict)
        prior_samples = get_priors_samples(
            jax.random.PRNGKey(0), self.kernel_params, self.priors, loglike, 10
        )
        assert len(prior_samples) == len(self.kernel_params)
        for key in self.kernel_params:
            assert prior_samples[key].shape == (10,)

    def test_get_psd_and_approx(self):
        f0, fM = 5.5e-3 / 20, 0.5 * 20
        loglike = get_log_likelihood(
            self.kernel_params, self.kernel_type, self.mean_type, self.t, self.y, self.yerr
        )

        prior_samples = get_priors_samples(
            jax.random.PRNGKey(0), self.kernel_params, self.priors, loglike, 10
        )

        f, psd_models, psd_approx = get_psd_and_approx(
            self.kernel_type, self.kernel_params, prior_samples, f0, fM, n_frequencies=200
        )
        assert len(f) == 200
        assert psd_models.shape == (10, 200)
        assert psd_approx.shape == (10, 200)

    def test_run_prior_checks(self):
        loglike = get_log_likelihood(
            self.kernel_params, self.kernel_type, self.mean_type, self.t, self.y, self.yerr
        )

        fig1, fig2 = run_prior_checks(
            self.kernel_type, self.kernel_params, self.priors, loglike, 5.5e-2, 0.5
        )
        plt.fignum_exists(1)
        plt.fignum_exists(2)

    def test__get_coefficients_approximation(self):

        kernel_params = {"alpha_1": 0.3, "f_bend": 0.05, "alpha_2": 3.5, "variance": 0.2}
        f_min, f_max = 1 / (self.t[-1] - self.t[0]) / 20, 1 / (2 * np.min(np.diff(self.t))) * 20
        spectral_points, spectral_coefs = _get_coefficients_approximation(
            self.kernel_type, kernel_params, f_min, f_max, 25
        )
        assert spectral_points.shape == (25,)
        assert spectral_coefs.shape == (25,)


@pytest.mark.xfail
@pytest.mark.skipif(
    not (_HAS_TINYGP and _HAS_TFP and _HAS_JAXNS), reason="tinygp, tfp or jaxns not installed"
)
class TestGPResult(object):
    def setup_class(self):
        self.Times = np.linspace(0, 1, 64)
        kernel_params = {
            "arn": jnp.exp(1.5),
            "crn": jnp.exp(1.0),
        }
        mean_params = {"A": jnp.array([3.0]), "t0": jnp.array([0.2]), "sig": jnp.array([0.2])}
        kernel = get_kernel("RN", kernel_params)
        mean = get_mean("gaussian", mean_params)

        gp = GaussianProcess(kernel=kernel, X=self.Times, mean_value=mean(self.Times))
        self.counts = gp.sample(key=jax.random.PRNGKey(6))

        lc = Lightcurve(time=self.Times, counts=self.counts, dt=self.Times[1] - self.Times[0])

        self.params_list = get_gp_params(kernel_type="RN", mean_type="gaussian")

        T = self.Times[-1] - self.Times[0]
        f = 1 / (self.Times[1] - self.Times[0])
        span = jnp.max(self.counts) - jnp.min(self.counts)

        # The prior dictionary, with suitable tfpd prior distributions
        prior_dict = {
            "log_A": Prior(
                tfpd.Uniform(low=jnp.log(0.1 * span), high=jnp.log(2 * span)), name="log_A"
            ),
            "t0": tfpd.Uniform(low=self.Times[0] - 0.1 * T, high=self.Times[-1] + 0.1 * T),
            "log_sig": tfpd.Uniform(low=jnp.log(0.5 * 1 / f), high=jnp.log(2 * T)),
            "log_arn": tfpd.Uniform(low=jnp.log(0.1 * span), high=jnp.log(2 * span)),
            "log_crn": tfpd.Uniform(low=jnp.log(1 / T), high=jnp.log(f)),
        }

        prior_model = get_prior(self.params_list, prior_dict)
        likelihood_model = get_log_likelihood(
            self.params_list,
            kernel_type="RN",
            mean_type="gaussian",
            times=self.Times,
            counts=self.counts,
        )

        NSmodel = Model(prior_model=prior_model, log_likelihood=likelihood_model)
        NSmodel.sanity_check(random.PRNGKey(10), S=100)

        Exact_ns = ExactNestedSampler(NSmodel, num_live_points=500, max_samples=5e3)
        Termination_reason, State = Exact_ns(
            random.PRNGKey(42), term_cond=TerminationCondition(live_evidence_frac=1e-4)
        )
        self.Results = Exact_ns.to_results(State, Termination_reason)

        self.gpresult = GPResult(lc)
        self.gpresult.sample(
            prior_model=prior_model, likelihood_model=likelihood_model, max_samples=5e3
        )

    def test_sample(self):
        for key in self.params_list:
            assert (self.Results.samples[key]).all() == (self.gpresult.results.samples[key]).all()

    def test_get_evidence(self):
        assert self.Results.log_Z_mean == self.gpresult.get_evidence()

    def test_plot_diagnostics(self):
        self.gpresult.plot_diagnostics()
        assert plt.fignum_exists(1)

    def test_plot_cornerplot(self):
        self.gpresult.plot_cornerplot()
        assert plt.fignum_exists(1)

    def test_get_parameters_names(self):
        assert sorted(self.params_list) == self.gpresult.get_parameters_names()

    def test_print_summary(self):
        self.gpresult.print_summary()
        assert True

    def test_max_posterior_parameters(self):
        for key in self.params_list:
            assert key in self.gpresult.get_max_posterior_parameters()

    def test_max_likelihood_parameters(self):
        for key in self.params_list:
            assert key in self.gpresult.get_max_likelihood_parameters()

    def test_posterior_plot(self):
        self.gpresult.posterior_plot("log_A")
        assert plt.fignum_exists(1)

    def test_posterior_plot_labels_and_fname_default(self):
        clear_all_figs()
        outfname = "log_A_Posterior_plot.png"
        if os.path.exists(outfname):
            os.unlink(outfname)
        self.gpresult.posterior_plot("log_A", save=True)
        assert os.path.exists(outfname)
        os.unlink(outfname)

    def test_posterior_plot_labels_and_fname(self):
        clear_all_figs()
        outfname = "blabla.png"
        if os.path.exists(outfname):
            os.unlink(outfname)
        self.gpresult.posterior_plot("log_A", axis=[0, 14, 0, 0.5], save=True, filename=outfname)
        assert os.path.exists(outfname)
        os.unlink(outfname)

    def test_weighted_posterior_plot(self):
        self.gpresult.weighted_posterior_plot("log_A")
        assert plt.fignum_exists(1)

    def test_weighted_posterior_plot_labels_and_fname_default(self):
        clear_all_figs()
        outfname = "log_A_Weighted_Posterior_plot.png"
        if os.path.exists(outfname):
            os.unlink(outfname)
        self.gpresult.weighted_posterior_plot("log_A", save=True)
        assert os.path.exists(outfname)
        os.unlink(outfname)

    def test_weighted_posterior_plot_labels_and_fname(self):
        clear_all_figs()
        outfname = "blabla.png"
        if os.path.exists(outfname):
            os.unlink(outfname)
        self.gpresult.weighted_posterior_plot(
            "log_A", axis=[0, 14, 0, 0.5], save=True, filename=outfname
        )
        assert os.path.exists(outfname)
        os.unlink(outfname)

    def test_comparison_plot(self):
        self.gpresult.comparison_plot("log_A", "t0")
        assert plt.fignum_exists(1)

    def test_comparison_plot_labels_and_fname_default(self):
        clear_all_figs()
        outfname = "log_A_t0_Comparison_plot.png"
        if os.path.exists(outfname):
            os.unlink(outfname)
        self.gpresult.comparison_plot("log_A", "t0", save=True)
        assert os.path.exists(outfname)
        os.unlink(outfname)

    def test_comparison_plot_labels_and_fname(self):
        clear_all_figs()
        outfname = "blabla.png"
        if os.path.exists(outfname):
            os.unlink(outfname)
        self.gpresult.comparison_plot(
            "log_A", "t0", axis=[0, 0.5, 0, 5], save=True, filename=outfname
        )
        assert os.path.exists(outfname)
        os.unlink(outfname)

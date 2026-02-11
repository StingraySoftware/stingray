import pytest
import numpy as np
from collections.abc import Iterable
from scipy import stats
from stingray.stats import (
    p_single_trial_from_p_multitrial,
    p_multitrial_from_single_trial,
    fold_detection_level,
    phase_dispersion_detection_level,
    z2_n_detection_level,
    fold_profile_probability,
    phase_dispersion_probability,
    z2_n_probability,
    classical_pvalue,
    equivalent_gaussian_Nsigma_from_logp,
    chi2_logp,
    fold_profile_logprobability,
    phase_dispersion_logprobability,
    z2_n_logprobability,
    power_upper_limit,
    amplitude_upper_limit,
    power_confidence_limits,
)


@pytest.mark.parametrize("ntrial", [1, 10, 100, 1000, 10000, 100000])
def test_p_single_from_multi(ntrial):
    epsilon_1 = 0.00000001
    epsilon_n = p_multitrial_from_single_trial(epsilon_1, ntrial)
    epsilon_1_corr = p_single_trial_from_p_multitrial(epsilon_n, ntrial)

    assert np.isclose(epsilon_1_corr, epsilon_1, rtol=1e-2)


def test_p_single_from_multi_fails():
    epsilon_n = 1
    with pytest.warns(UserWarning) as record:
        p1 = p_single_trial_from_p_multitrial(epsilon_n, 1000)

    assert np.any(
        ["Multi-trial probability is very close to 1" in r.message.args[0] for r in record]
    )
    assert np.isnan(p1)


def test_fold_detection_level():
    """Test pulse phase calculation, frequency only."""
    np.testing.assert_almost_equal(fold_detection_level(16, 0.01), 30.577914166892498)
    epsilon_corr = p_single_trial_from_p_multitrial(0.01, 2)
    np.testing.assert_almost_equal(
        fold_detection_level(16, 0.01, ntrial=2), fold_detection_level(16, epsilon_corr)
    )


def test_pdm_detection_level():
    """Test pulse phase calculation, frequency only."""
    nsamples = 10000
    nbin = 32
    beta_ppf = 0.9947853493529972
    np.testing.assert_almost_equal(phase_dispersion_detection_level(nsamples, nbin), beta_ppf)
    epsilon_corr = p_single_trial_from_p_multitrial(0.01, 2)
    np.testing.assert_almost_equal(
        phase_dispersion_detection_level(nsamples, nbin, 0.01, ntrial=2),
        phase_dispersion_detection_level(nsamples, nbin, epsilon_corr),
    )


def test_zn_detection_level():
    np.testing.assert_almost_equal(z2_n_detection_level(2), 13.276704135987625)
    epsilon_corr = p_single_trial_from_p_multitrial(0.01, 2)
    np.testing.assert_almost_equal(
        z2_n_detection_level(4, 0.01, ntrial=2), z2_n_detection_level(4, epsilon_corr)
    )


@pytest.mark.parametrize("ntrial", [1, 10, 100, 1000, 100000])
def test_fold_probability(ntrial):
    detlev = fold_detection_level(16, 0.1, ntrial=ntrial)
    np.testing.assert_almost_equal(fold_profile_probability(detlev, 16, ntrial=ntrial), 0.1)


@pytest.mark.parametrize("ntrial", [1, 10, 100, 1000, 100000])
def test_pdm_probability(ntrial):
    nsamples = 10000
    nbin = 32
    detec_level = 0.01
    detlev = phase_dispersion_detection_level(nsamples, nbin, epsilon=detec_level, ntrial=ntrial)
    np.testing.assert_almost_equal(
        phase_dispersion_probability(detlev, nsamples, nbin, ntrial=ntrial), detec_level
    )


@pytest.mark.parametrize("ntrial", [1, 10, 100, 1000, 100000])
def test_zn_probability(ntrial):
    detlev = z2_n_detection_level(2, 0.1, ntrial=ntrial)
    np.testing.assert_almost_equal(z2_n_probability(detlev, 2, ntrial=ntrial), 0.1)


def ncx2_ppf(xs, n=1, c=0.99, summed_flag=True):
    if isinstance(xs, Iterable):
        return [ncx2_ppf(x, n, c, summed_flag) for x in xs]
    if not summed_flag:
        xs = xs * n
    rv = stats.ncx2(2 * n, xs)
    val = rv.ppf(1 - c)
    if not summed_flag:
        val = val / n
    return val


class TestClassicalSignificances(object):
    def test_function_runs(self):
        power = 2.0
        nspec = 1.0
        with pytest.warns(DeprecationWarning):
            classical_pvalue(power, nspec)

    def test_power_is_not_infinite(self):
        power = np.inf
        nspec = 1
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                classical_pvalue(power, nspec)

    def test_power_is_not_infinite2(self):
        power = -np.inf
        nspec = 1
        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                classical_pvalue(power, nspec)

    def test_power_is_non_nan(self):
        power = np.nan
        nspec = 1
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                classical_pvalue(power, nspec)

    def test_power_is_positive(self):
        power = -2.0
        nspec = 1.0
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                classical_pvalue(power, nspec)

    def test_nspec_is_not_infinite(self):
        power = 2.0
        nspec = np.inf
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                classical_pvalue(power, nspec)

    def test_nspec_is_not_infinite2(self):
        power = 2.0
        nspec = -np.inf
        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                classical_pvalue(power, nspec)

    def test_nspec_is_not_nan(self):
        power = 2.0
        nspec = np.nan
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                classical_pvalue(power, nspec)

    def test_nspec_is_positive(self):
        power = 2.0
        nspec = -1.0
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                classical_pvalue(power, nspec)

    def test_nspec_is_nonzero(self):
        power = 2.0
        nspec = 0.0
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                classical_pvalue(power, nspec)

    def test_nspec_is_an_integer_number(self):
        power = 2.0
        nspec = 2.5
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                classical_pvalue(power, nspec)

    def test_nspec_float_type_okay(self):
        power = 2.0
        nspec = 2.0
        with pytest.warns(DeprecationWarning):
            classical_pvalue(power, nspec)

    def test_pvalue_decreases_with_increasing_power(self):
        power1 = 2.0
        power2 = 20.0
        nspec = 1.0
        with pytest.warns(DeprecationWarning):
            pval1 = classical_pvalue(power1, nspec)
        with pytest.warns(DeprecationWarning):
            pval2 = classical_pvalue(power2, nspec)

        assert pval1 - pval2 > 0.0

    def test_pvalue_must_decrease_with_increasing_nspec(self):
        power = 3.0
        nspec1 = 1.0
        nspec2 = 10.0

        with pytest.warns(DeprecationWarning):
            pval1 = classical_pvalue(power, nspec1)
        with pytest.warns(DeprecationWarning):
            pval2 = classical_pvalue(power, nspec2)

        assert pval1 - pval2 > 0.0

    def test_very_large_powers_produce_zero_prob(self):
        power = 31000.0
        nspec = 1
        with pytest.warns((DeprecationWarning, UserWarning)):
            pval = classical_pvalue(power, nspec)
        assert np.isclose(pval, 0.0)

    def test_equivalent_Nsigma_logp(self):
        pvalues = [
            0.15865525393145707,
            0.0013498980316301035,
            9.865877004244794e-10,
            6.661338147750939e-16,
            3.09e-138,
        ]
        log_pvalues = np.log(np.array(pvalues))
        sigmas = np.array([1, 3, 6, 8, 25])
        # Single number
        assert np.isclose(
            equivalent_gaussian_Nsigma_from_logp(log_pvalues[0]), sigmas[0], atol=0.01
        )
        # Array
        assert np.allclose(equivalent_gaussian_Nsigma_from_logp(log_pvalues), sigmas, atol=0.01)

    def test_chi2_logp(self):
        chi2 = 31
        # Test check on dof
        with pytest.raises(ValueError) as excinfo:
            chi2_logp(chi2, 1)
        message = str(excinfo.value)
        assert "The number of degrees of freedom cannot be < 2" in message

        # Test that approximate function works as expected. chi2 / dof > 15,
        # but small and safe number in order to compare to scipy.stats
        assert np.isclose(chi2_logp(chi2, 2), stats.chi2.logsf(chi2, 2), atol=0.1)
        chi2 = np.array([5, 32])
        assert np.allclose(chi2_logp(chi2, 2), stats.chi2.logsf(chi2, 2), atol=0.1)

    @pytest.mark.parametrize("nbin", [8, 16, 23, 72])
    def test_compare_fold_logprob_with_prob(self, nbin):
        stat = np.random.uniform(5, 200, 5)
        logp = fold_profile_logprobability(stat, nbin)
        p = fold_profile_probability(stat, nbin)
        assert np.allclose(logp, np.log(p))

    @pytest.mark.parametrize("nbin", [8, 16, 23, 72])
    def test_compare_pdm_logprob_with_prob(self, nbin):
        nsamples = 10000
        stat = np.random.uniform(5, 200, 5)
        logp = phase_dispersion_logprobability(stat, nsamples, nbin)
        p = phase_dispersion_probability(stat, nsamples, nbin)
        assert np.allclose(logp, np.log(p))

    @pytest.mark.parametrize("n", [2, 16, 23, 72])
    def test_compare_z2n_logprob_with_prob(self, n):
        stat = np.random.uniform(5, 200, 5)
        logp = z2_n_logprobability(stat, n=n)
        p = z2_n_probability(stat, n=n)
        assert np.allclose(logp, np.log(p))

    def test_power_upper_limit(self):
        # Use example from Vaughan+94
        assert np.isclose(power_upper_limit(40, 1, 0.99), 75, rtol=0.1)

    def test_amplitude_upper_limit(self):
        # The measured power is much lower than the expected noise level
        # (100 vs ~200 for n=100). The upper limit should be 0.
        assert np.isclose(
            amplitude_upper_limit(
                100, 100000, n=100, c=0.95, fft_corr=False, nyq_ratio=0.01, summed_flag=True
            ),
            0.0,
            atol=1e-4,
        )

    def test_amplitude_upper_limit_averaging(self):
        assert np.isclose(
            amplitude_upper_limit(
                100, 100000, n=100, c=0.95, fft_corr=False, nyq_ratio=0.01, summed_flag=False
            ),
            0.045,
            rtol=0.1,
        )

    def test_power_upper_limit_averaging(self):
        assert np.isclose(power_upper_limit(100, 10, 0.997, summed_flag=False), 115, rtol=0.1)

    def test_power_confidence_limits(self):
        from scipy.stats import ncx2

        preal = 25.0
        n = 2
        c = 0.95
        low, high = power_confidence_limits(preal, n, c)

        # Check boundaries against Scipy CDF definition
        assert np.isclose(ncx2.cdf(high, df=2 * n, nc=preal), 0.975, atol=1e-4)
        assert np.isclose(ncx2.cdf(low, df=2 * n, nc=preal), 0.025, atol=1e-4)

    @pytest.mark.parametrize("n", [1, 10, 100, 1000])
    @pytest.mark.parametrize("c", [0.50, 0.95, 0.99])
    @pytest.mark.parametrize("summed_flag", [True, False])
    def test_power_upper_limit_roundtrip(self, n, c, summed_flag):
        # our pmeas should be the lower limit of a certain preal. Let us go backwards,
        # and simulate preal.
        preal = 10 ** np.random.uniform(-1, 3, 100)
        # and create the pmeas from there
        pmeas = ncx2_ppf(preal, n=n, c=c, summed_flag=summed_flag)
        # Now calculate the upper limit, which should coincide with preal.
        uplim = [power_upper_limit(p, c=c, n=n, summed_flag=summed_flag) for p in pmeas]
        assert np.allclose(uplim, preal, rtol=0.1)

    @pytest.mark.parametrize("n", [1, 100])
    @pytest.mark.parametrize("c", [0.50, 0.99])
    @pytest.mark.parametrize("summed_flag", [True, False])
    def test_power_upper_limit_low_val(self, n, c, summed_flag):
        # consistent with power==0 if the measured power is below the given confidence limit
        # for no power
        maxval = ncx2_ppf(0, n=n, c=c, summed_flag=summed_flag)

        vals = np.random.uniform(0, maxval, 10)
        for val in vals:
            assert np.isclose(power_upper_limit(val, c=c, n=n, summed_flag=summed_flag), 0.0)

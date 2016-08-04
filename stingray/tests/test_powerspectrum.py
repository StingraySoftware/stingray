
import numpy as np

from astropy.tests.helper import pytest

from stingray import Lightcurve
from stingray import Powerspectrum, AveragedPowerspectrum, \
                        DynamicalPowerspectrum
from stingray.powerspectrum import classical_pvalue

np.random.seed(20150907)


class TestPowerspectrum(object):

    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 100.0
        mean_counts = mean_count_rate * dt

        poisson_counts = np.random.poisson(mean_counts,
                                           size=time.shape[0])

        cls.lc = Lightcurve(time, counts=poisson_counts)

    def test_make_empty_periodogram(self):
        ps = Powerspectrum()
        assert ps.norm == "frac"
        assert ps.freq is None
        assert ps.power is None
        assert ps.df is None
        assert ps.m == 1
        assert ps.n is None

    def test_make_periodogram_from_lightcurve(self):
        ps = Powerspectrum(lc=self.lc)
        assert ps.freq is not None
        assert ps.power is not None
        assert ps.df == 1.0 / self.lc.tseg
        assert ps.norm == "frac"
        assert ps.m == 1
        assert ps.n == self.lc.time.shape[0]
        assert ps.nphots == np.sum(self.lc.counts)

    def test_periodogram_types(self):
        ps = Powerspectrum(lc=self.lc)
        assert isinstance(ps.freq, np.ndarray)
        assert isinstance(ps.power, np.ndarray)

    def test_init_with_lightcurve(self):
        assert Powerspectrum(self.lc)

    def test_init_without_lightcurve(self):
        with pytest.raises(TypeError):
            assert Powerspectrum(self.lc.counts)

    def test_init_with_nonsense_data(self):
        nonsense_data = [None for i in range(100)]
        with pytest.raises(TypeError):
            assert Powerspectrum(nonsense_data)

    def test_init_with_nonsense_norm(self):
        nonsense_norm = "bla"
        with pytest.raises(ValueError):
            assert Powerspectrum(self.lc, norm=nonsense_norm)

    def test_init_with_wrong_norm_type(self):
        nonsense_norm = 1.0
        with pytest.raises(TypeError):
            assert Powerspectrum(self.lc, norm=nonsense_norm)

    def test_total_variance(self):
        """
        the integral of powers (or Riemann sum) should be close
        to the variance divided by twice the length of the light curve.

        Note: make sure the factors of ncounts match!
        Also, make sure to *exclude* the zeroth power!
        """
        ps = Powerspectrum(lc=self.lc)
        nn = ps.n
        pp = ps.unnorm_power / np.float(nn)**2
        p_int = np.sum(pp[:-1]*ps.df) + (pp[-1]*ps.df)/2
        var_lc = np.var(self.lc.counts) / (2.*self.lc.tseg)
        assert np.isclose(p_int, var_lc, atol=0.01, rtol=0.01)

    def test_frac_normalization_is_standard(self):
        """
        Make sure the standard normalization of a periodogram is
        rms and it stays that way!
        """
        ps = Powerspectrum(lc=self.lc)
        assert ps.norm == "frac"

    def test_frac_normalization_correct(self):
        """
        In rms normalization, the integral of the powers should be
        equal to the variance of the light curve divided by the mean
        of the light curve squared.
        """
        ps = Powerspectrum(lc=self.lc, norm="frac")
        ps_int = np.sum(ps.power[:-1]*ps.df) + ps.power[-1]*ps.df/2
        std_lc = np.var(self.lc.counts) / np.mean(self.lc.counts)**2
        print(ps_int)
        print(std_lc)
        assert np.isclose(ps_int, std_lc, atol=0.01, rtol=0.01)

    def test_fractional_rms_in_frac_norm(self):
        ps = Powerspectrum(lc=self.lc, norm="frac")
        rms_ps, rms_err = ps.compute_rms(min_freq=ps.freq[1],
                                         max_freq=ps.freq[-1])
        rms_lc = np.std(self.lc.counts) / np.mean(self.lc.counts)
        assert np.isclose(rms_ps, rms_lc, atol=0.01)

    def test_leahy_norm_correct(self):
        time = np.linspace(0, 10.0, 1e6)
        counts = np.random.poisson(1000, size=time.shape[0])

        lc = Lightcurve(time, counts)
        ps = Powerspectrum(lc, norm="leahy")
        print(np.mean(ps.power[1:]))

        assert np.isclose(np.mean(ps.power[1:]), 2.0, atol=0.01, rtol=0.01)

    def test_leahy_norm_total_variance(self):
        """
        In Leahy normalization, the total variance should be the sum of
        powers multiplied by the number of counts and divided by the
        square of the number of data points in the light curve
        """
        ps = Powerspectrum(lc=self.lc, norm="Leahy")
        ps_var = (np.sum(self.lc.counts)/ps.n**2.) * \
            (np.sum(ps.power[:-1]) + ps.power[-1]/2.)

        assert np.isclose(ps_var, np.var(self.lc.counts), atol=0.01)

    def test_fractional_rms_in_leahy_norm(self):
        """
        fractional rms should only be *approximately* equal the standard
        deviation divided by the mean of the light curve. Therefore, we allow
        for a larger tolerance in np.isclose()
        """
        ps = Powerspectrum(lc=self.lc, norm="Leahy")
        rms_ps, rms_err = ps.compute_rms(min_freq=ps.freq[0],
                                         max_freq=ps.freq[-1])

        rms_lc = np.std(self.lc.counts) / np.mean(self.lc.counts)
        assert np.isclose(rms_ps, rms_lc, atol=0.01)

    def test_fractional_rms_fails_when_rms_not_leahy(self):
        with pytest.raises(Exception):
            ps = Powerspectrum(lc=self.lc, norm="rms")
            rms_ps, rms_err = ps.compute_rms(min_freq=ps.freq[0],
                                            max_freq=ps.freq[-1])


    def test_fractional_rms_error(self):
        """
        TODO: Need to write a test for the fractional rms error.
        But I don't know how!
        """
        pass

    def test_rebin_makes_right_attributes(self):
        ps = Powerspectrum(lc=self.lc, norm="Leahy")
        # replace powers
        ps.power = np.ones_like(ps.power) * 2.0
        rebin_factor = 2.0
        bin_ps = ps.rebin(rebin_factor*ps.df)

        assert bin_ps.freq is not None
        assert bin_ps.power is not None
        assert bin_ps.df == rebin_factor * 1.0 / self.lc.tseg
        assert bin_ps.norm.lower() == "leahy"
        assert bin_ps.m == 2
        assert bin_ps.n == self.lc.time.shape[0]
        assert bin_ps.nphots == np.sum(self.lc.counts)

    def test_rebin_uses_mean(self):
        """
        Make sure the rebin-method uses "mean" to average instead of summing
        powers by default, and that this is not changed in the future!
        Note: function defaults come as a tuple, so the first keyword argument
        had better be 'method'
        """
        ps = Powerspectrum(lc=self.lc, norm="Leahy")
        assert ps.rebin.__defaults__[0] == "mean"

    def rebin_several(self, df):
        """
        TODO: Not sure how to write tests for the rebin method!
        """
        ps = Powerspectrum(lc=self.lc, norm="Leahy")
        bin_ps = ps.rebin(df)
        assert np.isclose(bin_ps.freq[0], bin_ps.df, atol=1e-4, rtol=1e-4)

    def test_rebin(self):
        df_all = [2, 3, 5, 1.5, 1, 85]
        for df in df_all:
            yield self.rebin_several, df

    def test_classical_significances_runs(self):
        ps = Powerspectrum(lc=self.lc, norm="Leahy")
        ps.classical_significances()

    def test_classical_significances_fails_in_rms(self):
        ps = Powerspectrum(lc=self.lc, norm="frac")
        with pytest.raises(ValueError):
            ps.classical_significances()

    def test_classical_significances_threshold(self):
        ps = Powerspectrum(lc=self.lc, norm="leahy")

        # change the powers so that just one exceeds the threshold
        ps.power = np.zeros_like(ps.power)+2.0

        index = 1
        ps.power[index] = 10.0

        threshold = 0.01

        pval = ps.classical_significances(threshold=threshold,
                                          trial_correction=False)
        assert pval[0, 0] < threshold
        assert pval[1, 0] == index

    def test_classical_significances_trial_correction(self):
        ps = Powerspectrum(lc=self.lc, norm="leahy")
        # change the powers so that just one exceeds the threshold
        ps.power = np.zeros_like(ps.power) + 2.0
        index = 1
        ps.power[index] = 10.0
        threshold = 0.01
        pval = ps.classical_significances(threshold=threshold,
                                          trial_correction=True)
        assert np.size(pval) == 0

    def test_pvals_is_numpy_array(self):
        ps = Powerspectrum(lc=self.lc, norm="leahy")
        # change the powers so that just one exceeds the threshold
        ps.power = np.zeros_like(ps.power)+2.0

        index = 1
        ps.power[index] = 10.0

        threshold = 1.0

        pval = ps.classical_significances(threshold=threshold,
                                          trial_correction=True)

        assert isinstance(pval, np.ndarray)
        assert pval.shape[0] == 2


class TestAveragedPowerspectrum(object):

    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 10.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate*dt

        poisson_counts = np.random.poisson(mean_counts,
                                           size=time.shape[0])

        cls.lc = Lightcurve(time, counts=poisson_counts)

    def test_one_segment(self):
        segment_size = self.lc.tseg
        ps = AveragedPowerspectrum(self.lc, segment_size)
        assert np.isclose(ps.segment_size, segment_size)

    def test_n_segments(self):
        nseg_all = [1, 2, 3, 5, 10, 20, 100]
        for nseg in nseg_all:
            yield self.check_segment_size, nseg

    def check_segment_size(self, nseg):
        segment_size = self.lc.tseg/nseg
        ps = AveragedPowerspectrum(self.lc, segment_size)
        assert ps.m == nseg

    def test_segments_with_leftover(self):
        segment_size = self.lc.tseg/2. - 1.
        ps = AveragedPowerspectrum(self.lc, segment_size)
        assert np.isclose(ps.segment_size, segment_size)
        assert ps.m == 2

    def test_init_without_segment(self):
        with pytest.raises(TypeError):
            assert AveragedPowerspectrum(self.lc)

    def test_init_with_nonsense_segment(self):
        segment_size = "foo"
        with pytest.raises(TypeError):
            assert AveragedPowerspectrum(self.lc, segment_size)

    def test_init_with_none_segment(self):
        segment_size = None
        with pytest.raises(TypeError):
            assert AveragedPowerspectrum(self.lc, segment_size)

    def test_init_with_inf_segment(self):
        segment_size = np.inf
        with pytest.raises(ValueError):
            assert AveragedPowerspectrum(self.lc, segment_size)

    def test_init_with_nan_segment(self):
        segment_size = np.nan
        with pytest.raises(ValueError):
            assert AveragedPowerspectrum(self.lc, segment_size)

    def test_list_of_light_curves(self):
        n_lcs = 10

        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate*dt

        lc_all = []
        for n in range(n_lcs):
            poisson_counts = np.random.poisson(mean_counts,
                                               size=len(time))

            lc = Lightcurve(time, counts=poisson_counts)
            lc_all.append(lc)

        segment_size = 0.5
        assert AveragedPowerspectrum(lc_all, segment_size)

    def test_list_with_nonsense_component(self):
        n_lcs = 10

        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate*dt

        lc_all = []
        for n in range(n_lcs):
            poisson_counts = np.random.poisson(mean_counts,
                                               size=len(time))

            lc = Lightcurve(time, counts=poisson_counts)
            lc_all.append(lc)

        lc_all.append(1.0)
        segment_size = 0.5

        with pytest.raises(TypeError):
            assert AveragedPowerspectrum(lc_all, segment_size)

    def test_leahy_correct_for_multiple(self):

        n = 100
        lc_all = []
        for i in range(n):
            time = np.arange(0.0, 10.0, 10./100000)
            counts = np.random.poisson(1000, size=time.shape[0])
            lc = Lightcurve(time, counts)
            lc_all.append(lc)

        ps = AveragedPowerspectrum(lc_all, 10.0, norm="leahy")

        assert np.isclose(np.mean(ps.power), 2.0, atol=1e-3, rtol=1e-3)
        assert np.isclose(np.std(ps.power), 2.0/np.sqrt(n), atol=0.1, rtol=0.1)


class TestClassicalSignificances(object):

    def test_function_runs(self):
        power = 2.0
        nspec = 1.0
        classical_pvalue(power, nspec)

    def test_power_is_not_infinite(self):
        power = np.inf
        nspec = 1
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_power_is_not_infinite2(self):
        power = -np.inf
        nspec = 1
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_power_is_non_nan(self):
        power = np.nan
        nspec = 1
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_power_is_positive(self):
        power = -2.0
        nspec = 1.0
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_nspec_is_not_infinite(self):
        power = 2.0
        nspec = np.inf
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_nspec_is_not_infinite2(self):
        power = 2.0
        nspec = -np.inf
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_nspec_is_not_nan(self):
        power = 2.0
        nspec = np.nan
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_nspec_is_positive(self):
        power = 2.0
        nspec = -1.0
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_nspec_is_nonzero(self):
        power = 2.0
        nspec = 0.0
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_nspec_is_an_integer_number(self):
        power = 2.0
        nspec = 2.5
        with pytest.raises(ValueError):
            classical_pvalue(power, nspec)

    def test_nspec_float_type_okay(self):
        power = 2.0
        nspec = 2.0
        classical_pvalue(power, nspec)

    def test_pvalue_decreases_with_increasing_power(self):
        power1 = 2.0
        power2 = 20.0
        nspec = 1.0
        pval1 = classical_pvalue(power1, nspec)
        pval2 = classical_pvalue(power2, nspec)

        assert pval1-pval2 > 0.0

    def test_pvalue_must_decrease_with_increasing_nspec(self):

        power = 3.0
        nspec1 = 1.0
        nspec2 = 10.0

        pval1 = classical_pvalue(power, nspec1)
        pval2 = classical_pvalue(power, nspec2)

        assert pval1-pval2 > 0.0

    def test_very_large_powers_produce_zero_prob(self):
        power = 31000.0
        nspec = 1
        pval = classical_pvalue(power, nspec)
        assert np.isclose(pval, 0.0)


class TestDynamicalPowerspectrum(object):

    def setup_class(self):
        lc = Lightcurve([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        [4, 5, 6, 2, 10, 5, 12, 16, 18, 10, 2, 0, 22, 25, 30])

        self.lc = lc

    def test_with_bad_seg_size(self):
        with pytest.raises(ValueError):
            dps = DynamicalPowerspectrum(self.lc, segment_size=0)

    def test_matrix(self):
        dps = DynamicalPowerspectrum(self.lc, segment_size=10)
        assert dps.matrix.shape == (4, 2)

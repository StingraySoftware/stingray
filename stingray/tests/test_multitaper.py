import numpy as np
import copy
import warnings

import pytest
from numpy.random import poisson, standard_cauchy
from scipy.signal import TransferFunction

from stingray import Lightcurve
from stingray.events import EventList
from stingray import Multitaper, Powerspectrum

pytestmark = pytest.mark.slow
np.random.seed(1)


class TestMultitaper(object):
    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 1.484
        dt = 0.0001

        time = np.arange(tstart + 0.5 * dt, tend + 0.5 * dt, dt)

        mean_count_rate = 100.0
        mean_counts = mean_count_rate * dt

        poisson_counts = np.random.poisson(mean_counts, size=time.shape[0])

        cls.lc = Lightcurve(time, counts=poisson_counts, dt=dt, gti=[[tstart, tend]])

        mean = 0
        standard_deviation = 0.1
        gauss_counts = np.random.normal(mean, standard_deviation, size=time.shape[0])

        cls.lc_gauss = Lightcurve(
            time, counts=gauss_counts, dt=dt, gti=[[tstart, tend]], err_dist="gauss"
        )

    def test_lc_keyword_deprecation(self):
        mtp1 = Multitaper(self.lc)
        with pytest.warns(DeprecationWarning) as record:
            mtp2 = Multitaper(lc=self.lc)
        assert np.any(["lc keyword" in r.message.args[0] for r in record])
        assert np.allclose(mtp1.power, mtp2.power)
        assert np.allclose(mtp1.freq, mtp2.freq)

    def test_make_empty_multitaper(self):
        mtp = Multitaper()
        assert mtp.norm == "frac"
        assert mtp.freq is None
        assert mtp.power is None
        assert mtp.multitaper_norm_power is None
        assert mtp.eigvals is None
        assert mtp.power_err is None
        assert mtp.df is None
        assert mtp.m == 1
        assert mtp.nphots is None
        assert mtp.jk_var_deg_freedom is None

    @pytest.mark.parametrize("lightcurve", ["lc", "lc_gauss"])
    def test_make_multitaper_from_lightcurve(self, lightcurve):
        mtp = Multitaper(getattr(self, lightcurve))
        assert mtp.norm == "frac"
        assert mtp.fullspec is False
        assert mtp.meancounts == getattr(self, lightcurve).meancounts
        assert mtp.nphots == np.float64(np.sum(getattr(self, lightcurve).counts))
        assert mtp.err_dist == getattr(self, lightcurve).err_dist
        assert mtp.dt == getattr(self, lightcurve).dt
        assert mtp.n == getattr(self, lightcurve).time.shape[0]
        assert mtp.df == 1.0 / getattr(self, lightcurve).tseg
        assert mtp.m == 1
        assert mtp.freq is not None
        assert mtp.multitaper_norm_power is not None
        assert mtp.power is not None
        assert mtp.power_err is not None
        assert mtp.jk_var_deg_freedom is not None

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            mpt = Multitaper(norm=1)

    def test_init_with_lightcurve(self):
        assert Multitaper(self.lc)

    def test_init_without_lightcurve(self):
        with pytest.raises(TypeError):
            assert Multitaper(self.lc.counts)

    def test_init_with_nonsense_data(self):
        nonsense_data = [None for i in range(100)]
        with pytest.raises(TypeError):
            assert Multitaper(nonsense_data)

    def test_init_with_nonsense_norm(self):
        nonsense_norm = "bla"
        with pytest.raises(ValueError):
            assert Multitaper(self.lc, norm=nonsense_norm)

    def test_init_with_wrong_norm_type(self):
        nonsense_norm = 1.0
        with pytest.raises(TypeError):
            assert Multitaper(self.lc, norm=nonsense_norm)

    @pytest.mark.parametrize("low_bias", [False, True])
    def test_make_multitaper_adaptive_and_low_bias(self, low_bias):
        mtp = Multitaper(self.lc, low_bias=low_bias, adaptive=True)

        if low_bias:
            assert np.min(mtp.eigvals) >= 0.9
        assert mtp.jk_var_deg_freedom is not None
        assert mtp.freq is not None
        assert mtp.multitaper_norm_power is not None

    @pytest.mark.parametrize("lightcurve", ["lc", "lc_gauss"])
    def test_make_multitaper_var(self, lightcurve):
        if getattr(self, lightcurve).err_dist == "poisson":
            mtp = Multitaper(getattr(self, lightcurve))
            assert mtp.err_dist == "poisson"
            assert mtp.var == getattr(self, lightcurve).meancounts
        else:
            with pytest.warns(UserWarning) as record:
                mtp = Multitaper(getattr(self, lightcurve))
            assert mtp.err_dist == "gauss"
            assert mtp.var == np.mean(getattr(self, lightcurve).counts_err) ** 2
            assert np.any(["not poisson" in r.message.args[0] for r in record])

    @pytest.mark.parametrize("lombscargle", [False, True])
    def test_fourier_multitaper_with_invalid_NW(self, lombscargle):
        with pytest.raises(ValueError):
            mtp = Multitaper(self.lc, NW=0.1, lombscargle=lombscargle)

    @pytest.mark.parametrize(
        "adaptive, jackknife", [(a, j) for a in (True, False) for j in (True, False)]
    )
    def test_fourier_multitaper_with_adaptive_jackknife_combos(self, adaptive, jackknife):
        mtp = Multitaper(self.lc, adaptive=adaptive, jackknife=jackknife)
        assert mtp.multitaper_norm_power is not None
        assert mtp.jk_var_deg_freedom is not None

    def test_fractional_rms_in_frac_norm_is_consistent(self):
        """
        Copied from test_powerspectrum.py
        """
        time = np.arange(0, 100, 1) + 0.5

        poisson_counts = np.random.poisson(100.0, size=time.shape[0])

        lc = Lightcurve(time, counts=poisson_counts, dt=1, gti=[[0, 100]])
        mtp = Multitaper(lc, norm="leahy")
        with pytest.warns(UserWarning, match="M<30"):
            rms_mtp_l, rms_err_l = mtp.compute_rms(
                min_freq=mtp.freq[1], max_freq=mtp.freq[-1], poisson_noise_level=0
            )

        mtp = Multitaper(lc, norm="frac")
        with pytest.warns(UserWarning, match="M<30"):
            rms_mtp, rms_err = mtp.compute_rms(
                min_freq=mtp.freq[1], max_freq=mtp.freq[-1], poisson_noise_level=0
            )
        assert np.allclose(rms_mtp, rms_mtp_l, atol=0.01)
        assert np.allclose(rms_err, rms_err_l, atol=0.01)

    def test_classical_significances_threshold(self):
        """
        Copied from test_powerspectrum.py
        """
        mtp = Multitaper(self.lc, norm="leahy")

        # change the powers so that just one exceeds the threshold
        mtp.power = np.zeros_like(mtp.power) + 2.0

        index = 1
        mtp.power[index] = 10.0

        threshold = 0.01

        pval = mtp.classical_significances(threshold=threshold, trial_correction=False)
        assert pval[0, 0] < threshold
        assert pval[1, 0] == index

    @pytest.mark.parametrize("df", [2, 3, 5, 1.5, 1, 85])
    def test_rebin(self, df):
        """
        TODO: Not sure how to write tests for the rebin method!
        """
        mtp = Multitaper(self.lc, norm="Leahy")
        bin_mtp = mtp.rebin(df)
        assert np.isclose(bin_mtp.freq[1] - bin_mtp.freq[0], bin_mtp.df, atol=1e-4, rtol=1e-4)
        assert np.isclose(
            bin_mtp.freq[0], (mtp.freq[0] - mtp.df * 0.5 + bin_mtp.df * 0.5), atol=1e-4, rtol=1e-4
        )

    def test_rebin_uses_mean(self):
        """
        Make sure the rebin-method uses "mean" to average instead of summing
        powers by default, and that this is not changed in the future!
        Note: function defaults come as a tuple, so the first keyword argument
        had better be 'method'
        """
        mtp = Multitaper(self.lc, norm="Leahy")
        assert mtp.rebin.__defaults__[2] == "mean"

    def test_rebin_output_shapes(self):
        """
        Test whether all the rebinned spectral attributes have the same shape.
        """
        mtp = Multitaper(self.lc, norm="Leahy")
        mtp_rebin = mtp.rebin(df=1.5)
        assert (
            mtp_rebin.power.shape
            == mtp_rebin.freq.shape
            == mtp_rebin.unnorm_power.shape
            == mtp_rebin.multitaper_norm_power.shape
        )

    def test_rebin_error(self):
        mtp = Multitaper(self.lc)
        with pytest.raises(ValueError):
            mtp.rebin()

    def test_rebin_smaller_resolution(self):
        # Original df is between 0.9 and 1.0
        mtp = Multitaper(self.lc)
        with pytest.raises(ValueError):
            new_mtp = mtp.rebin(df=0.1)

    def test_rebin(self):
        mtp = Multitaper(self.lc)
        new_mtp = mtp.rebin(df=1.5)
        assert new_mtp.df == 1.5

    def test_rebin_factor(self):
        mtp = Multitaper(self.lc)
        new_mtp = mtp.rebin(f=1.5)
        assert new_mtp.df == mtp.df * 1.5

    def test_get_adaptive_psd_with_less_tapers(self):
        with pytest.warns(UserWarning) as record:
            mtp = Multitaper(data=self.lc, NW=1.5, adaptive=True)
        assert np.any(["Not adaptively" in r.message.args[0] for r in record])
        assert mtp.multitaper_norm_power is not None

    @pytest.mark.parametrize("lombscargle", [False, True])
    def test_max_eigval_less_than_threshold(self, lombscargle):
        with pytest.warns(UserWarning) as record:
            mtp = Multitaper(data=self.lc, NW=0.5, low_bias=True, lombscargle=lombscargle)
        assert np.any(["not properly use low_bias" in r.message.args[0] for r in record])
        assert len(mtp.eigvals) > 0

    def test_init_data_eventlist(self):
        el = EventList.from_lc(self.lc)
        mtp_el = Multitaper(data=el, dt=self.lc.dt)

        mtp = Multitaper(data=self.lc)

        assert max(mtp_el.multitaper_norm_power - mtp.multitaper_norm_power) == 0

    def test_init_data_eventlist_no_dt(self):
        with pytest.raises(ValueError):
            el = EventList.from_lc(self.lc)
            mtp_el = Multitaper(data=el)

    def test_multitaper_lombscargle(self):
        rng = np.random.default_rng()
        N = 1000

        white_noise_irregular = rng.normal(loc=0.0, scale=7, size=N)
        start = 0.0
        end = 9.0

        # Generating uneven sampling times by adding white noise. Do tell a better way
        time_irregular = np.linspace(start, end, N) + rng.normal(
            loc=0.0, scale=(end - start) / (3 * N), size=N
        )
        time_irregular = np.sort(time_irregular)

        with pytest.warns(UserWarning) as record:
            lc_nonuni = Lightcurve(
                time=time_irregular,
                counts=white_noise_irregular,
                err_dist="gauss",
                err=np.ones_like(time_irregular) + np.sqrt(0.0),
            )  # Zero mean
        assert np.any(["aren't equal" in r.message.args[0] for r in record])

        mtls_white = Multitaper(lc_nonuni, lombscargle=True, low_bias=True, NW=4)
        assert mtls_white.norm == "frac"
        assert mtls_white.fullspec is False
        assert mtls_white.meancounts == lc_nonuni.meancounts
        assert mtls_white.nphots == np.float64(np.sum(lc_nonuni.counts))
        assert mtls_white.err_dist == lc_nonuni.err_dist
        assert mtls_white.dt == lc_nonuni.dt
        assert mtls_white.n == lc_nonuni.time.shape[0]
        assert mtls_white.df == 1.0 / lc_nonuni.tseg
        assert mtls_white.m == 1
        assert mtls_white.freq is not None
        assert mtls_white.multitaper_norm_power is not None
        assert mtls_white.power is not None
        assert mtls_white.power_err is not None
        assert mtls_white.jk_var_deg_freedom is None  # Not supported yet
        assert len(mtls_white.eigvals) > 0

    @pytest.mark.parametrize("norm", ["frac", "leahy", "abs", "none"])
    def test_multitaper_lombscargle_consistency(self, norm):
        mtp = Multitaper(self.lc, adaptive=False, norm=norm)
        mtp_ls = Multitaper(self.lc, lombscargle=True, adaptive=False, norm=norm)

        # Check if 99% of the points in the PSDs are within the set tolerance
        assert (
            np.sum(np.isclose(mtp.power, mtp_ls.power, atol=0.022 * np.max(mtp_ls.power)))
            >= 0.99 * mtp_ls.power.size
        )

        # Check if the freq vals are the same
        ps = Powerspectrum(self.lc, norm=norm)

        assert np.allclose(mtp.freq, mtp_ls.freq)
        assert np.allclose(mtp.freq, ps.freq)
        assert mtp.power.shape == ps.power.shape

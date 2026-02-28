import os

from collections.abc import Iterable
import pytest
import numpy as np
from astropy.table import Table

from stingray.fourier import fft, fftfreq, normalize_abs, normalize_frac, poisson_level
from stingray.fourier import (
    get_flux_iterable_from_segments,
    avg_pds_from_timeseries,
    avg_cs_from_timeseries,
    avg_pds_from_events,
    avg_cs_from_events,
)
from stingray.fourier import normalize_periodograms, estimate_intrinsic_coherence
from stingray.fourier import raw_coherence, intrinsic_coherence
from stingray.fourier import bias_term, error_on_averaged_cross_spectrum, unnormalize_periodograms
from stingray.fourier import impose_symmetry_lsft, lsft_slow, lsft_fast, rms_calculation
from stingray.fourier import get_average_ctrate, normalize_leahy_from_variance
from stingray.fourier import integrate_power_in_frequency_range
from stingray.fourier import get_rms_from_rms_norm_periodogram, get_rms_from_unnorm_periodogram
from stingray.fourier import check_powers_for_intrinsic_coherence

from stingray.utils import check_allclose_and_print
from astropy.modeling.models import Lorentz1D

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


rng = np.random.RandomState(137259723)


def compare_tables(table1, table2, rtol=0.001, discard=[]):
    for key in table1.meta.keys():
        if key in discard:
            continue
        oe, oc = table1.meta[key], table2.meta[key]

        if isinstance(oe, (int, str)):
            assert oe == oc
        elif oe is None:
            assert oc is None
        elif isinstance(oe, Iterable):
            assert np.allclose(oe, oc, rtol=rtol)
        else:
            assert np.isclose(oe, oc, rtol=rtol)
    for col in table1.colnames:
        if col in discard:
            continue
        oe, oc = table1[col], table2[col]
        assert np.allclose(oe, oc, rtol=rtol)


def test_norm():
    mean = var = 100000
    N = 1000000
    dt = 0.2
    meanrate = mean / dt
    lc = rng.poisson(mean, N)
    pds = np.abs(fft(lc)) ** 2
    freq = fftfreq(N, dt)
    good = slice(1, N // 2)

    pdsabs = normalize_abs(pds, dt, lc.size)
    pdsfrac = normalize_frac(pds, dt, lc.size, mean)
    pois_abs = poisson_level(meanrate=meanrate, norm="abs")
    pois_frac = poisson_level(meanrate=meanrate, norm="frac")

    assert np.isclose(pdsabs[good].mean(), pois_abs, rtol=0.01)
    assert np.isclose(pdsfrac[good].mean(), pois_frac, rtol=0.01)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_flux_iterables(dtype):
    times = np.arange(4)
    fluxes = np.ones(4).astype(dtype)
    errors = np.ones(4).astype(dtype) * np.sqrt(2)
    gti = np.asanyarray([[-0.5, 3.5]])
    iter = get_flux_iterable_from_segments(times, gti, 2, n_bin=None, fluxes=fluxes, errors=errors)
    cast_kind = float
    if np.iscomplexobj(fluxes):
        cast_kind = complex
    for it, er in iter:
        assert np.allclose(it, 1, rtol=0.01)
        assert np.allclose(er, np.sqrt(2), rtol=0.01)
        assert isinstance(it[0], cast_kind)
        assert isinstance(er[0], cast_kind)


def test_avg_pds_imperfect_lc_size():
    times = np.arange(100)
    fluxes = np.ones(100).astype(float)
    gti = np.asanyarray([[-0.5, 99.5]])
    segment_size = 5.99
    dt = 1
    res = avg_pds_from_timeseries(times, gti, segment_size, dt, fluxes=fluxes)
    assert res.meta["segment_size"] == 6
    assert res.meta["dt"] == 1


def test_avg_pds_from_events_warns():
    times = np.arange(100)
    fluxes = np.ones(100).astype(float)
    gti = np.asanyarray([[-0.5, 99.5]])
    segment_size = 5.99
    dt = 1
    with pytest.warns(DeprecationWarning, match="avg_pds_from_events is deprecated"):
        res = avg_pds_from_events(times, gti, segment_size, dt, fluxes=fluxes)
    assert res.meta["segment_size"] == 6
    assert res.meta["dt"] == 1


def test_avg_cs_imperfect_lc_size():
    times1 = times2 = np.arange(100)
    fluxes1 = np.ones(100).astype(float)
    fluxes2 = np.ones(100).astype(float)
    gti = np.asanyarray([[-0.5, 99.5]])
    segment_size = 5.99
    dt = 1
    res = avg_cs_from_timeseries(
        times1, times2, gti, segment_size, dt, fluxes1=fluxes1, fluxes2=fluxes2
    )
    assert res.meta["segment_size"] == 6
    assert res.meta["dt"] == 1


def test_avg_cs_from_events_warns():
    times1 = times2 = np.arange(100)
    fluxes1 = np.ones(100).astype(float)
    fluxes2 = np.ones(100).astype(float)
    gti = np.asanyarray([[-0.5, 99.5]])
    segment_size = 5.99
    dt = 1
    with pytest.warns(DeprecationWarning, match="avg_cs_from_events is deprecated"):
        res = avg_cs_from_events(
            times1, times2, gti, segment_size, dt, fluxes1=fluxes1, fluxes2=fluxes2
        )
    assert res.meta["segment_size"] == 6
    assert res.meta["dt"] == 1


class TestCoherence(object):
    @classmethod
    def setup_class(cls):
        data = (
            Table.read(os.path.join(datadir, "sample_variable_series.fits"))["data"][:10000] * 1000
        )
        cls.data1 = rng.poisson(data)
        cls.data2 = rng.poisson(data)
        ft1 = np.fft.fft(cls.data1)
        ft2 = np.fft.fft(cls.data2)
        dt = 0.01
        cls.N = data.size
        mean = np.mean(data)
        meanrate = mean / dt
        freq = np.fft.fftfreq(data.size, dt)
        good = (freq > 0) & (freq < 0.1)
        ft1, ft2 = ft1[good], ft2[good]
        cls.cross = normalize_periodograms(
            ft1.conj() * ft2, dt, cls.N, mean, norm="abs", power_type="all"
        )
        cls.pds1 = normalize_periodograms(
            ft1 * ft1.conj(), dt, cls.N, mean, norm="abs", power_type="real"
        )
        cls.pds2 = normalize_periodograms(
            ft2 * ft2.conj(), dt, cls.N, mean, norm="abs", power_type="real"
        )

        cls.p1noise = poisson_level(meanrate=meanrate, norm="abs")
        cls.p2noise = poisson_level(meanrate=meanrate, norm="abs")

    def test_old_estimate_intrinsic_coherence(self):
        with pytest.warns(DeprecationWarning):
            coh = estimate_intrinsic_coherence(
                self.cross, self.pds1, self.pds2, self.p1noise, self.p2noise, self.N
            )
        assert np.allclose(coh, 1, atol=0.001)

    def test_raw_high_coherence(self):
        coh = raw_coherence(self.cross, self.pds1, self.pds2, self.p1noise, self.p2noise, self.N)
        assert np.allclose(coh, 1, atol=0.001)

    def test_raw_low_coherence(self):
        nbins = 2
        C, P1, P2 = self.cross[:nbins], self.pds1[:nbins], self.pds2[:nbins]
        bsq = bias_term(P1, P2, self.p1noise, self.p2noise, self.N)
        # must be lower than bsq!
        low_coh_cross = rng.normal(bsq**0.5 / 10, bsq**0.5 / 100) + 0.0j
        coh = raw_coherence(low_coh_cross, P1, P2, self.p1noise, self.p2noise, self.N)
        assert np.allclose(coh, 0)
        # Do it with a single number
        coh = raw_coherence(low_coh_cross[0], P1[0], P2[0], self.p1noise, self.p2noise, self.N)
        # Do it with a single complex object
        coh = raw_coherence(
            complex(low_coh_cross[0]), P1[0], P2[0], self.p1noise, self.p2noise, self.N
        )

    def test_intrinsic_neg_pow_warns(self):
        nbins = 2
        C, P1, P2 = np.array([4, 4]), np.array([3, 3]), np.array([1, 1])

        with pytest.warns(
            UserWarning, match="NaN values detected in intrinsic_coherence calculation."
        ):
            coh = intrinsic_coherence(C, P1, P2, 2, 2, 40)
            assert np.all(np.isnan(coh))
        # Do it with a single number

        with pytest.warns(
            UserWarning, match="NaN values detected in intrinsic_coherence calculation."
        ):
            coh = intrinsic_coherence(C[0], P1[0], P2[0], 2, 2, 40)
            assert np.isnan(coh)

    def test_intrinsic_low_coh_iteration(self):
        # bsq = (8 * 8 - 6 * 6) / 2. = 14 for coherence = 1, 36 for coherence = 0
        # C**2 = 36, so that C**2 - bsq is positive, but coherence is less than 1.
        # This will trigger the iteration.

        C, P1, P2 = np.array([6, 6]), np.array([8, 8]), np.array([8, 8])
        coh = intrinsic_coherence(C, P1, P2, 2, 2, 2, adjust_bias=True)
        assert np.allclose(coh, 0.22, atol=0.01)

        coh = intrinsic_coherence(C[0], P1[0], P2[0], 2, 2, 2, adjust_bias=True)
        assert np.isclose(coh, 0.22, atol=0.01)

    def test_intrinsic_low_coh_many_iteration(self):
        # bsq = (8 * 8 - 6 * 6) / 2. = 14 for coherence = 1, 36 for coherence = 0
        # C**2 > 36, so that C**2 - bsq is positive, but coherence is less than 1.
        # This will trigger the iteration.

        C, P1, P2 = np.array([6.1, 6.1]), np.array([8, 8]), np.array([8, 8])
        with pytest.warns(
            UserWarning,
            match="The iterative procedure to adjust the bias term did not converge after 40 iterations.",
        ):
            coh = intrinsic_coherence(C, P1, P2, 2, 2, 2, adjust_bias=True, atol=1e-15)
        assert np.allclose(coh, 0.2894444, atol=0.00001)

        with pytest.warns(
            UserWarning,
            match="The iterative procedure to adjust the bias term did not converge after 40 iterations.",
        ):
            coh = intrinsic_coherence(C[0], P1[0], P2[0], 2, 2, 2, adjust_bias=True, atol=1e-15)
        assert np.isclose(coh, 0.28944444, atol=0.00001)

    @pytest.mark.parametrize("adjust_bias", [True, False])
    def test_intrinsic_neg_0(self, adjust_bias):
        # bsq = (8 * 8 - gamma * 6 * 6) / 2. = 14 for gamma=1 and 36 for gamma=0
        # C**2 = 9, so that C**2 - bsq is always negative.

        C, P1, P2 = np.array([3, 3]), np.array([8, 8]), np.array([8, 8])
        with pytest.warns(UserWarning, match="Zero values detected in intrinsic_coherence"):
            coh, unc = intrinsic_coherence(
                C, P1, P2, 2, 2, 2, adjust_bias=adjust_bias, return_uncertainty=True
            )
        assert np.allclose(coh, 0)
        assert np.all(np.isnan(unc))

        with pytest.warns(UserWarning, match="Zero values detected in intrinsic_coherence"):
            coh = intrinsic_coherence(C[0], P1[0], P2[0], 2, 2, 2, adjust_bias=adjust_bias)
        assert np.isclose(coh, 0)

    @pytest.mark.parametrize("adjust_bias", [True, False])
    def test_intrinsic_invalid_power(self, adjust_bias):
        # bsq = (8 * 8 - gamma * 6 * 6) / 2. = 14 for gamma=1 and 36 for gamma=0
        # C**2 = 9, so that C**2 - bsq is always negative.

        C, P1, P2 = np.array([3, 3]), np.array([1, 1]), np.array([8, 8])
        with pytest.warns(UserWarning, match="NaN values detected in intrinsic_coherence"):
            coh, unc = intrinsic_coherence(
                C, P1, P2, 2, 2, 2, adjust_bias=adjust_bias, return_uncertainty=True
            )
        assert np.all(np.isnan(unc))
        assert np.all(np.isnan(coh))

        with pytest.warns(UserWarning, match="NaN values detected in intrinsic_coherence"):
            coh = intrinsic_coherence(C[0], P1[0], P2[0], 2, 2, 2, adjust_bias=adjust_bias)
        assert np.isnan(coh)

    def test_raw_high_bias(self):
        """Test when squared bias higher than squared norm of cross spec"""
        # Values chosen to have a high bias term, larger than |C|^2
        C = np.array([12986.0 + 8694.0j])
        P1 = np.array([476156.0])
        P2 = np.array([482751.0])
        P1noise = 495955
        P2noise = 494967
        coh = raw_coherence(C, P1, P2, P1noise, P2noise, 499, 1)
        coh_sngl = raw_coherence(C[0], P1[0], P2[0], P1noise, P2noise, 499, 1)
        assert np.allclose(coh, 0)
        assert np.isclose(coh_sngl, 0)

    def test_check_powers_for_intrinsic_coherence(self):
        pow1 = np.array([10, 20, 30])
        pow2 = np.array([10, 20, 30])
        pow1_noise = 5
        pow2_noise = 5
        n_ave = np.array([1, 10, 10])
        # Only one power is below the threshold, due to the low number of averaged powers.
        res = check_powers_for_intrinsic_coherence(pow1, pow2, pow1_noise, pow2_noise, n_ave, 3.0)
        assert np.all(res == np.array([True, False, False]))


class TestFourier(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 1
        cls.length = 100
        cls.ctrate = 10000
        cls.N = np.rint(cls.length / cls.dt).astype(int)
        cls.dt = cls.length / cls.N
        cls.times = np.sort(rng.uniform(0, cls.length, int(cls.length * cls.ctrate)))
        cls.gti = np.asanyarray([[0, cls.length]])
        cls.counts, bins = np.histogram(cls.times, bins=np.linspace(0, cls.length, cls.N + 1))
        cls.errs = np.ones_like(cls.counts) * np.sqrt(cls.ctrate)
        cls.bin_times = (bins[:-1] + bins[1:]) / 2
        cls.segment_size = 5.0
        cls.times2 = np.sort(rng.uniform(0, cls.length, int(cls.length * cls.ctrate)))
        cls.counts2, _ = np.histogram(cls.times2, bins=np.linspace(0, cls.length, cls.N + 1))
        cls.errs2 = np.ones_like(cls.counts2) * np.sqrt(cls.ctrate)

    def test_error_on_averaged_cross_spectrum_low_nave(self):
        with pytest.warns(UserWarning) as record:
            error_on_averaged_cross_spectrum(4 + 1.0j, 2, 4, 29, 2, 2)
        assert np.any(["n_ave is below 30." in r.message.args[0] for r in record])

    def test_ctrate_events(self):
        assert get_average_ctrate(self.times, self.gti, self.segment_size) == self.ctrate

    def test_ctrate_counts(self):
        assert (
            get_average_ctrate(self.bin_times, self.gti, self.segment_size, self.counts)
            == self.ctrate
        )

    def test_fts_from_segments_invalid(self):
        with pytest.raises(ValueError) as excinfo:
            # N and counts are both None. This should make the function fail immediately
            for _ in get_flux_iterable_from_segments(1, 2, 3, n_bin=None, fluxes=None):
                pass
        assert "At least one between fluxes" in str(excinfo.value)

    def test_fts_from_segments_cts_and_events_are_equal(self):
        N = np.rint(self.segment_size / self.dt).astype(int)
        fts_evts = [
            f
            for f in get_flux_iterable_from_segments(
                self.times, self.gti, self.segment_size, n_bin=N
            )
        ]
        fts_cts = [
            f
            for f in get_flux_iterable_from_segments(
                self.bin_times, self.gti, self.segment_size, fluxes=self.counts
            )
        ]
        for fe, fc in zip(fts_evts, fts_cts):
            assert np.allclose(fe, fc)

    def test_avg_pds_bad_input(self):
        times = np.sort(rng.uniform(0, 1000, 1))
        out_ev = avg_pds_from_timeseries(times, self.gti, self.segment_size, self.dt)
        assert out_ev is None

    @pytest.mark.parametrize("return_subcs", [True, False])
    @pytest.mark.parametrize("return_auxil", [True, False])
    def test_avg_cs_bad_input(self, return_auxil, return_subcs):
        times1 = np.sort(rng.uniform(0, 1000, 1))
        times2 = np.sort(rng.uniform(0, 1000, 1))
        out_ev = avg_cs_from_timeseries(
            times1,
            times2,
            self.gti,
            self.segment_size,
            self.dt,
            return_auxil=return_auxil,
            return_subcs=return_subcs,
        )
        assert out_ev is None

    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_pds_use_common_mean_similar_stats(self, norm):
        out_comm = avg_pds_from_timeseries(
            self.times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=True,
            silent=True,
            fluxes=None,
        )
        out = avg_pds_from_timeseries(
            self.times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=False,
            silent=True,
            fluxes=None,
        )
        assert np.isclose(out_comm["power"].std(), out["power"].std(), rtol=0.1)

    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_cs_use_common_mean_similar_stats(self, norm):
        out_comm = avg_cs_from_timeseries(
            self.times,
            self.times2,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=True,
            silent=True,
            return_subcs=True,
        )
        out = avg_cs_from_timeseries(
            self.times,
            self.times2,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=False,
            silent=True,
            return_subcs=True,
        )

        assert np.isclose(out_comm["power"].std(), out["power"].std(), rtol=0.1)
        assert np.isclose(out_comm["unnorm_power"].std(), out["unnorm_power"].std(), rtol=0.1)
        # Run the same check on the subcs
        assert np.isclose(out_comm.meta["subcs"].std(), out.meta["subcs"].std(), rtol=0.1)
        assert np.isclose(
            out_comm.meta["unnorm_subcs"].std(), out.meta["unnorm_subcs"].std(), rtol=0.1
        )
        # Now verify that the normalizations are consistent between single power and subcs
        assert np.isclose(
            out_comm["unnorm_power"].std() / out_comm["power"].std(),
            out_comm.meta["unnorm_subcs"].std() / out_comm.meta["subcs"].std(),
        )

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_pds_cts_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_pds_from_timeseries(
            self.times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=True,
            fluxes=None,
            return_subcs=True,
        )
        out_ct = avg_pds_from_timeseries(
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=True,
            fluxes=self.counts,
            return_subcs=True,
        )
        compare_tables(out_ev, out_ct)

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_pds_cts_and_err_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_pds_from_timeseries(
            self.times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=True,
            fluxes=None,
            return_subcs=True,
        )
        out_ct = avg_pds_from_timeseries(
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=True,
            fluxes=self.counts,
            errors=self.errs,
            return_subcs=True,
        )
        assert "subcs" in out_ct.meta
        assert "subcs" in out_ev.meta
        # The variance is not _supposed_ to be equal, when we specify errors
        if use_common_mean:
            compare_tables(out_ev, out_ct, rtol=0.01, discard=["variance"])
        else:
            compare_tables(out_ev, out_ct, rtol=0.1, discard=["variance"])

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_cs_cts_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_cs_from_timeseries(
            self.times,
            self.times2,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=False,
        )
        out_ct = avg_cs_from_timeseries(
            self.bin_times,
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=False,
            fluxes1=self.counts,
            fluxes2=self.counts2,
        )
        if use_common_mean:
            compare_tables(out_ev, out_ct, rtol=0.01)
        else:
            compare_tables(out_ev, out_ct, rtol=0.1)

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_cs_cts_and_err_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_cs_from_timeseries(
            self.times,
            self.times2,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=False,
        )
        out_ct = avg_cs_from_timeseries(
            self.bin_times,
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=False,
            fluxes1=self.counts,
            fluxes2=self.counts2,
            errors1=self.errs,
            errors2=self.errs2,
        )
        discard = [m for m in out_ev.meta.keys() if "variance" in m]
        if use_common_mean:
            compare_tables(out_ev, out_ct, rtol=0.01, discard=discard)
        else:
            compare_tables(out_ev, out_ct, rtol=0.1, discard=discard)


class TestNorms(object):
    @classmethod
    def setup_class(cls):
        cls.mean = cls.var = 100000.0
        cls.N = 800000
        cls.dt = 0.2
        cls.df = 1 / (cls.N * cls.dt)
        freq = fftfreq(cls.N, cls.dt)
        good = freq > 0
        cls.good = good
        cls.meanrate = cls.mean / cls.dt
        cls.lc = rng.poisson(cls.mean, cls.N).astype(float)
        cls.nph = np.sum(cls.lc)
        cls.pds = (np.abs(np.fft.fft(cls.lc)) ** 2)[good]
        cls.cross = ((np.fft.fft(cls.lc)) ** 2)[good]
        cls.lc_bksub = cls.lc - cls.mean
        cls.pds_bksub = (np.abs(np.fft.fft(cls.lc_bksub)) ** 2)[good]
        cls.lc_renorm = cls.lc / cls.mean
        cls.pds_renorm = (np.abs(np.fft.fft(cls.lc_renorm)) ** 2)[good]
        cls.lc_renorm_bksub = cls.lc_renorm - 1
        cls.pds_renorm_bksub = (np.abs(np.fft.fft(cls.lc_renorm_bksub)) ** 2)[good]

    def test_leahy_bksub_var_vs_standard(self):
        """Test that the Leahy norm. does not change with background-subtracted lcs"""
        leahyvar = normalize_leahy_from_variance(self.pds_bksub, np.var(self.lc_bksub), self.N)
        leahy = 2 * self.pds / np.sum(self.lc)
        ratio = np.mean(leahyvar / leahy)
        assert np.isclose(ratio, 1, rtol=0.01)

    def test_abs_bksub(self):
        """Test that the abs rms normalization does not change with background-subtracted lcs"""
        ratio = normalize_abs(self.pds_bksub, self.dt, self.N) / normalize_abs(
            self.pds, self.dt, self.N
        )
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_frac_renorm_constant(self):
        """Test that the fractional rms normalization is equivalent when renormalized"""
        ratio = normalize_frac(self.pds_renorm, self.dt, self.N, 1) / normalize_frac(
            self.pds, self.dt, self.N, self.mean
        )
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_frac_to_abs_ctratesq(self):
        """Test that fractional rms normalization x ctrate**2 is equivalent to abs renormalized"""
        ratio = (
            normalize_frac(self.pds, self.dt, self.N, self.mean)
            / normalize_abs(self.pds, self.dt, self.N)
            * self.meanrate**2
        )
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_total_variance(self):
        """Test that the total variance of the unnormalized pds is the same as
        the variance from the light curve
        Attention: VdK defines the variance as sum (x - x0)**2.
        The usual definition is divided by 'N'
        """
        vdk_total_variance = np.sum((self.lc - self.mean) ** 2)
        ratio = np.mean(self.pds) / vdk_total_variance
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level(self, norm):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm=norm
        )

        assert np.isclose(
            pdsnorm.mean(), poisson_level(meanrate=self.meanrate, norm=norm), rtol=0.01
        )

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level_real(self, norm):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm=norm, power_type="real"
        )

        assert np.isclose(
            pdsnorm.mean(), poisson_level(meanrate=self.meanrate, norm=norm), rtol=0.01
        )

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level_absolute(self, norm):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm=norm, power_type="abs"
        )

        assert np.isclose(
            pdsnorm.mean(), poisson_level(meanrate=self.meanrate, norm=norm), rtol=0.01
        )

    def test_normalize_with_variance(self):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, variance=self.var, norm="leahy"
        )
        assert np.isclose(pdsnorm.mean(), 2, rtol=0.01)

    def test_normalize_with_variance_fails_if_variance_zero(self):
        # If the variance is zero, it will fail:
        with pytest.raises(ValueError) as excinfo:
            pdsnorm = normalize_leahy_from_variance(self.pds, 0.0, self.N)
        assert "The variance used to normalize the" in str(excinfo.value)

    def test_normalize_none(self):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm="none"
        )
        assert np.isclose(pdsnorm.mean(), self.pds.mean(), rtol=0.01)

    def test_normalize_badnorm(self):
        with pytest.raises(ValueError):
            pdsnorm = normalize_periodograms(
                self.pds, self.var, self.N, self.mean, n_ph=self.nph, norm="asdfjlasdjf"
            )

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy", "none"])
    @pytest.mark.parametrize("power_type", ["all", "real", "abs"])
    def test_unnormalize_periodogram(self, norm, power_type):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm=norm, power_type=power_type
        )

        pdsunnorm = unnormalize_periodograms(
            pdsnorm, self.dt, self.N, n_ph=self.nph, norm=norm, power_type=power_type
        )

        check_allclose_and_print(self.pds, pdsunnorm, rtol=0.001)

    @pytest.mark.parametrize("norm", ["leahy"])
    @pytest.mark.parametrize("power_type", ["all", "real", "abs"])
    def test_unnorm_periodograms_variance(self, norm, power_type):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm=norm, power_type=power_type
        )

        pdsunnorm = unnormalize_periodograms(
            pdsnorm, self.dt, self.N, n_ph=self.nph, variance=None, norm=norm, power_type=power_type
        )

        pdsunnorm_var = unnormalize_periodograms(
            pdsnorm, self.dt, self.N, n_ph=self.nph, variance=1.0, norm=norm, power_type=power_type
        )

        check_allclose_and_print(pdsunnorm_var, pdsunnorm, rtol=0.001)

    @pytest.mark.parametrize("power_type", ["all", "real", "abs"])
    def test_unnorm_periodograms_background(self, power_type):
        background = 1.0
        pdsnorm = normalize_frac(self.pds, self.dt, self.N, self.mean, background_flux=background)
        pdsunnorm_bkg = unnormalize_periodograms(
            pdsnorm,
            self.dt,
            self.N,
            n_ph=self.nph,
            background_flux=background,
            norm="frac",
            power_type=power_type,
        )
        check_allclose_and_print(self.pds, pdsunnorm_bkg, rtol=0.001)

    def test_unorm_periodogram_wrong_norm(self):
        with pytest.raises(ValueError, match="Unknown value for the norm"):
            unnormalize_periodograms(
                self.pds, self.dt, self.N, n_ph=self.nph, norm="wrong", power_type="all"
            )

    def test_unnorm_periodogram_wrong_type(self):
        with pytest.raises(ValueError, match="Unrecognized power type"):
            unnormalize_periodograms(
                self.pds, self.dt, self.N, n_ph=self.nph, norm="frac", power_type="None"
            )

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    @pytest.mark.parametrize("power_type", ["all", "real"])
    def test_unnormalize_poisson_noise(self, norm, power_type):
        noise = poisson_level(norm, self.meanrate, self.nph)
        unnorm_noise = unnormalize_periodograms(
            noise, self.dt, self.N, n_ph=self.nph, norm=norm, power_type=power_type
        )
        noise_notnorm = poisson_level("none", self.meanrate, self.nph)

        assert np.isclose(noise_notnorm, unnorm_noise)


@pytest.mark.parametrize("phlag", [0.05, 0.1, 0.2, 0.4])
def test_lags(phlag):
    freq = 1.1123232252

    def func(time, phase=0):
        return 2 + np.sin(2 * np.pi * (time * freq - phase))

    time = np.sort(rng.uniform(0, 100, 3000))
    ft0 = lsft_slow(func(time, 0), time, np.array([freq]))
    ft1 = lsft_slow(func(time, phlag), time, np.array([freq]))
    measured_lag = (np.angle(ft1) - np.angle(ft0)) / 2 / np.pi
    while measured_lag > 0.5:
        measured_lag -= 1
    while measured_lag <= -0.5:
        measured_lag += 1

    assert np.isclose(measured_lag, phlag, atol=0.02, rtol=0.02)


def test_lsft_slow_fast():
    np.random.seed(0)
    rand = np.random.default_rng(42)
    n = 1000
    t = np.sort(rand.random(n)) * np.sqrt(n)
    y = np.sin(2 * np.pi * 3.0 * t)
    sub = np.min(y)
    y -= sub
    freqs = np.fft.fftfreq(n, np.median(np.diff(t, 1)))
    freqs = freqs[freqs >= 0]
    lsftslow = lsft_slow(y, t, freqs, sign=1)
    lsftfast = lsft_fast(y, t, freqs, sign=1, oversampling=10)
    assert np.argmax(lsftslow) == np.argmax(lsftfast)
    assert round(freqs[np.argmax(lsftslow)], 1) == round(freqs[np.argmax(lsftfast)], 1) == 3.0
    assert np.allclose((lsftslow * np.conjugate(lsftslow)).imag, [0]) & np.allclose(
        (lsftfast * np.conjugate(lsftfast)).imag, 0
    )


def test_impose_symmetry_lsft():
    np.random.seed(0)
    rand = np.random.default_rng(42)
    n = 1000
    t = np.sort(rand.random(n)) * np.sqrt(n)
    y = np.sin(2 * np.pi * 3.0 * t)
    sub = np.min(y)
    y -= sub
    freqs = np.fft.fftfreq(n, np.median(np.diff(t, 1)))
    freqs = freqs[freqs >= 0]
    lsftslow = lsft_slow(y, t, freqs, sign=1)
    lsftfast = lsft_fast(y, t, freqs, sign=1, oversampling=5)
    imp_sym_slow, freqs_new_slow = impose_symmetry_lsft(lsftslow, 0, n, freqs)
    imp_sym_fast, freqs_new_fast = impose_symmetry_lsft(lsftfast, 0, n, freqs)
    assert imp_sym_slow.shape == imp_sym_fast.shape == freqs_new_fast.shape == freqs_new_slow.shape
    assert np.all((imp_sym_slow.real) == np.flip(imp_sym_slow.real))
    assert np.all((imp_sym_slow.imag) == -np.flip(imp_sym_slow.imag))
    assert np.all((imp_sym_fast.real) == np.flip(imp_sym_fast.real))
    assert np.all((imp_sym_fast.imag) == (-np.flip(imp_sym_fast.imag)))
    assert np.all(freqs_new_slow == freqs_new_fast)


class TestIntegration(object):
    @classmethod
    def setup_class(cls):
        cls.freq = [0, 1, 2, 3]
        cls.power = [2, 2, 2, 2]
        cls.power_err = [1, 1, 1, 1]

    def test_power_integration_middle_bin(self):
        freq_range = [1, 2]
        pow, powe = integrate_power_in_frequency_range(self.freq, self.power, freq_range)
        assert np.isclose(pow, 2)
        assert np.isclose(powe, np.sqrt(2))

    def test_power_integration_precise(self):
        freq_range = [0.5, 2.5]
        df = 1
        pow, powe = integrate_power_in_frequency_range(self.freq, self.power, freq_range, df=df)
        assert np.allclose(pow, 4)
        assert np.allclose(powe, 2 * np.sqrt(2))

    def test_power_integration_poisson(self):
        freq_range = [0.5, 2.5]
        for poisson_power in (1, np.ones_like(self.power)):
            pow, powe = integrate_power_in_frequency_range(
                self.freq, self.power, freq_range, poisson_power=poisson_power
            )
            assert np.allclose(pow, 2)
            assert np.allclose(powe, 2 * np.sqrt(2))

    def test_power_integration_err(self):
        freq_range = [0.5, 2.5]
        pow, powe = integrate_power_in_frequency_range(
            self.freq, self.power, freq_range, power_err=self.power_err
        )
        assert np.allclose(pow, 4)
        assert np.allclose(powe, np.sqrt(2))

    def test_power_integration_m(self):
        freq_range = [0.5, 2.5]
        pow, powe = integrate_power_in_frequency_range(self.freq, self.power, freq_range, m=4)
        assert np.allclose(pow, 4)
        assert np.allclose(powe, np.sqrt(2))


class TestRMS(object):
    @classmethod
    def setup_class(cls):
        fwhm = 0.23456
        cls.segment_size = 256
        cls.df = 1 / cls.segment_size

        cls.freqs = np.arange(cls.df, 1.54232, cls.df)
        pds_shape_func = Lorentz1D(x_0=0, fwhm=fwhm)
        cls.pds_shape_raw = pds_shape_func(cls.freqs)

        pds_shape_func_qpo = Lorentz1D(x_0=0, fwhm=0.312567) + Lorentz1D(x_0=0.5, fwhm=0.1)
        cls.pds_shape_qpo_raw = pds_shape_func_qpo(cls.freqs)

    def _prepare_pds_for_rms_tests(self, rms, nphots, M, distort_poisson_by=1, with_qpo=False):
        meanrate = nphots / self.segment_size
        poisson_noise_rms = 2 / meanrate
        pds_shape = self.pds_shape_raw if not with_qpo else self.pds_shape_qpo_raw

        pds_shape_rms = pds_shape / np.sum(pds_shape * self.df) * rms**2
        pds_shape_rms += poisson_noise_rms * distort_poisson_by

        random_part = rng.chisquare(2 * M, size=pds_shape.size) / 2 / M
        pds_rms_noisy = random_part * pds_shape_rms

        pds_unnorm = pds_rms_noisy * meanrate / 2 * nphots
        return pds_rms_noisy, pds_unnorm

    @pytest.mark.parametrize("M", [100, 10000])
    @pytest.mark.parametrize("nphots", [100_000, 1_000_000])
    @pytest.mark.parametrize("rms", [0.05, 0.1, 0.32, 0.5])
    @pytest.mark.parametrize("with_qpo", [False, True])
    def test_rms(self, M, nphots, rms, with_qpo):
        meanrate = nphots / self.segment_size
        poisson_noise_rms = 2 / meanrate
        pds_rms_noisy, pds_unnorm = self._prepare_pds_for_rms_tests(
            rms, nphots, M, with_qpo=with_qpo
        )

        rms_from_unnorm, rmse_from_unnorm = get_rms_from_unnorm_periodogram(
            pds_unnorm,
            nphots,
            self.df,
            M=M,
        )
        rms_from_rms, rmse_from_rms = get_rms_from_rms_norm_periodogram(
            pds_rms_noisy, poisson_noise_rms, self.df, M
        )

        assert np.isclose(rms_from_rms, rms, atol=3 * rmse_from_rms)
        assert np.isclose(rms_from_unnorm, rms, atol=3 * rmse_from_unnorm)

    @pytest.mark.parametrize("M", [100, 10000])
    @pytest.mark.parametrize("nphots", [100_000, 1_000_000])
    @pytest.mark.parametrize("rms", [0.05, 0.1, 0.32, 0.5])
    def test_rms_abs(self, M, nphots, rms):
        meanrate = nphots / self.segment_size
        _, pds_unnorm = self._prepare_pds_for_rms_tests(rms, nphots, M)

        rms_from_unnorm, rmse_from_unnorm = get_rms_from_unnorm_periodogram(
            pds_unnorm, nphots, self.df, M=M, kind="abs"
        )
        assert np.isclose(rms_from_unnorm, rms * meanrate, atol=3 * rmse_from_unnorm * meanrate)

    @pytest.mark.parametrize("M", [1, 10])
    @pytest.mark.parametrize("nphots", [100_000, 1_000_000])
    @pytest.mark.parametrize("rms", [0.05, 0.1, 0.32, 0.5])
    def test_rms_M_low(self, M, nphots, rms):
        """Test that the warning is raised when M is low."""
        meanrate = nphots / self.segment_size
        poisson_noise_rms = 2 / meanrate

        pds_rms_noisy, pds_unnorm = self._prepare_pds_for_rms_tests(rms, nphots, M)

        with pytest.warns(UserWarning, match="All power spectral bins have M<30."):
            rms_from_unnorm, rmse_from_unnorm = get_rms_from_unnorm_periodogram(
                pds_unnorm,
                nphots,
                self.df,
                M=M,
            )
        with pytest.warns(UserWarning, match="All power spectral bins have M<30."):
            rms_from_rms, rmse_from_rms = get_rms_from_rms_norm_periodogram(
                pds_rms_noisy, poisson_noise_rms, self.df, M
            )

        assert np.isclose(rms_from_rms, rms, atol=3 * rmse_from_rms)
        assert np.isclose(rms_from_unnorm, rms, atol=3 * rmse_from_unnorm)

    @pytest.mark.parametrize("nphots", [100_000, 1_000_000])
    def test_rms_low(self, nphots):
        meanrate = nphots / self.segment_size
        poisson_noise_rms = 2 / meanrate
        M = 100

        pds_rms_noisy, pds_unnorm = self._prepare_pds_for_rms_tests(
            0, nphots, M, distort_poisson_by=0.9
        )

        with pytest.warns(UserWarning, match="Poisson-subtracted power is below 0"):
            get_rms_from_unnorm_periodogram(
                pds_unnorm,
                nphots,
                self.df,
                M=M,
                kind="frac",
            )
        with pytest.warns(UserWarning, match="Poisson-subtracted power is below 0"):
            get_rms_from_rms_norm_periodogram(pds_rms_noisy, poisson_noise_rms, self.df, M)

    def test_array_m_and_df(self):
        # Very safe, high-rms dataset
        nphots = 1_000_000
        rms = 0.5
        M = 1000

        meanrate = nphots / self.segment_size
        poisson_noise_rms = 2 / meanrate

        pds_rms_noisy, _ = self._prepare_pds_for_rms_tests(rms, nphots, M)

        M = np.zeros_like(pds_rms_noisy) + 100
        df = np.zeros_like(pds_rms_noisy) + self.df

        rms_from_rms, rmse_from_rms = get_rms_from_rms_norm_periodogram(
            pds_rms_noisy, poisson_noise_rms, df, M
        )

        assert np.isclose(rms_from_rms, rms, atol=3 * rmse_from_rms)

    def test_incompatible_m_and_df(self):
        # Make df non constant
        df = np.zeros_like(self.pds_shape_raw) + self.df
        df[-1] = 2 * self.df

        with pytest.raises(
            ValueError, match="M and df must be either both constant, or none of them."
        ):
            get_rms_from_rms_norm_periodogram(self.pds_shape_raw, 2, df, M=100)

    def test_invalid_kind(self):
        # Make df non constant

        with pytest.raises(ValueError, match="Only 'frac' or 'abs' rms are supported."):
            get_rms_from_unnorm_periodogram(self.pds_shape_raw, 2, 0.1, M=100, kind="asdfkhf")

    def test_deprecation_rms_calculation(self):
        nphots = 1_000_000
        rms = 0.5
        M = 1000
        _, pds_unnorm = self._prepare_pds_for_rms_tests(rms, nphots, M)
        with pytest.warns(DeprecationWarning, match="The rms_calculation function is deprecated"):
            rms, _ = rms_calculation(
                pds_unnorm,
                self.freqs.min(),
                self.freqs.max(),
                nphots,
                self.segment_size,
                M,
                1,
                len(self.freqs),
                poisson_noise_unnorm=nphots,
            )
        rms_from_unnorm, rmse_from_unnorm = get_rms_from_unnorm_periodogram(
            pds_unnorm,
            nphots,
            self.df,
            M=M,
        )
        assert np.isclose(rms, rms_from_unnorm, atol=3 * rmse_from_unnorm)


@pytest.mark.parametrize("ntimes", [100, 1000])
def test_shift_and_add_orbit(ntimes):
    # This time correct for orbital motion
    from stingray.fourier import shift_and_add

    fmid = 0.7
    freqs = np.linspace(0.699, 0.701, 1001)
    porb = 2.52 * 86400
    asini = 22.5
    t0 = porb / 2
    times = np.linspace(0, porb, ntimes + 1)[:-1]
    power_list = np.zeros((times.size, freqs.size))
    omega = 2 * np.pi / porb
    orbit_freqs = fmid * (1 - asini * omega * np.cos(omega * (times - t0)))

    idx = np.searchsorted(freqs, orbit_freqs)
    for i_t, power in zip(idx, power_list):
        power[i_t] = 1

    f, p, n = shift_and_add(freqs, power_list, orbit_freqs, nbins=5)
    # If we corrected well, the power should be the average of all max powers in the
    # original series
    assert np.max(p) == 1
    assert np.max(n) == times.size

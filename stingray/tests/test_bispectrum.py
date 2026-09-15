import os
import copy

import numpy as np
import pytest
import matplotlib.pyplot as plt

from stingray import Lightcurve, EventList, StingrayTimeseries
from stingray.bispectrum import (
    Bispectrum,
    AveragedBispectrum,
    CrossBispectrum,
    AveragedCrossBispectrum,
    DynamicalBispectrum,
    DynamicalCrossBispectrum,
    crossbispectrum_from_stingray_timeseries,
    crossbispectrum_from_lc_iterable,
)
from stingray.fourier import (
    fftfreq,
    positive_fft_bins,
    avg_bispectrum_from_iterable,
    avg_bispectrum_from_timeseries,
    avg_cross_bispectrum_from_iterables,
    bicoherence_from_sums,
    BICOHERENCE_NORMS,
    _bispectrum_frequency_grid,
)
import stingray.bispectrum as bispectrum_module

# The bispectrum module emits a one-time UserWarning about the changed interface
# the first time any bispectrum object is constructed. Fixtures and tests build
# such objects, which would otherwise fire the notice at an unpredictable point
# (an error under ``filterwarnings = ["error", ...]``). Pre-set the once-flag so
# the suite stays quiet; ``TestBispectrum.test_interface_change_warning_emitted``
# resets it to assert the notice is actually raised.
bispectrum_module._INTERFACE_CHANGE_WARNED = True


def clear_all_figs():
    fign = plt.get_fignums()
    for fig in fign:
        plt.close(fig)


rng = np.random.RandomState(20150907)

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


def _brute_force_bispectrum(segments, n_bin):
    """Reference bispectrum and the three bicoherence normalizations.

    A direct, slow, double-loop implementation of the Maccarone (2013)
    estimator, used to validate the vectorized ``avg_bispectrum_*`` functions.
    """
    fgt0 = positive_fft_bins(n_bin)
    kbins = np.arange(fgt0.start, fgt0.stop)
    nf = len(kbins)
    nyq = n_bin // 2
    num = np.zeros((nf, nf), dtype=complex)
    den1 = np.zeros((nf, nf))
    den2 = np.zeros((nf, nf))
    sum_abs = np.zeros((nf, nf))
    valid = np.zeros((nf, nf), dtype=bool)
    for s in segments:
        ft = np.fft.fft(np.asarray(s))
        for a, ka in enumerate(kbins):
            for b, kb in enumerate(kbins):
                if ka + kb > nyq:
                    continue
                valid[a, b] = True
                t = ft[ka] * ft[kb] * np.conj(ft[ka + kb])
                num[a, b] += t
                den1[a, b] += np.abs(ft[ka] * ft[kb]) ** 2
                den2[a, b] += np.abs(ft[ka + kb]) ** 2
                sum_abs[a, b] += np.abs(t)
    k = len(segments)
    bispec = np.full((nf, nf), np.nan, dtype=complex)
    norms = {n: np.full((nf, nf), np.nan) for n in BICOHERENCE_NORMS}
    bispec[valid] = num[valid] / k
    norms["kim_powers"][valid] = np.abs(num[valid]) ** 2 / (den1[valid] * den2[valid])
    norms["sigl_chamoun"][valid] = np.abs(num[valid]) / np.sqrt(den1[valid] * den2[valid])
    norms["hagihira"][valid] = np.abs(num[valid]) / sum_abs[valid]
    return bispec, norms, valid


def _coupled_segments(rng_local, n_seg, n_bin, dt, f1, f2, couple=True):
    """Segments with (optionally) quadratic phase coupling of f1, f2 -> f1+f2."""
    t = np.arange(n_bin) * dt
    segs = []
    for _ in range(n_seg):
        p1, p2 = rng_local.uniform(0, 2 * np.pi, size=2)
        p3 = p1 + p2 if couple else rng_local.uniform(0, 2 * np.pi)
        s = (
            np.cos(2 * np.pi * f1 * t + p1)
            + np.cos(2 * np.pi * f2 * t + p2)
            + np.cos(2 * np.pi * (f1 + f2) * t + p3)
            + 20
        )
        segs.append(s)
    return segs


class TestBispectrum(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.1
        cls.n = 256
        cls.time = np.arange(cls.n) * cls.dt
        cls.counts = rng.poisson(50, cls.n).astype(float)
        cls.lc = Lightcurve(cls.time, cls.counts, dt=cls.dt, skip_checks=True)
        cls.events = EventList(
            np.sort(rng.uniform(0, cls.n * cls.dt, 5000)), gti=[[0, cls.n * cls.dt]]
        )
        cls.bs = Bispectrum(cls.lc)

    def test_interface_change_warning_emitted(self):
        # The interface-change notice is shown once, on the first bispectrum
        # object constructed in a session. Reset the module flag so we can assert
        # the notice is raised, then restore it so the rest of the suite stays
        # quiet (the warning is otherwise an error under filterwarnings=error).
        bispectrum_module._INTERFACE_CHANGE_WARNED = False
        try:
            with pytest.warns(UserWarning, match="bispectrum interface has changed"):
                Bispectrum(self.lc)
        finally:
            bispectrum_module._INTERFACE_CHANGE_WARNED = True

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_initialize_empty(self, skip_checks):
        bs = Bispectrum(skip_checks=skip_checks)
        assert bs.freq is None
        assert bs.bispec is None
        assert bs.m == 1

    def test_make_empty_bispectrum(self):
        bs = Bispectrum()
        assert bs.bicoherence is None
        assert bs.biphase is None
        assert bs.bispec_mag is None

    def test_make_bispectrum_from_lightcurve(self):
        bs = Bispectrum(self.lc)
        assert isinstance(bs, Bispectrum)
        assert bs.m == 1
        assert bs.n == self.n

    def test_bispectrum_types(self):
        assert self.bs.type == "bispectrum"

    def test_type_change(self):
        bs = copy.deepcopy(self.bs)
        assert bs.type == "bispectrum"
        bs.type = "astdfawerfsaf"
        assert bs.type == "astdfawerfsaf"

    def test_shapes(self):
        nf = self.bs.freq.size
        assert self.bs.bispec.shape == (nf, nf)
        assert self.bs.bicoherence.shape == (nf, nf)
        assert self.bs.biphase.shape == (nf, nf)
        assert self.bs.bispec_mag.shape == (nf, nf)
        assert self.bs.bispec_err.shape == (nf, nf)

    def test_init_without_lightcurve(self):
        with pytest.raises(TypeError):
            Bispectrum(self.lc.counts)

    def test_init_with_nonsense_data(self):
        nonsense_data = [None for i in range(100)]
        with pytest.raises(TypeError):
            Bispectrum(nonsense_data)

    def test_init_with_wrong_data_type(self):
        with pytest.raises(TypeError):
            Bispectrum(1)

    def test_eventlist_needs_dt(self):
        with pytest.raises(ValueError):
            Bispectrum(self.events)

    def test_lc_keyword_deprecation(self):
        with pytest.warns(DeprecationWarning) as record:
            bs1 = Bispectrum(lc=self.lc)
        assert np.any(["deprecated" in r.message.args[0].lower() for r in record])
        bs2 = Bispectrum(data=self.lc)
        good = np.isfinite(bs1.bispec)
        assert np.allclose(bs1.bispec[good], bs2.bispec[good])

    def test_bispec_mag_and_phase(self):
        good = np.isfinite(self.bs.bispec)
        assert np.allclose(self.bs.bispec_mag[good], np.abs(self.bs.bispec[good]))
        assert np.allclose(self.bs.bispec_phase[good], self.bs.biphase[good])

    def test_redundant_region_is_nan(self):
        assert np.any(np.isnan(self.bs.bispec))
        assert np.array_equal(np.isnan(self.bs.bispec), ~self.bs.valid)

    def test_from_lightcurve_works(self):
        bs = Bispectrum.from_lightcurve(self.lc)
        good = np.isfinite(bs.bispec)
        assert np.allclose(bs.bispec[good], self.bs.bispec[good])

    def test_from_events_works(self):
        bs = Bispectrum.from_events(self.events, dt=self.dt)
        assert isinstance(bs, Bispectrum)

    def test_from_time_array_works(self):
        bs = Bispectrum.from_time_array(self.events.time, dt=self.dt, gti=[[0, self.n * self.dt]])
        assert isinstance(bs, Bispectrum)

    def test_from_stingray_timeseries_works(self):
        ts = StingrayTimeseries(
            self.time, array_attrs={"flux": self.counts}, dt=self.dt, skip_checks=True
        )
        ts.gti = self.lc.gti
        bs = Bispectrum.from_stingray_timeseries(ts, "flux")
        good = np.isfinite(bs.bispec)
        assert np.allclose(bs.bispec[good], self.bs.bispec[good])


class TestAveragedBispectrum(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.1
        cls.n = 512
        cls.segment_size = 5.0
        cls.time = np.arange(cls.n) * cls.dt
        cls.counts = rng.poisson(40, cls.n).astype(float)
        cls.lc = Lightcurve(cls.time, cls.counts, dt=cls.dt, skip_checks=True)
        cls.gauss_lc = Lightcurve(
            cls.time,
            cls.counts,
            err=np.sqrt(cls.counts + 1),
            err_dist="gauss",
            dt=cls.dt,
            skip_checks=True,
        )
        cls.events = EventList(
            np.sort(rng.uniform(0, cls.n * cls.dt, 8000)), gti=[[0, cls.n * cls.dt]]
        )
        cls.bs = AveragedBispectrum(cls.lc, segment_size=cls.segment_size)

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_initialize_empty(self, skip_checks):
        bs = AveragedBispectrum(skip_checks=skip_checks)
        assert bs.freq is None
        assert bs.m == 1

    def test_averaged_bispectrum_from_lightcurve(self):
        assert isinstance(self.bs, AveragedBispectrum)
        assert self.bs.m > 1

    @pytest.mark.parametrize("nseg", [1, 2, 5, 10])
    def test_n_segments(self, nseg):
        segment_size = self.n * self.dt / nseg
        bs = AveragedBispectrum(self.lc, segment_size=segment_size)
        assert bs.m == nseg

    def test_segments_with_leftover(self):
        # A segment size that does not divide the light curve evenly
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size * 1.5)
        assert bs.m > 1

    def test_init_without_segment(self):
        with pytest.raises(ValueError):
            AveragedBispectrum(self.lc)

    def test_init_with_none_segment(self):
        with pytest.raises(ValueError):
            AveragedBispectrum(self.lc, segment_size=None)

    def test_lc_keyword_deprecation(self):
        with pytest.warns(DeprecationWarning):
            AveragedBispectrum(lc=self.lc, segment_size=self.segment_size)

    def test_from_lightcurve_works(self):
        bs = AveragedBispectrum.from_lightcurve(self.lc, segment_size=self.segment_size)
        good = np.isfinite(bs.bispec)
        assert np.allclose(bs.bispec[good], self.bs.bispec[good])

    def test_from_events_works(self):
        bs = AveragedBispectrum.from_events(self.events, dt=self.dt, segment_size=self.segment_size)
        assert isinstance(bs, AveragedBispectrum)
        assert bs.m > 1

    def test_from_time_array_works(self):
        bs = AveragedBispectrum.from_time_array(
            self.events.time,
            dt=self.dt,
            segment_size=self.segment_size,
            gti=[[0, self.n * self.dt]],
        )
        assert bs.m > 1

    def test_list_of_light_curves(self):
        bs = AveragedBispectrum([self.lc, self.lc], segment_size=self.segment_size)
        assert isinstance(bs, AveragedBispectrum)

    def test_from_lc_iterable_works(self):
        bs = AveragedBispectrum.from_lc_iterable(
            [self.lc, self.lc], self.dt, segment_size=self.segment_size
        )
        assert isinstance(bs, AveragedBispectrum)

    def test_list_with_nonsense_component(self):
        with pytest.raises(TypeError):
            AveragedBispectrum([self.lc, 3], segment_size=self.segment_size)

    def test_save_all(self):
        bs = AveragedBispectrum.from_lightcurve(
            self.lc, segment_size=self.segment_size, save_all=True
        )
        assert hasattr(bs, "bispec_all")
        assert len(bs.bispec_all) == bs.m

    def test_gti_is_threaded_through(self):
        full = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        half = AveragedBispectrum(
            self.lc, segment_size=self.segment_size, gti=[[0, self.n * self.dt / 2]]
        )
        assert half.m < full.m

    def test_skip_checks(self):
        AveragedBispectrum(self.lc, segment_size=self.segment_size, skip_checks=True)

    def test_initial_checks_none_returns_false(self):
        assert Bispectrum().initial_checks(data=None) is False

    def test_averaged_segment_too_small(self):
        with pytest.raises(ValueError):
            AveragedBispectrum(self.lc, segment_size=self.dt)

    def test_eventlist_init_dispatch(self):
        bs = Bispectrum(self.events, dt=0.1)
        assert bs.bispec is not None

    def test_gauss_errors_lightcurve(self):
        bs = AveragedBispectrum(self.gauss_lc, segment_size=self.segment_size)
        assert bs.m > 1

    def test_from_lc_iterable_with_gti_and_errors(self):
        bs = AveragedBispectrum.from_lc_iterable(
            [self.gauss_lc], self.dt, self.segment_size, gti=[[0, self.n * self.dt]]
        )
        assert bs.m > 1

    def test_from_lc_iterable_arrays(self):
        n_bin = int(self.segment_size / self.dt)
        segs = [self.counts[i : i + n_bin] for i in range(0, self.n - n_bin, n_bin)]
        bs = AveragedBispectrum.from_lc_iterable(segs, self.dt, self.segment_size)
        assert bs.m > 1

    def test_averaged_from_stingray_timeseries_with_errors(self):
        ts = StingrayTimeseries(
            self.time,
            array_attrs={"flux": self.counts, "flux_err": np.sqrt(self.counts + 1)},
            dt=self.dt,
            skip_checks=True,
        )
        ts.gti = self.lc.gti
        bs = AveragedBispectrum.from_stingray_timeseries(
            ts, "flux", self.segment_size, error_flux_attr="flux_err"
        )
        assert bs.m > 1

    def test_averaged_segment_too_small_events(self):
        with pytest.raises(ValueError):
            AveragedBispectrum(self.events, segment_size=0.05, dt=0.1)

    def test_generator_input_warns(self):
        with pytest.warns(UserWarning):
            AveragedBispectrum((lc for lc in [self.lc]), segment_size=self.segment_size)


class TestBispectrumEstimator(object):
    """Tests of the underlying Fourier estimator functions in ``fourier.py``
    (``avg_bispectrum_from_iterable``, ``avg_bispectrum_from_timeseries``,
    ``bicoherence_from_sums``, ``_bispectrum_frequency_grid``).
    """

    @classmethod
    def setup_class(cls):
        cls.dt = 0.1
        cls.n_bin = 32
        cls.segments = [rng.poisson(25, cls.n_bin).astype(float) for _ in range(40)]

        # For the counts-vs-events equality test
        cls.length = 100.0
        cls.ctrate = 1000
        cls.segment_size = 5.0
        cls.N = int(cls.length / cls.dt)
        cls.times = np.sort(rng.uniform(0, cls.length, int(cls.length * cls.ctrate)))
        cls.gti = np.asanyarray([[0, cls.length]])
        cls.counts, bins = np.histogram(cls.times, bins=np.linspace(0, cls.length, cls.N + 1))
        cls.bin_times = (bins[:-1] + bins[1:]) / 2

    def test_frequency_grid(self):
        freq, idx1, idx2, idx3, valid = _bispectrum_frequency_grid(self.n_bin, self.dt)
        fgt0 = positive_fft_bins(self.n_bin)
        assert np.allclose(freq, fftfreq(self.n_bin, self.dt)[fgt0])
        assert idx1.shape == (freq.size, freq.size)
        # The valid region is where f1 + f2 is at or below the Nyquist bin
        assert np.all(idx3[valid] <= self.n_bin // 2)

    def test_no_segments_returns_none(self):
        assert avg_bispectrum_from_iterable(iter([]), self.dt, silent=True) is None

    def test_matches_brute_force(self):
        res = avg_bispectrum_from_iterable(iter(self.segments), self.dt, silent=True)
        bispec_ref, norms_ref, valid = _brute_force_bispectrum(self.segments, self.n_bin)
        assert np.array_equal(res.meta["valid"], valid)
        assert np.allclose(res.meta["bispec"][valid], bispec_ref[valid], atol=1e-9)
        assert np.allclose(
            res.meta["bicoherence"][valid], norms_ref["kim_powers"][valid], atol=1e-12
        )

    @pytest.mark.parametrize("norm", BICOHERENCE_NORMS)
    def test_all_norms_match_brute_force(self, norm):
        _, norms_ref, valid = _brute_force_bispectrum(self.segments, self.n_bin)
        res = avg_bispectrum_from_iterable(
            iter(self.segments), self.dt, bicoherence_norm=norm, silent=True
        )
        assert np.allclose(res.meta["bicoherence"][valid], norms_ref[norm][valid], atol=1e-12)

    def test_kim_powers_is_sigl_chamoun_squared(self):
        rk = avg_bispectrum_from_iterable(
            iter(self.segments), self.dt, bicoherence_norm="kim_powers", silent=True
        )
        rs = avg_bispectrum_from_iterable(
            iter(self.segments), self.dt, bicoherence_norm="sigl_chamoun", silent=True
        )
        valid = rk.meta["valid"]
        assert np.allclose(
            rk.meta["bicoherence"][valid], rs.meta["bicoherence"][valid] ** 2, atol=1e-12
        )

    def test_biphase_is_angle_of_bispectrum(self):
        res = avg_bispectrum_from_iterable(iter(self.segments), self.dt, silent=True)
        valid = res.meta["valid"]
        d = np.angle(
            np.exp(1j * (res.meta["biphase"][valid] - np.angle(res.meta["bispec"][valid])))
        )
        assert np.allclose(d, 0, atol=1e-9)

    def test_single_segment_bicoherence_is_one(self):
        res = avg_bispectrum_from_iterable(iter(self.segments[:1]), self.dt, silent=True)
        valid = res.meta["valid"]
        assert np.allclose(res.meta["bicoherence"][valid], 1.0)
        assert np.allclose(res.meta["biphase_err"][valid], 0.0)
        assert res.meta["m"] == 1

    def test_bad_norm_raises(self):
        with pytest.raises(ValueError):
            avg_bispectrum_from_iterable(
                iter(self.segments), self.dt, bicoherence_norm="bogus", silent=True
            )

    def test_bicoherence_from_sums_bad_norm(self):
        ones = np.ones((2, 2))
        with pytest.raises(ValueError):
            bicoherence_from_sums("bogus", ones, ones, ones, ones)

    def test_cts_and_events_are_equal(self):
        bs_evts = avg_bispectrum_from_timeseries(
            self.times, self.gti, self.segment_size, self.dt, silent=True
        )
        bs_cts = avg_bispectrum_from_timeseries(
            self.bin_times, self.gti, self.segment_size, self.dt, fluxes=self.counts, silent=True
        )
        valid = bs_evts.meta["valid"]
        assert np.allclose(bs_evts.meta["bispec"][valid], bs_cts.meta["bispec"][valid])
        assert np.allclose(bs_evts.meta["bicoherence"][valid], bs_cts.meta["bicoherence"][valid])

    @pytest.mark.parametrize("norm", BICOHERENCE_NORMS)
    def test_detects_quadratic_coupling(self, norm):
        rng_local = np.random.RandomState(99)
        dt, n_bin = 0.01, 128
        f1, f2 = 5.0, 12.0
        coupled = _coupled_segments(rng_local, 300, n_bin, dt, f1, f2, couple=True)
        control = _coupled_segments(rng_local, 300, n_bin, dt, f1, f2, couple=False)
        res_c = avg_bispectrum_from_iterable(iter(coupled), dt, bicoherence_norm=norm, silent=True)
        res_u = avg_bispectrum_from_iterable(iter(control), dt, bicoherence_norm=norm, silent=True)
        freq = res_c.meta["freq"]
        i1 = np.argmin(np.abs(freq - f1))
        i2 = np.argmin(np.abs(freq - f2))
        # Strong coupling gives a high bicoherence; the control (same power
        # spectrum, random relative phase) gives a low one, for every norm.
        assert res_c.meta["bicoherence"][i1, i2] > 0.7
        assert res_u.meta["bicoherence"][i1, i2] < 0.4
        assert res_c.meta["bicoherence"][i1, i2] > res_u.meta["bicoherence"][i1, i2]

    def test_biphase_of_coupling(self):
        rng_local = np.random.RandomState(5)
        dt, n_bin = 0.01, 128
        f1, f2 = 5.0, 12.0
        segs = _coupled_segments(rng_local, 200, n_bin, dt, f1, f2, couple=True)
        res = avg_bispectrum_from_iterable(iter(segs), dt, silent=True)
        freq = res.meta["freq"]
        i1 = np.argmin(np.abs(freq - f1))
        i2 = np.argmin(np.abs(freq - f2))
        assert abs(res.meta["biphase"][i1, i2]) < 0.2


class TestBispectrumCumulant(object):
    """The legacy 3rd-order-cumulant estimator, exposed as method='cumulant'."""

    @classmethod
    def setup_class(cls):
        cls.dt = 0.1
        cls.n = 256
        cls.lc = Lightcurve(
            np.arange(cls.n) * cls.dt,
            rng.poisson(50, cls.n).astype(float),
            dt=cls.dt,
            skip_checks=True,
        )

    def teardown_method(self):
        clear_all_figs()

    def test_default_method_is_fourier(self):
        bs = Bispectrum(self.lc)
        assert bs.method == "fourier"
        assert bs.bicoherence is not None
        assert bs.cum3 is None

    def test_bad_method_raises(self):
        with pytest.raises(ValueError):
            Bispectrum(self.lc, method="nope")
        with pytest.raises(ValueError):
            AveragedBispectrum(self.lc, segment_size=2.0, method="nope")

    def test_reproduces_reference_values(self):
        # Golden values from the original stingray cumulant Bispectrum docstring.
        lc = Lightcurve(
            np.array([1, 2, 3, 4, 5]), np.array([2, 3, 1, 1, 2]), dt=1, skip_checks=True
        )
        bs = Bispectrum(lc, method="cumulant", maxlag=1)
        assert np.allclose(bs.lags, [-1, 0, 1])
        assert np.allclose(bs.freq, [-0.5, 0.0, 0.5])
        cum3_ref = [[-0.2976, 0.1024, 0.1408], [0.1024, 0.144, -0.2976], [0.1408, -0.2976, 0.1024]]
        assert np.allclose(bs.cum3, cum3_ref, atol=1e-4)
        mag_ref = [[1.263368, 0.0032, 0.0032], [0.0032, 0.16, 0.0032], [0.0032, 0.0032, 1.263368]]
        assert np.allclose(bs.bispec_mag, mag_ref, atol=1e-4)

    def test_cumulant_attributes_and_shapes(self):
        maxlag = 30
        bs = Bispectrum(self.lc, method="cumulant", maxlag=maxlag)
        nlag = 2 * maxlag + 1
        assert bs.method == "cumulant"
        assert bs.cum3.shape == (nlag, nlag)
        assert bs.bispec.shape == (nlag, nlag)
        assert bs.lags.shape == (nlag,)
        assert bs.freq.shape == (nlag,)
        assert bs.maxlag == maxlag
        assert bs.scale == "biased"
        # the cumulant method produces no bicoherence
        assert bs.bicoherence is None

    def test_bispec_is_fft_of_cumulant(self):
        from stingray.utils import fftshift, fft2, ifftshift

        bs = Bispectrum(self.lc, method="cumulant", maxlag=20)
        expected = fftshift(fft2(ifftshift(bs.cum3)))
        assert np.allclose(bs.bispec, expected)
        assert np.allclose(bs.bispec_mag, np.abs(bs.bispec))
        assert np.allclose(bs.biphase, np.angle(bs.bispec))

    def test_windowed_differs_from_unwindowed(self):
        plain = Bispectrum(self.lc, method="cumulant", maxlag=30)
        windowed = Bispectrum(self.lc, method="cumulant", maxlag=30, window="parzen")
        assert not np.allclose(plain.bispec_mag, windowed.bispec_mag)
        assert windowed.window == "parzen"

    def test_biased_differs_from_unbiased(self):
        biased = Bispectrum(self.lc, method="cumulant", maxlag=30, scale="biased")
        unbiased = Bispectrum(self.lc, method="cumulant", maxlag=30, scale="unbiased")
        assert not np.allclose(biased.cum3, unbiased.cum3)

    def test_invalid_cumulant_params(self):
        with pytest.raises(ValueError):
            Bispectrum(self.lc, method="cumulant", scale="nope")
        with pytest.raises(ValueError):
            Bispectrum(self.lc, method="cumulant", window="not-a-window")
        with pytest.raises(ValueError):
            Bispectrum(self.lc, method="cumulant", maxlag=10 * self.n)

    def test_from_eventlist(self):
        ev = EventList(np.sort(rng.uniform(0, 100, 5000)), gti=[[0, 100]])
        bs = Bispectrum(ev, dt=0.1, method="cumulant", maxlag=20)
        assert bs.method == "cumulant"
        assert bs.cum3 is not None

    def test_averaged_cumulant(self):
        abs_c = AveragedBispectrum(self.lc, segment_size=2.0, method="cumulant", maxlag=15)
        assert abs_c.method == "cumulant"
        assert abs_c.m > 1
        assert abs_c.cum3.shape == (31, 31)
        assert abs_c.bicoherence is None

    def test_plot_cum3(self):
        bs = Bispectrum(self.lc, method="cumulant", maxlag=20)
        ax = bs.plot_cum3()
        assert ax is not None

    def test_plot_cum3_requires_cumulant(self):
        bs = Bispectrum(self.lc)  # fourier
        with pytest.raises(ValueError):
            bs.plot_cum3()

    def test_default_maxlag(self):
        bs = Bispectrum(self.lc, method="cumulant")  # maxlag defaults to n // 2
        assert bs.maxlag == self.n // 2

    def test_window_must_be_string(self):
        with pytest.raises(TypeError):
            Bispectrum(self.lc, method="cumulant", maxlag=20, window=5)

    def test_maxlag_must_be_integer(self):
        with pytest.raises(ValueError):
            Bispectrum(self.lc, method="cumulant", maxlag=1.5)

    def test_eventlist_without_dt_raises(self):
        ev = EventList(np.sort(rng.uniform(0, 100, 2000)), gti=[[0, 100]])
        with pytest.raises(ValueError):
            Bispectrum(ev, method="cumulant", maxlag=10, skip_checks=True)

    def test_bad_input_type_raises(self):
        with pytest.raises(TypeError):
            Bispectrum([self.lc], method="cumulant", maxlag=10)

    def test_plot_cum3_save(self, tmp_path):
        bs = Bispectrum(self.lc, method="cumulant", maxlag=20)
        fname = str(tmp_path / "cum3.png")
        bs.plot_cum3(save=True, filename=fname)
        assert os.path.exists(fname)

    def test_averaged_segment_longer_than_lc_raises(self):
        # segment longer than the light curve is rejected (no full segment)
        with pytest.raises((ValueError, AssertionError)):
            AveragedBispectrum(
                self.lc, segment_size=self.n * self.dt * 10, method="cumulant", maxlag=5
            )

    def test_averaged_cumulant_skips_zero_segments(self):
        # a partially-zero light curve exercises the all-zero segment skip
        counts = self.lc.counts.copy()
        counts[: counts.size // 2] = 0.0
        lc = Lightcurve(self.lc.time, counts, dt=self.dt, skip_checks=True)
        bs = AveragedBispectrum(lc, segment_size=2.0, method="cumulant", maxlag=5)
        assert bs.m >= 1

    def test_averaged_cumulant_all_zero_raises(self):
        lc = Lightcurve(self.lc.time, np.zeros(self.n), dt=self.dt, skip_checks=True)
        with pytest.raises(ValueError):
            AveragedBispectrum(lc, segment_size=2.0, method="cumulant", maxlag=5)


class TestBispectrumNormalization(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.01
        n = 256 * 20
        cls.lc = Lightcurve(
            np.arange(n) * cls.dt, rng.poisson(60, n).astype(float), dt=cls.dt, skip_checks=True
        )
        cls.segment_size = 2.56

    def test_default_is_kim_powers(self):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        assert bs.bicoherence_norm == "kim_powers"

    @pytest.mark.parametrize("norm", ["kim_powers", "sigl_chamoun", "hagihira"])
    def test_norm_parameter(self, norm):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size, bicoherence_norm=norm)
        assert bs.bicoherence_norm == norm
        valid = bs.valid
        assert np.all(bs.bicoherence[valid] >= 0)
        assert np.all(bs.bicoherence[valid] <= 1)

    @pytest.mark.parametrize("norm", ["kim_powers", "sigl_chamoun", "hagihira"])
    def test_recompute_matches_direct(self, norm):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        direct = AveragedBispectrum(self.lc, segment_size=self.segment_size, bicoherence_norm=norm)
        recomputed = bs.recompute_bicoherence(norm)
        valid = bs.valid
        assert np.allclose(recomputed[valid], direct.bicoherence[valid], atol=1e-12)

    def test_recompute_inplace(self):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        bs.recompute_bicoherence("hagihira", inplace=True)
        assert bs.bicoherence_norm == "hagihira"

    def test_recompute_bad_norm(self):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        with pytest.raises(ValueError):
            bs.recompute_bicoherence("nope")

    def test_recompute_on_empty_raises(self):
        with pytest.raises(ValueError):
            Bispectrum().recompute_bicoherence("kim_powers")

    def test_bias_subtract_removes_1_over_m(self):
        # For the squared (kim_powers) norm the debiased value is raw - 1/M.
        raw = AveragedBispectrum(
            self.lc, segment_size=self.segment_size, bicoherence_norm="kim_powers"
        )
        deb = AveragedBispectrum(
            self.lc,
            segment_size=self.segment_size,
            bicoherence_norm="kim_powers",
            bias_subtract=True,
        )
        assert deb.bias_subtract is True
        assert raw.bias_subtract is False
        valid = raw.valid
        # Exactly raw - 1/M, with NO clipping (below-floor values may be negative).
        expected = raw.bicoherence[valid] - 1.0 / raw.m
        assert np.allclose(deb.bicoherence[valid], expected, atol=1e-12)
        # debiasing lowers the (noise-floor) bicoherence on average
        assert np.nanmean(deb.bicoherence) <= np.nanmean(raw.bicoherence)

    def test_bias_subtract_not_clipped_below_zero(self):
        # The debiased squared bicoherence must be allowed to go negative, so that
        # below-noise-floor / suspect statistics stay visible rather than hidden.
        deb = AveragedBispectrum(
            self.lc,
            segment_size=self.segment_size,
            bicoherence_norm="kim_powers",
            bias_subtract=True,
        )
        assert np.nanmin(deb.bicoherence) < 0.0

    def test_bias_subtract_recompute_matches_constructor(self):
        raw = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        deb = AveragedBispectrum(self.lc, segment_size=self.segment_size, bias_subtract=True)
        post = raw.recompute_bicoherence(bias_subtract=True)
        assert np.allclose(np.nan_to_num(post), np.nan_to_num(deb.bicoherence), atol=1e-12)

    def test_bias_subtract_sigl_finite_and_signed(self):
        bs = AveragedBispectrum(
            self.lc, segment_size=self.segment_size, bicoherence_norm="sigl_chamoun"
        )
        deb = bs.recompute_bicoherence(norm="sigl_chamoun", bias_subtract=True)
        valid = bs.valid
        # signed root: finite, <= 1, and may go negative (not clipped)
        assert np.all(np.isfinite(deb[valid]))
        assert np.all(deb[valid] <= 1.0)

    def test_bias_subtract_hagihira_raises(self):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        with pytest.raises(ValueError):
            bs.recompute_bicoherence(norm="hagihira", bias_subtract=True)

    def test_bicoherence_from_sums_bias_requires_nseg(self):
        a = np.ones((3, 3))
        with pytest.raises(ValueError):
            bicoherence_from_sums("kim_powers", a, a, a, a, bias_subtract=True)

    def test_bias_subtract_cross(self):
        xbs = AveragedCrossBispectrum(
            self.lc, self.lc, self.lc, segment_size=self.segment_size, bias_subtract=True
        )
        assert xbs.bias_subtract is True
        valid = xbs.valid
        # debiased values are finite and <= 1 (may be negative, so no >= 0 check)
        assert np.all(np.isfinite(xbs.bicoherence[valid]))
        assert np.all(xbs.bicoherence[valid] <= 1.0)


class TestBispectrumIO(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.1
        cls.n = 256
        lc = Lightcurve(
            np.arange(cls.n) * cls.dt,
            rng.poisson(40, cls.n).astype(float),
            dt=cls.dt,
            skip_checks=True,
        )
        cls.bs = AveragedBispectrum(lc, segment_size=5.0)

    def test_astropy_table_roundtrip(self):
        ts = self.bs.to_astropy_table()
        back = AveragedBispectrum.from_astropy_table(ts)
        assert np.allclose(back.freq, self.bs.freq)
        assert np.allclose(np.nan_to_num(back.bispec), np.nan_to_num(self.bs.bispec))
        assert np.allclose(np.nan_to_num(back.bicoherence), np.nan_to_num(self.bs.bicoherence))

    def test_recompute_survives_roundtrip(self):
        ts = self.bs.to_astropy_table()
        back = AveragedBispectrum.from_astropy_table(ts)
        valid = self.bs.valid
        assert np.allclose(
            back.recompute_bicoherence("sigl_chamoun")[valid],
            self.bs.recompute_bicoherence("sigl_chamoun")[valid],
            atol=1e-12,
        )

    @pytest.mark.parametrize("fmt", ["pickle"])
    def test_file_roundtrip(self, fmt):
        fname = f"test_bispec.{fmt}"
        try:
            self.bs.write(fname, fmt=fmt)
            back = AveragedBispectrum.read(fname, fmt=fmt)
            assert np.allclose(back.freq, self.bs.freq)
            assert np.allclose(np.nan_to_num(back.bispec), np.nan_to_num(self.bs.bispec))
        finally:
            if os.path.exists(fname):
                os.remove(fname)


class TestBispectrumPlots(object):
    @classmethod
    def setup_class(cls):
        lc = Lightcurve(
            np.arange(256) * 0.1, rng.poisson(40, 256).astype(float), dt=0.1, skip_checks=True
        )
        cls.bs = AveragedBispectrum(lc, segment_size=5.0)

    def teardown_method(self):
        clear_all_figs()

    def test_plot_mag(self):
        ax = self.bs.plot_mag()
        assert ax is not None

    def test_plot_phase(self):
        ax = self.bs.plot_phase()
        assert ax is not None

    def test_plot_bicoherence(self):
        ax = self.bs.plot_bicoherence()
        assert ax is not None

    def test_plot_bicoherence_log(self):
        ax = self.bs.plot_bicoherence(log=True)
        assert ax is not None
        assert "log10" in ax.get_title().lower()

    def test_plot_on_given_axis(self):
        _, ax = plt.subplots()
        out = self.bs.plot_mag(ax=ax)
        assert out is ax

    def test_plot_save(self, tmp_path):
        fname = str(tmp_path / "mag.png")
        self.bs.plot_mag(save=True, filename=fname)
        assert os.path.exists(fname)

    def test_plot_empty_raises(self):
        with pytest.raises(ValueError):
            Bispectrum().plot_mag()


class TestBispectrumJellyfish(object):
    @classmethod
    def setup_class(cls):
        # Small grid so the per-segment (2D) store stays light.
        rng_local = np.random.RandomState(11)
        dt, seg, nseg = 0.02, 2.0, 60
        nb = int(seg / dt)  # 100 bins -> nf ~ 49
        t = np.arange(nb) * dt
        cls.f0 = 5.0
        chunks = []
        for _ in range(nseg):
            p = rng_local.uniform(0, 2 * np.pi)
            # fundamental at f0 plus a phase-locked harmonic at 2*f0
            s = np.cos(2 * np.pi * cls.f0 * t + p) + np.cos(2 * np.pi * 2 * cls.f0 * t + 2 * p)
            chunks.append(rng_local.poisson(np.clip(100 * (1 + 0.3 * s), 0, None) * dt))
        c = np.concatenate(chunks).astype(float)
        cls.lc = Lightcurve(np.arange(c.size) * dt, c, dt=dt, skip_checks=True)
        cls.seg = seg
        cls.bs = AveragedBispectrum(cls.lc, segment_size=seg, save_all=True)
        cls.bs_diag = AveragedBispectrum(cls.lc, segment_size=seg, save_diagonal=True)

    def teardown_method(self):
        clear_all_figs()

    def test_requires_per_segment_store(self):
        bs = AveragedBispectrum(self.lc, segment_size=self.seg)  # neither store
        with pytest.raises(ValueError):
            bs.plot_jellyfish()

    def test_save_diagonal_is_light(self):
        nf = self.bs.freq.size
        full = np.asarray(self.bs.bispec_all)
        diag = np.asarray(self.bs_diag.bispec_diagonal)
        assert full.shape == (self.bs.m, nf, nf)
        assert diag.shape == (self.bs_diag.m, nf)
        # save_diagonal keeps no full cube
        assert self.bs_diag.bispec_all is None
        assert diag.nbytes < full.nbytes

    def test_save_diagonal_matches_full_diagonal(self):
        full = np.asarray(self.bs.bispec_all)
        diag = np.asarray(self.bs_diag.bispec_diagonal)
        nf = self.bs.freq.size
        assert np.allclose(diag, full[:, np.arange(nf), np.arange(nf)])

    def test_jellyfish_from_diagonal_matches_full(self):
        ax_full = self.bs.plot_jellyfish(f0=self.f0)
        ax_diag = self.bs_diag.plot_jellyfish(f0=self.f0)
        lines_full = ax_full.get_lines()
        lines_diag = ax_diag.get_lines()
        assert len(lines_full) == len(lines_diag)
        for lf, ld in zip(lines_full, lines_diag):
            assert np.allclose(lf.get_xdata(), ld.get_xdata())
            assert np.allclose(lf.get_ydata(), ld.get_ydata())

    def test_diagonal_store_plots(self):
        ax = self.bs_diag.plot_jellyfish(f0=self.f0)
        assert ax is not None

    def test_empty_requires_save_all(self):
        with pytest.raises(ValueError):
            Bispectrum().plot_jellyfish()

    def test_returns_axes(self):
        ax = self.bs.plot_jellyfish()
        assert ax is not None

    def test_highlight_f0(self):
        ax = self.bs.plot_jellyfish(f0=self.f0)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "QPO fundamental" in labels
        assert "subharmonic" in labels

    def test_freqs_subset(self):
        ax = self.bs.plot_jellyfish(freqs=[self.f0])
        assert ax is not None

    def test_no_resolved_diagonal_freqs_raises(self):
        # frequencies beyond the resolved diagonal leave nothing to draw
        with pytest.raises(ValueError):
            self.bs.plot_jellyfish(freqs=[10 * self.bs.freq[-1]])

    def test_endpoint_radius_is_bicoherence(self):
        # The drawn fundamental path endpoint amplitude equals the (sigl_chamoun)
        # bicoherence, and its angle is the biphase.
        j = np.argmin(np.abs(self.bs.freq - self.f0))
        subbs = np.asarray(self.bs.bispec_all)
        norm = np.sqrt(self.bs._bicoh_denom1[j, j] * self.bs._bicoh_denom2[j, j])
        endpoint = np.sum(subbs[:, j, j]) / norm
        assert np.isclose(
            np.abs(endpoint), self.bs.recompute_bicoherence("sigl_chamoun")[j, j], atol=1e-9
        )
        assert np.isclose(np.angle(endpoint), self.bs.biphase[j, j], atol=1e-9)

    def test_plot_on_given_axis(self):
        _, ax = plt.subplots()
        out = self.bs.plot_jellyfish(ax=ax)
        assert out is ax

    def test_save(self, tmp_path):
        fname = str(tmp_path / "jelly.png")
        self.bs.plot_jellyfish(save=True, filename=fname)
        assert os.path.exists(fname)


class TestBispectrumPoisson(object):
    @classmethod
    def setup_class(cls):
        rng_local = np.random.RandomState(55)
        cls.dt = 0.02
        cls.n_bin = 64
        cls.segments = [rng_local.poisson(30, cls.n_bin).astype(float) for _ in range(80)]

    def test_matches_manual_wirnitzer_correction(self):
        freq, i1, i2, i3, valid = _bispectrum_frequency_grid(self.n_bin, self.dt)
        acc = np.zeros((freq.size, freq.size), dtype=complex)
        for s in self.segments:
            ft = np.fft.fft(s)
            p = (ft * ft.conj()).real
            triple = ft[i1] * ft[i2] * np.conj(ft[i3])
            triple = triple - (p[i1] + p[i2] + p[i3] - 2.0 * s.sum())
            acc += triple
        acc /= len(self.segments)
        res = avg_bispectrum_from_iterable(
            iter(self.segments), self.dt, poisson_subtract=True, silent=True
        )
        assert np.allclose(res.meta["bispec"][valid], acc[valid], atol=1e-9)

    def test_flag_in_meta(self):
        res = avg_bispectrum_from_iterable(
            iter(self.segments), self.dt, poisson_subtract=True, silent=True
        )
        assert res.meta["poisson_subtract"] is True
        res2 = avg_bispectrum_from_iterable(iter(self.segments), self.dt, silent=True)
        assert res2.meta["poisson_subtract"] is False

    def test_removes_bias_on_pure_noise(self):
        # Pure Poisson noise: raw Re(B) is biased upward by ~ N, corrected ~ 0.
        raw = avg_bispectrum_from_iterable(iter(self.segments), self.dt, silent=True)
        cor = avg_bispectrum_from_iterable(
            iter(self.segments), self.dt, poisson_subtract=True, silent=True
        )
        v = raw.meta["valid"]
        n = raw.meta["nphots"]
        assert np.nanmean(raw.meta["bispec"][v].real) > 0.5 * n
        assert abs(np.nanmean(cor.meta["bispec"][v].real)) < 0.1 * n

    def test_class_flag_propagates(self):
        rng_local = np.random.RandomState(7)
        c = rng_local.poisson(20, 64 * 40).astype(float)
        lc = Lightcurve(np.arange(c.size) * 0.02, c, dt=0.02, skip_checks=True)
        bs = AveragedBispectrum(lc, segment_size=64 * 0.02, poisson_subtract=True)
        assert bs.poisson_subtracted is True
        assert AveragedBispectrum(lc, segment_size=64 * 0.02).poisson_subtracted is False

    def test_empty_flag_default(self):
        assert Bispectrum().poisson_subtracted is False


def _cross_bands(rng_local, n_seg, n_bin, dt, f1, f2, couple=True):
    """Two simultaneous bands: band A carries f1, f2; band B carries f1+f2,
    phase-locked to A (couple=True) or with an independent phase."""
    t = np.arange(n_bin) * dt
    ca, cb = [], []
    for _ in range(n_seg):
        p1, p2 = rng_local.uniform(0, 2 * np.pi, size=2)
        a = 0.5 * np.cos(2 * np.pi * f1 * t + p1) + 0.5 * np.cos(2 * np.pi * f2 * t + p2)
        pb = (p1 + p2) if couple else rng_local.uniform(0, 2 * np.pi)
        b = 0.5 * np.cos(2 * np.pi * (f1 + f2) * t + pb)
        ca.append(rng_local.poisson(np.clip(500 * (1 + a), 0, None) * dt))
        cb.append(rng_local.poisson(np.clip(500 * (1 + b), 0, None) * dt))
    a = np.concatenate(ca).astype(float)
    b = np.concatenate(cb).astype(float)
    lca = Lightcurve(np.arange(a.size) * dt, a, dt=dt, skip_checks=True)
    lcb = Lightcurve(np.arange(b.size) * dt, b, dt=dt, skip_checks=True)
    return lca, lcb


class TestCrossBispectrum(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.05
        cls.n = 512
        cls.segment_size = 5.0
        cls.time = np.arange(cls.n) * cls.dt
        rng2 = np.random.RandomState(42)
        cls.lc1 = Lightcurve(
            cls.time, rng2.poisson(50, cls.n).astype(float), dt=cls.dt, skip_checks=True
        )
        cls.lc2 = Lightcurve(
            cls.time, rng2.poisson(50, cls.n).astype(float), dt=cls.dt, skip_checks=True
        )
        cls.lc3 = Lightcurve(
            cls.time, rng2.poisson(50, cls.n).astype(float), dt=cls.dt, skip_checks=True
        )
        cls.xbs = CrossBispectrum(cls.lc1, cls.lc2, cls.lc3)
        cls.gti = [[0, cls.n * cls.dt]]
        rng_ev = np.random.RandomState(99)
        cls.evs = [
            EventList(np.sort(rng_ev.uniform(0, cls.n * cls.dt, 4000)), gti=cls.gti)
            for _ in range(3)
        ]

    def teardown_method(self):
        clear_all_figs()

    def test_type_and_hierarchy(self):
        assert self.xbs.type == "crossbispectrum"
        assert isinstance(Bispectrum(), CrossBispectrum)
        assert isinstance(AveragedBispectrum(), AveragedCrossBispectrum)

    def test_signed_grid(self):
        # cross-bispectrum uses the full signed frequency plane
        assert np.any(self.xbs.freq < 0)
        assert np.any(self.xbs.freq > 0)
        nf = self.xbs.freq.size
        assert self.xbs.bispec.shape == (nf, nf)

    def test_empty(self):
        xbs = CrossBispectrum()
        assert xbs.freq is None
        assert xbs.type == "crossbispectrum"

    def test_nphots_per_channel(self):
        assert self.xbs.nphots1 is not None
        assert self.xbs.nphots3 is not None

    def test_single_arg_defaults_to_auto(self):
        # CrossBispectrum(lc) should set data2 = data3 = data1
        xbs = CrossBispectrum(self.lc1)
        assert xbs.bispec is not None

    def test_mismatched_kinds_raise(self):
        ev = EventList(np.sort(np.random.uniform(0, 10, 50)), gti=[[0, 10]])
        with pytest.raises((ValueError, TypeError)):
            CrossBispectrum(self.lc1, ev, self.lc3, dt=self.dt)

    def test_mismatched_time_bins_raise(self):
        other = Lightcurve(
            np.arange(self.n) * self.dt * 2, self.lc1.counts, dt=self.dt * 2, skip_checks=True
        )
        with pytest.raises(ValueError):
            CrossBispectrum(self.lc1, self.lc2, other)

    def test_reduces_to_auto_when_identical(self):
        auto = Bispectrum(self.lc1)
        cross = CrossBispectrum(self.lc1, self.lc1, self.lc1)
        i1a, i2a = 3, 5
        f1, f2 = auto.freq[i1a], auto.freq[i2a]
        c1 = int(np.argmin(np.abs(cross.freq - f1)))
        c2 = int(np.argmin(np.abs(cross.freq - f2)))
        assert np.isclose(auto.bispec[i1a, i2a], cross.bispec[c1, c2])

    def test_from_lightcurve(self):
        xbs = CrossBispectrum.from_lightcurve(self.lc1, self.lc2, self.lc3)
        assert isinstance(xbs, CrossBispectrum)

    def test_from_events(self):
        rng2 = np.random.RandomState(1)
        evs = [EventList(np.sort(rng2.uniform(0, 100, 3000)), gti=[[0, 100]]) for _ in range(3)]
        xbs = CrossBispectrum.from_events(*evs, dt=0.1)
        assert isinstance(xbs, CrossBispectrum)

    def test_averaged(self):
        xbs = AveragedCrossBispectrum(self.lc1, self.lc2, self.lc3, segment_size=self.segment_size)
        assert isinstance(xbs, AveragedCrossBispectrum)
        assert xbs.m > 1

    def test_averaged_needs_segment(self):
        with pytest.raises(ValueError):
            AveragedCrossBispectrum(self.lc1, self.lc2, self.lc3)

    @pytest.mark.parametrize("norm", BICOHERENCE_NORMS)
    def test_recompute_bicoherence(self, norm):
        b = self.xbs.recompute_bicoherence(norm)
        valid = self.xbs.valid
        # Bounded to [0, 1] by construction; the output is not clipped, so allow a
        # numerical tolerance at the upper edge (a single segment gives b == 1).
        assert np.all(b[valid] >= 0)
        assert np.all(b[valid] <= 1 + 1e-10)

    def test_detects_cross_coupling(self):
        rng_local = np.random.RandomState(7)
        dt, n_bin, n_seg = 0.01, 128, 400
        f1, f2 = 5.0, 12.0
        lca_c, lcb_c = _cross_bands(rng_local, n_seg, n_bin, dt, f1, f2, couple=True)
        lca_u, lcb_u = _cross_bands(rng_local, n_seg, n_bin, dt, f1, f2, couple=False)
        xc = AveragedCrossBispectrum(
            lca_c, lca_c, lcb_c, segment_size=n_bin * dt, bicoherence_norm="sigl_chamoun"
        )
        xu = AveragedCrossBispectrum(
            lca_u, lca_u, lcb_u, segment_size=n_bin * dt, bicoherence_norm="sigl_chamoun"
        )
        i1 = int(np.argmin(np.abs(xc.freq - f1)))
        i2 = int(np.argmin(np.abs(xc.freq - f2)))
        assert xc.bicoherence[i1, i2] > 0.7
        assert xu.bicoherence[i1, i2] < 0.3

    def test_index_convention_broken_swap_symmetry(self):
        # Three DISTINCT bands: X only at f1, Y only at f2, Z only at f1+f2.
        # Coupling appears at (f1, f2) but not at the swapped (f2, f1); this also
        # documents that bispec[i, j] is indexed (f1=freq[i], f2=freq[j]).
        rng_local = np.random.RandomState(3)
        dt, n_bin, n_seg = 0.01, 128, 400
        f1, f2 = 5.0, 12.0
        t = np.arange(n_bin) * dt
        cx, cy, cz = [], [], []
        for _ in range(n_seg):
            p1, p2 = rng_local.uniform(0, 2 * np.pi, size=2)
            cx.append(
                rng_local.poisson(
                    np.clip(500 * (1 + 0.5 * np.cos(2 * np.pi * f1 * t + p1)), 0, None) * dt
                )
            )
            cy.append(
                rng_local.poisson(
                    np.clip(500 * (1 + 0.5 * np.cos(2 * np.pi * f2 * t + p2)), 0, None) * dt
                )
            )
            cz.append(
                rng_local.poisson(
                    np.clip(500 * (1 + 0.5 * np.cos(2 * np.pi * (f1 + f2) * t + p1 + p2)), 0, None)
                    * dt
                )
            )

        def lc(ch):
            c = np.concatenate(ch).astype(float)
            return Lightcurve(np.arange(c.size) * dt, c, dt=dt, skip_checks=True)

        xbs = AveragedCrossBispectrum(
            lc(cx), lc(cy), lc(cz), segment_size=n_bin * dt, bicoherence_norm="sigl_chamoun"
        )
        i1 = int(np.argmin(np.abs(xbs.freq - f1)))
        i2 = int(np.argmin(np.abs(xbs.freq - f2)))
        assert xbs.bicoherence[i1, i2] > 0.7  # (f1, f2): coupled
        assert xbs.bicoherence[i2, i1] < 0.3  # (f2, f1): swap is absent
        # reality symmetry: B(-f1, -f2) = conj(B(f1, f2)) -> same bicoherence
        mi1 = int(np.argmin(np.abs(xbs.freq + f1)))
        mi2 = int(np.argmin(np.abs(xbs.freq + f2)))
        assert np.isclose(xbs.bicoherence[i1, i2], xbs.bicoherence[mi1, mi2], atol=1e-6)

    def test_astropy_table_roundtrip(self):
        xbs = AveragedCrossBispectrum(self.lc1, self.lc2, self.lc3, segment_size=self.segment_size)
        ts = xbs.to_astropy_table()
        back = AveragedCrossBispectrum.from_astropy_table(ts)
        assert np.allclose(back.freq, xbs.freq)
        assert np.allclose(np.nan_to_num(back.bispec), np.nan_to_num(xbs.bispec))

    def test_cross_jellyfish(self):
        xbs = AveragedCrossBispectrum(
            self.lc1, self.lc2, self.lc3, segment_size=self.segment_size, save_diagonal=True
        )
        ax = xbs.plot_jellyfish()
        assert ax is not None
        plt.close("all")

    def test_poisson_only_with_overlap(self):
        # poisson_subtract has no effect for independent channels
        base = AveragedCrossBispectrum(self.lc1, self.lc2, self.lc3, segment_size=self.segment_size)
        indep = AveragedCrossBispectrum(
            self.lc1, self.lc2, self.lc3, segment_size=self.segment_size, poisson_subtract=True
        )
        assert indep.poisson_subtracted is False
        valid = base.valid
        assert np.allclose(np.nan_to_num(base.bispec[valid]), np.nan_to_num(indep.bispec[valid]))
        overlap = AveragedCrossBispectrum(
            self.lc1,
            self.lc1,
            self.lc1,
            segment_size=self.segment_size,
            poisson_subtract=True,
            channels_overlap=True,
        )
        assert overlap.poisson_subtracted is True

    # --- input-dispatch paths and factories -------------------------------

    def test_init_from_events_dispatch(self):
        xbs = CrossBispectrum(*self.evs, dt=0.1)
        assert xbs.bispec is not None

    def test_init_from_lc_iterables_dispatch(self):
        xbs = CrossBispectrum([self.lc1], [self.lc2], [self.lc3])
        assert xbs.bispec is not None

    def test_initial_checks_none_returns_false(self):
        assert CrossBispectrum().initial_checks(data1=None) is False

    def test_initial_checks_bad_type(self):
        with pytest.raises(TypeError):
            CrossBispectrum(1, 1, 1)

    def test_segment_size_too_small(self):
        with pytest.raises(ValueError):
            AveragedCrossBispectrum(self.lc1, self.lc2, self.lc3, segment_size=self.dt, dt=self.dt)

    def test_from_time_array(self):
        xbs = CrossBispectrum.from_time_array(
            self.evs[0].time, self.evs[1].time, self.evs[2].time, 0.1, gti=self.gti
        )
        assert isinstance(xbs, CrossBispectrum)

    def test_averaged_from_lightcurve_static(self):
        xbs = AveragedCrossBispectrum.from_lightcurve(
            self.lc1, self.lc2, self.lc3, self.segment_size
        )
        assert xbs.m > 1

    def test_averaged_from_events_static(self):
        xbs = AveragedCrossBispectrum.from_events(*self.evs, 0.1, self.segment_size)
        assert isinstance(xbs, AveragedCrossBispectrum)

    def test_from_stingray_timeseries(self):
        tss = [
            StingrayTimeseries(
                self.time, array_attrs={"flux": lc.counts}, dt=self.dt, skip_checks=True
            )
            for lc in (self.lc1, self.lc2, self.lc3)
        ]
        for ts in tss:
            ts.gti = np.asarray(self.gti)
        xbs = crossbispectrum_from_stingray_timeseries(*tss, "flux", segment_size=self.segment_size)
        assert xbs.m > 1

    def test_from_lc_iterable(self):
        xbs = crossbispectrum_from_lc_iterable(
            [self.lc1], [self.lc2], [self.lc3], self.dt, segment_size=self.segment_size
        )
        assert isinstance(xbs, AveragedCrossBispectrum)

    def test_events_without_dt_raises(self):
        with pytest.raises(ValueError):
            CrossBispectrum(*self.evs)

    def test_averaged_single_arg_defaults_to_auto(self):
        xbs = AveragedCrossBispectrum(self.lc1, segment_size=self.segment_size)
        assert xbs.m > 1

    def test_from_lc_iterable_with_gti(self):
        xbs = crossbispectrum_from_lc_iterable(
            [self.lc1],
            [self.lc2],
            [self.lc3],
            self.dt,
            segment_size=self.segment_size,
            gti=[[0, self.n * self.dt]],
        )
        assert isinstance(xbs, AveragedCrossBispectrum)

    def test_from_lc_iterable_arrays(self):
        n_bin = int(self.segment_size / self.dt)
        seg = [self.lc1.counts[:n_bin], self.lc1.counts[n_bin : 2 * n_bin]]
        xbs = crossbispectrum_from_lc_iterable(
            list(seg), list(seg), list(seg), self.dt, segment_size=self.segment_size
        )
        assert xbs.m == 2

    def test_from_lc_iterable_bad_input(self):
        with pytest.raises(TypeError):
            crossbispectrum_from_lc_iterable([1], [1], [1], self.dt, segment_size=self.segment_size)

    def test_save_all_cross(self):
        xbs = AveragedCrossBispectrum(
            self.lc1, self.lc2, self.lc3, segment_size=self.segment_size, save_all=True
        )
        assert xbs.bispec_all is not None
        assert len(xbs.bispec_all) == xbs.m

    def test_all_zero_channels_raise(self):
        zero = Lightcurve(self.time, np.zeros(self.n), dt=self.dt, skip_checks=True)
        with pytest.raises(ValueError):
            AveragedCrossBispectrum(zero, zero, zero, segment_size=self.segment_size)

    def test_estimator_accepts_error_tuples(self):
        # the estimator unwraps (flux, error) tuples yielded per segment
        rng2 = np.random.RandomState(1)
        n_bin = 128

        def segs():
            return [(rng2.poisson(50, n_bin).astype(float), np.ones(n_bin)) for _ in range(20)]

        res = avg_cross_bispectrum_from_iterables(segs(), segs(), segs(), 0.01)
        assert res is not None


def _blinking_diagonal_lc(rng_local, n_blocks, seg_per_block, n_bin, dt, nu, on):
    """Auto light curve with harmonic (nu -> 2 nu) coupling switched per block.

    ``on`` is a per-block boolean sequence; the diagonal bicoherence at ``nu``
    should be high in the blocks where it is True and near the noise floor
    elsewhere.
    """
    t = np.arange(n_bin) * dt
    flux = []
    for b in range(n_blocks):
        for _ in range(seg_per_block):
            pa = rng_local.uniform(0, 2 * np.pi)
            s = 0.5 * np.cos(2 * np.pi * nu * t + pa)
            phase2 = 2 * pa if on[b] else rng_local.uniform(0, 2 * np.pi)
            s += 0.5 * np.cos(2 * np.pi * 2 * nu * t + phase2)
            flux.append(rng_local.poisson(np.clip(500 * (1 + s), 0, None) * dt))
    y = np.concatenate(flux).astype(float)
    return Lightcurve(np.arange(y.size) * dt, y, dt=dt, skip_checks=True)


class TestDynamicalBispectrum(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.01
        cls.segment_size = 2.0
        cls.bin_size = 80.0
        cls.n_bin = int(cls.segment_size / cls.dt)
        cls.nu = 4.0
        cls.n_blocks = 8
        cls.seg_per_block = 40
        rng_local = np.random.RandomState(11)
        # harmonic coupling ON only in the middle four blocks
        cls.on = [False, False, True, True, True, True, False, False]
        cls.lc = _blinking_diagonal_lc(
            rng_local, cls.n_blocks, cls.seg_per_block, cls.n_bin, cls.dt, cls.nu, cls.on
        )
        cls.db = DynamicalBispectrum(
            cls.lc,
            segment_size=cls.segment_size,
            bin_size=cls.bin_size,
            bicoherence_norm="sigl_chamoun",
        )
        cls.full = DynamicalBispectrum(
            cls.lc,
            segment_size=cls.segment_size,
            bin_size=cls.bin_size,
            store="full",
            bicoherence_norm="sigl_chamoun",
        )

    def teardown_method(self):
        clear_all_figs()

    def test_type_and_hierarchy(self):
        assert self.db.type == "bispectrum"
        assert isinstance(self.db, DynamicalCrossBispectrum)
        # the auto case is a special case of the cross case
        assert isinstance(DynamicalBispectrum(), DynamicalCrossBispectrum)

    def test_shapes_and_two_time_scales(self):
        assert self.db.time.size == self.n_blocks
        # diagonal store: (n_time, nf)
        assert self.db.dyn_bicoherence.shape == (self.n_blocks, self.db.freq.size)
        assert self.db.dt == self.bin_size
        assert self.db.m == self.seg_per_block
        assert np.isclose(self.db.df, 1.0 / self.segment_size)

    def test_empty(self):
        db = DynamicalBispectrum()
        assert db.freq is None
        assert db.dyn_bicoherence is None
        assert db.type == "bispectrum"

    def test_requires_both_time_scales(self):
        with pytest.raises(TypeError):
            DynamicalBispectrum(self.lc, segment_size=self.segment_size)
        with pytest.raises(TypeError):
            DynamicalBispectrum(self.lc, bin_size=self.bin_size)

    def test_bin_size_at_least_segment(self):
        with pytest.raises(ValueError):
            DynamicalBispectrum(self.lc, segment_size=2.0, bin_size=1.0)

    def test_invalid_store(self):
        with pytest.raises(ValueError):
            DynamicalBispectrum(self.lc, segment_size=2.0, bin_size=80.0, store="nope")

    def test_few_segments_per_bin_warns(self):
        with pytest.warns(UserWarning):
            DynamicalBispectrum(self.lc, segment_size=2.0, bin_size=6.0)

    def test_detects_blinking_diagonal(self):
        _, bic, _ = self.db.trace(self.nu, self.nu)
        on = np.array(self.on)
        assert np.all(bic[on] > 0.7)
        assert np.all(bic[~on] < 0.5)

    def test_trace_maximum_follows_the_harmonic(self):
        # in the coupled blocks the peak diagonal bicoherence sits at nu
        pos = self.db.trace_maximum(min_freq=1.0, max_freq=20.0)
        peak_freqs = self.db.freq[pos]
        on = np.array(self.on)
        assert np.allclose(peak_freqs[on], self.nu)

    def test_rebin_invariant(self):
        # collapsing every time bin into one must equal a single AveragedBispectrum
        merged = self.db.rebin_by_n_intervals(self.db.time.size)
        full = AveragedBispectrum(
            self.lc, segment_size=self.segment_size, bicoherence_norm="sigl_chamoun"
        )
        got = merged.dyn_bicoherence[0]
        ref = np.diag(full.bicoherence)
        assert np.allclose(np.nan_to_num(got), np.nan_to_num(ref), atol=1e-6)

    def test_rebin_time(self):
        rt = self.db.rebin_time(2 * self.bin_size)
        assert rt.time.size == self.n_blocks // 2
        assert rt.dt == 2 * self.bin_size
        assert rt.m == 2 * self.seg_per_block

    def test_rebin_time_must_increase(self):
        with pytest.raises(ValueError):
            self.db.rebin_time(self.bin_size / 2)

    def test_rebin_frequency(self):
        rf = self.db.rebin_frequency(2 * self.db.df)
        assert rf.freq.size < self.db.freq.size
        assert np.isclose(rf.df, 2 * self.db.df)

    def test_shift_and_add(self):
        # an always-coupled signal, so the aligned co-add stays strongly coherent
        rng_local = np.random.RandomState(5)
        lc = _blinking_diagonal_lc(
            rng_local,
            self.n_blocks,
            self.seg_per_block,
            self.n_bin,
            self.dt,
            self.nu,
            [True] * self.n_blocks,
        )
        db = DynamicalBispectrum(
            lc,
            segment_size=self.segment_size,
            bin_size=self.bin_size,
            bicoherence_norm="sigl_chamoun",
        )
        rel_freq, bic, _ = db.shift_and_add(np.full(db.time.size, self.nu))
        # the aligned coupling piles up at zero relative frequency
        assert np.isclose(rel_freq[np.nanargmax(bic)], 0.0)
        assert np.nanmax(bic) > 0.7

    def test_shift_and_add_bad_length(self):
        with pytest.raises(ValueError):
            self.db.shift_and_add([self.nu, self.nu])

    def test_plot_diagonal(self):
        ax = self.db.plot_diagonal()
        assert ax is not None

    def test_plot_diagonal_log(self):
        from matplotlib.colors import LogNorm

        ax = self.db.plot_diagonal(log=True)
        assert ax is not None
        # the mesh should carry a logarithmic norm
        assert any(isinstance(c.norm, LogNorm) for c in ax.collections)

    def test_diagonal_store_rejects_offdiagonal_ops(self):
        with pytest.raises(ValueError):
            self.db.trace(3.0, 7.0)
        with pytest.raises(ValueError):
            self.db.plot_slice(5.0)
        with pytest.raises(ValueError):
            self.db.plot_frame(self.db.time[0])

    def test_full_store_slice_and_frame(self):
        db = DynamicalBispectrum(
            self.lc,
            segment_size=self.segment_size,
            bin_size=self.bin_size,
            store="full",
            bicoherence_norm="sigl_chamoun",
        )
        assert db.dyn_bicoherence.shape == (self.n_blocks, db.freq.size, db.freq.size)
        assert db.plot_slice(self.nu) is not None
        assert db.plot_frame(db.time[0]) is not None
        # log colour scale on the full-store plots
        assert db.plot_slice(self.nu, log=True) is not None
        assert db.plot_frame(db.time[0], log=True) is not None
        assert db.plot_montage(log=True) is not None
        # trace works off-diagonal with the full store
        _, bic, _ = db.trace(self.nu, self.nu)
        assert np.all(bic[np.array(self.on)] > 0.7)

    def test_requires_data(self):
        with pytest.raises(TypeError):
            DynamicalBispectrum(segment_size=self.segment_size, bin_size=self.bin_size)

    def test_eventlist_needs_sample_time(self):
        ev = EventList(np.sort(rng.uniform(0, 200, 3000)), gti=[[0, 200]])
        with pytest.raises(ValueError):
            DynamicalBispectrum(ev, segment_size=self.segment_size, bin_size=self.bin_size)

    def test_segment_too_short(self):
        with pytest.raises(ValueError):
            DynamicalBispectrum(self.lc, segment_size=self.dt, bin_size=self.bin_size)

    def test_explicit_gti(self):
        db = DynamicalBispectrum(
            self.lc,
            segment_size=self.segment_size,
            bin_size=self.bin_size,
            gti=[[0, self.n_blocks * self.bin_size]],
            bicoherence_norm="sigl_chamoun",
        )
        assert db.time.size >= 1

    def test_plot_diagonal_empty_raises(self):
        with pytest.raises(ValueError):
            DynamicalBispectrum().plot_diagonal()

    def test_plot_trace(self):
        axes = self.db.plot_trace(self.nu, self.nu)
        assert len(axes) == 2

    def test_trace_maximum_defaults(self):
        pos = self.db.trace_maximum()
        assert pos.size == self.db.time.size

    def test_trace_maximum_full_store(self):
        pos = self.full.trace_maximum(min_freq=1, max_freq=20)
        assert pos.size == self.full.time.size

    def test_rebin_by_n_intervals_identity(self):
        assert self.db.rebin_by_n_intervals(1).time.size == self.db.time.size

    def test_rebin_by_n_intervals_noninteger_warns(self):
        with pytest.warns(UserWarning):
            self.db.rebin_by_n_intervals(2.0)

    def test_rebin_by_n_intervals_bad_n(self):
        with pytest.raises(ValueError):
            self.db.rebin_by_n_intervals(0)

    def test_rebin_frequency_identity(self):
        assert self.db.rebin_frequency(self.db.df).freq.size == self.db.freq.size

    def test_rebin_frequency_must_increase(self):
        with pytest.raises(ValueError):
            self.db.rebin_frequency(self.db.df / 2)

    def test_rebin_frequency_full_store_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.full.rebin_frequency(2 * self.full.df)

    def test_plot_montage_times_and_diagonal_guard(self):
        axes = self.full.plot_montage(times=[self.full.time[0], self.full.time[-1]])
        assert axes is not None
        with pytest.raises(ValueError):
            self.db.plot_montage()  # diagonal store cannot montage

    def test_single_time_bin_plot(self):
        # bin_size spanning the whole observation -> a single time bin, which
        # exercises the size-1 branch of the pcolormesh edge helper.
        total = self.n_blocks * self.bin_size
        db = DynamicalBispectrum(
            self.lc,
            segment_size=self.segment_size,
            bin_size=total,
            bicoherence_norm="sigl_chamoun",
        )
        assert db.time.size == 1
        assert db.plot_diagonal() is not None

    def test_skips_unusable_bins(self):
        # zero out the first bin -> that bin has no usable segments and is dropped
        counts = self.lc.counts.copy()
        counts[: int(self.bin_size / self.dt)] = 0.0
        lc = Lightcurve(self.lc.time, counts, dt=self.dt, skip_checks=True)
        with pytest.warns(UserWarning):
            db = DynamicalBispectrum(
                lc,
                segment_size=self.segment_size,
                bin_size=self.bin_size,
                bicoherence_norm="sigl_chamoun",
            )
        assert db.time.size == self.n_blocks - 1

    def test_all_bins_unusable_raises(self):
        n = self.n_blocks * self.seg_per_block * self.n_bin
        lc = Lightcurve(np.arange(n) * self.dt, np.zeros(n), dt=self.dt, skip_checks=True)
        with pytest.raises(ValueError):
            DynamicalBispectrum(lc, segment_size=self.segment_size, bin_size=self.bin_size)


class TestDynamicalCrossBispectrum(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.01
        cls.segment_size = 2.0
        cls.bin_size = 60.0
        cls.n_bin = int(cls.segment_size / cls.dt)
        cls.f1, cls.f2 = 5.0, 12.0
        rng_local = np.random.RandomState(7)
        n_seg = 8 * 30
        t = np.arange(cls.n_bin) * cls.dt
        cx, cy, cz = [], [], []
        for _ in range(n_seg):
            p1, p2 = rng_local.uniform(0, 2 * np.pi, size=2)
            cx.append(
                rng_local.poisson(
                    np.clip(500 * (1 + 0.5 * np.cos(2 * np.pi * cls.f1 * t + p1)), 0, None) * cls.dt
                )
            )
            cy.append(
                rng_local.poisson(
                    np.clip(500 * (1 + 0.5 * np.cos(2 * np.pi * cls.f2 * t + p2)), 0, None) * cls.dt
                )
            )
            cz.append(
                rng_local.poisson(
                    np.clip(
                        500 * (1 + 0.5 * np.cos(2 * np.pi * (cls.f1 + cls.f2) * t + p1 + p2)),
                        0,
                        None,
                    )
                    * cls.dt
                )
            )

        def lc(ch):
            c = np.concatenate(ch).astype(float)
            return Lightcurve(np.arange(c.size) * cls.dt, c, dt=cls.dt, skip_checks=True)

        cls.lcX, cls.lcY, cls.lcZ = lc(cx), lc(cy), lc(cz)

    def teardown_method(self):
        clear_all_figs()

    def test_signed_grid(self):
        db = DynamicalCrossBispectrum(
            self.lcX,
            self.lcY,
            self.lcZ,
            segment_size=self.segment_size,
            bin_size=self.bin_size,
            store="full",
            bicoherence_norm="sigl_chamoun",
        )
        assert db.type == "crossbispectrum"
        assert np.any(db.freq < 0) and np.any(db.freq > 0)
        # coupling at (f1, f2), absent at the swapped (f2, f1)
        _, bic, _ = db.trace(self.f1, self.f2)
        _, bsw, _ = db.trace(self.f2, self.f1)
        assert np.all(bic > 0.7)
        assert np.all(bsw < 0.5)

    def test_reduces_to_auto(self):
        # three identical inputs -> the diagonal matches the auto DynamicalBispectrum
        cross = DynamicalCrossBispectrum(
            self.lcX,
            self.lcX,
            self.lcX,
            segment_size=self.segment_size,
            bin_size=self.bin_size,
            bicoherence_norm="sigl_chamoun",
        )
        auto = DynamicalBispectrum(
            self.lcX,
            segment_size=self.segment_size,
            bin_size=self.bin_size,
            bicoherence_norm="sigl_chamoun",
        )
        # match on the positive diagonal frequencies present in both grids
        for nu in (self.f1, self.f2):
            ic = int(np.argmin(np.abs(cross.freq - nu)))
            ia = int(np.argmin(np.abs(auto.freq - nu)))
            assert np.allclose(
                np.nan_to_num(cross.dyn_bicoherence[:, ic]),
                np.nan_to_num(auto.dyn_bicoherence[:, ia]),
                atol=1e-6,
            )

    def test_mismatched_kinds_raise(self):
        ev = EventList(np.sort(rng.uniform(0, 100, 500)), gti=[[0, 100]])
        with pytest.raises(ValueError):
            DynamicalCrossBispectrum(
                self.lcX,
                ev,
                self.lcZ,
                sample_time=self.dt,
                segment_size=self.segment_size,
                bin_size=self.bin_size,
            )

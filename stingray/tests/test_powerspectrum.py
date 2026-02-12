import os
import numpy as np
import copy
import warnings
import importlib
import shutil
import tempfile

import pytest
import matplotlib.pyplot as plt
from astropy.io import fits
from stingray import Lightcurve
from stingray.events import EventList
from stingray.utils import HAS_NUMBA
from stingray import Powerspectrum, AveragedPowerspectrum, DynamicalPowerspectrum
from stingray.powerspectrum import GtiCorrPowerspectrum
from stingray.powerspectrum import powerspectrum_from_time_array
from astropy.modeling.models import Lorentz1D
from stingray.filters import filter_for_deadtime

_HAS_XARRAY = importlib.util.find_spec("xarray") is not None
_HAS_PANDAS = importlib.util.find_spec("pandas") is not None
_HAS_H5PY = importlib.util.find_spec("h5py") is not None
_HAS_FLX2XSP = (os.getenv("HEADAS") is not None) and (shutil.which("ftflx2xsp") is not None)


def clear_all_figs():
    fign = plt.get_fignums()
    for fig in fign:
        plt.close(fig)


rng = np.random.RandomState(20150907)

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


class TestAveragedPowerspectrumEvents(object):
    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 10.0
        cls.dt = 0.0001
        cls.segment_size = tend - tstart

        times = np.sort(rng.uniform(tstart, tend, 1000))
        gti = np.array([[tstart, tend]])

        cls.events = EventList(times, gti=gti)
        cls.events.fake_weights = np.ones_like(cls.events.time)

        cls.lc = cls.events
        cls.leahy_pds = AveragedPowerspectrum(
            cls.lc, segment_size=cls.segment_size, dt=cls.dt, norm="leahy", silent=True
        )

        cls.leahy_pds_sng = Powerspectrum(cls.lc, dt=cls.dt, norm="leahy")

    def test_save_all(self):
        cs = AveragedPowerspectrum(self.lc, dt=self.dt, segment_size=1, save_all=True)
        assert hasattr(cs, "cs_all")

    def test_plot_simple(self):
        clear_all_figs()
        self.leahy_pds.plot()
        assert plt.fignum_exists("crossspectrum")
        plt.close("crossspectrum")

    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs", "none"])
    def test_common_mean_gives_comparable_scatter(self, norm):
        acs = AveragedPowerspectrum(
            self.events,
            dt=self.dt,
            silent=True,
            segment_size=self.segment_size,
            norm=norm,
            use_common_mean=False,
        )
        acs_comm = AveragedPowerspectrum(
            self.events,
            dt=self.dt,
            silent=True,
            segment_size=self.segment_size,
            norm=norm,
            use_common_mean=True,
        )

        assert np.isclose(acs_comm.power.std(), acs.power.std(), rtol=0.1)

    @pytest.mark.parametrize("norm", ["frac", "leahy", "none", "abs"])
    def test_modulation_upper_limit(self, norm):
        val = 70
        unnorm_val = 70 * self.leahy_pds.nphots / 2
        pds = copy.deepcopy(self.leahy_pds)
        pds.power[25] = val
        pds.unnorm_power[25] = unnorm_val
        pds_norm = pds.to_norm(norm)
        assert np.isclose(pds_norm.modulation_upper_limit(2, 5, 0.99), 0.5412103, atol=1e-4)

    def test_type_change(self):
        pds = copy.deepcopy(self.leahy_pds)
        assert pds.type == "powerspectrum"
        pds.type = "astdfawerfsaf"
        assert pds.type == "astdfawerfsaf"

    def test_from_events_works_ps(self):
        pds_ev = Powerspectrum.from_events(self.events, dt=self.dt, norm="leahy")
        assert np.allclose(self.leahy_pds_sng.power, pds_ev.power)

    def test_from_events_works_aps(self):
        pds_ev = AveragedPowerspectrum.from_events(
            self.events, segment_size=self.segment_size, dt=self.dt, norm="leahy", silent=True
        )
        assert np.allclose(self.leahy_pds.power, pds_ev.power)

    def test_from_events_works_ps_dt(self):
        events = copy.deepcopy(self.events)
        # Use an incompatible dt
        events.dt = self.dt * np.pi / 3
        with pytest.warns(UserWarning, match="The input event list has a time resolution of"):
            pds_ev = Powerspectrum.from_events(events, dt=self.dt, norm="leahy", silent=True)
            assert np.isclose(pds_ev.dt, events.suggest_compatible_dt(self.dt))

    def test_from_events_works_aps_dt(self):
        events = copy.deepcopy(self.events)
        # Use an incompatible dt
        events.dt = self.dt * np.pi / 3
        with pytest.warns(UserWarning, match="The input event list has a time resolution of"):
            pds_ev = AveragedPowerspectrum.from_events(
                events, segment_size=self.segment_size, dt=self.dt, norm="leahy", silent=True
            )
            assert np.isclose(pds_ev.dt, events.suggest_compatible_dt(self.dt))

    def test_from_lc_iter_works(self):
        pds_ev = AveragedPowerspectrum.from_lc_iterable(
            self.events.to_lc_iter(self.dt, self.segment_size),
            segment_size=self.segment_size,
            dt=self.dt,
            norm="leahy",
            silent=True,
            gti=self.events.gti,
        )
        assert np.allclose(self.leahy_pds.power, pds_ev.power)

    @pytest.mark.parametrize("err_dist", ["poisson", "gauss"])
    @pytest.mark.parametrize("norm", ["leahy", "abs", "frac", "none"])
    def test_method_norm(self, norm, err_dist):
        lc = self.events.to_lc(dt=self.dt)
        if err_dist == "gauss":
            factor = 1 / lc.counts.max()
            lc.counts = lc.counts * factor
            lc.counts_err = lc.counts_err * factor
            lc.err_dist = "gauss"

        pds = AveragedPowerspectrum.from_lightcurve(
            lc, segment_size=self.segment_size, norm="leahy", silent=True
        )

        loc_pds = AveragedPowerspectrum.from_lightcurve(
            lc, segment_size=self.segment_size, norm=norm, silent=True
        )

        renorm_pds = pds.to_norm(norm)

        assert loc_pds.norm == renorm_pds.norm
        for attr in ["power", "unnorm_power", "power_err"]:
            loc = getattr(loc_pds, attr)
            renorm = getattr(renorm_pds, attr)
            assert np.allclose(loc, renorm, atol=0.5)
        for attr in ["norm", "nphots1", "df", "dt", "n", "m"]:
            loc = getattr(loc_pds, attr)
            renorm = getattr(renorm_pds, attr)
            assert loc == renorm

    def test_from_lc_iter_with_err_works(self):
        def iter_lc_with_errs(iter_lc):
            for lc in iter_lc:
                # In order for error bars to be considered,
                # err_dist has to be gauss.
                lc.err_dist = "gauss"
                lc._counts_err = np.zeros_like(lc.counts) + lc.counts.mean() ** 0.5
                yield lc

        lccs = AveragedPowerspectrum.from_lc_iterable(
            iter_lc_with_errs(self.events.to_lc_iter(self.dt, self.segment_size)),
            segment_size=self.segment_size,
            dt=self.dt,
            norm="leahy",
            silent=True,
        )
        power1 = lccs.power.real
        power2 = self.leahy_pds.power.real
        assert np.allclose(power1, power2, rtol=0.01)

    def test_from_lc_iter_with_err_ignored_with_wrong_err_dist(self):
        def iter_lc_with_errs(iter_lc):
            for lc in iter_lc:
                # Not supposed to have error bars
                lc.err_dist = "poisson"
                # use a completely wrong error bar, for fun
                lc._counts_err = np.zeros_like(lc.counts) + 14.2345425252462
                yield lc

        lccs = AveragedPowerspectrum.from_lc_iterable(
            iter_lc_with_errs(self.events.to_lc_iter(self.dt, self.segment_size)),
            segment_size=self.segment_size,
            dt=self.dt,
            norm="leahy",
            silent=True,
        )
        power1 = lccs.power.real
        power2 = self.leahy_pds.power.real
        assert np.allclose(power1, power2, rtol=0.01)

    def test_from_lc_iter_counts_only_works(self):
        def iter_lc_counts_only(iter_lc):
            for lc in iter_lc:
                yield lc.counts

        lccs = AveragedPowerspectrum.from_lc_iterable(
            iter_lc_counts_only(self.events.to_lc_iter(self.dt, self.segment_size)),
            segment_size=self.segment_size,
            dt=self.dt,
            norm="leahy",
            silent=True,
        )
        power1 = lccs.power.real
        power2 = self.leahy_pds.power.real
        assert np.allclose(power1, power2, rtol=0.01)

    @pytest.mark.slow
    def test_from_time_array_works_with_memmap(self):
        with fits.open(os.path.join(datadir, "monol_testA.evt"), memmap=True) as hdul:
            times = hdul[1].data["TIME"]

            gti = np.array([[hdul[2].data["START"][0], hdul[2].data["STOP"][0]]])

            _ = AveragedPowerspectrum.from_time_array(
                times, segment_size=128, dt=self.dt, gti=gti, norm="none", use_common_mean=False
            )

    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_from_lc_with_err_works(self, norm):
        lc = self.events.to_lc(self.dt)
        lc._counts_err = np.sqrt(lc.counts.mean()) + np.zeros_like(lc.counts)
        pds = AveragedPowerspectrum.from_lightcurve(
            lc, segment_size=self.segment_size, norm=norm, silent=True, gti=lc.gti
        )
        pds_ev = AveragedPowerspectrum.from_events(
            self.events,
            segment_size=self.segment_size,
            dt=self.dt,
            norm=norm,
            silent=True,
            gti=self.events.gti,
        )
        for attr in ["power", "freq", "m", "n", "nphots", "segment_size"]:
            assert np.allclose(getattr(pds, attr), getattr(pds_ev, attr))

    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_from_timeseries_with_err_works(self, norm):
        lc = self.events.to_binned_timeseries(self.dt)
        lc._counts_err = np.sqrt(lc.counts.mean()) + np.zeros_like(lc.counts)
        pds = AveragedPowerspectrum.from_stingray_timeseries(
            lc,
            "counts",
            "_counts_err",
            segment_size=self.segment_size,
            norm=norm,
            silent=True,
            gti=lc.gti,
        )
        pds_weight = AveragedPowerspectrum.from_stingray_timeseries(
            lc,
            "fake_weights",
            "_counts_err",
            segment_size=self.segment_size,
            norm=norm,
            silent=True,
        )
        pds_ev = AveragedPowerspectrum.from_events(
            self.events,
            segment_size=self.segment_size,
            dt=self.dt,
            norm=norm,
            silent=True,
            gti=self.events.gti,
        )
        for attr in ["power", "freq", "m", "n", "nphots", "segment_size"]:
            assert np.allclose(getattr(pds, attr), getattr(pds_ev, attr))
            assert np.allclose(getattr(pds_weight, attr), getattr(pds_ev, attr))

    def test_init_without_segment(self):
        with pytest.raises(ValueError):
            assert AveragedPowerspectrum(self.lc, dt=self.dt)

    def test_init_with_nonsense_segment(self):
        segment_size = "foo"
        with pytest.raises(TypeError):
            assert AveragedPowerspectrum(self.lc, segment_size, dt=self.dt)

    def test_init_with_none_segment(self):
        segment_size = None
        with pytest.raises(ValueError):
            assert AveragedPowerspectrum(self.lc, segment_size, dt=self.dt)

    def test_init_with_inf_segment(self):
        segment_size = np.inf
        with pytest.raises(ValueError):
            assert AveragedPowerspectrum(self.lc, segment_size, dt=self.dt)

    def test_init_with_nan_segment(self):
        segment_size = np.nan
        with pytest.raises(ValueError):
            assert AveragedPowerspectrum(self.lc, segment_size, dt=self.dt)

    @pytest.mark.parametrize("df", [2, 3, 5, 1.5, 1, 85])
    def test_rebin(self, df):
        """
        TODO: Not sure how to write tests for the rebin method!
        """

        aps = AveragedPowerspectrum(
            self.lc, segment_size=self.segment_size, norm="Leahy", dt=self.dt
        )
        bin_aps = aps.rebin(df)
        assert np.isclose(bin_aps.freq[1] - bin_aps.freq[0], bin_aps.df, atol=1e-4, rtol=1e-4)
        assert np.isclose(
            bin_aps.freq[0], (aps.freq[0] - aps.df * 0.5 + bin_aps.df * 0.5), atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize("f", [20, 30, 50, 15, 1, 850])
    def test_rebin_factor(self, f):
        """
        TODO: Not sure how to write tests for the rebin method!
        """

        aps = AveragedPowerspectrum(self.lc, segment_size=1, norm="Leahy", dt=self.dt)
        bin_aps = aps.rebin(f=f)
        assert np.isclose(bin_aps.freq[1] - bin_aps.freq[0], bin_aps.df, atol=1e-4, rtol=1e-4)
        assert np.isclose(
            bin_aps.freq[0], (aps.freq[0] - aps.df * 0.5 + bin_aps.df * 0.5), atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize("df", [0.01, 0.1])
    def test_rebin_log(self, df):
        # For now, just verify that it doesn't crash
        aps = AveragedPowerspectrum(self.lc, segment_size=1, norm="Leahy", dt=self.dt)
        bin_aps = aps.rebin_log(df)

    @pytest.mark.parametrize("use_common_mean", [True, False])
    def test_leahy_correct_for_multiple(self, use_common_mean):
        n = 10
        lc_all = []
        for i in range(n):
            time = np.arange(0.0, 10.0, 10.0 / 10000)
            counts = rng.poisson(1000, size=time.shape[0])
            lc = Lightcurve(time, counts)
            lc_all.append(lc)

        ps = AveragedPowerspectrum(lc_all, 1.0, norm="leahy", use_common_mean=use_common_mean)

        assert ps.m == 100
        assert np.isclose(np.mean(ps.power), 2.0, atol=1e-2, rtol=1e-2)
        assert np.isclose(np.std(ps.power), 2.0 / np.sqrt(ps.m), atol=0.1, rtol=0.1)

    @pytest.mark.skipif("not HAS_NUMBA")
    def test_deadtime_corr(self):
        tmax = 100.0
        segment_size = 1
        events = np.sort(np.random.uniform(0, tmax, 10000))
        events_dt = filter_for_deadtime(events, deadtime=0.0025)
        pds_dt = powerspectrum_from_time_array(
            events_dt,
            gti=[[0, tmax]],
            dt=0.001,
            segment_size=segment_size,
            save_all=True,
            norm="leahy",
        )

        pds = pds_dt.deadtime_correct(dead_time=0.0025, rate=np.size(events_dt) / tmax)
        assert np.isclose(np.mean(pds.power), 2, rtol=0.1)
        assert np.isclose(np.std(pds.power), 2 / np.sqrt(tmax / segment_size), rtol=0.1)


class TestGtiCorrPowerspectrum(object):
    @classmethod
    def setup_class(cls):
        """Set up a light curve and an event list with GTIs for testing GtiCorrPowerspectrum."""
        tstart = 0.0
        tend = 100.0
        dt = 0.01

        time = np.arange(tstart + 0.5 * dt, tend + 0.5 * dt, dt)

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate * dt

        poisson_counts = rng.poisson(mean_counts, size=time.shape[0])

        cls.lc = Lightcurve(time, counts=poisson_counts, gti=[[tstart, tend]], dt=dt)
        cls.events = EventList(
            np.sort(
                np.random.uniform(
                    tstart, tend, np.random.poisson(mean_count_rate * (tend - tstart))
                )
            ),
            gti=[[tstart, tend]],
        )

    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs", "none"])
    def test_gti_corr_ps(self, norm):
        """GtiCorrPowerspectrum results match Powerspectrum when GTIs are unimportant."""
        gcps = GtiCorrPowerspectrum(self.lc, norm=norm)
        ps = Powerspectrum(self.lc, norm=norm)
        for attr in [
            "freq",
            "power",
            "power_err",
            "unnorm_power",
            "unnorm_power_err",
            "df",
            "m",
            "n",
            "nphots",
        ]:
            assert np.array_equal(getattr(gcps, attr), getattr(ps, attr))

    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs", "none"])
    def test_gti_corr_ps_events(self, norm):
        """GtiCorrPowerspectrum results match Powerspectrum for event lists."""
        gcps = GtiCorrPowerspectrum(self.events, dt=0.01, norm=norm)
        ps = Powerspectrum(self.events, dt=0.01, norm=norm)
        for attr in [
            "freq",
            "power",
            "power_err",
            "unnorm_power",
            "unnorm_power_err",
            "df",
            "m",
            "n",
            "nphots",
        ]:
            assert np.array_equal(getattr(gcps, attr), getattr(ps, attr))

    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs", "none"])
    def test_gti_corr_ps_fill_different(self, norm):
        """Filling BTIs or not changes the power spectrum."""
        lc = copy.deepcopy(self.lc)
        lc.gti = [[0, 30], [35, 100]]  # Two GTIs, one gap
        gcps_fill = GtiCorrPowerspectrum(lc, norm=norm, fill_lc=True)
        gcps_nofill = GtiCorrPowerspectrum(lc, norm=norm, fill_lc=False)
        gcps_fill = gcps_fill.clean_gti_features()
        gcps_nofill = gcps_nofill.clean_gti_features()
        mean_gcps_fill = np.mean(gcps_fill.power)
        mean_gcps_nofill = np.mean(gcps_nofill.power)
        assert not np.allclose(mean_gcps_fill, mean_gcps_nofill, rtol=0.01)

    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs", "none"])
    def test_gti_corr_ps_fill(self, norm):
        """Filling BTIs and correcting the normalization adjusts the power spectrum."""
        lc = copy.deepcopy(self.lc)
        lc.gti = [[0, 30], [35, 100]]  # Two GTIs, one gap
        gcps = GtiCorrPowerspectrum(lc, norm=norm, fill_lc=True)
        gcps = gcps.clean_gti_features()
        ps = Powerspectrum(self.lc, norm=norm)
        mean_ps = np.mean(ps.power)
        mean_gcps = np.mean(gcps.power)
        assert np.isclose(mean_gcps, mean_ps, rtol=0.01)

    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs", "none"])
    def test_gti_corr_ps_fill_events(self, norm):
        """Filling BTIs and correcting the normalization adjusts the power spectrum."""
        lc = copy.deepcopy(self.events)
        lc.gti = [[0, 30], [35, 100]]  # Two GTIs, one gap
        gcps = GtiCorrPowerspectrum(lc, dt=0.01, norm=norm, fill_lc=True)
        gcps = gcps.clean_gti_features()
        ps = Powerspectrum(self.events, dt=0.01, norm=norm)
        mean_ps = np.mean(ps.power)
        mean_gcps = np.mean(gcps.power)
        assert np.isclose(mean_gcps, mean_ps, rtol=0.01)

    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs", "none"])
    def test_gti_corr_ps_fill_rebin(self, norm):
        """Rebinning works on GtiCorrPowerspectrum."""
        lc = copy.deepcopy(self.lc)
        lc.gti = [[0, 30], [35, 100]]  # Two GTIs, one gap
        gcps = GtiCorrPowerspectrum(lc, norm=norm, fill_lc=True)
        gcps = gcps.clean_gti_features()
        ps = Powerspectrum(self.lc, norm=norm)
        ps = ps.rebin_log(0.01)
        gcps = gcps.rebin_log(0.01)

        mean_ps = np.mean(ps.power)
        mean_gcps = np.mean(gcps.power)
        assert np.isclose(mean_gcps, mean_ps, rtol=0.1)

    def test_gti_corr_apply_gti_lc_fails(self):
        """Applying GTIs to a light curve with gaps in the time array raises an error."""
        lc = copy.deepcopy(self.lc)
        lc.gti = [[0, 30], [35, 100]]  # Two GTIs, one gap
        lc.apply_gtis()
        with pytest.raises(ValueError, match="The time array in the light"):
            GtiCorrPowerspectrum(lc, norm="leahy", fill_lc=True)

    def test_gti_corr_plot(self):
        """Plotting GtiCorrPowerspectrum works."""
        lc = copy.deepcopy(self.events)
        lc.gti = [[0, 30], [35, 100]]  # Two GTIs, one gap
        gcps = GtiCorrPowerspectrum(lc, dt=0.01, norm="leahy", fill_lc=True)
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            figname = tmpfile.name
        try:
            gcps = gcps.clean_gti_features(plot=True, figname=figname)
            jpg_name = figname + ".jpg"
            assert os.path.exists(jpg_name)
            os.unlink(jpg_name)
        finally:
            if os.path.exists(figname):
                os.unlink(figname)


class TestPowerspectrum(object):
    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.arange(tstart + 0.5 * dt, tend + 0.5 * dt, dt)

        mean_count_rate = 100.0
        mean_counts = mean_count_rate * dt

        poisson_counts = rng.poisson(mean_counts, size=time.shape[0])

        cls.lc = Lightcurve(time, counts=poisson_counts, dt=dt, gti=[[tstart, tend]])

    def test_single_ps_of_lc_with_tight_gtis_does_not_crash(self):
        tstart = 1.0
        tend = 10.0
        gti = [[1.0, 9.0]]

        time = np.linspace(tstart, tend, 10000)
        counts = np.random.poisson(10, size=time.shape[0])

        lc = Lightcurve(time, counts, gti=gti)
        Powerspectrum(lc, norm="leahy")

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_initialize_empty(self, skip_checks):
        cs = Powerspectrum(skip_checks=skip_checks)
        assert cs.freq is None

    def test_make_empty_periodogram(self):
        ps = Powerspectrum()
        assert ps.norm == "frac"
        assert ps.freq is None
        assert ps.power is None
        assert ps.power_err is None
        assert ps.df is None
        assert ps.m == 1
        assert ps.n is None

    def test_make_periodogram_from_lightcurve(self):
        ps = Powerspectrum(self.lc)
        assert ps.freq is not None
        assert ps.power is not None
        assert ps.power_err is not None
        assert np.isclose(ps.df, 1.0 / self.lc.tseg)
        assert ps.norm == "frac"
        assert ps.m == 1
        assert ps.n == self.lc.time.shape[0]
        assert ps.nphots == np.sum(self.lc.counts)

    def test_periodogram_types(self):
        ps = Powerspectrum(self.lc)
        assert isinstance(ps.freq, np.ndarray)
        assert isinstance(ps.power, np.ndarray)
        assert isinstance(ps.power_err, np.ndarray)

    def test_init_with_lightcurve(self):
        assert Powerspectrum(self.lc)

    def test_init_without_lightcurve(self):
        with pytest.raises(TypeError):
            assert Powerspectrum(self.lc.counts)

    def test_init_with_nonsense_list(self):
        nonsense_data = [None for i in range(100)]
        with pytest.raises(TypeError):
            assert Powerspectrum(nonsense_data)

    def test_init_with_nonsense_data(self):
        nonsense_data = 34
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
        ps = Powerspectrum(self.lc)
        nn = ps.n
        pp = ps.unnorm_power / float(nn) ** 2
        p_int = np.sum(pp[:-1] * ps.df) + (pp[-1] * ps.df) / 2
        var_lc = np.var(self.lc.counts) / (2.0 * self.lc.tseg)
        assert np.isclose(p_int, var_lc, atol=0.01, rtol=0.01)

    def test_frac_normalization_is_standard(self):
        """
        Make sure the standard normalization of a periodogram is
        rms and it stays that way!
        """
        ps = Powerspectrum(self.lc)
        assert ps.norm == "frac"

    def test_frac_normalization_correct(self):
        """
        In fractional rms normalization, the integral of the powers should be
        equal to the variance of the light curve divided by the mean
        of the light curve squared.
        """
        ps = Powerspectrum(self.lc, norm="frac")
        ps_int = np.sum(ps.power[:-1] * ps.df) + ps.power[-1] * ps.df / 2
        std_lc = np.var(self.lc.counts) / np.mean(self.lc.counts) ** 2
        assert np.isclose(ps_int, std_lc, atol=0.01, rtol=0.01)

    def test_leahy_norm_Poisson_noise(self):
        """
        In Leahy normalization, the poisson noise level (so, in the absence of
        a signal, the average power) should be equal to 2.
        """
        time = np.linspace(0, 10.0, 10**5)
        counts = rng.poisson(1000, size=time.shape[0])

        lc = Lightcurve(time, counts)
        ps = Powerspectrum(lc, norm="leahy")

        assert np.isclose(np.mean(ps.power[1:]), 2.0, atol=0.01, rtol=0.01)

    def test_leahy_norm_total_variance(self):
        """
        In Leahy normalization, the total variance should be the sum of
        powers multiplied by the number of counts and divided by the
        square of the number of data points in the light curve
        """
        ps = Powerspectrum(self.lc, norm="Leahy")
        ps_var = (np.sum(self.lc.counts) / ps.n**2) * (np.sum(ps.power[:-1]) + ps.power[-1] / 2.0)

        assert np.isclose(ps_var, np.var(self.lc.counts), atol=0.01)

    def test_abs_norm_Poisson_noise(self):
        """
        Poisson noise level for a light curve with absolute rms-squared
        normalization should be approximately 2 * the mean count rate of the
        light curve.
        """

        time = np.linspace(0, 1.0, 10**4)
        counts = rng.poisson(0.01, size=time.shape[0])

        lc = Lightcurve(time, counts)
        ps = Powerspectrum(lc, norm="abs")
        abs_noise = 2.0 * 100  # expected Poisson noise level;
        # hardcoded value from above
        assert np.isclose(np.mean(ps.power[1:]), abs_noise, atol=50)

    def test_fractional_rms_error(self):
        """
        TODO: Need to write a test for the fractional rms error.
        But I don't know how!
        """
        pass

    def test_rebin_makes_right_attributes(self):
        ps = Powerspectrum(self.lc, norm="Leahy")
        # replace powers
        ps.power = np.ones_like(ps.power) * 2.0

        rebin_factor = 2
        bin_ps = ps.rebin(rebin_factor * ps.df)

        assert bin_ps.freq is not None
        assert bin_ps.power is not None
        assert bin_ps.power is not None
        assert np.isclose(bin_ps.df, rebin_factor * 1.0 / self.lc.tseg)
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
        ps = Powerspectrum(self.lc, norm="Leahy")
        assert ps.rebin.__defaults__[2] == "mean"

    @pytest.mark.parametrize("df", [2, 3, 5, 1.5, 1, 85])
    def test_rebin(self, df):
        """
        TODO: Not sure how to write tests for the rebin method!
        """
        ps = Powerspectrum(self.lc, norm="Leahy")
        bin_ps = ps.rebin(df)
        assert np.isclose(bin_ps.freq[1] - bin_ps.freq[0], bin_ps.df, atol=1e-4, rtol=1e-4)
        assert np.isclose(
            bin_ps.freq[0], (ps.freq[0] - ps.df * 0.5 + bin_ps.df * 0.5), atol=1e-4, rtol=1e-4
        )

    def test_lc_keyword_deprecation(self):
        cs1 = Powerspectrum(self.lc, norm="Leahy")
        with pytest.warns(DeprecationWarning) as record:
            cs2 = Powerspectrum(lc=self.lc, norm="Leahy")
        assert np.any(["lc keyword" in r.message.args[0] for r in record])
        assert np.allclose(cs1.power, cs2.power)
        assert np.allclose(cs1.freq, cs2.freq)

    def test_classical_significances_runs(self):
        ps = Powerspectrum(self.lc, norm="Leahy")
        ps.classical_significances()

    def test_classical_significances_fails_in_rms(self):
        ps = Powerspectrum(self.lc, norm="frac")
        with pytest.raises(ValueError):
            ps.classical_significances()

    def test_classical_significances_threshold(self):
        ps = Powerspectrum(self.lc, norm="leahy")

        # change the powers so that just one exceeds the threshold
        ps.power = np.zeros_like(ps.power) + 2.0

        index = 1
        ps.power[index] = 10.0

        threshold = 0.01

        pval = ps.classical_significances(threshold=threshold, trial_correction=False)
        assert pval[0, 0] < threshold
        assert pval[1, 0] == index

    def test_classical_significances_trial_correction(self):
        ps = Powerspectrum(self.lc, norm="leahy")
        # change the powers so that just one exceeds the threshold
        ps.power = np.zeros_like(ps.power) + 2.0
        index = 1
        ps.power[index] = 10.0
        threshold = 0.01
        pval = ps.classical_significances(threshold=threshold, trial_correction=True)
        assert np.size(pval) == 0

    def test_classical_significances_with_logbinned_psd(self):
        ps = Powerspectrum(self.lc, norm="leahy")
        ps_log = ps.rebin_log()
        pval = ps_log.classical_significances(threshold=1.1, trial_correction=False)

        assert len(pval[0]) == len(ps_log.power)

    def test_pvals_is_numpy_array(self):
        ps = Powerspectrum(self.lc, norm="leahy")
        # change the powers so that just one exceeds the threshold
        ps.power = np.zeros_like(ps.power) + 2.0

        index = 1
        ps.power[index] = 10.0

        threshold = 1.0

        pval = ps.classical_significances(threshold=threshold, trial_correction=True)

        assert isinstance(pval, np.ndarray)
        assert pval.shape[0] == 2


class TestAveragedPowerspectrum(object):
    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 10.0
        dt = 0.0001

        time = np.arange(tstart + 0.5 * dt, tend + 0.5 * dt, dt)

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate * dt

        poisson_counts = rng.poisson(mean_counts, size=time.shape[0])

        cls.lc = Lightcurve(time, counts=poisson_counts, gti=[[tstart, tend]], dt=dt)

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_initialize_empty(self, skip_checks):
        cs = AveragedPowerspectrum(skip_checks=skip_checks)
        assert cs.freq is None

    def test_one_segment(self):
        segment_size = self.lc.tseg

        ps = AveragedPowerspectrum(self.lc, segment_size)
        assert np.isclose(ps.segment_size, segment_size)

    def test_lc_keyword_deprecation(self):
        cs1 = AveragedPowerspectrum(self.lc, segment_size=self.lc.tseg)
        with pytest.warns(DeprecationWarning) as record:
            cs2 = AveragedPowerspectrum(lc=self.lc, segment_size=self.lc.tseg)
        assert np.any(["lc keyword" in r.message.args[0] for r in record])
        assert np.allclose(cs1.power, cs2.power)
        assert np.allclose(cs1.freq, cs2.freq)

    def test_make_empty_periodogram(self):
        ps = AveragedPowerspectrum()
        assert ps.norm == "frac"
        assert ps.freq is None
        assert ps.power is None
        assert ps.power_err is None
        assert ps.df is None
        assert ps.m == 1
        assert ps.n is None

    @pytest.mark.parametrize("nseg", [1, 2, 3, 5, 10, 20, 100])
    def test_n_segments(self, nseg):
        segment_size = self.lc.tseg / nseg
        ps = AveragedPowerspectrum(self.lc, segment_size)
        assert ps.m == nseg

    def test_segments_with_leftover(self):
        segment_size = self.lc.tseg / 2.0 - 1.0
        ps = AveragedPowerspectrum(self.lc, segment_size)
        assert np.isclose(ps.segment_size, segment_size)
        assert ps.m == 2

    def test_init_without_segment(self):
        with pytest.raises(ValueError):
            assert AveragedPowerspectrum(self.lc)

    def test_init_with_nonsense_segment(self):
        segment_size = "foo"
        with pytest.raises(TypeError):
            assert AveragedPowerspectrum(self.lc, segment_size)

    def test_init_with_none_segment(self):
        segment_size = None
        with pytest.raises(ValueError):
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

        time = np.arange(tstart + 0.5 * dt, tend + 0.5 * dt, dt)

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate * dt

        lc_all = []
        for n in range(n_lcs):
            poisson_counts = rng.poisson(mean_counts, size=len(time))

            lc = Lightcurve(time, counts=poisson_counts, gti=[[tstart, tend]], dt=dt)
            lc_all.append(lc)

        segment_size = 0.5
        assert AveragedPowerspectrum(lc_all, segment_size)

    def test_with_zero_counts(self):
        nbins = 100
        x = np.linspace(0, 10, nbins)
        y0 = rng.normal(loc=10, scale=0.5, size=int(0.4 * nbins))
        y1 = np.zeros(int(0.6 * nbins))
        y = np.hstack([y0, y1])

        lc = Lightcurve(x, y)
        aps = AveragedPowerspectrum(lc, segment_size=5.0, norm="leahy")
        assert aps.m == 1

    def test_with_iterable_of_lightcurves(self):
        def iter_lc(lc, n):
            "Generator of n parts of lc."
            t0 = int(len(lc) / n)
            t = t0
            i = 0
            while True:
                lc_seg = lc[i:t]
                yield lc_seg
                if t + t0 > len(lc):
                    break
                else:
                    i, t = t, t + t0

        with pytest.warns(UserWarning) as record:
            cs = AveragedPowerspectrum(iter_lc(self.lc, 1), segment_size=1, gti=self.lc.gti)
        message = "The averaged power spectrum from a generator"

        assert np.any([message in r.message.args[0] for r in record])

    def test_with_iterable_of_variable_length_lightcurves(self):
        gti = [[0, 0.05], [0.05, 0.5], [0.555, 1.0]]
        lc = copy.deepcopy(self.lc)
        lc.gti = gti
        lc_split = lc.split_by_gti()

        cs = AveragedPowerspectrum(lc_split, segment_size=0.05, norm="leahy", gti=lc.gti)
        cs_lc = AveragedPowerspectrum(lc, segment_size=0.05, norm="leahy", gti=lc.gti)
        for attr in ("power", "unnorm_power", "power_err", "unnorm_power_err", "freq"):
            assert np.allclose(getattr(cs, attr), getattr(cs_lc, attr))

        for attr in ("m", "n", "norm"):
            assert getattr(cs, attr) == getattr(cs_lc, attr)

    @pytest.mark.parametrize("df", [2, 3, 5, 1.5, 1, 85])
    def test_rebin(self, df):
        """
        TODO: Not sure how to write tests for the rebin method!
        """

        aps = AveragedPowerspectrum(self.lc, segment_size=1, norm="Leahy")
        bin_aps = aps.rebin(df)
        assert np.isclose(bin_aps.freq[1] - bin_aps.freq[0], bin_aps.df, atol=1e-4, rtol=1e-4)
        assert np.isclose(
            bin_aps.freq[0], (aps.freq[0] - aps.df * 0.5 + bin_aps.df * 0.5), atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize("f", [20, 30, 50, 15, 1, 850])
    def test_rebin_factor(self, f):
        """
        TODO: Not sure how to write tests for the rebin method!
        """

        aps = AveragedPowerspectrum(self.lc, segment_size=1, norm="Leahy")
        bin_aps = aps.rebin(f=f)
        assert np.isclose(bin_aps.freq[1] - bin_aps.freq[0], bin_aps.df, atol=1e-4, rtol=1e-4)
        assert np.isclose(
            bin_aps.freq[0], (aps.freq[0] - aps.df * 0.5 + bin_aps.df * 0.5), atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize("df", [0.01, 0.1])
    def test_rebin_log(self, df):
        # For now, just verify that it doesn't crash
        aps = AveragedPowerspectrum(self.lc, segment_size=1, norm="Leahy")
        bin_aps = aps.rebin_log(df)

    def test_list_with_nonsense_component(self):
        n_lcs = 10

        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend - tstart) / dt))

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate * dt

        lc_all = []
        for n in range(n_lcs):
            poisson_counts = rng.poisson(mean_counts, size=len(time))

            lc = Lightcurve(time, counts=poisson_counts)
            lc_all.append(lc)

        lc_all.append(1.0)
        segment_size = 0.5

        with pytest.raises(TypeError):
            assert AveragedPowerspectrum(lc_all, segment_size)

    @pytest.mark.parametrize("use_common_mean", [True, False])
    def test_leahy_correct_for_multiple(self, use_common_mean):
        n = 10
        lc_all = []
        for i in range(n):
            time = np.arange(0.0, 10.0, 10.0 / 10000)
            counts = rng.poisson(1000, size=time.shape[0])
            lc = Lightcurve(time, counts)
            lc_all.append(lc)

        ps = AveragedPowerspectrum(lc_all, 1.0, norm="leahy", use_common_mean=use_common_mean)

        assert np.isclose(np.mean(ps.power), 2.0, atol=1e-2, rtol=1e-2)
        assert np.isclose(np.std(ps.power), 2.0 / np.sqrt(n * 10), atol=0.1, rtol=0.1)


class TestDynamicalPowerspectrum(object):
    def setup_class(cls):
        # generate timestamps
        timestamps = np.arange(0.005, 100.01, 0.01)
        dt = 0.01
        freq = 25 + 1.2 * np.sin(2 * np.pi * timestamps / 130)
        # variability signal with drifiting frequency
        vari = 25 * np.sin(2 * np.pi * freq * timestamps)
        signal = vari + 50
        # create a lightcurve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            lc = Lightcurve(timestamps, signal, err_dist="poisson", dt=dt, gti=[[0, 100]])

        cls.lc = lc

        # Simple lc to demonstrate rebinning of dyn ps
        # Simple lc to demonstrate rebinning of dyn ps
        test_times = np.arange(16)
        test_counts = [2, 3, 1, 3, 1, 5, 2, 1, 4, 2, 2, 2, 3, 4, 1, 7]
        cls.lc_test = Lightcurve(test_times, test_counts)

    def test_bad_args(self):
        with pytest.raises(TypeError, match=".must all be specified"):
            _ = DynamicalPowerspectrum(1)

    def test_with_short_seg_size(self):
        with pytest.raises(ValueError):
            dps = DynamicalPowerspectrum(self.lc, segment_size=0)

    def test_works_with_events(self):
        lc = copy.deepcopy(self.lc)
        lc.counts = np.floor(lc.counts)
        ev = EventList.from_lc(lc)
        dps = DynamicalPowerspectrum(lc, segment_size=10)
        with pytest.raises(ValueError):
            # Without dt, it fails
            _ = DynamicalPowerspectrum(ev, segment_size=10)

        dps_ev = DynamicalPowerspectrum(ev, segment_size=10, sample_time=self.lc.dt)
        assert np.allclose(dps.dyn_ps, dps_ev.dyn_ps)
        dps_ev.power_colors(freq_edges=[1 / 5, 1 / 2, 1, 2.0, 16.0])

    def test_with_long_seg_size(self):
        with pytest.raises(ValueError):
            dps = DynamicalPowerspectrum(self.lc, segment_size=1000)

    def test_matrix(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dps = DynamicalPowerspectrum(self.lc, segment_size=3)
        nsegs = int(self.lc.tseg / dps.segment_size)
        nfreq = int((1 / self.lc.dt) / (2 * (dps.freq[1] - dps.freq[0])) - (1 / self.lc.tseg))
        assert dps.dyn_ps.shape == (nfreq, nsegs)

    def test_trace_maximum_without_boundaries(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dps = DynamicalPowerspectrum(self.lc, segment_size=3)
        max_pos = dps.trace_maximum()

        assert np.max(dps.freq[max_pos]) <= 1 / self.lc.dt
        assert np.min(dps.freq[max_pos]) >= 1 / dps.segment_size

    def test_trace_maximum_with_boundaries(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dps = DynamicalPowerspectrum(self.lc, segment_size=3)
        minfreq = 21
        maxfreq = 24
        max_pos = dps.trace_maximum(min_freq=minfreq, max_freq=maxfreq)

        assert np.max(dps.freq[max_pos]) <= maxfreq
        assert np.min(dps.freq[max_pos]) >= minfreq

    def test_size_of_trace_maximum(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dps = DynamicalPowerspectrum(self.lc, segment_size=3)
        max_pos = dps.trace_maximum()
        nsegs = int(self.lc.tseg / dps.segment_size)
        assert len(max_pos) == nsegs

    def test_rebin_small_dt(self):
        segment_size = 3
        dps = DynamicalPowerspectrum(self.lc_test, segment_size=segment_size)
        with pytest.raises(ValueError):
            dps.rebin_time(dt_new=2.0)

    def test_rebin_small_df(self):
        segment_size = 3
        dps = DynamicalPowerspectrum(self.lc, segment_size=segment_size)
        with pytest.raises(ValueError):
            dps.rebin_frequency(df_new=dps.df / 2.0)

    def test_rebin_time_sum_method(self):
        segment_size = 3
        dt_new = 6.0
        rebin_time = np.array([2.5, 8.5])
        rebin_dps = np.array([[1.73611111, 0.81018519]])
        dps = DynamicalPowerspectrum(self.lc_test, segment_size=segment_size)
        new_dps = dps.rebin_time(dt_new=dt_new, method="sum")
        assert np.allclose(new_dps.time, rebin_time)
        assert np.allclose(new_dps.dyn_ps, rebin_dps)
        assert np.isclose(new_dps.dt, dt_new)

    def test_rebin_frequency_default_method(self):
        segment_size = 50
        df_new = 10.0
        rebin_freq = np.array([5.01, 15.01, 25.01, 35.01])
        rebin_dps = np.array(
            [
                [
                    [1.71989342e-04, 6.42756881e-05],
                    [7.54455204e-04, 2.14785049e-04],
                    [6.24831554e00, 6.24984615e00],
                    [6.71135792e-04, 7.42516599e-05],
                ]
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dps = DynamicalPowerspectrum(self.lc, segment_size=segment_size)
        new_dps = dps.rebin_frequency(df_new=df_new, method="sum")
        assert np.allclose(new_dps.freq, rebin_freq)
        assert np.allclose(new_dps.dyn_ps, rebin_dps, atol=0.01)
        assert np.isclose(new_dps.df, df_new)

    @pytest.mark.parametrize("method", ["mean", "average"])
    def test_rebin_time_mean_method(self, method):
        segment_size = 3
        dt_new = 6.0
        rebin_time = np.array([2.5, 8.5])
        rebin_dps = np.array([[1.73611111, 0.81018519]]) / 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dps = DynamicalPowerspectrum(self.lc_test, segment_size=segment_size)
        new_dps = dps.rebin_time(dt_new=dt_new, method=method)
        assert np.allclose(new_dps.time, rebin_time)
        assert np.allclose(new_dps.dyn_ps, rebin_dps)
        assert np.isclose(new_dps.dt, dt_new)

    @pytest.mark.parametrize("method", ["mean", "average"])
    def test_rebin_frequency_mean_method(self, method):
        segment_size = 50
        df_new = 10.0
        rebin_freq = np.array([5.01, 15.01, 25.01, 35.01])
        rebin_dps = (
            np.array(
                [
                    [
                        [1.71989342e-04, 6.42756881e-05],
                        [7.54455204e-04, 2.14785049e-04],
                        [6.24831554e00, 6.24984615e00],
                        [6.71135792e-04, 7.42516599e-05],
                    ]
                ]
            )
            / 500
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dps = DynamicalPowerspectrum(self.lc, segment_size=segment_size)
        new_dps = dps.rebin_frequency(df_new=df_new, method=method)
        assert np.allclose(new_dps.freq, rebin_freq)
        assert np.allclose(new_dps.dyn_ps, rebin_dps, atol=0.00001)
        assert np.isclose(new_dps.df, df_new)

    def test_shift_and_add(self):
        power_list = [[2, 5, 2, 2, 2], [1, 1, 5, 1, 1], [3, 3, 3, 5, 3]]
        power_list = np.array(power_list).T
        freqs = np.arange(5) * 0.1
        f0_list = [0.1, 0.2, 0.3, 0.4]
        dps = DynamicalPowerspectrum()
        dps.dyn_ps = power_list
        dps.freq = freqs
        dps.df = 0.1
        dps.m = 1
        output = dps.shift_and_add(f0_list, nbins=5)
        assert np.array_equal(output.m, [2, 3, 3, 3, 2])
        assert np.array_equal(output.power, [2.0, 2.0, 5.0, 2.0, 1.5])
        assert np.allclose(output.freq, [0.05, 0.15, 0.25, 0.35, 0.45])

    def test_shift_and_add_rebin(self):
        power_list = [[2, 5, 2, 2, 2], [1, 1, 5, 1, 1], [3, 3, 3, 5, 3]]
        power_list = np.array(power_list).T
        freqs = np.arange(5) * 0.1
        f0_list = [0.1, 0.2, 0.3, 0.4]
        dps = DynamicalPowerspectrum()
        dps.dyn_ps = power_list
        dps.freq = freqs
        dps.df = 0.1
        dps.m = 1
        output = dps.shift_and_add(f0_list, nbins=5, rebin=2)
        assert np.array_equal(output.m, [5, 6])
        assert np.array_equal(output.power, [2.0, 3.5])
        assert np.allclose(output.freq, [0.1, 0.3])


class TestRoundTrip:
    @classmethod
    def setup_class(cls):
        cls.cs = AveragedPowerspectrum()
        cls.cs.freq = np.arange(10)
        cls.cs.power = rng.uniform(0, 10, 10)
        cls.cs.m = 2
        cls.cs.nphots1 = 34
        cls.cs.df = 1
        cls.cs.power_err = np.ones_like(cls.cs.power) * 0.5

    def _check_equal(self, so, new_so):
        for attr in ["freq", "power"]:
            assert np.allclose(getattr(so, attr), getattr(new_so, attr))

        for attr in ["m", "nphots1"]:
            assert getattr(so, attr) == getattr(new_so, attr)

    def test_astropy_roundtrip(self):
        so = self.cs
        ts = so.to_astropy_table()
        new_so = so.from_astropy_table(ts)
        self._check_equal(so, new_so)

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip(self):
        so = self.cs
        ts = so.to_xarray()
        new_so = so.from_xarray(ts)

        self._check_equal(so, new_so)

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip(self):
        so = self.cs
        ts = so.to_pandas()
        new_so = so.from_pandas(ts)

        self._check_equal(so, new_so)

    @pytest.mark.skipif("not _HAS_H5PY")
    def test_hdf_roundtrip(self):
        so = self.cs
        so.write("dummy.hdf5")
        new_so = so.read("dummy.hdf5")
        os.unlink("dummy.hdf5")

        self._check_equal(so, new_so)

    @pytest.mark.skipif("not _HAS_FLX2XSP")
    def test_save_as_xspec(self):
        so = self.cs
        try:
            so.save_as_xspec("dummy_pow")
            assert os.path.exists("dummy_pow.pha")
            assert os.path.exists("dummy_pow.rsp")
            assert os.path.exists("dummy_pow.txt")
        finally:
            for ext in [".pha", ".rsp", ".txt"]:
                if os.path.exists(f"dummy_pow{ext}"):
                    os.unlink(f"dummy_pow{ext}")

    @pytest.mark.skipif("_HAS_FLX2XSP")
    def test_save_as_xspec_fails_no_flx2xsp(self):
        so = self.cs
        try:
            with pytest.raises(RuntimeError, match="install and initialize HEASOFT to save"):
                so.save_as_xspec("dummy")
        finally:
            if os.path.exists("dummy.txt"):
                os.unlink("dummy.txt")

    def test_save_as_xspec_fails_no_df(self):
        so = copy.deepcopy(self.cs)
        so.df = None
        with pytest.raises(
            ValueError, match="powerspectrum object has no attribute 'df' or it is None."
        ):
            so.save_as_xspec("dummy")

    def test_save_as_xspec_fails_no_power_err(self):
        so = copy.deepcopy(self.cs)
        so.power_err = None
        with pytest.raises(
            ValueError, match="powerspectrum object has no attribute 'power_err' or it is None."
        ):
            so.save_as_xspec("dummy")

    @pytest.mark.skipif("_HAS_FLX2XSP")
    def test_save_as_xspec_mock(self):
        from unittest.mock import patch

        def function(blah, root):
            for fnames in [root + ".pha", root + ".rsp", root + ".txt"]:
                with open(fnames, "w") as f:
                    f.write("dummy")

        so = self.cs

        try:
            with patch("stingray.io.run_flx2xsp", side_effect=function) as mock_flx2xsp:
                so.save_as_xspec("dummy_pow")

            for ext in [".pha", ".rsp", ".txt"]:
                assert os.path.exists(f"dummy_pow{ext}")
        finally:
            for ext in [".pha", ".rsp", ".txt"]:
                if os.path.exists(f"dummy_pow{ext}"):
                    os.unlink(f"dummy_pow{ext}")

    @pytest.mark.parametrize("fmt", ["pickle"])
    def test_file_roundtrip(self, fmt):
        so = self.cs
        fname = f"dummy.{fmt}"
        so.write(fname, fmt=fmt)
        new_so = so.read(fname, fmt=fmt)
        # os.unlink(fname)

        self._check_equal(so, new_so)

    @pytest.mark.parametrize("fmt", ["ascii", "ascii.ecsv", "fits"])
    def test_file_roundtrip_lossy(self, fmt):
        so = self.cs
        fname = f"dummy.{fmt}"
        with pytest.warns(UserWarning, match=".* output does not serialize the metadata"):
            so.write(fname, fmt=fmt)
        new_so = so.read(fname, fmt=fmt)
        os.unlink(fname)

        self._check_equal(so, new_so)


class TestRMS(object):
    @classmethod
    def setup_class(cls):
        fwhm = 0.23456
        cls.segment_size = 256
        cls.df = 1 / cls.segment_size

        cls.freqs = np.arange(cls.df, 1, cls.df)
        dt = 0.5 / cls.freqs.max()

        pds_shape_func = Lorentz1D(x_0=0, fwhm=fwhm)
        cls.pds_shape_raw = pds_shape_func(cls.freqs)
        cls.M = 1000
        cls.nphots = 1_000_000
        cls.rms = 0.5
        meanrate = cls.nphots / cls.segment_size
        cls.poisson_noise_rms = 2 / meanrate
        pds_shape_rms = cls.pds_shape_raw / np.sum(cls.pds_shape_raw * cls.df) * cls.rms**2
        pds_shape_rms += cls.poisson_noise_rms

        random_part = rng.chisquare(2 * cls.M, size=cls.pds_shape_raw.size) / 2 / cls.M
        pds_rms_noisy = random_part * pds_shape_rms

        pds_unnorm = pds_rms_noisy * meanrate / 2 * cls.nphots
        cls.pds = AveragedPowerspectrum()
        cls.pds.freq = cls.freqs
        cls.pds.unnorm_power = pds_unnorm
        cls.pds.power = pds_rms_noisy
        cls.pds.df = cls.df
        cls.pds.m = cls.M
        cls.pds.nphots = cls.nphots
        cls.pds.norm = "frac"
        cls.pds.dt = dt
        cls.pds.n = cls.pds.freq.size

    @pytest.mark.parametrize("norm", ["none", "frac", "leahy", "abs"])
    def test_rms(self, norm):
        pds = self.pds.to_norm(norm)
        rms_from_ps, rmse_from_ps = pds.compute_rms(self.freqs.min(), self.freqs.max())
        assert np.isclose(rms_from_ps, self.rms, atol=3 * rmse_from_ps)

    def test_rms_M_low(self):
        """Test that the warning is raised when M is low."""

        pds = copy.deepcopy(self.pds)
        pds.m = 23

        with pytest.warns(UserWarning, match="All power spectral bins have M<30."):
            pds.compute_rms(
                self.freqs.min(), self.freqs.max(), poisson_noise_level=self.poisson_noise_rms
            )

    @pytest.mark.parametrize("norm", ["none", "frac", "leahy", "abs"])
    def test_rms_rebinning(self, norm):
        pds = self.pds.to_norm(norm)
        pds = pds.rebin_log(0.04)
        rms_from_ps, rmse_from_ps = pds.compute_rms(self.freqs.min(), self.freqs.max())

        assert np.isclose(rms_from_ps, self.rms, atol=3 * rmse_from_ps)

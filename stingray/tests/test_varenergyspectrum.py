import copy
from multiprocessing import Event
import os
import shutil
import numpy as np
from stingray.events import EventList
from stingray.varenergyspectrum import VarEnergySpectrum
from stingray.varenergyspectrum import ComplexCovarianceSpectrum, CovarianceSpectrum
from stingray.varenergyspectrum import RmsSpectrum, RmsEnergySpectrum, CountSpectrum
from stingray.varenergyspectrum import LagSpectrum, LagEnergySpectrum
from stingray.varenergyspectrum import ExcessVarianceSpectrum
from stingray.lightcurve import Lightcurve

import pytest
from astropy.table import Table

_HAS_XARRAY = _HAS_PANDAS = _HAS_H5PY = True

try:
    import xarray
    from xarray import Dataset
except ImportError:
    _HAS_XARRAY = False

try:
    import pandas
    from pandas import DataFrame
except ImportError:
    _HAS_PANDAS = False

try:
    import h5py
except ImportError:
    _HAS_H5PY = False

_HAS_FLX2XSP = (os.getenv("HEADAS") is not None) and (shutil.which("ftflx2xsp") is not None)

np.random.seed(20150907)
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


class DummyVarEnergy(VarEnergySpectrum):
    def _spectrum_function(self):
        return None, None


class TestExcVarEnergySpectrum(object):
    @classmethod
    def setup_class(cls):
        from ..simulator import Simulator

        simulator = Simulator(0.1, 10000, rms=0.2, mean=200)
        test_lc = simulator.simulate(1)
        cls.test_ev1, cls.test_ev2 = EventList(), EventList()
        cls.test_ev1.simulate_times(test_lc)
        cls.test_ev1.energy = np.random.uniform(0.3, 12, len(cls.test_ev1.time))

    def test_allocate(self):
        _ = ExcessVarianceSpectrum(
            self.test_ev1, [0.0, 100], (0.3, 12, 5, "lin"), bin_time=1, segment_size=100
        )

    def test_invalid_norm(self):
        with pytest.raises(ValueError):
            _ = ExcessVarianceSpectrum(
                self.test_ev1,
                [0.0, 100],
                (0.3, 12, 5, "lin"),
                bin_time=1,
                segment_size=100,
                normalization="asdfghjkl",
            )


class TestVarEnergySpectrum(object):
    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 100.0
        nphot = 1000
        alltimes = np.random.uniform(tstart, tend, nphot)
        alltimes.sort()
        cls.events = EventList(
            alltimes, energy=np.random.uniform(0.3, 12, nphot), gti=[[tstart, tend]]
        )
        cls.vespec = DummyVarEnergy(
            cls.events, [0.0, 10000], (0.5, 5, 10, "lin"), [0.3, 10], bin_time=0.1
        )
        cls.vespeclog = DummyVarEnergy(cls.events, [0.0, 10000], (0.5, 5, 10, "log"), [0.3, 10])

    def test_no_spectrum_func_raises(self):
        with pytest.raises(TypeError):
            ref_int = VarEnergySpectrum(self.events, [0.0, 10000], (0.5, 5, 10, "log"), [0.3, 10])

    @pytest.mark.parametrize("energy_spec", [2, "a"])
    def test_invalid_energy_spec(self, energy_spec):
        with pytest.raises(
            ValueError,
            match=f"Energy specification must be a tuple or a list .input: {energy_spec}.",
        ):
            DummyVarEnergy(self.events, [0.0, 10000], energy_spec=energy_spec)

    def test_ref_band_none(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )
        vespec = DummyVarEnergy(events, [0.0, 10000], (0, 1, 2, "lin"), bin_time=0.1)
        assert np.allclose(vespec.ref_band, np.array([[0, np.inf]]))

    def test_energy_spec_wrong_list_not_tuple(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )
        # Test using a list instead of tuple
        # with pytest.raises(ValueError):
        vespec = DummyVarEnergy(events, [0.0, 10000], [0, 1, 2, "lin"], bin_time=0.1)

    def test_energy_spec_wrong_str(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )
        # Test using a list instead of tuple
        with pytest.raises(ValueError):
            vespec = DummyVarEnergy(events, [0.0, 10000], (0, 1, 2, "xxx"), bin_time=0.1)

    def test_energy_property(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.8, 1.4, 1.9], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )
        energy_spec = [0, 1, 2]
        vespec = DummyVarEnergy(events, [0.0, 10000], energy_spec, [0.5, 1.1], bin_time=0.1)
        assert np.allclose(vespec.energy, [0.5, 1.5])

    def test_construct_lightcurves(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )
        vespec = DummyVarEnergy(events, [0.0, 10000], (0, 1, 2, "lin"), [0.5, 1.1], bin_time=0.1)
        base_lc, ref_lc = vespec._construct_lightcurves([0, 0.5], tstart=0, tstop=0.65)
        np.testing.assert_allclose(base_lc.counts, [1, 0, 2, 1, 0, 0])
        np.testing.assert_allclose(ref_lc.counts, [0, 0, 0, 0, 1, 1])

    def test_construct_lightcurves_no_exclude(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54], energy=[0, 0, 0, 0, 1, 1], gti=[[0, 0.65]]
        )

        vespec = DummyVarEnergy(events, [0.0, 10000], (0, 1, 2, "lin"), [0, 0.5], bin_time=0.1)
        base_lc, ref_lc = vespec._construct_lightcurves(
            [0, 0.5], tstart=0, tstop=0.65, exclude=False
        )
        np.testing.assert_equal(base_lc.counts, ref_lc.counts)

    def test_construct_lightcurves_pi(self):
        events = EventList(
            [0.09, 0.21, 0.23, 0.32, 0.4, 0.54],
            pi=np.asanyarray([0, 0, 0, 0, 1, 1]),
            gti=[[0, 0.65]],
        )
        vespec = DummyVarEnergy(
            events, [0.0, 10000], (0, 1, 2, "lin"), [0.5, 1.1], use_pi=True, bin_time=0.1
        )
        base_lc, ref_lc = vespec._construct_lightcurves([0, 0.5], tstart=0, tstop=0.65)
        np.testing.assert_allclose(base_lc.counts, [1, 0, 2, 1, 0, 0])
        np.testing.assert_allclose(ref_lc.counts, [0, 0, 0, 0, 1, 1])


class TestCountSpectrum(object):
    @classmethod
    def setup_class(cls):
        cls.times = [0.1, 2, 4, 5.5]
        cls.energy = [3, 5, 2, 4]

        cls.events = EventList(time=cls.times, energy=cls.energy, pi=cls.energy, gti=[[0, 6.0]])

    @pytest.mark.parametrize("use_pi", [False, True])
    def test_counts(self, use_pi):
        ctsspec = CountSpectrum(self.events, [1.5, 3.5, 6.5], use_pi=use_pi)
        assert np.allclose(ctsspec.spectrum, 2)


@pytest.mark.slow
class TestRmsAndCovSpectrum(object):
    @classmethod
    def setup_class(cls):
        from ..simulator import Simulator

        cls.bin_time = 0.01

        data = Table.read(os.path.join(datadir, "sample_variable_series.fits"))["data"]
        # No need for huge count rates
        flux = data / 40
        times = np.arange(data.size) * cls.bin_time
        gti = np.asanyarray([[0, data.size * cls.bin_time]])
        test_lc = Lightcurve(
            times, flux, err_dist="gauss", gti=gti, dt=cls.bin_time, skip_checks=True
        )

        cls.test_ev1, cls.test_ev2 = EventList(), EventList()
        cls.test_ev1.simulate_times(test_lc)
        cls.test_ev2.simulate_times(test_lc)
        N1 = cls.test_ev1.time.size
        N2 = cls.test_ev2.time.size
        cls.test_ev1.energy = np.random.uniform(0.3, 12, N1)
        cls.test_ev2.energy = np.random.uniform(0.3, 12, N2)

        mask = np.sort(np.random.randint(0, min(N1, N2) - 1, 200000))
        cls.test_ev1_small = cls.test_ev1.apply_mask(mask)
        cls.test_ev2_small = cls.test_ev2.apply_mask(mask)

    def test_create_complexcovariance(self):
        spec = ComplexCovarianceSpectrum(
            self.test_ev1_small,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=200,
            norm="abs",
            events2=self.test_ev2_small,
        )
        assert np.all(np.iscomplex(spec.spectrum))

    @pytest.mark.parametrize("cross", [True, False])
    def test_empty_subband_lag(self, cross):
        ev2 = None
        if cross:
            ev2 = self.test_ev2_small

        # Note: energy_spec is a list, so it's actually the edges
        # of the energy bins. So, the covariance spectrum will be
        # calculated in two bands: 0.3-12 keV and 12-15 keV. But
        # the 12-15 keV band is empty (see definition of test_ev1_small)
        spec = LagSpectrum(
            self.test_ev1_small,
            freq_interval=[0.00001, 0.1],
            energy_spec=[0.3, 12, 15],
            ref_band=[[0.3, 12]],
            bin_time=self.bin_time / 2,
            segment_size=200,
            events2=ev2,
        )
        good = ~np.isnan(spec.spectrum)
        assert np.count_nonzero(good) == 1

    @pytest.mark.parametrize("cross", [True, False])
    @pytest.mark.parametrize("kind", ["rms", "cov"])
    def test_empty_subband_cov(self, cross, kind):
        ev2 = None
        if cross:
            ev2 = self.test_ev2_small

        if kind == "rms":
            func = RmsSpectrum
        elif kind == "cov":
            func = ComplexCovarianceSpectrum
        # Note: energy_spec is a list, so it's actually the edges
        # of the energy bins. So, the covariance spectrum will be
        # calculated in two bands: 0.3-12 keV and 12-15 keV. But
        # the 12-15 keV band is empty (see definition of test_ev1_small)
        with pytest.warns(UserWarning, match="Low count rate in the 12-15 subject band"):
            spec = func(
                self.test_ev1_small,
                freq_interval=[0.00001, 0.1],
                energy_spec=[0.3, 12, 15],
                ref_band=[[0.3, 12]],
                bin_time=self.bin_time / 2,
                segment_size=200,
                events2=ev2,
            )
        good = ~np.isnan(spec.spectrum)
        assert np.count_nonzero(good) == 1

    def test_empty_subband_cov_ev2(self):
        ev2 = copy.deepcopy(self.test_ev2_small)
        # We empty out only the second event list above 5 keV
        ev2.filter_energy_range([0.3, 5], inplace=True)

        with pytest.warns(UserWarning, match="Low count rate in the 5-12 subject band"):
            spec = RmsSpectrum(
                self.test_ev1_small,
                freq_interval=[0.00001, 0.1],
                energy_spec=[0.3, 5, 12],
                bin_time=self.bin_time / 2,
                segment_size=200,
                events2=ev2,
            )
        good = ~np.isnan(spec.spectrum)
        assert np.count_nonzero(good) == 1

    @pytest.mark.parametrize("norm", ["frac", "abs"])
    def test_correct_rms_values_vs_cross(self, norm):
        """The rms calculated with independent event lists (from the cospectrum)
        is equivalent to the one calculated with one event list (from the PDS)"""

        rmsspec_cross = RmsEnergySpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=9999,
            events2=self.test_ev2,
            norm=norm,
        )
        rmsspec_pds = RmsEnergySpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=9999,
            norm=norm,
        )
        pds = rmsspec_pds.spectrum
        cross = rmsspec_cross.spectrum
        err = rmsspec_pds.spectrum_error
        cerr = rmsspec_cross.spectrum_error
        assert np.allclose(err, cerr, rtol=0.2)
        assert np.allclose(pds, cross, atol=3 * err)

        if norm == "frac":
            assert np.allclose(pds, 0.20, atol=3 * err)

    @pytest.mark.parametrize("norm", ["frac", "abs"])
    def test_correct_cov_values_vs_cross(self, norm):
        """The rms calculated with independent event lists (from the cospectrum)
        is equivalent to the one calculated with one event list (from the PDS)"""
        covar = CovarianceSpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=100,
            norm=norm,
        )

        covar_cross = CovarianceSpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=100,
            norm=norm,
            events2=self.test_ev2,
        )

        cov = covar.spectrum
        cross = covar_cross.spectrum
        coverr = covar.spectrum_error
        crosserr = covar_cross.spectrum_error

        assert np.allclose(cov, cross, atol=3 * coverr)

    @pytest.mark.parametrize("cross", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs"])
    def test_correct_rms_values_vs_cov(self, cross, norm):
        """The rms calculated with independent event lists (from the cospectrum)
        is equivalent to the one calculated with one event list (from the PDS)"""
        ev2 = None
        if cross:
            ev2 = self.test_ev2
        covar = CovarianceSpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=100,
            norm=norm,
            events2=ev2,
        )
        rmsspec = RmsSpectrum(
            self.test_ev1,
            freq_interval=[0.00001, 0.1],
            energy_spec=(0.3, 12, 2, "lin"),
            bin_time=self.bin_time / 2,
            segment_size=100,
            norm=norm,
            events2=ev2,
        )

        cov = covar.spectrum
        rms = rmsspec.spectrum
        coverr = covar.spectrum_error
        rmserr = covar.spectrum_error

        assert np.allclose(cov, rms, atol=3 * coverr)

    def test_cov_invalid_evlist_warns(self):
        ev = EventList(time=[], energy=[], gti=self.test_ev1.gti)
        with pytest.warns(UserWarning) as record:
            rms = CovarianceSpectrum(
                ev, [0.0, 100], (0.3, 12, 5, "lin"), bin_time=0.01, segment_size=100
            )
        assert np.all(np.isnan(rms.spectrum))
        assert np.all(np.isnan(rms.spectrum_error))

    def test_rms_invalid_evlist_warns(self):
        ev = EventList(time=[], energy=[], gti=self.test_ev1.gti)
        with pytest.warns(UserWarning) as record:
            rms = RmsEnergySpectrum(
                ev,
                [0.0, 100],
                (0.3, 12, 5, "lin"),
                bin_time=0.01,
                segment_size=100,
                events2=self.test_ev2,
            )
        assert np.all(np.isnan(rms.spectrum))
        assert np.all(np.isnan(rms.spectrum_error))


import abc


class BaseTestIO(abc.ABC):
    @property
    @abc.abstractmethod
    def variant(self):
        pass

    @classmethod
    def setup_class(cls):
        if cls.variant == "rms":
            cls.func = RmsSpectrum
        elif cls.variant == "complcov":
            cls.func = ComplexCovarianceSpectrum
        elif cls.variant == "cov":
            cls.func = CovarianceSpectrum
        elif cls.variant == "lag":
            cls.func = LagSpectrum
        spec = cls.func(energy_spec=[0.3, 12, 15])
        spec.freq_interval = [0.1, 0.2]
        spec.ref_band = [0.3, 12]
        spec.bin_time = 0.01
        spec.segment_size = 100
        spec.cross = cls.variant == "complcov"
        spec.spectrum = np.array([1.0, 2.0])
        spec.spectrum_error = np.array([0.1, 0.2])
        if spec.cross:
            spec.spectrum = spec.spectrum + 1.0j
            spec.spectrum_error = spec.spectrum_error + 0.1j
        spec.energy_intervals = np.array([[0.3, 12], [12, 15]])
        cls.sting_obj = spec

    def test_astropy_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        ts = so.to_astropy_table()
        new_so = self.func.from_astropy_table(ts)
        assert so == new_so

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        ts = so.to_xarray()
        new_so = self.func.from_xarray(ts)
        assert so == new_so

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        ts = so.to_pandas()
        new_so = self.func.from_pandas(ts)
        assert so == new_so

    def test_astropy_roundtrip_empty(self):
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        so = self.func()
        ts = so.to_astropy_table()
        new_so = self.func.from_astropy_table(ts)
        assert new_so.energy == []
        assert so == new_so

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip_empty(self):
        so = self.func()
        ts = so.to_xarray()
        new_so = self.func.from_xarray(ts)
        assert new_so.energy == []
        assert so == new_so

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip_empty(self):
        so = self.func()
        ts = so.to_pandas()
        new_so = self.func.from_pandas(ts)
        assert new_so.energy == []
        assert so == new_so

    @pytest.mark.skipif("not _HAS_H5PY")
    def test_hdf_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        so.write("dummy.hdf5")
        new_so = so.read("dummy.hdf5")
        os.unlink("dummy.hdf5")

        assert so == new_so

    def test_file_roundtrip_fits(self):
        so = copy.deepcopy(self.sting_obj)
        with pytest.warns(
            UserWarning, match=".* output does not serialize the metadata at the moment"
        ):
            so.write("dummy.fits")
        new_so = self.func.read("dummy.fits")
        os.unlink("dummy.fits")
        assert so == new_so

    @pytest.mark.parametrize("fmt", ["ascii", "ascii.ecsv"])
    def test_file_roundtrip(self, fmt):
        so = copy.deepcopy(self.sting_obj)
        with pytest.warns(UserWarning, match=".* output does not serialize the metadata"):
            so.write(f"dummy.{fmt}", fmt=fmt)
        new_so = self.func.read(f"dummy.{fmt}", fmt=fmt)
        os.unlink(f"dummy.{fmt}")

        assert so == new_so

    def test_file_roundtrip_pickle(self):
        fmt = "pickle"
        so = copy.deepcopy(self.sting_obj)
        so.write(f"dummy.{fmt}", fmt=fmt)
        new_so = self.func.read(f"dummy.{fmt}", fmt=fmt)
        os.unlink(f"dummy.{fmt}")

        assert so == new_so

    def test_save_as_xspec(self):
        from unittest.mock import patch

        def function(blah, root):
            for fnames in [root + ".pha", root + ".rsp", root + ".txt"]:
                with open(fnames, "w") as f:
                    f.write("dummy")

        so = copy.deepcopy(self.sting_obj)

        with patch("stingray.io.run_flx2xsp", side_effect=function) as mock_flx2xsp:
            if self.variant == "complcov":
                try:
                    with pytest.warns(UserWarning, match="No header keywords provided. "):
                        so.save_as_xspec("dummy")
                    for part in ["real", "imag"]:
                        for ext in ["pha", "rsp", "txt"]:
                            assert os.path.exists(f"dummy_{part}.{ext}")
                finally:
                    for part in ["real", "imag"]:
                        for ext in ["pha", "rsp", "txt"]:
                            if os.path.exists(f"dummy_{part}.{ext}"):
                                os.unlink(f"dummy_{part}.{ext}")
            else:
                try:
                    with pytest.warns(UserWarning, match="No header keywords provided. "):
                        so.save_as_xspec("dummy_pow")
                    assert os.path.exists("dummy_pow.pha")
                    assert os.path.exists("dummy_pow.rsp")
                    assert os.path.exists("dummy_pow.txt")
                finally:
                    for ext in [".pha", ".rsp", ".txt"]:
                        if os.path.exists(f"dummy_pow{ext}"):
                            os.unlink(f"dummy_pow{ext}")


class TestCovarianceIO(BaseTestIO):
    variant = "cov"


class TestComplexCovarianceIO(BaseTestIO):
    variant = "complcov"


class TestRmsIO(BaseTestIO):
    variant = "rms"


class TestLagIO(BaseTestIO):
    variant = "lag"


@pytest.mark.slow
class TestLagEnergySpectrum(object):
    @classmethod
    def setup_class(cls):
        from ..simulator import Simulator

        dt = 0.01
        cls.time_lag = 5.0
        data = Table.read(os.path.join(datadir, "sample_variable_series.fits"))["data"]
        flux = data
        times = np.arange(data.size) * dt
        maxfreq = 0.15 / cls.time_lag
        roll_amount = int(cls.time_lag // dt)
        good = slice(roll_amount, roll_amount + int(200 // dt))

        # When rolling, a positive delay is introduced
        rolled_flux = np.array(np.roll(flux, roll_amount))
        times, flux, rolled_flux = times[good], flux[good], rolled_flux[good]

        length = times[-1] - times[0]
        test_ref = Lightcurve(times, flux, err_dist="gauss", dt=dt, skip_checks=True)
        test_sub = Lightcurve(test_ref.time, rolled_flux, err_dist=test_ref.err_dist, dt=dt)
        test_ref_ev, test_sub_ev = EventList(), EventList()
        test_ref_ev.simulate_times(test_ref)
        test_sub_ev.simulate_times(test_sub)

        test_sub_ev.energy = np.random.uniform(0.3, 9, len(test_sub_ev.time))
        test_ref_ev.energy = np.random.uniform(9, 12, len(test_ref_ev.time))

        cls.lag = LagEnergySpectrum(
            test_sub_ev,
            freq_interval=[maxfreq / 2, maxfreq],
            energy_spec=(0.3, 9, 1, "lin"),
            ref_band=[9, 12],
            bin_time=dt / 2,
            segment_size=length,
            events2=test_ref_ev,
        )

        # Make single event list
        test_ev = test_sub_ev.join(test_ref_ev)

        cls.lag_same = LagEnergySpectrum(
            test_ev,
            freq_interval=[0, maxfreq],
            energy_spec=(0.3, 9, 1, "lin"),
            ref_band=[9, 12],
            bin_time=dt / 2,
            segment_size=length,
        )

    def test_lagspectrum_values_and_errors(self):
        assert np.all(np.abs(self.lag.spectrum - self.time_lag) < 3 * self.lag.spectrum_error)

    def test_lagspectrum_values_and_errors_same(self):
        assert np.all(np.abs(self.lag_same.spectrum - self.time_lag) < 3 * self.lag.spectrum_error)

    def test_lagspectrum_invalid_warns(self):
        ev = EventList(time=[], energy=[], gti=self.lag.events1.gti)
        with pytest.warns(UserWarning) as record:
            lag = LagSpectrum(
                ev,
                [0.0, 0.5],
                (0.3, 9, 4, "lin"),
                [9, 12],
                bin_time=0.1,
                segment_size=30,
                events2=self.lag.events2,
            )

        assert np.all(np.isnan(lag.spectrum))
        assert np.all(np.isnan(lag.spectrum_error))

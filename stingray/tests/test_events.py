import importlib
import copy
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from astropy.io import fits
import astropy.units as u
from astropy.time import Time

from ..events import EventList
from ..lightcurve import Lightcurve

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

_HAS_XARRAY = importlib.util.find_spec("xarray") is not None
_HAS_PANDAS = importlib.util.find_spec("pandas") is not None
_HAS_H5PY = importlib.util.find_spec("h5py") is not None
_HAS_YAML = importlib.util.find_spec("yaml") is not None


class TestEvents(object):
    @classmethod
    def setup_class(self):
        np.random.seed(57239875)
        self.time = [0.5, 1.5, 2.5, 3.5]
        self.counts = [3000, 2000, 2200, 3600]
        self.counts_flat = [3000, 3000, 3000, 3000]
        self.spectrum = [[1, 2, 3, 4, 5, 6], [1000, 2040, 1000, 3000, 4020, 2070]]
        self.gti = np.asanyarray([[0, 4]])

    def test_warn_wrong_keywords(self):
        with pytest.warns(UserWarning) as record:
            _ = EventList(self.time, self.counts, gti=self.gti, bubu="settete")
        assert np.any(["Unrecognized keywords:" in r.message.args[0] for r in record])

    def test_warn_wrong_keywords_ncounts(self):
        with pytest.warns(DeprecationWarning, match="The ncounts keyword does nothing"):
            _ = EventList(self.time, self.counts, gti=self.gti, ncounts=10)

    def test_initiate_from_ndarray(self):
        times = np.sort(np.random.uniform(1e8, 1e8 + 1000, 101).astype(np.longdouble))
        ev = EventList(times, mjdref=54600)
        assert np.allclose(ev.time, times, atol=1e-15)
        assert np.allclose(ev.mjdref, 54600)

    def test_print(self):
        times = [1.01, 2, 3]
        ev = EventList(times, mjdref=54600)

        print(ev)

    def test_initiate_from_astropy_time(self):
        times = np.sort(np.random.uniform(1e8, 1e8 + 1000, 101).astype(np.longdouble))
        mjdref = 54600
        mjds = Time(mjdref + times / 86400, format="mjd")
        ev = EventList(mjds, mjdref=mjdref)
        assert np.allclose(ev.time, times, atol=1e-15)
        assert np.allclose(ev.mjdref, mjdref)

    def test_create_high_precision_object(self):
        times = np.sort(np.random.uniform(1e8, 1e8 + 1000, 101).astype(np.longdouble))
        ev = EventList(times, high_precision=True)
        assert np.allclose(ev.time, times, atol=1e-15)

    def test_inequal_length(self):
        """Check that exception is raised in case of
        disparity in length of 'time' and 'energy'
        """
        with pytest.raises(ValueError):
            EventList(time=[1, 2, 3], energy=[10, 12])

    def test_concatenate(self):
        """Join two overlapping event lists."""
        ev = EventList(time=[1, 1.1, 2, 3, 4], energy=[3, 4, 7, 4, 3], gti=[[1, 2], [3, 4]])
        ev_other1 = EventList(time=[5, 6, 6.1], energy=[4, 3, 8], gti=[[6, 6.2]])
        ev_other2 = EventList(time=[7, 10], energy=[1, 2], gti=[[6.5, 7]])
        ev_new = ev.concatenate([ev_other1, ev_other2])

        assert (ev_new.time == np.array([1, 1.1, 2, 3, 4, 5, 6, 6.1, 7, 10])).all()
        assert (ev_new.energy == np.array([3, 4, 7, 4, 3, 4, 3, 8, 1, 2])).all()
        assert (ev_new.gti == np.array([[1, 2], [3, 4], [6, 6.2], [6.5, 7]])).all()

    def test_to_lc(self):
        """Create a light curve from event list."""
        ev = EventList(self.time, gti=self.gti)
        lc = ev.to_lc(1)
        assert np.allclose(lc.time, [0.5, 1.5, 2.5, 3.5])
        assert (lc.gti == self.gti).all()

    def test_to_lc_intrinsic_dt(self):
        """Create a light curve from event list."""
        ev = EventList(self.time, gti=self.gti)
        ev.dt = np.pi / 3
        with pytest.warns(UserWarning, match="The input event list has a time resolution"):
            lc = ev.to_lc(1)
        assert lc.dt == np.pi / 3

        ev.dt = np.pi / 6
        with pytest.warns(UserWarning, match="The input event list has a time resolution"):
            lc = ev.to_lc(1)
        assert lc.dt == np.pi / 3

    def test_to_timeseries(self):
        """Create a time series from event list."""
        ev = EventList(self.time, gti=self.gti)
        ev.bla = np.zeros_like(ev.time) + 2
        lc = ev.to_lc(1)
        ts = ev.to_binned_timeseries(1)
        assert np.allclose(ts.time, [0.5, 1.5, 2.5, 3.5])
        assert (ts.gti == self.gti).all()
        assert np.array_equal(ts.counts, lc.counts)
        assert np.array_equal(ts.bla, ts.counts * 2)

    def test_from_lc(self):
        """Load event list from lightcurve"""
        lc = Lightcurve(time=[0.5, 1.5, 2.5], counts=[2, -1, 2])
        ev = EventList.from_lc(lc)

        assert np.array_equal(ev.time, np.array([0.5, 0.5, 2.5, 2.5]))

    def test_simulate_times_warns_bin_time(self):
        """Simulate photon arrival times for an event list
        from light curve.
        """
        lc = Lightcurve(self.time, self.counts_flat, gti=self.gti)
        ev = EventList()
        with pytest.warns(DeprecationWarning) as record:
            ev.simulate_times(lc, bin_time=lc.dt)
        assert np.any(["Bin time will be ignored" in r.message.args[0] for r in record])
        lc_sim = ev.to_lc(dt=lc.dt, tstart=lc.tstart, tseg=lc.tseg)
        assert np.all((lc - lc_sim).counts < 3 * np.sqrt(lc.counts))

    @pytest.mark.parametrize("use_spline", [True, False])
    def test_simulate_times(self, use_spline):
        """Simulate photon arrival times, with use_spline option
        enabled.
        """
        lc = Lightcurve(self.time, self.counts_flat, gti=self.gti)
        ev = EventList()
        ev.simulate_times(lc, use_spline=use_spline)
        lc_sim = ev.to_lc(dt=lc.dt, tstart=lc.tstart, tseg=lc.tseg)
        assert np.all((lc - lc_sim).counts < 3 * np.sqrt(lc.counts))

    def test_simulate_energies(self):
        """Assign photon energies to an event list."""
        ev = EventList(np.arange(10))
        ev.simulate_energies(self.spectrum)

    def test_simulate_energies_with_1d_spectrum(self):
        """Test that simulate_energies() method raises index
        error exception is spectrum is 1-d.
        """
        ev = EventList(np.arange(10))
        with pytest.raises(IndexError):
            ev.simulate_energies(self.spectrum[0])

    def test_simulate_energies_with_wrong_spectrum_type(self):
        """Test that simulate_energies() method raises type error
        exception when wrong spectrum type is supplied.
        """
        ev = EventList(np.arange(10))
        with pytest.raises(TypeError):
            ev.simulate_energies(1)

    def test_simulate_energies_with_counts_not_set(self):
        ev = EventList()
        with pytest.warns(UserWarning, match="empty event list"):
            ev.simulate_energies(self.spectrum)

    def test_compare_energy(self):
        """Compare the simulated energy distribution to actual distribution."""
        fluxes = np.array(self.spectrum[1])
        ev = EventList(np.arange(1000))
        ev.simulate_energies(self.spectrum)

        # Note: I'm passing the edges: when the bin center is 1, the
        # edge is at 0.5
        lc, _ = np.histogram(ev.energy, bins=np.arange(0.5, 7, 1))

        # Calculate probabilities and compare
        lc_prob = lc / float(sum(lc))
        fluxes_prob = fluxes / float(sum(fluxes))

        assert np.all(np.abs(lc_prob - fluxes_prob) < 3 * np.sqrt(fluxes_prob))
        assert np.all((ev.energy >= 0.5) & (ev.energy < 6.5))

    @pytest.mark.skipif("not (_HAS_YAML)")
    def test_io_with_ascii(self):
        ev = EventList(self.time)
        with pytest.warns(UserWarning, match=".* output does not serialize the metadata"):
            ev.write("ascii_ev.ecsv", fmt="ascii")
        ev = ev.read("ascii_ev.ecsv", fmt="ascii")
        assert np.allclose(ev.time, self.time)
        os.remove("ascii_ev.ecsv")

    def test_io_with_pickle(self):
        ev = EventList(self.time, mjdref=54000)
        ev.write("ev.pickle", fmt="pickle")
        ev = ev.read("ev.pickle", fmt="pickle")
        assert np.allclose(ev.time, self.time)
        os.remove("ev.pickle")

    @pytest.mark.skipif("not _HAS_H5PY")
    def test_io_with_hdf5_auto(self):
        ev = EventList(time=self.time, mjdref=54000)
        ev.write("ev.hdf5")

        ev = ev.read("ev.hdf5")
        assert np.allclose(ev.time, self.time)
        os.remove("ev.hdf5")

    @pytest.mark.skipif("not _HAS_H5PY")
    def test_io_with_hdf5(self):
        ev = EventList(time=self.time, mjdref=54000)
        ev.write("ev.hdf5", fmt="hdf5")

        ev = ev.read("ev.hdf5", fmt="hdf5")
        assert np.allclose(ev.time, self.time)
        os.remove("ev.hdf5")

    def test_io_with_fits(self):
        ev = EventList(time=self.time, mjdref=54000)
        with pytest.warns(UserWarning, match=".* output does not serialize the metadata"):
            ev.write("ev.fits", fmt="fits")
        ev = ev.read("ev.fits", fmt="fits")
        assert np.allclose(ev.time, self.time)
        os.remove("ev.fits")

    def test_calibrate_directly_warns(self):
        rmf_file = os.path.join(datadir, "test.rmf")
        with pytest.warns(UserWarning, match="PI channels must be provided"):
            EventList(time=self.time, mjdref=54000, rmf_file=rmf_file)

    def test_calibrate_directly(self):
        rmf_file = os.path.join(datadir, "test.rmf")
        pis = np.random.randint(0, 1000, np.size(self.time))
        ev1 = EventList(time=self.time, pi=pis, mjdref=54000, rmf_file=rmf_file)
        ev2 = EventList(time=self.time, pi=pis, mjdref=54000)
        ev2.convert_pi_to_energy(rmf_file)

        assert np.array_equal(ev1.energy, ev2.energy)

    def test_fits_standard(self):
        """Test that fits works with a standard event list
        file.
        """
        fname = os.path.join(datadir, "monol_testA.evt")
        ev = EventList()
        ev = ev.read(fname, fmt="hea")
        assert np.isclose(ev.mjdref, 55197.00076601852)

    def test_fits_with_standard_file_and_calibrate_directly(self):
        """Test that fits works and calibration works."""
        fname = os.path.join(datadir, "monol_testA.evt")
        rmf_file = os.path.join(datadir, "test.rmf")
        ev1 = EventList()
        ev1 = ev1.read(fname, fmt="hea")
        ev2 = EventList()
        ev2 = ev2.read(fname, fmt="hea", rmf_file=rmf_file)
        ev1.convert_pi_to_energy(rmf_file)
        assert np.array_equal(ev1.energy, ev2.energy)

    @pytest.mark.parametrize(
        "fname",
        ["xte_test.evt.gz", "xte_gx_test.evt.gz", "chandra_test.fits", "chandra_noener_test.fits"],
    )
    def test_event_file_read(self, fname):
        """Test event file reading."""
        fname = os.path.join(datadir, fname)
        EventList.read(fname, fmt="hea")

    def test_event_file_read_laxpc(self):
        """Test LAXPC file reading"""
        fname = os.path.join(datadir, "laxpc_file_read.fits")
        with pytest.warns(UserWarning, match="HDU EVENTS"):
            with pytest.warns(UserWarning, match="No valid GTI"):
                ev = EventList.read(fname, fmt="hea")
        assert ev.instr.lower() == "laxpc3"
        assert ev.mission.lower() == "astrosat"
        assert ((ev.filter_detector_id(detector_id=1)).main_array_length) == 453
        assert ((ev.filter_detector_id(detector_id=[1])).main_array_length) == 453

    def test_fits_with_additional(self):
        """Test that fits works with a standard event list
        file.
        """
        fname = os.path.join(datadir, "xmm_test.fits")
        ev = EventList()
        with pytest.warns(UserWarning, match="HDU EVENTS not found"):
            ev = ev.read(fname, fmt="hea", additional_columns=["PRIOR"])
        assert hasattr(ev, "prior")

    def test_timeseries_empty_evts(self):
        N = len(self.time)
        ev = EventList()
        ts = ev.to_astropy_timeseries()
        assert len(ts.columns) == 0

    def test_timeseries_roundtrip(self):
        N = len(self.time)
        ev = EventList(
            time=self.time,
            gti=self.gti,
            energy=np.zeros(N),
            pi=np.ones(N),
            mission="BUBU",
            instr="BABA",
            mjdref=53467.0,
        )
        ts = ev.to_astropy_timeseries()
        new_ev = ev.from_astropy_timeseries(ts)
        for attr in ["time", "energy", "pi", "gti"]:
            assert np.allclose(getattr(ev, attr), getattr(new_ev, attr))
        for attr in ["mission", "instr", "mjdref"]:
            assert getattr(ev, attr) == getattr(new_ev, attr)

    def test_table_roundtrip(self):
        N = len(self.time)
        ev = EventList(
            time=self.time,
            gti=self.gti,
            energy=np.zeros(N),
            pi=np.ones(N),
            mission="BUBU",
            instr="BABA",
            mjdref=53467.0,
        )
        ts = ev.to_astropy_table()
        new_ev = ev.from_astropy_table(ts)
        for attr in ["time", "energy", "pi", "gti"]:
            assert np.allclose(getattr(ev, attr), getattr(new_ev, attr))
        for attr in ["mission", "instr", "mjdref"]:
            assert getattr(ev, attr) == getattr(new_ev, attr)

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip(self):
        N = len(self.time)
        ev = EventList(
            time=self.time,
            gti=self.gti,
            energy=np.zeros(N),
            pi=np.ones(N),
            mission="BUBU",
            instr="BABA",
            mjdref=53467.0,
        )
        ts = ev.to_xarray()
        new_ev = ev.from_xarray(ts)
        for attr in ["time", "energy", "pi", "gti"]:
            assert np.allclose(getattr(ev, attr), getattr(new_ev, attr))
        for attr in ["mission", "instr", "mjdref"]:
            assert getattr(ev, attr) == getattr(new_ev, attr)

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip(self):
        N = len(self.time)
        ev = EventList(
            time=self.time,
            gti=self.gti,
            energy=np.zeros(N),
            pi=np.ones(N),
            mission="BUBU",
            instr="BABA",
            mjdref=53467.0,
        )
        ts = ev.to_pandas()
        new_ev = ev.from_pandas(ts)
        for attr in ["time", "energy", "pi", "gti"]:
            assert np.allclose(getattr(ev, attr), getattr(new_ev, attr))
        for attr in ["mission", "instr", "mjdref"]:
            assert getattr(ev, attr) == getattr(new_ev, attr)


class TestEventsFermi(object):
    @classmethod
    def setup_class(self):
        self.fermi_file = os.path.join(datadir, "fermi_gbm_tte.fits")
        with fits.open(self.fermi_file) as hdul:
            self.header = hdul[0].header
            self.time = hdul[2].data["TIME"]
            self.gti = hdul[3].data[0]

            self.pi = hdul[2].data["PHA"]
            elower = hdul[1].data["E_MIN"]
            ehigher = hdul[1].data["E_MAX"]
            emid = elower + (ehigher - elower) / 2.0
            self.emin, self.emax = np.min(elower), np.max(ehigher)
            self.energy = np.array([emid[c] for c in self.pi])

    def test_read_fermi(self):
        assert EventList.read(self.fermi_file) is not None

    def test_check_energy_pi(self):
        evt = EventList.read(self.fermi_file)

        assert_allclose(evt.energy, self.energy, atol=1e-8)
        assert_array_equal(evt.pi, self.pi)

    def test_high_precision(self):
        evt = EventList.read(self.fermi_file, high_precision=True)

        assert np.issubdtype(evt.time.dtype, np.float128)
        assert np.issubdtype(evt.pi.dtype, np.float128)

    @pytest.mark.parametrize(
        "emin, emax",
        [
            (2, 45.2),
            (2 * u.keV, 45.2 * u.keV),
            (2000 * u.eV, 45.2 * u.keV),
            (None, 45.2 * u.keV),
            (2000 * u.eV, None),
            (None, None),
        ],
    )
    def test_mask_ebound(self, emin, emax):
        evt = EventList.read(self.fermi_file, emin=emin, emax=emax)

        emin = emin if emin is not None else self.emin
        emax = emax if emax is not None else self.emax

        if isinstance(emin, u.Quantity):
            emin = emin.to(u.keV).value
        if isinstance(emax, u.Quantity):
            emax = emax.to(u.keV).value

        mask = (self.energy > emin) & (self.energy < emax)
        energy = self.energy[mask]
        time = self.time[mask]
        pi = self.pi[mask]

        assert_array_equal(evt.time, time)
        assert_array_equal(evt.pi, pi)
        assert_array_equal(evt.energy, energy)

    def test_check_time_gti(self):
        evt = EventList.read(self.fermi_file)

        assert_allclose(evt.time, self.time, atol=1e-8)
        assert_array_equal(evt.gti, np.array(self.gti).reshape(1, 2))
        assert_equal(evt.ncounts, self.time.shape[0])
        assert_allclose(evt.mjdref, self.header["MJDREFI"])

    def test_kwargs(self):
        with pytest.warns(UserWarning, match="Unrecognized keywords:"):
            EventList.read(self.fermi_file, alpha=1)


class TestJoinEvents:
    def test_join_without_times_simulated(self):
        """Test if exception is raised when join method is
        called before first simulating times.
        """
        ev = EventList()
        ev_other = EventList()

        with pytest.warns(UserWarning, match="One of the time series you are joining is empty."):
            assert ev.join(ev_other, strategy="union").time is None

    def test_join_empty_lists(self):
        """Test if an empty event list can be concatenated
        with a non-empty event list.
        """
        ev = EventList(time=[1, 2, 3])
        ev_other = EventList()
        with pytest.warns(UserWarning, match="One of the time series you are joining is empty."):
            ev_new = ev.join(ev_other, strategy="union")
        assert np.allclose(ev_new.time, [1, 2, 3])

        ev = EventList()
        ev_other = EventList(time=[1, 2, 3])
        ev_new = ev.join(ev_other, strategy="union")
        assert np.allclose(ev_new.time, [1, 2, 3])

        ev = EventList()
        ev_other = EventList()
        with pytest.warns(UserWarning, match="One of the time series you are joining is empty."):
            ev_new = ev.join(ev_other, strategy="union")
        assert ev_new.time is None
        assert ev_new.gti is None
        assert ev_new.pi is None
        assert ev_new.energy is None

        ev = EventList(time=[1, 2, 3])
        ev_other = EventList([])
        with pytest.warns(UserWarning, match="One of the time series you are joining is empty."):
            ev_new = ev.join(ev_other, strategy="union")
        assert np.allclose(ev_new.time, [1, 2, 3])

        ev = EventList([])
        ev_other = EventList(time=[1, 2, 3])
        ev_new = ev.join(ev_other, strategy="union")
        assert np.allclose(ev_new.time, [1, 2, 3])

    def test_join_different_dt(self):
        ev = EventList(time=[10, 20, 30], dt=1)
        ev_other = EventList(time=[40, 50, 60], dt=3)
        with pytest.warns(UserWarning, match="The time resolution is different."):
            ev_new = ev.join(ev_other, strategy="union")

        assert np.array_equal(ev_new.dt, [1, 1, 1, 3, 3, 3])
        assert np.allclose(ev_new.time, [10, 20, 30, 40, 50, 60])

    def test_join_different_instr(self):
        ev = EventList(time=[10, 20, 30], instr="fpma")
        ev_other = EventList(time=[40, 50, 60], instr="fpmb")
        with pytest.warns(
            UserWarning,
            match="Attribute instr is different in the time series being merged.",
        ):
            ev_new = ev.join(ev_other, strategy="union")

        assert ev_new.instr == "fpma,fpmb"

    def test_join_different_meta_attribute(self):
        ev = EventList(time=[10, 20, 30])
        ev_other = EventList(time=[40, 50, 60])
        ev_other.bubu = "settete"
        ev.whatstheanswer = 42
        ev.unmovimentopara = "arriba"
        ev_other.unmovimentopara = "abajo"

        with pytest.warns(
            UserWarning,
            match=(
                "Attribute (bubu|whatstheanswer|unmovimentopara) is different "
                "in the time series being merged."
            ),
        ):
            ev_new = ev.join(ev_other, strategy="union")

        assert ev_new.bubu == (None, "settete")
        assert ev_new.whatstheanswer == (42, None)
        assert ev_new.unmovimentopara == "arriba,abajo"

    def test_join_without_energy(self):
        ev = EventList(time=[1, 2, 3], energy=[3, 3, 3])
        ev_other = EventList(time=[4, 5])
        with pytest.warns(
            UserWarning, match="The energy array is empty in one of the time series being merged."
        ):
            ev_new = ev.join(ev_other, strategy="union")

        assert np.allclose(ev_new.energy, [3, 3, 3, np.nan, np.nan], equal_nan=True)

    def test_join_without_pi(self):
        ev = EventList(time=[1, 2, 3], pi=[3, 3, 3])
        ev_other = EventList(time=[4, 5])
        with pytest.warns(
            UserWarning, match="The pi array is empty in one of the time series being merged."
        ):
            ev_new = ev.join(ev_other, strategy="union")

        assert np.allclose(ev_new.pi, [3, 3, 3, np.nan, np.nan], equal_nan=True)

    def test_join_with_arbitrary_attribute(self):
        ev = EventList(time=[1, 2, 4])
        ev_other = EventList(time=[3, 5])
        ev.u = [3, 3, 3]
        ev_other.q = [1, 2]
        with pytest.warns(
            UserWarning, match="The (u|q) array is empty in one of the time series being merged."
        ):
            ev_new = ev.join(ev_other, strategy="union")

        assert np.allclose(ev_new.q, [np.nan, np.nan, 1, np.nan, 2], equal_nan=True)
        assert np.allclose(ev_new.u, [3, 3, np.nan, 3, np.nan], equal_nan=True)

    def test_join_with_gti_none(self):
        ev = EventList(time=[1, 2, 3])
        ev_other = EventList(time=[4, 5], gti=[[3.5, 5.5]])
        ev_new = ev.join(ev_other, strategy="union")

        assert np.allclose(ev_new.gti, [[1, 3], [3.5, 5.5]])

        ev = EventList(time=[1, 2, 3], gti=[[0.5, 3.5]])
        ev_other = EventList(time=[4, 5])
        ev_new = ev.join(ev_other, strategy="union")

        assert np.allclose(ev_new.gti, [[0.5, 3.5], [4, 5]])

        ev = EventList(time=[1, 2, 3])
        ev_other = EventList(time=[4, 5])
        ev_new = ev.join(ev_other, strategy="union")

        assert ev_new._gti is None

    def test_non_overlapping_join_infer(self):
        """Join two overlapping event lists."""
        ev = EventList(time=[1, 1.1, 2, 3, 4], energy=[3, 4, 7, 4, 3], gti=[[1, 2], [3, 4]])
        ev_other = EventList(time=[5, 6, 6.1, 7, 10], energy=[4, 3, 8, 1, 2], gti=[[6, 7]])
        ev_new = ev.join(ev_other, strategy="infer")

        assert (ev_new.time == np.array([1, 1.1, 2, 3, 4, 5, 6, 6.1, 7, 10])).all()
        assert (ev_new.energy == np.array([3, 4, 7, 4, 3, 4, 3, 8, 1, 2])).all()
        assert (ev_new.gti == np.array([[1, 2], [3, 4], [6, 7]])).all()

    def test_overlapping_join_infer(self):
        """Join two non-overlapping event lists."""
        with pytest.warns(UserWarning, match="The time array is not sorted."):
            ev = EventList(time=[1, 1.1, 10, 6, 5], energy=[10, 6, 3, 11, 2], gti=[[1, 3], [5, 6]])
        with pytest.warns(UserWarning, match="The time array is not sorted."):
            ev_other = EventList(
                time=[5.1, 7, 6.1, 6.11, 10.1], energy=[2, 3, 8, 1, 2], gti=[[5, 7], [8, 10]]
            )
        ev_new = ev.join(ev_other, strategy="infer")

        assert (ev_new.time == np.array([1, 1.1, 5, 5.1, 6, 6.1, 6.11, 7, 10, 10.1])).all()
        assert (ev_new.energy == np.array([10, 6, 2, 2, 11, 8, 1, 3, 3, 2])).all()
        assert (ev_new.gti == np.array([[5, 6]])).all()

    def test_overlapping_join_change_mjdref(self):
        """Join two non-overlapping event lists."""
        with pytest.warns(UserWarning, match="The time array is not sorted."):
            ev = EventList(
                time=[1, 1.1, 10, 6, 5],
                energy=[10, 6, 3, 11, 2],
                gti=[[1, 3], [5, 6]],
                mjdref=57001,
            )
        with pytest.warns(UserWarning, match="The time array is not sorted."):
            ev_other = EventList(
                time=np.asanyarray([5.1, 7, 6.1, 6.11, 10.1]) + 86400,
                energy=[2, 3, 8, 1, 2],
                gti=np.asanyarray([[5, 7], [8, 10]]) + 86400,
                mjdref=57000,
            )
        with pytest.warns(UserWarning, match="Attribute mjdref is different"):
            ev_new = ev.join(ev_other, strategy="intersection")

        assert np.allclose(ev_new.time, np.array([1, 1.1, 5, 5.1, 6, 6.1, 6.11, 7, 10, 10.1]))
        assert (ev_new.energy == np.array([10, 6, 2, 2, 11, 8, 1, 3, 3, 2])).all()
        assert np.allclose(ev_new.gti, np.array([[5, 6]]))

    def test_multiple_join(self):
        """Test if multiple event lists can be joined."""
        ev = EventList(time=[1, 2, 4], instr="a", mission=1)
        ev_other = EventList(time=[3, 5, 7], instr="b", mission=2)
        ev_other2 = EventList(time=[6, 8, 9], instr="c", mission=3)

        ev.pibiri = [1, 1, 1]
        ev_other.pibiri = [2, 2, 2]
        ev_other2.pibiri = [3, 3, 3]

        with pytest.warns(
            UserWarning,
            match="Attribute (instr|mission) is different in the time series being merged.",
        ):
            ev_new = ev.join([ev_other, ev_other2], strategy="union")
        assert np.allclose(ev_new.time, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert np.allclose(ev_new.pibiri, [1, 1, 2, 1, 2, 3, 2, 3, 3])
        assert ev_new.instr == "a,b,c"
        assert ev_new.mission == (1, 2, 3)


class TestFilters(object):
    @classmethod
    def setup_class(cls):
        events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
        events = EventList(events, gti=[[0, 3.3]])
        events.pi = np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
        events.energy = np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
        events.mjdref = 10
        cls.events = events

    @pytest.mark.parametrize("inplace", [True, False])
    def test_apply_mask(self, inplace):
        events = copy.deepcopy(self.events)
        mask = [True, False, False, False, False, True, True, True, False, True]
        filt_events = events.apply_mask(mask, inplace=inplace)
        if inplace:
            assert filt_events is events
            assert np.allclose(events.pi, 1)
        else:
            assert filt_events is not events
            assert not np.allclose(events.pi, 1)

        expected = np.array([1, 2, 2.2, 3, 3.2])
        assert np.allclose(filt_events.time, expected)
        assert np.allclose(filt_events.pi, 1)
        assert np.allclose(filt_events.energy, 1)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("use_pi", [True, False])
    def test_filter_energy_range(self, inplace, use_pi):
        events = copy.deepcopy(self.events)

        filt_events = events.filter_energy_range([0.5, 1.5], use_pi=use_pi, inplace=inplace)
        if inplace:
            assert filt_events is events
            assert np.allclose(events.pi, 1)
        else:
            assert filt_events is not events
            assert not np.allclose(events.pi, 1)

        expected = np.array([1, 2, 2.2, 3, 3.2])
        assert np.allclose(filt_events.time, expected)
        assert np.allclose(filt_events.pi, 1)
        assert np.allclose(filt_events.energy, 1)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_apply_deadtime(self, inplace):
        events = copy.deepcopy(self.events)
        filt_events, _ = events.apply_deadtime(
            0.11, inplace=inplace, verbose=False, return_all=True
        )
        if inplace:
            assert filt_events is events
            assert np.allclose(events.pi, 1)
        else:
            assert filt_events is not events
            assert not np.allclose(events.pi, 1)

        expected = np.array([1, 2, 2.2, 3, 3.2])
        assert np.allclose(filt_events.time, expected)
        assert np.allclose(filt_events.pi, 1)
        assert np.allclose(filt_events.energy, 1)


class TestColors(object):
    @classmethod
    def setup_class(cls):
        cls.events = EventList(
            time=np.arange(100000) + 0.5, energy=np.random.choice([2, 5], 100000), gti=[[0, 100000]]
        )

    def test_bad_interval_color(self):
        with pytest.raises(ValueError, match=" 2x2 array"):
            self.events.get_color_evolution([[0, 3], [4, 6], [7, 8]], 10000)
        with pytest.raises(ValueError, match=" 2x2 array"):
            self.events.get_color_evolution([[0, 3, 8]], 10000)
        with pytest.raises(ValueError, match=" 2x2 array"):
            self.events.get_color_evolution([0], 10000)
        with pytest.raises(ValueError, match=" 2x2 array"):
            self.events.get_color_evolution([[0, 1]], 10000)

    def test_bad_interval_intensity(self):
        with pytest.raises(ValueError, match="2-element list"):
            self.events.get_intensity_evolution([[0, 3], [4, 6], [7, 8]], 10000)
        with pytest.raises(ValueError, match="2-element list"):
            self.events.get_intensity_evolution([[0, 3, 8]], 10000)
        with pytest.raises(ValueError, match="2-element list"):
            self.events.get_intensity_evolution([0], 10000)
        with pytest.raises(ValueError, match="2-element list"):
            self.events.get_intensity_evolution([[0, 1]], 10000)

    def test_colors(self):
        start, stop, colors, color_errs = self.events.get_color_evolution([[0, 3], [4, 6]], 10000)
        # 5000 / 5000 = 1
        # 2 x sqrt(5000) / 5000 = 0.0282
        assert np.allclose(colors, 1, rtol=0.1)
        assert np.allclose(color_errs, 0.0282, atol=0.003)
        assert np.allclose(start, np.arange(10) * 10000)
        assert np.allclose(stop, np.arange(1, 11) * 10000)

    def test_colors_missing_energies(self):
        events = copy.deepcopy(self.events)
        events.filter_energy_range([0, 3], inplace=True)
        with pytest.warns(UserWarning, match="No counts in one of the energy ranges"):
            events.get_color_evolution([[0, 3], [4, 6]], 10000)

    def test_colors_no_segment(self):
        start, stop, colors, color_errs = self.events.get_color_evolution([[0, 3], [4, 6]])
        # 50000 / 50000 = 1
        # 2 x sqrt(50000) / 50000 = 0.0089
        assert np.allclose(colors, 1, rtol=0.1)
        assert np.allclose(color_errs, 0.0089, atol=0.001)
        assert np.allclose(start, 0)
        assert np.allclose(stop, 100000)

    def test_intensity(self):
        start, stop, rate, rate_errs = self.events.get_intensity_evolution([0, 6], 10000)

        assert np.allclose(rate, 1, rtol=0.1)
        assert np.allclose(rate_errs, 0.01, atol=0.003)
        assert np.allclose(start, np.arange(10) * 10000)
        assert np.allclose(stop, np.arange(1, 11) * 10000)

    def test_intensity_no_segment(self):
        start, stop, rate, rate_errs = self.events.get_intensity_evolution([0, 6])

        assert np.allclose(rate, 1, rtol=0.1)
        assert np.allclose(rate_errs, 0.003, atol=0.001)
        assert np.allclose(start, 0)
        assert np.allclose(stop, 100000)

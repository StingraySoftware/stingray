import numpy as np
import os
import matplotlib.pyplot as plt

import pytest
from astropy.utils.exceptions import AstropyUserWarning

from ..io import split_numbers
from ..io import ref_mjd
from ..io import high_precision_keyword_read
from ..io import load_events_and_gtis, read_mission_info
from ..io import read_header_key, FITSTimeseriesReader, DEFAULT_FORMAT
from ..events import EventList

import warnings

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

_H5PY_INSTALLED = True

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False

skip_condition = pytest.mark.skipif(not _H5PY_INSTALLED, reason="H5PY not installed.")


class TestIO(object):
    def test_common_name(self):
        """Test the common_name function."""
        from ..io import common_name

        a = "A_3-50_A"
        b = "B_3-50_B"
        assert common_name(a, b) == "3-50"

    def test_high_precision_keyword(self):
        """Test high precision FITS keyword read."""
        hdr = {"MJDTESTI": 100, "MJDTESTF": np.longdouble(0.5), "CIAO": np.longdouble(0.0)}
        assert high_precision_keyword_read(hdr, "MJDTEST") == np.longdouble(
            100.5
        ), "Keyword MJDTEST read incorrectly"
        assert high_precision_keyword_read(hdr, "MJDTESTA") == np.longdouble(
            100.5
        ), "Keyword MJDTESTA read incorrectly"
        assert high_precision_keyword_read(hdr, "CIAO") == np.longdouble(
            0.0
        ), "Keyword CIAO read incorrectly"
        assert high_precision_keyword_read(hdr, "BU") is None, "Inexistent key read incorrectly"

    def test_xselect_mdb_is_found_headas(self, monkeypatch, tmp_path):
        """Test event file reading."""
        path = tmp_path / "bin"
        path.mkdir()
        f = path / "xselect.mdb"
        f.write_text("MAXI:submkey       NONE\nMAXI:instkey       INSTRUME")

        monkeypatch.setenv("HEADAS", str(tmp_path))

        info = read_mission_info()
        assert "NUSTAR" not in info

    def test_read_whole_mission_info(self):
        """Test event file reading."""
        info = read_mission_info()
        assert "NUSTAR" in info
        assert "XMM" in info
        assert "NICER" in info

    def test_event_file_read_and_automatic_sort(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "monol_testA_calib.evt")
        with pytest.warns(AstropyUserWarning, match="No valid GTI extensions"):
            evdata = load_events_and_gtis(fname)
        fname_unsrt = os.path.join(datadir, "monol_testA_calib_unsrt.evt")
        with pytest.warns(UserWarning, match="not sorted. Sorting them for you"):
            with pytest.warns(AstropyUserWarning, match="No valid GTI extensions"):
                evdata_unsrt = load_events_and_gtis(fname_unsrt)

        for attr in "ev_list", "energy_list", "pi_list":
            assert np.allclose(getattr(evdata, attr), getattr(evdata_unsrt, attr))

    def test_event_file_read_additional_warns_uncal(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "monol_testA.evt")
        with pytest.warns(UserWarning) as record:
            vals = load_events_and_gtis(fname, additional_columns=["energy"])
        assert np.any(["Column energy not found" in r.message.args[0] for r in record])
        # This is the default calibration for nustar data, as returned
        # from rough_calibration
        assert np.allclose(vals.energy_list, vals.pi_list * 0.04 + 1.62)

    def test_event_file_read_additional_energy_cal(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "monol_testA_calib.evt")
        with pytest.warns(AstropyUserWarning, match="No valid GTI extensions"):
            vals = load_events_and_gtis(fname, additional_columns=["energy"])
        # These energies were calibrated with a different calibration than
        # returned from rough_calibration, on purpose! (notice the +1.)
        assert np.allclose(vals.energy_list, vals.pi_list * 0.04 + 1.6 + 1.0)

    def test_event_file_read_xmm(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "xmm_test.fits")
        with pytest.warns(UserWarning) as record:
            load_events_and_gtis(fname, additional_columns=["PRIOR"])
        assert np.any(["Trying first extension" in r.message.args[0] for r in record])

    def test_event_file_read_rxte(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "xte_test.evt.gz")
        data = load_events_and_gtis(fname)
        assert data.mission.lower() == "xte"

    def test_event_file_read_rxte_gx(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "xte_gx_test.evt.gz")
        data = load_events_and_gtis(fname)
        assert data.mission.lower() == "xte"
        assert np.min(data.pi_list) == 2
        assert np.max(data.pi_list) == 255
        assert np.allclose(data.energy_list.min(), 2.06)
        assert np.allclose(data.energy_list.max(), 117.86)

    def test_event_file_read_chandra(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "chandra_test.fits")
        data = load_events_and_gtis(fname)
        assert data.mission.lower() == "axaf"
        assert hasattr(data, "energy_list")
        assert data.energy_list.max() < 18

    def test_event_file_read_chandra_noE(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "chandra_noener_test.fits")
        data = load_events_and_gtis(fname)
        assert data.mission.lower() == "axaf"
        assert hasattr(data, "energy_list")
        assert data.energy_list.max() < 18

    def test_event_file_read_laxpc(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "laxpc_file_read.fits")
        with pytest.warns(UserWarning, match="No valid GTI*"):
            data = load_events_and_gtis(fname, additional_columns=["Layer"], hduname="event file")
        assert data.mission.lower() == "astrosat"
        assert read_header_key(fname, "INSTRUME", hdu=0) == "LAXPC3"

    def test_event_file_read_no_mission(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "nomission.evt")
        with pytest.warns(UserWarning, match="Sorting them"):
            load_events_and_gtis(fname)

    def test_event_file_read_no_additional(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "monol_testA.evt")
        load_events_and_gtis(fname)

    def test_event_file_read_no_pi(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "monol_testA.evt")
        load_events_and_gtis(fname)

    def test_read_header_key(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "monol_testA.evt")
        assert read_header_key(fname, "INSTRUME") == "FPMA"
        assert read_header_key(fname, "BU") == ""

    def test_read_mjdref(self):
        """Test event file reading."""
        fname = os.path.join(datadir, "monol_testA.evt")
        assert ref_mjd(fname) is not None

    def test_split_number(self):
        """Test split with high precision numbers."""
        numbers = np.array(
            [57401.0000003423423400453453, 0.00000574010000003426646], dtype=np.longdouble
        )
        number_I, number_F = split_numbers(numbers)
        r_numbers = np.longdouble(number_I) + np.longdouble(number_F)

        assert (numbers == r_numbers).all()

        n = [1234.567, 12.345]
        shift = -2
        n_i, n_f = split_numbers(n, shift)
        assert np.allclose(n_i, [1200, 0])
        r_n = n_i + n_f
        assert (n == r_n).all()


class TmpIOReadWrite(object):
    """A temporary helper class to test all the read and write functions."""

    def __init__(self):
        self.number = 10
        self.str = "Test"
        self.list = [1, 2, 3]
        self.array = np.array([1, 2, 3])
        self.long_number = np.longdouble(1.25)
        self.long_array = np.longdouble([1, 2, 3])

    def test_operation(self):
        return self.number * 10


class TestFileFormats(object):
    def test_savefig_without_plot(self):
        from ..io import savefig

        plt.close("all")
        with pytest.warns(UserWarning, match="plot the image first"):
            savefig("test.png")
        os.unlink("test.png")

    def test_savefig(self):
        from ..io import savefig

        plt.plot([1, 2, 3])
        savefig("test.png")
        os.unlink("test.png")


class TestCalibrate(object):
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")

        cls.rmf = os.path.join(cls.datadir, "test.rmf")

    def test_calibrate_spectrum(self):
        from ..io import pi_to_energy

        pis = np.array([1, 2, 3])
        energies = pi_to_energy(pis, self.rmf)
        assert np.allclose(energies, [1.66, 1.70, 1.74])


class TestFITSTimeseriesReader(object):
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")
        cls.fname = os.path.join(datadir, "monol_testA.evt")

    def test_read_fits_timeseries_bad_kind(self):
        with pytest.raises(
            NotImplementedError, match="Only events are supported by FITSTimeseriesReader"
        ):
            FITSTimeseriesReader(self.fname, output_class="bubu", data_kind="BAD_KIND")

    def test_read_fits_timeseries(self):
        reader = FITSTimeseriesReader(self.fname, output_class=EventList)
        # Full slice
        all_ev = reader[:]
        assert np.all((all_ev.time > 80000000) & (all_ev.time < 80001024))

    @pytest.mark.parametrize("root_file_name", [None, "test"])
    @pytest.mark.parametrize("gti_kind", ["same", "one", "multiple"])
    def test_read_apply_gti_lists(self, root_file_name, gti_kind):
        reader = FITSTimeseriesReader(self.fname, output_class=EventList)
        if gti_kind == "same":
            gti_list = [reader.gti]
        elif gti_kind == "one":
            gti_list = [[[80000000, 80001024]]]
        elif gti_kind == "multiple":
            gti_list = [[[80000000, 80000512]], [[80000513, 80001024]]]

        evs = list(reader.apply_gti_lists(gti_list, root_file_name=root_file_name))

        # Check that the number of event lists is the same as the number of GTI lists we input
        assert len(evs) == len(gti_list)

        # If the root_file_name is not None, read the event lists and delete the file(s)
        if root_file_name is not None:
            ev_str = evs
            evs = [EventList.read(ev, fmt=DEFAULT_FORMAT) for ev in ev_str]
            for ev in ev_str:
                os.unlink(ev)

        for i, ev in enumerate(evs):
            # Check that the gtis of the output event lists are the same we input
            assert np.allclose(ev.gti, gti_list[i])

    def test_read_apply_gti_lists_ignore_empty(self):
        reader = FITSTimeseriesReader(self.fname, output_class=EventList)
        gti_list = [[], [[80000000, 80000512]], [[80000513, 80001024]]]
        evs = list(reader.apply_gti_lists(gti_list))
        assert np.allclose(evs[0].gti, gti_list[1])
        assert np.allclose(evs[1].gti, gti_list[2])

    def test_read_fits_timeseries_by_nsamples(self):
        reader = FITSTimeseriesReader(self.fname, output_class=EventList)
        # Full slice
        outfnames = list(reader.split_by_number_of_samples(500, root_file_name="test"))
        assert len(outfnames) == 2
        ev0 = EventList.read(outfnames[0], fmt=DEFAULT_FORMAT)
        ev1 = EventList.read(outfnames[1], fmt=DEFAULT_FORMAT)
        assert np.all(ev0.time < 80000512.5)
        assert np.all(ev1.time > 80000512.5)
        for fname in outfnames:
            os.unlink(fname)

    def test_read_fits_timeseries_by_time_intv(self):
        reader = FITSTimeseriesReader(self.fname, output_class=EventList)
        # Full slice
        outfnames = list(
            reader.filter_at_time_intervals([80000100, 80001100], root_file_name="test")
        )
        assert len(outfnames) == 1
        ev0 = EventList.read(outfnames[0], fmt=DEFAULT_FORMAT)
        assert np.all((ev0.time > 80000100) & (ev0.time < 80001100))
        assert np.all((ev0.gti >= 80000100) & (ev0.gti < 80001100))
        for fname in outfnames:
            os.unlink(fname)

    def test_read_fits_timeseries_by_nsamples_generator(self):
        reader = FITSTimeseriesReader(self.fname, output_class=EventList)
        # Full slice
        ev0, ev1 = list(reader.split_by_number_of_samples(500))

        assert np.all(ev0.time < 80000512.5)
        assert np.all(ev1.time > 80000512.5)

    def test_read_fits_timeseries_by_time_intv_generator(self):
        reader = FITSTimeseriesReader(self.fname, output_class=EventList)
        # Full slice
        evs = list(reader.filter_at_time_intervals([80000100, 80001100]))
        assert len(evs) == 1
        ev0 = evs[0]
        assert np.all((ev0.time > 80000100) & (ev0.time < 80001100))
        assert np.all((ev0.gti >= 80000100) & (ev0.gti < 80001100))

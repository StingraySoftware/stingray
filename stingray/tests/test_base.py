import os
import importlib
import copy
import pytest
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from stingray.base import StingrayObject, StingrayTimeseries
from stingray.io import FITSTimeseriesReader

_HAS_XARRAY = importlib.util.find_spec("xarray") is not None
_HAS_PANDAS = importlib.util.find_spec("pandas") is not None
_HAS_H5PY = importlib.util.find_spec("h5py") is not None
_HAS_YAML = importlib.util.find_spec("yaml") is not None


class DummyStingrayObj(StingrayObject):
    main_array_attr = "guefus"

    def __init__(self, dummy=None):
        self.guefus = dummy
        self._mask = None
        # StingrayObject.__init__(self)


class DummyStingrayTs(StingrayTimeseries):
    main_array_attr = "time"

    def __init__(self, dummy=None):
        self.guefus = dummy
        # StingrayObject.__init__(self)


# A StingrayObject with no main_array_attr
class BadStingrayObj(StingrayObject):
    def __init__(self, dummy=None):
        self.guefus = dummy
        StingrayObject.__init__(self)


class TestStingrayObject:
    @classmethod
    def setup_class(cls):
        cls.arr = np.asanyarray([4, 5, 2])
        sting_obj = DummyStingrayObj(cls.arr)
        sting_obj.pardulas = [3.0 + 1.0j, 2.0j, 1.0 + 0.0j]
        sting_obj.sebadas = [[0, 1], [2, 3], [4, 5]]
        sting_obj._sebadas = [[0, 1], [2, 3], [4, 5]]
        sting_obj.pirichitus = np.int64(4)
        sting_obj.pistochedus = np.float64(4)
        sting_obj.parafritus = "bonus!"
        sting_obj.panesapa = [[41, 25], [98, 3]]
        cls.sting_obj = sting_obj

    def test_print(self, capsys):
        print(self.sting_obj)
        captured = capsys.readouterr()
        assert "guefus" in captured.out

    def test_preliminary(self):
        assert np.allclose(self.sting_obj.guefus, self.arr)

    def test_instantiate_without_main_array_attr(self):
        with pytest.raises(RuntimeError):
            BadStingrayObj(self.arr)

    def test_equality(self):
        ts1 = copy.deepcopy(self.sting_obj)
        ts2 = copy.deepcopy(self.sting_obj)
        assert ts1 == ts2

    def test_different_array_attributes(self):
        ts1 = copy.deepcopy(self.sting_obj)
        ts2 = copy.deepcopy(self.sting_obj)
        # Add a meta attribute only to ts1. This will fail
        ts1.blah = 2
        assert ts1 != ts2

        # Add a non-scalar meta attribute, but with the same name, to ts2.
        ts2.blah = [2]
        assert ts1 != ts2

        # Get back to normal
        del ts1.blah, ts2.blah
        assert ts1 == ts2

        # Add a non-scalar meta attribute to both, just slightly different
        ts1.blah = [2]
        ts2.blah = [3]
        assert ts1 != ts2

        # Get back to normal
        del ts1.blah, ts2.blah
        assert ts1 == ts2

        # Add a meta attribute only to ts2. This will also fail
        ts2.blah = 3
        assert ts1 != ts2

    def test_different_meta_attributes(self):
        ts1 = copy.deepcopy(self.sting_obj)
        ts2 = copy.deepcopy(self.sting_obj)
        # Add an array attribute to ts1. This will fail
        ts1.blah = ts1.guefus
        assert ts1 != ts2

        # Get back to normal
        del ts1.blah
        assert ts1 == ts2
        # Add an array attribute to ts2. This will fail
        ts2.blah = ts1.guefus
        assert ts1 != ts2

        # Get back to normal
        del ts2.blah
        assert ts1 == ts2

    @pytest.mark.parametrize("inplace", [True, False])
    def test_apply_mask(self, inplace):
        ts = copy.deepcopy(self.sting_obj)
        obj = ts.apply_mask([True, True, False], inplace=inplace)

        assert obj.parafritus == "bonus!"
        assert np.array_equal(obj.guefus, [4, 5])
        assert np.array_equal(obj.panesapa, ts.panesapa)
        assert np.array_equal(obj.pardulas, [3.0 + 1.0j, 2.0j])
        assert np.array_equal(obj.sebadas, [[0, 1], [2, 3]])
        if inplace:
            # Only if masking in place, the final object will be the same as the starting one.
            assert ts is obj
        else:
            # If not, the objects have to be different
            assert ts is not obj

    @pytest.mark.parametrize("inplace", [True, False])
    def test_partial_apply_mask(self, inplace):
        ts = copy.deepcopy(self.sting_obj)
        obj = ts.apply_mask([True, True, False], inplace=inplace, filtered_attrs=["pardulas"])
        assert obj.parafritus == "bonus!"
        assert np.array_equal(obj.guefus, [4, 5])
        assert np.array_equal(obj.panesapa, ts.panesapa)
        assert np.array_equal(obj.pardulas, [3.0 + 1.0j, 2.0j])
        assert obj.sebadas is None

        if inplace:
            assert ts is obj
        else:
            assert ts is not obj

    def test_operations(self):
        guefus = [5, 10, 15]
        count1 = [300, 100, 400]
        count2 = [600, 1200, 800]
        ts1 = DummyStingrayObj(guefus)
        ts2 = DummyStingrayObj(guefus)
        ts1.counts = count1
        ts2.counts = count2
        lc = ts1 + ts2  # Test __add__
        assert np.allclose(lc.counts, [900, 1300, 1200])
        assert np.array_equal(lc.guefus, guefus)
        lc = ts1 - ts2  # Test __sub__
        assert np.allclose(lc.counts, [-300, -1100, -400])
        assert np.array_equal(lc.guefus, guefus)
        lc = -ts2 + ts1  # Test __neg__
        assert np.allclose(lc.counts, [-300, -1100, -400])
        assert np.array_equal(lc.guefus, guefus)

    def test_inplace_add(self):
        guefus = [5, 10, 15]
        count1 = [300, 100, 400]
        count2 = [600, 1200, 800]
        ts1 = DummyStingrayObj(guefus)
        ts2 = DummyStingrayObj(guefus)
        ts1.counts = count1
        ts2.counts = count2
        lc = ts1 + ts2  # Test __add__
        ts1 += ts2  # Test __iadd__
        assert np.allclose(ts1.counts, [900, 1300, 1200])
        assert np.array_equal(ts1.guefus, guefus)
        assert lc == ts1

    def test_inplace_sub(self):
        guefus = [5, 10, 15]
        count1 = [300, 100, 400]
        count2 = [600, 1200, 800]
        ts1 = DummyStingrayObj(guefus)
        ts2 = DummyStingrayObj(guefus)
        ts1.counts = count1
        ts2.counts = count2
        lc = ts1 - ts2  # Test __sub__
        ts1 -= ts2  # Test __isub__
        assert np.allclose(ts1.counts, [-300, -1100, -400])
        assert np.array_equal(ts1.guefus, guefus)
        assert lc == ts1

    def test_inplace_add_with_method(self):
        guefus = [5, 10, 15]
        count1 = [300, 100, 400]
        count2 = [600, 1200, 800]
        ts1 = DummyStingrayObj(guefus)
        ts2 = DummyStingrayObj(guefus)
        ts1.counts = count1
        ts2.counts = count2
        lc = ts1.add(ts2)
        assert lc is not ts1
        lc_ip = ts2.add(ts1, inplace=True)
        assert lc == lc_ip
        assert lc_ip is ts2
        assert np.allclose(lc.counts, [900, 1300, 1200])
        assert np.array_equal(lc.guefus, guefus)

    def test_inplace_sub_with_method(self):
        guefus = [5, 10, 15]
        count1 = [300, 100, 400]
        count2 = [600, 1200, 800]
        ts1 = DummyStingrayObj(guefus)
        ts2 = DummyStingrayObj(guefus)
        ts1.counts = count1
        ts2.counts = count2
        lc = ts1.sub(ts2)
        assert lc is not ts1
        lc_ip = ts2.sub(ts1, inplace=True)
        assert lc == -lc_ip
        assert lc_ip is ts2
        assert np.allclose(lc.counts, [-300, -1100, -400])
        assert np.array_equal(lc.guefus, guefus)

    def test_failed_operations(self):
        guefus = [5, 10, 15]
        count1 = [300, 100, 400]
        ts1 = DummyStingrayObj(guefus)
        ts2 = DummyStingrayObj(np.array(guefus) + 1)
        ts1.counts = np.array(count1)
        ts2.counts = np.array(count1)
        with pytest.raises(TypeError, match=".*objects can only be operated with other.*"):
            ts1._operation_with_other_obj(float(3), np.add)
        with pytest.raises(ValueError, match="The values of guefus are different"):
            ts1 + ts2

    def test_len(self):
        assert len(self.sting_obj) == 3

    def test_slice(self):
        ts1 = self.sting_obj
        ts_filt = ts1[1]
        assert np.array_equal(ts_filt.guefus, [5])
        assert ts_filt.parafritus == "bonus!"
        assert np.array_equal(ts_filt.panesapa, ts1.panesapa)
        assert np.array_equal(ts_filt.pardulas, [2.0j])

        ts_filt = ts1[:2]
        assert np.array_equal(ts_filt.guefus, [4, 5])
        assert ts_filt.parafritus == "bonus!"
        assert np.array_equal(ts_filt.panesapa, ts1.panesapa)
        assert np.array_equal(ts_filt.pardulas, [3.0 + 1.0j, 2.0j])

        with pytest.raises(IndexError, match="The index must be either an integer or a slice"):
            ts1[1.0]

    def test_side_effects(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        new_so = DummyStingrayObj([4, 5, 6])
        assert not hasattr(new_so, "panesapa")

    def test_astropy_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        so.stingattr = DummyStingrayObj([3, 4, 5])
        ts = so.to_astropy_table()
        new_so = DummyStingrayObj.from_astropy_table(ts)
        assert so == new_so
        assert not hasattr(new_so, "stingattr")

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        ts = so.to_xarray()
        new_so = DummyStingrayObj.from_xarray(ts)
        assert so == new_so

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        ts = so.to_pandas()
        new_so = DummyStingrayObj.from_pandas(ts)
        # assert not hasattr(new_so, "sebadas")
        # new_so.sebadas = so.sebadas
        assert so == new_so

    def test_astropy_roundtrip_empty(self):
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        so = DummyStingrayObj([])
        ts = so.to_astropy_table()
        new_so = DummyStingrayObj.from_astropy_table(ts)
        assert new_so.guefus is None or len(new_so.guefus) == 0

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip_empty(self):
        so = DummyStingrayObj([])
        ts = so.to_xarray()
        new_so = DummyStingrayObj.from_xarray(ts)

        assert new_so.guefus is None or len(new_so.guefus) == 0

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip_empty(self):
        so = DummyStingrayObj([])
        ts = so.to_pandas()
        new_so = DummyStingrayObj.from_pandas(ts)

        assert new_so.guefus is None or len(new_so.guefus) == 0

    @pytest.mark.skipif("not _HAS_H5PY")
    def test_hdf_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        so.write("dummy.hdf5")
        new_so = so.read("dummy.hdf5")
        os.unlink("dummy.hdf5")

        assert so == new_so

    def test_file_roundtrip_fits(self):
        so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        with pytest.warns(
            UserWarning, match=".* output does not serialize the metadata at the moment"
        ):
            so.write("dummy.fits")
        new_so = DummyStingrayObj.read("dummy.fits")
        os.unlink("dummy.fits")
        # panesapa is invalid for FITS header and got lost
        assert not hasattr(new_so, "panesapa")
        new_so.panesapa = so.panesapa
        assert so == new_so

    @pytest.mark.parametrize("fmt", ["ascii", "ascii.ecsv"])
    def test_file_roundtrip(self, fmt):
        so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        with pytest.warns(UserWarning, match=".* output does not serialize the metadata"):
            so.write(f"dummy.{fmt}", fmt=fmt)
        new_so = DummyStingrayObj.read(f"dummy.{fmt}", fmt=fmt)
        os.unlink(f"dummy.{fmt}")

        assert so == new_so

    def test_file_roundtrip_pickle(self):
        fmt = "pickle"
        so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        so.write(f"dummy.{fmt}", fmt=fmt)
        new_so = DummyStingrayObj.read(f"dummy.{fmt}", fmt=fmt)
        os.unlink(f"dummy.{fmt}")

        assert so == new_so


class TestStingrayTimeseries:
    @classmethod
    def setup_class(cls):
        cls.time = np.arange(0, 10, 1)
        cls.arr = cls.time + 2
        sting_obj = StingrayTimeseries(
            time=cls.time,
            mjdref=59777.000,
            array_attrs=dict(guefus=cls.arr),
            parafritus="bonus!",
            panesapa=np.asanyarray([[41, 25], [98, 3]]),
            gti=np.asanyarray([[-0.5, 10.5]]),
        )
        sting_obj_highp = StingrayTimeseries(
            time=cls.time,
            mjdref=59777.000,
            array_attrs=dict(guefus=cls.arr),
            parafritus="bonus!",
            panesapa=np.asanyarray([[41, 25], [98, 3]]),
            gti=np.asanyarray([[-0.5, 10.5]]),
            high_precision=True,
        )
        cls.sting_obj = sting_obj
        cls.sting_obj_highp = sting_obj_highp

    def test_print(self, capsys):
        print(self.sting_obj)
        captured = capsys.readouterr()
        assert "59777" in captured.out

    def test_invalid_instantiation(self):
        with pytest.raises(ValueError, match="Lengths of time and guefus must be equal"):
            StingrayTimeseries(time=np.arange(10), array_attrs=dict(guefus=np.arange(11)))
        with pytest.raises(ValueError, match="Lengths of time and guefus must be equal"):
            StingrayTimeseries(time=np.arange(10), array_attrs=dict(guefus=np.zeros((5, 2))))

    def test_mask_is_none_then_isnt_no_gti(self):
        ts = copy.deepcopy(self.sting_obj)
        assert ts._mask is None
        # Unset GTIs
        ts.gti = None
        # But when I use the mask property, it's an array
        assert np.array_equal(ts.mask, np.ones(len(ts.time), dtype=bool))

    def test_apply_mask(self):
        ts = copy.deepcopy(self.sting_obj)
        mask = [True, True] + 8 * [False]
        newts0 = ts.apply_mask(mask, inplace=False)
        newts1 = ts.apply_mask(mask, inplace=True)
        for obj in [newts1, newts0]:
            for attr in ["parafritus", "mjdref"]:
                assert getattr(obj, attr) == getattr(ts, attr)
            for attr in ["panesapa", "gti"]:
                assert np.array_equal(getattr(obj, attr), getattr(ts, attr))

            assert np.array_equal(obj.guefus, [2, 3])
            assert np.array_equal(obj.time, [0, 1])
        assert ts is newts1
        assert ts is not newts0

    def test_comparison(self):
        time = [5, 10, 15]
        count1 = [300, 100, 400]

        ts1 = StingrayTimeseries(
            time=time,
            array_attrs=dict(counts=np.array(count1), _counts=np.array(count1)),
            mjdref=55000,
        )
        ts2 = StingrayTimeseries(
            time=time,
            array_attrs=dict(counts=np.array(count1), _counts=np.array(count1)),
            mjdref=55000,
        )

        assert ts1 == ts2
        # Change one attribute, check that they are not equal
        ts2.counts[0] += 1
        assert ts1 != ts2
        # Put back, check there are no side effects
        ts2.counts = np.array(count1)
        assert ts1 == ts2
        # Now check a meta attribute
        ts2.mjdref += 1
        assert ts1 != ts2
        # Put back, check there are no side effects
        ts2.mjdref = 55000
        assert ts1 == ts2
        # Now check an internal array attribute
        ts2._counts[0] += 1
        assert ts1 != ts2
        # Put back, check there are no side effects
        ts2._counts = np.array(count1)
        assert ts1 == ts2

    def test_zero_out_timeseries(self):
        time = [5, 10, 15]
        count1 = [300, 100, 400]

        ts1 = StingrayTimeseries(
            time=time,
            array_attrs=dict(counts=np.array(count1), _bla=np.array(count1)),
            mjdref=55000,
        )
        # All has been set correctly
        assert np.array_equal(ts1.counts, count1)
        assert np.array_equal(ts1._bla, count1)

        # Now zero out times and verify that everything else has been zeroed out
        ts1.time = None
        assert ts1.counts is None
        assert ts1.time is None
        assert ts1._bla is None

    def test_n_property(self):
        ts = StingrayTimeseries()
        assert ts.n == 0

        time = [5, 10, 15]
        count1 = [300, 100, 400]

        ts1 = StingrayTimeseries(
            time=time,
            array_attrs=dict(counts=np.array(count1)),
            mjdref=55000,
        )
        # All has been set correctly
        assert ts1.n == 3

        ts1.time = None
        assert ts1.n == 0

    def test_what_is_array_and_what_is_not(self):
        """Test that array_attrs are not confused with other attributes.

        In particular, time, gti and panesapa have the same length. Verify that panesapa
        is considered an array attribute, but not gti."""
        ts = StingrayTimeseries(
            [0, 3],
            gti=[[0.5, 1.5], [2.5, 3.5]],
            array_attrs=dict(
                panesapa=np.asanyarray([[41, 25], [98, 3]]),
                _panesapa=np.asanyarray([[41, 25], [98, 3]]),
            ),
            dt=1,
        )
        array_attrs = ts.array_attrs()
        internal_array_attrs = ts.internal_array_attrs()
        assert "panesapa" in array_attrs
        assert "_panesapa" not in array_attrs
        assert "_panesapa" in internal_array_attrs
        assert "gti" not in array_attrs
        assert "time" not in array_attrs

    def test_operations(self):
        time = [5, 10, 15]
        count1 = [300, 100, 400]
        count2 = [600, 1200, 800]

        ts1 = StingrayTimeseries(time=time)
        ts2 = StingrayTimeseries(time=time)
        ts1.counts = count1
        ts2.counts = count2
        ts1.counts_err = np.zeros_like(count1) + 1
        ts2.counts_err = np.zeros_like(count2) + 1

        lc = ts1 + ts2  # Test __add__
        assert np.allclose(lc.counts, [900, 1300, 1200])
        assert np.array_equal(lc.time, time)
        assert np.allclose(lc.counts_err, np.sqrt(2))
        lc = ts1 - ts2  # Test __sub__
        assert np.allclose(lc.counts, [-300, -1100, -400])
        assert np.array_equal(lc.time, time)
        assert np.allclose(lc.counts_err, np.sqrt(2))
        lc = -ts2 + ts1  # Test __neg__
        assert np.allclose(lc.counts, [-300, -1100, -400])
        assert np.array_equal(lc.time, time)
        assert np.allclose(lc.counts_err, np.sqrt(2))

    def test_operations_different_mjdref(self):
        time = [5, 10, 15]
        count1 = [300, 100, 400]
        count2 = [600, 1200, 800]

        ts1 = StingrayTimeseries(time=time, array_attrs=dict(counts=count1), mjdref=55000)
        # Now I create a second time series, I make sure that the times are the same, but
        # Then I change the mjdref. From now on, the time array will not have the same
        # values as `ts1.time`. The sum operation will warn the user about this, but then
        # change the mjdref of the second time series to match the first one, and the times
        # will be aligned again.
        ts2 = StingrayTimeseries(time=time, array_attrs=dict(counts=count2), mjdref=55000)
        ts2.change_mjdref(54000, inplace=True)
        with pytest.warns(UserWarning, match="MJDref is different in the two time series"):
            lc = ts1 + ts2  # Test __add__
        assert np.allclose(lc.counts, [900, 1300, 1200])
        assert np.array_equal(lc.time, time)
        assert np.array_equal(lc.mjdref, ts1.mjdref)
        with pytest.warns(UserWarning, match="MJDref is different in the two time series"):
            lc = ts2 + ts1  # Test __add__ of the other curve. The mjdref will be the other one now
        assert np.allclose(lc.counts, [900, 1300, 1200])
        assert np.array_equal(lc.time, ts2.time)
        assert np.array_equal(lc.mjdref, ts2.mjdref)

    def test_operation_with_diff_gti(self):
        time = [10, 20, 30, 40]
        count1 = [600, 1200, 800, 400]
        count2 = [300, 100, 400, 100]
        gti1 = [[0, 35]]
        gti2 = [[5, 40]]
        ts1 = StingrayTimeseries(time, array_attrs=dict(counts=count1), gti=gti1, dt=10)
        ts2 = StingrayTimeseries(time, array_attrs=dict(counts=count2), gti=gti2, dt=10)
        with pytest.warns(
            UserWarning, match="The good time intervals in the two time series are different."
        ):
            lc = ts1 - ts2
        assert np.allclose(lc.counts, [300, 1100, 400])
        assert np.allclose(lc.time, [10, 20, 30])

    def test_len(self):
        assert len(self.sting_obj) == 10

    def test_slice(self):
        ts1 = copy.deepcopy(self.sting_obj)

        ts_filt = ts1[1]
        assert np.array_equal(ts_filt.guefus, [3])
        assert ts_filt.parafritus == "bonus!"
        assert np.array_equal(ts_filt.panesapa, ts1.panesapa)

        ts_filt = ts1[:2]
        assert np.array_equal(ts_filt.guefus, [2, 3])
        assert ts_filt.parafritus == "bonus!"
        assert np.array_equal(ts_filt.panesapa, ts1.panesapa)

        ts_filt = ts1[0:3:2]
        # If dt >0, gtis are also altered. Otherwise they're left alone
        ts1.dt = 1
        ts_filt_dt1 = ts1[0:3:2]
        # Also try with array dt
        ts1.dt = np.ones_like(ts1.time)
        ts_filt_dtarr = ts1[0:3:2]
        for ts_f in [ts_filt, ts_filt_dt1, ts_filt_dtarr]:
            assert np.array_equal(ts_f.guefus, [2, 4])
            assert ts_f.parafritus == "bonus!"
            assert np.array_equal(ts_f.panesapa, ts1.panesapa)
        assert np.allclose(ts_filt_dt1.gti, [[-0.5, 0.5], [1.5, 2.5]])
        assert np.allclose(ts_filt_dtarr.gti, [[-0.5, 0.5], [1.5, 2.5]])
        assert np.allclose(ts_filt.gti, [[0, 2.0]])

        with pytest.raises(IndexError, match="The index must be either an integer or a slice"):
            ts1[1.0]

    @pytest.mark.parametrize("inplace", [True, False])
    def test_apply_gti(self, inplace):
        so = copy.deepcopy(self.sting_obj)
        so.gti = np.asanyarray([[-0.1, 2.1]])
        so2 = so.apply_gtis()
        if inplace:
            assert so2 is so

        assert np.allclose(so2.time, [0, 1, 2])
        assert np.allclose(so2.guefus, [2, 3, 4])
        assert np.allclose(so2.gti, [[-0.1, 2.1]])
        assert np.allclose(so2.mjdref, 59777.000)

    def test_split_ts_by_gtis(self):
        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        counts = [1, 1, 1, 1, 2, 3, 3, 2, 3, 3]
        bg_counts = [0, 0, 0, 1, 0, 1, 2, 0, 0, 1]
        bg_ratio = [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2]
        frac_exp = [1, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 1]
        gti = [[0.5, 4.5], [5.5, 7.5], [8.5, 9.5]]

        ts = StingrayTimeseries(
            times,
            array_attrs=dict(
                counts=counts, bg_counts=bg_counts, bg_ratio=bg_ratio, frac_exp=frac_exp
            ),
            gti=gti,
        )
        list_of_tss = ts.split_by_gti(min_points=0)
        assert len(list_of_tss) == 3

        ts0 = list_of_tss[0]
        ts1 = list_of_tss[1]
        ts2 = list_of_tss[2]
        assert np.allclose(ts0.time, [1, 2, 3, 4])
        assert np.allclose(ts1.time, [6, 7])
        assert np.allclose(ts2.time, [9])
        assert np.allclose(ts0.counts, [1, 1, 1, 1])
        assert np.allclose(ts1.counts, [3, 3])
        assert np.allclose(ts1.counts, [3])
        assert np.allclose(ts0.gti, [[0.5, 4.5]])
        assert np.allclose(ts1.gti, [[5.5, 7.5]])
        assert np.allclose(ts2.gti, [[8.5, 9.5]])
        # Check if new attributes are also split accordingly
        assert np.allclose(ts0.bg_counts, [0, 0, 0, 1])
        assert np.allclose(ts1.bg_counts, [1, 2])
        assert np.allclose(ts0.bg_ratio, [0.1, 0.1, 0.1, 0.2])
        assert np.allclose(ts1.bg_ratio, [0.2, 0.2])
        assert np.allclose(ts0.frac_exp, [1, 0.5, 1, 1])
        assert np.allclose(ts1.frac_exp, [0.5, 0.5])

    def test_truncate(self):
        time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        count = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        lc = StingrayTimeseries(time, array_attrs={"counts": count}, dt=1)
        lc_new = lc.truncate(start=2, stop=8, method="index")
        assert np.allclose(lc_new.counts, [30, 40, 50, 60, 70, 80])
        assert np.array_equal(lc_new.time, [3, 4, 5, 6, 7, 8])

        # Truncation can also be done by time values
        lc_new = lc.truncate(start=6, method="time")
        assert np.array_equal(lc_new.time, [6, 7, 8, 9])
        assert np.allclose(lc_new.counts, [60, 70, 80, 90])

    def test_truncate_not_str(self):
        with pytest.raises(TypeError, match="The method keyword argument"):
            self.sting_obj.truncate(method=1)

    def test_truncate_invalid(self):
        with pytest.raises(ValueError, match="Unknown method type"):
            self.sting_obj.truncate(method="ababalksdfja")

    def test_concatenate(self):
        time0 = [1, 2, 3]
        time1 = [5, 6, 7, 8]
        time2 = [10]
        count0 = [10, 20, 30]
        count1 = [50, 60, 70, 80]
        count2 = [100]
        gti0 = [[0.5, 3.5]]
        gti1 = [[4.5, 8.5]]
        gti2 = [[9.5, 10.5]]
        lc0 = StingrayTimeseries(
            time0, array_attrs={"counts": count0, "_bla": count0}, dt=1, gti=gti0
        )
        lc1 = StingrayTimeseries(
            time1, array_attrs={"counts": count1, "_bla": count1}, dt=1, gti=gti1
        )
        lc2 = StingrayTimeseries(
            time2, array_attrs={"counts": count2, "_bla": count2}, dt=1, gti=gti2
        )
        lc = lc0.concatenate([lc1, lc2])
        assert np.allclose(lc._bla, count0 + count1 + count2)
        assert np.allclose(lc.counts, count0 + count1 + count2)
        assert np.allclose(lc.time, time0 + time1 + time2)
        assert np.allclose(lc.gti, [[0.5, 3.5], [4.5, 8.5], [9.5, 10.5]])

    def test_concatenate_invalid(self):
        with pytest.raises(TypeError, match="objects can only be merged with other"):
            self.sting_obj.concatenate(1)

    def test_concatenate_gtis_overlap(self):
        time0 = [1, 2, 3, 4]
        time1 = [5, 6, 7, 8, 9]
        count0 = [10, 20, 30, 40]
        count1 = [50, 60, 70, 80, 90]
        gti0 = [[0.5, 4.5]]
        gti1 = [[3.5, 9.5]]
        lc0 = StingrayTimeseries(
            time0, array_attrs={"counts": count0, "_bla": count0}, dt=1, gti=gti0
        )
        lc1 = StingrayTimeseries(
            time1, array_attrs={"counts": count1, "_bla": count1}, dt=1, gti=gti1
        )
        with pytest.raises(ValueError, match="In order to append, GTIs must be mutually"):
            lc0.concatenate(lc1)

        # Instead, this will work
        lc0.concatenate(lc1, check_gti=False)

    def test_concatenate_diff_mjdref(self):
        time0 = [1, 2, 3, 4]
        time1 = [5, 6, 7, 8, 9]
        count0 = [10, 20, 30, 40]
        count1 = [50, 60, 70, 80, 90]
        gti0 = [[0.5, 4.49]]
        gti1 = [[4.51, 9.5]]
        lc0 = StingrayTimeseries(
            time0, array_attrs={"counts": count0, "_bla": count0}, dt=1, gti=gti0, mjdref=55000
        )
        lc1 = StingrayTimeseries(
            time1, array_attrs={"counts": count1, "_bla": count1}, dt=1, gti=gti1, mjdref=55000
        )
        lc1.change_mjdref(50001, inplace=True)
        with pytest.warns(UserWarning, match="mjdref is different"):
            lc = lc0.concatenate(lc1)
        assert lc.mjdref == 55000

    def test_rebin(self):
        time0 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        count0 = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        count0_err = [1] * 9
        gti0 = [[0.5, 9.5]]
        lc0 = StingrayTimeseries(
            time0,
            array_attrs={"counts": count0, "counts_err": count0_err, "_bla": count0},
            dt=1,
            gti=gti0,
        )
        # With new dt=2
        lc1 = lc0.rebin(dt_new=2)
        assert np.allclose(lc1.counts, [30, 70, 110, 150])
        assert np.allclose(lc1.counts_err, [np.sqrt(2)] * 4)
        assert np.allclose(lc1._bla, [30, 70, 110, 150])
        assert np.allclose(lc1.time, [1.5, 3.5, 5.5, 7.5])
        assert lc1.dt == 2
        # With a factor of two. Should give the same result
        lc1 = lc0.rebin(f=2)
        assert np.allclose(lc1.counts, [30, 70, 110, 150])
        assert np.allclose(lc1.counts_err, [np.sqrt(2)] * 4)
        assert np.allclose(lc1._bla, [30, 70, 110, 150])
        assert np.allclose(lc1.time, [1.5, 3.5, 5.5, 7.5])
        assert lc1.dt == 2

    def test_rebin_irregular(self):
        x0 = [-10]
        x1 = np.linspace(0, 10, 11)
        x2 = np.linspace(10.33, 20.0, 30)
        x3 = np.linspace(21, 30, 10)
        x = np.hstack([x0, x1, x2, x3])
        dt = np.hstack(
            [
                [1],
                [np.diff(x1).mean()] * x1.size,
                [np.diff(x2).mean()] * x3.size,
                [np.diff(x3).mean()] * x3.size,
            ]
        )

        counts = 2.0
        y = np.zeros_like(x) + counts

        yerr = np.sqrt(y)
        # Note: the first point of x is outside the GTIs
        lc0 = StingrayTimeseries(
            x,
            array_attrs={"counts": y, "counts_err": yerr, "_bla": y},
            dt=dt,
            gti=[[x1[0] - 0.5 * dt[1], x3[-1] + 0.5 * dt[-1]]],
        )

        dx_new = 1.5
        lc1 = lc0.rebin(dt_new=dx_new)
        from stingray import utils

        # Verify that the rebinning of irregular data is sampled correctly,
        # Including all data but the first point, which is outside GTIs
        xbin, ybin, yerr_bin, step_size = utils.rebin_data(x[1:], y[1:], dx_new, yerr[1:])

        assert np.allclose(lc1.time, xbin)
        assert np.allclose(lc1.counts, ybin)
        assert np.allclose(lc1._bla, ybin)
        assert np.allclose(lc1.counts_err, yerr_bin)

    def test_rebin_no_good_gtis(self):
        time0 = [1, 2, 3, 4]
        count0 = [10, 20, 30, 40]
        gti0 = [[0.5, 4.5]]
        lc0 = StingrayTimeseries(
            time0,
            array_attrs={"counts": count0},
            dt=1,
            gti=gti0,
        )
        with pytest.raises(ValueError, match="No valid GTIs after rebin."):
            print(lc0.rebin(dt_new=5).counts)

    def test_rebin_no_input(self):
        with pytest.raises(ValueError, match="You need to specify at least one between f and"):
            self.sting_obj.rebin()

    def test_rebin_less_than_dt(self):
        time0 = [1, 2, 3, 4]
        count0 = [10, 20, 30, 40]
        lc0 = StingrayTimeseries(time0, array_attrs={"counts": count0}, dt=1)
        with pytest.raises(ValueError, match="The new time resolution must be larger than"):
            lc0.rebin(dt_new=0.1)

    def test_sort(self):
        times = [2, 1, 3, 4]
        blah = np.asanyarray([40, 10, 20, 5])
        bleh = [4, 1, 2, 0.5]
        mjdref = 57000

        with pytest.warns(UserWarning, match="The time array is not sorted."):
            lc = StingrayTimeseries(
                times, array_attrs={"blah": blah, "_bleh": bleh}, dt=1, mjdref=mjdref
            )

        lc_new = lc.sort()

        assert np.allclose(lc_new._bleh, np.array([1, 4, 2, 0.5]))
        assert np.allclose(lc_new.blah, np.array([10, 40, 20, 5]))
        assert np.allclose(lc_new.time, np.array([1, 2, 3, 4]))
        assert lc_new.mjdref == mjdref

        lc_new = lc.sort(reverse=True)

        assert np.allclose(lc_new._bleh, np.array([0.5, 2, 4, 1]))
        assert np.allclose(lc_new.blah, np.array([5, 20, 40, 10]))
        assert np.allclose(lc_new.time, np.array([4, 3, 2, 1]))
        assert lc_new.mjdref == mjdref

    @pytest.mark.parametrize("highprec", [True, False])
    def test_astropy_roundtrip(self, highprec):
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        ts = so.to_astropy_table()
        new_so = StingrayTimeseries.from_astropy_table(ts)
        assert so == new_so

    def test_setting_property_fails(self):
        ts = Table(dict(time=[1, 2, 3]))
        ts.meta["exposure"] = 10
        with pytest.warns(
            UserWarning, match=r".*protected attribute\(s\) of StingrayTimeseries: exposure"
        ):
            StingrayTimeseries.from_astropy_table(ts)

    @pytest.mark.parametrize("highprec", [True, False])
    def test_astropy_ts_roundtrip(self, highprec):
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        ts = so.to_astropy_timeseries()
        new_so = StingrayTimeseries.from_astropy_timeseries(ts)
        assert so == new_so

    @pytest.mark.skipif("not _HAS_XARRAY")
    @pytest.mark.parametrize("highprec", [True, False])
    def test_xarray_roundtrip(self, highprec):
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        ts = so.to_xarray()
        new_so = StingrayTimeseries.from_xarray(ts)
        assert so == new_so

    @pytest.mark.skipif("not _HAS_PANDAS")
    @pytest.mark.parametrize("highprec", [True, False])
    def test_pandas_roundtrip(self, highprec):
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        ts = so.to_pandas()
        new_so = StingrayTimeseries.from_pandas(ts)
        # assert not hasattr(new_so, "sebadas")
        # new_so.sebadas = so.sebadas
        assert so == new_so

    @pytest.mark.skipif("not _HAS_H5PY")
    @pytest.mark.parametrize("highprec", [True, False])
    def test_hdf_roundtrip(self, highprec):
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        so.write("dummy.hdf5")
        new_so = so.read("dummy.hdf5")
        os.unlink("dummy.hdf5")

        assert so == new_so

    @pytest.mark.parametrize("highprec", [True, False])
    def test_file_roundtrip_fits(self, highprec):
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, self.time.shape)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        with pytest.warns(
            UserWarning, match=".* output does not serialize the metadata at the moment"
        ):
            so.write("dummy.fits")
        new_so = StingrayTimeseries.read("dummy.fits")
        os.unlink("dummy.fits")
        # panesapa is invalid for FITS header and got lost
        assert not hasattr(new_so, "panesapa")
        new_so.panesapa = so.panesapa
        new_so.gti = so.gti
        new_so == so

    @pytest.mark.parametrize("fmt", ["ascii", "ascii.ecsv"])
    @pytest.mark.parametrize("highprec", [True, False])
    def test_file_roundtrip(self, fmt, highprec):
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        with pytest.warns(
            UserWarning, match=".* output does not serialize the metadata at the moment"
        ):
            so.write(f"dummy.{fmt}", fmt=fmt)
        new_so = StingrayTimeseries.read(f"dummy.{fmt}", fmt=fmt)
        os.unlink(f"dummy.{fmt}")

        assert so == new_so

    @pytest.mark.parametrize("highprec", [True, False])
    def test_file_roundtrip_pickle(self, highprec):
        fmt = "pickle"
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        so.write(f"dummy.{fmt}", fmt=fmt)
        new_so = StingrayTimeseries.read(f"dummy.{fmt}", fmt=fmt)
        os.unlink(f"dummy.{fmt}")

        assert so == new_so

    @pytest.mark.parametrize("highprec", [True, False])
    def test_shift_time(self, highprec):
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        new_so = so.shift(1)
        assert np.allclose(new_so.time - 1, self.sting_obj.time)
        assert np.allclose(new_so.gti - 1, self.sting_obj.gti)

    @pytest.mark.parametrize("highprec", [True, False])
    def test_change_mjdref(self, highprec):
        if highprec:
            so = copy.deepcopy(self.sting_obj_highp)
        else:
            so = copy.deepcopy(self.sting_obj)
        new_so = so.change_mjdref(59776.5)
        assert new_so.mjdref == 59776.5
        assert np.allclose(new_so.time - 43200, self.sting_obj.time)
        assert np.allclose(new_so.gti - 43200, self.sting_obj.gti)

    def test_plot_simple(self):
        time0 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        count0 = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        count0_err = [1] * 9
        gti0 = [[0.5, 3.5], [4.5, 9.5]]
        lc0 = StingrayTimeseries(
            time0,
            array_attrs={"counts": count0, "counts_err": count0_err, "_bla": count0},
            dt=1,
            gti=gti0,
        )
        plt.close("all")
        lc0.plot("counts", title="Counts", witherrors=True)
        assert plt.fignum_exists("counts")
        plt.close("all")

    def test_plot_default_filename(self):
        self.sting_obj.plot("guefus", save=True)
        assert os.path.isfile("out.png")
        os.unlink("out.png")
        plt.close("all")

    def test_plot_custom_filename(self):
        self.sting_obj.plot("guefus", save=True, filename="lc.png")
        assert os.path.isfile("lc.png")
        os.unlink("lc.png")
        plt.close("all")


class TestStingrayTimeseriesSubclass:
    @classmethod
    def setup_class(cls):
        cls.arr = [4, 5, 2]
        sting_obj = DummyStingrayTs(cls.arr)
        sting_obj.time = np.asanyarray([0, 1, 2])
        sting_obj.mjdref = 59777.000
        sting_obj.parafritus = "bonus!"
        sting_obj.panesapa = np.asanyarray([[41, 25], [98, 3]])
        sting_obj.gti = np.asanyarray([[-0.5, 2.5]])
        cls.sting_obj = sting_obj
        curdir = os.path.abspath(os.path.dirname(__file__))
        datadir = os.path.join(curdir, "data")
        cls.fname = os.path.join(datadir, "monol_testA.evt")

    def test_print(self, capsys):
        print(self.sting_obj)
        captured = capsys.readouterr()
        assert "59777" in captured.out

    def test_astropy_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        ts = so.to_astropy_table()
        new_so = DummyStingrayTs.from_astropy_table(ts)
        assert so == new_so

    def test_astropy_ts_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        ts = so.to_astropy_timeseries()
        new_so = DummyStingrayTs.from_astropy_timeseries(ts)
        assert so == new_so

    def test_shift_time(self):
        new_so = self.sting_obj.shift(1)
        assert np.allclose(new_so.time - 1, self.sting_obj.time)
        assert np.allclose(new_so.gti - 1, self.sting_obj.gti)

    def test_change_mjdref(self):
        new_so = self.sting_obj.change_mjdref(59776.5)
        assert new_so.mjdref == 59776.5
        assert np.allclose(new_so.time - 43200, self.sting_obj.time)
        assert np.allclose(new_so.gti - 43200, self.sting_obj.gti)

    @pytest.mark.parametrize("check_gtis", [True, False])
    def test_read_timeseries_by_time_intv(self, check_gtis):
        reader = FITSTimeseriesReader(self.fname, output_class=DummyStingrayTs)[:]

        # Full slice
        evs = list(reader.filter_at_time_intervals([80000100, 80001000], check_gtis=check_gtis))
        assert len(evs) == 1
        ev0 = evs[0]
        assert np.all((ev0.time > 80000100) & (ev0.time < 80001000))
        assert np.all((ev0.gti >= 80000100) & (ev0.gti <= 80001000))
        assert np.isclose(ev0.gti[0, 0], 80000100)
        assert np.isclose(ev0.gti[-1, 1], 80001000)

    def test_read_timeseries_by_time_intv_check_bad_gtis(self):
        reader = FITSTimeseriesReader(self.fname, output_class=DummyStingrayTs)[:]

        # Full slice
        evs = list(reader.filter_at_time_intervals([80000100, 80001100], check_gtis=False))
        assert len(evs) == 1
        ev0 = evs[0]
        assert np.all((ev0.time > 80000100) & (ev0.time < 80001025))
        assert np.isclose(ev0.gti[0, 0], 80000100)
        # This second gti will be ugly, larger than the original gti boundary
        assert np.isclose(ev0.gti[-1, 1], 80001100)

    @pytest.mark.parametrize("gti_kind", ["same", "one", "multiple"])
    def test_read_apply_gti_lists(self, gti_kind):
        reader = FITSTimeseriesReader(self.fname, output_class=DummyStingrayTs)[:]
        if gti_kind == "same":
            gti_list = [reader.gti]
        elif gti_kind == "one":
            gti_list = [[[80000000, 80001024]]]
        elif gti_kind == "multiple":
            gti_list = [[[80000000, 80000512]], [[80000513, 80001024]]]

        evs = list(reader.apply_gti_lists(gti_list))

        # Check that the number of event lists is the same as the number of GTI lists we input
        assert len(evs) == len(gti_list)

        for i, ev in enumerate(evs):
            # Check that the gtis of the output event lists are the same we input
            assert np.allclose(ev.gti, gti_list[i])

    def test_read_apply_gti_lists_ignore_empty(self):
        reader = FITSTimeseriesReader(self.fname, output_class=DummyStingrayTs)[:]
        gti_list = [[], [[80000000, 80000512]], [[80000513, 80001024]]]
        evs = list(reader.apply_gti_lists(gti_list))
        assert np.allclose(evs[0].gti, gti_list[1])
        assert np.allclose(evs[1].gti, gti_list[2])


class TestJoinEvents:
    def test_join_without_times_simulated(self):
        """Test if exception is raised when join method is
        called before first simulating times.
        """
        ts = StingrayTimeseries()
        ts_other = StingrayTimeseries()

        with pytest.warns(UserWarning, match="One of the time series you are joining is empty."):
            assert ts.join(ts_other, strategy="union").time is None

    def test_join_empty_lists(self):
        """Test if an empty event list can be concatenated
        with a non-empty event list.
        """
        ts = StingrayTimeseries(time=[1, 2, 3])
        ts_other = StingrayTimeseries()
        with pytest.warns(UserWarning, match="One of the time series you are joining is empty."):
            ts_new = ts.join(ts_other, strategy="union")
        assert np.allclose(ts_new.time, [1, 2, 3])

        ts = StingrayTimeseries()
        ts_other = StingrayTimeseries(time=[1, 2, 3])
        ts_new = ts.join(ts_other, strategy="union")
        assert np.allclose(ts_new.time, [1, 2, 3])

        ts = StingrayTimeseries()
        ts_other = StingrayTimeseries()
        with pytest.warns(UserWarning, match="One of the time series you are joining is empty."):
            ts_new = ts.join(ts_other, strategy="union")
        assert ts_new.time is None
        assert ts_new.gti is None

        ts = StingrayTimeseries(time=[1, 2, 3])
        ts_other = StingrayTimeseries([])
        with pytest.warns(UserWarning, match="One of the time series you are joining is empty."):
            ts_new = ts.join(ts_other, strategy="union")
        assert np.allclose(ts_new.time, [1, 2, 3])

        ts = StingrayTimeseries([])
        ts_other = StingrayTimeseries(time=[1, 2, 3])
        ts_new = ts.join(ts_other, strategy="union")
        assert np.allclose(ts_new.time, [1, 2, 3])

    def test_join_different_dt(self):
        ts = StingrayTimeseries(time=[10, 20, 30], dt=1)
        ts_other = StingrayTimeseries(time=[40, 50, 60], dt=3)
        with pytest.warns(UserWarning, match="The time resolution is different."):
            ts_new = ts.join(ts_other, strategy="union")

        assert np.array_equal(ts_new.dt, [1, 1, 1, 3, 3, 3])
        assert np.allclose(ts_new.time, [10, 20, 30, 40, 50, 60])

    def test_join_different_instr(self):
        ts = StingrayTimeseries(time=[10, 20, 30], instr="fpma")
        ts_other = StingrayTimeseries(time=[40, 50, 60], instr="fpmb")
        with pytest.warns(
            UserWarning,
            match="Attribute instr is different in the time series being merged.",
        ):
            ts_new = ts.join(ts_other, strategy="union")

        assert ts_new.instr == "fpma,fpmb"

    def test_join_different_meta_attribute(self):
        ts = StingrayTimeseries(time=[10, 20, 30])
        ts_other = StingrayTimeseries(time=[40, 50, 60])
        ts_other.bubu = "settete"
        ts.whatstheanswer = 42
        ts.unmovimentopara = "arriba"
        ts_other.unmovimentopara = "abajo"

        with pytest.warns(
            UserWarning,
            match=(
                "Attribute (bubu|whatstheanswer|unmovimentopara) is different "
                "in the time series being merged."
            ),
        ):
            ts_new = ts.join(ts_other, strategy="union")

        assert ts_new.bubu == (None, "settete")
        assert ts_new.whatstheanswer == (42, None)
        assert ts_new.unmovimentopara == "arriba,abajo"

    def test_join_without_energy(self):
        ts = StingrayTimeseries(time=[1, 2, 3], energy=[3, 3, 3])
        ts_other = StingrayTimeseries(time=[4, 5])
        with pytest.warns(
            UserWarning, match="The energy array is empty in one of the time series being merged."
        ):
            ts_new = ts.join(ts_other, strategy="union")

        assert np.allclose(ts_new.energy, [3, 3, 3, np.nan, np.nan], equal_nan=True)

    def test_join_without_pi(self):
        ts = StingrayTimeseries(time=[1, 2, 3], pi=[3, 3, 3])
        ts_other = StingrayTimeseries(time=[4, 5])
        with pytest.warns(
            UserWarning, match="The pi array is empty in one of the time series being merged."
        ):
            ts_new = ts.join(ts_other, strategy="union")

        assert np.allclose(ts_new.pi, [3, 3, 3, np.nan, np.nan], equal_nan=True)

    def test_join_with_arbitrary_attribute(self):
        ts = StingrayTimeseries(time=[1, 2, 4])
        ts_other = StingrayTimeseries(time=[3, 5])
        ts.u = [3, 3, 3]
        ts_other.q = [1, 2]
        with pytest.warns(
            UserWarning, match="The (u|q) array is empty in one of the time series being merged."
        ):
            ts_new = ts.join(ts_other, strategy="union")

        assert np.allclose(ts_new.q, [np.nan, np.nan, 1, np.nan, 2], equal_nan=True)
        assert np.allclose(ts_new.u, [3, 3, np.nan, 3, np.nan], equal_nan=True)

    def test_join_with_gti_none(self):
        ts = StingrayTimeseries(time=[1, 2, 3])
        ts_other = StingrayTimeseries(time=[4, 5], gti=[[3.5, 5.5]])
        ts_new = ts.join(ts_other, strategy="union")

        assert np.allclose(ts_new.gti, [[1, 3], [3.5, 5.5]])

        ts = StingrayTimeseries(time=[1, 2, 3], gti=[[0.5, 3.5]])
        ts_other = StingrayTimeseries(time=[4, 5])
        ts_new = ts.join(ts_other, strategy="union")

        assert np.allclose(ts_new.gti, [[0.5, 3.5], [4, 5]])

        ts = StingrayTimeseries(time=[1, 2, 3])
        ts_other = StingrayTimeseries(time=[4, 5])
        ts_new = ts.join(ts_other, strategy="union")

        assert ts_new._gti is None

    def test_non_overlapping_join_infer(self):
        """Join two overlapping event lists."""
        ts = StingrayTimeseries(
            time=[1, 1.1, 2, 3, 4], energy=[3, 4, 7, 4, 3], gti=[[1, 2], [3, 4]]
        )
        ts_other = StingrayTimeseries(time=[5, 6, 6.1, 7, 10], energy=[4, 3, 8, 1, 2], gti=[[6, 7]])
        ts_new = ts.join(ts_other, strategy="infer")

        assert (ts_new.time == np.array([1, 1.1, 2, 3, 4, 5, 6, 6.1, 7, 10])).all()
        assert (ts_new.energy == np.array([3, 4, 7, 4, 3, 4, 3, 8, 1, 2])).all()
        assert (ts_new.gti == np.array([[1, 2], [3, 4], [6, 7]])).all()

    def test_overlapping_join_infer(self):
        """Join two non-overlapping event lists."""
        with pytest.warns(UserWarning, match="The time array is not sorted."):
            ts = StingrayTimeseries(
                time=[1, 1.1, 10, 6, 5], energy=[10, 6, 3, 11, 2], gti=[[1, 3], [5, 6]]
            )
        with pytest.warns(UserWarning, match="The time array is not sorted."):
            ts_other = StingrayTimeseries(
                time=[5.1, 7, 6.1, 6.11, 10.1], energy=[2, 3, 8, 1, 2], gti=[[5, 7], [8, 10]]
            )
        ts_new = ts.join(ts_other, strategy="infer")

        assert (ts_new.time == np.array([1, 1.1, 5, 5.1, 6, 6.1, 6.11, 7, 10, 10.1])).all()
        assert (ts_new.energy == np.array([10, 6, 2, 2, 11, 8, 1, 3, 3, 2])).all()
        assert (ts_new.gti == np.array([[5, 6]])).all()

    def test_overlapping_join_change_mjdref(self):
        """Join two non-overlapping event lists."""
        with pytest.warns(UserWarning, match="The time array is not sorted."):
            ts = StingrayTimeseries(
                time=[1, 1.1, 10, 6, 5],
                energy=[10, 6, 3, 11, 2],
                gti=[[1, 3], [5, 6]],
                mjdref=57001,
            )
        with pytest.warns(UserWarning, match="The time array is not sorted."):
            ts_other = StingrayTimeseries(
                time=np.asanyarray([5.1, 7, 6.1, 6.11, 10.1]) + 86400,
                energy=[2, 3, 8, 1, 2],
                gti=np.asanyarray([[5, 7], [8, 10]]) + 86400,
                mjdref=57000,
            )
        with pytest.warns(UserWarning, match="Attribute mjdref is different"):
            ts_new = ts.join(ts_other, strategy="intersection")

        assert np.allclose(ts_new.time, np.array([1, 1.1, 5, 5.1, 6, 6.1, 6.11, 7, 10, 10.1]))
        assert (ts_new.energy == np.array([10, 6, 2, 2, 11, 8, 1, 3, 3, 2])).all()
        assert np.allclose(ts_new.gti, np.array([[5, 6]]))

    def test_multiple_join(self):
        """Test if multiple event lists can be joined."""
        ts = StingrayTimeseries(time=[1, 2, 4], instr="a", mission=1)
        ts_other = StingrayTimeseries(time=[3, 5, 7], instr="b", mission=2)
        ts_other2 = StingrayTimeseries(time=[6, 8, 9], instr="c", mission=3)

        ts.pibiri = [1, 1, 1]
        ts_other.pibiri = [2, 2, 2]
        ts_other2.pibiri = [3, 3, 3]

        with pytest.warns(
            UserWarning,
            match="Attribute (instr|mission) is different in the time series being merged.",
        ):
            ts_new = ts.join([ts_other, ts_other2], strategy="union")
        assert np.allclose(ts_new.time, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert np.allclose(ts_new.pibiri, [1, 1, 2, 1, 2, 3, 2, 3, 3])
        assert ts_new.instr == "a,b,c"
        assert ts_new.mission == (1, 2, 3)

    def test_join_ignore_attr(self):
        """Test if multiple event lists can be joined."""
        ts = StingrayTimeseries(time=[1, 2, 4], instr="a", mission=1)
        ts_other = StingrayTimeseries(time=[3, 5, 7], instr="b", mission=2)

        with pytest.warns(
            UserWarning,
            match="Attribute mission is different in the time series being merged.",
        ):
            ts_new = ts.join([ts_other], strategy="union", ignore_meta=["instr"])

        assert np.allclose(ts_new.time, [1, 2, 3, 4, 5, 7])
        assert not hasattr(ts_new, "instr")
        assert ts_new.mission == (1, 2)


class TestFillBTI(object):
    @classmethod
    def setup_class(cls):
        cls.rand_time = np.sort(np.random.uniform(0, 1000, 100000))
        cls.rand_ener = np.random.uniform(0, 100, 100000)
        cls.gti = [[0, 900], [950, 1000]]
        blablas = np.random.normal(0, 1, cls.rand_ener.size)
        cls.ev_like = StingrayTimeseries(
            cls.rand_time, energy=cls.rand_ener, blablas=blablas, gti=cls.gti
        )
        time_edges = np.linspace(0, 1000, 1001)
        counts = np.histogram(cls.rand_time, bins=time_edges)[0]
        blablas = np.random.normal(0, 1, 1000)
        cls.lc_like = StingrayTimeseries(
            time=time_edges[:-1] + 0.5, counts=counts, blablas=blablas, gti=cls.gti, dt=1
        )

    def test_no_btis_returns_copy(self):
        ts = StingrayTimeseries([1, 2, 3], energy=[4, 6, 8], gti=[[0.5, 3.5]])
        ts_new = ts.fill_bad_time_intervals()
        assert ts == ts_new

    def test_event_like(self):
        ev_like_filt = copy.deepcopy(self.ev_like)
        # I introduce a small gap in the GTIs
        ev_like_filt.gti = np.asanyarray(
            [[0, 498], [500, 520], [522, 700], [702, 900], [950, 1000]]
        )
        ev_new = ev_like_filt.fill_bad_time_intervals(max_length=3, buffer_size=4)

        assert np.allclose(ev_new.gti, self.gti)

        # Now, I set the same GTIs as the original event list, and the data
        # should be the same
        ev_new.gti = ev_like_filt.gti

        new_masked, filt_masked = ev_new.apply_gtis(), ev_like_filt.apply_gtis()
        for attr in ["time", "energy", "blablas"]:
            assert np.allclose(getattr(new_masked, attr), getattr(filt_masked, attr))

    def test_no_counts_in_buffer(self):
        ev_like_filt = copy.deepcopy(self.ev_like)
        # I introduce a small gap in the GTIs
        ev_like_filt.gti = np.asanyarray([[0, 490], [491, 498], [500, 505], [510, 520], [522, 700]])

        # I empty out two GTIs
        bad = (ev_like_filt.time > 490) & (ev_like_filt.time < 510)
        ev_like_filt = ev_like_filt.apply_mask(~bad)

        with pytest.warns(UserWarning, match="simulate the time series in interval 498-500"):
            ev_like_filt.fill_bad_time_intervals(max_length=3, buffer_size=2)

    def test_lc_like(self):
        lc_like_filt = copy.deepcopy(self.lc_like)
        # I introduce a small gap in the GTIs
        lc_like_filt.gti = np.asanyarray(
            [[0, 498], [500, 520], [522, 700], [702, 900], [950, 1000]]
        )
        lc_new = lc_like_filt.fill_bad_time_intervals(max_length=3, buffer_size=4)
        assert np.allclose(lc_new.gti, self.gti)

        lc_like_gtifilt = self.lc_like.apply_gtis(inplace=False)
        # In this case, the time array should also be the same as the original
        assert np.allclose(lc_new.time, lc_like_gtifilt.time)

        # Now, I set the same GTIs as the original event list, and the data
        # should be the same
        lc_new.gti = lc_like_filt.gti

        new_masked, filt_masked = lc_new.apply_gtis(), lc_like_filt.apply_gtis()
        for attr in ["time", "counts", "blablas"]:
            assert np.allclose(getattr(new_masked, attr), getattr(filt_masked, attr))

    def test_ignore_attrs_ev_like(self):
        ev_like_filt = copy.deepcopy(self.ev_like)
        # I introduce a small gap in the GTIs
        ev_like_filt.gti = np.asanyarray([[0, 498], [500, 900], [950, 1000]])
        ev_new0 = ev_like_filt.fill_bad_time_intervals(seed=1234)
        ev_new1 = ev_like_filt.fill_bad_time_intervals(seed=1234, attrs_to_randomize=["energy"])
        assert np.allclose(ev_new0.gti, ev_new1.gti)
        assert np.allclose(ev_new0.time, ev_new1.time)

        assert np.count_nonzero(np.isnan(ev_new0.blablas)) == 0
        assert np.count_nonzero(np.isnan(ev_new1.blablas)) > 0
        assert np.count_nonzero(np.isnan(ev_new1.energy)) == 0

    def test_ignore_attrs_lc_like(self):
        lc_like_filt = copy.deepcopy(self.lc_like)
        # I introduce a small gap in the GTIs
        lc_like_filt.gti = np.asanyarray([[0, 498], [500, 900], [950, 1000]])
        lc_new0 = lc_like_filt.fill_bad_time_intervals(seed=1234)
        lc_new1 = lc_like_filt.fill_bad_time_intervals(seed=1234, attrs_to_randomize=["counts"])
        assert np.allclose(lc_new0.gti, lc_new1.gti)
        assert np.allclose(lc_new0.time, lc_new1.time)

        assert np.count_nonzero(np.isnan(lc_new0.blablas)) == 0
        assert np.count_nonzero(np.isnan(lc_new1.blablas)) > 0
        assert np.count_nonzero(np.isnan(lc_new1.counts)) == 0

    def test_forcing_non_uniform(self):
        ev_like_filt = copy.deepcopy(self.ev_like)
        # I introduce a small gap in the GTIs
        ev_like_filt.gti = np.asanyarray([[0, 498], [500, 900], [950, 1000]])
        # Results should be exactly the same
        ev_new0 = ev_like_filt.fill_bad_time_intervals(even_sampling=False, seed=201903)
        ev_new1 = ev_like_filt.fill_bad_time_intervals(even_sampling=None, seed=201903)
        for attr in ["time", "energy"]:
            assert np.allclose(getattr(ev_new0, attr), getattr(ev_new1, attr))

    def test_forcing_uniform(self):
        lc_like_filt = copy.deepcopy(self.lc_like)
        # I introduce a small gap in the GTIs
        lc_like_filt.gti = np.asanyarray([[0, 498], [500, 900], [950, 1000]])
        # Results should be exactly the same
        lc_new0 = lc_like_filt.fill_bad_time_intervals(even_sampling=True, seed=201903)
        lc_new1 = lc_like_filt.fill_bad_time_intervals(even_sampling=None, seed=201903)
        for attr in ["time", "counts", "blablas"]:
            assert np.allclose(getattr(lc_new0, attr), getattr(lc_new1, attr))

    def test_bti_close_to_edge_event_like(self):
        ev_like_filt = copy.deepcopy(self.ev_like)
        # I introduce a small gap in the GTIs
        ev_like_filt.gti = np.asanyarray([[0, 0.5], [1, 900], [950, 1000]])
        ev_new = ev_like_filt.fill_bad_time_intervals()
        assert np.allclose(ev_new.gti, self.gti)

        ev_like_filt = copy.deepcopy(self.ev_like)
        # I introduce a small gap in the GTIs
        ev_like_filt.gti = np.asanyarray([[0, 900], [950, 999], [999.5, 1000]])
        ev_new = ev_like_filt.fill_bad_time_intervals()
        assert np.allclose(ev_new.gti, self.gti)

    def test_bti_close_to_edge_lc_like(self):
        lc_like_filt = copy.deepcopy(self.lc_like)
        # I introduce a small gap in the GTIs
        lc_like_filt.gti = np.asanyarray([[0, 0.5], [1, 900], [950, 1000]])
        lc_new = lc_like_filt.fill_bad_time_intervals()
        assert np.allclose(lc_new.gti, self.gti)

        lc_like_filt = copy.deepcopy(self.lc_like)
        # I introduce a small gap in the GTIs
        lc_like_filt.gti = np.asanyarray([[0, 900], [950, 999], [999.5, 1000]])
        lc_new = lc_like_filt.fill_bad_time_intervals()
        assert np.allclose(lc_new.gti, self.gti)


class TestAnalyzeChunks(object):
    @classmethod
    def setup_class(cls):
        cls.time = np.arange(150)
        counts = np.zeros_like(cls.time) + 3
        cls.ts = StingrayTimeseries(cls.time, counts=counts, dt=1)
        cls.ts_no_dt = StingrayTimeseries(cls.time, counts=counts, dt=0)
        cls.ts_no_counts = StingrayTimeseries(cls.time, dt=0)

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="You have to specify at least one of"):
            self.ts.estimate_segment_size()
        with pytest.raises(ValueError, match="You have to specify at least one of"):
            self.ts_no_dt.estimate_segment_size()

    def test_no_total_counts(self):
        assert self.ts.estimate_segment_size(min_samples=2) == 2
        assert self.ts_no_dt.estimate_segment_size(min_samples=2) == 2
        assert self.ts_no_counts.estimate_segment_size(min_samples=2) == 2

    def test_estimate_segment_size(self):
        # Here, the total counts dominate
        assert self.ts.estimate_segment_size(min_counts=10, min_samples=3) == 4.0

    def test_estimate_segment_size_more_bins(self):
        # Here, the time bins dominate
        assert self.ts.estimate_segment_size(min_counts=10, min_samples=5) == 5.0

    def test_estimate_segment_size_lower_counts(self):
        counts = np.zeros_like(self.time) + 3
        counts[2:4] = 1
        ts = StingrayTimeseries(self.time, counts=counts, dt=1)
        assert ts.estimate_segment_size(min_counts=3, min_samples=1) == 3.0

    def test_estimate_segment_size_lower_dt(self):
        # A slightly more complex example
        dt = 0.2
        time = np.arange(0, 1000, dt)
        counts = np.random.poisson(100, size=len(time))
        ts = StingrayTimeseries(time, counts=counts, dt=dt)
        assert ts.estimate_segment_size(min_counts=100, min_samples=2) == 0.4

        assert ts.estimate_segment_size(100, min_samples=40) == 8.0

    @pytest.mark.parametrize("n_outs", [0, 1, 2, 3])
    def test_analyze_segments_bad_intv(self, n_outs):
        ts = StingrayTimeseries(time=np.arange(10), dt=1, gti=[[-0.5, 0.5], [1.5, 10.5]])

        def func(x):
            if n_outs == 0:
                return np.size(x.time)
            return [np.size(x.time) for _ in range(n_outs)]

        # I do not specify the segment_size, which means results will be calculated per-GTI
        with pytest.warns(UserWarning, match="has one data point or less."):
            ts.analyze_segments(func, segment_size=None)

    def test_analyze_segments_by_gti(self):
        ts = StingrayTimeseries(time=np.arange(11), dt=1, gti=[[-0.5, 5.5], [6.5, 10.5]])

        def func(x):
            return np.size(x.time)

        _, _, results_as = ts.analyze_segments(func, segment_size=None)
        _, _, results_ag = ts.analyze_by_gti(func)

        assert np.allclose(results_as, results_ag)
        assert np.allclose(results_as, [6, 4])

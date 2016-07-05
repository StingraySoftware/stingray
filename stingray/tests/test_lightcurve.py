import numpy as np

from astropy.tests.helper import pytest
import warnings
import os
import matplotlib.pyplot as plt

from stingray import Lightcurve

np.random.seed(20150907)

_H5PY_INSTALLED = True

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False

class TestLightcurve(object):

    @classmethod
    def setup_class(cls):
        cls.times = [1, 2, 3, 4]
        cls.counts = [2, 2, 2, 2]
        cls.dt = 1.0

    def test_create(self):
        """
        Demonstrate that we can create a trivial Lightcurve object.
        """
        lc = Lightcurve(self.times, self.counts)

    def test_irregular_time_warning(self):
        """
        Check if inputting an irregularly spaced time iterable throws out
        a warning.
        """
        times = [1, 2, 3, 5, 6]
        counts = [2, 2, 2, 2, 2]
        warn_str = ("SIMON says: Bin sizes in input time array aren't equal "
                    "throughout! This could cause problems with Fourier "
                    "transforms. Please make the input time evenly sampled.")

        with warnings.catch_warnings(record=True) as w:
            lc = Lightcurve(times, counts)
            assert str(w[0].message) == warn_str

    def test_lightcurve_from_toa(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)

    def test_tstart(self):
        tstart = 0.0
        lc = Lightcurve.make_lightcurve(self.times, self.dt, tstart=0.0)
        assert lc.tstart == tstart
        assert lc.times[0] == tstart + 0.5*self.dt

    def test_tseg(self):
        tstart = 0.0
        tseg = 5.0
        lc = Lightcurve.make_lightcurve(self.times, self.dt,
                                        tseg=tseg, tstart=tstart)

        assert lc.tseg == tseg
        assert lc.times[-1] - lc.times[0] == tseg - self.dt

    def test_nondivisble_tseg(self):
        """
        If the light curve length input is not divisible by the time
        resolution, the last (fractional) time bin will be dropped.
        """
        tstart = 0.0
        tseg = 5.5
        lc = Lightcurve.make_lightcurve(self.times, self.dt,
                                        tseg=tseg, tstart=tstart)
        assert lc.tseg == int(tseg/self.dt)

    def test_correct_timeresolution(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)
        assert np.isclose(lc.dt, self.dt)

    def test_bin_correctly(self):
        ncounts = np.array([2, 1, 0, 3])
        tstart = 0.0
        tseg = 4.0

        toa = np.hstack([np.random.uniform(i, i+1, size=n)
                         for i, n in enumerate(ncounts)])

        dt = 1.0
        lc = Lightcurve.make_lightcurve(toa, dt, tseg=tseg, tstart=tstart)

        assert np.allclose(lc.counts, ncounts)

    def test_countrate(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        counts = np.zeros_like(times) + mean_counts
        lc = Lightcurve(times, counts)
        assert np.allclose(lc.countrate, np.zeros_like(counts) +
                           mean_counts/dt)

    def test_input_countrate(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        countrate = np.zeros_like(times) + mean_counts
        lc = Lightcurve(times, countrate, input_counts=False)
        assert np.allclose(lc.counts, np.zeros_like(countrate) +
                           mean_counts*dt)

    def test_creating_lightcurve_raises_type_error_when_input_is_none(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        counts = np.array([None] * times.shape[0])
        with pytest.raises(TypeError):
            lc = Lightcurve(times, counts)

    def test_creating_lightcurve_raises_type_error_when_input_is_inf(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        counts = np.array([np.inf] * times.shape[0])
        with pytest.raises(AssertionError):
            lc = Lightcurve(times, counts)

    def test_creating_lightcurve_raises_type_error_when_input_is_nan(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        counts = np.array([np.nan] * times.shape[0])
        with pytest.raises(AssertionError):
            lc = Lightcurve(times, counts)

    def test_init_with_diff_array_lengths(self):
        time = [1, 2, 3]
        counts = [2, 2, 2, 2]
        
        with pytest.raises(AssertionError):
            lc = Lightcurve(time, counts)

    def test_add_with_different_time_arrays(self):
        _times = [1, 2, 3, 4, 5]
        _counts = [2, 2, 2, 2, 2]

        with pytest.raises(AssertionError):

            lc1 = Lightcurve(self.times, self.counts)
            lc2 = Lightcurve(_times, _counts)

            lc = lc1 + lc2

    def test_add_with_unequal_time_arrays(self):
        _times = [1, 3, 5, 7]

        with pytest.raises(AssertionError):
            lc1 = Lightcurve(self.times, self.counts)
            lc2 = Lightcurve(_times, self.counts)

            lc = lc1 + lc2

    def test_add_with_equal_time_arrays(self):
        _counts = [1, 1, 1, 1]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(self.times, _counts)

        lc = lc1 + lc2

        assert np.all(lc.counts == lc1.counts + lc2.counts)
        assert np.all(lc.countrate == lc1.countrate + lc2.countrate)

    def test_sub_with_diff_time_arrays(self):
        _times = [1, 2, 3, 4, 5]
        _counts = [2, 2, 2, 2, 2]

        with pytest.raises(AssertionError):
            lc1 = Lightcurve(self.times, self.counts)
            lc2 = Lightcurve(_times, _counts)

            lc = lc1 - lc2

    def test_subtraction(self):
        _counts = [3, 4, 5, 6]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(self.times, _counts)

        lc = lc2 - lc1

        expected_counts = np.array([1, 2, 3, 4])
        assert np.all(lc.counts == expected_counts)

    def test_negation(self):
        lc = Lightcurve(self.times, self.counts)

        _lc = lc + (-lc)

        assert not np.all(_lc.counts)

    def test_len_function(self):
        lc = Lightcurve(self.times, self.counts)

        assert len(lc) == 4

    def test_indexing_with_unexpected_type(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(IndexError):
            count = lc['first']

    def test_indexing(self):
        lc = Lightcurve(self.times, self.counts)

        assert lc[0] == lc[1] == lc[2] == lc[3] == 2

    def test_slicing(self):
        lc = Lightcurve(self.times, self.counts)

        assert np.all(lc[1:3].counts == np.array([2, 2]))
        assert np.all(lc[:2].counts == np.array([2, 2]))
        assert np.all(lc[2:].counts == np.array([2, 2]))
        assert np.all(lc[:].counts == np.array([2, 2, 2, 2]))

    def test_slicing_index_error(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(AssertionError):
            lc_new = lc[1:2]

    def test_join_with_different_dt(self):
        _times = [5, 5.5, 6]
        _counts = [2, 2, 2]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(_times, _counts)

        with warnings.catch_warnings(record=True) as w:
            lc1.join(lc2)
            assert "different bin widths" in str(w[0].message)

    def test_join_disjoint_time_arrays(self):
        _times = [5, 6, 7, 8]
        _counts = [2, 2, 2, 2]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(_times, _counts)

        lc = lc1.join(lc2)

        assert len(lc.counts) == len(lc.times) == 8
        assert np.all(lc.counts == 2)

    def test_join_overlapping_time_arrays(self):
        _times = [3, 4, 5, 6]
        _counts = [4, 4, 4, 4]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(_times, _counts)

        with warnings.catch_warnings(record=True) as w:
            lc = lc1.join(lc2)
            assert "overlapping time ranges" in str(w[0].message)

        assert len(lc.counts) == len(lc.times) == 6
        assert np.all(lc.counts == np.array([2, 2, 3, 3, 4, 4]))

    def test_truncate_by_index(self):
        lc = Lightcurve(self.times, self.counts)

        lc1 = lc.truncate(start=1)
        assert np.all(lc1.times == np.array([2, 3, 4]))
        assert np.all(lc1.counts == np.array([2, 2, 2]))

        lc2 = lc.truncate(stop=2)
        assert np.all(lc2.times == np.array([1, 2]))
        assert np.all(lc2.counts == np.array([2, 2]))

    def test_truncate_by_time_stop_less_than_start(self):
        lc = Lightcurve(self.times, self.counts)
 
        with pytest.raises(AssertionError):
            lc1 = lc.truncate(start=2, stop=1, method='time')

    def test_truncate_by_time(self):
        lc = Lightcurve(self.times, self.counts)

        lc1 = lc.truncate(start=1, method='time')
        assert np.all(lc1.times == np.array([1, 2, 3, 4]))
        assert np.all(lc1.counts == np.array([2, 2, 2, 2]))

        lc2 = lc.truncate(stop=3, method='time')
        assert np.all(lc2.times == np.array([1, 2]))
        assert np.all(lc2.counts == np.array([2, 2]))

    def test_sort(self):
        _times = [1, 2, 3, 4]
        _counts = [40, 10, 20, 5]
        lc = Lightcurve(_times, _counts)

        lc.sort()

        assert np.all(lc.counts == np.array([ 5, 10, 20, 40]))
        assert np.all(lc.times == np.array([4, 2, 3, 1]))

        lc.sort(reverse=True)

        assert np.all(lc.counts == np.array([40, 20, 10,  5]))
        assert np.all(lc.times == np.array([1, 3, 2, 4]))

    def test_plot_matplotlib_not_installed(self):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:

            lc = Lightcurve(self.times, self.counts)
            try:
                lc.plot()
            except Exception as e:
                assert type(e) is ImportError
                assert str(e) == "Matplotlib required for plot()"

    def test_plot_simple(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot()
        assert plt.fignum_exists(1)

    def test_plot_wrong_label_type(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(TypeError):
            with warnings.catch_warnings(record=True) as w:
                lc.plot(labels=123)
                assert "must be either a list or tuple" in str(w[0].message)

    def test_plot_labels_index_error(self):
        lc = Lightcurve(self.times, self.counts)
        with warnings.catch_warnings(record=True) as w:
            lc.plot(labels=('x'))
            assert "must have two labels" in str(w[0].message)

    def test_plot_default_filename(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(save=True)
        assert os.path.isfile('out.png')
        os.unlink('out.png')

    def test_plot_custom_filename(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(save=True, filename='lc.png')
        assert os.path.isfile('lc.png')
        os.unlink('lc.png')

    def test_plot_axis(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    def test_plot_title(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(title="Test Lightcurve")
        assert plt.fignum_exists(1)

    def test_io_with_ascii(self):
        lc = Lightcurve(self.times, self.counts)
        lc.write('ascii_lc.txt',format_='ascii')
        lc.read('ascii_lc.txt', format_='ascii')
        os.remove('ascii_lc.txt')

    def test_io_with_pickle(self):
        lc = Lightcurve(self.times, self.counts)
        lc.write('lc.pickle', format_='pickle')
        lc.read('lc.pickle',format_='pickle')
        assert np.all(lc.times == self.times)
        assert np.all(lc.counts == self.counts)
        os.remove('lc.pickle')

    def test_io_with_hdf5(self):
        lc = Lightcurve(self.times, self.counts)
        lc.write('lc.hdf5', format_='hdf5')
        
        if _H5PY_INSTALLED:
            data = lc.read('lc.hdf5',format_='hdf5')
            assert np.all(data['times'] == self.times)
            assert np.all(data['counts'] == self.counts)
            os.remove('lc.hdf5')

        else:
            lc.read('lc.pickle',format_='pickle')
            assert np.all(lc.times == self.times)
            assert np.all(lc.counts == self.counts)
            os.remove('lc.pickle')

        
class TestLightcurveRebin(object):

    @classmethod
    def setup_class(cls):
        dt = 0.0001220703125
        n = 1384132
        mean_counts = 2.0
        times = np.arange(dt/2, dt/2 + n*dt, dt)
        counts = np.zeros_like(times) + mean_counts
        cls.lc = Lightcurve(times, counts)

    def test_rebin_even(self):
        dt_new = 2.0
        lc_binned = self.lc.rebin_lightcurve(dt_new)
        assert np.isclose(lc_binned.dt, dt_new)
        counts_test = np.zeros_like(lc_binned.times) + \
            self.lc.counts[0]*dt_new/self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)

    def test_rebin_odd(self):
        dt_new = 1.5
        lc_binned = self.lc.rebin_lightcurve(dt_new)
        assert np.isclose(lc_binned.dt, dt_new)

        counts_test = np.zeros_like(lc_binned.times) + \
            self.lc.counts[0]*dt_new/self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)

    def rebin_several(self, dt):
        """
        TODO: Not sure how to write tests for the rebin method!
        """
        lc_binned = self.lc.rebin_lightcurve(dt)
        assert len(lc_binned.times) == len(lc_binned.counts)

    def test_rebin_equal_numbers(self):
        dt_all = [2, 3, np.pi, 5]
        for dt in dt_all:
            yield self.rebin_several, dt

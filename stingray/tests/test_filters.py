import pytest
import numpy as np

from astropy.modeling import models

from ..filters import Window1D, Optimal1D, filter_for_deadtime, get_deadtime_mask
from ..filters import _paralyzable_dead_time

from stingray.events import EventList


class TestFilters(object):
    @classmethod
    def setup_class(self):
        self.x = np.linspace(0, 10, 100)

        self.amplitude_0 = 5.0
        self.x_0_0 = 5.0
        self.fwhm_0 = 1.0
        self.amplitude_1 = -5
        self.lorentz = models.Lorentz1D(
            amplitude=self.amplitude_0, x_0=self.x_0_0, fwhm=self.fwhm_0
        )
        self.const = models.Const1D(amplitude=self.amplitude_1)
        self.model = self.lorentz + self.const
        self.y = self.model(self.x)

    def test_window(self):
        tophat_filter = Window1D(self.model)
        filtered_y = self.y * tophat_filter(self.x)
        filter_w = [1.0 if np.abs(x_i - self.x_0_0) <= self.fwhm_0 / 2 else 0.0 for x_i in self.x]
        y_w = self.y * filter_w
        assert np.allclose(filtered_y, y_w)

    def test_optimal(self):
        optimal_filter = Optimal1D(self.model)
        filtered_y = self.y * optimal_filter(self.x)
        filter_o = (self.lorentz / self.model)(self.x)
        y_o = self.y * filter_o
        assert np.allclose(filtered_y, y_o)


def test_filter_for_deadtime_nonpar():
    """Test dead time filter, non-paralyzable case."""
    events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
    filt_events, info = filter_for_deadtime(events, 0.11, return_all=True)
    expected = np.array([1, 2, 2.2, 3, 3.2])
    assert np.allclose(filt_events, expected), "Wrong: {} vs {}".format(filt_events, expected)
    assert np.allclose(filt_events, info.uf_events[info.is_event])


def test_filter_for_deadtime_evlist():
    """Test dead time filter, non-paralyzable case."""
    events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
    events = EventList(events)
    events.pi = np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
    events.energy = np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
    events.mjdref = 10
    filt_events = filter_for_deadtime(events, 0.11)

    expected = np.array([1, 2, 2.2, 3, 3.2])
    assert np.allclose(filt_events.time, expected), "Wrong: {} vs {}".format(filt_events, expected)

    assert np.allclose(filt_events.pi, 1)
    assert np.allclose(filt_events.energy, 1)


def test_filter_for_deadtime_lt0():
    """Test dead time filter, non-paralyzable case."""
    events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
    with pytest.raises(ValueError) as excinfo:
        _ = filter_for_deadtime(events, -0.11)
    assert "Dead time is less than 0. Please check." in str(excinfo.value)


def test_filter_for_deadtime_0():
    """Test dead time filter, non-paralyzable case."""
    events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
    filt_events, info = filter_for_deadtime(events, 0, return_all=True)
    assert np.allclose(events, filt_events)
    assert np.allclose(filt_events, info.uf_events[info.is_event])


def test_filter_for_deadtime_nonpar_sigma():
    """Test dead time filter, non-paralyzable case."""
    events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
    filt_events = filter_for_deadtime(events, 0.11, dt_sigma=0.001)
    expected = np.array([1, 2, 2.2, 3, 3.2])
    assert np.allclose(filt_events, expected), "Wrong: {} vs {}".format(filt_events, expected)


def test_filter_for_deadtime_nonpar_bkg():
    """Test dead time filter, non-paralyzable case, with background."""
    events = np.array([1.1, 2, 2.2, 3, 3.2])
    bkg_events = np.array([1, 3.1])
    filt_events, info = filter_for_deadtime(events, 0.11, bkg_ev_list=bkg_events, return_all=True)
    expected_ev = np.array([2, 2.2, 3, 3.2])
    expected_bk = np.array([1])
    assert np.allclose(filt_events, expected_ev), "Wrong: {} vs {}".format(filt_events, expected_ev)
    assert np.allclose(info.bkg, expected_bk), "Wrong: {} vs {}".format(info.bkg, expected_bk)
    assert np.allclose(filt_events, info.uf_events[info.is_event])


def test_filter_for_deadtime_par():
    """Test dead time filter, paralyzable case."""
    events = np.array([1, 1.1, 2, 2.2, 3, 3.1, 3.2])
    assert np.all(filter_for_deadtime(events, 0.11, paralyzable=True) == np.array([1, 2, 2.2, 3]))


def test_paralyzable_dead_time_variable():
    """A short dead time does not end a longer one that is still running."""
    events = np.array([0.0, 1.0, 2.0, 20.0])
    dead_time = np.array([10.0, 0.5, 0.5, 0.5])
    filt_events, mask = _paralyzable_dead_time(events, dead_time)
    assert np.array_equal(mask, [True, False, False, True])
    assert np.array_equal(filt_events, [0.0, 20.0])


def test_filter_for_deadtime_par_sigma():
    """No event is kept while an earlier event's dead time is still running."""
    rng = np.random.default_rng(1)
    events = np.sort(rng.uniform(0, 100, 2000))
    np.random.seed(2)
    mask = get_deadtime_mask(events, 0.05, dt_sigma=0.04, paralyzable=True)
    np.random.seed(2)
    dead_time = np.clip(np.random.normal(0.05, 0.04, events.size), 0, None)
    dead_end = np.maximum.accumulate(events + dead_time)
    assert np.array_equal(mask[1:], events[1:] >= dead_end[:-1])


def test_filter_for_deadtime_par_bkg():
    """Test dead time filter, paralyzable case, with background."""
    events = np.array([1.1, 2, 2.2, 3, 3.2])
    bkg_events = np.array([1, 3.1])
    filt_events, info = filter_for_deadtime(
        events, 0.11, bkg_ev_list=bkg_events, paralyzable=True, return_all=True
    )
    expected_ev = np.array([2, 2.2, 3])
    expected_bk = np.array([1])
    assert np.allclose(filt_events, expected_ev), "Wrong: {} vs {}".format(filt_events, expected_ev)
    assert np.allclose(info.bkg, expected_bk), "Wrong: {} vs {}".format(info.bkg, expected_bk)
    assert np.allclose(filt_events, info.uf_events[info.is_event])

import numpy as np

import pytest
import warnings

from stingray import AveragedCovariancespectrum, Covariancespectrum, Lightcurve
from stingray.events import EventList


class TestCovariancespectrumwithEvents(object):
    def setup_class(self):
        event_list = np.array(
            [
                [1, 2],
                [1, 4],
                [1, 8],
                [3, 6],
                [3, 1],
                [4, 2],
                [4, 4],
                [4, 6],
                [4, 7],
                [4, 5],
                [4, 9],
                [5, 1],
                [5, 6],
                [5, 7],
                [5, 9],
                [5, 3],
                [5, 8],
                [6, 1],
                [6, 3],
                [7, 3],
                [7, 4],
                [7, 5],
                [7, 7],
                [8, 8],
                [9, 2],
                [9, 5],
                [9, 9],
            ]
        )
        self.event_list = event_list

    def test_class_fails_if_events_is_set_but_dt_is_not(self):
        with pytest.raises(ValueError):
            c = Covariancespectrum(self.event_list)

    def test_set_lc_flag_correctly(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.event_list, dt=1)
        assert c.use_lc is False

    def test_covar_without_any_band_interest(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.event_list, dt=1)
        assert len(c.covar) == 9

    def test_creates_band_interest_correctly(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.event_list, dt=1)
        energies = np.arange(1, 10, 1)
        band_low = np.zeros_like(energies)
        band_high = np.zeros_like(energies)
        ediff = np.diff(energies)

        band_low[:-1] = energies[:-1] - 0.5 * ediff
        band_high[:-1] = energies[:-1] + 0.5 * ediff
        band_low[-1] = energies[-1] - 0.5 * ediff[-1]
        band_high[-1] = energies[-1] + 0.5 * ediff[-1]

        band_interest = np.vstack([band_low, band_high]).T

        assert np.allclose(band_interest, c.band_interest)

    def test_init_with_invalid_band_interest(self):
        with pytest.raises(ValueError):
            c = Covariancespectrum(self.event_list, dt=1, band_interest=(1))

    def test_init_with_invalid_ref_band_interest(self):
        with pytest.raises(ValueError):
            c = Covariancespectrum(self.event_list, dt=1, ref_band_interest=(0))

    def test_covar_with_both_bands(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(
                self.event_list, dt=1, band_interest=[(2, 6), (8, 9)], ref_band_interest=(1, 10)
            )
        assert len(c.covar) == 2

    def test_with_separate_reference_band(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(
                self.event_list, dt=1, band_interest=[(2, 6)], ref_band_interest=((7, 10))
            )

    def test_with_unsorted_event_list(self):
        event_list = np.array([[2, 1], [2, 3], [1, 2], [5, 2], [4, 1]])
        with pytest.warns(UserWarning) as record:
            _ = Covariancespectrum(event_list, dt=1)
        assert np.any(["must be sorted" in str(r.message) for r in record])

    def test_with_std_as_iterable(self):
        c = Covariancespectrum(self.event_list, dt=1, std=[1, 2])

    def test_with_std_as_a_single_float(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.event_list, dt=1, std=2.55)

    def test_with_eventlist_object(self):
        e = EventList(time=self.event_list[:, 0], energy=self.event_list[:, 1])
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(e, dt=1)


class TestCovariancewithLightcurves(object):
    def setup_class(self):
        time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        counts1 = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0])
        counts2 = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1])
        counts3 = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0])
        counts4 = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
        counts5 = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1])
        counts6 = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0])
        counts7 = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0])
        counts8 = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0])
        counts9 = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1])

        lc1 = Lightcurve(time, counts1)
        lc2 = Lightcurve(time, counts2)
        lc3 = Lightcurve(time, counts3)
        lc4 = Lightcurve(time, counts4)
        lc5 = Lightcurve(time, counts5)
        lc6 = Lightcurve(time, counts6)
        lc7 = Lightcurve(time, counts7)
        lc8 = Lightcurve(time, counts8)
        lc9 = Lightcurve(time, counts9)

        self.lcs = [lc1, lc2, lc3, lc4, lc5, lc6, lc7, lc8, lc9]

        self.ref_band_interest = Lightcurve(time, np.array([2, 3, 1, 4, 4, 0, 1, 3, 2]))

    def test_class_initializes_with_lightcurves(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.lcs)

    def test_set_lc_flag_correctly(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.lcs)
        assert c.use_lc is True
        assert c.lcs == self.lcs

    def test_single_lc_as_ref_band_interest(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.lcs, ref_band_interest=self.ref_band_interest)

    def test_lc_with_no_band_interest(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.lcs, ref_band_interest=None)

        band_interest = np.vstack([np.arange(len(self.lcs)), np.arange(1, len(self.lcs) + 1, 1)]).T

        assert np.allclose(c.band_interest, band_interest)

    def test_failure_without_valid_band_interest(self):
        with pytest.raises(ValueError):
            c = Covariancespectrum(self.lcs, band_interest=1.0)

    def test_ref_band_interest_created_correctly(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.lcs, ref_band_interest=None)
        assert len(c.ref_band_lcs) == len(self.lcs)

    def test_class_fails_with_wrong_number_of_ref_band_interest_lcs(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.lcs, ref_band_interest=self.lcs[::-1])
        assert len(c.lcs) == len(c.ref_band_lcs)

    def test_ref_band_interest_get_saved_correctly(self):
        with pytest.raises(AssertionError):
            c = Covariancespectrum(self.lcs, ref_band_interest=self.lcs[:-1])


class TestAveragedCovariancespectrum(object):
    def setup_class(self):
        event_list = np.array(
            [
                [1, 2],
                [1, 4],
                [1, 8],
                [3, 6],
                [3, 1],
                [4, 2],
                [4, 4],
                [4, 6],
                [4, 7],
                [4, 5],
                [4, 9],
                [5, 1],
                [5, 6],
                [5, 7],
                [5, 9],
                [5, 3],
                [5, 8],
                [6, 1],
                [6, 3],
                [7, 3],
                [7, 4],
                [7, 5],
                [7, 7],
                [8, 8],
                [9, 2],
                [9, 5],
                [9, 9],
                [10, 1],
                [10, 2],
            ]
        )
        self.event_list = event_list

    def test_with_full_segment(self):
        with pytest.warns(UserWarning, match="excess variance"):
            c = Covariancespectrum(self.event_list, 1)
        with pytest.warns(UserWarning, match="excess variance"):
            avg_c = AveragedCovariancespectrum(self.event_list, segment_size=10, dt=1)
        assert np.allclose(avg_c.unnorm_covar, c.unnorm_covar)

    def test_with_two_segments(self):
        with pytest.warns(UserWarning, match="excess variance"):
            avg_c = AveragedCovariancespectrum(self.event_list, segment_size=5, dt=1)

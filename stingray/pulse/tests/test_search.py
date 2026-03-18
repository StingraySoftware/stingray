import numpy as np
import pytest
import matplotlib.pyplot as plt

from stingray.pulse import epoch_folding_search, z_n_search
from stingray.pulse import phase_dispersion_search
from stingray.pulse import phaseogram, plot_phaseogram
from stingray.pulse.search import _profile_fast
from stingray.pulse import plot_profile
from stingray.pulse import fold_events
from stingray import Lightcurve
from stingray.events import EventList

pytestmark = pytest.mark.slow

np.random.seed(20150907)


class TestAll(object):
    """Unit tests for the stingray.pulse.search module."""

    @classmethod
    def setup_class(cls):
        cls.pulse_frequency = 1 / 0.101
        cls.tstart = 0
        cls.tend = 25.25
        cls.tseg = cls.tend - cls.tstart
        cls.dt = 0.01212
        cls.times = np.arange(cls.tstart, cls.tend, cls.dt) + cls.dt / 2
        cls.counts = 100 + 20 * np.cos(2 * np.pi * cls.times * cls.pulse_frequency)
        cls.gti = [[cls.tstart, cls.tend]]
        lc = Lightcurve(cls.times, cls.counts, gti=cls.gti, err_dist="gauss")
        events = EventList()
        events.simulate_times(lc)
        counts = np.random.poisson(lc.counts)
        cls.lc = Lightcurve(lc.time, counts)
        cls.event_times = events.time

    def test_prepare(self):
        pass

    def test_phaseogram(self):
        phaseogr, phases, times, additional_info = phaseogram(
            self.event_times, self.pulse_frequency
        )
        assert np.all(times < 25.6)
        assert np.any(times > 25)
        assert np.all((phases >= 0) & (phases <= 2))
        plt.close(plt.gcf())

    def test_phaseogram_bad_weights(self):
        with pytest.raises(ValueError) as excinfo:
            phaseogr, phases, times, additional_info = phaseogram(
                self.event_times, self.pulse_frequency, weights=[0, 2]
            )
        assert "must match" in str(excinfo.value)

    def test_phaseogram_weights(self):
        phaseogr, phases, times, additional_info = phaseogram(
            self.times, self.pulse_frequency, weights=self.counts, nph=16
        )
        assert np.all(times < 25.6)
        assert np.any(times > 25)
        assert np.all((phases >= 0) & (phases <= 2))
        fig = plt.figure("Phaseogram direct weights")
        plot_phaseogram(phaseogr, phases, times)
        plt.savefig("phaseogram_weights.png")
        plt.close(fig)

    def test_phaseogram_mjdref(self):
        phaseogr, phases, times, additional_info = phaseogram(
            self.event_times,
            self.pulse_frequency,
            mjdref=57000,
            out_filename="phaseogram_mjdref.png",
        )
        assert np.all(times >= 57000)
        assert np.all((phases >= 0) & (phases <= 2))
        plt.close(plt.gcf())

    def test_phaseogram_mjdref_pepoch(self):
        phaseogr, phases, times, additional_info = phaseogram(
            self.event_times,
            self.pulse_frequency,
            mjdref=57000,
            out_filename="phaseogram_mjdref.png",
            pepoch=57000,
        )
        assert np.all(times >= 57000)
        assert np.all((phases >= 0) & (phases <= 2))

    def test_plot_phaseogram_fromfunc(self):
        fig = plt.figure("Phaseogram from func")
        ax = plt.subplot()
        phaseogr, phases, times, additional_info = phaseogram(
            self.event_times,
            self.pulse_frequency,
            mjdref=57000,
            pepoch=57000,
            phaseogram_ax=ax,
            plot=True,
        )
        plt.savefig("phaseogram_fromfunc.png")
        plt.close(fig)

    def test_plot_phaseogram_direct(self):
        phaseogr, phases, times, additional_info = phaseogram(
            self.event_times, self.pulse_frequency
        )
        plot_phaseogram(phaseogr, phases, times)
        plt.savefig("phaseogram_direct.png")
        plt.close(plt.gcf())

    def test_search_wrong_key_fails(self):
        with pytest.raises(
            ValueError, match=r"Unidentified keyword\(s\) to fold_events: fdot, fddot"
        ):
            phase, prof, _ = fold_events(self.event_times, self.pulse_frequency, fdot=12, fddot=34)

    def test_plot_profile(self):
        phase, prof, _ = fold_events(self.event_times, self.pulse_frequency)
        ax = plot_profile(phase, prof)
        plt.savefig("profile_direct.png")
        plt.close(plt.gcf())

    def test_plot_profile_existing_ax(self):
        fig = plt.figure("Pulse profile")
        ax = plt.subplot()
        phase, prof, _ = fold_events(self.event_times, self.pulse_frequency)
        ax = plot_profile(phase, prof, ax=ax)
        plt.savefig("profile_existing_ax.png")
        plt.close(fig)

    def test_plot_profile_errorbars(self):
        fig = plt.figure("Pulse profile")
        ax = plt.subplot()
        phase, prof, err = fold_events(self.event_times, self.pulse_frequency)

        ax = plot_profile(phase, prof, err=err, ax=ax)
        plt.savefig("profile_errorbars.png")
        plt.close(fig)

    def test_profile_fast(self):
        test_phase = np.arange(0, 1, 1 / 16)
        prof = _profile_fast(test_phase, nbin=16)
        assert np.allclose(prof, np.ones(16))

    def test_epoch_folding_search(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        freq, stat = epoch_folding_search(self.event_times, frequencies, nbin=43)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)

    def test_phase_dispersion_search(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        freq, stat = phase_dispersion_search(self.lc.time, self.lc.counts, frequencies, nbin=43)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        minstatbin = freq[np.argmin(stat)]
        assert np.allclose(minstatbin, frequencies[minbin], atol=0.1 / self.tseg)

    def test_epoch_folding_search_fdot(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        fdots = [-0.1, 0, 0.1]
        freq, fdot, stat = epoch_folding_search(self.event_times, frequencies, nbin=43, fdots=fdots)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq.flatten()[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / self.tseg)

    def test_phase_dispersion_search_fdot(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        fdots = [-0.1, 0, 0.1]
        freq, fdot, stat = phase_dispersion_search(
            self.lc.time, self.lc.counts, frequencies, nbin=43, fdots=fdots
        )

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        minstatbin = freq.flatten()[np.argmin(stat)]
        assert np.allclose(minstatbin, frequencies[minbin], atol=0.1 / self.tseg)
        minfdot = fdot.flatten()[np.argmin(stat)]
        assert np.allclose(minfdot, 0.0, atol=0.1 / self.tseg)

    def test_epoch_folding_search_fdot_longdouble(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg, dtype=np.longdouble)
        fdots = np.array([-0.1, 0, 0.1], dtype=np.longdouble)
        freq, fdot, stat = epoch_folding_search(
            self.event_times.astype(np.longdouble), frequencies, nbin=43, fdots=fdots
        )

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq.flatten()[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / self.tseg)

    def test_epoch_folding_search_expocorr_fails(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        with pytest.raises(ValueError) as excinfo:
            freq, stat = epoch_folding_search(self.event_times, frequencies, nbin=23, expocorr=True)
        assert "To calculate exposure correction" in str(excinfo.value)

    def test_epoch_folding_search_expocorr(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        with pytest.warns(UserWarning):
            freq, stat = epoch_folding_search(
                self.event_times, frequencies, nbin=42, expocorr=True, gti=self.gti
            )

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)

    def test_epoch_folding_search_expocorr_fdot(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        fdots = [-0.1, 0, 0.1]
        with pytest.warns(UserWarning):
            freq, fdot, stat = epoch_folding_search(
                self.event_times, frequencies, nbin=42, expocorr=True, gti=self.gti, fdots=fdots
            )

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq.flatten()[np.argmax(stat)]

        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / self.tseg)

    def test_epoch_folding_search_weights(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        freq, stat = epoch_folding_search(self.times, frequencies, nbin=16, weights=self.counts)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)

    def test_epoch_folding_search_weights_fdot(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        fdots = [-0.1, 0, 0.1]
        freq, fdot, stat = epoch_folding_search(
            self.times, frequencies, nbin=16, weights=self.counts, fdots=fdots
        )

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq.flatten()[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / self.tseg)

    def test_z_n_search(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        freq, stat = z_n_search(self.event_times, frequencies, nbin=25, nharm=2)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)

    def test_z_n_search_fdot(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        fdots = [-0.1, 0, 0.1]
        freq, fdot, stat = z_n_search(self.event_times, frequencies, nbin=25, nharm=2, fdots=fdots)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq.flatten()[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / self.tseg)

    def test_z_n_search_fdot_longdouble(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg, dtype=np.longdouble)
        fdots = np.array([-0.1, 0, 0.1], dtype=np.longdouble)
        freq, fdot, stat = z_n_search(
            self.event_times.astype(np.longdouble), frequencies, nbin=25, nharm=2, fdots=fdots
        )

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq.flatten()[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / self.tseg)

    def test_z_n_search_expocorr(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        with pytest.warns(UserWarning):
            freq, stat = z_n_search(
                self.event_times, frequencies, nbin=64, nharm=2, expocorr=True, gti=self.gti
            )

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)

    def test_z_n_search_expocorr_fdot(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        fdots = [-0.1, 0, 0.1]
        with pytest.warns(UserWarning):
            freq, fdot, stat = z_n_search(
                self.event_times,
                frequencies,
                nbin=64,
                nharm=2,
                expocorr=True,
                gti=self.gti,
                fdots=fdots,
            )

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq.flatten()[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / self.tseg)

    def test_z_n_search_expocorr_fails(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        with pytest.raises(ValueError) as excinfo:
            freq, stat = z_n_search(self.event_times, frequencies, nharm=1, nbin=35, expocorr=True)
        assert "To calculate exposure correction" in str(excinfo.value)

    def test_z_n_search_weights(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        freq, stat = z_n_search(self.times, frequencies, nbin=44, nharm=1, weights=self.counts)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)

    def test_z_n_search_weights_fdot(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.8, 9.99, 0.1 / self.tseg)
        fdots = [-0.1, 0, 0.1]
        freq, fdot, stat = z_n_search(
            self.times, frequencies, nbin=44, nharm=1, weights=self.counts, fdots=fdots
        )

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq.flatten()[np.argmax(stat)]
        assert np.allclose(maxstatbin, frequencies[minbin], atol=0.1 / self.tseg)
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / self.tseg)

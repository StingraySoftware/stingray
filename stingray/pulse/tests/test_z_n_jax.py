import numpy as np
import pytest

from stingray.pulse.search import z_n_search, HAS_JAX
from stingray import Lightcurve
from stingray.events import EventList


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pulsed_events():
    """Generate a reproducible pulsed event list for testing."""
    np.random.seed(20150907)
    pulse_frequency = 1 / 0.101
    tstart = 0
    tend = 25.25
    dt = 0.01212
    times = np.arange(tstart, tend, dt) + dt / 2
    counts = 100 + 20 * np.cos(2 * np.pi * times * pulse_frequency)
    gti = [[tstart, tend]]
    lc = Lightcurve(times, counts, gti=gti, err_dist="gauss")
    events = EventList()
    events.simulate_times(lc)
    return {
        "event_times": events.time,
        "pulse_frequency": pulse_frequency,
        "tseg": tend - tstart,
        "frequencies": np.arange(9.8, 9.99, 0.1 / (tend - tstart)),
    }


class TestZnSearchNoJax:
    """Tests for z_n_search without JAX (standard path)."""

    def test_z_n_search_1d(self, pulsed_events):
        """Test that standard z_n_search finds the correct frequency."""
        ev = pulsed_events
        freq, stat = z_n_search(ev["event_times"], ev["frequencies"], nharm=2, nbin=25)
        minbin = np.argmin(np.abs(ev["frequencies"] - ev["pulse_frequency"]))
        maxstatbin = freq[np.argmax(stat)]
        assert np.allclose(maxstatbin, ev["frequencies"][minbin], atol=0.1 / ev["tseg"])

    def test_z_n_search_1d_output_shape(self, pulsed_events):
        """Test output shape for 1D search."""
        ev = pulsed_events
        freq, stat = z_n_search(ev["event_times"], ev["frequencies"], nharm=2, nbin=25)
        assert freq.shape == ev["frequencies"].shape
        assert stat.shape == ev["frequencies"].shape

    def test_z_n_search_2d(self, pulsed_events):
        """Test that standard z_n_search with fdots finds correct peak."""
        ev = pulsed_events
        fdots = [-0.1, 0, 0.1]
        freq, fdot, stat = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, nbin=25, fdots=fdots
        )
        minbin = np.argmin(np.abs(ev["frequencies"] - ev["pulse_frequency"]))
        maxstatbin = freq.flatten()[np.argmax(stat)]
        assert np.allclose(maxstatbin, ev["frequencies"][minbin], atol=0.1 / ev["tseg"])
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / ev["tseg"])


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
class TestZnSearchJax:
    """Tests for z_n_search with JAX acceleration."""

    def test_z_n_search_jax_1d(self, pulsed_events):
        """Test that JAX z_n_search finds the correct frequency."""
        ev = pulsed_events
        freq, stat = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, use_jax=True
        )
        minbin = np.argmin(np.abs(ev["frequencies"] - ev["pulse_frequency"]))
        maxstatbin = freq[np.argmax(stat)]
        assert np.allclose(maxstatbin, ev["frequencies"][minbin], atol=0.1 / ev["tseg"])

    def test_z_n_search_jax_1d_output_shape(self, pulsed_events):
        """Test output shape for JAX 1D search."""
        ev = pulsed_events
        freq, stat = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, use_jax=True
        )
        assert freq.shape == ev["frequencies"].shape
        assert stat.shape == ev["frequencies"].shape

    def test_z_n_search_jax_2d(self, pulsed_events):
        """Test that JAX z_n_search with fdots finds the correct peak."""
        ev = pulsed_events
        fdots = [-0.1, 0, 0.1]
        freq, fdot, stat = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, fdots=fdots, use_jax=True
        )
        minbin = np.argmin(np.abs(ev["frequencies"] - ev["pulse_frequency"]))
        maxstatbin = freq.flatten()[np.argmax(stat)]
        assert np.allclose(maxstatbin, ev["frequencies"][minbin], atol=0.1 / ev["tseg"])
        maxfdot = fdot.flatten()[np.argmax(stat)]
        assert np.allclose(maxfdot, 0.0, atol=0.1 / ev["tseg"])

    def test_z_n_search_jax_2d_output_shape(self, pulsed_events):
        """Test output shape for JAX 2D search."""
        ev = pulsed_events
        fdots = np.array([-0.1, 0, 0.1])
        freq, fdot, stat = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, fdots=fdots, use_jax=True
        )
        assert freq.shape == (len(fdots), len(ev["frequencies"]))
        assert fdot.shape == (len(fdots), len(ev["frequencies"]))
        assert stat.shape == (len(fdots), len(ev["frequencies"]))

    def test_z_n_search_jax_scalar_fdot(self, pulsed_events):
        """Test that scalar fdot=0 returns 1D output (same as no fdots)."""
        ev = pulsed_events
        freq, stat = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, fdots=0, use_jax=True
        )
        assert freq.ndim == 1
        assert stat.ndim == 1

    def test_z_n_search_jax_list_fdots(self, pulsed_events):
        """Test that passing fdots as a Python list works correctly."""
        ev = pulsed_events
        fdots_list = [-0.1, 0, 0.1]
        freq, fdot, stat = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, fdots=fdots_list, use_jax=True
        )
        assert stat.shape == (len(fdots_list), len(ev["frequencies"]))

    def test_z_n_search_jax_different_nharm(self, pulsed_events):
        """Test JAX search with different number of harmonics."""
        ev = pulsed_events
        for nharm in [1, 2, 4]:
            freq, stat = z_n_search(
                ev["event_times"], ev["frequencies"], nharm=nharm, use_jax=True
            )
            assert stat.shape == ev["frequencies"].shape
            # Peak should still be near the true frequency
            minbin = np.argmin(np.abs(ev["frequencies"] - ev["pulse_frequency"]))
            maxstatbin = freq[np.argmax(stat)]
            assert np.allclose(
                maxstatbin, ev["frequencies"][minbin], atol=0.1 / ev["tseg"]
            )


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
class TestZnSearchJaxVsStandard:
    """Comparison tests: JAX results should agree with the standard method."""

    def test_comparison_1d_peak_location(self, pulsed_events):
        """Both methods should find the same peak frequency."""
        ev = pulsed_events
        freq_std, stat_std = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, nbin=128
        )
        freq_jax, stat_jax = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, use_jax=True
        )
        peak_std = freq_std[np.argmax(stat_std)]
        peak_jax = freq_jax[np.argmax(stat_jax)]
        assert np.allclose(peak_std, peak_jax, atol=0.1 / ev["tseg"])

    def test_comparison_2d_peak_location(self, pulsed_events):
        """Both methods should find the same peak in (frequency, fdot) space."""
        ev = pulsed_events
        fdots = [-0.1, 0, 0.1]
        f_std, fd_std, s_std = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, nbin=128, fdots=fdots
        )
        f_jax, fd_jax, s_jax = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, fdots=fdots, use_jax=True
        )
        # Same grid shapes
        assert f_std.shape == f_jax.shape
        assert fd_std.shape == fd_jax.shape
        assert s_std.shape == s_jax.shape

        # Same grids
        np.testing.assert_array_equal(f_std, f_jax)
        np.testing.assert_array_equal(fd_std, fd_jax)

        # Same peak location
        peak_f_std = f_std.flatten()[np.argmax(s_std)]
        peak_f_jax = f_jax.flatten()[np.argmax(s_jax)]
        assert np.allclose(peak_f_std, peak_f_jax, atol=0.1 / ev["tseg"])

        peak_fd_std = fd_std.flatten()[np.argmax(s_std)]
        peak_fd_jax = fd_jax.flatten()[np.argmax(s_jax)]
        assert np.allclose(peak_fd_std, peak_fd_jax, atol=0.1 / ev["tseg"])

    def test_comparison_1d_stat_values(self, pulsed_events):
        """JAX and standard Z^2_n statistics should be correlated.

        Note: exact values may differ because the standard method uses binned
        profiles while JAX computes the exact unbinned statistic. We check that
        the overall shape (ranking of frequencies) is consistent.
        """
        ev = pulsed_events
        _, stat_std = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, nbin=128
        )
        _, stat_jax = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, use_jax=True
        )
        # The rank correlation should be very high
        from scipy.stats import spearmanr

        corr, _ = spearmanr(stat_std, stat_jax)
        assert corr > 0.95, f"Spearman correlation {corr} is too low"

    def test_comparison_returns_numpy(self, pulsed_events):
        """JAX path should return numpy arrays, not JAX arrays."""
        ev = pulsed_events
        freq, stat = z_n_search(
            ev["event_times"], ev["frequencies"], nharm=2, use_jax=True
        )
        # stat.flatten() returns numpy in the current code path
        assert isinstance(freq, np.ndarray)
        stat_np = np.asarray(stat)
        assert stat_np.dtype in [np.float32, np.float64]


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
class TestZnSearchBenchmark:
    """Benchmark comparing JAX vs standard z_n_search performance."""

    def test_benchmark_1d(self, pulsed_events, capsys):
        """Benchmark 1D z_n_search: JAX vs standard."""
        import time

        ev = pulsed_events

        # Warmup both paths (numba JIT + JAX JIT)
        z_n_search(ev["event_times"], ev["frequencies"], nharm=2, nbin=128)
        z_n_search(ev["event_times"], ev["frequencies"], nharm=2, use_jax=True)

        n_runs = 5

        # Benchmark standard
        std_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            z_n_search(ev["event_times"], ev["frequencies"], nharm=2, nbin=128)
            std_times.append(time.perf_counter() - t0)

        # Benchmark JAX
        jax_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            z_n_search(ev["event_times"], ev["frequencies"], nharm=2, use_jax=True)
            jax_times.append(time.perf_counter() - t0)

        std_median = np.median(std_times)
        jax_median = np.median(jax_times)
        speedup = std_median / jax_median if jax_median > 0 else float("inf")

        with capsys.disabled():
            print(
                f"\n  1D benchmark ({len(ev['frequencies'])} frequencies, "
                f"{len(ev['event_times'])} events, {n_runs} runs):"
            )
            print(f"    Standard: {std_median:.4f}s (median)")
            print(f"    JAX:      {jax_median:.4f}s (median)")
            print(f"    Speedup:  {speedup:.2f}x")

        # Sanity check: both should complete without error (no assertion on speed,
        # as relative performance depends on hardware and problem size)
        assert std_median > 0
        assert jax_median > 0

    def test_benchmark_2d(self, pulsed_events, capsys):
        """Benchmark 2D z_n_search (with fdots): JAX vs standard.

        On CPU-only environments, the speedup may be modest (~1.5-2x) since
        the standard path uses numba-compiled loops. Larger gains are expected
        on GPU-enabled JAX backends.
        """
        import time

        ev = pulsed_events
        frequencies = np.linspace(9.5, 10.5, 200)
        fdots = np.linspace(-0.5, 0.5, 3)

        # Warmup both paths (numba JIT + JAX JIT)
        z_n_search(
            ev["event_times"], frequencies, nharm=2, nbin=128, fdots=fdots
        )
        z_n_search(
            ev["event_times"], frequencies, nharm=2, fdots=fdots, use_jax=True
        )

        n_runs = 3

        # Benchmark standard
        std_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            z_n_search(
                ev["event_times"], frequencies, nharm=2, nbin=128, fdots=fdots
            )
            std_times.append(time.perf_counter() - t0)

        # Benchmark JAX
        jax_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            z_n_search(
                ev["event_times"], frequencies, nharm=2, fdots=fdots, use_jax=True
            )
            jax_times.append(time.perf_counter() - t0)

        std_median = np.median(std_times)
        jax_median = np.median(jax_times)
        speedup = std_median / jax_median if jax_median > 0 else float("inf")

        with capsys.disabled():
            print(
                f"\n  2D benchmark ({len(frequencies)} frequencies x "
                f"{len(fdots)} fdots, {len(ev['event_times'])} events, {n_runs} runs):"
            )
            print(f"    Standard: {std_median:.4f}s (median)")
            print(f"    JAX:      {jax_median:.4f}s (median)")
            print(f"    Speedup:  {speedup:.2f}x")

        assert std_median > 0
        assert jax_median > 0

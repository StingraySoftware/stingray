import os
import pytest
import numpy as np
from scipy.interpolate import interp1d

from stingray.lightcurve import Lightcurve
from stingray.powerspectrum import AveragedPowerspectrum
from stingray.deadtime.model import r_det, r_in, pds_model_zhang, non_paralyzable_dead_time_model
from stingray.deadtime.model import check_A, check_B, A, B
from stingray.filters import filter_for_deadtime
from stingray.utils import HAS_NUMBA


pytestmark = pytest.mark.slow


def simulate_events(rate, length, deadtime=2.5e-3, bkg_rate=0.0, **filter_kwargs):
    events = np.random.uniform(0, length, int(rate * length))

    events = np.sort(events)

    events_back = None
    if bkg_rate > 0:
        events_back = np.sort(np.random.uniform(0, length, int(rate * length)))

    events_dt = filter_for_deadtime(events, deadtime, bkg_ev_list=events_back, **filter_kwargs)
    return events, events_dt


def test_deadtime_conversion():
    """Test the functions for count rate conversion."""
    original_rate = np.arange(1, 1000, 10)
    deadtime = 2.5e-3
    rdet = r_det(deadtime, original_rate)
    rin = r_in(deadtime, rdet)
    np.testing.assert_almost_equal(rin, original_rate)


@pytest.mark.parametrize("rate", [1.0, 100.0])
def test_zhang_model_accurate(rate):
    bintime = 0.0002
    deadtime = 2.5e-3
    length = 2000
    fftlen = 10

    _, events_dt = simulate_events(rate, length, deadtime=deadtime)
    lc_dt = Lightcurve.make_lightcurve(events_dt, bintime, tstart=0, tseg=length)
    pds = AveragedPowerspectrum(lc_dt, fftlen, norm="leahy")

    zh_f, zh_p = pds_model_zhang(100, rate, deadtime, bintime, limit_k=600)

    deadtime_fun = interp1d(zh_f, zh_p, bounds_error=False, fill_value="extrapolate")
    ratio = pds.power / deadtime_fun(pds.freq)
    assert np.isclose(np.mean(ratio), 1, rtol=0.01)
    assert np.isclose(np.std(ratio), 1 / np.sqrt(pds.m), rtol=0.1)


@pytest.mark.skipif("not HAS_NUMBA")
@pytest.mark.parametrize("rates", [(1.0, 0.0), (1.0, 1.0), (100.0, 10.0), (100, 200)])
def test_non_paralyzable_model_accurate(rates):
    bintime = 0.0002
    deadtime = 2.5e-3
    length = 2000
    fftlen = 10
    rate, bkg_rate = rates
    events, events_dt = simulate_events(rate, length, deadtime=deadtime, bkg_rate=bkg_rate)
    det_rate = events_dt.size / length
    det_bkg_rate = events.size / length - det_rate

    lc_dt = Lightcurve.make_lightcurve(events_dt, bintime, tstart=0, tseg=length)
    pds = AveragedPowerspectrum(lc_dt, fftlen, norm="leahy")

    model_power = non_paralyzable_dead_time_model(
        pds.freq, deadtime, rate=det_rate, background_rate=det_bkg_rate
    )
    ratio = pds.power / model_power
    assert np.isclose(np.mean(ratio), 1, rtol=0.01)


@pytest.mark.parametrize("is_incident", [True, False])
def test_checkA(is_incident):
    check_A(300, 2.5e-3, 0.001, max_k=100, save_to="check_A.png", rate_is_incident=is_incident)
    assert os.path.exists("check_A.png")
    os.unlink("check_A.png")


@pytest.mark.parametrize("is_incident", [True, False])
def test_checkB(is_incident):
    check_B(300, 2.5e-3, 0.001, max_k=100, save_to="check_B.png", rate_is_incident=is_incident)
    assert os.path.exists("check_B.png")
    os.unlink("check_B.png")


@pytest.mark.parametrize("tb", [0.0001, 0.1])
def test_A_and_B_array(tb):
    td = 2.5e-3
    ks = np.array([1, 5, 20, 70])
    rate = 10
    tau = 1 / rate
    r0 = r_det(td, rate)
    assert np.array_equal(
        np.array([A(k, r0, td, tb, tau) for k in ks]), A(ks, r0, td, tb, tau), equal_nan=False
    )
    assert np.array_equal(
        np.array([B(k, r0, td, tb, tau) for k in ks]), B(ks, r0, td, tb, tau), equal_nan=False
    )


def test_pds_model_warns():
    with pytest.warns(UserWarning, match="The bin time is much larger than the "):
        pds_model_zhang(10, 100.0, 2.5e-3, 0.1, limit_k=10)


def test_non_paralyzable_model_fail_bad_rate():
    """Fail if combined rate is larger than 1 / deadtime."""
    with pytest.raises(
        ValueError,
        match="The sum of the source and background count rates is larger than the inverse",
    ):
        non_paralyzable_dead_time_model(np.arange(10), 2.5e-3, rate=300, background_rate=300)


@pytest.mark.skipif("not HAS_NUMBA")
def test_non_paralyzable_model_restricted_band():
    """The model on a restricted band must agree with the full grid sliced to it.

    The Zhang grid always spans 0 to the Nyquist frequency of ``bin_time``, so its
    resolution has to be derived from ``bin_time`` and the spacing of the requested
    frequencies. Deriving it from ``max(freqs)`` instead gives a 25% error here.
    """
    bin_time = 1 / 2048
    deadtime = 2.5e-3
    rate = 200.0

    full = np.fft.rfftfreq(8192, bin_time)[1:]
    band = (full >= 0.5) & (full <= 10.0)

    on_full = non_paralyzable_dead_time_model(full, deadtime, rate, bin_time=bin_time)[band]
    on_band = non_paralyzable_dead_time_model(full[band], deadtime, rate, bin_time=bin_time)

    assert np.allclose(on_band, on_full, rtol=0.01)


@pytest.mark.skipif("not HAS_NUMBA")
@pytest.mark.parametrize("f_live", [0.75, 0.5, 0.25])
def test_non_paralyzable_model_adaptive_limit_k(f_live):
    """The ``limit_k=None`` default must be converged across the live time grid.

    The number of inner-sum terms needed grows as the live time falls, so the default
    is derived from the rate; here it is checked against a large fixed ``limit_k``.
    """
    bin_time = 1 / 2048
    deadtime = 2.5e-3
    rate = (1 - f_live) / deadtime
    freqs = np.fft.rfftfreq(2048, bin_time)[1:]

    kwargs = dict(bin_time=bin_time, n_approx=256)
    reference = non_paralyzable_dead_time_model(freqs, deadtime, rate, limit_k=200, **kwargs)
    default = non_paralyzable_dead_time_model(freqs, deadtime, rate, **kwargs)

    assert np.allclose(default, reference, rtol=0.001)


@pytest.mark.skipif("not HAS_NUMBA")
def test_non_paralyzable_model_limit_k_capped():
    """At very low live time the adaptive ``limit_k`` is capped, with a warning.

    The model does not converge there whatever ``limit_k`` is, so an uncapped
    rate-derived value would only make the call slow for nothing.
    """
    with pytest.warns(UserWarning, match="does not converge"):
        non_paralyzable_dead_time_model(
            np.arange(1, 10.0), 2.5e-3, 395.0, bin_time=1 / 2048, n_approx=16
        )


@pytest.mark.skipif("not HAS_NUMBA")
@pytest.mark.parametrize("bins_per_dead_time", [5, 41, 164])
def test_default_limit_k_follows_the_bin_time(bins_per_dead_time):
    """The dead-time correlations span a fixed *time*, so the number of inner-sum terms
    grows as dead_time / bin_time. Whatever the binning, the low-frequency level must be
    2 f_live^2; a limit_k from the live time alone was 2.5x too high at 164 bins."""
    deadtime = 2.5e-3
    rate = 200.0
    f_live = 1 - rate * deadtime
    bin_time = deadtime / bins_per_dead_time

    level = non_paralyzable_dead_time_model(np.array([1.0]), deadtime, rate, bin_time=bin_time)
    # A long grid: Zhang's (N - k)/N weights bias a short one by 2% at 164 bins per td.
    _, zhang = pds_model_zhang(65536, rate, deadtime, bin_time, rate_is_incident=False)

    assert level[0] == pytest.approx(2 * f_live**2, rel=0.01)
    assert zhang[1] == pytest.approx(2 * f_live**2, rel=0.01)

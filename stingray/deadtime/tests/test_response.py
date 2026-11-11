import numpy as np
import pytest

from stingray.deadtime.model import (
    deadtime_response,
    pds_model_zhang,
    r_det,
    vikhlinin_dead_time_model,
)
from stingray.utils import HAS_NUMBA

DEAD_TIME = 2.5e-3


def test_deadtime_response_is_one_at_low_frequency_and_a_delayed_sinc():
    """The unbinned response is a running mean over the last dead time: 1 at f=0,
    zero at f=1/td, and with the phase of a delay of td/2."""
    freqs = np.array([1e-6, 0.5 / DEAD_TIME, 1 / DEAD_TIME])
    response = deadtime_response(freqs, DEAD_TIME)

    assert response[0] == pytest.approx(1.0, abs=1e-6)
    assert np.angle(response[1]) == pytest.approx(-np.pi / 2)
    assert abs(response[2]) < 1e-12


def test_binned_response_tends_to_the_unbinned_one_for_fine_bins():
    """Aliasing only matters near the Nyquist frequency of the bins; far below it the
    binned response is the unbinned one, and the difference shrinks with the bin."""
    freqs = np.linspace(1, 200, 50)
    exact = deadtime_response(freqs, DEAD_TIME)
    coarse = deadtime_response(freqs, DEAD_TIME, bin_time=DEAD_TIME / 4)
    fine = deadtime_response(freqs, DEAD_TIME, bin_time=DEAD_TIME / 64)

    assert np.max(np.abs(fine - exact)) < 1e-3
    assert np.max(np.abs(fine - exact)) < np.max(np.abs(coarse - exact))


@pytest.mark.parametrize("bin_time", [None, DEAD_TIME / 164])
def test_vikhlinin_model_low_frequency_level(bin_time):
    """Well below 1/td the Leahy level is 2 f_live^2 whatever the binning, including
    the 164 bins per dead time where the Zhang model with its default limit_k fails."""
    rate = 200.0
    f_live = 1 - rate * DEAD_TIME
    level = vikhlinin_dead_time_model(np.array([0.5]), DEAD_TIME, rate, bin_time=bin_time)

    assert level[0] == pytest.approx(2 * f_live**2, rel=1e-3)


@pytest.mark.skipif("not HAS_NUMBA")
@pytest.mark.parametrize("bins_per_dead_time", [2, 5])
def test_binned_vikhlinin_model_matches_zhang(bins_per_dead_time):
    """The Zhang model is the unbinned Vikhlinin model aliased by the bin window, so the
    binned form must agree with it at every frequency up to Nyquist, not just low down.
    Zhang needs a long grid: its (N - k)/N weights bias it by 2% at N = 256."""
    incident = 400.0
    bin_time = DEAD_TIME / bins_per_dead_time
    freqs, zhang = pds_model_zhang(2048, incident, DEAD_TIME, bin_time, limit_k=100)
    model = vikhlinin_dead_time_model(
        freqs[1:], DEAD_TIME, r_det(DEAD_TIME, incident), bin_time=bin_time
    )

    assert np.allclose(model, zhang[1:], rtol=3e-3)


def test_vikhlinin_model_rejects_an_impossible_rate():
    """A detected rate above 1/td cannot come out of a non-paralyzable detector."""
    with pytest.raises(ValueError, match="inverse of the dead time"):
        vikhlinin_dead_time_model(np.arange(1, 10.0), DEAD_TIME, 300, background_rate=300)

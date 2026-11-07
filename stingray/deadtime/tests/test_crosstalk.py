import numpy as np
import pytest

from stingray import EventList
from stingray.crossspectrum import AveragedCrossspectrum
from stingray.deadtime.crosstalk import (
    coherence_from_cross_spectral_matrix,
    cross_module_cross_spectral_matrix,
    deadtime_mixing_matrix,
    live_time_fraction,
    unmix_cross_spectral_matrix,
)
from stingray.filters import filter_for_deadtime

NORMS = ["none", "leahy", "frac"]


def _true_matrix(power, coherence, phase):
    """A true 2x2 cross-spectral matrix with equal band powers."""
    cross = power * np.sqrt(coherence) * np.exp(-1j * phase)
    return power, power, cross


def _mix(power1, power2, cross, matrix):
    """Apply ``S_meas = M S_true M^T`` to the three independent entries."""
    s = np.array([[power1, cross], [np.conj(cross), power2]])
    out = matrix @ s @ matrix.T
    return out[0, 0].real, out[1, 1].real, out[0, 1]


# ---------------------------------------------------------------- the live time


def test_live_time_fraction_is_exact_for_non_paralyzable_dead_time():
    """``1 - r_det t_d`` must equal the surviving fraction of a dead-time filter.

    The identity is exact for non-paralyzable dead time, so a user with no
    housekeeping file can get ``f_live`` from the detected rate alone.
    """
    dead_time, length, rate = 2.5e-3, 2000.0, 200.0
    rng = np.random.default_rng(1234)
    times = np.sort(rng.uniform(0, length, int(rate * length)))
    kept = filter_for_deadtime(times, dead_time)

    surviving = kept.size / times.size
    assert live_time_fraction(kept.size / length, dead_time) == pytest.approx(surviving, rel=0.01)


# ------------------------------------------------------------- the mixing matrix


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("band_fraction", [0.5, 0.25, 0.1])
def test_mixing_matrix_determinant_is_the_live_time_fraction(norm, band_fraction):
    """``det M = f_live`` in every normalisation and at every band split.

    The determinant is what scales the imaginary part of the cross spectrum, so it has
    to be free of the band split and of the choice of units.
    """
    f_live = 0.4
    matrix = deadtime_mixing_matrix(f_live, band_fraction, norm=norm)
    assert np.linalg.det(matrix) == pytest.approx(f_live)


def test_only_the_leahy_mixing_matrix_is_symmetric():
    """In Leahy units the matrix is ``I - a sqrt(p_i p_j)``, hence symmetric.

    The other two normalisations give the transpose of each other, which is why a
    matrix built for one of them silently misbehaves when used with another.
    """
    leahy = deadtime_mixing_matrix(0.5, 0.25, norm="leahy")
    counts = deadtime_mixing_matrix(0.5, 0.25, norm="none")
    frac = deadtime_mixing_matrix(0.5, 0.25, norm="frac")

    assert leahy == pytest.approx(leahy.T)
    assert counts != pytest.approx(counts.T)
    assert counts == pytest.approx(frac.T)


def test_mixing_matrices_agree_at_an_equal_band_split():
    """All three normalisations coincide when the two bands share the counts equally.

    Worth pinning down: it is the reason a matrix written for the wrong normalisation
    can look correct for a long time.
    """
    matrices = [deadtime_mixing_matrix(0.5, 0.5, norm=n) for n in NORMS]
    for other in matrices[1:]:
        assert other == pytest.approx(matrices[0])


def test_mixing_matrices_are_one_matrix_in_three_sets_of_units():
    """The three forms are the same operator rescaled by the per-band amplitude units.

    Leahy divides band ``i`` by ``sqrt(m_i)`` and fractional rms by ``m_i``, so each
    form is ``D^-1 M D`` for the matching diagonal ``D``.
    """
    f_live, p1 = 0.45, 0.2
    counts = deadtime_mixing_matrix(f_live, p1, norm="none")
    m = np.array([p1, 1 - p1])

    for norm, scale in (("leahy", np.sqrt(m)), ("frac", m)):
        d = np.diag(scale)
        expected = np.linalg.inv(d) @ counts @ d
        assert deadtime_mixing_matrix(f_live, p1, norm=norm) == pytest.approx(expected)


def test_mixing_matrix_rejects_an_unknown_normalization():
    """An unsupported normalisation must fail loudly, not pick a default silently."""
    with pytest.raises(ValueError, match="normalization"):
        deadtime_mixing_matrix(0.5, 0.5, norm="abs")


# ------------------------------------------------------------------ the unmixing


@pytest.mark.parametrize("norm", NORMS)
def test_unmixing_inverts_the_mixing(norm):
    """Mixing a known matrix and unmixing it must return the original entries.

    This is the whole correction: ``S_meas = M S_true M^T`` with ``M`` known, so the
    measurement simply inverts.
    """
    f_live, p1 = 0.4, 0.2
    truth = _true_matrix(3.0, 0.5, 0.3)
    measured = _mix(*truth, deadtime_mixing_matrix(f_live, p1, norm=norm))

    recovered = unmix_cross_spectral_matrix(*measured, f_live, p1, norm=norm)
    assert recovered[0] == pytest.approx(truth[0])
    assert recovered[1] == pytest.approx(truth[1])
    assert recovered[2] == pytest.approx(truth[2])


def test_unmixing_broadcasts_over_frequency_and_over_segments():
    """A per-segment ``f_live`` must broadcast against a (frequency, segment) matrix.

    This is the shape ``DynamicalCrossspectrum.dyn_ps`` has, and a per-segment
    correction is the one the dead-time study recommends for a variable source.
    """
    nfreq, nseg = 7, 4
    f_live = np.linspace(0.3, 0.8, nseg)
    p1 = np.full(nseg, 0.3)
    truth = _true_matrix(np.full((nfreq, nseg), 2.0), 0.6, 0.2)

    measured = [np.empty((nfreq, nseg), dtype=complex) for _ in range(3)]
    for j in range(nseg):
        matrix = deadtime_mixing_matrix(f_live[j], p1[j])
        for i in range(nfreq):
            entries = _mix(truth[0][i, j], truth[1][i, j], truth[2][i, j], matrix)
            for store, value in zip(measured, entries):
                store[i, j] = value

    recovered = unmix_cross_spectral_matrix(
        measured[0].real, measured[1].real, measured[2], f_live, p1
    )
    assert recovered[2].shape == (nfreq, nseg)
    assert recovered[2] == pytest.approx(truth[2])


def test_perfect_coherence_survives_the_mixing():
    """A coherence of 1 is a protected special case of a linear mixing.

    Two perfectly coherent signals stay perfectly coherent however they are mixed,
    which is why the dead-time distortion of the coherence went unnoticed for years.
    """
    measured = _mix(*_true_matrix(5.0, 1.0, 0.3), deadtime_mixing_matrix(0.5, 0.35))
    assert coherence_from_cross_spectral_matrix(*measured) == pytest.approx(1.0)


def test_mixing_stretches_a_small_fully_coherent_phase_by_one_over_f_live():
    """At coherence 1 the phase stretch tends to ``1/f_live`` as the lag goes to zero.

    The scalar rule quoted in the literature is this limit: the exact ratio is
    ``f_live^2 cos(phi) / ((u^2 + v^2) cos(phi) + 2uv)``, so it converges from above
    with two digits gained per decade of lag.
    """
    f_live = 0.5
    matrix = deadtime_mixing_matrix(f_live, 0.5)
    ratios = []
    for phase in (0.2, 0.02, 0.002):
        measured = _mix(*_true_matrix(4.0, 1.0, phase), matrix)
        ratios.append(np.tan(-np.angle(measured[2])) / (np.tan(phase) / f_live))

    assert ratios[-1] == pytest.approx(1.0, rel=1e-5)
    assert ratios[0] - 1 > ratios[1] - 1 > ratios[2] - 1 > 0


def test_a_partly_coherent_phase_is_distorted_by_more_than_one_over_f_live():
    """Below coherence 1 the real part picks up a term from the diagonal.

    ``Re C_meas = (u^2 + v^2) Re C + 2uv P`` with ``2uv < 0``, so the two pieces nearly
    cancel and the phase runs away far beyond the ``1/f_live`` of the coherent limit.
    """
    f_live, phase = 0.5, 0.3
    matrix = deadtime_mixing_matrix(f_live, 0.5)
    ratios = []
    for coherence in (1.0, 0.5, 0.25):
        measured = _mix(*_true_matrix(4.0, coherence, phase), matrix)
        ratios.append(-np.angle(measured[2]) / phase)

    assert ratios[0] == pytest.approx(1 / f_live, rel=0.05)
    assert ratios[1] > 1.5 / f_live
    assert ratios[2] > ratios[1]


# ------------------------------------------------------- the cross-module matrix


def _two_band_events(rng, length, dt, rate, band_fraction, phase, frequency, amplitude):
    """Event times for a soft and a hard band sharing one sinusoidal signal."""
    t = np.arange(0, length, dt)
    shape = 1 + amplitude * np.sin(2 * np.pi * frequency * t)
    lagged = 1 + amplitude * np.sin(2 * np.pi * frequency * t - phase)

    out = []
    for fraction, curve in ((band_fraction, shape), (1 - band_fraction, lagged)):
        counts = rng.poisson(rate * fraction * dt * curve)
        times = np.repeat(t, counts) + rng.uniform(0, dt, counts.sum())
        out.append(np.sort(times))
    return out


def _module(rng, length, dt, rate, band_fraction, phase, frequency, amplitude, dead_time):
    """One module: draw both bands, merge, apply dead time, split back."""
    soft, hard = _two_band_events(rng, length, dt, rate, band_fraction, phase, frequency, amplitude)
    merged = np.concatenate([soft, hard])
    label = np.concatenate([np.zeros(soft.size), np.ones(hard.size)])
    order = np.argsort(merged)
    merged, label = merged[order], label[order]

    kept = np.isin(merged, filter_for_deadtime(merged, dead_time))
    gti = [[0, length]]
    return (
        EventList(merged[kept & (label == 0)], gti=gti),
        EventList(merged[kept & (label == 1)], gti=gti),
        kept.sum() / merged.size,
    )


@pytest.mark.slow
def test_cross_module_matrix_carries_no_poisson_level():
    """Crossing two modules leaves no white-noise level in any entry, diagonal included.

    A single module's Leahy power of a steady source sits at a level near 2 -- below it
    here, because dead time suppresses it, which is the level nobody can model well.
    The cross-module diagonal must sit at 0 instead, which is what makes the matrix
    correctable with no model at all.
    """
    length, dt, rate, dead_time = 512.0, 1 / 256, 300.0, 2.5e-3
    rng = np.random.default_rng(42)
    kwargs = dict(
        length=length,
        dt=dt,
        rate=rate,
        band_fraction=0.5,
        phase=0.0,
        frequency=1.0,
        amplitude=0.0,
        dead_time=dead_time,
    )
    soft_a, hard_a, _ = _module(rng, **kwargs)
    soft_b, hard_b, _ = _module(rng, **kwargs)

    matrix = cross_module_cross_spectral_matrix(
        soft_a, hard_a, soft_b, hard_b, segment_size=8.0, dt=dt
    )
    single = AveragedCrossspectrum(soft_a, soft_a, segment_size=8.0, dt=dt, norm="leahy")

    assert np.mean(single.power.real) > 1.0
    assert abs(np.mean(matrix.power1)) < 0.1
    assert abs(np.mean(matrix.power2)) < 0.1


@pytest.mark.slow
def test_unmixing_recovers_a_lag_through_dead_time_at_an_unequal_split():
    """The whole chain, on an unequal band split, where the normalisation matters.

    With Leahy entries only the symmetric matrix is right. At a 10/90 split the other
    two normalisations miss by a factor of two and twenty percent respectively, in
    opposite directions, so this pins down which matrix goes with which units. The
    surviving few percent is the linearisation residual at ``f_live`` near 0.5.
    """
    length, dt, rate, dead_time = 512.0, 1 / 256, 400.0, 2.5e-3
    phase, frequency, split = 0.5, 4.0, 0.1
    rng = np.random.default_rng(7)
    kwargs = dict(
        length=length,
        dt=dt,
        rate=rate,
        band_fraction=split,
        phase=phase,
        frequency=frequency,
        amplitude=0.5,
        dead_time=dead_time,
    )
    soft_a, hard_a, live_a = _module(rng, **kwargs)
    soft_b, hard_b, live_b = _module(rng, **kwargs)
    f_live = 0.5 * (live_a + live_b)

    matrix = cross_module_cross_spectral_matrix(
        soft_a, hard_a, soft_b, hard_b, segment_size=8.0, dt=dt
    )
    peak = np.argmin(np.abs(matrix.freq - frequency))

    def measured_phase(norm):
        entries = unmix_cross_spectral_matrix(
            matrix.power1[peak],
            matrix.power2[peak],
            matrix.cross_power[peak],
            f_live,
            matrix.band_fraction,
            norm=norm,
        )
        return -np.angle(entries[2])

    assert measured_phase("leahy") == pytest.approx(phase, rel=0.12)
    assert measured_phase("frac") < 0.7 * phase
    assert measured_phase("none") > 1.1 * phase

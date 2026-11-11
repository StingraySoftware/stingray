import functools

import numpy as np
import pytest

from stingray import EventList
from stingray.crossspectrum import AveragedCrossspectrum
from stingray.deadtime.crosstalk import (
    cross_module_lag_errors,
    coherence_from_cross_spectral_matrix,
    cross_module_cross_spectral_matrix,
    deadtime_mixing_matrix,
    dynamical_cross_module_cross_spectral_matrix,
    live_time_fraction,
    module_fourier_amplitudes,
    unmix_amplitudes,
    unmix_cross_spectral_matrix,
)
from stingray.filters import filter_for_deadtime
from stingray.utils import njit

NORMS = ["none", "leahy", "frac"]


def _true_matrix(power, coherence, phase):
    """A true 2x2 cross-spectral matrix with equal band powers."""
    cross = power * np.sqrt(coherence) * np.exp(-1j * phase)
    return power, power, cross


def _mix(power1, power2, cross, matrix):
    """Apply ``S_meas = conj(M) S_true M^T`` to the three independent entries.

    The cross spectrum is ``conj(X_1) X_2``, so the first factor is conjugated; it only
    matters for the complex, frequency-dependent matrix.
    """
    s = np.array([[power1, cross], [np.conj(cross), power2]])
    out = np.conj(matrix) @ s @ matrix.T
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


def _two_band_events(
    rng, length, dt, rate, band_fraction, phase, frequency, amplitude, amplitude2=None
):
    """Event times for a soft and a hard band sharing one sinusoidal signal.

    ``amplitude2``, if given, is the hard band's own fractional amplitude.
    """
    amplitude2 = amplitude if amplitude2 is None else amplitude2
    t = np.arange(0, length, dt)
    shape = 1 + amplitude * np.sin(2 * np.pi * frequency * t)
    lagged = 1 + amplitude2 * np.sin(2 * np.pi * frequency * t - phase)

    out = []
    for fraction, curve in ((band_fraction, shape), (1 - band_fraction, lagged)):
        counts = rng.poisson(rate * fraction * dt * curve)
        times = np.repeat(t, counts) + rng.uniform(0, dt, counts.sum())
        out.append(np.sort(times))
    return out


def _module(
    rng, length, dt, rate, band_fraction, phase, frequency, amplitude, dead_time, amplitude2=None
):
    """One module: draw both bands, merge, apply dead time, split back."""
    soft, hard = _two_band_events(
        rng, length, dt, rate, band_fraction, phase, frequency, amplitude, amplitude2
    )
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


def _module_with_segments(
    rng, length, dt, segment_size, rate_curve, band_fraction, phase, frequency, amplitude, dead_time
):
    """One module on a time-varying mean rate, with the live time fraction per segment.

    Returns the two bands as event lists, plus the fraction of events that survived the
    dead-time filter inside each Fourier segment.
    """
    t = np.arange(0, length, dt)
    shape = 1 + amplitude * np.sin(2 * np.pi * frequency * t)
    lagged = 1 + amplitude * np.sin(2 * np.pi * frequency * t - phase)

    times, labels = [], []
    for label, (fraction, curve) in enumerate(
        ((band_fraction, shape), (1 - band_fraction, lagged))
    ):
        counts = rng.poisson(rate_curve * fraction * dt * curve)
        drawn = np.repeat(t, counts) + rng.uniform(0, dt, counts.sum())
        times.append(drawn)
        labels.append(np.full(drawn.size, label))

    merged = np.concatenate(times)
    label = np.concatenate(labels)
    order = np.argsort(merged)
    merged, label = merged[order], label[order]
    kept = np.isin(merged, filter_for_deadtime(merged, dead_time))

    nseg = int(length // segment_size)
    edges = np.arange(nseg + 1) * segment_size
    drawn_per_seg, _ = np.histogram(merged, bins=edges)
    kept_per_seg, _ = np.histogram(merged[kept], bins=edges)

    gti = [[0, length]]
    return (
        EventList(merged[kept & (label == 0)], gti=gti),
        EventList(merged[kept & (label == 1)], gti=gti),
        kept_per_seg / drawn_per_seg,
    )


@pytest.mark.slow
def test_the_per_segment_correction_is_the_one_that_does_not_depend_on_the_bookkeeping():
    """On a source whose live time swings between segments, the per-segment matrix gives
    the same lag whichever segment normalisation it is built in, and one global matrix
    does not.

    This is the end-to-end recipe and the argument for it. A single matrix has to be fed
    some single live time, and the answer then depends on that choice and on whether the
    segments were normalised by their own counts or by the observation mean; the
    per-segment chain has no such choice to get wrong, and what is left is the few
    percent linearisation residual at ``f_live`` near 0.5.
    """
    length, dt, segment_size, dead_time = 512.0, 1 / 256, 8.0, 2.5e-3
    phase, frequency, split = 0.5, 4.0, 0.25
    rng = np.random.default_rng(99)

    # A square wave in the mean rate: the live time swings between segments.
    t = np.arange(0, length, dt)
    rate_curve = np.where((t // 64) % 2 == 0, 150.0, 700.0)
    kwargs = dict(
        length=length,
        dt=dt,
        segment_size=segment_size,
        rate_curve=rate_curve,
        band_fraction=split,
        phase=phase,
        frequency=frequency,
        amplitude=0.5,
        dead_time=dead_time,
    )
    soft_a, hard_a, live_a = _module_with_segments(rng, **kwargs)
    soft_b, hard_b, live_b = _module_with_segments(rng, **kwargs)
    f_live = 0.5 * (live_a + live_b)
    assert f_live.max() - f_live.min() > 0.2

    matrices = {
        common: dynamical_cross_module_cross_spectral_matrix(
            soft_a,
            hard_a,
            soft_b,
            hard_b,
            segment_size=segment_size,
            sample_time=dt,
            use_common_mean=common,
        )
        for common in (False, True)
    }
    assert f_live.size == matrices[False].power1.shape[1]

    def recovered(common, live, band):
        matrix = matrices[common]
        peak = np.argmin(np.abs(matrix.freq - frequency))
        entries = unmix_cross_spectral_matrix(
            matrix.power1, matrix.power2, matrix.cross_power, live, band
        )
        return -np.angle(np.mean(entries[2][peak]))

    per_segment = recovered(False, f_live, matrices[False].band_fraction)
    common_mean = recovered(True, f_live, matrices[True].band_fraction)
    global_only = recovered(True, np.mean(f_live), np.mean(matrices[True].band_fraction))
    uncorrected = -np.angle(
        np.mean(matrices[False].cross_power[np.argmin(np.abs(matrices[False].freq - frequency))])
    )

    assert uncorrected > 1.5 * phase
    assert per_segment == pytest.approx(phase, rel=0.1)
    assert common_mean == pytest.approx(per_segment, rel=0.05)
    assert abs(global_only - phase) > 2 * abs(per_segment - phase)


# ------------------------------------------------- the per-segment matrix


def _steady_two_band_module(rng, length, dt, rate, band_fraction):
    """Two bands of one module, steady source, no dead time: shapes only."""
    out = []
    for fraction in (band_fraction, 1 - band_fraction):
        n = rng.poisson(rate * fraction * length)
        out.append(EventList(np.sort(rng.uniform(0, length, n)), gti=[[0, length]]))
    return out


def test_dynamical_matrix_has_one_column_per_segment():
    """The per-segment builder returns ``(frequency, segment)`` entries and a
    band fraction and a mid-time per segment.

    These are the shapes :func:`unmix_cross_spectral_matrix` broadcasts against, so a
    wrong one would silently correct the whole observation with one segment's matrix.
    """
    length, dt, segment_size = 64.0, 1 / 64, 8.0
    rng = np.random.default_rng(3)
    soft_a, hard_a = _steady_two_band_module(rng, length, dt, 200.0, 0.4)
    soft_b, hard_b = _steady_two_band_module(rng, length, dt, 200.0, 0.4)

    matrix = dynamical_cross_module_cross_spectral_matrix(
        soft_a, hard_a, soft_b, hard_b, segment_size=segment_size, sample_time=dt
    )

    nseg = int(length // segment_size)
    nfreq = matrix.freq.size
    assert matrix.power1.shape == (nfreq, nseg)
    assert matrix.power2.shape == (nfreq, nseg)
    assert matrix.cross_power.shape == (nfreq, nseg)
    assert matrix.band_fraction.shape == (nseg,)
    assert matrix.time.shape == (nseg,)
    assert matrix.m == 1
    assert np.iscomplexobj(matrix.cross_power)
    assert not np.iscomplexobj(matrix.power1)


def test_per_segment_entries_average_to_the_segment_averaged_matrix():
    """Averaging the per-segment entries must reproduce the averaged builder exactly.

    Both normalise each segment by its own counts here, so the averaged matrix is
    nothing but the mean of the per-segment one and any difference is a bug.
    """
    length, dt, segment_size = 64.0, 1 / 64, 8.0
    rng = np.random.default_rng(4)
    soft_a, hard_a = _steady_two_band_module(rng, length, dt, 200.0, 0.3)
    soft_b, hard_b = _steady_two_band_module(rng, length, dt, 200.0, 0.3)
    bands = (soft_a, hard_a, soft_b, hard_b)

    per_segment = dynamical_cross_module_cross_spectral_matrix(
        *bands, segment_size=segment_size, sample_time=dt
    )
    averaged = cross_module_cross_spectral_matrix(
        *bands, segment_size=segment_size, dt=dt, use_common_mean=False
    )

    assert averaged.freq == pytest.approx(per_segment.freq)
    assert np.mean(per_segment.power1, axis=1) == pytest.approx(averaged.power1)
    assert np.mean(per_segment.power2, axis=1) == pytest.approx(averaged.power2)
    assert np.mean(per_segment.cross_power, axis=1) == pytest.approx(averaged.cross_power)
    assert np.mean(per_segment.band_fraction) == pytest.approx(averaged.band_fraction, rel=0.01)


def test_per_segment_band_fraction_follows_a_changing_band_split():
    """A source whose counts move between the bands gets a band fraction per segment.

    The averaged builder can only return the exposure mean, and the mixing matrix of a
    segment is built from that segment's split, so this is the second thing -- besides
    the live time -- that a single matrix for the whole observation gets wrong.
    """
    length, dt, segment_size = 64.0, 1 / 64, 8.0
    rng = np.random.default_rng(5)
    half = length / 2

    def module():
        soft = np.concatenate(
            [
                rng.uniform(0, half, rng.poisson(240.0 * half)),
                rng.uniform(half, length, rng.poisson(60.0 * half)),
            ]
        )
        hard = rng.uniform(0, length, rng.poisson(150.0 * length))
        return (
            EventList(np.sort(soft), gti=[[0, length]]),
            EventList(np.sort(hard), gti=[[0, length]]),
        )

    soft_a, hard_a = module()
    soft_b, hard_b = module()
    matrix = dynamical_cross_module_cross_spectral_matrix(
        soft_a, hard_a, soft_b, hard_b, segment_size=segment_size, sample_time=dt
    )

    nseg = matrix.band_fraction.size
    first, second = matrix.band_fraction[: nseg // 2], matrix.band_fraction[nseg // 2 :]
    assert np.mean(first) == pytest.approx(240 / 390, rel=0.05)
    assert np.mean(second) == pytest.approx(60 / 210, rel=0.1)


# ------------------------------------------- the frequency-dependent correction

DEAD_TIME = 2.5e-3


def _mix_per_frequency(power1, power2, cross, matrix):
    """``conj(M) S M^T`` with one matrix per frequency, ``matrix`` of shape (nf, 2, 2)."""
    s = np.empty(matrix.shape, dtype=complex)
    s[:, 0, 0], s[:, 1, 1], s[:, 0, 1] = power1, power2, cross
    s[:, 1, 0] = np.conj(cross)
    out = np.conj(matrix) @ s @ np.swapaxes(matrix, -1, -2)
    return out[:, 0, 0].real, out[:, 1, 1].real, out[:, 0, 1]


@pytest.mark.parametrize("bin_time", [None, DEAD_TIME / 4])
@pytest.mark.parametrize("norm", NORMS)
def test_frequency_dependent_unmixing_inverts_its_mixing(norm, bin_time):
    """With the dead time, the matrix is complex and changes with frequency; mixing and
    unmixing must still round-trip at an unequal split, on a grid reaching 2/td."""
    f_live, p1 = 0.5, 0.3
    freq = np.linspace(0.5, 2 / DEAD_TIME, 200)
    options = dict(norm=norm, dead_time=DEAD_TIME, freq=freq, bin_time=bin_time)
    truth = _true_matrix(3.0, 0.5, 0.3)
    measured = _mix_per_frequency(*truth, deadtime_mixing_matrix(f_live, p1, **options))

    recovered = unmix_cross_spectral_matrix(*measured, f_live, p1, **options)
    assert np.allclose(recovered[0], truth[0])
    assert np.allclose(recovered[1], truth[1])
    assert np.allclose(recovered[2], truth[2])


def test_frequency_dependent_matrix_tends_to_the_instantaneous_one():
    """Far below 1/(2 pi td) the live time follows the rate instantaneously and the two
    corrections agree; at a few hundred Hz they do not."""
    measured = _true_matrix(3.0, 0.5, 0.3)
    instantaneous = unmix_cross_spectral_matrix(*measured, 0.5, 0.3)
    low, high = (
        unmix_cross_spectral_matrix(*measured, 0.5, 0.3, dead_time=DEAD_TIME, freq=f)
        for f in (1.0, 300.0)
    )

    assert abs(low[2] / instantaneous[2] - 1) < 0.01
    assert abs(high[2] / instantaneous[2] - 1) > 0.1


def test_frequency_dependent_matrix_needs_the_frequencies():
    """A dead time without frequencies cannot give W(f), and a bin time without a dead
    time would be silently ignored; both are errors."""
    with pytest.raises(ValueError, match="freq"):
        deadtime_mixing_matrix(0.5, 0.5, dead_time=DEAD_TIME)
    with pytest.raises(ValueError, match="dead_time"):
        deadtime_mixing_matrix(0.5, 0.5, freq=1.0, bin_time=1e-3)


@pytest.mark.slow
def test_frequency_dependent_correction_fixes_the_lag_of_unequal_bands():
    """Unequal bands (30/70 counts, amplitudes 0.3/0.15) at 40 Hz: the instantaneous
    matrix returns 0.19 rad for 0.30, the frequency-dependent one recovers it. The
    single-realisation scatter is 0.017 rad; the gap between the two is paired."""
    length, dt, rate = 1024.0, 1 / 512, 400.0
    phase, frequency = 0.3, 40.0
    rng = np.random.default_rng(11)
    kwargs = dict(
        length=length,
        dt=dt,
        rate=rate,
        band_fraction=0.3,
        phase=phase,
        frequency=frequency,
        amplitude=0.3,
        amplitude2=0.15,
        dead_time=DEAD_TIME,
    )
    soft_a, hard_a, _ = _module(rng, **kwargs)
    soft_b, hard_b, _ = _module(rng, **kwargs)
    matrix = cross_module_cross_spectral_matrix(
        soft_a, hard_a, soft_b, hard_b, segment_size=8.0, dt=dt
    )
    detected = (soft_a.time.size + hard_a.time.size + soft_b.time.size + hard_b.time.size) / 2
    f_live = live_time_fraction(detected / length, DEAD_TIME)
    peak = np.argmin(np.abs(matrix.freq - frequency))
    entries = (matrix.power1[peak], matrix.power2[peak], matrix.cross_power[peak])

    def recovered(**options):
        unmixed = unmix_cross_spectral_matrix(*entries, f_live, matrix.band_fraction, **options)
        return -np.angle(unmixed[2])

    with_w = recovered(dead_time=DEAD_TIME, freq=matrix.freq[peak])
    without_w = recovered()
    assert with_w == pytest.approx(phase, rel=0.15)
    assert with_w - without_w > 0.05


@njit()
def _solve_dead_time_equation(r1, r2, k, dt):
    """Detected rates of two bands from ``m_i = r_i [1 - int_{t-td}^t m_tot]``.

    Trapezoid rule over the last ``k`` steps, implicit in the current one, started from
    the steady state so that there is no start-up ringing.
    """
    n = r1.size
    m1, m2, mt = np.zeros(n), np.zeros(n), np.zeros(n)
    steady = (r1[0] + r2[0]) / (1.0 + (r1[0] + r2[0]) * k * dt)
    window = k * steady
    for i in range(n):
        old = mt[i - k] if i >= k else steady
        window -= old
        rt = r1[i] + r2[i]
        mt[i] = rt * (1.0 - dt * (window + 0.5 * old)) / (1.0 + 0.5 * dt * rt)
        m1[i], m2[i] = r1[i] * mt[i] / rt, r2[i] * mt[i] / rt
        window += mt[i]
    return m1, m2


@functools.cache
def _time_domain_matrices():
    """Measured and dead-time-free spectra of a deterministic, noise-free source.

    Two bands with 15/85 of 1600 c/s (f_live = 0.2) and small sinusoids of different
    amplitude and phase at 3, 211 and 700 Hz; two runs, so the matrix is not rank one.
    """
    dt, eps, freqs = DEAD_TIME / 100, 1e-4, np.array([3.0, 211.0, 700.0])
    t = np.arange(0, 1.5, dt)
    rng = np.random.default_rng(5)
    keep = t >= 0.5
    runs = []
    for _ in range(2):
        s1, s2 = (
            sum(amp * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in freqs)
            for amp in (0.4, 1.0)
        )
        r1, r2 = 1600 * 0.15 * (1 + eps * s1), 1600 * 0.85 * (1 + eps * s2)
        m1, m2 = _solve_dead_time_equation(r1, r2, 100, dt)
        runs.append([x[keep] for x in (m1, m2, r1, r2)])
    m1bar = np.mean([r[0].mean() for r in runs])
    m2bar = np.mean([r[1].mean() for r in runs])
    f_live = 1 - (m1bar + m2bar) * DEAD_TIME

    def amplitude(x):
        return np.exp(-2j * np.pi * np.outer(freqs, t[keep])) @ (x - x.mean())

    def matrix(x1, x2, scale1, scale2):
        a1, a2 = amplitude(x1) / scale1, amplitude(x2) / scale2
        return np.array([np.abs(a1) ** 2, np.abs(a2) ** 2, np.conj(a1) * a2])

    scales = {
        "none": (1.0, 1.0),
        "frac": (m1bar, m2bar),
        "leahy": (np.sqrt(m1bar), np.sqrt(m2bar)),
    }
    out = {}
    for norm, (sc1, sc2) in scales.items():
        meas = sum(matrix(m1, m2, sc1, sc2) for m1, m2, _, _ in runs)
        true = sum(matrix(f_live * r1, f_live * r2, sc1, sc2) for _, _, r1, r2 in runs)
        out[norm] = (meas, true)
    return freqs, f_live, m1bar / (m1bar + m2bar), out


@pytest.mark.parametrize("norm", NORMS)
def test_frequency_dependent_unmixing_matches_the_exact_dead_time_equation(norm):
    """Against the dead-time equation solved in time, with no algebra shared with the
    module: in every normalisation W(f) recovers the dead-time-free matrix up to 700 Hz
    (left: the time step), and the instantaneous matrix is a factor of several off."""
    freqs, f_live, p1, matrices = _time_domain_matrices()
    meas, true = matrices[norm]
    scale = np.max(np.abs(true), axis=0)

    def error(**options):
        got = unmix_cross_spectral_matrix(*meas, f_live, p1, norm=norm, **options)
        return np.max(np.abs(np.array(got) - true) / scale, axis=0)

    assert np.all(error(dead_time=DEAD_TIME, freq=freqs) < 2e-3)
    assert np.all(error()[1:] > 1)


# ----------------------------------------------------------- unmixed amplitudes


def _matrix_from_amplitudes(det1, det2):
    """The cross-module entries of two detectors' (band 1, band 2) amplitudes."""
    (a1, a2), (b1, b2) = det1, det2
    return (
        (np.conj(a1) * b1).real,
        (np.conj(a2) * b2).real,
        0.5 * (np.conj(a1) * b2 + np.conj(b1) * a2),
    )


@pytest.mark.parametrize("norm", NORMS)
def test_unmixing_the_amplitudes_is_unmixing_the_matrix(norm):
    """Unmixing each detector's amplitudes and then crossing them gives the unmixed
    matrix (the identity behind the error bars), per segment and with W(f)."""
    rng = np.random.default_rng(8)
    nfreq, nseg = 40, 5
    freq = np.linspace(0.5, 2 / DEAD_TIME, nfreq)[:, None]
    f_live = rng.uniform(0.2, 0.9, nseg)
    p1 = rng.uniform(0.1, 0.9, nseg)
    dets = [
        [rng.normal(size=(nfreq, nseg)) + 1j * rng.normal(size=(nfreq, nseg)) for _ in "ab"]
        for _ in "AB"
    ]
    options = dict(norm=norm, dead_time=DEAD_TIME, freq=freq)

    unmixed = [unmix_amplitudes(*det, f_live, p1, **options) for det in dets]
    expected = unmix_cross_spectral_matrix(*_matrix_from_amplitudes(*dets), f_live, p1, **options)

    for got, want in zip(_matrix_from_amplitudes(*unmixed), expected):
        assert np.allclose(got, want)


@pytest.mark.parametrize("norm", NORMS)
def test_module_amplitudes_reproduce_the_per_segment_matrix(norm):
    """Crossing the amplitudes of two detectors must give exactly the entries of
    dynamical_cross_module_cross_spectral_matrix, on the same frequencies and segments."""
    length, dt, segment_size = 64.0, 1 / 64, 8.0
    rng = np.random.default_rng(3)
    det_a = _steady_two_band_module(rng, length, dt, 200.0, 0.4)
    det_b = _steady_two_band_module(rng, length, dt, 200.0, 0.4)

    matrix = dynamical_cross_module_cross_spectral_matrix(
        *det_a, *det_b, segment_size=segment_size, sample_time=dt, norm=norm
    )
    amp_a, amp_b = (
        module_fourier_amplitudes(*det, segment_size=segment_size, dt=dt, norm=norm)
        for det in (det_a, det_b)
    )

    assert np.allclose(amp_a.freq, matrix.freq)
    assert np.allclose(amp_a.time, matrix.time)
    edges = np.arange(0, length + segment_size, segment_size)
    counts1, counts2 = (np.histogram(ev.time, bins=edges)[0] for ev in det_a)
    assert np.allclose(amp_a.band_fraction, counts1 / (counts1 + counts2))
    entries = _matrix_from_amplitudes(
        (amp_a.amplitude1, amp_a.amplitude2), (amp_b.amplitude1, amp_b.amplitude2)
    )
    for got, want in zip(entries, (matrix.power1, matrix.power2, matrix.cross_power)):
        assert np.allclose(got, want)


# ------------------------------------------------------------------ error bars


def test_lag_error_without_noise_is_bendat_piersol():
    """With identical, noise-free detectors (T = P) the error formula must reduce
    exactly to the Bendat & Piersol one, sqrt((1 - gamma^2) / (2 gamma^2 N))."""
    rng = np.random.default_rng(11)
    nseg = 30
    x, y = (rng.normal(size=(4, nseg)) + 1j * rng.normal(size=(4, nseg)) for _ in "xy")
    y = y + 0.8 * x
    res = cross_module_lag_errors((x, y), (x, y))

    gamma2 = np.abs(res.cross_power) ** 2 / (res.power1 * res.power2)
    assert res.m == nseg
    assert np.allclose(res.total_power1, res.power1)
    assert np.allclose(res.phase_lag_err, np.sqrt((1 - gamma2) / (2 * gamma2 * nseg)))


def test_lag_error_matches_the_scatter_of_simulated_lags():
    """On normal amplitudes with a known lag, partial coherence and independent noise
    in each detector, the predicted lag error matches the scatter of the measured lags."""
    rng = np.random.default_rng(12)
    nreal, nseg, gamma2, lag, noise = 2000, 200, 0.5, 0.7, 2.0

    def cnormal(power):
        return np.sqrt(power / 2) * (
            rng.normal(size=(nreal, nseg)) + 1j * rng.normal(size=(nreal, nseg))
        )

    s1 = cnormal(1.0)
    s2 = np.exp(1j * lag) * (np.sqrt(gamma2) * s1 + np.sqrt(1 - gamma2) * cnormal(1.0))
    det_a = (s1 + cnormal(noise), s2 + cnormal(noise))
    det_b = (s1 + cnormal(noise), s2 + cnormal(noise))
    res = cross_module_lag_errors(det_a, det_b)

    assert np.isclose(np.mean(res.phase_lag), lag, atol=0.02)
    assert np.isclose(np.std(res.phase_lag), np.median(res.phase_lag_err), rtol=0.1)

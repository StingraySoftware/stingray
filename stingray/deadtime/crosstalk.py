r"""Dead-time cross-talk between energy bands, and how to undo it.

Dead time couples the energy bands of a single detector. The live time of a
non-paralyzable detector is a gain that fluctuates with the *total* detected rate and
is shared by every band, so a fluctuation in one band suppresses every band, its own
included. Linearising :math:`m_i = r_i/(1 + r_{\rm tot} t_d)` leaves a cross-spectral
matrix that has been mixed by a known real matrix,

.. math:: S_{\rm meas} = M\, S_{\rm true}\, M^{T},

with :math:`\det M = f_{\rm live}`. The consequences are not subtle. A completely
steady source shows a cross spectrum between two of its bands with a purely real,
negative value and manufactures a coherence of several tenths; a real partial coherence
is suppressed by up to a factor of four; and a phase lag is stretched, by
:math:`1/f_{\rm live}` when the two bands are perfectly coherent and by considerably
more when they are not.

Because :math:`M` is known from the live time fraction and from the way the detected
counts split between the bands, the measurement simply inverts:

.. math:: S_{\rm true} = M^{-1} S_{\rm meas} (M^{-1})^{T}.

The one prerequisite is a cross-spectral matrix with no Poisson noise in it, including
on the diagonal, since a subtracted noise level would have to be modelled and the
dead-time power spectrum is exactly what is hard to model. Two detectors looking at the
same source provide that for free: their Poisson noise is independent, so crossing them
leaves every entry of the matrix noise-free. See
:func:`cross_module_cross_spectral_matrix`.

**The normalisation matters.** :math:`M` acts on Fourier amplitudes, so rescaling each
band by a different factor -- which is what every per-band normalisation does --
changes the matrix by a diagonal similarity transform. In Leahy normalisation it is
symmetric, :math:`M_{ij} = \delta_{ij} - a\sqrt{p_i p_j}`; in fractional rms
normalisation it is not. Passing the matrix of one normalisation and the spectra of
another is harmless when the two bands share the counts equally and wrong otherwise, so
``norm`` is an explicit argument everywhere below and must match the spectra.

For a variable source the live time is a function of time and one matrix does not
describe a whole observation: build and unmix the matrix *per segment*, with that
segment's own live time, and average afterwards.
:func:`dynamical_cross_module_cross_spectral_matrix` builds that per-segment matrix, and
every function here broadcasts, so a ``f_live`` of one value per segment corrects its
``(frequency, segment)`` entries segment by segment::

    matrix = dynamical_cross_module_cross_spectral_matrix(
        soft_a, hard_a, soft_b, hard_b, segment_size=8.0, sample_time=1 / 256)
    f_live = live_time_fraction(detected_rate_per_segment, dead_time)
    power1, power2, cross = unmix_cross_spectral_matrix(
        matrix.power1, matrix.power2, matrix.cross_power, f_live, matrix.band_fraction)
    power1, power2, cross = (np.mean(x, axis=-1) for x in (power1, power2, cross))

Averaging only after the unmixing is the point: the mean of the corrected segments is
not the correction of the mean segment. How much that costs depends on the segment
length against the source. A live time that varies only *between* segments is already
handled well by one global inverse; a live time that swings *within* a segment, as it
does under fast flaring, separates the two by about 20% on the lag and is not fully
fixed by either, because then no single matrix describes even one segment. Normalise
each segment by *its own* counts while you are at it -- the per-segment builder defaults
to that -- because the matrix of a segment is built from that segment's counts and the
spectra it acts on have to be in the same units.

References
----------
Zhang et al. 1995, ApJ 449, 930; Bachetti et al. 2015, ApJ 800, 109;
Vaughan & Nowak 1997, ApJ 474, L43.
"""

from collections import namedtuple

import numpy as np

from stingray.crossspectrum import AveragedCrossspectrum, DynamicalCrossspectrum

__all__ = [
    "live_time_fraction",
    "deadtime_mixing_matrix",
    "unmix_cross_spectral_matrix",
    "coherence_from_cross_spectral_matrix",
    "cross_module_cross_spectral_matrix",
    "dynamical_cross_module_cross_spectral_matrix",
    "CrossSpectralMatrix",
]

VALID_NORMS = ("none", "leahy", "frac")

CrossSpectralMatrix = namedtuple(
    "CrossSpectralMatrix",
    ["freq", "power1", "power2", "cross_power", "band_fraction", "m", "norm", "time"],
    defaults=(None,),
)
CrossSpectralMatrix.__doc__ = """A Poisson-free 2x2 cross-spectral matrix of two energy bands.

Attributes
----------
freq : `np.ndarray`
    Fourier frequencies.
power1, power2 : `np.ndarray`
    The diagonal: the power of each band, with no Poisson level in it. Shape
    ``(frequency,)`` for a segment-averaged matrix and ``(frequency, segment)`` for a
    per-segment one.
cross_power : `np.ndarray`
    The off-diagonal cross spectrum, complex. The other off-diagonal entry is its
    complex conjugate, because the matrix is Hermitian.
band_fraction : float or `np.ndarray`
    Fraction of the detected counts that fall in band 1, the ``p_1`` the mixing matrix
    needs. One value per segment for a per-segment matrix.
m : int
    Number of averaged segments, hence 1 for a per-segment matrix.
norm : str
    Normalisation the entries are in, to be passed on to the unmixing.
time : `np.ndarray` or None
    Mid-time of each segment for a per-segment matrix, `None` when the segments have
    already been averaged.
"""


def live_time_fraction(detected_rate, dead_time):
    r"""Live time fraction from the detected count rate.

    For non-paralyzable dead time :math:`r_{\rm det} = r_{\rm in}/(1 + r_{\rm in} t_d)`,
    so :math:`r_{\rm det} t_d = 1 - f_{\rm live}` *exactly*: the live time fraction
    follows from the detected rate alone, with no model and no knowledge of the
    incident rate. Use the value the mission reports when there is one -- NuSTAR gives
    it at 1 s cadence -- and this when there is not.

    Parameters
    ----------
    detected_rate : float or `np.ndarray`
        Detected count rate of the whole detector, counts/s. Every detected event
        causes dead time, so this must count all events, not only those in the bands
        under analysis.
    dead_time : float
        Dead time per event, s.

    Returns
    -------
    f_live : float or `np.ndarray`
        Live time fraction, in [0, 1].

    Examples
    --------
    >>> float(live_time_fraction(200.0, 2.5e-3))
    0.5
    """
    return 1.0 - np.asanyarray(detected_rate) * dead_time


def deadtime_mixing_matrix(f_live, band_fraction, norm="leahy"):
    r"""The dead-time mixing matrix of two energy bands of one detector.

    With :math:`a = 1 - f_{\rm live}` and :math:`p_i` the fraction of detected counts
    in band :math:`i`, the matrix acting on Leahy-normalised amplitudes is

    .. math::

        M = \begin{pmatrix} 1 - a p_1 & -a\sqrt{p_1 p_2} \\
                            -a\sqrt{p_1 p_2} & 1 - a p_2 \end{pmatrix},
        \qquad \det M = f_{\rm live}.

    Its off-diagonal is the cross-talk term of Bachetti et al. 2015. The three
    supported normalisations are the same operator in different units, related by the
    diagonal rescaling each normalisation applies to the bands, and they coincide when
    :math:`p_1 = p_2 = 1/2`.

    Parameters
    ----------
    f_live : float or `np.ndarray`
        Live time fraction. An array gives one matrix per element, stacked on the
        leading axes, which is how a per-segment correction is built.
    band_fraction : float or `np.ndarray`
        Fraction :math:`p_1` of the detected counts in band 1. The two bands are
        assumed to hold all the detected counts, so :math:`p_2 = 1 - p_1`; photons
        outside both bands still cause dead time and would need a wider matrix.

    Other Parameters
    ----------------
    norm : {'leahy', 'frac', 'none'}, default 'leahy'
        Normalisation of the amplitudes the matrix will act on. It has to match the
        normalisation of the spectra, or the correction is wrong away from an equal
        band split.

    Returns
    -------
    matrix : `np.ndarray`
        Shape ``(..., 2, 2)``, the leading axes broadcast from the inputs.

    Examples
    --------
    >>> m = deadtime_mixing_matrix(0.5, 0.5)
    >>> bool(np.isclose(np.linalg.det(m), 0.5))
    True
    """
    if norm not in VALID_NORMS:
        raise ValueError(
            f"Unknown normalization {norm!r}: the mixing matrix is only defined for "
            f"{VALID_NORMS}. Absolute rms normalization rescales each band by its own "
            "mean rate and is not implemented."
        )

    a = 1.0 - np.asanyarray(f_live, dtype=float)
    p1 = np.asanyarray(band_fraction, dtype=float)
    p2 = 1.0 - p1

    if norm == "leahy":
        upper = lower = -a * np.sqrt(p1 * p2)
    elif norm == "frac":
        # Dividing band i by its mean rate m_i sends -a p_i to -a p_j.
        upper, lower = -a * p2, -a * p1
    else:
        upper, lower = -a * p1, -a * p2

    diag1, diag2 = 1.0 - a * p1, 1.0 - a * p2
    rows = [
        np.stack(np.broadcast_arrays(diag1, upper), axis=-1),
        np.stack(np.broadcast_arrays(lower, diag2), axis=-1),
    ]
    return np.stack(rows, axis=-2)


def unmix_cross_spectral_matrix(power1, power2, cross_power, f_live, band_fraction, norm="leahy"):
    r"""Undo the dead-time mixing of a Poisson-free cross-spectral matrix.

    Applies :math:`S_{\rm true} = M^{-1} S_{\rm meas} (M^{-1})^{T}` element by element,
    so it corrects the two band powers, the coherence and the phase lag in one step.
    There is no separate scalar correction for the lag: the familiar
    :math:`1/f_{\rm live}` stretch of the phase is only the limit of this expression at
    a coherence of 1, and it under-corrects by a factor of several when the bands are
    only partly coherent.

    The inputs must carry **no Poisson noise level**, on the diagonal as well as off
    it; :func:`cross_module_cross_spectral_matrix` builds such a matrix from two
    detectors. Feeding it a single detector's power spectra, which do have a noise
    level, corrects the wrong quantity.

    Parameters
    ----------
    power1, power2 : float or `np.ndarray`
        The diagonal of the measured matrix: the two band powers.
    cross_power : complex or `np.ndarray`
        The measured cross spectrum between the bands.
    f_live : float or `np.ndarray`
        Live time fraction. Broadcast against the spectra, so an array of one value per
        segment corrects a ``(frequency, segment)`` matrix segment by segment.
    band_fraction : float or `np.ndarray`
        Fraction of the detected counts in band 1.

    Other Parameters
    ----------------
    norm : {'leahy', 'frac', 'none'}, default 'leahy'
        Normalisation the spectra are in. Must match them.

    Returns
    -------
    power1, power2 : `np.ndarray`
        The unmixed band powers, real.
    cross_power : `np.ndarray`
        The unmixed cross spectrum, complex. Its phase is the intrinsic phase lag and
        :func:`coherence_from_cross_spectral_matrix` turns the three of them into the
        intrinsic coherence.

    Examples
    --------
    >>> m = deadtime_mixing_matrix(0.5, 0.5)
    >>> s = np.array([[2.0, 1.0 + 0.5j], [1.0 - 0.5j, 2.0]])
    >>> meas = m @ s @ m.T
    >>> out = unmix_cross_spectral_matrix(
    ...     meas[0, 0].real, meas[1, 1].real, meas[0, 1], 0.5, 0.5)
    >>> bool(np.isclose(out[2], 1.0 + 0.5j))
    True
    """
    matrix = deadtime_mixing_matrix(f_live, band_fraction, norm=norm)
    inverse = np.linalg.inv(matrix)
    a, b = inverse[..., 0, 0], inverse[..., 0, 1]
    c, d = inverse[..., 1, 0], inverse[..., 1, 1]

    power1 = np.asanyarray(power1)
    power2 = np.asanyarray(power2)
    cross_power = np.asanyarray(cross_power)
    conj = np.conj(cross_power)

    # S = [[p1, c], [conj(c), p2]] and T = Minv S Minv^T, written out term by term so
    # that it broadcasts over frequency and segment without building a matrix per bin.
    out1 = a * a * power1 + a * b * (cross_power + conj) + b * b * power2
    out2 = c * c * power1 + c * d * (cross_power + conj) + d * d * power2
    out12 = a * c * power1 + a * d * cross_power + b * c * conj + b * d * power2
    return np.real(out1), np.real(out2), out12


def coherence_from_cross_spectral_matrix(power1, power2, cross_power):
    r"""Intrinsic coherence of a Poisson-free cross-spectral matrix.

    .. math:: \gamma^2 = |S_{12}|^2 / (S_{11} S_{22})

    with nothing subtracted, because a matrix built by crossing two detectors never had
    a Poisson level in it and needs no bias term. Pass it the output of
    :func:`unmix_cross_spectral_matrix`, not the raw matrix, or dead time will have
    suppressed the answer.

    .. note::

        Average or rebin the matrix **first** and form the ratio afterwards. Taken per
        Fourier bin, the denominator crosses zero and the ratio is meaningless; this is
        the easiest way to produce a coherence of -58, with or without dead time.

    Parameters
    ----------
    power1, power2 : float or `np.ndarray`
        The diagonal of the matrix.
    cross_power : complex or `np.ndarray`
        The off-diagonal entry.

    Returns
    -------
    coherence : `np.ndarray`
        The intrinsic coherence, which is 1 for perfectly correlated bands.
    """
    power1 = np.real(np.asanyarray(power1))
    power2 = np.real(np.asanyarray(power2))
    return np.abs(np.asanyarray(cross_power)) ** 2 / (power1 * power2)


def _four_cross_spectra(cls, band1_det1, band2_det1, band1_det2, band2_det2, options):
    """The four cross spectra a two-band, two-detector matrix is made of.

    In one place so that the segment-averaged and the per-segment builder cannot
    disagree on which bands and which detectors are crossed with which.
    """
    return (
        cls(band1_det1, band1_det2, **options),
        cls(band2_det1, band2_det2, **options),
        cls(band1_det1, band2_det2, **options),
        cls(band1_det2, band2_det1, **options),
    )


def cross_module_cross_spectral_matrix(
    band1_det1, band2_det1, band1_det2, band2_det2, segment_size, norm="leahy", **kwargs
):
    r"""Build a cross-spectral matrix of two energy bands by crossing two detectors.

    Two detectors observing the same source share its variability and have independent
    Poisson noise, so every entry of

    .. math::

        S_{11} = {\rm Re}\langle A_1^* B_1\rangle, \qquad
        S_{22} = {\rm Re}\langle A_2^* B_2\rangle, \qquad
        S_{12} = \langle A_1^* B_2 + B_1^* A_2\rangle / 2

    is free of a white-noise level, the diagonal included -- where :math:`A`, :math:`B`
    are the two detectors and the subscript is the band. Nothing has to be calibrated
    and no dead-time power spectrum has to be modelled, which is what makes the
    inversion in :func:`unmix_cross_spectral_matrix` possible. This is the cospectrum
    trick of Bachetti et al. 2015 extended to the whole matrix.

    It removes the *Poisson* bias only. The dead-time mixing between the bands is
    common to both detectors and survives, so the result still has to be unmixed.

    Parameters
    ----------
    band1_det1, band2_det1 : `stingray.EventList` or `stingray.Lightcurve`
        The two energy bands of the first detector.
    band1_det2, band2_det2 : `stingray.EventList` or `stingray.Lightcurve`
        The same two bands of the second detector.
    segment_size : float
        Length of each Fourier segment, s.

    Other Parameters
    ----------------
    norm : {'leahy', 'frac', 'none'}, default 'leahy'
        Normalisation of the entries, to be carried through to the unmixing. Leahy is
        the default because it is the one in which the mixing matrix is symmetric.
    **kwargs : dict
        Passed to :class:`stingray.crossspectrum.AveragedCrossspectrum`; ``dt`` is
        required for event lists.

    Returns
    -------
    matrix : `CrossSpectralMatrix`
        The three independent entries, the band fraction the unmixing needs, and the
        number of averaged segments.

    See Also
    --------
    dynamical_cross_module_cross_spectral_matrix : the same matrix, one per segment.

    Notes
    -----
    For a variable source, use
    :func:`dynamical_cross_module_cross_spectral_matrix` instead and unmix each segment
    with its own live time: a single matrix from the exposure-mean live time is biased
    by the variance of the live time across segments.
    """
    options = dict(segment_size=segment_size, norm=norm, **kwargs)
    diag1, diag2, off_a, off_b = _four_cross_spectra(
        AveragedCrossspectrum, band1_det1, band2_det1, band1_det2, band2_det2, options
    )

    counts1 = diag1.nphots1 + diag1.nphots2
    counts2 = diag2.nphots1 + diag2.nphots2

    return CrossSpectralMatrix(
        freq=diag1.freq,
        power1=diag1.power.real,
        power2=diag2.power.real,
        cross_power=0.5 * (off_a.power + off_b.power),
        band_fraction=counts1 / (counts1 + counts2),
        m=diag1.m,
        norm=norm,
    )


def dynamical_cross_module_cross_spectral_matrix(
    band1_det1,
    band2_det1,
    band1_det2,
    band2_det2,
    segment_size,
    norm="leahy",
    use_common_mean=False,
    **kwargs,
):
    r"""Build one Poisson-free cross-spectral matrix per Fourier segment.

    Same matrix as :func:`cross_module_cross_spectral_matrix`, kept segment by segment
    instead of averaged, so that each segment can be unmixed with its own live time
    fraction and its own band split. For a variable source that is not a refinement but
    the correct thing to do: the live time follows the count rate, so a bright segment
    and a faint segment have been mixed by *different* matrices, and one matrix built
    from the exposure mean leaves a bias that grows with the variance of the live time
    across segments.

    The entries have shape ``(frequency, segment)``, which is what
    :func:`unmix_cross_spectral_matrix` broadcasts a per-segment ``f_live`` against.
    Average over the segment axis **after** unmixing, never before.

    Parameters
    ----------
    band1_det1, band2_det1 : `stingray.EventList` or `stingray.Lightcurve`
        The two energy bands of the first detector.
    band1_det2, band2_det2 : `stingray.EventList` or `stingray.Lightcurve`
        The same two bands of the second detector.
    segment_size : float
        Length of each Fourier segment, s. It sets the time resolution of the
        correction: short enough that the live time is roughly constant inside one
        segment, long enough to reach the lowest frequency of interest.

    Other Parameters
    ----------------
    norm : {'leahy', 'frac', 'none'}, default 'leahy'
        Normalisation of the entries, to be carried through to the unmixing.
    use_common_mean : bool, default False
        Normalise each segment by its own counts rather than by the mean over the
        observation. The default is the opposite of stingray's elsewhere, and
        deliberately so: the mixing matrix of a segment is built from *that segment's*
        counts, so the spectra it acts on have to be in that segment's units too.
    **kwargs : dict
        Passed to :class:`stingray.crossspectrum.DynamicalCrossspectrum`;
        ``sample_time`` is required for event lists. Note the name: the averaged
        builder calls the same quantity ``dt``.

    Returns
    -------
    matrix : `CrossSpectralMatrix`
        ``power1``, ``power2`` and ``cross_power`` of shape ``(frequency, segment)``,
        ``band_fraction`` and ``time`` of shape ``(segment,)``, and ``m = 1``.

    See Also
    --------
    cross_module_cross_spectral_matrix : the same matrix, averaged over the segments.

    Notes
    -----
    ``band_fraction`` is built from the per-segment photon counts stingray stores for
    each cross spectrum, which are the *geometric* mean of the two detectors' counts in
    that band. The band fraction is a ratio of the two bands, so any factor common to
    the two detectors cancels and the value is exact whenever the two detectors split
    the counts between the bands in the same proportion -- the same response, which is
    the usual case for two modules of one telescope. It differs from the per-detector
    fraction only at second order in the difference between the two responses.

    The ``f_live`` to pass to the unmixing is one value per segment, lined up with
    ``time``. Take it from the mission's housekeeping where there is one, or from the
    detected rate of the whole detector with :func:`live_time_fraction`; in either case
    it has to be the rate of *all* events, not only those in the two bands, since every
    event causes dead time.
    """
    options = dict(segment_size=segment_size, norm=norm, use_common_mean=use_common_mean, **kwargs)
    diag1, diag2, off_a, off_b = _four_cross_spectra(
        DynamicalCrossspectrum, band1_det1, band2_det1, band1_det2, band2_det2, options
    )

    counts1 = np.asanyarray(diag1.nphots_all)
    counts2 = np.asanyarray(diag2.nphots_all)

    return CrossSpectralMatrix(
        freq=diag1.freq,
        power1=diag1.dyn_ps.real,
        power2=diag2.dyn_ps.real,
        cross_power=0.5 * (off_a.dyn_ps + off_b.dyn_ps),
        band_fraction=counts1 / (counts1 + counts2),
        m=1,
        norm=norm,
        time=diag1.time,
    )

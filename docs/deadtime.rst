Dealing with dead time
**********************

Stingray implements a few features to deal with instrumental dead time.
This is particularly useful in missions with long dead time, such as NuSTAR or IXPE.
In this tutorial, we will show the effects of dead time on X-ray observations, and explain how Stingray can help model it and, under some conditions, even correct for it.

.. toctree::
   :maxdepth: 2

   notebooks/Deadtime/Dead time model in Stingray.ipynb
   notebooks/Deadtime/FAD correction in Stingray.ipynb

Cross-talk between energy bands
===============================

Dead time does more than suppress the white-noise level: it *couples* the energy
bands of a single detector. A photon that is detected in the soft band blinds the
detector to a photon that would have arrived in the hard band, so the two bands of
one module share a common deficit. For a perfectly steady source, the measured
Leahy-normalized periodogram of a band holding a fraction :math:`p` of the counts is

.. math::

   P_\mathrm{band}(f) - 2 = p \, \left[P_\mathrm{tot}(f) - 2\right],

and the cross spectrum between a soft and a hard band of the *same* module is purely
real,

.. math::

   \mathrm{Re}\,C_{sh}(f) = \sqrt{p_s p_h} \, \left[P_\mathrm{tot}(f) - 2\right],
   \qquad \mathrm{Im}\,C_{sh}(f) = 0 ,

where :math:`P_\mathrm{tot}` is the dead-time-distorted total-band periodogram that
:func:`stingray.deadtime.model.pds_model_zhang` predicts (or, equivalently and much
faster, :func:`stingray.deadtime.model.vikhlinin_dead_time_model` with ``bin_time``,
which needs no truncated sum to converge at fine binning). This term is a
noise--noise correlation: it is there even when the source does not vary at all, and
it does **not** average down with more data.

Two things follow. A time lag measured between two bands of one module can be
confidently wrong rather than merely noisy, because the real part of the cross
spectrum is dominated by an instrumental term while the imaginary part is not. And a
steady source *manufactures* coherence: raw :math:`\gamma^2` between two bands of one
module can read several tenths with no intrinsic variability whatsoever, while a real
partial coherence is pushed the other way and badly underestimated. Perfect
coherence, :math:`\gamma^2 = 1`, is a protected special case -- a linear mixing of
two perfectly coherent signals is still perfectly coherent -- which is why the
problem is easy to miss in simulations.

A Poisson-free cross-spectral matrix
====================================

Two independent detectors observing the same source do not share dead time, so
crossing them removes every noise term. This is the cospectrum of `Bachetti et al.
2015 <https://ui.adsabs.harvard.edu/abs/2015ApJ...800..109B/abstract>`_, extended
from a single cross spectrum to the whole :math:`2 \times 2` matrix over the two
energy bands:

.. math::

   S_{11} = \mathrm{Re}\langle A_1^* B_1 \rangle, \quad
   S_{22} = \mathrm{Re}\langle A_2^* B_2 \rangle, \quad
   S_{12} = \tfrac{1}{2} \langle A_1^* B_2 + B_1^* A_2 \rangle ,

with :math:`A` and :math:`B` the two detectors and the subscript the energy band.
No entry, the diagonal included, carries a Poisson level.
:func:`stingray.deadtime.crosstalk.cross_module_cross_spectral_matrix` builds it from
four event lists::

    from stingray.deadtime.crosstalk import cross_module_cross_spectral_matrix

    matrix = cross_module_cross_spectral_matrix(
        soft_a, hard_a, soft_b, hard_b, segment_size=8.0, norm="leahy"
    )

Undoing the mixing
==================

The cross-talk above is one instance of a linear mixing. Linearising the detected
rate :math:`m_i = r_i / (1 + r_\mathrm{tot} t_d)` around the mean gives a measured
matrix

.. math::

   S_\mathrm{meas} = M \, S_\mathrm{true} \, M^T ,

so the intrinsic matrix is recovered as :math:`M^{-1} S_\mathrm{meas} (M^{-1})^T`.
With :math:`a = 1 - f_\mathrm{live}` and :math:`f_\mathrm{live} = 1 - r_\mathrm{det}
t_d` exactly, the Leahy-normalized mixing matrix is symmetric,

.. math::

   M_{ij} = \delta_{ij} - a \sqrt{p_i p_j} ,

and its off-diagonal entry is exactly the cross-talk term quoted above.
:func:`stingray.deadtime.crosstalk.unmix_cross_spectral_matrix` applies the
inverse, and
:func:`stingray.deadtime.crosstalk.coherence_from_cross_spectral_matrix` turns the
result into a coherence::

    import numpy as np
    from stingray.deadtime.crosstalk import (
        live_time_fraction, unmix_cross_spectral_matrix,
        coherence_from_cross_spectral_matrix,
    )

    f_live = live_time_fraction(detected_rate, dead_time)
    power1, power2, cross = unmix_cross_spectral_matrix(
        matrix.power1, matrix.power2, matrix.cross_power,
        f_live, matrix.band_fraction, norm=matrix.norm,
    )

    coherence = coherence_from_cross_spectral_matrix(power1, power2, cross)
    lag = -np.angle(cross) / (2 * np.pi * matrix.freq)

.. warning::

   **The matrix depends on the normalization.** Rescaling each band by its own
   amplitude unit is a diagonal similarity transformation :math:`D^{-1} M D`, so the
   same physical mixing takes three different forms: symmetric for ``"leahy"``,
   and two different asymmetric forms for raw counts (``"none"``) and fractional rms
   (``"frac"``). All three have determinant :math:`f_\mathrm{live}`, and all three
   *coincide* at an equal band split :math:`p_1 = p_2 = 1/2`, which makes the mistake
   invisible in the symmetric case that is easiest to simulate. At a 10/90 split,
   using the wrong form recovers roughly half, or 120%, of the true lag. Pass the
   ``norm`` the matrix was actually built with.

.. note::

   Form the coherence and the lag from a *band-averaged or rebinned* matrix, never
   from a single Fourier bin. Per bin the denominator :math:`P_1 P_2` crosses zero
   and the ratio explodes; average every entry of the matrix first, then divide.

The matrix depends on frequency
-------------------------------

The matrix above assumes that the live time follows the total count rate
instantaneously. It does not: the detector is dead because of an event in the
*preceding* :math:`t_d`, so the live time responds to the rate averaged over that
interval. In Fourier space the average is the response of
:func:`stingray.deadtime.model.deadtime_response`,

.. math::

   W(f) = e^{-i\pi f t_d} \, \mathrm{sinc}(f t_d) ,

the same function on which the Vikhlinin et al. 1994 noise level,
:func:`stingray.deadtime.model.vikhlinin_dead_time_model`, is built. To first order
the mixing matrix becomes :math:`M(f) = I - A(f) \, q q^T`, with
:math:`q = (\sqrt{p_1}, \sqrt{p_2})` and

.. math::

   A(f) = \frac{w W(f)}{1 + w W(f)}, \qquad w = \frac{1 - f_\mathrm{live}}{f_\mathrm{live}} ,

which is complex, and the correction is
:math:`\overline{M^{-1}} S_\mathrm{meas} (M^{-1})^T`. :math:`W \to 1` well below
:math:`1/(2\pi t_d)` (about 60 Hz for NuSTAR) and gives back the matrix above. Two
consequences:

* Above a few tens of Hz the instantaneous matrix is simply wrong: it returns half
  the true lag or less above 100 Hz at :math:`f_\mathrm{live} = 0.5`.
* The phase of :math:`W` is a delay of about :math:`f_\mathrm{live} t_d / 2`. It
  cancels in the lag between two bands with equal counts and equal fractional rms,
  and only there: otherwise the instantaneous matrix leaves up to 10% on the lag even
  at a few Hz.

Pass the dead time and the frequencies to use it::

    power1, power2, cross = unmix_cross_spectral_matrix(
        matrix.power1, matrix.power2, matrix.cross_power,
        f_live, matrix.band_fraction, norm=matrix.norm,
        dead_time=dead_time, freq=matrix.freq,
    )

For a per-segment matrix (below), pass ``freq=matrix.freq[:, None]`` so that it
broadcasts against an :math:`f_\mathrm{live}` per segment. ``bin_time`` selects the
response aliased onto the grid of a binned light curve, the same folding that turns
the Vikhlinin noise level into the Zhang et al. 1995 one; it is the right choice
only for a signal with power up to the Nyquist frequency, and a red signal is better
served by the default, unbinned response.

Variable sources: one matrix per segment
========================================

:math:`f_\mathrm{live}` is a function of the count rate, so a source that varies
between Fourier segments is described by a *different* matrix in each one. Averaging
the cross spectra first and inverting once uses an inverse that belongs to none of
the segments, and the bias does not vanish with more data: for a source that flares
fast enough to change its live time *within* a segment, one global inverse over-corrects
the lag by about 20%, while a per-segment inverse leaves only the linearisation residual
of the matrix itself. For a source that varies only between segments the two are much
closer; see the warning below.

The per-segment matrix comes from
:func:`stingray.deadtime.crosstalk.dynamical_cross_module_cross_spectral_matrix`, which
returns the same three entries with shape ``(nfreq, nseg)`` and one band fraction and
one mid-time per segment. Every function in the module broadcasts, so an
:math:`f_\mathrm{live}` of shape ``(nseg,)`` then corrects each segment with its own
matrix::

    from stingray.deadtime.crosstalk import (
        dynamical_cross_module_cross_spectral_matrix,
    )

    matrix = dynamical_cross_module_cross_spectral_matrix(
        soft_a, hard_a, soft_b, hard_b, segment_size=8.0, sample_time=1 / 256
    )

    # One live time fraction per segment, from the rate measured in that segment,
    # lined up with ``matrix.time``.
    f_live = live_time_fraction(rate_per_segment, dead_time)

    power1, power2, cross = unmix_cross_spectral_matrix(
        matrix.power1, matrix.power2, matrix.cross_power, f_live, matrix.band_fraction
    )

    # Average over segments (axis 1) only after unmixing, then form the ratios.
    coherence = coherence_from_cross_spectral_matrix(
        power1.mean(axis=1), power2.mean(axis=1), cross.mean(axis=1)
    )
    lag = -np.angle(cross.mean(axis=1)) / (2 * np.pi * matrix.freq)

.. warning::

   Normalise each segment by **its own** counts, not by the mean over the observation.
   This is why ``dynamical_cross_module_cross_spectral_matrix`` defaults to
   ``use_common_mean=False``, against stingray's usual default: the matrix of a segment
   is built from that segment's counts and band split, so the spectra it acts on have
   to be in the same units. Getting this wrong is not a detail -- in the module's own
   tests the error of a *single* global inverse moves from under 1% to about 14% on
   this choice alone, while the per-segment answer moves by less than 1%.

How much the per-segment inverse buys you over a single global one depends on *how* the
source varies, and the distinction is the segment length:

* When :math:`f_\mathrm{live}` is near-constant **within** each segment and varies only
  **between** them -- a slow flare, or a source stepping between two levels -- a single
  global inverse is already close to right, and the two agree. What is left in both is
  the few percent linearisation residual of the matrix itself, which is present even for
  a perfectly steady source and shrinks as :math:`f_\mathrm{live}` approaches 1.
* When :math:`f_\mathrm{live}` swings **inside** a segment -- fast flaring, where it can
  go from 0.9 to 0.2 in a few seconds -- no single matrix describes even one segment.
  The two schemes then separate by about 20% on the lag, with the global inverse
  over-correcting, and shortening the segments is what brings them back together.

So the per-segment builder is not a refinement that only matters at the margin, but
neither does it rescue a segment that is too long for the source. Shorten the segments
until the correction stops changing, and prefer the per-segment inverse because it is
the one with no single :math:`f_\mathrm{live}` to choose badly.

Missions that report the live time at a cadence finer than the segment length -- as
NuSTAR does, at 1 s -- already carry the information this needs. Note that the
mixing matrix is a linearisation around the mean rate of a segment, so it stops
describing a segment whose rate varies strongly *within* it: shorten the segments
until the correction stops changing. Dropping the brightest segments on a live-time
threshold does not help, and makes the answer worse.

Unmixing the amplitudes instead of the matrix
---------------------------------------------

The cross-module matrix is a product of the two detectors' Fourier amplitudes, so
unmixing the matrix is the same as unmixing each detector's two band amplitudes,
:math:`u = M^{-1} (X_1, X_2)`, and then crossing them.
:func:`stingray.deadtime.crosstalk.module_fourier_amplitudes` returns the
per-segment amplitudes of one detector, normalised so that their cross products are
exactly the entries of the per-segment matrix, and
:func:`stingray.deadtime.crosstalk.unmix_amplitudes` unmixes them::

    amp_a = module_fourier_amplitudes(soft_a, hard_a, segment_size, dt=dt)
    amp_b = module_fourier_amplitudes(soft_b, hard_b, segment_size, dt=dt)
    u_a = unmix_amplitudes(amp_a.amplitude1, amp_a.amplitude2, f_live_a,
                           amp_a.band_fraction, dead_time=dead_time,
                           freq=amp_a.freq[:, None])
    u_b = ...  # the same for detector B
    cross = 0.5 * (np.conj(u_a[0]) * u_b[1] + np.conj(u_b[0]) * u_a[1])

The result is the same as with :func:`unmix_cross_spectral_matrix`, but each detector
now gets its own live time fraction and band split, and the powers of the unmixed
amplitudes are what the error bars on the corrected lag and coherence are built from.
Use the same good time intervals for the two detectors so that the segments line up.

:func:`stingray.deadtime.fad.FAD` is complementary rather than an alternative. FAD
corrects the per-band white-noise *levels* under a variable source, but it does not
touch the phase; the unmixing corrects the matrix. The two compose.

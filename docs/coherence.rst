Coherence and its uncertainty
*****************************

What the coherence measures
===========================

Take the light curves of the same source in two energy bands. At each Fourier frequency,
the *coherence* tells how much of the variability in one band can be predicted from the
other with a linear filter (a time delay, a smoothing, a change of amplitude). It goes
from 0, when the two bands vary independently, to 1, when one band is a filtered copy of
the other.

Its starting point is the cross spectrum :math:`C = A_1^* A_2`, where :math:`A_1` and
:math:`A_2` are the Fourier amplitudes of the two bands, and the power spectra
:math:`P_1 = |A_1|^2` and :math:`P_2 = |A_2|^2`. For a single light curve segment,
:math:`|C|^2 = P_1 P_2` always, so a single segment gives no information. The coherence
appears when we average over :math:`m` segments (and/or over neighbouring frequencies):

.. math::

    \gamma^2 = \frac{|\langle C \rangle|^2}{\langle P_1 \rangle \langle P_2 \rangle}.

If the phase difference between the two bands changes randomly from segment to segment,
the averaged cross spectrum shrinks, and so does the coherence.

Poisson noise and the *intrinsic* coherence
===========================================

Photon counting noise (Poisson noise) is independent in the two bands, so it lowers the
coherence of the measured light curves even when the source itself is perfectly coherent.
:func:`~stingray.fourier.raw_coherence` calculates the coherence of the data as they are.
:func:`~stingray.fourier.intrinsic_coherence` (also available as
:meth:`~stingray.AveragedCrossspectrum.intrinsic_coherence`) estimates the coherence of
the source alone, following Vaughan & Nowak (1997, ApJ 474, L43, "VN97"):

.. math::

    \gamma^2 = \frac{|\langle C \rangle|^2 - b^2}{S_1 S_2 - c^2}

* :math:`S_n = \langle P_n \rangle - N_n` are the source powers, after subtracting the
  Poisson noise level :math:`N_n`.
* :math:`b^2 = \langle P_1 \rangle \langle P_2 \rangle / m` is the *bias* of the numerator.
  The averaged cross spectrum scatters around its true value, and the square of a
  scattering quantity is, on average, larger than the square of its true value, by the
  variance of the scatter. This variance includes the Poisson noise and also the part of
  the source variability that is not coherent between the bands.
* :math:`c^2 = \max(|\langle C \rangle|^2 - b^2, 0) / m` corrects a smaller bias of the
  denominator: when the bands are coherent, the two powers scatter together.

The bias :math:`b^2` is exact, and :math:`c^2` is correct to first order, for averages
of Gaussian Fourier amplitudes, a good
description of the noise-like variability of accreting sources once a few tens of
segments are averaged.

The estimate is not clipped. If the cross spectrum is weaker than expected from noise
alone, it is negative. This is what an unbiased estimate of a small value does: averaged
over many frequencies or observations, the negative and positive fluctuations cancel. The
result is NaN only where a power is not above its noise level, because then the source
powers in the denominator do not exist.

>>> import numpy as np
>>> from stingray.fourier import intrinsic_coherence
>>> cross = np.array([2 + 1j, 4, 1])
>>> power1, power2 = np.array([10.0, 10.0, 10.0]), np.array([8.0, 8.0, 8.0])
>>> coh = intrinsic_coherence(cross, power1, power2, 2.0, 2.0, 50)
>>> print(np.round(coh, 3))
[ 0.071  0.302 -0.013]

Here the noise levels are 2 in both bands, and the spectra were averaged over 50
segments.

Error bars
==========

The distribution of the coherence estimate is far from Gaussian when the signal is weak
or the coherence is close to 0 or 1, so a single "plus or minus" number does not describe
it well. ``intrinsic_coherence`` gives a 1-sigma *interval* instead:

>>> coh, lower, upper = intrinsic_coherence(
...     cross, power1, power2, 2.0, 2.0, 50, return_interval=True
... )
>>> print(np.round(lower, 3))
[ 0.023  0.2   -0.04 ]
>>> print(np.round(upper, 3))
[0.167 0.419 0.049]

``return_uncertainty=True`` gives instead half the width of the interval, a symmetric
error bar convenient for plotting:

>>> coh, err = intrinsic_coherence(cross, power1, power2, 2.0, 2.0, 50, return_uncertainty=True)
>>> print(np.round(err, 3))
[0.072 0.109 0.044]

How the interval is built
-------------------------

For every candidate true value :math:`\gamma^2_0`, we calculate how much the estimate
would scatter if :math:`\gamma^2_0` were true, :math:`\sigma(\gamma^2_0)`. The interval
contains all the candidates for which the measured value is within one
:math:`\sigma(\gamma^2_0)` of :math:`\gamma^2_0`:

.. math::

    |\hat{\gamma}^2 - \gamma^2_0| \le \sigma(\gamma^2_0).

The difference from the usual error bar is that :math:`\sigma` is calculated under each
candidate, not at the measured value. The usual way fails exactly where the interval
matters most: with a measured coherence near zero, the scatter calculated at the measured
value is also near zero, and the error bar is far too small. The interval is asymmetric
when :math:`\sigma` changes quickly with :math:`\gamma^2_0`, as it does near 0 and 1.

The variance has two parts. With :math:`r_n = N_n / S_n` the noise-to-signal ratio of
each band and :math:`q = r_1 + r_2 + r_1 r_2`,

.. math::

    \sigma^2(\gamma^2_0) = \frac{2\gamma^2_0 (1 - \gamma^2_0)(1 - \gamma^2_0 + q)
    + (\gamma^2_0)^2 (r_1 + r_2)^2}{m}
    + \left(\kappa \frac{\langle P_1 \rangle \langle P_2 \rangle - \gamma^2_0 S_1 S_2}
    {m S_1 S_2}\right)^2 .

* The first term is the standard error propagation, done exactly for Gaussian Fourier
  amplitudes, including the correlations between the cross spectrum and the two powers.
  Without noise it gives back the classic result of Bendat & Piersol,
  :math:`2\gamma^2(1-\gamma^2)^2/m`; at full coherence :math:`\sigma` reduces to
  :math:`(r_1 + r_2)/\sqrt{m}`, the effect of noise alone.
* The second term dominates at low coherence, where the first one vanishes. With no
  coherent signal, :math:`|\langle C \rangle|^2` follows an exponential distribution with
  mean :math:`b^2`. A band :math:`\pm\kappa b^2` around that mean contains 68.27% of the
  distribution for :math:`\kappa = 0.830`.

Like the estimate, the interval is not restricted to [0, 1]. An interval entirely below
zero means that the cross spectrum is weaker than expected from noise alone; one entirely
above 1 means that the powers fluctuated low. Restricting it to [0, 1] would make it
shrink to the single point [0, 0] (or [1, 1]) in about 15% of the cases where the true
coherence sits at the edge, which would wrongly suggest an exact value with no
uncertainty. Candidates are searched up to 2; an upper end of 2 means that the data do
not constrain the coherence from above.

How well it works
-----------------

We tested the interval with simulated Fourier amplitudes, checking how often it contains
the true coherence (its *coverage*): about 68% of the time for a 1-sigma interval. The
table shows the coverage for 2000 simulated spectra per entry, as a function of the
number of averaged segments :math:`m`, the source-to-noise power ratio :math:`S/N`, and
the true coherence. The statistical uncertainty of each entry is about 0.01.

==== ======== ======== ======== ======== ========
m    S/N      γ² = 0   γ² = 0.1 γ² = 0.6 γ² = 1
==== ======== ======== ======== ======== ========
10   0.3      0.74     0.76     0.75     0.74
10   1        0.70     0.72     0.67     0.67
10   10       0.70     0.70     0.67     0.65
50   0.3      0.70     0.77     0.72     0.70
50   1        0.70     0.72     0.67     0.69
50   10       0.69     0.69     0.65     0.66
300  0.3      0.69     0.71     0.67     0.68
300  1        0.69     0.66     0.70     0.68
300  10       0.70     0.68     0.68     0.69
==== ======== ======== ======== ======== ========

The interval is slightly conservative (it covers more often than 68%) only when the
signal is too weak to measure the coherence well. For comparison, the formula used in
previous versions of Stingray (VN97, eq. 8) covered anywhere between 0% and 100% of the time on the same
kind of grid, and gave no error bar at all for about 30% of the spectra.

When the subject band is part of the reference band
---------------------------------------------------

Everything above assumes that the two bands contain different photons. A common choice
is instead a broad reference band that also contains the subject band. The two bands
then share the Poisson noise of the subject band, and the averaged cross spectrum
contains a noise term that does not vanish on average. Following Ingram (2019, MNRAS 489,
3927, eqs. 6 and 11), subtract it from the cross spectrum before calling
``intrinsic_coherence``: it is the noise level of the subject band, :math:`N_1`, in
absolute (or no) normalisation, and :math:`N_1 \mu_1 / \mu_2` in fractional
normalisation, where :math:`\mu_n` are the mean count rates. The noise level to pass for
the reference band is that of the whole reference band.
:meth:`~stingray.AveragedCrossspectrum.intrinsic_coherence` does not do this
subtraction, even with ``channels_overlap=True``; call
:func:`~stingray.fourier.intrinsic_coherence` directly.

Even after the subtraction, the shared noise makes the two powers and the cross spectrum
scatter together more than the formulas above assume. We checked with simulated Fourier
amplitudes, for a subject band that is half of the reference band:

* :math:`b^2` is still exact, but the estimate is biased low, by 0.03–0.07 for
  :math:`m = 30`, 0.008–0.02 for :math:`m = 100` and 0.002–0.007 for :math:`m = 300`.
  The formula used in previous versions of Stingray was biased low too, by 60–90% as
  much;
* the interval is too wide: it covers 87–100% of the time instead of 68%.

Both effects are negligible when the subject band is a small part of the reference band
(we checked one fiftieth). When it is a large part, use a reference band that excludes
the subject band.

About VN97, eq. 8
-----------------

The previous uncertainty followed eq. 8 of VN97. Checking it against simulations and
against the exact calculation showed a few problems, listed here for whoever compares
with older results:

* The text after eq. 7 gives the scatter of :math:`|C|^2` as :math:`\sqrt{2}\,n^2`;
  eq. 7 gives :math:`\sqrt{2}\,|C|\,n`. The same typo makes the first term of eq. 8 too
  small by a factor of about :math:`m`.
* The coherence error quoted from Bendat & Piersol is a *relative* error, but eq. 8 uses
  it as an absolute one. Stingray followed the literal reading, which inflates the error
  that term by :math:`1/\gamma^4` (a factor of 100 at :math:`\gamma^2 = 0.1`).
* The noise terms leave out the scatter of the source's own power, and the terms are added
  as if they were independent, while the cross spectrum and the powers are correlated.
* The bias :math:`n^2` of VN97 leaves out the incoherent part of the source variability.
  The bias :math:`b^2` above includes it.
* Eq. 9, meant for low coherence, contains a term that grows without limit as the
  coherence goes to zero.

Two detectors: no noise level to subtract
=========================================

The intrinsic coherence above needs the Poisson noise levels :math:`N_n`. They are not
always well known: dead time, for example, distorts the noise level of NuSTAR and other
instruments in a way that depends on frequency and count rate. When the same source is
observed by two independent detectors at the same time, such as the two NuSTAR modules,
the noise level is not needed at all. The Poisson noise of the two detectors is
independent, so it averages out when one detector's light curve is multiplied by the
other's (this is the *cospectrum* of Bachetti et al. 2015).

:func:`~stingray.crossspectrum.cross_detector_spectral_matrix` builds this crossing for
two energy bands. With :math:`A` and :math:`B` the Fourier amplitudes of the two
detectors, it returns the averaged cospectra of each band and the cross spectrum between
the bands,

.. math::

    S_{11} = {\rm Re}\langle A_1^* B_1\rangle, \qquad
    S_{22} = {\rm Re}\langle A_2^* B_2\rangle, \qquad
    S_{12} = \langle A_1^* B_2 + B_1^* A_2\rangle / 2,

none of which contains Poisson noise on average. It also returns the power spectrum of
each detector in each band, :math:`p_{A1}, p_{B1}, p_{A2}, p_{B2}`, which are needed for
the bias and the error bars. :func:`~stingray.fourier.cross_detector_coherence`
calculates

.. math::

    \gamma^2 = \frac{|S_{12}|^2 - b^2}{S_{11} S_{22} - c^2}, \qquad
    b^2 = \frac{p_{A1} p_{B2} + p_{B1} p_{A2} + 2 S_{11} S_{22}}{4m}, \qquad
    c^2 = \frac{\max(|S_{12}|^2 - b^2, 0)}{m}.

As before, :math:`b^2` is the average excess of :math:`|S_{12}|^2` over its true value,
due to the scatter of the averaged cross spectrum, and :math:`c^2` corrects the
correlation between the two cospectra. For Gaussian Fourier amplitudes, :math:`b^2` is
exact and :math:`c^2` is correct to first order. The result is not clipped, and it
is NaN where a cospectrum is not positive: there, no signal is detected in that band.

>>> from stingray.fourier import cross_detector_coherence
>>> auto = [np.full(3, p) for p in (5.0, 6.0, 2.0, 3.0)]  # pA1, pB1, pA2, pB2
>>> s11, s22 = np.full(3, 4.0), np.full(3, 1.0)
>>> s12 = np.array([0.5, 1 + 1j, 1.8])
>>> coh, lower, upper = cross_detector_coherence(
...     s11, s22, s12, auto, 50, return_interval=True
... )
>>> print(np.round(coh, 3))
[0.019 0.46  0.778]
>>> print(np.round(lower, 3))
[-0.018  0.318  0.613]
>>> print(np.round(upper, 3))
[0.121 0.626 0.954]

With light curves (or event lists) of two bands from two detectors, averaging the
matrix over a frequency range before calculating the coherence:

.. code-block:: python

    from stingray.crossspectrum import cross_detector_spectral_matrix

    mat = cross_detector_spectral_matrix(
        soft_a, hard_a, soft_b, hard_b, segment_size=8.0
    )
    band = (mat.freq > 0.5) & (mat.freq < 10)
    coh, lower, upper = cross_detector_coherence(
        mat.power1[band].mean(),
        mat.power2[band].mean(),
        mat.cross_power[band].mean(),
        [p[band].mean() for p in mat.auto_powers],
        mat.m * band.sum(),  # segments times frequencies averaged
        return_interval=True,
        # only needed with dead time, see below
        same_detector_cross=[c[band].mean() for c in mat.same_detector_cross],
    )

The 1-sigma interval is built in the same way as for the intrinsic coherence. The
variance under each candidate value :math:`x` has the same two parts. With
:math:`r_{Xn} = (p_{Xn} - S_{nn}) / S_{nn}` the noise-to-signal ratio of detector
:math:`X` in band :math:`n`,

.. math::

    m\,\sigma^2_1(x) = 2x(1-x)^2 + \frac{x(1-x)}{2}(r_{A1} + r_{B1} + r_{A2} + r_{B2})
    + \frac{x^2}{2}(r_{A1} r_{B1} + r_{A2} r_{B2}) + \frac{x}{2}(r_{A1} r_{B2} + r_{A2} r_{B1}),

plus the low-coherence term :math:`(\kappa\,(m b^2 - x S_{11} S_{22}) / (m S_{11} S_{22}))^2`.
In simulations with equal and unequal detectors and at least 30 averaged segments, the
interval contained the true value 63-76% of the time wherever the coherence could be
measured, both with simulated Fourier amplitudes and with simulated light curves. With
only 10 segments, it is too narrow at high coherence, and covers as little as 56% of
the time.

Dead time: correlated noise within each detector
------------------------------------------------

Dead time does not only change the noise level: it also correlates the noise of the two
bands *in the same detector*. After a photon of either band, the detector is blind to
both, so a band that receives more photons than average leaves fewer for the other. The
cross spectrum of the two bands in one detector, :math:`C_A = \langle A_1^* A_2\rangle`,
then contains a noise term that depends on frequency and count rate (in the simulation
described below, it is negative up to about 200 Hz and positive near 300 Hz). This term
cancels in all the entries of the matrix, so the coherence is not biased. However, it correlates the fluctuations of
:math:`S_{11}` and :math:`S_{22}` with those of :math:`S_{12}`, and this changes the
scatter of the coherence: the formula above, which ignores it, gives intervals too wide or
too narrow.

The more general calculation uses all the averaged products of the four amplitudes, which
fix the scatter of any quantity built from them, for Gaussian amplitudes (Isserlis'
theorem). Besides those in the formula above, these products include :math:`C_A` and
:math:`C_B = \langle B_1^* B_2\rangle`. ``cross_detector_spectral_matrix`` returns both, as
``same_detector_cross``; passing them to ``cross_detector_coherence`` gives the general
variance. Without dead time, the result is the same as the formula above. In a simulation
of two detectors with 2.5 ms of dead time, recording about half of 400 photons/s each, the
interval contained the true coherence 64-72% of the time with :math:`C_A` and
:math:`C_B`, in ten frequency windows from 0.1 to 400 Hz. Without them, it was too narrow
below 150 Hz, where the crosstalk is negative (coverage down to 49%), and slightly too wide
near 300 Hz, where it is positive (up to 78%).

Tutorial
========

The notebook below explains these methods step by step and checks them with simulations,
including two detectors affected by dead time.

.. toctree::
   :maxdepth: 2

   notebooks/Coherence/Coherence and its uncertainty.ipynb

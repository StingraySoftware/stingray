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

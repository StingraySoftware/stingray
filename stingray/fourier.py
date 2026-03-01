import copy
import warnings
from collections.abc import Iterable
from typing import Optional

import numpy as np
import numpy.typing as npt
from astropy.table import Table
from astropy.timeseries.periodograms.lombscargle.implementations.utils import trig_sum

from .gti import (
    generate_indices_of_segment_boundaries_binned,
    generate_indices_of_segment_boundaries_unbinned,
)

from .utils import (
    fft,
    fftfreq,
    histogram,
    show_progress,
    sum_if_not_none_or_initialize,
    fix_segment_size_to_integer_samples,
    rebin_data,
    njit,
    vectorize,
)


__all__ = [
    "integrate_power_in_frequency_range",
    "positive_fft_bins",
    "poisson_level",
    "normalize_frac",
    "normalize_abs",
    "normalize_leahy_from_variance",
    "normalize_leahy_poisson",
    "normalize_periodograms",
    "unnormalize_periodograms",
    "bias_term",
    "raw_coherence",
    "coherence",
    "intrinsic_coherence",
]


def integrate_power_in_frequency_range(
    frequency,
    power,
    frequency_range,
    power_err=None,
    df=None,
    m=1,
    poisson_power=0,
):
    """
    Integrate the power in a given frequency range.

    Parameters
    ----------
    frequency : iterable
        The frequencies of the power spectrum
    power : iterable
        The power at each frequency
    frequency_range : iterable of length 2
        The frequency range to integrate
    power_err : iterable, optional, default None
        The power error bar at each frequency
    df : float or float iterable, optional, default None
        The frequency resolution of the input data. If None, it is calculated
        from the median difference of input frequencies.
    m : int, optional, default 1
        The number of segments and/or contiguous frequency bins averaged to obtain power.
        Only needed if ``power_err`` is None
    poisson_power : float, optional, default 0
        The Poisson noise level of the power spectrum.

    Returns
    -------
    power_integrated : float
        The integrated power
    power_integrated_err : float
        The error on the integrated power

    """
    frequency = np.array(frequency, dtype=float)
    power = np.array(power)
    if not np.iscomplexobj(power):
        power = power.astype(float)
    if not isinstance(poisson_power, Iterable):
        poisson_power = np.ones_like(frequency) * poisson_power
    if df is None:
        df = np.median(np.diff(frequency))
    if not isinstance(df, Iterable):
        df = np.ones_like(frequency) * df

    # The frequency_range is in the middle or at the edge of each bin. When only a part of the
    # bin is included, we need to add the fraction of the bin that is included.
    frequency_mask = (frequency + df / 2 > frequency_range[0]) & (
        frequency - df / 2 < frequency_range[1]
    )

    freqs_to_integrate = frequency[frequency_mask]
    poisson_power = poisson_power[frequency_mask]
    correction_ratios = np.ones_like(freqs_to_integrate)
    dfs_to_integrate = df[frequency_mask]

    # The first and last bins are only partially included. We need to correct for the
    # fraction of the bin actually included.
    correction_ratios[0] = (
        freqs_to_integrate[0] + 0.5 * dfs_to_integrate[0] - frequency_range[0]
    ) / dfs_to_integrate[0]
    correction_ratios[-1] = (
        frequency_range[-1] - freqs_to_integrate[-1] + 0.5 * dfs_to_integrate[-1]
    ) / dfs_to_integrate[-1]
    dfs_to_integrate = dfs_to_integrate * correction_ratios

    powers_to_integrate = power[frequency_mask]

    if power_err is None:
        power_err_to_integrate = powers_to_integrate / np.sqrt(m)
    else:
        power_err_to_integrate = np.asanyarray(power_err)[frequency_mask]

    power_integrated = np.sum((powers_to_integrate - poisson_power) * dfs_to_integrate)
    power_err_integrated = np.sqrt(np.sum((power_err_to_integrate * dfs_to_integrate) ** 2))
    return power_integrated, power_err_integrated


def positive_fft_bins(n_bin, include_zero=False):
    """
    Give the range of positive frequencies of a complex FFT.

    This assumes we are using Numpy's FFT, or something compatible
    with it, like ``pyfftw.interfaces.numpy_fft``, where the positive
    frequencies come before the negative ones, the Nyquist frequency is
    included in the negative frequencies but only in even number of bins,
    and so on.
    This is mostly to avoid using the ``freq > 0`` mask, which is
    memory-hungry and inefficient with large arrays. We use instead a
    slice object, giving the range of bins of the positive frequencies.

    See https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details

    Parameters
    ----------
    n_bin : int
        The number of bins in the FFT, including all frequencies

    Other Parameters
    ----------------
    include_zero : bool, default False
        Include the zero frequency in the output slice

    Returns
    -------
    positive_bins : `slice`
        Slice object encoding the positive frequency bins. See examples.

    Examples
    --------
    Let us calculate the positive frequencies using the usual mask
    >>> freq = np.fft.fftfreq(10)
    >>> good = freq > 0

    This works well, but it is highly inefficient in large arrays.
    This function will instead return a `slice object`, which will work
    as an equivalent mask for the positive bins. Below, a few tests that
    this works as expected.
    >>> goodbins = positive_fft_bins(10)
    >>> assert np.allclose(freq[good], freq[goodbins])
    >>> freq = np.fft.fftfreq(11)
    >>> good = freq > 0
    >>> goodbins = positive_fft_bins(11)
    >>> assert np.allclose(freq[good], freq[goodbins])
    >>> freq = np.fft.fftfreq(10)
    >>> good = freq >= 0
    >>> goodbins = positive_fft_bins(10, include_zero=True)
    >>> assert np.allclose(freq[good], freq[goodbins])
    >>> freq = np.fft.fftfreq(11)
    >>> good = freq >= 0
    >>> goodbins = positive_fft_bins(11, include_zero=True)
    >>> assert np.allclose(freq[good], freq[goodbins])
    """
    # The zeroth bin is 0 Hz. We usually don't include it, but
    # if the user wants it, we do.
    minbin = 1
    if include_zero:
        minbin = 0

    if n_bin % 2 == 0:
        return slice(minbin, n_bin // 2)

    return slice(minbin, (n_bin + 1) // 2)


def poisson_level(norm="frac", meanrate=None, n_ph=None, backrate=0):
    r"""
    Poisson (white)-noise level in a periodogram of pure counting noise.

    For Leahy normalization, this is:

    .. math::

        P = 2

    For the fractional r.m.s. normalization, this is

    .. math::

        P = \frac{2}{\mu}

    where :math:`\mu` is the average count rate

    For the absolute r.m.s. normalization, this is

    .. math::

        P = 2 \mu

    Finally, for the unnormalized periodogram, this is

    .. math::

        P = N_{\rm ph}

    Parameters
    ----------
    norm : str, default "frac"
        Normalization of the periodogram. One of ["abs", "frac", "leahy",
        "none"].

    Other Parameters
    ----------------
    meanrate : float, default None
        Mean count rate in counts/s. Needed for r.m.s. norms ("abs" and
        "frac").
    n_ph : float, default None
        Total number of counts in the light curve. Needed if ``norm=="none"``.
    backrate : float, default 0
        Background count rate in counts/s. Optional for fractional r.m.s. norm.

    Raises
    ------
    ValueError
        If the inputs are incompatible with the required normalization.

    Returns
    -------
    power_noise : float
        The Poisson noise level in the wanted normalization.

    Examples
    --------
    >>> poisson_level(norm="leahy")
    2.0
    >>> poisson_level(norm="abs", meanrate=10.)
    20.0
    >>> poisson_level(norm="frac", meanrate=10.)
    0.2
    >>> poisson_level(norm="none", n_ph=10)
    10.0
    >>> poisson_level(norm="asdfwrqfasdh3r", meanrate=10.)
    Traceback (most recent call last):
    ...
    ValueError: Unknown value for norm: asdfwrqfasdh3r...
    >>> poisson_level(norm="none", meanrate=10)
    Traceback (most recent call last):
    ...
    ValueError: Bad input parameters for norm none...
    >>> poisson_level(norm="abs", n_ph=10)
    Traceback (most recent call last):
    ...
    ValueError: Bad input parameters for norm abs...
    """
    # Various ways the parameters are wrong.
    # We want the noise in rms norm, but don't specify the mean rate.
    bad_input = norm.lower() in ["abs", "frac"] and meanrate is None
    # We want the noise in unnormalized powers, without giving n_ph.
    bad_input = bad_input or (norm.lower() == "none" and n_ph is None)

    if bad_input:
        raise ValueError(
            f"Bad input parameters for norm {norm}: n_ph={n_ph}, " f"meanrate={meanrate}"
        )

    if norm == "abs":
        return 2.0 * meanrate
    if norm == "frac":
        return 2.0 / (meanrate - backrate) ** 2 * meanrate
    if norm == "leahy":
        return 2.0
    if norm == "none":
        return float(n_ph)

    raise ValueError(f"Unknown value for norm: {norm}")


def normalize_frac(unnorm_power, dt, n_bin, mean_flux, background_flux=0):
    r"""
    Fractional rms normalization.

    .. math::

        P = \frac{P_{\rm Leahy}}{\mu} = \frac{2T}{N_{\rm ph}^2}P_{\rm unnorm}

    where :math:`\mu` is the mean count rate, :math:`T` is the length of
    the observation, and :math:`N_{\rm ph}` the number of photons.
    Alternative formulas found in the literature substitute :math:`T=N\,dt`,
    :math:`\mu=N_{\rm ph}/T`, which give equivalent results.

    If the background can be estimated, one can calculate the source rms
    normalized periodogram as

    .. math::

        P = P_{\rm Leahy} \frac{\mu}{(\mu - \beta)^2}

    or

    .. math::

        P = \frac{2T}{(N_{\rm ph} - \beta T)^2}P_{\rm unnorm}

    where :math:`\beta` is the background count rate.

    This is also called the Belloni or Miyamoto normalization.
    In this normalization, the periodogram is in units of
    :math:`(rms/mean)^2 Hz^{-1}`, and the squared root of the
    integrated periodogram will give the fractional rms in the
    required frequency range.

    Belloni & Hasinger (1990) A&A 230, 103

    Miyamoto et al. (1991), ApJ 383, 784

    Parameters
    ----------
    unnorm_power : `np.array` of `float` or `complex`
        The unnormalized (cross-)spectral powers
    dt : float
        The sampling time
    n_bin : int
        The number of bins in the light curve
    mean_flux : float
        The mean of the light curve used to calculate the periodogram.
        If the light curve is in counts, it gives the mean counts per
        bin.

    Other parameters
    ----------------
    background_flux : float, default 0
        The background flux, in the same units as `mean_flux`.

    Returns
    -------
    power : `np.array` of the same kind and shape as `unnorm_power`
        The normalized powers.

    Examples
    --------
    >>> mean = var = 1000000
    >>> back = 100000
    >>> n_bin = 1000000
    >>> dt = 0.2
    >>> meanrate = mean / dt
    >>> backrate = back / dt
    >>> lc = np.random.poisson(mean, n_bin)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_frac(pds, dt, lc.size, mean)
    >>> assert np.isclose(
    ...     pdsnorm[1:n_bin//2].mean(), poisson_level(meanrate=meanrate,norm="frac"), rtol=0.01)
    >>> pdsnorm = normalize_frac(pds, dt, lc.size, mean, background_flux=back)
    >>> assert np.isclose(pdsnorm[1:n_bin//2].mean(),
    ...                   poisson_level(meanrate=meanrate,norm="frac",backrate=backrate),
    ...                   rtol=0.01)
    """
    #     (mean * n_bin) / (mean /dt) = n_bin * dt
    #     It's Leahy / meanrate;
    #     n_ph = mean * n_bin
    #     meanrate = mean / dt
    #     norm = 2 / (n_ph * meanrate) = 2 * dt / (mean**2 * n_bin)

    if background_flux > 0:
        power = unnorm_power * 2.0 * dt / ((mean_flux - background_flux) ** 2 * n_bin)
    else:
        # Note: this corresponds to eq. 3 in Uttley+14
        power = unnorm_power * 2.0 * dt / (mean_flux**2 * n_bin)
    return power


def normalize_abs(unnorm_power, dt, n_bin):
    r"""
    Absolute rms normalization.

    .. math::

        P = P_{\rm frac} * \mu^2

    where :math:`\mu` is the mean count rate, or equivalently

    .. math::

        P = \frac{2}{T}P_{\rm unnorm}

    In this normalization, the periodogram is in units of
    :math:`rms^2 Hz^{-1}`, and the squared root of the
    integrated periodogram will give the absolute rms in the
    required frequency range.

    e.g. Uttley & McHardy, MNRAS 323, L26

    Parameters
    ----------
    unnorm_power : `np.array` of `float` or `complex`
        The unnormalized (cross-)spectral powers
    dt : float
        The sampling time
    n_bin : int
        The number of bins in the light curve

    Returns
    -------
    power : `np.array` of the same kind and shape as `unnorm_power`
        The normalized powers.

    Examples
    --------
    >>> mean = var = 100000
    >>> n_bin = 1000000
    >>> dt = 0.2
    >>> meanrate = mean / dt
    >>> lc = np.random.poisson(mean, n_bin)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_abs(pds, dt, lc.size)
    >>> assert np.isclose(
    ...     pdsnorm[1:n_bin//2].mean(), poisson_level(norm="abs", meanrate=meanrate), rtol=0.01)
    """
    #     It's frac * meanrate**2; Leahy / meanrate * meanrate**2
    #     n_ph = mean * n_bin
    #     meanrate = mean / dt
    #     norm = 2 / (n_ph * meanrate) * meanrate**2 = 2 * dt / (mean**2 * n_bin) * mean**2 / dt**2

    return unnorm_power * 2.0 / n_bin / dt


def normalize_leahy_from_variance(unnorm_power, variance, n_bin):
    r"""
    Leahy+83 normalization, from the variance of the lc.

    .. math::

        P = \frac{P_{\rm unnorm}}{N <\delta{x}^2>}

    In this normalization, the periodogram of a single light curve
    is distributed according to a chi squared distribution with two
    degrees of freedom.

    In this version, the normalization is obtained by the variance
    of the light curve bins, instead of the more usual version with the
    number of photons. This allows to obtain this normalization also
    in the case of non-Poisson distributed data.

    Parameters
    ----------
    unnorm_power : `np.array` of `float` or `complex`
        The unnormalized (cross-)spectral powers
    variance : float
        The mean variance of the light curve bins
    n_bin : int
        The number of bins in the light curve

    Returns
    -------
    power : `np.array` of the same kind and shape as `unnorm_power`
        The normalized powers.

    Examples
    --------
    >>> mean = var = 100000.
    >>> n_bin = 1000000
    >>> lc = np.random.poisson(mean, n_bin).astype(float)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_leahy_from_variance(pds, var, lc.size)
    >>> assert np.isclose(pdsnorm[0], 2 * np.sum(lc), rtol=0.01)
    >>> assert np.isclose(pdsnorm[1:n_bin//2].mean(), poisson_level(norm="leahy"), rtol=0.01)

    If the variance is zero, it will fail:
    >>> pdsnorm = normalize_leahy_from_variance(pds, 0., lc.size)
    Traceback (most recent call last):
    ...
    ValueError: The variance used to normalize the ...
    """
    if variance == 0.0:
        raise ValueError("The variance used to normalize the periodogram is 0.")
    return unnorm_power * 2.0 / (variance * n_bin)


def normalize_leahy_poisson(unnorm_power, n_ph):
    r"""
    Leahy+83 normalization.

    .. math::

        P = \frac{2}{N_{\rm ph}} P_{\rm unnorm}

    In this normalization, the periodogram of a single light curve
    is distributed according to a chi squared distribution with two
    degrees of freedom.

    Leahy et al. 1983, ApJ 266, 160

    Parameters
    ----------
    unnorm_power : `np.array` of `float` or `complex`
        The unnormalized (cross-)spectral powers
    variance : float
        The mean variance of the light curve bins
    n_bin : int
        The number of bins in the light curve

    Returns
    -------
    power : `np.array` of the same kind and shape as `unnorm_power`
        The normalized powers.

    Examples
    --------
    >>> mean = var = 100000.
    >>> n_bin = 1000000
    >>> lc = np.random.poisson(mean, n_bin).astype(float)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_leahy_poisson(pds, np.sum(lc))
    >>> assert np.isclose(pdsnorm[0], 2 * np.sum(lc), rtol=0.01)
    >>> assert np.isclose(pdsnorm[1:n_bin//2].mean(), poisson_level(norm="leahy"), rtol=0.01)
    """
    return unnorm_power * 2.0 / n_ph


def normalize_periodograms(
    unnorm_power,
    dt,
    n_bin,
    mean_flux=None,
    n_ph=None,
    variance=None,
    background_flux=0.0,
    norm="frac",
    power_type="all",
):
    """
    Wrapper around all the normalize_NORM methods.

    Normalize the cross-spectrum or the power-spectrum to Leahy, absolute rms^2,
    fractional rms^2 normalization, or not at all.

    Parameters
    ----------
    unnorm_power: numpy.ndarray
        The unnormalized cross spectrum.

    dt: float
        The sampling time of the light curve

    n_bin: int
        The number of bins in the light curve

    Other parameters
    ----------------
    mean_flux: float
        The mean of the light curve used to calculate the powers
        (If a cross spectrum, the geometrical mean of the light
        curves in the two channels). Only relevant for "frac" normalization

    n_ph: int or float
        The number of counts in the light curve used to calculate
        the unnormalized periodogram. Only relevant for Leahy normalization.

    variance: float
        The average variance of the measurements in light curve (if a cross
        spectrum,  the geometrical mean of the variances in the two channels).
        **NOT** the variance of the light curve, but of each flux measurement
        (square of light curve error bar)! Only relevant for the Leahy
        normalization of non-Poissonian data.

    norm : str
        One of ``leahy`` (Leahy+83), ``frac`` (fractional rms), ``abs``
        (absolute rms),

    power_type : str
        One of ``real`` (real part), ``all`` (all complex powers), ``abs``
        (absolute value)

    background_flux : float, default 0
        The background flux, in the same units as `mean_flux`.

    Returns
    -------
    power: numpy.nd.array
        The normalized co-spectrum (real part of the cross spectrum). For
        'none' normalization, imaginary part is returned as well.
    """

    if norm == "leahy" and variance is not None:
        pds = normalize_leahy_from_variance(unnorm_power, variance, n_bin)
    elif norm == "leahy":
        pds = normalize_leahy_poisson(unnorm_power, n_ph)
    elif norm == "frac":
        pds = normalize_frac(unnorm_power, dt, n_bin, mean_flux, background_flux=background_flux)
    elif norm == "abs":
        pds = normalize_abs(unnorm_power, dt, n_bin)
    elif norm == "none":
        pds = unnorm_power
    else:
        raise ValueError("Unknown value for the norm")

    if power_type == "all":
        return pds
    if power_type == "real":
        return pds.real
    if power_type in ["abs", "absolute"]:
        return np.abs(pds)
    raise ValueError("Unrecognized power type")


def unnormalize_periodograms(
    norm_power, dt, n_bin, n_ph, variance=None, background_flux=0.0, norm=None, power_type="all"
):
    """
    Wrapper around all the normalize_NORM methods.

    Unnormalize the power of the cross-spectrum to Leahy, absolute rms^2,
    fractional rms^2 normalization, or not at all.

    Parameters
    ----------
    norm_power: numpy.ndarray
        The normalized cross-spectrum or poisson noise

    dt: float
        The sampling time of the light curve

    n_bin: int
        The number of bins in the light curve

    Other parameters
    ----------------
    mean_flux: float
        The mean of the light curve used to calculate the powers
        (If a cross spectrum, the geometrical mean of the light
        curves in the two channels). Only relevant for "frac" normalization

    n_ph: int or float
        The number of counts in the light curve used to calculate
        the unnormalized periodogram. Only relevant for Leahy normalization.

    variance: float
        The average variance of the measurements in light curve (if a cross
        spectrum,  the geometrical mean of the variances in the two channels).
        **NOT** the variance of the light curve, but of each flux measurement
        (square of light curve error bar)! Only relevant for the Leahy
        normalization of non-Poissonian data.

    norm : str
        One of ``leahy`` (Leahy+83), ``frac`` (fractional rms), ``abs``
        (absolute rms),

    power_type : str
        One of ``real`` (real part), ``all`` (all complex powers), ``abs``
        (absolute value)

    background_flux : float, default 0
        The background flux, in the same units as `mean_flux`.

    Returns
    -------
    power: numpy.nd.array
        The normalized co-spectrum (real part of the cross spectrum). For
        'none' normalization, imaginary part is returned as well.
    """

    if norm == "leahy" and variance is not None:
        unnorm_power = norm_power * (variance * n_ph) / 2.0
    elif norm == "leahy":
        unnorm_power = norm_power * n_ph / 2.0
    elif norm == "frac":
        if background_flux > 0:
            unnorm_power = norm_power * ((n_ph / n_bin - background_flux) ** 2 * n_bin) / (2.0 * dt)
        else:
            unnorm_power = norm_power * (n_ph**2 / n_bin) / (2.0 * dt)
    elif norm == "abs":
        unnorm_power = norm_power * dt * n_bin / 2.0
    elif norm == "none":
        unnorm_power = norm_power
    else:
        raise ValueError("Unknown value for the norm")

    if power_type == "all":
        return unnorm_power
    if power_type == "real":
        return unnorm_power.real
    if power_type in ["abs", "absolute"]:
        return np.abs(unnorm_power)
    raise ValueError("Unrecognized power type")


@vectorize(
    [
        "float64(float64, float64, float64, float64, int64, float64)",
        "float64(float64, float64, float64, float64, float64, float64)",
    ],
    nopython=True,
)
def _bias_term(power1, power2, power1_noise, power2_noise, n_ave, input_intrinsic_coherence):
    if n_ave > 500:
        return 0.0
    bsq = power1 * power2 - input_intrinsic_coherence * (power1 - power1_noise) * (
        power2 - power2_noise
    )
    return bsq / n_ave


def bias_term(power1, power2, power1_noise, power2_noise, n_ave, intrinsic_coherence=1.0):
    """
    Bias term needed to calculate the coherence.

    Introduced by
    Vaughan & Nowak 1997, ApJ 474, L43

    but implemented here according to the formulation in
    Ingram 2019, MNRAS 489, 392

    As recommended in the latter paper, returns 0 if n_ave > 500

    Parameters
    ----------
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int or float
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    intrinsic_coherence : float, default 1
        If known, the intrinsic coherence.

    Returns
    -------
    bias : float `np.array`, same shape as ``power1`` and ``power2``
        The bias term
    """

    return _bias_term(power1, power2, power1_noise, power2_noise, n_ave, intrinsic_coherence)


@vectorize(["float64(float64, float64, float64)"], nopython=True)
def _apply_low_lim_to_coherence_uncertainty(coherence, uncertainty, min_uncertainty):
    """
    Apply a low limit to the uncertainty on the coherence, to avoid zero or negative uncertainties.

    Parameters
    ----------
    coherence : float or float `np.array`
        The coherence values at all frequencies.
    uncertainty : float or float `np.array`
        The uncertainty on the coherence at all frequencies. Must have the same
        shape as `coherence`.
    min_uncertainty : float or float `np.array`
        The minimum uncertainty to apply. Can be a single value or an array of
        the same shape as `coherence` and `uncertainty`.

    Returns
    -------
    uncertainty : float or float `np.array`
        The uncertainty on the coherence, with the low limit applied.

    Examples
    --------
    >>> coherence = np.array([0.5, 0.0, 0.5])
    >>> uncertainty = np.array([0.1, 0.0, 0.05])
    >>> min_uncertainty = 0.01
    >>> expected = np.array([0.1, 0.01, 0.05])
    >>> res = _apply_low_lim_to_coherence_uncertainty(coherence, uncertainty, min_uncertainty)
    >>> assert np.allclose(res, expected)

    # The same, but with a different minimum uncertainty for each frequency.
    >>> min_uncertainty = np.array([0.01, 0.1, 0.1])
    >>> res = _apply_low_lim_to_coherence_uncertainty(coherence, uncertainty, min_uncertainty)
    >>> expected = np.array([0.1, 0.1, 0.1])
    >>> assert np.allclose(res, expected)

    # All the same, but with scalar inputs.
    >>> coherence = 0.5
    >>> uncertainty = 0.0
    >>> min_uncertainty = 0.01
    >>> expected = 0.01
    >>> res = _apply_low_lim_to_coherence_uncertainty(coherence, uncertainty, min_uncertainty)
    >>> assert np.isclose(res, expected)

    """

    bad = (coherence == 0) | (uncertainty < min_uncertainty)
    if bad:
        uncertainty = min_uncertainty
    return uncertainty


@vectorize(
    [
        "float64(complex128, float64, float64, float64, float64, int64, float64)",
        "float64(complex128, float64, float64, float64, float64, float64, float64)",
    ],
    nopython=True,
)
def _raw_coherence(
    cross_power,
    power1,
    power2,
    power1_noise,
    power2_noise,
    n_ave,
    intrinsic_coherence,
):
    """
    Raw coherence estimations from cross and power spectra.

    Vaughan & Nowak 1997, ApJ 474, L43

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int or float
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    intrinsic_coherence : float, default 1
        If known, the intrinsic coherence.
    return_uncertainty : bool, default False
        Whether to return the uncertainty on the coherence, calculated according to VN97

    Returns
    -------
    coherence : float `np.array`
        The raw coherence values at all frequencies.
    """
    bsq = _bias_term(power1, power2, power1_noise, power2_noise, n_ave, intrinsic_coherence)
    num = (cross_power * np.conj(cross_power)).real - bsq
    if num < 0:
        num = 0

    den = power1 * power2

    coherence = num / den

    return coherence


def raw_coherence(
    cross_power,
    power1,
    power2,
    power1_noise,
    power2_noise,
    n_ave,
    intrinsic_coherence=1.0,
    return_uncertainty=False,
):
    r"""
    Raw coherence estimations from cross and power spectra.

    The function is defined as (see Ingram 2019 [#]_ :

    .. math::

        \tilde{g}^2(f) = \frac{|\langle \tilde{C}(f) \rangle|^2 - \tilde{b}^2}
        {\langle \tilde{P}_1(f) \rangle \langle \tilde{P}_2(f) \rangle}

    where :math:`\tilde{b}^2` is the bias term that accounts for the contribution of Poisson
    noise to the cross spectrum (see :func:`stingray.fourier.bias_term`), and tilde generally indicates noisy
    measurements of the cross spectrum :math:`C` and the power spectra :math:`P_n`.

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int or float
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    intrinsic_coherence : float, default 1
        If known, the intrinsic coherence.
    return_uncertainty : bool, default False
        Whether to return the uncertainty on the coherence, calculated according to VN97

    Returns
    -------
    coherence : float `np.array`
        The raw coherence values at all frequencies.

    References
    ----------
    .. [#] https://doi.org/10.1093/mnras/stz2409
    """

    coherence = _raw_coherence(
        cross_power,
        power1,
        power2,
        power1_noise,
        power2_noise,
        n_ave,
        intrinsic_coherence,
    )
    if return_uncertainty:
        min_uncertainty = 1 / n_ave
        uncertainty = (2**0.5 * coherence * (1 - coherence)) / (np.sqrt(coherence) * n_ave**0.5)
        uncertainty = _apply_low_lim_to_coherence_uncertainty(
            coherence, uncertainty, min_uncertainty
        )
        return coherence, uncertainty
    return coherence


@njit
def _intrinsic_coherence_uncertainties(
    bsq, num, power1_sub, power2_sub, power1_noise, power2_noise, coherence, n_ave
):
    """Calculate the uncertainty on the intrinsic coherence, according to VN97, eq. 8.

    Parameters
    ----------
    bsq : float
        The bias term squared, calculated according to the bias_term function.
    num : float
        The numerator of the coherence calculation, i.e. (cross_power * np.conj(cross_power)).real - bsq.
    power1_sub : float
        The subtracted power of the first band, i.e. power1 - power1_noise
    power2_sub : float
        The subtracted power of the second band, i.e. power2 - power2_noise
    power1_noise : float
        The Poisson noise level of the first band periodogram
    power2_noise : float
        The Poisson noise level of the second band periodogram
    coherence : float
        The intrinsic coherence calculated according to the _intrinsic_coherence function.
    n_ave : int or float
        The number of intervals that have been averaged to obtain the input spectra
    """
    # Terms from VN97, eq. 8, for the uncertainty on the coherence.
    err1 = 2.0 * (bsq**2) * n_ave / (num - bsq) ** 2
    err2 = (power1_noise / power1_sub) ** 2 + (power2_noise / power2_sub) ** 2
    err3 = 2.0 * ((1 - coherence) / (coherence**1.5)) ** 2
    uncertainty = coherence / np.sqrt(n_ave) * np.sqrt(err1 + err2 + err3)
    return uncertainty


@vectorize(
    [
        "bool(float64, float64, float64, float64, int64, float64)",
        "bool(float64, float64, float64, float64, float64, float64)",
    ],
    nopython=True,
)
def check_powers_for_intrinsic_coherence(
    power1, power2, power1_noise, power2_noise, n_ave, threshold
):
    """Check if the powers are above the threshold for the intrinsic coherence to be well defined.

    Parameters
    ----------
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int or float
        number of intervals that have been averaged to obtain the input spectra
    threshold : float
        The threshold in sigma above the noise level for the powers to be considered "high".

    Other Parameters
    ----------------

    Returns
    -------
    low_power1 : bool `np.array`
        Whether the powers of the first band are below the threshold.
    low_power2 : bool `np.array`
        Whether the powers of the second band are below the threshold.

    """
    # Consider low power when the power is less than 3 sigma above the noise level.
    # This is somewhat arbitrary, better criteria suggestions are welcome.
    low_power1 = (power1 - power1_noise) <= threshold * power1_noise / np.sqrt(n_ave)
    low_power2 = (power2 - power2_noise) <= threshold * power2_noise / np.sqrt(n_ave)
    return low_power1 or low_power2


@njit
def _intrinsic_coherence(
    cross_power,
    power1,
    power2,
    power1_noise,
    power2_noise,
    n_ave,
):
    """
    Intrinsic coherence estimations from cross and power spectra.

    Vaughan & Nowak 1997, ApJ 474, L43

    For errors, assumes **high powers, high coherence**. See eq. 8 of the paper.
    Powers below 3 sigma above the noise level (P_noise / sqrt(MW)) are considered too low,
    and the coherence will be set to NaN.

    TODO: implement a more general treatment of the uncertainty.

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int or float
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    return_uncertainty : bool, default False
        Whether to return the uncertainty on the coherence, calculated according to
        Vaughan & Nowak 1997, ApJ 474, L43, eq. 8.

    Returns
    -------
    coherence : float `np.array` or tuple of two float `np.array`
        The intrinsic coherence values at all frequencies. It is 0 if the numerator
        of the coherence calculation is negative, and NaN if the powers are too low compared
        to the noise level.
    uncertainty : float `np.array`, optional
        The uncertainty on the intrinsic coherence, calculated according to Vaughan & Nowak
        1997, ApJ 474, L43, eq. 8.
    """

    if check_powers_for_intrinsic_coherence(power1, power2, power1_noise, power2_noise, n_ave, 3):
        # If the powers are too close to the noise level, the coherence is not well defined.
        return np.nan, np.nan

    bsq = _bias_term(power1, power2, power1_noise, power2_noise, n_ave, 1)
    num = (cross_power * np.conj(cross_power)).real - bsq
    if num < 0:
        return 0.0, np.nan

    power1_sub = power1 - power1_noise
    power2_sub = power2 - power2_noise

    den = power1_sub * power2_sub

    coherence = num / den

    uncertainty = _intrinsic_coherence_uncertainties(
        bsq, num, power1_sub, power2_sub, power1_noise, power2_noise, coherence, n_ave
    )

    return coherence, uncertainty


@njit
def _intrinsic_coherence_with_adjusted_bias(
    cross_power,
    power1,
    power2,
    power1_noise,
    power2_noise,
    n_ave,
    atol=0.01,
):
    """
    Intrinsic coherence estimations from cross and power spectra, with adjusted bias term.

    Follows the procedure from sec. 5 of Ingram 2019, MNRAS 489, 3927, which iteratively
    adjusts the bias term

    For errors, assumes **high powers, high coherence**. See eq. 8 of Vaughan & Nowak 1997.
    Powers below 3 sigma above the noise level (P_noise / sqrt(MW)) are considered too low,
    and the coherence will be set to NaN.

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int or float
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    atol: float, default 0.01
        The absolute tolerance for the convergence of the iterative procedure to adjust
        the bias term.

    Returns
    -------
    coherence : float `np.array` or tuple of two float `np.array`
        The intrinsic coherence values at all frequencies. It is 0 if the numerator
        of the coherence calculation is negative, and NaN if the powers are too low compared
        to the noise level.
    uncertainty : float `np.array`, optional
        The uncertainty on the intrinsic coherence, calculated according to Vaughan & Nowak
        1997, ApJ 474, L43, eq. 8.
    not_converged_flag : bool `np.array`
        Whether the bias term adjustment procedure converged (False) or not (True).

    """

    if check_powers_for_intrinsic_coherence(power1, power2, power1_noise, power2_noise, n_ave, 3):
        # If the powers are too close to the noise level, the coherence is not well defined.
        return np.nan, np.nan, False

    # Consider low power when the power is less than 3 sigma above the noise level.
    # This is somewhat arbitrary, better criteria suggestions are welcome.
    power1_sub = power1 - power1_noise
    power2_sub = power2 - power2_noise

    current_coherence = 1.0

    converged = False
    for _ in range(40):
        bsq = _bias_term(power1, power2, power1_noise, power2_noise, n_ave, current_coherence)
        num = (cross_power * np.conj(cross_power)).real - bsq
        if num <= 0:
            # if the current coherence still drops, the bias term increases,
            # so at this point it is useless to keep iterating.
            return 0.0, np.nan, False

        den = power1_sub * power2_sub

        coherence = num / den
        if np.abs(current_coherence - coherence) < atol:
            converged = True
            break
        current_coherence = coherence

    uncertainty = _intrinsic_coherence_uncertainties(
        bsq, num, power1_sub, power2_sub, power1_noise, power2_noise, coherence, n_ave
    )
    return coherence, uncertainty, not converged


@njit
def _intrinsic_coherence_array(
    cross_power,
    power1,
    power2,
    power1_noise,
    power2_noise,
    n_ave,
    adjust_bias=False,
    atol=0.01,
):
    r"""
    Intrinsic coherence estimations from cross and power spectra.

    Vaughan & Nowak 1997, ApJ 474, L43

    .. math::

        \tilde{\gamma}^2(f) = \frac{|\langle \tilde{C}(f) \rangle|^2 - \tilde{b}^2}
        {(\langle \tilde{P}_1(f) \rangle - \tilde{P}_{1, \rm noise})
        (\langle \tilde{P}_2(f) \rangle - \tilde{P}_{2, \rm noise})}

    where :math:`\tilde{b}^2` is the bias term that accounts for the contribution of Poisson
    noise to the cross spectrum (see :func:`stingray.fourier.bias_term`), and tilde generally
    indicates noisy measurements of the cross spectrum :math:`C` and the power spectra :math:`P_n`.
    The terms :math:`\tilde{P}_{\rm n, \rm noise}` are the estimates of the contribution of
    Poisson noise to the power spectra.

    The bias term depends on the intrinsic coherence itself, so it can be calculated iteratively
    following the procedure from sec. 5 of Ingram 2019, MNRAS 489, 3927, which adjusts the bias
    term until convergence is reached. This is done if ``adjust_bias`` is set to True. It is
    typically a very small correction.

    For errors, assumes **high powers, high coherence**. See eq. 8 of the paper.
    Powers below 3 sigma above the noise level (P_noise / sqrt(MW)) are considered too low,
    and the coherence will be set to NaN.

    TODO: implement a more general treatment of the uncertainty.

    Parameters
    ----------
    cross_power : complex `np.array`
        Cross spectrum
    power1 : float `np.array`
        Sub-band periodogram. Must have the same shape as `cross_power`.
    power2 : float `np.array`
        Reference-band periodogram. Must have the same shape as `cross_power`.
    power1_noise : float `np.array`
        Poisson noise level of the sub-band periodogram. Can have a single value
        or the same shape as `cross_power`.
    power2_noise : float `np.array`
        Poisson noise level of the reference-band periodogram. Can have a single value
        or the same shape as `cross_power`.
    n_ave : int or float `np.array`
        number of intervals that have been averaged to obtain the input spectra. Can have
        a single value or the same shape as `cross_power`.

    Other Parameters
    ----------------
    atol: float, default 0.01
        The absolute tolerance for the convergence of the iterative procedure to adjust
        the bias term. Only relevant if ``adjust_bias`` is True.

    Returns
    -------
    coherence : float `np.array`
        The intrinsic coherence values at all frequencies. It is 0 if the numerator
        of the coherence calculation is negative, and NaN if the powers are too low compared
        to the noise level.
    uncertainty : float `np.array`
        The uncertainty on the intrinsic coherence, calculated according to Vaughan & Nowak
        1997, ApJ 474, L43, eq. 8.
    flags : bool `np.array`
        Whether the bias term adjustment procedure converged (False) or not (True). Only
        relevant if ``adjust_bias`` is True.

    """

    coherence = np.empty(cross_power.shape, dtype=np.float64)
    uncertainty = np.empty(cross_power.shape, dtype=np.float64)
    no_conversion_flags = []

    for i in range(cross_power.size):
        if np.size(power1_noise) == np.size(cross_power):
            p1_n = float(power1_noise[i])
        else:
            p1_n = float(power1_noise[0])

        if np.size(power2_noise) == np.size(cross_power):
            p2_n = float(power2_noise[i])
        else:
            p2_n = float(power2_noise[0])

        if np.size(n_ave) == np.size(cross_power):
            n_a = int(n_ave[i])
        else:
            n_a = int(n_ave[0])

        if adjust_bias:
            coherence[i], uncertainty[i], flag = _intrinsic_coherence_with_adjusted_bias(
                cross_power[i], power1[i], power2[i], p1_n, p2_n, n_a, atol=atol
            )
            if flag:
                no_conversion_flags.append(i)

        else:
            coherence[i], uncertainty[i] = _intrinsic_coherence(
                cross_power[i], power1[i], power2[i], p1_n, p2_n, n_a
            )

    return coherence, uncertainty, no_conversion_flags


def intrinsic_coherence(
    cross_power,
    power1,
    power2,
    power1_noise,
    power2_noise,
    n_ave,
    return_uncertainty=False,
    adjust_bias=False,
    atol=0.01,
):
    r"""
    Intrinsic coherence estimations from cross and power spectra.

    Vaughan & Nowak 1997, ApJ 474, L43

    .. math::

        \tilde{\gamma}^2(f) = \frac{|\langle \tilde{C}(f) \rangle|^2 - \tilde{b}^2}
        {(\langle \tilde{P}_1(f) \rangle - \tilde{P}_{1, \rm noise})
        (\langle \tilde{P}_2(f) \rangle - \tilde{P}_{2, \rm noise})}

    where :math:`\tilde{b}^2` is the bias term that accounts for the contribution of Poisson
    noise to the cross spectrum (see :func:`stingray.fourier.bias_term`), and tilde generally
    indicates noisy measurements of the cross spectrum :math:`C` and the power spectra :math:`P_n`.
    The terms :math:`\tilde{P}_{\rm n, \rm noise}` are the estimates of the contribution of
    Poisson noise to the power spectra.

    The bias term depends on the intrinsic coherence itself, so it can be calculated iteratively
    following the procedure from sec. 5 of Ingram 2019, MNRAS 489, 3927, which adjusts the bias
    term until convergence is reached. This is done if ``adjust_bias`` is set to True. It is
    typically a very small correction.

    For errors, assumes **high powers, high coherence**. See eq. 8 of the paper.
    Powers below 3 sigma above the noise level (P_noise / sqrt(MW)) are considered too low,
    and the coherence will be set to NaN.

    TODO: implement a more general treatment of the uncertainty.

    Parameters
    ----------
    cross_power : complex `np.array`
        Cross spectrum
    power1 : float `np.array`
        Sub-band periodogram. Must have the same shape as `cross_power`.
    power2 : float `np.array`
        Reference-band periodogram. Must have the same shape as `cross_power`.
    power1_noise : float or float `np.array` of the same shape as `cross_power`
        Poisson noise level of the sub-band periodogram
    power2_noise : float or float `np.array` of the same shape as `cross_power`
        Poisson noise level of the reference-band periodogram
    n_ave : int or float or int `np.array` of the same shape as `cross_power`
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    return_uncertainty : bool, default False
        Whether to return the uncertainty on the coherence, calculated according to
        Vaughan & Nowak 1997, ApJ 474, L43, eq. 8.
    atol: float, default 0.01
        The absolute tolerance for the convergence of the iterative procedure to adjust
        the bias term.Only relevant if ``adjust_bias`` is True.

    Returns
    -------
    coherence : float `np.array` or tuple of two float `np.array`
        The intrinsic coherence values at all frequencies. It is 0 if the numerator
        of the coherence calculation is negative, and NaN if the powers are too low compared
        to the noise level.
    uncertainty : float `np.array`, optional
        The uncertainty on the intrinsic coherence, calculated according to Vaughan & Nowak
        1997, ApJ 474, L43, eq. 8.

    """
    single_power = False
    if not isinstance(cross_power, np.ndarray):
        cross_power = np.asanyarray([cross_power])
        power1 = np.asanyarray([power1])
        power2 = np.asanyarray([power2])
        single_power = True

    if not isinstance(power1_noise, Iterable):
        power1_noise = np.asanyarray([power1_noise])
    if not isinstance(power2_noise, Iterable):
        power2_noise = np.asanyarray([power2_noise])
    if not isinstance(n_ave, Iterable):
        n_ave = np.asanyarray([n_ave])

    coherence, uncertainty, flagged = _intrinsic_coherence_array(
        cross_power,
        power1,
        power2,
        power1_noise,
        power2_noise,
        n_ave,
        adjust_bias=adjust_bias,
        atol=atol,
    )

    if len(flagged) > 0:
        warnings.warn(
            "The iterative procedure to adjust the bias term did not converge after 40 "
            "iterations. Consider rebinning the spectra to increase the signal-to-noise "
            "ratio, or to use a more permissive tolerance (``atol`` parameter)."
        )
    if np.any(np.isnan(coherence)):
        warnings.warn(
            "NaN values detected in intrinsic_coherence calculation. This happens when the powers "
            "are too close to the noise level, and the coherence is not well defined. Consider "
            "rebinning the spectra to increase the signal-to-noise ratio."
        )
    if np.any(np.isclose(coherence, 0)):
        warnings.warn(
            "Zero values detected in intrinsic_coherence calculation. This usually happens"
            " when the bias term is larger than the cross spectrum, and the coherence is not"
            " well defined. "
        )

    if single_power:
        coherence = coherence[0]
        uncertainty = uncertainty[0]

    if return_uncertainty:
        return coherence, uncertainty
    return coherence


def estimate_intrinsic_coherence(cross_power, power1, power2, power1_noise, power2_noise, n_ave):
    """
    Estimate intrinsic coherence

    Use the iterative procedure from sec. 5 of

    Ingram 2019, MNRAS 489, 392

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int or float
        number of intervals that have been averaged to obtain the input spectra

    Returns
    -------
    coherence : float `np.array`
        The estimated intrinsic coherence, at all frequencies.
    """
    warnings.warn(
        "estimate_intrinsic_coherence is deprecated. Use intrinsic_coherence with adjust_bias=True instead.",
        DeprecationWarning,
    )
    new_coherence = intrinsic_coherence(
        cross_power,
        power1,
        power2,
        power1_noise,
        power2_noise,
        n_ave,
        return_uncertainty=False,
        adjust_bias=True,
    )
    return new_coherence


def get_rms_from_rms_norm_periodogram(power_sqrms, poisson_noise_sqrms, df, M, low_M_buffer_size=4):
    r"""Calculate integrated rms spectrum (frac or abs).

    If M=1, it starts by rebinning the powers slightly in order to get a slightly
    better approximation for the error bars.

    Parameters
    ----------
    power_sqrms: array-like
        Powers, in units of fractional rms ($(rms/mean)^2 Hz{-1}$)
    poisson_noise_sqrms: float
        Poisson noise level, in units of fractional rms ($(rms/mean)^2 Hz{-1}$
    df: float or ``np.array``, same dimension of ``power_sqrms``
        The frequency resolution of each power
    M: int or ``np.array``, same dimension of ``power_sqrms``
        The number of powers averaged to obtain each value of power.

    Other Parameters
    ----------------
    low_M_buffer_size : int, default 4
        If M=1, the powers are rebinned to have a minimum of ``low_M_buffer_size`` powers
        in each bin. This is done to get a better estimate of the error bars.
    """
    from stingray.utils import rebin_data

    m_is_iterable = isinstance(M, Iterable)
    df_is_iterable = isinstance(df, Iterable)

    # if M is an iterable but all values are the same, let's simplify
    if m_is_iterable and len(list(set(M))) == 1:
        M = M[0]
        m_is_iterable = False
    # the same with df
    if df_is_iterable and len(list(set(df))) == 1:
        df = df[0]
        df_is_iterable = False

    low_M_values = M < 30

    if np.any(M < 30):
        quantity = "Some"
        if not m_is_iterable or np.count_nonzero(low_M_values) == M.size:
            quantity = "All"
        warnings.warn(
            f"{quantity} power spectral bins have M<30. The error bars on the rms might be wrong. "
            "In some cases one might try to increase the number of segments, for example by "
            "reducing the segment size, in order to obtain at least 30 segments."
        )
    # But they cannot be of different kind. There would be something wrong with the data
    if m_is_iterable != df_is_iterable:
        raise ValueError("M and df must be either both constant, or none of them.")

    # If powers are not rebinned and M=1, we rebin them slightly. The error on the power is tricky,
    # because powers follow a non-central chi squared distribution. If we combine powers with
    # very different underlying signal level, the error bars will be completely wrong. But
    # nearby powers have a higher chance of having similar values, and so, by combining them,
    # we have a higher chance of obtaining sensible quasi-Gaussian error bars, easier to
    # propagate through standard quadrature summation
    if not m_is_iterable and M == 1 and power_sqrms.size > low_M_buffer_size:
        _, local_power_sqrms, _, local_M = rebin_data(
            np.arange(power_sqrms.size) * df,
            power_sqrms,
            df * low_M_buffer_size,
            yerr=None,
            method="average",
            dx=df,
        )
        local_df = df * local_M
        total_local_powers = local_M * local_power_sqrms.size
        total_power_err = np.sqrt(np.sum(local_power_sqrms**2 * local_df**2 / local_M))
        total_power_err *= power_sqrms.size / total_local_powers
    else:
        total_power_err = np.sqrt(np.sum(power_sqrms**2 * df**2 / M))

    powers_sub = power_sqrms - poisson_noise_sqrms

    total_power_sub = np.sum(powers_sub * df)

    if total_power_sub < 0:
        # By the definition, it makes no sense to define an error bar here.
        warnings.warn("Poisson-subtracted power is below 0")
        return 0.0, 0.0

    rms = np.sqrt(total_power_sub)

    high_snr_err = 0.5 / rms * total_power_err

    return rms, high_snr_err


def get_rms_from_unnorm_periodogram(
    unnorm_powers,
    nphots_per_segment,
    df,
    M=1,
    poisson_noise_unnorm=None,
    segment_size=None,
    kind="frac",
):
    """Calculate the fractional rms amplitude from unnormalized powers.

    We assume the powers come from an unnormalized Bartlett periodogram.
    If so, the Poisson noise level is ``nphots_per_segment``, but the user
    can specify otherwise (e.g. if the Poisson noise level is altered by dead time).
    The ``segment_size`` and ``nphots_per_segment`` parameters refer to the length
    and averaged counts of each segment of data used for the Bartlett periodogram.

    Parameters
    ----------
    unnorm_powers : np.ndarray
        The unnormalized power spectrum
    nphots_per_segment : float
        The averaged number of photons per segment of the data used for the Bartlett periodogram
    df : float
        The frequency resolution of the periodogram

    Other parameters
    ----------------
    poisson_noise_unnorm : float
        The unnormalized Poisson noise level
    segment_size : float
        The size of the segment, in seconds
    M : int
        The number of segments averaged to obtain the periodogram
    kind : str
        One of "frac" or "abs"
    """
    if segment_size is None:
        segment_size = 1 / np.min(df)

    if poisson_noise_unnorm is None:
        poisson_noise_unnorm = nphots_per_segment

    meanrate = nphots_per_segment / segment_size

    def to_leahy(powers):
        return powers * 2.0 / nphots_per_segment

    def to_frac(powers):
        return to_leahy(powers) / meanrate

    def to_abs(powers):
        return to_leahy(powers) * meanrate

    if kind.startswith("frac"):
        to_norm = to_frac
    elif kind.startswith("abs"):
        to_norm = to_abs
    else:
        raise ValueError("Only 'frac' or 'abs' rms are supported.")

    poisson = to_norm(poisson_noise_unnorm)
    powers = to_norm(unnorm_powers)

    return get_rms_from_rms_norm_periodogram(powers, poisson, df, M)


def rms_calculation(
    unnorm_powers,
    min_freq,
    max_freq,
    nphots,
    T,
    M_freqs,
    K_freqs,
    freq_bins,
    poisson_noise_unnorm,
    deadtime=0.0,
):
    """
    Compute the fractional rms amplitude in the given power or cross spectrum

    NOTE: all array quantities are already in the correct energy range

    Parameters
    ----------
    unnorm_powers: array of float
        unnormalised power or cross spectrum, the array has already been
        filtered for the given frequency range

    min_freq: float
        The lower frequency bound for the calculation (from the freq grid).

    max_freq: float
        The upper frequency bound for the calculation (from the freq grid).

    nphots: float
        Number of photons for the full power or cross spectrum

    T: float
        Time length of the light curve

    M_freq: scalar or array of float
        If scalar, it is the number of segments in the AveragedCrossspectrum
        If array, it is the number of segments times the rebinning sample
        in the given frequency range.

    K_freq: scalar or array of float
        If scalar, the power or cross spectrum is not rebinned (K_freq = 1)
        If array,  the power or cross spectrum is rebinned and it is the
        rebinned sample in the given frequency range.

    freq_bins: integer
        if the cross or power spectrum is rebinned freq_bins = 1,
        if it NOT rebinned freq_bins is the number of frequency bins
        in the given frequency range.

    poisson_noise_unnorm : float
        This is the Poisson noise level unnormalised.

    Other parameters
    ----------------
    deadtime: float
        Deadtime of the instrument

    Returns
    -------
    rms: float
        The fractional rms amplitude contained between ``min_freq`` and
        ``max_freq``.

    rms_err: float
        The error on the fractional rms amplitude.

    """
    warnings.warn(
        "The rms_calculation function is deprecated. Use get_rms_from_unnorm_periodogram instead.",
        DeprecationWarning,
    )
    rms_norm_powers = unnorm_powers * 2 * T / nphots**2
    rms_poisson_noise = poisson_noise_unnorm * 2 * T / nphots**2

    df = 1 / T * K_freqs

    rms, rms_err = get_rms_from_rms_norm_periodogram(
        rms_norm_powers, rms_poisson_noise, df, M_freqs
    )
    return rms, rms_err


def error_on_averaged_cross_spectrum(
    cross_power, seg_power, ref_power, n_ave, seg_power_noise, ref_power_noise, common_ref=False
):
    """
    Error on cross spectral quantities.

    `Ingram 2019<https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.3927I/abstract>`__ details
    the derivation of the corrected formulas when the subject band is included in the reference
    band (and photons are presents in both bands, so this does _not_ involve overlapping bands from
    different detectors).
    The keyword argument ``common_ref`` indicates if the reference band also contains the photons
    of the subject band.

    Note: this is only valid for a very large number of averaged powers.
    Beware if n_ave < 50 or so.

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    seg_power : float `np.array`
        sub-band periodogram
    ref_power : float `np.array`
        reference-band periodogram
    seg_power_noise : float
        Poisson noise level of the sub-band periodogram
    ref_power_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int or float
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    common_ref : bool, default False
        Are data in the sub-band also included in the reference band?

    Returns
    -------
    dRe : float `np.array`
        Error on the real part of the cross spectrum
    dIm : float `np.array`
        Error on the imaginary part of the cross spectrum
    dphi : float `np.array`
        Error on the angle (or phase lag)
    dG : float `np.array`
        Error on the modulus of the cross spectrum

    """
    if (not isinstance(n_ave, Iterable) and n_ave < 30) or (
        isinstance(n_ave, Iterable) and np.any(n_ave) < 30
    ):
        warnings.warn(
            "n_ave is below 30. Please note that the error bars "
            "on the quantities derived from the cross spectrum "
            "are only reliable for a large number of averaged "
            "powers."
        )
    two_n_ave = 2 * n_ave
    if common_ref:
        Gsq = (cross_power * np.conj(cross_power)).real
        bsq = bias_term(seg_power, ref_power, seg_power_noise, ref_power_noise, n_ave)
        frac = (Gsq - bsq) / (ref_power - ref_power_noise)
        power_over_2n = ref_power / two_n_ave

        # Eq. 18
        dRe = dIm = dG = np.sqrt(power_over_2n * (seg_power - frac))
        # Eq. 19
        dphi = np.sqrt(
            power_over_2n * (seg_power / (Gsq - bsq) - 1 / (ref_power - ref_power_noise))
        )

    else:
        PrPs = ref_power * seg_power
        dRe = np.sqrt((PrPs + cross_power.real**2 - cross_power.imag**2) / two_n_ave)
        dIm = np.sqrt((PrPs - cross_power.real**2 + cross_power.imag**2) / two_n_ave)

        gsq = raw_coherence(
            cross_power,
            seg_power,
            ref_power,
            seg_power_noise,
            ref_power_noise,
            n_ave,
            return_uncertainty=False,
        )
        dphi = np.sqrt((1 - gsq) / (2 * gsq * n_ave))
        dG = np.sqrt(PrPs / n_ave)

    return dRe, dIm, dphi, dG


def cross_to_covariance(cross_power, ref_power, ref_power_noise, delta_nu):
    """
    Convert a cross spectrum into a covariance spectrum.

    Covariance:
    Wilkinson & Uttley 2009, MNRAS, 397, 666

    Complex covariance:
    Mastroserio et al. 2018, MNRAS, 475, 4027

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    ref_power : float `np.array`
        reference-band periodogram
    ref_power_noise : float
        Poisson noise level of the reference-band periodogram
    delta_nu : float or `np.array`
        spectral resolution. Can be a float, or an array if the spectral
        resolution is not constant throughout the periodograms

    Returns
    -------
    covariance: complex `np.array`
        The cross spectrum, normalized as a covariance.

    """
    return cross_power * np.sqrt(delta_nu / (ref_power - ref_power_noise))


def _which_segment_idx_fun(binned=False, dt=None):
    """
    Select which segment index function from ``gti.py`` to use.

    If ``binned`` is ``False``, call the unbinned function.

    If ``binned`` is not ``True``, call the binned function.

    Note that in the binned function ``dt`` is an optional parameter.
    We pass it if the user specifies it.
    """
    # Make function interface equal (fluxes gets ignored)
    if not binned:
        fun = generate_indices_of_segment_boundaries_unbinned

        # Define a new function, make sure that, by default, the sort check
        # is disabled.
        def fun(*args, **kwargs):
            check_sorted = kwargs.pop("check_sorted", False)
            return generate_indices_of_segment_boundaries_unbinned(
                *args, check_sorted=check_sorted, **kwargs
            )

    else:
        # Define a new function, so that we can pass the correct dt as an
        # argument.
        def fun(*args, **kwargs):
            return generate_indices_of_segment_boundaries_binned(*args, dt=dt, **kwargs)

    return fun


def get_average_ctrate(times, gti, segment_size, counts=None):
    """
    Calculate the average count rate during the observation.

    This function finds the same segments that the averaged periodogram
    functions (``avg_cs_from_iterables``, ``avg_pds_from_iterables`` etc) will
    use, and returns the mean count rate.
    If ``counts`` is ``None``, the input times are interpreted as events.
    Otherwise, the number of events is taken from ``counts``

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments

    Other parameters
    ----------------
    counts : float `np.array`, default None
        Array of counts per bin

    Returns
    -------
    ctrate : float
        The average count rate in the segments that are used for the analysis.

    Examples
    --------
    >>> times = np.sort(np.random.uniform(0, 1000, 1000))
    >>> gti = np.asanyarray([[0, 1000]])
    >>> counts, _ = np.histogram(times, bins=np.linspace(0, 1000, 11))
    >>> bin_times = np.arange(50, 1000, 100)
    >>> assert get_average_ctrate(bin_times, gti, 1000, counts=counts) == 1.0
    >>> assert get_average_ctrate(times, gti, 1000) == 1.0
    """
    n_ph = 0
    n_intvs = 0
    binned = counts is not None
    func = _which_segment_idx_fun(binned)

    for _, _, idx0, idx1 in func(times, gti, segment_size):
        if not binned:
            n_ph += idx1 - idx0
        else:
            n_ph += np.sum(counts[idx0:idx1])
        n_intvs += 1

    return n_ph / (n_intvs * segment_size)


def get_flux_iterable_from_segments(
    times, gti, segment_size, n_bin=None, dt=None, fluxes=None, errors=None
):
    """
    Get fluxes from different segments of the observation.

    If ``fluxes`` is ``None``, the input times are interpreted as events, and
    they are split into many binned series of length ``segment_size`` with
    ``n_bin`` bins.

    If ``fluxes`` is an array, the number of events corresponding to each time
    bin is taken from ``fluxes``

    Therefore, at least one of either ``n_bin`` and ``fluxes`` needs to be
    specified.

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments. If ``None``, the full light curve is used.

    Other parameters
    ----------------
    n_bin : int, default None
        Number of bins to divide the ``segment_size`` in
    fluxes : float `np.array`, default None
        Array of fluxes.
    errors : float `np.array`, default None
        Array of error bars corresponding to the flux values above.
    dt : float, default None
        Time resolution of the light curve used to produce periodograms

    Yields
    ------
    flux : `np.array`
        Array of fluxes
    err : `np.array`
        (optional) if ``errors`` is None, an array of errors in the segment

    """
    if fluxes is None and n_bin is None:
        raise ValueError(
            "At least one between fluxes (if light curve) and " "n_bin (if events) has to be set"
        )

    binned = fluxes is not None
    cast_kind = float
    if dt is None and binned:
        dt = np.median(np.diff(times[:100]))

    if binned:
        fluxes = np.asanyarray(fluxes)
        if np.iscomplexobj(fluxes):
            cast_kind = complex

    if segment_size is None:
        segment_size = gti[-1, 1] - gti[0, 0]

        def fun(times, gti, segment_size):
            return [[gti[0, 0], gti[-1, 1], 0, times.size]]

    else:
        fun = _which_segment_idx_fun(binned, dt)

    for s, e, idx0, idx1 in fun(times, gti, segment_size):
        if idx1 - idx0 < 2:
            yield None
            continue
        if not binned:
            # astype here serves to avoid integer rounding issues in Windows,
            # where long is a 32-bit integer.
            cts = histogram(
                (times[idx0:idx1] - s).astype(float), bins=n_bin, range=[0, segment_size]
            ).astype(float)
            cts = np.array(cts)
        else:
            cts = fluxes[idx0:idx1].astype(cast_kind)
            if errors is not None:
                cts = cts, errors[idx0:idx1].astype(cast_kind)

        yield cts


@njit()
def _safe_array_slice_indices(input_size, input_center_idx, nbins):
    """Calculate the indices needed to extract a n-bin slice of an array, centered at an index.

    Let us say we have an array of size ``input_size`` and we want to extract a slice of
    ``nbins`` centered at index ``input_center_idx``. We should be robust when the slice goes
    beyond the edges of the input array, possibly leaving missing values in the output array.
    This function calculates the indices needed to extract the slice from the input array, and
    the indices in the output array that will be filled.

    In the most common case, the slice is entirely contained within the input array, so that the
    output slice will just be ``[0:nbins]`` and the input slice
    ``[input_center_idx - nbins // 2: input_center_idx - nbins // 2 + nbins]``.

    Parameters
    ----------
    input_size : int
        Input array size
    center_idx : int
        Index of the center of the slice
    nbins : int
        Number of bins to extract

    Returns
    -------
    input_slice : list
        Indices to extract the slice from the input array
    output_slice : list
        Indices to fill the output array

    Examples
    --------
    >>> _safe_array_slice_indices(input_size=10, input_center_idx=5, nbins=3)
    ([4, 7], [0, 3])

    If the slice goes beyond the right edge: the output slice will only cover
    the first two bins of the output array, and up to the end of the input array.
    >>> _safe_array_slice_indices(input_size=6, input_center_idx=5, nbins=3)
    ([4, 6], [0, 2])

    """

    minbin = input_center_idx - nbins // 2
    maxbin = minbin + nbins

    if minbin < 0:
        output_slice = [-minbin, min(nbins, input_size - minbin)]
        input_slice = [0, minbin + nbins]
    elif maxbin > input_size:
        output_slice = [0, nbins - (maxbin - input_size)]
        input_slice = [minbin, input_size]
    else:
        output_slice = [0, nbins]
        input_slice = [minbin, maxbin]

    return input_slice, output_slice


@njit()
def extract_pds_slice_around_freq(freqs, powers, f0, nbins=100):
    """Extract a slice of PDS around a given frequency.

    This function extracts a slice of the power spectrum around a given frequency.
    The slice has a length of ``nbins``. If the slice goes beyond the edges of the
    power spectrum, the missing values are filled with NaNs.

    Parameters
    ----------
    freqs : np.array
        Array of frequencies, the same for all powers
    powers : np.array
        Array of powers
    f0 : float
        Central frequency

    Other parameters
    ----------------
    nbins : int, default 100
        Number of bins to extract

    Examples
    --------
    >>> freqs = np.arange(1, 100) * 0.1
    >>> powers = 10 / freqs
    >>> f0 = 0.3
    >>> p = extract_pds_slice_around_freq(freqs, powers, f0)
    >>> assert np.isnan(p[0])
    >>> assert not np.any(np.isnan(p[48:]))
    """
    powers = np.asarray(powers)
    chunk = np.zeros(nbins) + np.nan
    # fchunk = np.zeros(nbins)

    start_f_idx = np.searchsorted(freqs, f0)

    input_slice, output_slice = _safe_array_slice_indices(powers.size, start_f_idx, nbins)
    chunk[output_slice[0] : output_slice[1]] = powers[input_slice[0] : input_slice[1]]
    return chunk


@njit()
def _shift_and_average_core(input_array_list, weight_list, center_indices, nbins):
    """Core function to shift_and_add, JIT-compiled for your convenience.

    Parameters
    ----------
    input_array_list : list of np.array
        List of input arrays
    weight_list : list of float
        List of weights for each input array
    center_indices : list of int
        Central indices of the slice of each input array to be summed
    nbins : int
        Number of bins to extract around the central index of each input array

    Returns
    -------
    output_array : np.array
        Average of the input arrays, weighted by the weights
    sum_of_weights : np.array
        Sum of the weights at each output bin
    """
    input_size = input_array_list[0].size
    output_array = np.zeros(nbins)
    sum_of_weights = np.zeros(nbins)
    for idx, array, weight in zip(center_indices, input_array_list, weight_list):
        input_slice, output_slice = _safe_array_slice_indices(input_size, idx, nbins)

        for i in range(input_slice[1] - input_slice[0]):
            output_array[output_slice[0] + i] += array[input_slice[0] + i] * weight

            sum_of_weights[output_slice[0] + i] += weight

    output_array = output_array / sum_of_weights

    return output_array, sum_of_weights


def shift_and_add(freqs, power_list, f0_list, nbins=100, rebin=None, df=None, M=None):
    """Add a list of power spectra, centered on different frequencies.

    This is the basic operation for the shift-and-add operation used to track
    kHz QPOs in X-ray binaries (e.g. Méndez et al. 1998, ApJ, 494, 65).

    Parameters
    ----------
    freqs : np.array
        Array of frequencies, the same for all powers. Must be sorted and on a uniform
        grid.
    power_list : list of np.array
        List of power spectra. Each power spectrum must have the same length
        as the frequency array.
    f0_list : list of float
        List of central frequencies

    Other parameters
    ----------------
    nbins : int, default 100
        Number of bins to extract
    rebin : int, default None
        Rebin the final spectrum by this factor. At the moment, the rebinning
        is linear.
    df : float, default None
        Frequency resolution of the power spectra. If not given, it is calculated
        from the input frequencies.
    M : int or list of int, default None
        Number of segments used to calculate each power spectrum. If a list is
        given, it must have the same length as the power list.

    Returns
    -------
    f : np.array
        Array of output frequencies
    p : np.array
        Array of output powers
    n : np.array
        Number of contributing power spectra at each frequency

    Examples
    --------
    >>> power_list = [[2, 5, 2, 2, 2], [1, 1, 5, 1, 1], [3, 3, 3, 5, 3]]
    >>> freqs = np.arange(5) * 0.1
    >>> f0_list = [0.1, 0.2, 0.3, 0.4]
    >>> f, p, n = shift_and_add(freqs, power_list, f0_list, nbins=5)
    >>> assert np.array_equal(n, [2, 3, 3, 3, 2])
    >>> assert np.array_equal(p, [2. , 2. , 5. , 2. , 1.5])
    >>> assert np.allclose(f, [0.05, 0.15, 0.25, 0.35, 0.45])
    """

    # Check if the input list of power contains numpy arrays
    if not hasattr(power_list[0], "size"):
        power_list = np.asarray(power_list)
    # input_size = np.size(power_list[0])
    freqs = np.asarray(freqs)

    # mid_idx = np.searchsorted(freqs, np.mean(f0_list))
    if M is None:
        M = 1
    if not isinstance(M, Iterable):
        M = np.ones(len(power_list)) * M

    center_f_indices = np.searchsorted(freqs, f0_list)

    final_powers, count = _shift_and_average_core(power_list, M, center_f_indices, nbins)

    if df is None:
        df = freqs[1] - freqs[0]

    final_freqs = np.arange(-nbins // 2, nbins // 2 + 1)[:nbins] * df
    final_freqs = final_freqs - (final_freqs[0] + final_freqs[-1]) / 2 + np.mean(f0_list)

    if rebin is not None:
        _, count, _, _ = rebin_data(final_freqs, count, rebin * df)
        final_freqs, final_powers, _, _ = rebin_data(final_freqs, final_powers, rebin * df)
        final_powers = final_powers / rebin

    return final_freqs, final_powers, count


def avg_pds_from_iterable(
    flux_iterable, dt, norm="frac", use_common_mean=True, silent=False, return_subcs=False
):
    """
    Calculate the average periodogram from an iterable of light curves

    Parameters
    ----------
    flux_iterable : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal-length series of count measurements,
        or of tuples (fluxes, errors). They must all be of the same length.
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or on
        the full light curve. This gives different results (Alston+2013).
        Here we assume the mean is calculated on the full light curve, but
        the user can set ``use_common_mean`` to False to calculate it on a
        per-segment basis.
    silent : bool, default False
        Silence the progress bars
    return_subcs : bool, default False
        Return all sub spectra from each light curve chunk

    Returns
    -------
    results : :class:`astropy.table.Table`
        Table containing the following columns:
        freq : `np.array`
            The periodogram frequencies
        power : `np.array`
            The normalized periodogram powers
        unnorm_power : `np.array`
            The unnormalized periodogram powers

        And a number of other useful diagnostics in the metadata, including
        all attributes needed to allocate Powerspectrum objects, such as all
        the input arguments of this function (``dt``, ``segment_size``), and,
        e.g.
        n : int
            the number of bins in the light curves used in each segment
        m : int
            the number of averaged periodograms
        mean : float
            the mean flux
    """
    local_show_progress = show_progress
    if silent:

        def local_show_progress(a):
            return a

    # Initialize stuff
    cross = unnorm_cross = None
    n_ave = 0

    sum_of_photons = 0
    common_variance = None
    if return_subcs:
        subcs = []
        unnorm_subcs = []

    for flux in local_show_progress(flux_iterable):
        if flux is None or np.all(flux == 0):
            continue

        # If the iterable returns the uncertainty, use it to calculate the
        # variance.
        variance = None
        if isinstance(flux, tuple):
            flux, err = flux
            variance = np.mean(err) ** 2

        # Calculate the FFT
        n_bin = flux.size
        ft = fft(flux)

        # This will only be used by the Leahy normalization, so only if
        # the input light curve is in units of counts/bin
        n_ph = flux.sum()
        unnorm_power = (ft * ft.conj()).real

        # Accumulate the sum of means and variances, to get the final mean and
        # variance the end
        sum_of_photons += n_ph

        if variance is not None:
            common_variance = sum_if_not_none_or_initialize(common_variance, variance)

        # In the first loop, define the frequency and the freq. interval > 0
        if cross is None:
            fgt0 = positive_fft_bins(n_bin)
            freq = fftfreq(n_bin, dt)[fgt0]

        # No need for the negative frequencies
        unnorm_power = unnorm_power[fgt0]

        # If the user wants to normalize using the mean of the total
        # lightcurve, normalize it here
        cs_seg = copy.deepcopy(unnorm_power)
        if not use_common_mean:
            mean = n_ph / n_bin

            cs_seg = normalize_periodograms(
                unnorm_power,
                dt,
                n_bin,
                mean,
                n_ph=n_ph,
                norm=norm,
                variance=variance,
            )

        # Accumulate the total sum cross spectrum
        cross = sum_if_not_none_or_initialize(cross, cs_seg)
        unnorm_cross = sum_if_not_none_or_initialize(unnorm_cross, unnorm_power)

        if return_subcs:
            subcs.append(cs_seg)
            unnorm_subcs.append(unnorm_power)

        n_ave += 1

    # If there were no good intervals, return None
    if cross is None:
        return None

    # Calculate the mean number of photons per chunk
    n_ph = sum_of_photons / n_ave
    # Calculate the mean number of photons per bin
    common_mean = n_ph / n_bin

    if common_variance is not None:
        # Note: the variances we summed were means, not sums.
        # Hence M, not M*n_bin
        common_variance /= n_ave

    # Transform a sum into the average
    unnorm_cross = unnorm_cross / n_ave
    cross = cross / n_ave

    # Final normalization (If not done already!)
    if use_common_mean:
        factor = normalize_periodograms(
            1, dt, n_bin, common_mean, n_ph=n_ph, norm=norm, variance=common_variance
        )
        cross = unnorm_cross * factor
        if return_subcs:
            for i in range(len(subcs)):
                subcs[i] *= factor

    results = Table()
    results["freq"] = freq
    results["power"] = cross
    results["unnorm_power"] = unnorm_cross
    results.meta.update(
        {
            "n": n_bin,
            "m": n_ave,
            "dt": dt,
            "norm": norm,
            "df": 1 / (dt * n_bin),
            "nphots": n_ph,
            "mean": common_mean,
            "variance": common_variance,
            "segment_size": dt * n_bin,
        }
    )
    if return_subcs:
        results.meta["subcs"] = subcs
        results.meta["unnorm_subcs"] = unnorm_subcs

    return results


def avg_cs_from_iterables_quick(flux_iterable1, flux_iterable2, dt, norm="frac"):
    """Like `avg_cs_from_iterables`, with default options that make it quick.

    Assumes that:

    * the flux iterables return counts/bin, no other units
    * the mean is calculated over the whole light curve, and normalization
      is done at the end
    * no auxiliary PDSs are returned
    * only positive frequencies are returned
    * the spectrum is complex, no real parts or absolutes
    * no progress bars

    Parameters
    ----------
    flux_iterable1 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal-length series of count measurements, or
        of tuples (fluxes, errors). They must all be of the same length.
    flux_iterable2 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Same as ``flux_iterable1``, for the reference channel
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram

    Returns
    -------
    results : :class:`astropy.table.Table`
        Table containing the following columns:
        freq : `np.array`
            The periodogram frequencies
        power : `np.array`
            The normalized periodogram powers
        unnorm_power : `np.array`
            The unnormalized periodogram powers

        And a number of other useful diagnostics in the metadata, including
        all attributes needed to allocate Crossspectrum objects, such as all
        the input arguments of this function (``dt``, ``segment_size``), and,
        e.g.
        n : int
            the number of bins in the light curves used in each segment
        m : int
            the number of averaged periodograms
        mean : float
            the mean flux (geometrical average of the mean fluxes in the two
            channels)

    """
    # Initialize stuff
    unnorm_cross = None
    n_ave = 0

    sum_of_photons1 = sum_of_photons2 = 0

    for flux1, flux2 in zip(flux_iterable1, flux_iterable2):
        if flux1 is None or flux2 is None or np.all(flux1 == 0) or np.all(flux2 == 0):
            continue

        n_bin = flux1.size

        # Calculate the sum of each light curve, to calculate the mean
        n_ph1 = flux1.sum()
        n_ph2 = flux2.sum()

        # At the first loop, we define the frequency array and the range of
        # positive frequency bins (after the first loop, cross will not be
        # None anymore)
        if unnorm_cross is None:
            freq = fftfreq(n_bin, dt)
            fgt0 = positive_fft_bins(n_bin)

        # Calculate the FFTs
        ft1 = fft(flux1)
        ft2 = fft(flux2)

        # Calculate the unnormalized cross spectrum
        unnorm_power = ft1.conj() * ft2

        # Accumulate the sum to calculate the total mean of the lc
        sum_of_photons1 += n_ph1
        sum_of_photons2 += n_ph2

        # Take only positive frequencies
        unnorm_power = unnorm_power[fgt0]

        # Initialize or accumulate final averaged spectrum
        unnorm_cross = sum_if_not_none_or_initialize(unnorm_cross, unnorm_power)

        n_ave += 1

    # If no valid intervals were found, return only `None`s
    if unnorm_cross is None:
        return None

    # Calculate the mean number of photons per chunk
    n_ph1 = sum_of_photons1 / n_ave
    n_ph2 = sum_of_photons2 / n_ave
    n_ph = np.sqrt(n_ph1 * n_ph2)
    # Calculate the mean number of photons per bin
    common_mean1 = n_ph1 / n_bin
    common_mean2 = n_ph2 / n_bin
    common_mean = n_ph / n_bin

    # Transform the sums into averages
    unnorm_cross /= n_ave

    # Finally, normalize the cross spectrum (only if not already done on an
    # interval-to-interval basis)
    cross = normalize_periodograms(
        unnorm_cross,
        dt,
        n_bin,
        common_mean,
        n_ph=n_ph,
        norm=norm,
        variance=None,
        power_type="all",
    )

    # No negative frequencies
    freq = freq[fgt0]

    results = Table()
    results["freq"] = freq
    results["power"] = cross
    results["unnorm_power"] = unnorm_cross
    results.meta.update(
        {
            "n": n_bin,
            "m": n_ave,
            "dt": dt,
            "norm": norm,
            "df": 1 / (dt * n_bin),
            "nphots": n_ph,
            "nphots1": n_ph1,
            "nphots2": n_ph2,
            "variance": None,
            "mean": common_mean,
            "mean1": common_mean1,
            "mean2": common_mean2,
            "power_type": "all",
            "fullspec": False,
            "segment_size": dt * n_bin,
        }
    )

    return results


def avg_cs_from_iterables(
    flux_iterable1,
    flux_iterable2,
    dt,
    norm="frac",
    use_common_mean=True,
    silent=False,
    fullspec=False,
    power_type="all",
    return_auxil=False,
    return_subcs=False,
):
    """Calculate the average cross spectrum from an iterable of light curves

    Parameters
    ----------
    flux_iterable1 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal-length series of count measurements, or
        of tuples (fluxes, errors). They must all be of the same length.
    flux_iterable2 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Same as ``flux_iterable1``, for the reference channel
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or on
        the full light curve. This gives different results (Alston+2013).
        Here we assume the mean is calculated on the full light curve, but
        the user can set ``use_common_mean`` to False to calculate it on a
        per-segment basis.
    fullspec : bool, default False
        Return the full periodogram, including negative frequencies
    silent : bool, default False
        Silence the progress bars
    power_type : str, default 'all'
        If 'all', give complex powers. If 'abs', the absolute value; if 'real',
        the real part
    return_auxil : bool, default False
        Return the auxiliary unnormalized PDSs from the two separate channels
    return_subcs : bool, default False
        Return all sub spectra from each light curve chunk

    Returns
    -------
    results : :class:`astropy.table.Table`
        Table containing the following columns:
        freq : `np.array`
            The frequencies.
        power : `np.array`
            The normalized cross spectral powers.
        unnorm_power : `np.array`
            The unnormalized cross spectral power.
        unnorm_pds1 : `np.array`
            The unnormalized auxiliary PDS from channel 1. Only returned if
            ``return_auxil`` is ``True``.
        unnorm_pds2 : `np.array`
            The unnormalized auxiliary PDS from channel 2. Only returned if
            ``return_auxil`` is ``True``.

        And a number of other useful diagnostics in the metadata, including
        all attributes needed to allocate Crossspectrum objects, such as all
        the input arguments of this function (``dt``, ``segment_size``), and,
        e.g.
        n : int
            The number of bins in the light curves used in each segment.
        m : int
            The number of averaged periodograms.
        mean : float
            The mean flux (geometrical average of the mean fluxes in the two
            channels).
    """

    local_show_progress = show_progress
    if silent:

        def local_show_progress(a):
            return a

    # Initialize stuff
    cross = unnorm_cross = unnorm_pds1 = unnorm_pds2 = pds1 = pds2 = None
    n_ave = 0

    sum_of_photons1 = sum_of_photons2 = 0
    common_variance1 = common_variance2 = common_variance = None

    if return_subcs:
        subcs = []
        unnorm_subcs = []

    for flux1, flux2 in local_show_progress(zip(flux_iterable1, flux_iterable2)):
        if flux1 is None or flux2 is None or np.all(flux1 == 0) or np.all(flux2 == 0):
            continue

        # Does the flux iterable return the uncertainty?
        # If so, define the variances
        variance1 = variance2 = None
        if isinstance(flux1, tuple):
            flux1, err1 = flux1
            variance1 = np.mean(err1) ** 2
        if isinstance(flux2, tuple):
            flux2, err2 = flux2
            variance2 = np.mean(err2) ** 2

        # Only use the variance if both flux iterables define it.
        if variance1 is None or variance2 is None:
            variance1 = variance2 = None
        else:
            common_variance1 = sum_if_not_none_or_initialize(common_variance1, variance1)
            common_variance2 = sum_if_not_none_or_initialize(common_variance2, variance2)

        n_bin = flux1.size

        # At the first loop, we define the frequency array and the range of
        # positive frequency bins (after the first loop, cross will not be
        # None anymore)
        if cross is None:
            freq = fftfreq(n_bin, dt)
            fgt0 = positive_fft_bins(n_bin)

        # Calculate the FFTs
        ft1 = fft(flux1)
        ft2 = fft(flux2)

        # Calculate the sum of each light curve chunk, to calculate the mean
        n_ph1 = flux1.sum()
        n_ph2 = flux2.sum()
        n_ph = np.sqrt(n_ph1 * n_ph2)

        # Calculate the unnormalized cross spectrum
        unnorm_power = ft1.conj() * ft2
        unnorm_pd1 = unnorm_pd2 = 0

        # If requested, calculate the auxiliary PDSs
        if return_auxil:
            unnorm_pd1 = (ft1 * ft1.conj()).real
            unnorm_pd2 = (ft2 * ft2.conj()).real

        # Accumulate the sum to calculate the total mean of the lc
        sum_of_photons1 += n_ph1
        sum_of_photons2 += n_ph2

        # Take only positive frequencies unless the user wants the full
        # spectrum
        if not fullspec:
            unnorm_power = unnorm_power[fgt0]
            if return_auxil:
                unnorm_pd1 = unnorm_pd1[fgt0]
                unnorm_pd2 = unnorm_pd2[fgt0]

        cs_seg = copy.deepcopy(unnorm_power)
        p1_seg = unnorm_pd1
        p2_seg = unnorm_pd2

        # If normalization has to be done interval by interval, do it here.
        if not use_common_mean:
            mean1 = n_ph1 / n_bin
            mean2 = n_ph2 / n_bin
            mean = n_ph / n_bin
            variance = None

            if variance1 is not None:
                variance = np.sqrt(variance1 * variance2)

            cs_seg = normalize_periodograms(
                unnorm_power,
                dt,
                n_bin,
                mean,
                n_ph=n_ph,
                norm=norm,
                power_type=power_type,
                variance=variance,
            )
            if return_auxil:
                p1_seg = normalize_periodograms(
                    unnorm_pd1,
                    dt,
                    n_bin,
                    mean1,
                    n_ph=n_ph1,
                    norm=norm,
                    power_type=power_type,
                    variance=variance1,
                )
                p2_seg = normalize_periodograms(
                    unnorm_pd2,
                    dt,
                    n_bin,
                    mean2,
                    n_ph=n_ph2,
                    norm=norm,
                    power_type=power_type,
                    variance=variance2,
                )

        if return_subcs:
            subcs.append(cs_seg)
            unnorm_subcs.append(unnorm_power)

        # Initialize or accumulate final averaged spectra
        cross = sum_if_not_none_or_initialize(cross, cs_seg)
        unnorm_cross = sum_if_not_none_or_initialize(unnorm_cross, unnorm_power)

        if return_auxil:
            unnorm_pds1 = sum_if_not_none_or_initialize(unnorm_pds1, unnorm_pd1)
            unnorm_pds2 = sum_if_not_none_or_initialize(unnorm_pds2, unnorm_pd2)
            pds1 = sum_if_not_none_or_initialize(pds1, p1_seg)
            pds2 = sum_if_not_none_or_initialize(pds2, p2_seg)

        n_ave += 1

    # If no valid intervals were found, return only `None`s
    if cross is None:
        return None

    # Calculate the mean number of photons per chunk
    n_ph1 = sum_of_photons1 / n_ave
    n_ph2 = sum_of_photons2 / n_ave
    n_ph = np.sqrt(n_ph1 * n_ph2)

    # Calculate the common mean number of photons per bin
    common_mean1 = n_ph1 / n_bin
    common_mean2 = n_ph2 / n_bin
    common_mean = n_ph / n_bin

    if common_variance1 is not None:
        # Note: the variances we summed were means, not sums. Hence M, not M*N
        common_variance1 /= n_ave
        common_variance2 /= n_ave
        common_variance = np.sqrt(common_variance1 * common_variance2)

    # Transform the sums into averages
    cross /= n_ave
    unnorm_cross /= n_ave
    if return_auxil:
        unnorm_pds1 /= n_ave
        unnorm_pds2 /= n_ave

    # Finally, normalize the cross spectrum (only if not already done on an
    # interval-to-interval basis)
    if use_common_mean:
        cross = normalize_periodograms(
            unnorm_cross,
            dt,
            n_bin,
            common_mean,
            n_ph=n_ph,
            norm=norm,
            variance=common_variance,
            power_type=power_type,
        )

        if return_subcs:
            for i in range(len(subcs)):
                subcs[i] = normalize_periodograms(
                    unnorm_subcs[i],
                    dt,
                    n_bin,
                    common_mean,
                    n_ph=n_ph,
                    norm=norm,
                    variance=common_variance,
                    power_type=power_type,
                )

        if return_auxil:
            pds1 = normalize_periodograms(
                unnorm_pds1,
                dt,
                n_bin,
                common_mean1,
                n_ph=n_ph1,
                norm=norm,
                variance=common_variance1,
                power_type=power_type,
            )
            pds2 = normalize_periodograms(
                unnorm_pds2,
                dt,
                n_bin,
                common_mean2,
                n_ph=n_ph2,
                norm=norm,
                variance=common_variance2,
                power_type=power_type,
            )
    # If the user does not want negative frequencies, don't give them
    if not fullspec:
        freq = freq[fgt0]

    results = Table()
    results["freq"] = freq
    results["power"] = cross
    results["unnorm_power"] = unnorm_cross
    results.meta.update(
        {
            "n": n_bin,
            "m": n_ave,
            "dt": dt,
            "norm": norm,
            "df": 1 / (dt * n_bin),
            "segment_size": dt * n_bin,
            "nphots": n_ph,
            "nphots1": n_ph1,
            "nphots2": n_ph2,
            "countrate1": common_mean1 / dt,
            "countrate2": common_mean2 / dt,
            "mean": common_mean,
            "mean1": common_mean1,
            "mean2": common_mean2,
            "power_type": power_type,
            "fullspec": fullspec,
            "variance": common_variance,
            "variance1": common_variance1,
            "variance2": common_variance2,
        }
    )

    if return_auxil:
        results["pds1"] = pds1
        results["pds2"] = pds2
        results["unnorm_pds1"] = unnorm_pds1
        results["unnorm_pds2"] = unnorm_pds2

    if return_subcs:
        results.meta["subcs"] = np.array(subcs)
        results.meta["unnorm_subcs"] = np.array(unnorm_subcs)

    return results


def avg_pds_from_events(*args, **kwargs):
    warnings.warn(
        "avg_pds_from_events is deprecated, use avg_cs_from_timeseries instead", DeprecationWarning
    )

    return avg_pds_from_timeseries(*args, **kwargs)


def avg_pds_from_timeseries(
    times,
    gti,
    segment_size,
    dt,
    norm="frac",
    use_common_mean=True,
    silent=False,
    fluxes=None,
    errors=None,
    return_subcs=False,
):
    """
    Calculate the average periodogram from a list of event times or a light
    curve.

    If the input is a light curve, the time array needs to be uniformly sampled
    inside GTIs (it can have gaps outside), and the fluxes need to be passed
    through the ``fluxes`` array.
    Otherwise, times are interpreted as photon arrival times.

    Parameters
    ----------
    times : float `np.array`
        Array of times.
    gti : [[gti00, gti01], [gti10, gti11], ...]
        Good time intervals.
    segment_size : float
        Length of segments. If ``None``, the full light curve is used.
    dt : float
        Time resolution of the light curves used to produce periodograms.

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or on
        the full light curve. This gives different results (Alston+2013).
        Here we assume the mean is calculated on the full light curve, but
        the user can set ``use_common_mean`` to False to calculate it on a
        per-segment basis.
    silent : bool, default False
        Silence the progress bars
    fluxes : float `np.array`, default None
        Array of counts per bin or fluxes
    errors : float `np.array`, default None
        Array of errors on the fluxes above
    return_subcs : bool, default False
        Return all sub spectra from each light curve chunk

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    power : `np.array`
        The normalized periodogram powers
    n_bin : int
        the number of bins in the light curves used in each segment
    n_ave : int or float
        the number of averaged periodograms
    mean : float
        the mean flux
    """
    binned = fluxes is not None
    if segment_size is not None:
        segment_size, n_bin = fix_segment_size_to_integer_samples(segment_size, dt)
    elif binned and segment_size is None:
        n_bin = fluxes.size
    else:
        _, n_bin = fix_segment_size_to_integer_samples(gti.max() - gti.min(), dt)

    flux_iterable = get_flux_iterable_from_segments(
        times, gti, segment_size, n_bin, dt=dt, fluxes=fluxes, errors=errors
    )
    cross = avg_pds_from_iterable(
        flux_iterable,
        dt,
        norm=norm,
        use_common_mean=use_common_mean,
        silent=silent,
        return_subcs=return_subcs,
    )
    if cross is not None:
        cross.meta["gti"] = gti
    return cross


def avg_cs_from_events(*args, **kwargs):
    warnings.warn(
        "avg_cs_from_events is deprecated, use avg_cs_from_timeseries instead", DeprecationWarning
    )
    return avg_cs_from_timeseries(*args, **kwargs)


def avg_cs_from_timeseries(
    times1,
    times2,
    gti,
    segment_size,
    dt,
    norm="frac",
    use_common_mean=True,
    fullspec=False,
    silent=False,
    power_type="all",
    fluxes1=None,
    fluxes2=None,
    errors1=None,
    errors2=None,
    return_auxil=False,
    return_subcs=False,
):
    """
    Calculate the average cross spectrum from a list of event times or a light
    curve.

    If the input is a light curve, the time arrays need to be uniformly sampled
    inside GTIs (they can have gaps outside), and the fluxes need to be passed
    through the ``fluxes1`` and ``fluxes2`` arrays.
    Otherwise, times are interpreted as photon arrival times

    Parameters
    ----------
    times1 : float `np.array`
        Array of times in the sub-band
    times2 : float `np.array`
        Array of times in the reference band
    gti : [[gti00, gti01], [gti10, gti11], ...]
        common good time intervals
    segment_size : float
        length of segments. If ``None``, the full light curve is used.
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or on
        the full light curve. This gives different results (Alston+2013).
        Here we assume the mean is calculated on the full light curve, but
        the user can set ``use_common_mean`` to False to calculate it on a
        per-segment basis.
    fullspec : bool, default False
        Return the full periodogram, including negative frequencies
    silent : bool, default False
        Silence the progress bars
    power_type : str, default 'all'
        If 'all', give complex powers. If 'abs', the absolute value; if 'real',
        the real part
    fluxes1 : float `np.array`, default None
        Array of fluxes or counts per bin for channel 1
    fluxes2 : float `np.array`, default None
        Array of fluxes or counts per bin for channel 2
    errors1 : float `np.array`, default None
        Array of errors on the fluxes on channel 1
    errors2 : float `np.array`, default None
        Array of errors on the fluxes on channel 2
    return_subcs : bool, default False
        Return all sub spectra from each light curve chunk

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    pds : `np.array`
        The normalized periodogram powers
    n_bin : int
        the number of bins in the light curves used in each segment
    n_ave : int or float
        the number of averaged periodograms
    """

    binned = fluxes1 is not None and fluxes2 is not None

    if segment_size is not None:
        segment_size, n_bin = fix_segment_size_to_integer_samples(segment_size, dt)
    elif binned and segment_size is None:
        n_bin = fluxes1.size
    else:
        _, n_bin = fix_segment_size_to_integer_samples(gti.max() - gti.min(), dt)

    flux_iterable1 = get_flux_iterable_from_segments(
        times1, gti, segment_size, n_bin=n_bin, dt=dt, fluxes=fluxes1, errors=errors1
    )
    flux_iterable2 = get_flux_iterable_from_segments(
        times2, gti, segment_size, n_bin=n_bin, dt=dt, fluxes=fluxes2, errors=errors2
    )

    is_events = np.all([val is None for val in (fluxes1, fluxes2, errors1, errors2)])

    if (
        is_events
        and silent
        and use_common_mean
        and power_type == "all"
        and not fullspec
        and not return_auxil
        and not return_subcs
    ):
        results = avg_cs_from_iterables_quick(flux_iterable1, flux_iterable2, dt, norm=norm)

    else:
        results = avg_cs_from_iterables(
            flux_iterable1,
            flux_iterable2,
            dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=silent,
            fullspec=fullspec,
            power_type=power_type,
            return_auxil=return_auxil,
            return_subcs=return_subcs,
        )

    if results is not None:
        results.meta["gti"] = gti
    return results


def lsft_fast(
    y: npt.ArrayLike,
    t: npt.ArrayLike,
    freqs: npt.ArrayLike,
    sign: Optional[int] = 1,
    oversampling: Optional[int] = 5,
) -> npt.NDArray:
    """
    Calculates the Lomb-Scargle Fourier transform of a light curve.
    Only considers non-negative frequencies.
    Subtracts mean from data as it is required for the working of the algorithm.

    Adapted from original Matlab code by J. D. Scargle, using Astropy's ``trig_sum`` for speed.

    Parameters
    ----------
    y : a :class:`numpy.array` of floats
        Observations to be transformed.

    t : :class:`numpy.array` of floats
        Times of the observations

    freqs : :class:`numpy.array`
        An array of frequencies at which the transform is sampled.

    sign : int, optional, default: 1
        The sign of the fourier transform. 1 implies positive sign and -1 implies negative sign.

    fullspec : bool, optional, default: False
        Return LSFT values for full frequency array (True) or just positive frequencies (False).

    oversampling : float, optional, default: 5
        Interpolation Oversampling Factor

    Returns
    -------
    ft_res : numpy.ndarray
        An array of Fourier transformed data.
    """
    y_ = copy.deepcopy(y) - np.mean(y)
    freqs = freqs[freqs >= 0]
    # Constants initialization
    sum_xx = np.sum(y_)
    num_xt = len(y_)
    num_ww = len(freqs)

    # Arrays initialization
    ft_real = ft_imag = np.zeros((num_ww))
    f0, df, N = freqs[0], freqs[1] - freqs[0], len(freqs)

    # Sum (y_i * cos(wt - wtau))
    Sh, Ch = trig_sum(t, y_, df, N, f0, oversampling=oversampling)

    # Angular Frequency
    w = freqs * 2 * np.pi

    # Summation of cos(2wt) and sin(2wt)
    csum, ssum = trig_sum(
        t, np.ones_like(y_) / len(y_), df, N, f0, freq_factor=2, oversampling=oversampling
    )

    wtau = 0.5 * np.arctan2(ssum, csum)

    S2, C2 = trig_sum(
        t,
        np.ones_like(y_),
        df,
        N,
        f0,
        freq_factor=2,
        oversampling=oversampling,
    )

    const = np.sqrt(0.5) * np.sqrt(num_ww)

    coswtau = np.cos(wtau)
    sinwtau = np.sin(wtau)

    sumr = Ch * coswtau + Sh * sinwtau
    sumi = Sh * coswtau - Ch * sinwtau

    cos2wtau = np.cos(2 * wtau)
    sin2wtau = np.sin(2 * wtau)

    scos2 = 0.5 * (N + C2 * cos2wtau + S2 * sin2wtau)
    ssin2 = 0.5 * (N - C2 * cos2wtau - S2 * sin2wtau)

    ft_real = const * sumr / np.sqrt(scos2)
    ft_imag = const * np.sign(sign) * sumi / np.sqrt(ssin2)

    ft_real[freqs == 0] = sum_xx / np.sqrt(num_xt)
    ft_imag[freqs == 0] = 0

    phase = wtau - w * t[0]

    ft_res = np.complex128(ft_real + (1j * ft_imag)) * np.exp(-1j * phase)

    return ft_res


def lsft_slow(
    y: npt.ArrayLike,
    t: npt.ArrayLike,
    freqs: npt.ArrayLike,
    sign: Optional[int] = 1,
) -> npt.NDArray:
    """
    Calculates the Lomb-Scargle Fourier transform of a light curve.
    Only considers non-negative frequencies.
    Subtracts mean from data as it is required for the working of the algorithm.

    Adapted from original Matlab code by J. D. Scargle.

    Parameters
    ----------
    y : a `:class:numpy.array` of floats
        Observations to be transformed.

    t : `:class:numpy.array` of floats
        Times of the observations

    freqs : numpy.ndarray
        An array of frequencies at which the transform is sampled.

    sign : int, optional, default: 1
        The sign of the fourier transform. 1 implies positive sign and -1 implies negative sign.

    fullspec : bool, optional, default: False
        Return LSFT values for full frequency array (True) or just positive frequencies (False).

    Returns
    -------
    ft_res : numpy.ndarray
        An array of Fourier transformed data.
    """
    y_ = y - np.mean(y)
    freqs = np.asanyarray(freqs[np.asanyarray(freqs) >= 0])

    ft_real = np.zeros_like(freqs)
    ft_imag = np.zeros_like(freqs)
    ft_res = np.zeros_like(freqs, dtype=np.complex128)

    num_y = y_.shape[0]
    num_freqs = freqs.shape[0]
    sum_y = np.sum(y_)
    const1 = np.sqrt(0.5 * num_y)
    const2 = const1 * np.sign(sign)
    ft_real = ft_imag = np.zeros(num_freqs)
    ft_res = np.zeros(num_freqs, dtype=np.complex128)

    for i in range(num_freqs):
        wrun = freqs[i] * 2 * np.pi
        if wrun == 0:
            ft_real = sum_y / np.sqrt(num_y)
            ft_imag = 0
            phase_this = 0
        else:
            # Calculation of \omega \tau (II.5) --
            csum = np.sum(np.cos(2.0 * wrun * t))
            ssum = np.sum(np.sin(2.0 * wrun * t))

            watan = np.arctan2(ssum, csum)
            wtau = 0.5 * watan
            # --
            # In the following, instead of t'_n we are using \omega t'_n = \omega t - \omega\tau

            # Terms of kind X_n * cos or sin (II.1) --
            sumr = np.sum(y_ * np.cos(wrun * t - wtau))
            sumi = np.sum(y_ * np.sin(wrun * t - wtau))
            # --

            # A and B before the square root and inversion in (II.3) --
            scos2 = np.sum(np.power(np.cos(wrun * t - wtau), 2))
            ssin2 = np.sum(np.power(np.sin(wrun * t - wtau), 2))

            # const2 is const1 times the sign.
            # It's the F0 in II.2 without the phase factor
            # The sign decides whether we are calculating the direct or inverse transform
            ft_real = const1 * sumr / np.sqrt(scos2)
            ft_imag = const2 * sumi / np.sqrt(ssin2)

            phase_this = wtau - wrun * t[0]

        ft_res[i] = np.complex128(ft_real + (1j * ft_imag)) * np.exp(-1j * phase_this)
    return ft_res


def impose_symmetry_lsft(
    ft_res: npt.ArrayLike,
    sum_y: float,
    len_y: int,
    freqs: npt.ArrayLike,
) -> npt.ArrayLike:
    """
    Impose symmetry on the input fourier transform.

    Parameters
    ----------
    ft_res : np.array
        The Fourier transform of the signal.
    sum_y : float
        The sum of the values of the signal.
    len_y : int
        The length of the signal.
    freqs : np.array
        An array of frequencies at which the transform is sampled.

    Returns
    -------
    lsft_res : np.array
        The Fourier transform of the signal with symmetry imposed.
    freqs_new : np.array
        The new frequencies
    """
    ft_res_new = np.concatenate([np.conjugate(np.flip(ft_res)), [0.0], ft_res])
    freqs_new = np.concatenate([np.flip(-freqs), [0.0], freqs])
    return ft_res_new, freqs_new

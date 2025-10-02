"""
Basic pulsar-related functions and statistics.
"""

import functools
import math
from collections.abc import Iterable
import warnings
from scipy.optimize import minimize, basinhopping
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from .fftfit import fftfit as taylor_fftfit
from ..utils import simon, jit
from . import HAS_PINT, get_model, toa


__all__ = [
    "pulse_phase",
    "phase_exposure",
    "fold_events",
    "ef_profile_stat",
    "pdm_profile_stat",
    "z_n",
    "fftfit",
    "get_TOA",
    "z_n_binned_events",
    "z_n_gauss",
    "z_n_events",
    "htest",
    "p_to_f",
    "z_n_binned_events_all",
    "z_n_gauss_all",
    "z_n_events_all",
    "get_orbital_correction_from_ephemeris_file",
]


def _default_value_if_no_key(dictionary, key, default):
    try:
        return dictionary[key]
    except:
        return default


def p_to_f(*period_derivatives):
    """Convert periods into frequencies, and vice versa.

    For now, limited to third derivative. Raises when a
    fourth derivative is passed.

    Parameters
    ----------
    p, pdot, pddot, ... : floats
        period derivatives, starting from zeroth and in
        increasing order

    Examples
    --------
    >>> assert p_to_f() == []
    >>> assert np.allclose(p_to_f(1), [1])
    >>> assert np.allclose(p_to_f(1, 2), [1, -2])
    >>> assert np.allclose(p_to_f(1, 2, 3), [1, -2, 5])
    >>> assert np.allclose(p_to_f(1, 2, 3, 4), [1, -2, 5, -16])
    """
    nder = len(period_derivatives)
    if nder == 0:
        return []
    fder = np.zeros_like(period_derivatives)
    p = period_derivatives[0]
    fder[0] = 1 / p

    if nder > 1:
        pd = period_derivatives[1]
        fder[1] = -1 / p**2 * pd

    if nder > 2:
        pdd = period_derivatives[2]
        fder[2] = 2 / p**3 * pd**2 - 1 / p**2 * pdd

    if nder > 3:
        pddd = period_derivatives[3]
        fder[3] = -6 / p**4 * pd**3 + 6 / p**3 * pd * pdd - 1 / p**2 * pddd
    if nder > 4:
        warnings.warn("Derivatives above third are not supported")

    return fder


def pulse_phase(times, *frequency_derivatives, **opts):
    """
    Calculate pulse phase from the frequency and its derivatives.

    Parameters
    ----------
    times : array of floats
        The times at which the phase is calculated
    *frequency_derivatives: floats
        List of derivatives in increasing order, starting from zero.

    Other Parameters
    ----------------
    ph0 : float
        The starting phase
    to_1 : bool, default True
        Only return the fractional part of the phase, normalized from 0 to 1

    Returns
    -------
    phases : array of floats
        The absolute pulse phase

    """

    ph0 = opts.pop("ph0", 0)
    to_1 = opts.pop("to_1", True)
    ph = ph0

    for i_f, f in enumerate(frequency_derivatives):
        ph += 1 / math.factorial(i_f + 1) * times ** (i_f + 1) * f

    if to_1:
        ph -= np.floor(ph)
    return ph


def phase_exposure(start_time, stop_time, period, nbin=16, gti=None):
    """Calculate the exposure on each phase of a pulse profile.

    Parameters
    ----------
    start_time, stop_time : float
        Starting and stopping time (or phase if ``period==1``)
    period : float
        The pulse period (if 1, equivalent to phases)

    Other parameters
    ----------------
    nbin : int, optional, default 16
        The number of bins in the profile
    gti : [[gti00, gti01], [gti10, gti11], ...], optional, default None
        Good Time Intervals

    Returns
    -------
    expo : array of floats
        The normalized exposure of each bin in the pulse profile (1 is the
        highest exposure, 0 the lowest)
    """
    if gti is None:
        gti = np.array([[start_time, stop_time]])

    # Use precise floating points -------------
    start_time = np.longdouble(start_time)
    stop_time = np.longdouble(stop_time)
    period = np.longdouble(period)
    gti = np.array(gti, dtype=np.longdouble)
    # -----------------------------------------

    expo = np.zeros(nbin)
    phs = np.linspace(0, 1, nbin + 1)
    phs = np.array(list(zip(phs[0:-1], phs[1:])))

    # Discard gtis outside [start, stop]
    good = np.logical_and(gti[:, 0] < stop_time, gti[:, 1] > start_time)
    gti = gti[good]

    for g in gti:
        g0 = g[0]
        g1 = g[1]
        if g0 < start_time:
            # If the start of the fold is inside a gti, start from there
            g0 = start_time
        if g1 > stop_time:
            # If the end of the fold is inside a gti, end there
            g1 = stop_time
        length = g1 - g0
        # How many periods inside this length?
        nraw = length / period
        # How many integer periods?
        nper = nraw.astype(int)

        # First raw exposure: the number of periods
        expo += nper / nbin

        # FRACTIONAL PART =================
        # What remains is additional exposure for part of the profile.
        start_phase = np.fmod(g0 / period, 1)
        end_phase = nraw - nper + start_phase

        limits = [[start_phase, end_phase]]
        # start_phase is always < 1. end_phase not always. In this case...
        if end_phase > 1:
            limits = [[0, end_phase - 1], [start_phase, 1]]

        for l in limits:
            l0 = l[0]
            l1 = l[1]
            # Discards bins untouched by these limits
            goodbins = np.logical_and(phs[:, 0] <= l1, phs[:, 1] >= l0)
            idxs = np.arange(len(phs), dtype=int)[goodbins]
            for i in idxs:
                start = np.max([phs[i, 0], l0])
                stop = np.min([phs[i, 1], l1])
                w = stop - start
                expo[i] += w

    return expo / np.max(expo)


def fold_events(
    times,
    *frequency_derivatives,
    mode="ef",
    nbin=16,
    weights=1,
    gti=None,
    ref_time=0,
    expocorr=False,
    variances=None,
):
    """Epoch folding with exposure correction.

    By default, the keyword `times` accepts a list of
    unbinned photon arrival times. If the input data is
    a (binned) light curve, then `times` will contain the
    time stamps of the observation, and `weights` should
    be set to the corresponding fluxes or counts.

    Parameters
    ----------
    times : array of floats
        Photon arrival times, or, if `weights` is set,
        time stamps of a light curve.

    f, fdot, fddot... : float
        The frequency and any number of derivatives.

    Other Parameters
    ----------------
    nbin : int, optional, default 16
        The number of bins in the pulse profile

    weights : float or array of floats, optional, default 1
        The weights of the data. It can either be specified as a single value
        for all points, or an array with the same length as ``time``

    gti : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...], optional, default None
        Good time intervals

    ref_time : float, optional, default 0
        Reference time for the timing solution

    expocorr : bool, optional, default False
        Correct each bin for exposure (use when the period of the pulsar is
        comparable to that of GTIs)

    mode : str, ["ef", "pdm"], optional, default "ef"
        Whether to calculate the epoch folding or phase dispersion
        minimization folded profile. For "ef", it calculates the (weighted)
        sum of the data points in each phase bin, for "pdm", the variance
        in each phase bin

    Returns
    -------
    phase_bins : array of floats
        The phases corresponding to the pulse profile

    profile : array of floats
        The pulse profile

    profile_err : array of floats
        The uncertainties on the pulse profile
    """

    # If no key is passed, *or gti is None*, defaults to the
    # initial and final event
    if gti is None:
        gti = [[times[0], times[-1]]]
    # Be safe if gtis are a list
    gti = np.asanyarray(gti)

    if not isinstance(weights, Iterable):
        weights *= np.ones(len(times))

    gti = gti - ref_time
    times = times - ref_time
    # This dt has not the same meaning as in the Lightcurve case.
    # it's just to define stop_time as a meaningful value after
    # the last event.
    dt = np.abs(times[1] - times[0])
    start_time = times[0]
    stop_time = times[-1] + dt

    phases = pulse_phase(times, *frequency_derivatives, to_1=True)
    gti_phases = pulse_phase(gti, *frequency_derivatives, to_1=False)
    start_phase, stop_phase = pulse_phase(
        np.array([start_time, stop_time]), *frequency_derivatives, to_1=False
    )

    if mode == "ef":
        raw_profile, bins = np.histogram(phases, bins=np.linspace(0, 1, nbin + 1), weights=weights)
        # TODO: this is wrong. Need to extend this to non-1 weights
        raw_profile_err = np.sqrt(raw_profile)

        if expocorr:
            expo_norm = phase_exposure(start_phase, stop_phase, 1, nbin, gti=gti_phases)
            simon("For exposure != 1, the uncertainty might be incorrect")
        else:
            expo_norm = 1

        raw_profile = raw_profile / expo_norm
        raw_profile_err = raw_profile_err / expo_norm

    elif mode == "pdm":
        # Convert the weights to a numpy array just in case it is a list
        weights = np.asarray(weights)
        if np.allclose(weights, 1.0):
            raise ValueError(
                "Can only calculate PDM for binned light curves!"
                + "`weights` attribute must be set to fluxes!"
            )

        # For weighted pdm, the measured uncertainties (variances) for each event are required
        # set the event_weights as 1/variances or else 1s if variances=None
        event_weights = (
            (1.0 / np.asarray(variances)) if variances is not None else np.ones_like(weights)
        )

        bins = np.linspace(0, 1, nbin + 1)
        # phases are already fractional values in [0, 1]
        # casting float to int removes the fractional part
        bin_indices = (phases * nbin).astype(int)
        # to wrap: for phase = 1 then bin_index = nbin; clipping changes it to bin_index = 0
        bin_indices = np.clip(bin_indices, 0, nbin - 1)

        # for efficiency, the sum-of-squared deviations for each bin is split
        # sosdev = sum_in_bin( (values_in_bin - mean_in_bin)**2 )
        # mean_in_bin = sum_in_bin(values_in_bin) / n_in_bin
        # so need n_in_bin, sum_in_bin, sos_in_bin (sum-of-squares)
        # now compute the stats for each bin using bincount
        # minlength=nbin prevents array shortening when a bin index is not represented in bin_indices
        n_in_bin = np.bincount(bin_indices, weights=event_weights, minlength=nbin)
        sum_in_bin = np.bincount(bin_indices, weights=weights * event_weights, minlength=nbin)
        sos_in_bin = np.bincount(bin_indices, weights=weights**2 * event_weights, minlength=nbin)
        # put it together to make sum-of-squared deviations: sosdev; avoid division by zero
        raw_profile = sos_in_bin - sum_in_bin**2 / np.maximum(n_in_bin, 1)

        # dummy array for the error, which we don't have for the variance
        raw_profile_err = np.zeros_like(raw_profile)

    else:
        raise ValueError(
            "mode can only be `ef` for Epoch Folding or "
            + "`pdm` for Phase Dispersion Minimization!"
        )

    return bins[:-1] + np.diff(bins) / 2, raw_profile, raw_profile_err


def ef_profile_stat(profile, err=None):
    """Calculate the epoch folding statistics \'a la Leahy et al. (1983).

    Parameters
    ----------
    profile : array
        The pulse profile

    Other Parameters
    ----------------
    err : float or array
        The uncertainties on the pulse profile

    Returns
    -------
    stat : float
        The epoch folding statistics
    """
    mean = np.mean(profile)
    if err is None:
        err = np.sqrt(mean)
    return np.sum((profile - mean) ** 2 / err**2)


def pdm_profile_stat(profile, sum_dev2, nsample):
    """Calculate the phase dispersion minimization
    statistic following Stellingwerf (1978)

    Parameters
    ----------
    profile : array
        The PDM pulse profile (variance as a function
        of phase)

    sum_dev2 : float
        The sum-of-squared-deviations of the sample,
        which uses the grand (whole sample together) mean

    nsample : int
        The number of time bins in the initial time
        series.

    Returns
    -------
    stat : float
        The epoch folding statistics
    """
    # Get the mean-of-squared-deviations from sum-of-squared-deviations by
    # considering the number of degrees of freedom:
    # nsample - nbin for the in_bin sum_dev2 because there are nbin means used
    # nsample - 1 for the grand sum_dev2 because there is 1 grand mean used
    mean_dev2_in_bin = np.sum(profile) / (nsample - len(profile))
    mean_dev2_grand = sum_dev2 / (nsample - 1)
    stat = mean_dev2_in_bin / mean_dev2_grand
    return stat


@functools.lru_cache(maxsize=128)
def _cached_sin_harmonics(nbin, z_n_n):
    """Cached sine values corresponding to each of the nbin bins.

    Parameters
    ----------
    nbin : int
        Number of bins
    z_n_n : int
        The number of harmonics (n) in the Z^2_n search
    """
    dph = 1.0 / nbin
    twopiphases = np.pi * 2 * np.arange(dph / 2, 1, dph)
    cached_sin = np.zeros(z_n_n * nbin)
    for i in range(z_n_n):
        cached_sin[i * nbin : (i + 1) * nbin] = np.sin(twopiphases)
    return cached_sin


@functools.lru_cache(maxsize=128)
def _cached_cos_harmonics(nbin, z_n_n):
    """Cached cosine values corresponding to each of the nbin bins.

    Parameters
    ----------
    nbin : int
        Number of bins
    z_n_n : int
        The number of harmonics (n) in the Z^2_n search
    """
    dph = 1.0 / nbin
    twopiphases = np.pi * 2 * np.arange(dph / 2, 1, dph)
    cached_cos = np.zeros(z_n_n * nbin)
    for i in range(z_n_n):
        cached_cos[i * nbin : (i + 1) * nbin] = np.cos(twopiphases)
    return cached_cos


@jit(nopython=True)
def _z_n_fast_cached_sums_unnorm(prof, ks, cached_sin, cached_cos):
    """Calculate the unnormalized Z^2_k, for (k=1,.. n), of a pulsed profile.

    Parameters
    ----------
    prof : :class:`numpy.array`
        The pulsed profile
    ks : :class:`numpy.array` of int
        The harmonic numbers, from 1 to n
    cached_sin : :class:`numpy.array`
        Cached sine values for each phase bin in the profile
    cached_cos : :class:`numpy.array`
        Cached cosine values for each phase bin in the profile
    """

    all_zs = np.zeros(ks.size)
    N = prof.size

    total_sum = 0
    for k in ks:
        local_z = (
            np.sum(cached_cos[: N * k : k] * prof) ** 2
            + np.sum(cached_sin[: N * k : k] * prof) ** 2
        )
        total_sum += local_z
        all_zs[k - 1] = total_sum

    return all_zs


def z_n_binned_events_all(profile, nmax=20):
    """Z^2_n statistic for multiple harmonics and binned events

    See Bachetti+2021, arXiv:2012.11397

    Parameters
    ----------
    profile : array of floats
        The folded pulse profile (containing the number of
        photons falling in each pulse bin)
    n : int
        Number of harmonics, including the fundamental

    Returns
    -------
    ks : list of ints
        Harmonic numbers, from 1 to nmax (included)
    z2_n : float
        The value of the statistic for all ks
    """
    cached_sin = _cached_sin_harmonics(profile.size, nmax)
    cached_cos = _cached_cos_harmonics(profile.size, nmax)
    ks = np.arange(1, nmax + 1, dtype=int)

    total = np.sum(profile)
    if total == 0:
        return ks, np.zeros(nmax)
    all_zs = _z_n_fast_cached_sums_unnorm(profile, ks, cached_sin, cached_cos)

    return ks, all_zs * 2 / total


def z_n_gauss_all(profile, err, nmax=20):
    """Z^2_n statistic for n harmonics and normally-distributed profiles

    See Bachetti+2021, arXiv:2012.11397

    Parameters
    ----------
    profile : array of floats
        The folded pulse profile
    err : float
        The (assumed constant) uncertainty on the flux in each bin.
    nmax : int
        Maximum number of harmonics, including the fundamental

    Returns
    -------
    ks : list of ints
        Harmonic numbers, from 1 to nmax (included)
    z2_n : list of floats
        The value of the statistic for all ks
    """
    cached_sin = _cached_sin_harmonics(profile.size, nmax)
    cached_cos = _cached_cos_harmonics(profile.size, nmax)
    ks = np.arange(1, nmax + 1, dtype=int)

    all_zs = _z_n_fast_cached_sums_unnorm(profile, ks, cached_sin, cached_cos)

    return ks, all_zs * (2 / profile.size / err**2)


@jit(nopython=True)
def z_n_events_all(phase, nmax=20):
    """Z^2_n statistics, a` la Buccheri+83, A&A, 128, 245, eq. 2.

    Parameters
    ----------
    phase : array of floats
        The phases of the events
    n : int, default 2
        Number of harmonics, including the fundamental

    Returns
    -------
    ks : list of ints
        Harmonic numbers, from 1 to nmax (included)
    z2_n : float
        The Z^2_n statistic for all ks
    """
    all_zs = np.zeros(nmax)
    ks = np.arange(1, nmax + 1)
    nphot = phase.size

    total_sum = 0
    phase = phase * 2 * np.pi

    for k in ks:
        local_z = np.sum(np.cos(k * phase)) ** 2 + np.sum(np.sin(k * phase)) ** 2
        total_sum += local_z
        all_zs[k - 1] = total_sum

    return ks, 2 / nphot * all_zs


def z_n_binned_events(profile, n):
    """Z^2_n statistic for pulse profiles from binned events

    See Bachetti+2021, arXiv:2012.11397

    Parameters
    ----------
    profile : array of floats
        The folded pulse profile (containing the number of
        photons falling in each pulse bin)
    n : int
        Number of harmonics, including the fundamental

    Returns
    -------
    z2_n : float
        The value of the statistic
    """
    _, all_zs = z_n_binned_events_all(profile, nmax=n)
    return all_zs[-1]


def z_n_gauss(profile, err, n):
    """Z^2_n statistic for normally-distributed profiles

    See Bachetti+2021, arXiv:2012.11397

    Parameters
    ----------
    profile : array of floats
        The folded pulse profile
    err : float
        The (assumed constant) uncertainty on the flux in each bin.
    n : int
        Number of harmonics, including the fundamental

    Returns
    -------
    z2_n : float
        The value of the statistic
    """
    _, all_zs = z_n_gauss_all(profile, err, nmax=n)
    return all_zs[-1]


def z_n_events(phase, n):
    """Z^2_n statistics, a` la Buccheri+83, A&A, 128, 245, eq. 2.

    Parameters
    ----------
    phase : array of floats
        The phases of the events
    n : int, default 2
        Number of harmonics, including the fundamental

    Returns
    -------
    z2_n : float
        The Z^2_n statistic
    """
    ks, all_zs = z_n_events_all(phase, nmax=n)
    return all_zs[-1]


def z_n(data, n, datatype="events", err=None, norm=None):
    """Z^2_n statistics, a` la Buccheri+83, A&A, 128, 245, eq. 2.

    If datatype is "binned" or "gauss", uses the formulation from
    Bachetti+2021, ApJ, arxiv:2012.11397

    Parameters
    ----------
    data : array of floats
        Phase values or binned flux values
    n : int, default 2
        Number of harmonics, including the fundamental

    Other Parameters
    ----------------
    datatype : str
        The data type: "events" if phase values between 0 and 1,
        "binned" if folded pulse profile from photons, "gauss" if
        folded pulse profile with normally-distributed fluxes
    err : float
        The uncertainty on the pulse profile fluxes (required for
        datatype="gauss", ignored otherwise)
    norm : float
        For backwards compatibility; if norm is not None, it is
        substituted to ``data``, and data is ignored. This raises
        a DeprecationWarning

    Returns
    -------
    z2_n : float
        The Z^2_n statistics of the events.
    """
    data = np.asanyarray(data)

    if norm is not None:
        warnings.warn(
            "The use of ``z_n(phase, norm=profile)`` is deprecated. Use "
            "``z_n(profile, datatype='binned')`` instead",
            DeprecationWarning,
        )
        if isinstance(norm, Iterable):
            data = norm
            datatype = "binned"
        else:
            datatype = "events"

    if data.size == 0:
        return 0

    if datatype == "binned":
        return z_n_binned_events(data, n)
    elif datatype == "events":
        return z_n_events(data, n)
    elif datatype == "gauss":
        if err is None:
            raise ValueError("If datatype='gauss', you need to specify an uncertainty (err)")
        return z_n_gauss(data, n=n, err=err)

    raise ValueError(f"Unknown datatype requested for Z_n ({datatype})")


def htest(data, nmax=20, datatype="binned", err=None):
    """htest-test statistic, a` la De Jager+89, A&A, 221, 180D, eq. 2.

    If datatype is "binned" or "gauss", uses the formulation from
    Bachetti+2021, ApJ, arxiv:2012.11397

    Parameters
    ----------
    data : array of floats
        Phase values or binned flux values
    nmax : int, default 20
        Maximum of harmonics for Z^2_n

    Other Parameters
    ----------------
    datatype : str
        The datatype of data: "events" if phase values between 0 and 1,
        "binned" if folded pulse profile from photons, "gauss" if
        folded pulse profile with normally-distributed fluxes
    err : float
        The uncertainty on the pulse profile fluxes (required for
        datatype="gauss", ignored otherwise)

    Returns
    -------
    M : int
        The best number of harmonics that describe the signal.
    htest : float
        The htest statistics of the events.
    """
    if datatype == "binned":
        ks, zs = z_n_binned_events_all(data, nmax)
    elif datatype == "events":
        ks, zs = z_n_events_all(data, nmax)
    elif datatype == "gauss":
        if err is None:
            raise ValueError("If datatype='gauss', you need to specify an uncertainty (err)")
        ks, zs = z_n_gauss_all(data, nmax=nmax, err=err)
    else:
        raise ValueError(f"Unknown datatype requested for htest ({datatype})")

    Hs = zs - 4 * ks + 4
    bestidx = np.argmax(Hs)

    return ks[bestidx], Hs[bestidx]


def fftfit_fun(profile, template, amplitude, phase):
    """Function to be minimized for the FFTFIT method."""

    pass


def fftfit(prof, template=None, quick=False, sigma=None, use_bootstrap=False, **fftfit_kwargs):
    """Align a template to a pulse profile.

    Parameters
    ----------
    prof : array
        The pulse profile
    template : array, default None
        The template of the pulse used to perform the TOA calculation. If None,
        a simple sinusoid is used

    Other parameters
    ----------------
    sigma : array
        error on profile bins (currently has no effect)
    use_bootstrap : bool
        Calculate errors using a bootstrap method, with `fftfit_error`
    **fftfit_kwargs : additional arguments for `fftfit_error`

    Returns
    -------
    mean_amp, std_amp : floats
        Mean and standard deviation of the amplitude
    mean_phase, std_phase : floats
        Mean and standard deviation of the phase
    """
    prof = prof - np.mean(prof)

    template = template - np.mean(template)

    return taylor_fftfit(prof, template)


def _plot_TOA_fit(
    profile, template, toa, mod=None, toaerr=None, additional_phase=0.0, show=True, period=1
):
    """Plot diagnostic information on the TOA."""
    from scipy.interpolate import interp1d
    import time

    phases = np.arange(0, 2, 1 / len(profile))
    profile = np.concatenate((profile, profile))
    template = np.concatenate((template, template))
    if mod is None:
        mod = interp1d(phases, template, fill_value="extrapolate")

    fig = plt.figure()
    plt.plot(phases, profile, drawstyle="steps-mid")
    fine_phases = np.linspace(0, 1, 1000)
    fine_phases_shifted = fine_phases - toa / period + additional_phase
    model = mod(fine_phases_shifted - np.floor(fine_phases_shifted))
    model = np.concatenate((model, model))
    plt.plot(np.linspace(0, 2, 2000), model)
    if toaerr is not None:
        plt.axvline((toa - toaerr) / period)
        plt.axvline((toa + toaerr) / period)
    plt.axvline(toa / period - 0.5 / len(profile), ls="--")
    plt.axvline(toa / period + 0.5 / len(profile), ls="--")
    timestamp = int(time.time())
    plt.savefig("{}.png".format(timestamp))
    if not show:
        plt.close(fig)


def get_TOA(
    prof,
    period,
    tstart,
    template=None,
    additional_phase=0,
    quick=False,
    debug=False,
    use_bootstrap=False,
    **fftfit_kwargs,
):
    """Calculate the Time-Of-Arrival of a pulse.

    Parameters
    ----------
    prof : array
        The pulse profile
    template : array, default None
        The template of the pulse used to perform the TOA calculation, if any.
        Otherwise use the default of fftfit
    tstart : float
        The time at the start of the pulse profile

    Other parameters
    ----------------
    nstep : int, optional, default 100
        Number of steps for the bootstrap method

    Returns
    -------
    toa, toastd : floats
        Mean and standard deviation of the TOA
    """
    nbin = len(prof)

    ph = np.arange(0, 1, 1 / nbin)
    if template is None:
        template = np.cos(2 * np.pi * ph)

    mean_amp, std_amp, phase_res, phase_res_err = fftfit(
        prof, template=template, quick=quick, use_bootstrap=use_bootstrap, **fftfit_kwargs
    )
    phase_res = phase_res + additional_phase
    phase_res = phase_res - np.floor(phase_res)

    toa = tstart + phase_res * period
    toaerr = phase_res_err * period

    if debug:
        _plot_TOA_fit(
            prof,
            template,
            toa - tstart,
            toaerr=toaerr,
            additional_phase=additional_phase,
            period=period,
        )

    return toa, toaerr


def _load_and_prepare_TOAs(mjds, ephem="DE405"):
    toalist = [None] * len(mjds)
    for i, m in enumerate(mjds):
        toalist[i] = toa.TOA(m, obs="Barycenter", scale="tdb")

    toalist = toa.TOAs(toalist=toalist)
    if "tdb" not in toalist.table.colnames:
        toalist.compute_TDBs(ephem=ephem)
    if "ssb_obs_pos" not in toalist.table.colnames:
        toalist.compute_posvels(ephem, False)
    return toalist


def get_orbital_correction_from_ephemeris_file(
    mjdstart, mjdstop, parfile, ntimes=1000, ephem="DE405", return_pint_model=False
):
    """Get a correction for orbital motion from pulsar parameter file.

    Parameters
    ----------
    mjdstart, mjdstop : float
        Start and end of the time interval where we want the orbital solution
    parfile : str
        Any parameter file understood by PINT (Tempo or Tempo2 format)

    Other parameters
    ----------------
    ntimes : int
        Number of time intervals to use for interpolation. Default 1000

    Returns
    -------
    correction_sec : function
        Function that accepts in input an array of times in seconds and a
        floating-point MJDref value, and returns the deorbited times
    correction_mjd : function
        Function that accepts times in MJDs and returns the deorbited times.
    """
    from scipy.interpolate import interp1d
    from astropy import units

    if not HAS_PINT:
        raise ImportError(
            "You need the optional dependency PINT to use this "
            "functionality: github.com/nanograv/pint"
        )

    simon("Assuming events are already referred to the solar system barycenter (timescale is TDB)")

    mjds = np.linspace(mjdstart, mjdstop, ntimes)
    toalist = _load_and_prepare_TOAs(mjds, ephem=ephem)
    m = get_model(parfile)
    delays = m.delay(toalist)

    correction_mjd_rough = interp1d(
        mjds,
        (toalist.table["tdbld"] * units.d - delays).to(units.d).value,
        fill_value="extrapolate",
    )

    def correction_mjd(mjds):
        """Get the orbital correction.

        Parameters
        ----------
        mjds : array-like
            The input times in MJD

        Returns
        -------
        mjds: Corrected times in MJD
        """
        xvals = correction_mjd_rough.x
        # Maybe this will be fixed if scipy/scipy#9602 is accepted
        bad = (mjds < xvals[0]) | (np.any(mjds > xvals[-1]))
        if np.any(bad):
            warnings.warn(
                "Some points are outside the interpolation range:" " {}".format(mjds[bad])
            )
        return correction_mjd_rough(mjds)

    def correction_sec(times, mjdref):
        """Get the orbital correction.

        Parameters
        ----------
        times : array-like
            The input times in seconds of Mission Elapsed Time (MET)
        mjdref : float
            MJDREF, reference MJD for the mission

        Returns
        -------
        mets: array-like
            Corrected times in MET seconds
        """
        deorb_mjds = correction_mjd(times / 86400 + mjdref)
        return np.array((deorb_mjds - mjdref) * 86400)

    retvals = [correction_sec, correction_mjd]
    if return_pint_model:
        retvals.append(m)
    return retvals

import numpy as np
from collections.abc import Iterable
from .pulsar import ef_profile_stat, pdm_profile_stat
from .pulsar import fold_events, z_n, pulse_phase
from ..utils import jit, HAS_NUMBA
from ..utils import contiguous_regions
from astropy.stats import poisson_conf_interval
import matplotlib.pyplot as plt


__all__ = [
    "epoch_folding_search",
    "z_n_search",
    "search_best_peaks",
    "plot_profile",
    "plot_phaseogram",
    "phaseogram",
    "phase_dispersion_search",
]


@jit(nopython=True)
def _pulse_phase_fast(time, f, fdot, buffer_array):
    for i in range(len(time)):
        buffer_array[i] = time[i] * f + 0.5 * time[i] ** 2 * fdot
        buffer_array[i] -= np.floor(buffer_array[i])
    return buffer_array


def _folding_search(
    stat_func, times, frequencies, segment_size=np.inf, use_times=False, fdots=0, **kwargs
):
    fgrid, fdgrid = np.meshgrid(
        np.asanyarray(frequencies).astype(np.float64), np.asanyarray(fdots).astype(np.float64)
    )
    stats = np.zeros_like(fgrid)
    times = (times - times[0]).astype(np.float64)
    length = times[-1]
    if length < segment_size:
        segment_size = length
    start_times = np.arange(times[0], times[-1], segment_size)
    count = 0
    for s in start_times:
        good = (times >= s) & (times < s + segment_size)
        ts = times[good]
        if len(ts) < 1 or ts[-1] - ts[0] < 0.2 * segment_size:
            continue
        buffer = np.zeros_like(ts)
        for i in range(stats.shape[0]):
            for j in range(stats.shape[1]):
                f = fgrid[i, j]
                fd = fdgrid[i, j]
                if use_times:
                    kwargs_copy = {}
                    for key in kwargs.keys():
                        if isinstance(kwargs[key], Iterable) and len(kwargs[key]) == len(times):
                            kwargs_copy[key] = kwargs[key][good]
                        else:
                            kwargs_copy[key] = kwargs[key]
                    stats[i, j] += stat_func(ts, f, fd, **kwargs_copy)
                else:
                    phases = _pulse_phase_fast(ts, f, fd, buffer)
                    stats[i, j] += stat_func(phases)
        count += 1

    if fgrid.shape[0] == 1:
        return fgrid.flatten(), stats.flatten() / count
    else:
        return fgrid, fdgrid, stats / count


@jit(nopython=True)
def _bincount_fast(phase):
    return np.bincount(phase)


@jit(nopython=True)
def _profile_fast(phase, nbin=128):
    phase_bin = np.zeros(len(phase) + 2, dtype=np.int64)
    # This is done to force bincount from 0 to nbin -1
    phase_bin[-1] = nbin - 1
    phase_bin[-2] = 0
    for i in range(len(phase)):
        phase_bin[i] = np.int64(np.floor(phase[i] * nbin))
    bc = _bincount_fast(phase_bin)
    bc[0] -= 1
    bc[-1] -= 1
    return bc


def epoch_folding_search(
    times, frequencies, nbin=128, segment_size=np.inf, expocorr=False, gti=None, weights=1, fdots=0
):
    """Performs epoch folding at trial frequencies in photon data.

    If no exposure correction is needed and numba is installed, it uses a fast
    algorithm to perform the folding. Otherwise, it runs a *much* slower
    algorithm, which however yields a more precise result.
    The search can be done in segments and the results averaged. Use
    segment_size to control this

    Parameters
    ----------
    times : array-like
        the event arrival times

    frequencies : array-like
        the trial values for the frequencies

    Other Parameters
    ----------------
    nbin : int
        the number of bins of the folded profiles

    segment_size : float
        the length of the segments to be averaged in the periodogram

    fdots : array-like
        trial values of the first frequency derivative (optional)

    expocorr : bool
        correct for the exposure (Use it if the period is comparable to the
        length of the good time intervals). If True, GTIs have to be specified
        via the ``gti`` keyword

    gti : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good time intervals

    weights : array-like
        weight for each time. This might be, for example, the number of counts
        if the times array contains the time bins of a light curve

    Returns
    -------
    (fgrid, stats) or (fgrid, fdgrid, stats), as follows:

    fgrid : array-like
        frequency grid of the epoch folding periodogram
    fdgrid : array-like
        frequency derivative grid. Only returned if fdots is an array.
    stats : array-like
        the epoch folding statistics corresponding to each frequency bin.
    """

    if expocorr or not HAS_NUMBA or isinstance(weights, Iterable):
        if expocorr and gti is None:
            raise ValueError("To calculate exposure correction, you need to" " specify the GTIs")

        def stat_fun(t, f, fd=0, **kwargs):
            return ef_profile_stat(fold_events(t, f, fd, **kwargs)[1])

        return _folding_search(
            stat_fun,
            times,
            frequencies,
            segment_size=segment_size,
            use_times=True,
            expocorr=expocorr,
            weights=weights,
            gti=gti,
            nbin=nbin,
            fdots=fdots,
        )

    return _folding_search(
        lambda x: ef_profile_stat(_profile_fast(x, nbin=nbin)),
        times,
        frequencies,
        segment_size=segment_size,
        fdots=fdots,
    )


def phase_dispersion_search(
    times, flux, frequencies, nbin=128, segment_size=5000, expocorr=False, gti=None, fdots=0
):
    """Performs folding at trial frequencies in time series data (i.e.~a light curve
    of flux or photon counts) and computes the Phase Dispersion Minimization statistic.

    If no exposure correction is needed and numba is installed, it uses a fast
    algorithm to perform the folding. Otherwise, it runs a *much* slower
    algorithm, which however yields a more precise result.
    The search can be done in segments and the results averaged. Use
    segment_size to control this

    Parameters
    ----------
    times : array-like
        the time stamps of the time series

    flux : array-like
        the flux or photon count values of the time series

    frequencies : array-like
        the trial values for the frequencies

    Other Parameters
    ----------------
    nbin : int
        the number of bins of the folded profiles

    segment_size : float
        the length of the segments to be averaged in the periodogram

    fdots : array-like
        trial values of the first frequency derivative (optional)

    expocorr : bool
        correct for the exposure (Use it if the period is comparable to the
        length of the good time intervals). If True, GTIs have to be specified
        via the ``gti`` keyword

    gti : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good time intervals

    Returns
    -------
    (fgrid, stats) or (fgrid, fdgrid, stats), as follows:

    fgrid : array-like
        frequency grid of the epoch folding periodogram
    fdgrid : array-like
        frequency derivative grid. Only returned if fdots is an array.
    stats : array-like
        the epoch folding statistics corresponding to each frequency bin.
    """

    def stat_fun(t, f, fd=0, **kwargs):
        bins, profile, _ = fold_events(t, f, fd, **kwargs, mode="pdm")
        flux = kwargs["weights"]
        len_flux = len(flux)
        # ddof = 1 computes the unbiased estimator of the variance
        sigma = np.var(flux, ddof=1)
        return pdm_profile_stat(profile, sigma, len_flux)

    return _folding_search(
        stat_fun,
        times,
        frequencies,
        segment_size=segment_size,
        use_times=True,
        expocorr=expocorr,
        weights=flux,
        gti=gti,
        nbin=nbin,
        fdots=fdots,
    )


def z_n_search(
    times,
    frequencies,
    nharm=4,
    nbin=128,
    segment_size=np.inf,
    expocorr=False,
    weights=1,
    gti=None,
    fdots=0,
):
    """Calculates the Z^2_n statistics at trial frequencies in photon data.

    The "real" Z^2_n statistics is very slow. Therefore, in this function data
    are folded first, and then the statistics is calculated using the value of
    the profile as an additional normalization term.
    The two methods are mostly equivalent. However, the number of bins has to
    be chosen wisely: if the number of bins is too small, the search for high
    harmonics is ineffective.
    If no exposure correction is needed and numba is installed, it uses a fast
    algorithm to perform the folding. Otherwise, it runs a *much* slower
    algorithm, which however yields a more precise result.
    The search can be done in segments and the results averaged. Use
    segment_size to control this

    Parameters
    ----------
    times : array-like
        the event arrival times

    frequencies : array-like
        the trial values for the frequencies

    Other Parameters
    ----------------
    nbin : int
        the number of bins of the folded profiles

    segment_size : float
        the length of the segments to be averaged in the periodogram

    fdots : array-like
        trial values of the first frequency derivative (optional)

    expocorr : bool
        correct for the exposure (Use it if the period is comparable to the
        length of the good time intervals.)

    gti : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good time intervals

    weights : array-like
        weight for each time. This might be, for example, the number of counts
        if the times array contains the time bins of a light curve

    Returns
    -------
    (fgrid, stats) or (fgrid, fdgrid, stats), as follows:

    fgrid : array-like
        frequency grid of the epoch folding periodogram
    fdgrid : array-like
        frequency derivative grid. Only returned if fdots is an array.
    stats : array-like
        the Z^2_n statistics corresponding to each frequency bin.
    """
    phase = np.arange(0, 1, 1 / nbin)
    if expocorr or not HAS_NUMBA or isinstance(weights, Iterable):
        if expocorr and gti is None:
            raise ValueError("To calculate exposure correction, you need to" " specify the GTIs")

        def stat_fun(t, f, fd=0, **kwargs):
            return z_n(fold_events(t, f, fd, nbin=nbin, **kwargs)[1], n=nharm, datatype="binned")

        return _folding_search(
            stat_fun,
            times,
            frequencies,
            segment_size=segment_size,
            use_times=True,
            expocorr=expocorr,
            weights=weights,
            gti=gti,
            fdots=fdots,
        )

    return _folding_search(
        lambda x: z_n(_profile_fast(x, nbin=nbin), n=nharm, datatype="binned"),
        times,
        frequencies,
        segment_size=segment_size,
        fdots=fdots,
    )


def search_best_peaks(x, stat, threshold):
    """Search peaks above threshold in an epoch folding periodogram.

    If more values of stat are above threshold and are contiguous, only the
    largest one is returned (see Examples).

    Parameters
    ----------
    x : array-like
        The x axis of the periodogram (frequencies, periods, ...)

    stat : array-like
        The y axis. It must have the same shape as x

    threshold : float
        The threshold value over which we look for peaks in the stat array

    Returns
    -------
    best_x : array-like
        the array containing the x position of the peaks above threshold. If no
        peaks are above threshold, an empty list is returned. The array is
        sorted by inverse value of stat

    best_stat : array-like
        for each best_x, give the corresponding stat value. Empty if no peaks
        above threshold.

    Examples
    --------
    >>> # Test multiple peaks
    >>> x = np.arange(10)
    >>> stat = [0, 0, 0.5, 0, 0, 1, 1, 2, 1, 0]
    >>> best_x, best_stat = search_best_peaks(x, stat, 0.5)
    >>> len(best_x)
    2
    >>> assert np.isclose(best_x[0], 7.0)
    >>> assert np.isclose(best_x[1], 2.0)
    >>> stat = [0, 0, 2.5, 0, 0, 1, 1, 2, 1, 0]
    >>> best_x, best_stat = search_best_peaks(x, stat, 0.5)
    >>> assert np.isclose(best_x[0], 2.0)
    >>> # Test no peak above threshold
    >>> x = np.arange(10)
    >>> stat = [0, 0, 0.4, 0, 0, 0, 0, 0, 0, 0]
    >>> best_x, best_stat = search_best_peaks(x, stat, 0.5)
    >>> best_x
    []
    >>> best_stat
    []

    """
    stat = np.asanyarray(stat)
    x = np.asanyarray(x)
    peaks = stat >= threshold
    regions = contiguous_regions(peaks)
    if len(regions) == 0:
        return [], []
    best_x = np.zeros(len(regions))
    best_stat = np.zeros(len(regions))
    for i, r in enumerate(regions):
        stat_filt = stat[r[0] : r[1]]
        x_filt = x[r[0] : r[1]]
        max_arg = np.argmax(stat_filt)
        best_stat[i] = stat_filt[max_arg]
        best_x[i] = x_filt[max_arg]

    order = np.argsort(best_stat)[::-1]

    return best_x[order], best_stat[order]


def plot_profile(phase, profile, err=None, ax=None):
    """Plot a pulse profile showing some stats.

    If err is None, the profile is assumed in counts and the Poisson confidence
    level is plotted. Otherwise, err is shown as error bars

    Parameters
    ----------
    phase : array-like
        The bins on the x-axis

    profile : array-like
        The pulsed profile

    Other Parameters
    ----------------
    ax : `matplotlib.pyplot.axis` instance
        Axis to plot to. If None, create a new one.

    Returns
    -------
    ax : `matplotlib.pyplot.axis` instance
        Axis where the profile was plotted.
    """
    if ax is None:
        plt.figure("Pulse profile")
        ax = plt.subplot()
    mean = np.mean(profile)
    if np.all(phase < 1.5):
        phase = np.concatenate((phase, phase + 1))
        profile = np.concatenate((profile, profile))
    ax.plot(phase, profile, drawstyle="steps-mid")
    if err is None:
        err_low, err_high = poisson_conf_interval(mean, interval="frequentist-confidence", sigma=1)
        ax.axhspan(err_low, err_high, alpha=0.5)
    else:
        err = np.concatenate((err, err))
        ax.errorbar(phase, profile, yerr=err, fmt="none")

    ax.set_ylabel("Counts")
    ax.set_xlabel("Phase")
    return ax


def plot_phaseogram(phaseogram, phase_bins, time_bins, unit_str="s", ax=None, **plot_kwargs):
    """Plot a phaseogram.

    Parameters
    ----------
    phaseogram : NxM array
        The phaseogram to be plotted

    phase_bins : array of M + 1 elements
        The bins on the x-axis

    time_bins : array of N + 1 elements
        The bins on the y-axis

    Other Parameters
    ----------------
    unit_str : str
        String indicating the time unit (e.g. 's', 'MJD', etc)

    ax : `matplotlib.pyplot.axis` instance
        Axis to plot to. If None, create a new one.

    plot_kwargs : dict
        Additional arguments to be passed to pcolormesh

    Returns
    -------
    ax : `matplotlib.pyplot.axis` instance
        Axis where the phaseogram was plotted.
    """
    if ax is None:
        plt.figure("Phaseogram")
        ax = plt.subplot()

    ax.pcolormesh(phase_bins, time_bins, phaseogram.T, **plot_kwargs)
    ax.set_ylabel("Time ({})".format(unit_str))
    ax.set_xlabel("Phase")
    ax.set_xlim([0, np.max(phase_bins)])
    ax.set_ylim([np.min(time_bins), np.max(time_bins)])
    return ax


def phaseogram(
    times,
    f,
    nph=128,
    nt=32,
    ph0=0,
    mjdref=None,
    fdot=0,
    fddot=0,
    pepoch=None,
    plot=False,
    phaseogram_ax=None,
    weights=None,
    **plot_kwargs,
):
    """
    Calculate and plot the phaseogram of a pulsar observation.

    The phaseogram is a 2-D histogram where the x axis is the pulse phase and
    the y axis is the time. It shows how the pulse phase changes with time, and
    it is very useful to see if the pulse solution is correct and/or if there
    are additional frequency derivatives appearing in the data (due to spin up
    or down, or even orbital motion)

    Parameters
    ----------
    times : array
        Event arrival times

    f : float
        Pulse frequency

    Other parameters
    ----------------
    nph : int
        Number of phase bins

    nt : int
        Number of time bins

    ph0 : float
        The starting phase of the pulse

    mjdref : float
        MJD reference time. If given, the y axis of the plot will be in MJDs,
        otherwise it will be in seconds.

    fdot : float
        First frequency derivative

    fddot : float
        Second frequency derivative

    pepoch : float
        If the input pulse solution is referred to a given time, give it here.
        It has no effect (just a phase shift of the pulse) if `fdot` is zero.
        if `mjdref` is specified, pepoch MUST be in MJD

    weights : array
        Weight for each time

    plot : bool
        Return the axes in the additional_info, and don't close the plot, so
        that the user can add information to it.

    Returns
    -------
    phaseogr : 2-D matrix
        The phaseogram

    phases : array-like
        The x axis of the phaseogram (the x bins of the histogram),
        corresponding to the pulse phase in each column

    times : array-like
        The y axis of the phaseogram (the y bins of the histogram),
        corresponding to the time at each row

    additional_info : dict
        Additional information, like the pulse profile and the axes to modify
        the plot (the latter, only if `return_plot` is True)
    """

    use_mjdref = False
    if mjdref is not None:
        use_mjdref = True

    if pepoch is None:
        pepoch = (times[-1] + times[0]) / 2
        if use_mjdref:
            pepoch /= 86400

    plot_unit = "s"
    if use_mjdref:
        pepoch = (pepoch - mjdref) * 86400
        plot_unit = "MJD"

    phases = pulse_phase((times - pepoch), f, fdot, fddot, to_1=True, ph0=ph0)

    allphases = np.concatenate([phases, phases + 1]).astype("float64")
    allts = np.concatenate([times, times]).astype("float64")

    if weights is not None and isinstance(weights, Iterable):
        if len(weights) != len(times):
            raise ValueError("The length of weights must match the length of " "times")
        weights = np.concatenate([weights, weights]).astype("float64")

    if use_mjdref:
        allts = allts / 86400 + mjdref

    phas, binx, biny = np.histogram2d(
        allphases,
        allts,
        bins=(np.linspace(0, 2, nph * 2 + 1), np.linspace(np.min(allts), np.max(allts), nt + 1)),
        weights=weights,
    )

    if plot:
        phaseogram_ax = plot_phaseogram(
            phas, binx, biny, ax=phaseogram_ax, unit_str=plot_unit, **plot_kwargs
        )
        additional_info = {"ax": phaseogram_ax}
    else:
        additional_info = {}

    return phas, binx, biny, additional_info

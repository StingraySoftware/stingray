import warnings
import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from astropy.table import Table

from stingray.lightcurve import Lightcurve
from stingray.loggingconfig import setup_logger
from ..crossspectrum import AveragedCrossspectrum, get_flux_generator
from ..powerspectrum import AveragedPowerspectrum
from ..fourier import normalize_periodograms, fft, fftfreq, positive_fft_bins
from ..utils import show_progress
from ..gti import cross_two_gtis


__all__ = ["calculate_FAD_correction", "get_periodograms_from_FAD_results", "FAD"]

logger = setup_logger()


def FAD(
    data1,
    data2,
    segment_size,
    dt=None,
    norm="frac",
    plot=False,
    ax=None,
    smoothing_alg="gauss",
    smoothing_length=None,
    verbose=False,
    tolerance=0.05,
    strict=False,
    output_file=None,
    return_objects=False,
):
    r"""Calculate Frequency Amplitude Difference-corrected (cross)power spectra.

    Reference: Bachetti \& Huppenkothen, 2018, ApJ, 853L, 21

    The two input light curve must be strictly simultaneous, and recorded by
    two independent detectors with similar responses, so that the count rates
    are similar and dead time is independent.
    The method does not apply to different energy channels of the same
    instrument, or to the signal observed by two instruments with very
    different responses. See the paper for caveats.

    Parameters
    ----------
    data1 : `Lightcurve` or `EventList`
        Input data for channel 1
    data2 : `Lightcurve` or `EventList`
        Input data for channel 2. Must be strictly simultaneous to ``data1``
        and, if a light curve, have the same binning time. Also, it must be
        strictly independent, e.g. from a different detector. There must be
        no dead time cross-talk between the two time series.
    segment_size: float
        The final Fourier products are averaged over many segments of the
        input light curves. This is the length of each segment being averaged.
        Note that the light curve must be long enough to have at least 30
        segments, as the result gets better as one averages more and more
        segments.
    dt : float
        Time resolution of the light curves used to produce periodograms
    norm: {``frac``, ``abs``, ``leahy``, ``none``}, default ``none``
        The normalization of the (real part of the) cross spectrum.


    Other parameters
    ----------------
    plot : bool, default False
        Plot diagnostics: check if the smoothed Fourier difference scatter is
        a good approximation of the data scatter.
    ax : :class:`matplotlib.axes.axes` object
        If not None and ``plot`` is True, use this axis object to produce
         the diagnostic plot. Otherwise, create a new figure.
    smoothing_alg : {'gauss', ...}
        Smoothing algorithm. For now, the only smoothing algorithm allowed is
        ``gauss``, which applies a Gaussian Filter from `scipy`.
    smoothing_length : int, default ``segment_size * 3``
        Number of bins to smooth in gaussian window smoothing
    verbose: bool, default False
        Print out information on the outcome of the algorithm (recommended)
    tolerance : float, default 0.05
        Accepted relative error on the FAD-corrected Fourier amplitude, to be
        used as success diagnostics.
        Should be
        ```
        stdtheor = 2 / np.sqrt(n)
        std = (average_corrected_fourier_diff / n).std()
        np.abs((std - stdtheor) / stdtheor) < tolerance
        ```
    strict : bool, default False
        Decide what to do if the condition on tolerance is not met. If True,
        raise a ``RuntimeError``. If False, just throw a warning.
    output_file : str, default None
        Name of an output file (any extension automatically recognized by
        Astropy is fine)

    Returns
    -------
    results : class:`astropy.table.Table` object or ``dict`` or ``str``
        The content of ``results`` depends on whether ``return_objects`` is
        True or False.
        If ``return_objects==False``,
        ``results`` is a `Table` with the following columns:

        + pds1: the corrected PDS of ``lc1``
        + pds2: the corrected PDS of ``lc2``
        + cs: the corrected cospectrum
        + ptot: the corrected PDS of lc1 + lc2

        If ``return_objects`` is True, ``results`` is a ``dict``, with keys
        named like the columns
        listed above but with `AveragePowerspectrum` or
        `AverageCrossspectrum` objects instead of arrays.

    """
    gti = cross_two_gtis(data1.gti, data2.gti)
    data1.gti = data2.gti = gti
    if isinstance(data1, Lightcurve):
        dt = data1.dt

    flux_iterable1 = get_flux_generator(data1, segment_size, dt=dt)
    flux_iterable2 = get_flux_generator(data2, segment_size, dt=dt)
    # Initialize stuff
    freq = None
    # These will be the final averaged periodograms. Initializing with a single
    # scalar 0, but the final products will be arrays.
    pds1_unnorm = 0
    pds2_unnorm = 0
    ptot_unnorm = 0
    cs_unnorm = 0
    pds1 = 0
    pds2 = 0
    ptot = 0
    cs = 0j
    M = 0
    nph1_tot = nph2_tot = nph_tot = 0
    average_diff = average_diff_uncorr = 0

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

    for flux1, flux2 in show_progress(zip(flux_iterable1, flux_iterable2)):
        if flux1 is None or flux2 is None:
            continue

        N = flux1.size
        segment_size = N * dt
        if smoothing_length is None:
            smoothing_length = segment_size * 3
        if freq is None:
            fgt0 = positive_fft_bins(N)
            freq = fftfreq(N, dt)[fgt0]

        # Calculate the sum of each light curve, to calculate the mean
        # This will
        nph1 = flux1.sum()
        nph2 = flux2.sum()
        nphtot = nph1 + nph2

        # Calculate the FFTs
        f1 = fft(flux1)[fgt0]
        f2 = fft(flux2)[fgt0]
        ftot = fft(flux1 + flux2)[fgt0]

        f1_leahy = f1 * np.sqrt(2 / nph1)
        f2_leahy = f2 * np.sqrt(2 / nph2)
        ftot_leahy = ftot * np.sqrt(2 / nphtot)

        fourier_diff = f1_leahy - f2_leahy
        if plot:
            ax.scatter(freq, fourier_diff.real, s=1)

        if smoothing_alg == "gauss":
            smooth_real = gaussian_filter1d(fourier_diff.real**2, smoothing_length)
        else:
            raise ValueError("Unknown smoothing algorithm: {}".format(smoothing_alg))

        p1 = (f1 * f1.conj()).real
        p1 = p1 / smooth_real * 2
        p2 = (f2 * f2.conj()).real
        p2 = p2 / smooth_real * 2
        pt = (ftot * ftot.conj()).real
        pt = pt / smooth_real * 2

        c = f2 * f1.conj()
        c = c / smooth_real * 2

        nphgeom = np.sqrt(nph1 * nph2)
        power1 = normalize_periodograms(p1, dt, N, nph1 / N, n_ph=nph1, norm=norm)
        power2 = normalize_periodograms(p2, dt, N, nph2 / N, n_ph=nph2, norm=norm)
        power_tot = normalize_periodograms(pt, dt, N, nphtot / N, n_ph=nphtot, norm=norm)
        cs_power = normalize_periodograms(c, dt, N, nphgeom / N, n_ph=nphgeom, norm=norm)

        if M == 0 and plot:
            ax.plot(freq, smooth_real, zorder=10, lw=3)
            ax.plot(freq, f1_leahy.real, zorder=5, lw=1)
            ax.plot(freq, f2_leahy.real, zorder=5, lw=1)

        # Save the unnormalised (but smoothed) powerspectra and cross-spectrum
        pds1_unnorm += p1
        pds2_unnorm += p2
        ptot_unnorm += pt
        cs_unnorm += c

        # Save the normalised and smoothed powerspectra and cross-spectrum
        ptot += power_tot
        pds1 += power1
        pds2 += power2
        cs += cs_power

        average_diff += fourier_diff / smooth_real**0.5 * np.sqrt(2)
        average_diff_uncorr += fourier_diff
        nph1_tot += nph1
        nph2_tot += nph2
        nph_tot += nphtot
        M += 1

    std = (average_diff / M).std()
    stdtheor = 2 / np.sqrt(M)
    stduncorr = (average_diff_uncorr / M).std()
    is_compliant = np.abs((std - stdtheor) / stdtheor) < tolerance
    verbose_string = """
        -------- FAD correction ----------
        I smoothed over {smoothing_length} power spectral bins
        {M} intervals averaged.
        The uncorrected standard deviation of the Fourier
        differences is {stduncorr} (dead-time affected!)
        The final standard deviation of the FAD-corrected
        Fourier differences is {std}. For the results to be
        acceptable, this should be close to {stdtheor}
        to within {tolerance} %.
        In this case, the results ARE {compl}complying.
        {additional}
        ----------------------------------
        """.format(
        smoothing_length=smoothing_length,
        M=M,
        stduncorr=stduncorr,
        std=std,
        stdtheor=stdtheor,
        tolerance=tolerance * 100,
        compl="NOT " if not is_compliant else "",
        additional="Maybe something is not right." if not is_compliant else "",
    )

    if verbose and is_compliant:
        logger.info(verbose_string)
    elif not is_compliant:
        warnings.warn(verbose_string)

    if strict and not is_compliant:
        raise RuntimeError("Results are not compliant, and `strict` mode " "selected. Exiting.")

    results = Table()

    results["freq"] = freq
    results["pds1"] = pds1 / M
    results["pds2"] = pds2 / M
    results["cs"] = cs / M
    results["ptot"] = ptot / M
    results["pds1_unnorm"] = pds1_unnorm / M
    results["pds2_unnorm"] = pds2_unnorm / M
    results["cs_unnorm"] = cs_unnorm / M
    results["ptot_unnorm"] = ptot_unnorm / M
    results["fad"] = average_diff / M
    results.meta["fad_delta"] = (std - stdtheor) / stdtheor
    results.meta["is_compliant"] = is_compliant
    results.meta["M"] = M
    results.meta["dt"] = dt
    results.meta["nph1"] = nph1_tot / M
    results.meta["nph2"] = nph2_tot / M
    results.meta["nph"] = nph_tot / M
    results.meta["norm"] = norm
    results.meta["smoothing_length"] = smoothing_length
    results.meta["df"] = np.mean(np.diff(freq))

    if output_file is not None:
        results.write(output_file, overwrite=True)

    if return_objects:
        result_table = results
        results = {}
        results["pds1"] = get_periodograms_from_FAD_results(result_table, kind="pds1")
        results["pds2"] = get_periodograms_from_FAD_results(result_table, kind="pds2")
        results["cs"] = get_periodograms_from_FAD_results(result_table, kind="cs")
        results["ptot"] = get_periodograms_from_FAD_results(result_table, kind="ptot")
        results["fad"] = result_table["fad"]

    return results


def calculate_FAD_correction(
    lc1,
    lc2,
    segment_size,
    norm="frac",
    gti=None,
    plot=False,
    ax=None,
    smoothing_alg="gauss",
    smoothing_length=None,
    verbose=False,
    tolerance=0.05,
    strict=False,
    output_file=None,
    return_objects=False,
):
    r"""Calculate Frequency Amplitude Difference-corrected (cross)power spectra.

    Reference: Bachetti \& Huppenkothen, 2018, ApJ, 853L, 21

    The two input light curve must be strictly simultaneous, and recorded by
    two independent detectors with similar responses, so that the count rates
    are similar and dead time is independent.
    The method does not apply to different energy channels of the same
    instrument, or to the signal observed by two instruments with very
    different responses. See the paper for caveats.

    Parameters
    ----------
    lc1: class:`stingray.ligthtcurve.Lightcurve`
        Light curve from channel 1
    lc2: class:`stingray.ligthtcurve.Lightcurve`
        Light curve from channel 2. Must be strictly simultaneous to ``lc1``
        and have the same binning time. Also, it must be strictly independent,
        e.g. from a different detector. There must be no dead time cross-talk
        between the two light curves.
    segment_size: float
        The final Fourier products are averaged over many segments of the
        input light curves. This is the length of each segment being averaged.
        Note that the light curve must be long enough to have at least 30
        segments, as the result gets better as one averages more and more
        segments.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, default ``none``
        The normalization of the (real part of the) cross spectrum.


    Other parameters
    ----------------
    plot : bool, default False
        Plot diagnostics: check if the smoothed Fourier difference scatter is
        a good approximation of the data scatter.
    ax : :class:`matplotlib.axes.axes` object
        If not None and ``plot`` is True, use this axis object to produce
         the diagnostic plot. Otherwise, create a new figure.
    smoothing_alg : {'gauss', ...}
        Smoothing algorithm. For now, the only smoothing algorithm allowed is
        ``gauss``, which applies a Gaussian Filter from `scipy`.
    smoothing_length : int, default ``segment_size * 3``
        Number of bins to smooth in gaussian window smoothing
    verbose: bool, default False
        Print out information on the outcome of the algorithm (recommended)
    tolerance : float, default 0.05
        Accepted relative error on the FAD-corrected Fourier amplitude, to be
        used as success diagnostics.
        Should be
        ```
        stdtheor = 2 / np.sqrt(n)
        std = (average_corrected_fourier_diff / n).std()
        np.abs((std - stdtheor) / stdtheor) < tolerance
        ```
    strict : bool, default False
        Decide what to do if the condition on tolerance is not met. If True,
        raise a ``RuntimeError``. If False, just throw a warning.
    output_file : str, default None
        Name of an output file (any extension automatically recognized by
        Astropy is fine)

    Returns
    -------
    results : class:`astropy.table.Table` object or ``dict`` or ``str``
        The content of ``results`` depends on whether ``return_objects`` is
        True or False.
        If ``return_objects==False``,
        ``results`` is a `Table` with the following columns:

        + pds1: the corrected PDS of ``lc1``
        + pds2: the corrected PDS of ``lc2``
        + cs: the corrected cospectrum
        + ptot: the corrected PDS of lc1 + lc2

        If ``return_objects`` is True, ``results`` is a ``dict``, with keys
        named like the columns
        listed above but with `AveragePowerspectrum` or
        `AverageCrossspectrum` objects instead of arrays.

    """
    return FAD(
        lc1,
        lc2,
        segment_size,
        dt=lc1.dt,
        norm=norm,
        plot=plot,
        ax=ax,
        smoothing_alg=smoothing_alg,
        smoothing_length=smoothing_length,
        verbose=verbose,
        tolerance=tolerance,
        strict=strict,
        output_file=output_file,
        return_objects=return_objects,
    )


def get_periodograms_from_FAD_results(FAD_results, kind="ptot"):
    """Get Stingray periodograms from FAD results.

    Parameters
    ----------
    FAD_results : :class:`astropy.table.Table` object or `str`
        Results from `calculate_FAD_correction`, either as a Table or an output
        file name
    kind : :class:`str`, one of ['ptot', 'pds1', 'pds2', 'cs']
        Kind of periodogram to get (E.g., 'ptot' -> PDS from the sum of the two
        light curves, 'cs' -> cospectrum, etc.)

    Returns
    -------
    results : `AveragedCrossspectrum` or `Averagedpowerspectrum` object
        The periodogram.
    """
    if isinstance(FAD_results, str):
        FAD_results = Table.read(FAD_results)

    if kind.startswith("p") and kind in FAD_results.colnames:
        powersp = AveragedPowerspectrum()
        powersp.nphots = FAD_results.meta["nph"]
        if "1" in kind:
            powersp.nphots = FAD_results.meta["nph1"]
        elif "2" in kind:
            powersp.nphots = FAD_results.meta["nph2"]
    elif kind == "cs":
        powersp = AveragedCrossspectrum(power_type="all")
        powersp.pds1 = get_periodograms_from_FAD_results(FAD_results, kind="pds1")
        powersp.pds2 = get_periodograms_from_FAD_results(FAD_results, kind="pds2")
        powersp.nphots1 = FAD_results.meta["nph1"]
        powersp.nphots2 = FAD_results.meta["nph2"]
    else:
        raise ValueError("Unknown periodogram type")

    powersp.freq = FAD_results["freq"]
    powersp.power = FAD_results[kind]
    powersp.unnorm_power = FAD_results[kind + "_unnorm"]
    powersp.power_err = np.zeros_like(powersp.power)
    powersp.unnorm_power_err = np.zeros_like(powersp.unnorm_power)
    powersp.m = FAD_results.meta["M"]
    powersp.df = FAD_results.meta["df"]
    powersp.dt = FAD_results.meta["dt"]
    powersp.n = len(powersp.freq) * 2
    powersp.norm = FAD_results.meta["norm"]

    return powersp

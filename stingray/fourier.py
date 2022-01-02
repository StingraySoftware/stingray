import copy
import warnings
from collections.abc import Iterable
import numpy as np
from astropy.table import Table
from .utils import histogram, show_progress, sum_if_not_none_or_initialize
from .gti import generate_indices_of_segment_boundaries_unbinned, generate_indices_of_segment_boundaries_binned


try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft, fftfreq

    pyfftw.interfaces.cache.enable()
except ImportError:
    warnings.warn("pyfftw not installed. Using standard scipy fft")
    from scipy.fft import fft, fftfreq


def positive_fft_bins(N, include_zero=False):
    """See https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details

    Examples
    --------
    >>> freq = np.fft.fftfreq(10)
    >>> good = freq > 0
    >>> goodbins = positive_fft_bins(10)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    >>> freq = np.fft.fftfreq(11)
    >>> good = freq > 0
    >>> goodbins = positive_fft_bins(11)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    >>> freq = np.fft.fftfreq(10)
    >>> good = freq >= 0
    >>> goodbins = positive_fft_bins(10, include_zero=True)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    >>> freq = np.fft.fftfreq(11)
    >>> good = freq >= 0
    >>> goodbins = positive_fft_bins(11, include_zero=True)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    """
    minbin = 1
    if include_zero:
        minbin = 0

    if N % 2 == 0:
        return slice(minbin, N // 2)
    else:
        return slice(minbin, (N + 1) // 2)


def poisson_level(meanrate=None, Nph=None, norm="abs"):
    """Poisson (white)-noise level in a periodogram of pure counting noise.

    Other Parameters
    ----------
    meanrate : float, default None
        Mean count rate in counts/s
    Nph : float, default None
        Total number of counts in the light curve
    norm : str, default "abs"
        Normalization of the periodogram. One of ["abs", "frac", "leahy"]

    Examples
    --------
    >>> poisson_level(norm="leahy")
    2.0
    >>> poisson_level(meanrate=10., norm="abs")
    20.0
    >>> poisson_level(meanrate=10., norm="frac")
    0.2
    >>> poisson_level(Nph=10, norm="none")
    10.0
    >>> poisson_level(meanrate=10., norm="asdfwrqfasdh3r")
    Traceback (most recent call last):
    ...
    ValueError: Unknown value for norm: asdfwrqfasdh3r...
    >>> poisson_level(meanrate=10, norm="none")
    Traceback (most recent call last):
    ...
    ValueError: Bad input parameters for norm none...
    >>> poisson_level(Nph=10, norm="abs")
    Traceback (most recent call last):
    ...
    ValueError: Bad input parameters for norm abs...
    """
    bad_input = norm.lower() in ["abs", "frac"] and meanrate is None
    bad_input = bad_input or (norm.lower() == "none" and Nph is None)

    if bad_input:
        raise ValueError(f"Bad input parameters for norm {norm}: Nph={Nph}, meanrate={meanrate}")

    if norm == "abs":
        return 2. * meanrate
    if norm == "frac":
        return 2. / meanrate
    if norm == "leahy":
        return 2.0
    if norm == "none":
        return float(Nph)
    raise ValueError(f"Unknown value for norm: {norm}")


def normalize_frac(power, dt, N, mean):
    """Fractional rms normalization, from the variance of the lc.

    Examples
    --------
    >>> mean = var = 1000000
    >>> N = 1000000
    >>> dt = 0.2
    >>> meanrate = mean / dt
    >>> lc = np.random.poisson(mean, N)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_frac(pds, dt, lc.size, mean)
    >>> np.isclose(pdsnorm[1:N//2].mean(), poisson_level(meanrate=meanrate,norm="frac"), rtol=0.01)
    True
    """
    #     (mean * N) / (mean /dt) = N * dt
    #     It's Leahy / meanrate;
    #     Nph = mean * N
    #     meanrate = mean / dt
    #     norm = 2 / (Nph * meanrate) = 2 * dt / (mean**2 * N)

    return power * 2. * dt / (mean ** 2 * N)


def normalize_abs(power, dt, N):
    """Absolute rms normalization, from the variance of the lc.

    Examples
    --------
    >>> mean = var = 100000
    >>> N = 1000000
    >>> dt = 0.2
    >>> meanrate = mean / dt
    >>> lc = np.random.poisson(mean, N)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_abs(pds, dt, lc.size)
    >>> np.isclose(pdsnorm[1:N//2].mean(), poisson_level(meanrate=meanrate, norm="abs"), rtol=0.01)
    True
    """
    #     It's frac * meanrate**2; Leahy / meanrate * meanrate**2
    #     Nph = mean * N
    #     meanrate = mean / dt
    #     norm = 2 / (Nph * meanrate) * meanrate**2 = 2 * dt / (mean**2 * N) * mean**2 / dt**2

    return power * 2. / N / dt


def normalize_leahy_from_variance(power, variance, N):
    """Leahy+83 normalization, from the variance of the lc.

    Examples
    --------
    >>> mean = var = 100000.
    >>> N = 1000000
    >>> lc = np.random.poisson(mean, N).astype(float)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_leahy_from_variance(pds, var, lc.size)
    >>> np.isclose(pdsnorm[0], 2 * np.sum(lc), rtol=0.01)
    True
    >>> np.isclose(pdsnorm[1:N//2].mean(), poisson_level(norm="leahy"), rtol=0.01)
    True
    """
    return power * 2. / (variance * N)


def normalize_leahy_poisson(power, Nph):
    """Leahy+83 normalization, from the variance of the lc.

    Examples
    --------
    >>> mean = var = 100000.
    >>> N = 1000000
    >>> lc = np.random.poisson(mean, N).astype(float)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_leahy_poisson(pds, np.sum(lc))
    >>> np.isclose(pdsnorm[0], 2 * np.sum(lc), rtol=0.01)
    True
    >>> np.isclose(pdsnorm[1:N//2].mean(), poisson_level(norm="leahy"), rtol=0.01)
    True
    """
    return power * 2. / Nph


def normalize_periodograms(unnorm_power, dt, N, mean, variance=None, norm="abs", power_type="all"):
    """Wrapper around all the normalize_NORM methods.

    Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
    fractional rms^2 normalization, or not at all.

    Parameters
    ----------
    unnorm_power: numpy.ndarray
        The unnormalized cross spectrum.

    dt: float
        The sampling time of the light curve

    mean: float
        The mean counts per bin of the light curve (if a cross spectrum, the geometrical
        mean of the counts per bin in the two channels)

    N: int
        The number of bins in the light curve

    Other parameters
    ----------------
    variance: float
        The average variance of the measurements in light curve (if a cross spectrum,
        the geometrical mean of the variances in the two channels). **NOT** the
        variance of the light curve, but of each flux measurement (square of
        light curve error bar)! Only relevant for the Leahy normalization of
        non-Poissonian data.

    norm : str
        One of `'leahy'` (Leahy+83), `'frac'` (fractional rms), `'abs'`
        (absolute rms)

    power_type : str
        One of `'real'` (real part), `'all'` (all complex powers), `'abs'`
        (absolute value)

    Returns
    -------
    power: numpy.nd.array
        The normalized co-spectrum (real part of the cross spectrum). For
        'none' normalization, imaginary part is returned as well.
    """

    if norm == "leahy" and variance is not None:
        pds = normalize_leahy_from_variance(unnorm_power, variance, N)
    elif norm == "leahy":
        pds = normalize_leahy_poisson(unnorm_power, N * mean)
    elif norm == "frac":
        pds = normalize_frac(unnorm_power, dt, N, mean)
    elif norm == "abs":
        pds = normalize_abs(unnorm_power, dt, N)
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


def bias_term(C, P1, P2, P1noise, P2noise, Nave, intrinsic_coherence=1.0):
    """Bias term from Ingram 2019.

    As recommended in the paper, returns 0 if N > 500

    Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    P1 : float `np.array`
        sub-band periodogram
    P2 : float `np.array`
        reference-band periodogram
    P1noise : float
        Poisson noise level of the sub-band periodogram
    P2noise : float
        Poisson noise level of the reference-band periodogram
    Nave : int
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    intrinsic_coherence : float, default 1
        If known, the intrinsic coherence.
    """
    if Nave > 500:
        return 0. * P1
    bsq = P1 * P2 - intrinsic_coherence * (P1 - P1noise) * (P2 - P2noise)
    return bsq / Nave


def raw_coherence(C, P1, P2, P1noise, P2noise, Nave, intrinsic_coherence=1):
    """Raw coherence from Ingram 2019.

    Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    P1 : float `np.array`
        sub-band periodogram
    P2 : float `np.array`
        reference-band periodogram
    P1noise : float
        Poisson noise level of the sub-band periodogram
    P2noise : float
        Poisson noise level of the reference-band periodogram
    Nave : int
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    intrinsic_coherence : float, default 1
        If known, the intrinsic coherence.
    """
    bsq = bias_term(C, P1, P2, P1noise, P2noise, Nave, intrinsic_coherence=intrinsic_coherence)
    num = (C * np.conj(C)).real - bsq
    if isinstance(num, Iterable):
        num[num < 0] = (C * np.conj(C)).real[num < 0]
    elif num < 0:
        num = (C * np.conj(C)).real
    den = P1 * P2
    return num / den


def estimate_intrinsic_coherence(C, P1, P2, P1noise, P2noise, Nave):
    """Estimate intrinsic coherence

    Use the iterative procedure from sec. 5 of Ingram 2019

    Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    P1 : float `np.array`
        sub-band periodogram
    P2 : float `np.array`
        reference-band periodogram
    P1noise : float
        Poisson noise level of the sub-band periodogram
    P2noise : float
        Poisson noise level of the reference-band periodogram
    Nave : int
        number of intervals that have been averaged to obtain the input spectra

    """
    new_coherence = np.ones_like(P1)
    old_coherence = np.zeros_like(P1)
    count = 0
    while not np.allclose(new_coherence, old_coherence, atol=0.01) and count < 40:
        # TODO: make it only iterate over the places at low coherence
        old_coherence = new_coherence
        bsq = bias_term(C, P1, P2, P1noise, P2noise, Nave, intrinsic_coherence=new_coherence)
        # old_coherence = new_coherence
        den = (P1 - P1noise) * (P2 - P2noise)
        num = (C * C.conj()).real - bsq
        num[num < 0] = (C * C.conj()).real[num < 0]
        new_coherence = num / den
        count += 1

    return new_coherence


def error_on_averaged_cross_spectrum(C, Ps, Pr, Nave, Psnoise, Prnoise, common_ref="False"):
    """Error on cross spectral quantities, From Ingram 2019.

    Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    Ps : float `np.array`
        sub-band periodogram
    Pr : float `np.array`
        reference-band periodogram
    Psnoise : float
        Poisson noise level of the sub-band periodogram
    Prnoise : float
        Poisson noise level of the reference-band periodogram
    Nave : int
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
    twoN = 2 * Nave
    if common_ref:
        Gsq = (C * C.conj()).real
        bsq = bias_term(C, Ps, Pr, Psnoise, Prnoise, Nave)
        frac = (Gsq - bsq) / (Pr - Prnoise)
        PoN = Pr / twoN

        # Eq. 18
        dRe = dIm = dG = np.sqrt(PoN * (Ps - frac))
        # Eq. 19
        dphi = np.sqrt(PoN * (Ps / (Gsq - bsq) - 1 / (Pr - Prnoise)))

    else:
        PrPs = Pr * Ps
        dRe = np.sqrt((PrPs + C.real ** 2 - C.imag ** 2) / twoN)
        dIm = np.sqrt((PrPs - C.real ** 2 + C.imag ** 2) / twoN)
        gsq = raw_coherence(C, Ps, Pr, Psnoise, Prnoise, Nave)
        dphi = np.sqrt((1 - gsq) / (2 * gsq ** 2 * Nave))
        dG = np.sqrt(PrPs / Nave)

    return dRe, dIm, dphi, dG


def cross_to_covariance(C, Pr, Prnoise, delta_nu):
    """Convert a cross spectrum into a covariance.
     Parameters
    ----------
    C : complex `np.array`
        cross spectrum
    Pr : float `np.array`
        reference-band periodogram
    Prnoise : float
        Poisson noise level of the reference-band periodogram
    delta_nu : float or `np.array`
        spectral resolution. Can be a float, or an array if the spectral
        resolution is not constant throughout the periodograms

    """
    return C * np.sqrt(delta_nu / (Pr - Prnoise))


def _which_segment_idx_fun(counts=None, dt=None):
    # Make function interface equal (counts gets ignored)
    if counts is None:
        return generate_indices_of_segment_boundaries_unbinned

    def fun(*args, **kwargs):
        return generate_indices_of_segment_boundaries_binned(*args, dt=dt, **kwargs)
    return fun


def get_total_ctrate(times, gti, segment_size, counts=None):
    """Calculate the average count rate during the observation.

    This function finds the same segments that the PDS will use and
    returns the mean count rate.
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

    Examples
    --------
    >>> times = np.sort(np.random.uniform(0, 1000, 1000))
    >>> gti = np.asarray([[0, 1000]])
    >>> counts, _ = np.histogram(times, bins=np.linspace(0, 1000, 11))
    >>> bin_times = np.arange(50, 1000, 100)
    >>> get_total_ctrate(bin_times, gti, 1000, counts=counts)
    1.0
    >>> get_total_ctrate(times, gti, 1000)
    1.0
    """
    Nph = 0
    Nintvs = 0
    func = _which_segment_idx_fun(counts)

    for _, _, idx0, idx1 in func(times, gti, segment_size):
        if counts is None:
            Nph += idx1 - idx0
        else:
            Nph += np.sum(counts[idx0:idx1])
        Nintvs += 1

    return Nph / (Nintvs * segment_size)


def get_flux_iterable_from_segments(times, gti, segment_size, N=None, counts=None, errors=None):
    """Get fluxes from different segments of the observation.

    If ``counts`` is ``None``, the input times are interpreted as events.
    At least one between ``N`` and ``counts`` needs to be specified.
    Otherwise, they are assumed uniformly binned inside each GTI, and the number
    of events per bin is taken from ``counts``

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
    N : int, default None
        Number of bins to divide the ``segment_size`` in
    counts : float `np.array`, default None
        Array of counts per bin

    Returns
    -------
    cts : `np.array`
        Array of counts
    err : `np.array`
        (optional) if ``errors`` is None, an array of errors in the segment

    """
    if counts is None and N is None:
        raise ValueError(
            "At least one between counts (if light curve) and N (if events) has to be set"
        )

    dt = None
    if counts is not None:
        dt = np.median(np.diff(times[:100]))

    fun = _which_segment_idx_fun(counts, dt)

    for s, e, idx0, idx1 in fun(times, gti, segment_size):
        if idx1 - idx0 < 2:
            yield None
            continue
        if counts is None:
            event_times = times[idx0:idx1]
            # counts, _ = np.histogram(event_times - s, bins=bins)
            # astype here serves to avoid integer rounding issues in Windows,
            # where long is a 32-bit integer.
            cts = histogram((event_times - s).astype(float), bins=N,
                            range=[0, segment_size]).astype(float)
            cts = np.array(cts)
        else:
            cts = counts[idx0:idx1].astype(float)
            if errors is not None:
                cts = cts, errors[idx0:idx1]

        yield cts


def avg_pds_from_iterable(flux_iterable, dt, norm="abs", use_common_mean=True, silent=False):
    """Calculate the average periodogram from an iterable of light curves

    Parameters
    ----------
    flux_iterable : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal segments of light curve, or of light curve
        and errors. They must all be of the same length.
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "abs"
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

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    pds : `np.array`
        The normalized periodogram powers
    N : int
        the number of bins in the light curves used in each segment
    M : int
        the number of averaged periodograms
    mean : float
        the mean counts per bin
    """
    local_show_progress = show_progress
    if silent:

        def local_show_progress(a):
            return a

    # Initialize stuff
    cross = None
    M = 0

    sum_of_photons = 0
    common_variance = None
    for flux in local_show_progress(flux_iterable):
        if flux is None:
            continue

        # If the iterable returns the uncertainty, use it to calculate the variance.
        variance = None
        if isinstance(flux, tuple):
            flux, err = flux
            variance = np.mean(err) ** 2

        # Calculate the FFT
        N = flux.size
        ft = fft(flux)

        nph = flux.sum()
        unnorm_power = (ft * ft.conj()).real

        # Accumulate the sum of means and variances, to get the final mean and
        # variance the end
        sum_of_photons += nph

        if variance is not None:
            common_variance = \
                sum_if_not_none_or_initialize(common_variance, variance)

        # In the first loop, define the frequency and the freq. interval > 0
        if cross is None:
            fgt0 = positive_fft_bins(N)
            freq = fftfreq(N, dt)[fgt0]

        # No need for the negative frequencies
        unnorm_power = unnorm_power[fgt0]

        # If the user wants to normalize using the mean of the total lightcurve,
        # normalize it here
        cs_seg = unnorm_power
        if not use_common_mean:
            mean = nph / N

            cs_seg = normalize_periodograms(
                unnorm_power, dt, N, mean, norm=norm, variance=variance,
            )

        # Accumulate the total sum cross spectrum
        cross = sum_if_not_none_or_initialize(cross, cs_seg)
        M += 1

    # If there were no good intervals, return None
    if cross is None:
        return None

    # Calculate the mean number of photons per chunk
    Nph = sum_of_photons / M
    # Calculate the mean number of photons per bin
    common_mean = Nph / N

    if common_variance is not None:
        # Note: the variances we summed were means, not sums. Hence M, not M*N
        common_variance /= M

    # Transform a sum into the average
    unnorm_cross = cross / M

    # Final normalization (If not done already!)
    if use_common_mean:
        cross = normalize_periodograms(
            unnorm_cross, dt, N, common_mean, norm=norm, variance=common_variance
        )

    results = Table()
    results["freq"] = freq
    results["power"] = cross
    results["unnorm_power"] = unnorm_cross
    results.meta.update({"n": N, "m": M, "dt": dt,
                         "norm": norm,
                         "df": 1 / (dt * N),
                         "nphots": Nph,
                         "mean": common_mean,
                         "segment_size": dt * N})

    return results


def avg_cs_from_iterables_quick(
    flux_iterable1,
    flux_iterable2,
    dt,
    norm="abs"
):
    """Like `avg_cs_from_iterables`, with default options that make it quick.

    Assumes that:

    * the flux iterables return counts, no other units
    * the mean is calculated over the whole light curve, and normalization
      is done at the end
    * no auxiliary PDSs are returned
    * only positive frequencies are returned
    * the spectrum is complex, no real parts or absolutes
    * no progress bars

    Parameters
    ----------
    flux_iterable1 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal segments of light curve, or of light curve
        and errors. They must all be of the same length.
    flux_iterable2 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Same as ``flux_iterable1``, for the reference channel
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "abs"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    power : `np.array`
        The normalized cross spectral powers
    N : int
        the number of bins in the light curves used in each segment
    M : int
        the number of averaged periodograms
    mean : float
        the mean flux (geometrical average of the mean fluxes in the two channels)

    """
    # Initialize stuff
    unnorm_cross = unnorm_pds1 = unnorm_pds2 = None
    M = 0

    sum_of_photons1 = sum_of_photons2 = 0

    for flux1, flux2 in zip(flux_iterable1, flux_iterable2):
        if flux1 is None or flux2 is None:
            continue

        N = flux1.size

        # Calculate the sum of each light curve, to calculate the mean
        # This will
        nph1 = flux1.sum()
        nph2 = flux2.sum()

        # At the first loop, we define the frequency array and the range of positive
        # frequency bins (after the first loop, cross will not be None nymore)
        if unnorm_cross is None:
            freq = fftfreq(N, dt)
            fgt0 = positive_fft_bins(N)

        # Calculate the FFTs
        ft1 = fft(flux1)
        ft2 = fft(flux2)

        # Calculate the unnormalized cross spectrum
        unnorm_power = ft1 * ft2.conj()

        # Accumulate the sum to calculate the total mean of the lc
        sum_of_photons1 += nph1
        sum_of_photons2 += nph2

        # Take only positive frequencies
        unnorm_power = unnorm_power[fgt0]

        # Initialize or accumulate final averaged spectrum
        unnorm_cross = sum_if_not_none_or_initialize(unnorm_cross, unnorm_power)

        M += 1

    # If no valid intervals were found, return only `None`s
    if unnorm_cross is None:
        return None

    # Calculate the mean number of photons per chunk
    Nph1 = sum_of_photons1 / M
    Nph2 = sum_of_photons2 / M
    # Calculate the mean number of photons per bin
    common_mean1 = Nph1 / N
    common_mean2 = Nph2 / N
    common_mean = np.sqrt(common_mean1 * common_mean2)

    # Transform the sums into averages
    unnorm_cross /= M

    # Finally, normalize the cross spectrum (only if not already done on an
    # interval-to-interval basis)
    cross = normalize_periodograms(
        unnorm_cross,
        dt,
        N,
        common_mean,
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
    results.meta.update({"n": N, "m": M, "dt": dt,
                         "norm": norm,
                         "df": 1 / (dt * N),
                         "nphots1": Nph1, "nphots2": Nph2,
                         "mean": common_mean,
                         "mean1": common_mean1,
                         "mean2": common_mean2,
                         "power_type": "all",
                         "segment_size": dt * N})

    return results


def avg_cs_from_iterables(
    flux_iterable1,
    flux_iterable2,
    dt,
    norm="abs",
    use_common_mean=True,
    silent=False,
    fullspec=False,
    power_type="all",
    return_auxil=False
):
    """Calculate the average cross spectrum from an iterable of light curves

    Parameters
    ----------
    flux_iterable1 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal segments of light curve, or of light curve
        and errors. They must all be of the same length.
    flux_iterable2 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Same as ``flux_iterable1``, for the reference channel
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "abs"
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

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    power : `np.array`
        The normalized cross spectral powers
    N : int
        the number of bins in the light curves used in each segment
    M : int
        the number of averaged periodograms
    mean : float
        the mean flux (geometrical average of the mean fluxes in the two channels)
    unnorm_pds1 : `np.array`
        The unnormalized auxiliary PDS from channel 1. Only returned if ``return_auxil`` is ``True``
    mean1 : float
        The mean flux in channel 1. Only returned if ``return_auxil`` is ``True``
    unnorm_pds2 : `np.array`
        The unnormalized auxiliary PDS from channel 2. Only returned if ``return_auxil`` is ``True``
    mean2 : float
        The mean flux in channel 2. Only returned if ``return_auxil`` is ``True``
    unnorm_power : `np.array`
        The unnormalized cross spectral power
    """

    local_show_progress = show_progress
    if silent:
        def local_show_progress(a):
            return a

    # Initialize stuff
    cross = unnorm_cross = unnorm_pds1 = unnorm_pds2 = pds1 = pds2 = None
    M = 0

    sum_of_photons1 = sum_of_photons2 = 0
    common_variance1 = common_variance2 = common_variance = None

    for flux1, flux2 in local_show_progress(zip(flux_iterable1, flux_iterable2)):
        if flux1 is None or flux2 is None:
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

        N = flux1.size

        # At the first loop, we define the frequency array and the range of positive
        # frequency bins (after the first loop, cross will not be None nymore)
        if cross is None:
            freq = fftfreq(N, dt)
            fgt0 = positive_fft_bins(N)

        # Calculate the FFTs
        ft1 = fft(flux1)
        ft2 = fft(flux2)

        # Calculate the sum of each light curve, to calculate the mean
        # This will
        nph1 = flux1.sum()
        nph2 = flux2.sum()
        nph = np.sqrt(nph1 * nph2)

        # Calculate the unnormalized cross spectrum
        unnorm_power = ft1 * ft2.conj()
        unnorm_pd1 = unnorm_pd2 = 0

        # If requested, calculate the auxiliary PDSs
        if return_auxil:
            unnorm_pd1 = (ft1 * ft1.conj()).real
            unnorm_pd2 = (ft2 * ft2.conj()).real

        # Accumulate the sum to calculate the total mean of the lc
        sum_of_photons1 += nph1
        sum_of_photons2 += nph2

        # Take only positive frequencies unless the user wants the full spectrum
        if not fullspec:
            unnorm_power = unnorm_power[fgt0]
            if return_auxil:
                unnorm_pd1 = unnorm_pd1[fgt0]
                unnorm_pd2 = unnorm_pd2[fgt0]

        cs_seg = unnorm_power
        p1_seg = unnorm_pd1
        p2_seg = unnorm_pd2

        # If normalization has to be done interval by interval, do it here.
        if not use_common_mean:
            mean = nph / N
            variance = None

            if variance1 is not None:
                variance = np.sqrt(variance1 * variance2)

            cs_seg = normalize_periodograms(
                unnorm_power, dt, N, mean, norm=norm, power_type=power_type, variance=variance
            )
            p1_seg = normalize_periodograms(
                unnorm_pd1, dt, N, mean, norm=norm, power_type=power_type, variance=variance
            )
            p2_seg = normalize_periodograms(
                unnorm_pd2, dt, N, mean, norm=norm, power_type=power_type, variance=variance
            )

        # Initialize or accumulate final averaged spectra
        cross = sum_if_not_none_or_initialize(cross, cs_seg)
        unnorm_cross = sum_if_not_none_or_initialize(unnorm_cross, unnorm_power)

        if return_auxil:
            unnorm_pds1 = sum_if_not_none_or_initialize(unnorm_pds1, unnorm_pd1)
            unnorm_pds2 = sum_if_not_none_or_initialize(unnorm_pds2, unnorm_pd2)
            pds1 = sum_if_not_none_or_initialize(pds1, p1_seg)
            pds2 = sum_if_not_none_or_initialize(pds2, p2_seg)

        M += 1

    # If no valid intervals were found, return only `None`s
    if cross is None:
        return None

    # Calculate the mean number of photons per chunk
    Nph1 = sum_of_photons1 / M
    Nph2 = sum_of_photons2 / M

    # Calculate the common mean number of photons per bin
    common_mean1 = Nph1 / N
    common_mean2 = Nph2 / N
    common_mean = np.sqrt(common_mean1 * common_mean2)

    if common_variance1 is not None:
        # Note: the variances we summed were means, not sums. Hence M, not M*N
        common_variance1 /= M
        common_variance2 /= M
        common_variance = np.sqrt(common_variance1 * common_variance2)

    # Transform the sums into averages
    cross /= M
    unnorm_cross /= M
    if return_auxil:
        unnorm_pds1 /= M
        unnorm_pds2 /= M

    # Finally, normalize the cross spectrum (only if not already done on an
    # interval-to-interval basis)
    if use_common_mean:
        cross = normalize_periodograms(
            unnorm_cross,
            dt,
            N,
            common_mean,
            norm=norm,
            variance=common_variance,
            power_type=power_type,
        )
        if return_auxil:
            pds1 = normalize_periodograms(
                unnorm_pds1,
                dt,
                N,
                common_mean1,
                norm=norm,
                variance=common_variance1,
                power_type=power_type,
            )
            pds2 = normalize_periodograms(
                unnorm_pds2,
                dt,
                N,
                common_mean2,
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
    results.meta.update({"n": N, "m": M, "dt": dt,
                         "norm": norm,
                         "df": 1 / (dt * N),
                         "segment_size": dt * N,
                         "nphots1": Nph1, "nphots2": Nph2,
                         "countrate1": common_mean1 / dt,
                         "countrate2": common_mean2 / dt,
                         "mean": common_mean,
                         "mean1": common_mean1,
                         "mean2": common_mean2,
                         "power_type": power_type,
                         "variance": common_variance,
                         "variance1": common_variance1,
                         "variance2": common_variance2})

    if return_auxil:
        results["pds1"] = pds1
        results["pds2"] = pds2
        results["unnorm_pds1"] = unnorm_pds1
        results["unnorm_pds2"] = unnorm_pds2

    return results


def avg_pds_from_events(
    times,
    gti,
    segment_size,
    dt,
    norm="abs",
    use_common_mean=True,
    silent=False,
    counts=None,
    errors=None,
):
    """Calculate the average periodogram from a list of event times or a light curve.

    If the input is a light curve, the time array needs to be uniformly sampled
    inside GTIs (it can have gaps outside), and the counts need to be passed
    through the ``counts`` array.
    Otherwise, times are interpeted as photon arrival times.

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "abs"
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
    counts : float `np.array`, default None
        Array of counts per bin or fluxes per bin
    errors : float `np.array`, default None
        Array of errors on the counts above

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    pds : `np.array`
        The normalized periodogram powers
    N : int
        the number of bins in the light curves used in each segment
    M : int
        the number of averaged periodograms
    mean : float
        the mean counts per bin
    """
    if segment_size is None:
        segment_size = gti.max() - gti.min()
    N = np.rint(segment_size / dt).astype(int)
    dt = segment_size / N

    flux_iterable = get_flux_iterable_from_segments(
        times, gti, segment_size, N, counts=counts, errors=errors
    )
    cross = avg_pds_from_iterable(
        flux_iterable, dt, norm=norm, use_common_mean=use_common_mean, silent=silent
    )
    if cross is not None:
        cross.meta["gti"] = gti
    return cross


def avg_cs_from_events(
    times1,
    times2,
    gti,
    segment_size,
    dt,
    norm="abs",
    use_common_mean=True,
    fullspec=False,
    silent=False,
    power_type="all",
    counts1=None,
    counts2=None,
    errors1=None,
    errors2=None,
    return_auxil=False,
):
    """Calculate the average cross spectrum from a list of event times or a light curve.

    If the input is a light curve, the time arrays need to be uniformly sampled
    inside GTIs (they can have gaps outside), and the counts need to be passed
    through the ``counts1`` and ``counts2`` arrays.
    Otherwise, times are interpeted as photon arrival times

    Parameters
    ----------
    times1 : float `np.array`
        Array of times in the sub-band
    times2 : float `np.array`
        Array of times in the reference band
    gti : [[gti00, gti01], [gti10, gti11], ...]
        common good time intervals
    segment_size : float
        length of segments
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "abs"
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
    counts1 : float `np.array`, default None
        Array of counts per bin for channel 1
    counts2 : float `np.array`, default None
        Array of counts per bin for channel 2
    errors1 : float `np.array`, default None
        Array of errors on the counts on channel 1
    errors2 : float `np.array`, default None
        Array of errors on the counts on channel 2

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    pds : `np.array`
        The normalized periodogram powers
    N : int
        the number of bins in the light curves used in each segment
    M : int
        the number of averaged periodograms
    """
    if segment_size is None:
        segment_size = gti.max() - gti.min()
    N = np.rint(segment_size / dt).astype(int)
    # adjust dt
    dt = segment_size / N

    flux_iterable1 = get_flux_iterable_from_segments(
        times1, gti, segment_size, N, counts=counts1, errors=errors1
    )
    flux_iterable2 = get_flux_iterable_from_segments(
        times2, gti, segment_size, N, counts=counts2, errors=errors2
    )

    is_events = np.all([val is None for val in (counts1, counts2, errors1, errors2)])

    if (is_events
            and silent
            and use_common_mean
            and power_type == "all"
            and not fullspec
            and not return_auxil):
        results = avg_cs_from_iterables_quick(
            flux_iterable1,
            flux_iterable2,
            dt,
            norm=norm
        )

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
        )
    if results is not None:
        results.meta["gti"] = gti
    return results
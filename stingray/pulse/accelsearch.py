from collections.abc import Iterable
from multiprocessing import Pool
import warnings
import tempfile

import numpy as np
import scipy
from scipy import special
import scipy.signal
from astropy.table import Table
import matplotlib.pyplot as plt
from stingray.loggingconfig import setup_logger

logger = setup_logger()

try:
    from tqdm import tqdm as show_progress
except ImportError:

    def show_progress(a, **kwargs):
        return a


from stingray.pulse.overlapandsave.ols import ols

from ..utils import njit, fft, fftfreq, fftn, ifftn, HAS_PYFFTW
from ..stats import pds_probability, pds_detection_level
from ..gti import create_gti_mask


def convolve_ols(a, b, memout=None):
    """Convolution using overlap-and-save.

    The code for the convolution, as implemented by Ahmed Fasih, is under
    stingray.pulse.overlapandsave

    Mimics scipy.signal.fftconvolve with mode='save'.

    Examples
    --------
    >>> from scipy.signal import fftconvolve
    >>> nx, nh = 21, 7
    >>> x = (np.random.randint(-30, 30, size=(nx, nx)) + 1j *
    ...      np.random.randint(-30, 30, size=(nx, nx)))
    >>> h = (np.random.randint(-20, 20, size=(nh, nh)) + 1j *
    ...      np.random.randint(-20, 20, size=(nh, nh)))
    >>> ref = fftconvolve(x, h, mode='same')
    >>> y = convolve(x, h) # +doctest:ellipsis
    ...
    >>> assert np.allclose(ref, y)
    """

    if isinstance(a, str):
        a = np.lib.format.open_memmap(a)

    return ols(
        a,
        b,
        size=[max(4 * x, int(pow(100000, 1 / len(b.shape)))) for x in b.shape],
        rfftn=fftn,
        irfftn=ifftn,
        out=memout,
    )


def convolve(a, b, mode="ols", memout=None):
    if np.version.version.split(".") <= ["1", "15", "0"]:
        mode = "scipy"

    if mode == "ols":
        return convolve_ols(a, b, memout=memout)

    return scipy.signal.fftconvolve(a, b, mode="same")


@njit()
def pds_from_fft(spectr, nph):
    return (spectr * spectr.conj()).real * 2 / nph


def _create_responses(range_z):
    """Create responses corresponding to different accelerations.

    This is the implementation of Eq. 39 in Ransom, Eikenberry &
    Middleditch 2002. See that paper for details

    Parameters
    ----------
    range_z : int
        List of z values to be used for the calculation.

    Returns
    -------
    responses : list
        List of arrays describing the shape of the response function
        corresponding to each value of ``range_z``.
    """
    logger.info("Creating responses")
    responses = []
    for j, z in enumerate(show_progress(range_z)):
        # fdot = z / T**2
        if np.abs(z) < 0.01:
            responses.append(0)
            continue

        m = np.max([np.abs(int(2 * z)), 40])
        sign = z / np.abs(z)
        absz = np.abs(z)
        factor = sign * 1 / np.sqrt(2 * absz)

        q_ks = np.arange(-m / 2, m / 2 + 1)

        exponentials = np.exp(1j * np.pi * q_ks**2 / z)

        Yks = sign * np.sqrt(2 / absz) * q_ks
        Zks = sign * np.sqrt(2 / absz) * (q_ks + z)
        # print(Yks, Zks)
        [SZs, CZs] = special.fresnel(Zks.astype(float))
        [SYs, CYs] = special.fresnel(Yks.astype(float))
        weights = SZs - SYs + 1j * (CYs - CZs)
        responses.append(weights * exponentials * factor)
    return responses


def _convolve_with_response(
    A, detlev, freq_intv_to_search, response_and_j, interbin=False, memout=None
):
    """Accelerate the Fourier transform and find pulsations.

    This function convolves the initial Fourier transform with the response
    corresponding to a constant acceleration, and searches for signals
    corresponding to candidate pulsations.

    Parameters
    ----------
    A : complex array
        The initial FT, normalized so that || FT ||^2 are Leahy powers.
    response_and_j : tuple
        Tuple containing the response matrix corresponding to a given
        acceleration and its position in the list of responses allocated
        at the start of the procedure in ``accelsearch``.
    detlev : float
        The power level considered good for detection
    freq_intv_to_search : bool array
        Mask for ``A``, showing all spectral bins that should be searched
        for pulsations. Note that we use the full array to calculate the
        convolution with the responses, but only these bins to search for
        pulsations. Had we filtered the frequencies before the convolution,
        we would be sure to introduce boundary effects in the "Good"
        frequency interval

    Other parameters
    ----------------
    interbin : bool
        Calculate interbinning to improve sensitivity to frequencies close
        to the edge of PDS bins
    nproc : int
        Number of processors to be used for parallel computation.

    Returns
    -------
    result : list
        List containing tuples of the kind (r, j, power) where
        r is the frequency in units of 1/ T, j is the index of the
        acceleration response used and power is the spectral power
    """
    response, j = response_and_j
    r_freqs = np.arange(A.size)
    if np.asanyarray(response).size == 1:
        accel = A
    else:
        accel = convolve(A, response, memout=memout)
        # new_size = accel.size
        # diff = new_size - A.size
        # Now uses 'same'
        # accel = accel[diff // 2: diff // 2 + A.size]

    rf = r_freqs[freq_intv_to_search]
    accel = accel[freq_intv_to_search]
    if interbin:
        rf, accel = interbin_fft(rf, accel)

    powers_to_search = (accel * accel.conj()).real

    candidate = powers_to_search > detlev
    rs = rf[candidate]
    cand_powers = powers_to_search[candidate]
    results = []
    for i in range(len(rs)):
        r = rs[i]
        cand_power = cand_powers[i]
        results.append([r, j, cand_power])

    return results


def _calculate_all_convolutions(
    A, responses, freq_intv_to_search, detlev, debug=False, interbin=False, nproc=4
):
    """Accelerate the initial Fourier transform and find pulsations.

    This function convolves the initial Fourier transform with the responses
    corresponding to different amounts of constant acceleration, and searches
    for signals corresponding to candidate pulsations.

    Parameters
    ----------
    A : complex array
        The initial FT, normalized so that || FT ||^2 are Leahy powers.
    responses : list of complex arrays
        List of response functions corresponding to different values of
        constant acceleration.
    freq_intv_to_search : bool array
        Mask for ``A``, showing all spectral bins that should be searched
        for pulsations. Note that we use the full array to calculate the
        convolution with the responses, but only these bins to search for
        pulsations. Had we filtered the frequencies before the convolution,
        we would be sure to introduce boundary effects in the "Good"
        frequency interval
    detlev : float
        The power level considered good for detection

    Other parameters
    ----------------
    debug : bool
        Dump debugging information
    interbin : bool
        Calculate interbinning to improve sensitivity to frequencies close
        to the edge of PDS bins
    nproc : int
        Number of processors to be used for parallel computation.

    Returns
    -------
    candidate_rs: array of float
        Frequency of candidates in units of r = 1 / T
    candidate_js: array of float
        Index of the response used
    candidate_powers: array of float
        Power of candidates
    """
    logger.info("Convolving FFT with responses...")
    candidate_powers = [0.0]
    candidate_rs = [1]

    candidate_js = [2]
    # print(responses)
    len_responses = len(responses)
    # if debug:
    #     fobj = open('accelsearch_dump.dat', 'w')

    _, memmapfname = tempfile.mkstemp(suffix=".npy")
    memout = np.lib.format.open_memmap(memmapfname, mode="w+", dtype=A.dtype, shape=A.shape)

    from functools import partial

    func = partial(
        _convolve_with_response, A, detlev, freq_intv_to_search, interbin=interbin, memout=memout
    )

    if nproc == 1:
        results = []
        for j in show_progress(range(len_responses)):
            results.append(func((responses[j], j)))
    else:
        with Pool(processes=nproc) as pool:
            results = list(
                show_progress(
                    pool.imap_unordered(func, [(responses[j], j) for j in range(len_responses)]),
                    total=len_responses,
                )
            )
        pool.close()

    for res in results:
        for subr in res:
            candidate_powers.append(subr[2])
            candidate_rs.append(subr[0])
            candidate_js.append(subr[1])

    # if debug:
    #     fobj.close()
    return candidate_rs[1:], candidate_js[1:], candidate_powers[1:]


def accelsearch(
    times,
    signal,
    delta_z=1,
    fmin=1,
    fmax=1e32,
    gti=None,
    zmax=100,
    candidate_file=None,
    ref_time=0,
    debug=False,
    interbin=False,
    nproc=4,
    det_p_value=0.15,
    fft_rescale=None,
):
    """Find pulsars with accelerated search.

    The theory behind these methods is described in Ransom+02, AJ 124, 1788.

    Parameters
    ----------
    times : array of floats
        An evenly spaced list of times
    signal : array of floats
        The light curve, in counts; same length as ``times``

    Other parameters
    ----------------
    delta_z : float
        The spacing in ``z`` space (delta_z = 1 -> delta_fdot = 1/T**2)
    fmin : float, default 1.
        Minimum frequency to search
    fmax : float, default 1e32
        Maximum frequency to search
    gti : ``[[gti00, gti01], [gti10, gti11], ...]``, default None
        Good Time Intervals. If None, it assumes the full range
        ``[[time[0] - dt / 2 -- time[-1] + dt / 2]]``
    zmax : int, default 100
        Maximum frequency derivative to search (pos and neg), in bins.
        It corresponds to ``fdot_max = zmax / T**2``, where ``T`` is the
        length of the observation.
    candidate_file : str, default None
        Save the final candidate table to this file. If None, the table
        is just returned and not saved.
    ref_time : float, default 0
        Reference time for the times
    det_p_value : float, default 0.015
        Detection p-value (tail probability of noise powers, corrected for the
        number of trials)
    fft_rescale : function
        Any function to apply to the initial FFT, normalized by the number of
        photons as FT * np.sqrt(2/nph) so that || FT ||^2 are Leahy powers.
        For example, a filter to flatten the spectrum in the presence of strong
        red noise.

    Returns
    -------
    candidate_table: :class:`Table`
        Table containing the candidate frequencies and frequency derivatives,
        the spectral power in Leahy normalization, the detection probability,
        the time and the observation length.

    """
    times = np.asanyarray(times)
    signal = np.asanyarray(signal)

    dt = times[1] - times[0]
    n_photons = np.sum(signal)
    if gti is not None and isinstance(gti, Iterable) and len(gti) > 1:
        warnings.warn(
            "Data contain multiple GTIs. Bad time intervals will be "
            "filled with the mean of the signal."
        )
        gti = np.asanyarray(gti)
        # Fill in the data with a constant outside GTIs
        gti_mask = create_gti_mask(times, gti)
        expo_fraction = np.count_nonzero(gti_mask) / len(gti_mask)
        bti_mask = ~gti_mask
        mean_ops = np.mean
        if np.mean(signal) > 10:
            mean_ops = np.median
        signal[bti_mask] = mean_ops(signal[gti_mask])
    else:
        expo_fraction = 1
        gti = np.array([[times[0] - dt / 2, times[-1] + dt / 2]])

    spectr = fft(signal) * np.sqrt(2 / n_photons)
    freq = fftfreq(len(spectr), dt)

    if debug:
        _good_f = freq > 0
        fig = plt.figure(figsize=(12, 8))
        plt.plot(freq[_good_f], (spectr * spectr.conj()).real[_good_f], label="initial PDS")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (Leahy)")
        plt.loglog()

    if fft_rescale is not None:
        logger.info("Applying initial filters...")
        spectr = fft_rescale(spectr)

    if debug:
        plt.plot(
            freq[_good_f],
            (spectr * spectr.conj()).real[_good_f],
            label="PDS after filtering (if any)",
        )
        fname = candidate_file + "_initial_spec.png" if candidate_file else "initial_spec.png"
        plt.legend(loc=2)
        del _good_f
        plt.savefig(fname)
        plt.close(fig)

    T = times[-1] - times[0] + dt

    freq_intv_to_search = (freq >= fmin) & (freq < fmax)
    logger.info("Starting search over full plane...")
    start_z = -zmax
    end_z = zmax
    range_z = np.arange(start_z, end_z, delta_z)
    logger.info("min and max r_dot: {}--{}".format(delta_z / T**2, np.max(range_z) / T**2))
    freqs_to_search = freq[freq_intv_to_search]

    candidate_table = Table(
        names=[
            "time",
            "length",
            "frac_exposure",
            "power",
            "prob",
            "frequency",
            "fdot",
            "fddot",
            "ntrial",
        ],
        dtype=[float] * 8 + [int],
    )

    detlev = pds_detection_level(ntrial=freqs_to_search.size, epsilon=det_p_value)

    responses = _create_responses(range_z)

    candidate_rs, candidate_js, candidate_powers = _calculate_all_convolutions(
        spectr, responses, freq_intv_to_search, detlev, debug=debug, interbin=interbin, nproc=nproc
    )

    for r, j, cand_power in zip(candidate_rs, candidate_js, candidate_powers):
        z = range_z[j]
        cand_freq = r / T
        fdot = z / T**2
        prob = pds_probability(cand_power, freqs_to_search.size)
        candidate_table.add_row(
            [
                ref_time + gti[0, 0],
                T,
                expo_fraction,
                cand_power,
                prob,
                cand_freq,
                fdot,
                0,
                freqs_to_search.size,
            ]
        )

    if candidate_file is not None:
        candidate_table.write(candidate_file + ".csv", overwrite=True)

    return candidate_table


def interbin_fft(freq, fft):
    """Interbinning, a la van der Klis 1989.

    Allows to recover some sensitivity in a power density spectrum when
    the pulsation frequency is close to a bin edge. Here we oversample
    the Fourier transform that will be used to calculate the PDS, adding
    intermediate bins with the following values:

    A_{k+1/2} = \\pi /4 (A_k - A_{k + 1})

    Please note: The new bins are not statistically independent from the
    rest. Please use simulations to estimate the correct detection
    levels.

    Parameters
    ----------
    freq : array of floats
        The frequency array
    fft : array of complex numbers
        The Fourier Transform

    Returns
    new_freqs : array of floats, twice the length of the original array
        The new frequency array
    new_fft : array of complex numbers
        The interbinned Fourier Transform.

    Examples
    --------
    >>> import numpy as np
    >>> freq = [0, 0.5, 1, -1, -0.5]
    >>> fft = np.array([1, 0, 1, 1, 0], dtype=float)
    >>> f, F = interbin_fft(freq, fft)
    >>> assert np.allclose(f, [0, 0.25, 0.5, 0.75, 1, -1, -0.75, -0.5, -0.25])
    >>> pi_4 = np.pi / 4
    >>> assert np.allclose(F, [1, -pi_4, 0, pi_4, 1, 1, -pi_4, 0, pi_4])
    """
    import numpy as np

    freq = np.asanyarray(freq)
    fft = np.asanyarray(fft)

    neglast = freq[-1] < 0
    if neglast:
        order = np.argsort(freq)
        freq = freq[order]
        fft = fft[order]

    N = freq.size

    new_N = 2 * N - 1

    new_freqs = np.linspace(freq[0], freq[-1], new_N)

    new_fft = np.zeros(new_N, dtype=type(fft[0]))
    new_fft[::2] = fft
    new_fft[1::2] = (fft[1:] - fft[:-1]) * np.pi / 4

    if neglast:
        fneg = new_freqs < 0
        fpos = ~fneg
        new_freqs = np.concatenate((new_freqs[fpos], new_freqs[fneg]))
        new_fft = np.concatenate((new_fft[fpos], new_fft[fneg]))

    return new_freqs, new_fft

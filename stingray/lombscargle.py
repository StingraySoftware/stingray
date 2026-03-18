import copy
from collections.abc import Iterable
import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import scipy
import scipy.stats
from astropy.timeseries.periodograms import LombScargle

from .crossspectrum import Crossspectrum
from .events import EventList
from .exceptions import StingrayError
from .fourier import (
    impose_symmetry_lsft,
    lsft_fast,
    lsft_slow,
    get_rms_from_unnorm_periodogram,
    poisson_level,
    unnormalize_periodograms,
)
from .lightcurve import Lightcurve
from .utils import simon


__all__ = ["LombScarglePowerspectrum", "LombScargleCrossspectrum"]


def _autofrequency(min_freq=None, max_freq=None, df=None, dt=None, length=None, nyquist_factor=1):
    """Decide the frequency grid for the periodogram if not provided explicitly.

    Parameters
    ----------
    min_freq : float
        Minimum frequency to take the Lomb-Scargle Fourier Transform
    max_freq : float
        Maximum frequency to take the Lomb-Scargle Fourier Transform
    df : float
        The frequency resolution of the final periodogram. Defaults to 1 / length.
    dt : float
        The time resolution of the light curve.
    length : float
        The total length of the light curve.

    Returns
    -------
    freq : numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    Examples
    --------
    >>> freqs = _autofrequency(min_freq=0.1, max_freq=0.5, df=0.1)
    >>> assert np.allclose(freqs, [0.1, 0.2, 0.3, 0.4, 0.5])
    >>> freqs = _autofrequency(min_freq=0.1, max_freq=0.5, length=10)
    >>> assert np.allclose(freqs, [0.1, 0.2, 0.3, 0.4, 0.5])
    >>> freqs = _autofrequency(min_freq=0.1, dt=1, length=10)
    >>> assert np.allclose(freqs, [0.1, 0.2, 0.3, 0.4, 0.5])
    >>> freqs = _autofrequency(max_freq=0.5, df=0.2)
    >>> assert np.allclose(freqs, [0.1, 0.3, 0.5])
    """

    if (df is None or df <= 0) and length is None:
        raise ValueError("Either df or length must be specified.")
    elif df is None or df <= 0:
        df = 1 / length

    if max_freq is None and (dt is None or dt == 0):
        raise ValueError("Either max_freq or dt must be specified.")
    elif max_freq is None:
        max_freq = nyquist_factor * 0.5 / dt

    if min_freq is None:
        min_freq = df / 2
    elif min_freq <= 0:
        warnings.warn("min_freq must be positive and >0. Setting to df / 2.")
        min_freq = df / 2

    freq = np.arange(min_freq, max_freq + df, df)
    return freq


class LombScargleCrossspectrum(Crossspectrum):
    main_array_attr = "freq"
    type = "crossspectrum"
    """
    Make a cross spectrum from an unevenly sampled (binned) light curve.
    You can also make an empty :class:`LombScargleCrossspectrum` object to populate with your
    own Fourier-transformed data (this can sometimes be useful when making
    binned power spectra).

    Parameters
    ----------
    data1: :class:`stingray.lightcurve.Lightcurve` or :class:`stingray.events.EventList`, optional, default ``None``
        The dataset for the first channel/band of interest.

    data2: :class:`stingray.lightcurve.Lightcurve` or :class:`stingray.events.EventList`, optional, default ``None``
        The dataset for the second, or "reference", band.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, string, optional, default ``none``
        The normalization of the cross spectrum.

    power_type: {``real``, ``absolute``, ``all`}, string, optional, default ``all``
        Parameter to choose among complete, real part and magnitude of the cross spectrum.

    fullspec: boolean, optional, default ``False``
        If False, keep only the positive frequencies, or if True, keep all of them .

    Other Parameters
    ----------------
    dt: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where ``data1``, ``data2`` are
        :class:`EventList` objects

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!)

    min_freq : float
        Minimum frequency to take the Lomb-Scargle Fourier Transform

    max_freq: float
        Maximum frequency to take the Lomb-Scargle Fourier Transform

    df : float
        The frequency resolution of the final periodogram.

    method : str
        The method to be used by the Lomb-Scargle Fourier Transformation function. `fast`
        and `slow` are the allowed values. Default is `fast`. fast uses the optimized Press
        and Rybicki O(n*log(n))

    oversampling : float, optional, default: 5
        Interpolation Oversampling Factor (for the fast algorithm)

    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of cross spectra (complex numbers)

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging more than one spectra). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged cross-spectra amplitudes in each bin.

    n: int
        The number of data points/time bins in one segment of the light
        curves.

    k: array of int
        The rebinning scheme if the object has been rebinned otherwise is set to 1.

    nphots1: float
        The total number of photons in light curve 1

    nphots2: float
        The total number of photons in light curve 2

    References
    ----------
    .. [1] Scargle, J. D. , "Studies in astronomical time series analysis. III - Fourier
        transforms, autocorrelation functions, and cross-correlation
        functions of unevenly spaced data". ApJ 1:343, p874-887, 1989
    .. [2] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis
        of unevenly sampled data". ApJ 1:338, p277, 1989
    """

    def __init__(
        self,
        data1: Optional[Union[EventList, Lightcurve]] = None,
        data2: Optional[Union[EventList, Lightcurve]] = None,
        norm: Optional[str] = "none",
        power_type: Optional[str] = "all",
        dt: Optional[float] = None,
        fullspec: Optional[bool] = False,
        skip_checks: bool = False,
        min_freq: float = None,
        max_freq: float = None,
        df: float = None,
        method: str = "fast",
        oversampling: int = 5,
    ):
        self._type = None

        if data1 is None and data2 is None:
            self._initialize_empty()
            return

        if dt is None:
            if isinstance(data1, Lightcurve) or isinstance(data2, EventList):
                dt = data1.dt
            elif isinstance(data2, Lightcurve) or isinstance(data2, EventList) and dt is None:
                dt = data2.dt

        if not skip_checks:
            good_input = self.initial_checks(
                data1=data1,
                data2=data2,
                norm=norm,
                power_type=power_type,
                dt=dt,
                fullspec=fullspec,
                min_freq=min_freq,
                max_freq=max_freq,
                df=df,
                method=method,
                oversampling=oversampling,
            )

        if data1 is not None and data2 is not None:
            self._initialize_from_any_input(
                data1,
                data2,
                dt=dt,
                norm=norm,
                power_type=power_type,
                fullspec=fullspec,
                min_freq=min_freq,
                max_freq=max_freq,
                df=None,
                method=method,
                oversampling=oversampling,
            )

    def initial_checks(
        self,
        data1,
        data2,
        norm,
        power_type,
        dt,
        min_freq,
        max_freq,
        fullspec,
        df,
        method,
        oversampling,
    ):
        if not isinstance(norm, str):
            raise TypeError("norm must be a string")

        if not isinstance(power_type, str):
            raise TypeError("power_type must be a string")

        if norm.lower() not in ["frac", "abs", "leahy", "none"]:
            raise ValueError("norm must be one of ['frac','abs','leahy','none']")

        if power_type not in ["all", "absolute", "real"]:
            raise ValueError("power_type must be one of ['all','absolute','real']")

        if data1 is None or data2 is None:
            if data1 is not None or data2 is not None:
                raise ValueError("You can't do a cross spectrum with just one lightcurve")

        if min_freq is not None and min_freq < 0:
            raise ValueError("min_freq must be non-negative")

        if max_freq is not None and max_freq < 0:
            raise ValueError("max_freq must be non-negative")

        if max_freq is not None and min_freq is not None:
            if max_freq <= min_freq:
                raise ValueError("max_freq must be greater than min_freq")

        if method not in ["fast", "slow"]:
            raise ValueError("method must be one of ['fast','slow']")

        if not isinstance(oversampling, int):
            raise TypeError("oversampling must be an integer")

        if not isinstance(fullspec, bool):
            raise TypeError("fullspec must be a boolean")

        dt_is_invalid = (dt is None) or (dt <= np.finfo(float).resolution)
        if type(data1) != type(data2):
            raise TypeError("data1 and data2 must be of the same kind")

        if isinstance(data1, EventList):
            if dt_is_invalid:
                raise ValueError(
                    "If using event lists, please specify the bin time to generate lightcurves."
                )
        elif isinstance(data1, Lightcurve):
            if data1.err_dist.lower() != data2.err_dist.lower():
                simon(
                    "Your lightcurves have different statistics."
                    "The errors in the Crossspectrum will be incorrect."
                )
        else:
            raise TypeError("Input data are invalid")
        return True

    def _initialize_from_any_input(
        self,
        data1,
        data2,
        dt,
        norm,
        power_type,
        fullspec,
        min_freq,
        max_freq,
        df,
        method,
        oversampling,
    ):
        """Not required for unevenly sampled data"""
        if isinstance(data1, EventList):
            data1 = data1.to_lc(dt)
        if isinstance(data2, EventList):
            data2 = data2.to_lc(dt)

        self.lc1 = data1.apply_gtis(inplace=False)
        self.lc2 = data2.apply_gtis(inplace=False)

        spec = lscrossspectrum_from_lightcurve(
            self.lc1,
            self.lc2,
            norm=norm,
            power_type=power_type,
            fullspec=fullspec,
            min_freq=min_freq,
            max_freq=max_freq,
            df=df,
            method=method,
            oversampling=oversampling,
        )

        for key, val in spec.__dict__.items():
            setattr(self, key, val)

    def _initialize_empty(self):
        self.freq = None
        self.power = None
        self.power_err = None
        self.unnorm_power = None
        self.unnorm_power_err = None
        self.df = None
        self.dt = None
        self.nphots1 = None
        self.nphots2 = None
        self.m = 1
        self.n = None
        self.fullspec = None
        self.k = 1
        self.err_dist = None
        self.method = None
        self.meancounts1 = None
        self.meancounts2 = None
        self.oversampling = None
        self.variance1 = None
        self.variance2 = None
        self.variance = None
        return

    def time_lag(self):
        super().__doc__
        return self.phase_lag() / (2 * np.pi * self.freq)

    def classical_significances(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'classical_significances' ! Not applicable for unevenly sampled data"
        )

    def from_time_array(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'from_time_array' ! Not applicable for unevenly sampled data"
        )

    def from_events(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'from_events' ! Not applicable for unevenly sampled data"
        )

    def from_lightcurve(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'from_lightcurve' ! Not applicable for unevenly sampled data"
        )

    def from_lc_iterable(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'from_lc_iterable' ! Not applicable for unevenly sampled data"
        )

        if self.df is None:
            self.df = self.freq[1] - self.freq[0]

    def compute_rms(self, min_freq, max_freq, poisson_noise_level=None):
        """
        Compute the fractional rms amplitude in the power spectrum
        between two frequencies.

        Parameters
        ----------
        min_freq: float
            The lower frequency bound for the calculation.

        max_freq: float
            The upper frequency bound for the calculation.

        Other parameters
        ----------------
        poisson_noise_level : float, default is None
            This is the Poisson noise level of the PDS with same
            normalization as the PDS. If poissoin_noise_level is None,
            the Poisson noise is calculated in the idealcase
            e.g. 2./<countrate> for fractional rms normalisation
            Dead time and other instrumental effects can alter it.
            The user can fit the Poisson noise level outside
            this function using the same normalisation of the PDS
            and it will get subtracted from powers here.

        Returns
        -------
        rms: float
            The fractional rms amplitude contained between ``min_freq`` and
            ``max_freq``.

        rms_err: float
            The error on the fractional rms amplitude.

        """
        good = (self.freq >= min_freq) & (self.freq <= max_freq)

        M_freq = self.m
        K_freq = self.k

        if isinstance(self.k, Iterable):
            K_freq = self.k[good]

        if isinstance(self.m, Iterable):
            M_freq = self.m[good]

        if poisson_noise_level is None:
            poisson_noise_unnorm = poisson_level("none", n_ph=self.nphots)
        else:
            poisson_noise_unnorm = unnormalize_periodograms(
                poisson_noise_level, self.dt, self.n, self.nphots, norm=self.norm
            )

        rms, rmse = get_rms_from_unnorm_periodogram(
            self.unnorm_power[good],
            self.nphots,
            self.df * K_freq,
            M=M_freq,
            poisson_noise_unnorm=poisson_noise_unnorm,
            segment_size=None,
            kind="frac",
        )

        return rms, rmse

    def _rms_error(self, powers):
        r"""
        Compute the error on the fractional rms amplitude using error
        propagation.
        Note: this uses the actual measured powers, which is not
        strictly correct. We should be using the underlying power spectrum,
        but in the absence of an estimate of that, this will have to do.

        .. math::

           r = \sqrt{P}

        .. math::

           \delta r = \\frac{1}{2 * \sqrt{P}} \delta P

        Parameters
        ----------
        powers: iterable
            The list of powers used to compute the fractional rms amplitude.

        Returns
        -------
        delta_rms: float
            The error on the fractional rms amplitude.
        """
        nphots = self.nphots
        p_err = scipy.stats.chi2(2.0 * self.m).var() * powers / self.m / nphots

        rms = np.sum(powers) / nphots
        pow = np.sqrt(rms)

        drms_dp = 1 / (2 * pow)

        sq_sum_err = np.sqrt(np.sum(p_err**2))
        delta_rms = sq_sum_err * drms_dp
        return delta_rms


class LombScarglePowerspectrum(LombScargleCrossspectrum):
    type = "powerspectrum"
    """
    Make a :class:`LombScarglePowerspectrum` (also called periodogram) from a unevenly sampled (binned)
    light curve. Periodograms can be normalized by either Leahy normalization,
    fractional rms normalization, absolute rms normalization, or not at all.

    You can also make an empty :class:`LombScarglePowerspectrum` object to populate with
    your own fourier-transformed data (this can sometimes be useful when making
    binned power spectra).

    Parameters
    ----------
    data: :class:`stingray.lightcurve.Lightcurve` or :class:`stingray.events.EventList` object, optional, default ``None``
        The light curve data to be Fourier-transformed.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, string, optional, default ``none``
        The normalization of the power spectrum.

    power_type: {``real``, ``absolute``, ``all`}, string, optional, default ``all``
        Parameter to choose among complete, real part and magnitude of the power spectrum.

    fullspec: boolean, optional, default ``False``
        If False, keep only the positive frequencies, or if True, keep all of them .

    Other Parameters
    ----------------
    dt: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where ``data`` is a
        :class:`EventList` object

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!).

    min_freq : float
        Minimum frequency to take the Lomb-Scargle Fourier Transform

    max_freq: float
        Maximum frequency to take the Lomb-Scargle Fourier Transform

    df : float
        The time resolution of the light curve. Only needed where ``data`` is a :class`stingray.Eventlist` object

    method : str
        The method to be used by the Lomb-Scargle Fourier Transformation function. `fast`
        and `slow` are the allowed values. Default is `fast`. fast uses the optimized Press
        and Rybicki O(n*log(n))

    oversampling : float, optional, default: 5
        Interpolation Oversampling Factor (for the fast algorithm)

    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples.

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes.

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectra of segments of a light curve).
        Note that for a single realization (``m=1``) the error is equal to the
        power.

    df: float
        The frequency resolution.

    m: int
        The number of averaged powers in each bin.

    n: int
        The number of data points in the light curve.

    nphots: float
        The total number of photons in the light curve.
    """

    def __init__(
        self,
        data: Optional[Union[Lightcurve, EventList]] = None,
        norm: Optional[str] = "frac",
        power_type: Optional[str] = "all",
        dt: Optional[float] = None,
        fullspec: Optional[bool] = False,
        skip_checks: Optional[bool] = False,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
        df: Optional[float] = None,
        method: Optional[str] = "fast",
        oversampling: Optional[int] = 5,
    ):
        self._type = None
        if data is None:
            return self._initialize_empty()
        good_input = True
        if not skip_checks:
            good_input = self.initial_checks(
                data,
                data,
                norm,
                power_type,
                dt,
                min_freq,
                max_freq,
                fullspec,
                df,
                method,
                oversampling,
            )

        self._initialize_from_any_input(
            data1=data,
            data2=data,
            dt=dt,
            norm=norm,
            power_type=power_type,
            fullspec=fullspec,
            min_freq=min_freq,
            max_freq=max_freq,
            df=df,
            method=method,
            oversampling=oversampling,
        )
        self.nphots = self.nphots1
        self.dt = dt


def lscrossspectrum_from_lightcurve(
    lc1,
    lc2,
    norm="frac",
    power_type="all",
    fullspec=False,
    min_freq=None,
    max_freq=None,
    df=None,
    method="fast",
    oversampling=5,
    nyquist_factor=1,
):
    """Creates a Lomb Scargle Cross Spectrum from two light curves
    Parameters
    ----------
    lc1: :class:`stingray.lightcurve.Lightcurve` object
        Light curve from channel 1.
    lc2 : :class:`stingray.lightcurve.Lightcurve` object
        Light curve from channel 2.

    Other parameters
    ----------------
    norm : str, default "none"
        The normalization of the periodogram. "frac" is fractional rms,"abs" is absolute
        rms, "leahy" is Leahy normalization, and "none" is the unnormalized periodogram

    power_type : str, default "all"
        Parameter to choose among complete, real part and magnitude of the spectrum

    fullspec : bool, default False
        If False, keep only the positive frequencies, or if True, keep all of them

    min_freq : float, default 0
        Minimum frequency to take the Lomb-Scargle Fourier Transform

    max_freq : float, default None
        Maximum frequency to take the Lomb-Scargle Fourier Transform

    method : str, default "fast"
        The method to be used by the Lomb-Scargle Fourier Transformation function.
        `fast` and `slow` are the allowed values. Default is `fast`. fast uses the
        optimized Press and Rybicki O(n*log(n)) algorithm, while slow uses the original
        O(n^2) algorithm.

    oversampling : int, default 5
        Interpolation Oversampling Factor (for the fast algorithm)
    nyquist_factor : int, default 1
        How many times the Nyquist frequency to use as the maximum frequency
    """
    lscs = LombScargleCrossspectrum()

    length = max(lc1.time[-1], lc2.time[-1]) - min(lc1.time[0], lc2.time[0])
    dt = np.min([lc1.dt, lc2.dt])

    freq = _autofrequency(
        min_freq=min_freq,
        max_freq=max_freq,
        df=df,
        dt=dt,
        length=length,
        nyquist_factor=nyquist_factor,
    )
    freq, cross = _ls_cross(
        lc1,
        lc2,
        freq=freq,
        fullspec=fullspec,
        method=method,
        oversampling=oversampling,
    )
    lscs.unnorm_power = cross
    lscs.freq = freq
    lscs.lc1 = lc1
    lscs.lc2 = lc2
    lscs.norm = norm
    lscs.power_type = power_type
    lscs.fullspec = fullspec
    lscs.min_freq = min_freq
    lscs.max_freq = max_freq
    lscs.oversampling = oversampling
    lscs.nphots1 = lc1.counts.sum()
    lscs.nphots2 = lc2.counts.sum()
    lscs.dt = lc1.dt
    lscs.n = lc1.n
    lscs.method = method
    lscs.err_dist = "poisson"

    if lc1.err_dist == "poisson":
        lscs.variance1 = lc1.meancounts
    else:
        lscs.variance1 = np.mean(lc1.counts_err) ** 2
        lscs.err_dist = "gauss"
    if lc2.err_dist == "poisson":
        lscs.variance2 = lc2.meancounts
    else:
        lscs.variance2 = np.mean(lc2.counts_err) ** 2
        lscs.err_dist = "gauss"

    lscs.power = lscs._normalize_crossspectrum(lscs.unnorm_power)

    if power_type == "real":
        lscs.power = np.real(lscs.power)
        lscs.unnorm_power = np.real(lscs.power)
    elif power_type == "absolute":
        lscs.power = np.abs(lscs.power)
        lscs.unnorm_power = np.abs(lscs.power)
    return lscs


def _ls_cross(
    lc1,
    lc2,
    freq=None,
    fullspec=False,
    method="fast",
    oversampling=5,
):
    """
    Lomb-Scargle Fourier transform the two light curves, then compute the cross spectrum.
    Computed as CS = lc1 x lc2* (where lc2 is the one that gets
    complex-conjugated). The user has the option to either get just the
    positive frequencies or the full spectrum.

    Parameters
    ----------
    lc1: :class:`stingray.lightcurve.Lightcurve` object
        One light curve to be Lomb-Scargle Fourier transformed. This is the band of
        interest or channel of interest.

    lc2: :class:`stingray.lightcurve.Lightcurve` object
        Another light curve to be Fourier transformed.
        This is the reference band.

    fullspec: boolean. Default is False.
        If True, return the whole array of frequencies, or only positive frequencies (False).

    method : str
        The method to be used by the Lomb-Scargle Fourier Transformation function. `fast`
        and `slow` are the allowed values. Default is `fast`. fast uses the optimized Press
        and Rybicki O(n*log(n))

    Returns
    -------
    freq: numpy.ndarray
        The frequency grid at which the LSFT was evaluated

    cross: numpy.ndarray
        The cross spectrum value at each frequency.

    """
    if method == "slow":
        lsft1 = lsft_slow(lc1.counts, lc1.time, freq)
        lsft2 = lsft_slow(lc2.counts, lc2.time, freq)
    elif method == "fast":
        lsft1 = lsft_fast(lc1.counts, lc1.time, freq, oversampling=oversampling)
        lsft2 = lsft_fast(lc2.counts, lc2.time, freq, oversampling=oversampling)
    if fullspec:
        lsft1, _ = impose_symmetry_lsft(lsft1, np.sum((lc1.counts)), lc1.n, freq)
        lsft2, freq = impose_symmetry_lsft(lsft2, np.sum(lc2.counts), lc2.n, freq)
    cross = np.multiply(lsft1, np.conjugate(lsft2))
    return freq, cross

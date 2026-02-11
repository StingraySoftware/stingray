import copy
import warnings
from collections.abc import Iterable, Iterator, Generator

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

from stingray.exceptions import StingrayError
from stingray.utils import rebin_data, rebin_data_log, simon

from .base import StingrayObject
from .events import EventList
from .gti import cross_two_gtis, time_intervals_from_gtis
from .lightcurve import Lightcurve
from .fourier import avg_cs_from_iterables, error_on_averaged_cross_spectrum
from .fourier import avg_cs_from_timeseries, poisson_level
from .fourier import normalize_periodograms, raw_coherence
from .fourier import get_flux_iterable_from_segments
from .fourier import get_rms_from_unnorm_periodogram
from .power_colors import power_color

from scipy.special import factorial


__all__ = [
    "Crossspectrum",
    "AveragedCrossspectrum",
    "DynamicalCrossspectrum",
    "cospectra_pvalue",
    "normalize_crossspectrum",
    "time_lag",
    "coherence",
    "get_flux_generator",
]


def get_flux_generator(data, segment_size, dt=None):
    """Get a flux generator from different segments of a data object

    It is just a wrapper around
    ``stingray.fourier.get_flux_iterable_from_segments``, providing
    this method with the information it needs to create the iterables,
    starting from an event list or a light curve.

    Only accepts `Lightcurve`s and `EventList`s.

    Parameters
    ----------
    data : `Lightcurve` or `EventList`
        Input data
    segment_size : float
        Segment size in seconds

    Other parameters
    ----------------
    dt : float, default None
        Sampling time of the output flux iterables. Required if input data
        is an event list, otherwise the light curve sampling time is selected.

    Returns
    -------
    flux_iterable : ``generator``
        Generator of flux arrays.

    Examples
    --------
    >>> mean = 256
    >>> length = 128
    >>> times = np.sort(np.random.uniform(0, length, int(mean * length)))
    >>> events = EventList(time=times, gti=[[0, length]])
    >>> dt = 0.125
    >>> segment_size = 4

    Create a light curve
    >>> lc = events.to_lc(dt=dt)

    Create a light curve with a different error distribution
    >>> lc_renorm = copy.deepcopy(lc)
    >>> lc_renorm.counts = lc.counts / mean
    >>> lc_renorm.counts_err = lc.counts_err / mean
    >>> lc_renorm.err_dist = "gauss"

    Create an iterable from events, forgetting ``dt``. Should fail
    >>> get_flux_generator(events, segment_size, dt=None)
    Traceback (most recent call last):
    ...
    ValueError: If data is an EventList, you need to specify...

    Create an iterable from events
    >>> iter_ev = get_flux_generator(events, segment_size, dt=dt)

    Create an iterable from the light curve
    >>> iter_lc = get_flux_generator(lc, segment_size, dt=dt)

    Create an iterable from the non-poisson light curve
    >>> iter_lc_nonpois = get_flux_generator(lc_renorm, segment_size, dt=dt)

    Verify that they are equivalent
    >>> for l1, l2 in zip(iter_ev, iter_lc): assert np.allclose(l1, l2)

    Note that the iterable for non-Poissonian light curves also returns the uncertainty
    >>> for l1, (l2, l2e) in zip(iter_lc, iter_lc_nonpois): assert np.allclose(l1, l2 * mean)

    """
    times = data.time
    gti = data.gti

    counts = err = None
    if isinstance(data, Lightcurve):
        counts = data.counts
        N = counts.size
        if data.err_dist.lower() != "poisson":
            err = data.counts_err
    elif isinstance(data, EventList):
        if dt is None:
            raise ValueError("If data is an EventList, you need to specify the bin time dt")
        N = int(np.rint(segment_size / dt))

    flux_iterable = get_flux_iterable_from_segments(
        times, gti, segment_size, N, fluxes=counts, errors=err
    )
    return flux_iterable


def coherence(lc1, lc2):
    """
    Estimate coherence function of two light curves.
    For details on the definition of the coherence, see Vaughan and Nowak,
    1996 [#]_.

    Parameters
    ----------
    lc1: :class:`stingray.Lightcurve` object
        The first light curve data for the channel of interest.
    lc2: :class:`stingray.Lightcurve` object
        The light curve data for reference band

    Returns
    -------
    coh : ``np.ndarray``
        The array of coherence versus frequency

    References
    ----------
    .. [#] https://iopscience.iop.org/article/10.1086/310430
    """

    warnings.warn(
        "The coherence function, as implemented, does not work as expected. "
        "Please use the coherence function of AveragedCrossspectrum, with the "
        "correct parameters.",
        DeprecationWarning,
    )
    if not isinstance(lc1, Lightcurve):
        raise TypeError("lc1 must be a lightcurve.Lightcurve object")

    if not isinstance(lc2, Lightcurve):
        raise TypeError("lc2 must be a lightcurve.Lightcurve object")

    cs = Crossspectrum(lc1, lc2, norm="none")

    return cs.coherence()


def time_lag(lc1, lc2):
    """
    Estimate the time lag of two light curves.
    Calculate time lag and uncertainty.
    Equation from Bendat & Piersol, 2011 [bendat-2011]_.

    Parameters
    ----------
    lc1: :class:`stingray.Lightcurve` object
        The first light curve data for the channel of interest.
    lc2: :class:`stingray.Lightcurve` object
        The light curve data for reference band

    Returns
    -------
    lag : np.ndarray
        The time lag
    lag_err : np.ndarray
        The uncertainty in the time lag

    References
    ----------
    .. [bendat-2011] https://www.wiley.com/en-us/Random+Data%3A+Analysis+and+Measurement+Procedures%2C+4th+Edition-p-9780470248775
    """

    warnings.warn(
        "This standalone time_lag function is deprecated. "
        "Please use the time_lag method of AveragedCrossspectrum, with the "
        "correct parameters.",
        DeprecationWarning,
    )

    if not isinstance(lc1, Lightcurve):
        raise TypeError("lc1 must be a lightcurve.Lightcurve object")

    if not isinstance(lc2, Lightcurve):
        raise TypeError("lc2 must be a lightcurve.Lightcurve object")

    cs = Crossspectrum(lc1, lc2, norm="none")
    lag = cs.time_lag()

    return lag


def normalize_crossspectrum(
    unnorm_power, tseg, nbins, nphots1, nphots2, norm="none", power_type="real"
):
    """
    Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
    fractional rms^2 normalization, or not at all.

    Here for API compatibility purposes. Will be removed in the next
    major release.

    Parameters
    ----------
    unnorm_power: numpy.ndarray
        The unnormalized cross spectrum.

    tseg: int
        The length of the Fourier segment, in seconds.

    nbins : int
        Number of bins in the light curve

    nphots1 : int
        Number of photons in the light curve no. 1

    nphots2 : int
        Number of photons in the light curve no. 2

    Other parameters
    ----------------
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
    warnings.warn(
        "normalize_crossspectrum is now deprecated and will be removed "
        "in the next major release. Please use "
        "stingray.fourier.normalize_periodograms instead.",
        DeprecationWarning,
    )
    dt = tseg / nbins
    nph = np.sqrt(nphots1 * nphots2)
    mean = nph / nbins
    return normalize_periodograms(
        unnorm_power, dt, nbins, mean, n_ph=nph, norm=norm, power_type=power_type
    )


def normalize_crossspectrum_gauss(
    unnorm_power, mean_flux, var, dt, N, norm="none", power_type="real"
):
    """
    Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
    fractional rms^2 normalization, or not at all.

    Here for API compatibility purposes. Will be removed in the next
    major release.

    Parameters
    ----------
    unnorm_power: numpy.ndarray
        The unnormalized cross spectrum.

    mean_flux: float
        The mean flux of the light curve (if a cross spectrum, the geometrical
        mean of the flux in the two channels)

    var: float
        The variance of the light curve (if a cross spectrum, the geometrical
        mean of the variance in the two channels)

    dt: float
        The sampling time of the light curve

    N: int
        The number of bins in the light curve

    Other parameters
    ----------------
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
    warnings.warn(
        "normalize_crossspectrum_gauss is now deprecated and will be "
        "removed in the next major release. Please use "
        "stingray.fourier.normalize_periodograms instead.",
        DeprecationWarning,
    )
    mean = mean_flux * dt
    return normalize_periodograms(
        unnorm_power, dt, N, mean, variance=var, norm=norm, power_type=power_type
    )


def _averaged_cospectra_cdf(xcoord, n):
    """
    Function calculating the cumulative distribution function for
    averaged cospectra, Equation 19 of Huppenkothen & Bachetti (2018).

    Parameters
    ----------
    xcoord : float or iterable
        The cospectral power for which to calculate the CDF.

    n : int
        The number of averaged cospectra

    Returns
    -------
    cdf : float
        The value of the CDF at `xcoord` for `n` averaged cospectra
    """
    if np.size(xcoord) == 1:
        xcoord = [xcoord]

    cdf = np.zeros_like(xcoord)

    for i, x in enumerate(xcoord):
        prefac_bottom1 = factorial(n - 1)
        for j in range(n):
            prefac_top = factorial(n - 1 + j)
            prefac_bottom2 = factorial(n - 1 - j) * factorial(j)
            prefac_bottom3 = 2.0 ** (n + j)

            prefac = prefac_top / (prefac_bottom1 * prefac_bottom2 * prefac_bottom3)

            gf = -j + n

            first_fac = scipy.special.gamma(gf)
            if x >= 0:
                second_fac = scipy.special.gammaincc(gf, n * x) * first_fac
                fac = 2.0 * first_fac - second_fac
            else:
                fac = scipy.special.gammaincc(gf, -n * x) * first_fac

            cdf[i] += prefac * fac
        if np.size(xcoord) == 1:
            return cdf[i]

    return cdf


def cospectra_pvalue(power, nspec):
    """
    This function computes the single-trial p-value that the power was
    observed under the null hypothesis that there is no signal in
    the data.

    Important: the underlying assumption that make this calculation valid
    is that the powers in the cross spectrum follow a Laplace distribution,
    and this requires that:

    1. the co-spectrum is normalized according to [Leahy 1983]_
    2. there is only white noise in the light curve. That is, there is no
       aperiodic variability that would change the overall shape of the power
       spectrum.

    Also note that the p-value is for a *single trial*, i.e. the power
    currently being tested. If more than one power or more than one power
    spectrum are being tested, the resulting p-value must be corrected for the
    number of trials (Bonferroni correction).

    Mathematical formulation in [Huppenkothen 2017]_.

    Parameters
    ----------
    power :  float
        The squared Fourier amplitude of a spectrum to be evaluated

    nspec : int
        The number of spectra or frequency bins averaged in ``power``.
        This matters because averaging spectra or frequency bins increases
        the signal-to-noise ratio, i.e. makes the statistical distributions
        of the noise narrower, such that a smaller power might be very
        significant in averaged spectra even though it would not be in a single
        cross spectrum.

    Returns
    -------
    pval : float
        The classical p-value of the observed power being consistent with
        the null hypothesis of white noise

    References
    ----------

    * .. [Leahy 1983] https://ui.adsabs.harvard.edu/#abs/1983ApJ...266..160L/abstract
    * .. [Huppenkothen 2017] http://adsabs.harvard.edu/abs/2018ApJS..236...13H

    """
    if not np.all(np.isfinite(power)):
        raise ValueError("power must be a finite floating point number!")

    # if power < 0:
    #    raise ValueError("power must be a positive real number!")

    if not np.isfinite(nspec):
        raise ValueError("nspec must be a finite integer number")

    if not np.isclose(nspec % 1, 0):
        raise ValueError("nspec must be an integer number!")

    if nspec < 1:
        raise ValueError("nspec must be larger or equal to 1")

    elif nspec == 1:
        lapl = scipy.stats.laplace(0, 1)
        pval = lapl.sf(power)

    elif nspec > 50:
        exp_sigma = np.sqrt(2) / np.sqrt(nspec)
        gauss = scipy.stats.norm(0, exp_sigma)
        pval = gauss.sf(power)

    else:
        pval = 1.0 - _averaged_cospectra_cdf(power, nspec)

    return pval


class Crossspectrum(StingrayObject):
    main_array_attr = "freq"
    type = "crossspectrum"

    """
    Make a cross spectrum from a (binned) light curve.
    You can also make an empty :class:`Crossspectrum` object to populate with your
    own Fourier-transformed data (this can sometimes be useful when making
    binned power spectra). Stingray uses the scipy.fft standards for the sign
    of the Nyquist frequency.

    Parameters
    ----------
    data1: :class:`stingray.Lightcurve` or :class:`stingray.events.EventList`, optional, default ``None``
        The dataset for the first channel/band of interest.

    data2: :class:`stingray.Lightcurve` or :class:`stingray.events.EventList`, optional, default ``None``
        The dataset for the second, or "reference", band.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, default ``none``
        The normalization of the (real part of the) cross spectrum.

    power_type: string, optional, default ``real``
        Parameter to choose among complete, real part and magnitude of the cross spectrum.

    fullspec: boolean, optional, default ``False``
        If False, keep only the positive frequencies, or if True, keep all of them .

    Other Parameters
    ----------------
    gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time intervals. Defaults to the common GTIs from the two input
        objects. Could throw errors if these GTIs have overlaps with the input
        `Lightcurve` GTIs! If you're getting errors regarding your GTIs, don't
        use this and only give GTIs to the `Lightcurve` objects before making
        the cross spectrum.

    lc1: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects
        For backwards compatibility only. Like ``data1``, but no
        :class:`stingray.events.EventList` objects allowed

    lc2: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects
        For backwards compatibility only. Like ``data2``, but no
        :class:`stingray.events.EventList` objects allowed

    dt: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where ``data1``, ``data2`` are
        :class:`EventList` objects

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!)

    channels_overlap: bool, default False
        If True, the reference band contains all the photons of the subject band.
        This happens, for example, when calculating covariance spectra (see, e.g.,
        the docs for ``CovarianceSpectrum``). This will generally be false in a
        single cross spectrum.

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

    """

    def __init__(
        self,
        data1=None,
        data2=None,
        norm="frac",
        gti=None,
        lc1=None,
        lc2=None,
        power_type="all",
        dt=None,
        fullspec=False,
        skip_checks=False,
        save_all=False,
        channels_overlap=False,
    ):
        self._type = None
        # for backwards compatibility
        if data1 is None:
            data1 = lc1
        if data2 is None:
            data2 = lc2
        self.channels_overlap = channels_overlap
        empty = data1 is None and data2 is None
        good_input = not empty

        if not skip_checks:
            good_input = self.initial_checks(
                data1=data1,
                data2=data2,
                norm=norm,
                gti=gti,
                lc1=lc1,
                lc2=lc2,
                power_type=power_type,
                dt=dt,
                fullspec=fullspec,
            )

        self.dt = dt
        norm = norm.lower()
        self.norm = norm
        self.k = 1

        if empty or not good_input:
            return self._initialize_empty()

        return self._initialize_from_any_input(
            data1,
            data2,
            dt=dt,
            norm=norm,
            power_type=power_type,
            fullspec=fullspec,
            gti=gti,
            save_all=save_all,
        )

    def initial_checks(
        self,
        data1=None,
        data2=None,
        norm="frac",
        gti=None,
        lc1=None,
        lc2=None,
        segment_size=None,
        power_type="real",
        dt=None,
        fullspec=False,
    ):
        """Run initial checks on the input.

        Returns True if checks are passed, False if they are not.

        Raises various errors for different bad inputs

        Examples
        --------
        >>> times = np.arange(0, 10)
        >>> counts = np.random.poisson(100, 10)
        >>> lc1 = Lightcurve(times, counts, skip_checks=True)
        >>> lc2 = Lightcurve(times, counts, skip_checks=True)
        >>> ev1 = EventList(times)
        >>> ev2 = EventList(times)
        >>> c = Crossspectrum()
        >>> ac = AveragedCrossspectrum()

        If norm is not a string, raise a TypeError
        >>> Crossspectrum.initial_checks(c, norm=1)
        Traceback (most recent call last):
        ...
        TypeError: norm must be a string...

        If ``norm`` is not one of the valid norms, raise a ValueError
        >>> Crossspectrum.initial_checks(c, norm="blabla")
        Traceback (most recent call last):
        ...
        ValueError: norm must be 'frac'...

        If ``power_type`` is not one of the valid norms, raise a ValueError
        >>> Crossspectrum.initial_checks(c, power_type="blabla")
        Traceback (most recent call last):
        ...
        ValueError: `power_type` not recognized!

        If the user passes only one light curve, raise a ValueError

        >>> Crossspectrum.initial_checks(c, data1=lc1, data2=None)
        Traceback (most recent call last):
        ...
        ValueError: You can't do a cross spectrum...

        If the user passes an event list without dt, raise a ValueError

        >>> Crossspectrum.initial_checks(c, data1=ev1, data2=ev2, dt=None)
        Traceback (most recent call last):
        ...
        ValueError: If using event lists, please specify...
        """
        if isinstance(norm, str) is False:
            raise TypeError("norm must be a string")

        if norm.lower() not in ["frac", "abs", "leahy", "none"]:
            raise ValueError("norm must be 'frac', 'abs', 'leahy', or 'none'!")

        if power_type not in ["all", "absolute", "real"]:
            raise ValueError("`power_type` not recognized!")

        # check if input data is a Lightcurve object, if not make one or
        # make an empty Crossspectrum object if lc1 == ``None`` or lc2 == ``None``

        if lc1 is not None or lc2 is not None:
            warnings.warn(
                "The lcN keywords are now deprecated. Use dataN instead", DeprecationWarning
            )

        if data1 is None or data2 is None:
            if data1 is not None or data2 is not None:
                raise ValueError("You can't do a cross spectrum with just one light curve!")
            else:
                return False

        dt_is_invalid = (dt is None) or (dt <= np.finfo(float).resolution)

        if segment_size is None:
            # checks to be run for non-averaged spectra
            if gti is not None and len(gti) > 1:
                raise TypeError("Non-averaged cross spectra need a single GTI")

        if type(data1) != type(data2):
            raise TypeError("Input data have to be of the same kind")

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

            # If dt differs slightly, its propagated error must not be more than
            # 1/100th of the bin
            if not np.isclose(data1.dt, data2.dt, rtol=0.01 * data1.dt / data1.tseg):
                raise StingrayError("Light curves do not have same time binning dt.")

            if data1.tseg != data2.tseg:
                simon(
                    "Lightcurves do not have same tseg. This means that the data"
                    "from the two channels are not completely in sync. This "
                    "might or might not be an issue. Keep an eye on it."
                )
        elif isinstance(data1, (list, tuple)):
            if not isinstance(data1[0], Lightcurve) or not isinstance(data2[0], Lightcurve):
                raise TypeError("Inputs lists have to contain light curve objects")

            if data1[0].err_dist.lower() != data2[0].err_dist.lower():
                simon(
                    "Your lightcurves have different statistics."
                    "The errors in the Crossspectrum will be incorrect."
                )
        elif isinstance(data1, (Generator, Iterator)):
            pass
        else:
            raise TypeError("Input data are invalid")

        return True

    def rebin(self, df=None, f=None, method="mean"):
        """
        Rebin the cross spectrum to a new frequency resolution ``df``.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Other Parameters
        ----------------
        f: float
            the rebin factor. If specified, it substitutes df with ``f*self.df``

        Returns
        -------
        bin_cs = :class:`Crossspectrum` (or one of its subclasses) object
            The newly binned cross spectrum or power spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from :class:`AveragedPowerspectrum`, it will return an object of class
            :class:`AveragedPowerspectrum`, too.
        """

        if f is None and df is None:
            raise ValueError("You need to specify at least one between f and df")
        elif f is not None:
            df = f * self.df

        # rebin cross spectrum to new resolution
        binfreq, bincs, binerr, step_size = rebin_data(
            self.freq, self.power, df, self.power_err, method=method, dx=self.df
        )
        # make an empty cross spectrum object
        # note: syntax deliberate to work with subclass Powerspectrum
        bin_cs = copy.copy(self)

        # store the binned periodogram in the new object
        bin_cs.freq = binfreq
        bin_cs.power = bincs
        bin_cs.df = df
        bin_cs.power_err = binerr

        if hasattr(self, "unnorm_power") and self.unnorm_power is not None:
            unnorm_power_err = None
            if hasattr(self, "unnorm_power_err") and self.unnorm_power_err is not None:
                unnorm_power_err = self.unnorm_power_err

            _, binpower_unnorm, binpower_err_unnorm, _ = rebin_data(
                self.freq, self.unnorm_power, df, dx=self.df, yerr=unnorm_power_err, method=method
            )

            if hasattr(self, "unnorm_power_err") and self.unnorm_power_err is not None:
                bin_cs.unnorm_power_err = binpower_err_unnorm

            bin_cs.unnorm_power = binpower_unnorm

        if hasattr(self, "cs_all"):
            cs_all = []
            for c in self.cs_all:
                cs_all.append(
                    rebin_data(self.freq, c, dx_new=df, yerr=None, method=method, dx=self.df)[1]
                )
            bin_cs.cs_all = cs_all
        if hasattr(self, "pds1"):
            bin_cs.pds1 = self.pds1.rebin(df=df, f=f, method=method)
        if hasattr(self, "pds2"):
            bin_cs.pds2 = self.pds2.rebin(df=df, f=f, method=method)

        bin_cs.m = np.rint(step_size * self.m)

        return bin_cs

    def to_norm(self, norm, inplace=False):
        """Convert Cross spectrum to new normalization.

        Parameters
        ----------
        norm : str
            The new normalization of the spectrum

        Other parameters
        ----------------
        inplace: bool, default False
            If True, change the current instance. Otherwise, return a new one

        Returns
        -------
        new_spec : object, same class as input
            The new, normalized, spectrum.
        """
        if norm == self.norm:
            return copy.deepcopy(self)

        variance1 = variance2 = variance = None
        if self.type == "powerspectrum":
            # This is the case for Powerspectrum
            mean = mean1 = mean2 = self.nphots / self.n
            if hasattr(self, "err_dist") and self.err_dist != "poisson":
                variance = self.variance
            nph = self.nphots
        else:
            nph = np.sqrt(self.nphots1 * self.nphots2)
            mean1 = self.nphots1 / self.n
            mean2 = self.nphots2 / self.n
            mean = nph / self.n
            if hasattr(self, "err_dist") and self.err_dist != "poisson":
                variance1 = self.variance1
                variance2 = self.variance2
                variance = np.sqrt(self.variance1 * self.variance2)

        if inplace:
            new_spec = self
        else:
            new_spec = copy.deepcopy(self)

        power_type = "all"
        if hasattr(self, "power_type"):
            power_type = self.power_type

        for attr in ["power", "power_err"]:
            unnorm_attr = "unnorm_" + attr
            if not hasattr(self, unnorm_attr) or getattr(self, unnorm_attr) is None:
                continue
            power = normalize_periodograms(
                getattr(self, unnorm_attr),
                self.dt,
                self.n,
                mean,
                n_ph=nph,
                variance=variance,
                norm=norm,
                power_type=power_type,
            )
            setattr(new_spec, attr, power)
            new_spec.norm = norm
            if hasattr(self, "pds1"):
                p1 = normalize_periodograms(
                    getattr(self.pds1, unnorm_attr),
                    self.dt,
                    self.n,
                    mean1,
                    n_ph=self.nphots1,
                    variance=variance1,
                    norm=norm,
                    power_type=power_type,
                )
                setattr(new_spec.pds1, attr, p1)
                p2 = normalize_periodograms(
                    getattr(self.pds2, unnorm_attr),
                    self.dt,
                    self.n,
                    mean2,
                    n_ph=self.nphots2,
                    variance=variance2,
                    norm=norm,
                    power_type=power_type,
                )
                setattr(new_spec.pds2, attr, p2)
                new_spec.pds1.norm = new_spec.pds2.norm = norm

        return new_spec

    def _normalize_crossspectrum(self, unnorm_power):
        """
        Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
        fractional rms^2 normalization, or not at all.

        Parameters
        ----------
        unnorm_power: numpy.ndarray
            The unnormalized cross spectrum.

        Returns
        -------
        power: numpy.nd.array
            The normalized co-spectrum (real part of the cross spectrum). For
            'none' normalization, imaginary part is returned as well.
        """

        nph = np.sqrt(self.nphots1 * self.nphots2)
        mean = nph / self.n
        variance = None
        if self.err_dist != "poisson":
            variance = np.sqrt(self.variance1 * self.variance2)
        return normalize_periodograms(
            unnorm_power,
            self.dt,
            self.n,
            mean,
            n_ph=nph,
            variance=variance,
            norm=self.norm,
            power_type=self.power_type,
        )

    def rebin_log(self, f=0.01):
        """
        Logarithmic rebin of the periodogram.
        The new frequency depends on the previous frequency
        modified by a factor f:

        .. math::

            d\\nu_j = d\\nu_{j-1} (1+f)

        Parameters
        ----------
        f: float, optional, default ``0.01``
            parameter that steers the frequency resolution


        Returns
        -------
        new_spec : :class:`Crossspectrum` (or one of its subclasses) object
            The newly binned cross spectrum or power spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from :class:`AveragedPowerspectrum`, it will return an object of class
        """

        binfreq, binpower, binpower_err, nsamples = rebin_data_log(
            self.freq, self.power, f, y_err=self.power_err, dx=self.df
        )

        new_spec = copy.copy(self)
        new_spec.freq = binfreq
        new_spec.power = binpower
        new_spec.power_err = binpower_err
        new_spec.m = nsamples * self.m
        new_spec.dt = self.dt
        new_spec.k = nsamples

        if hasattr(self, "unnorm_power") and self.unnorm_power is not None:
            unnorm_power_err = None
            if hasattr(self, "unnorm_power_err") and self.unnorm_power_err is not None:
                unnorm_power_err = self.unnorm_power_err
            _, binpower_unnorm, binpower_err_unnorm, _ = rebin_data_log(
                self.freq, self.unnorm_power, f, dx=self.df, y_err=unnorm_power_err
            )

            new_spec.unnorm_power = binpower_unnorm
            if hasattr(self, "unnorm_power_err") and self.unnorm_power_err is not None:
                new_spec.unnorm_power_err = binpower_err_unnorm

        if hasattr(self, "pds1"):
            new_spec.pds1 = self.pds1.rebin_log(f)
        if hasattr(self, "pds2"):
            new_spec.pds2 = self.pds2.rebin_log(f)

        if hasattr(self, "cs_all"):
            cs_all = []

            for c in self.cs_all:
                cs_all.append(rebin_data_log(self.freq, c, f, dx=self.df)[1])
            new_spec.cs_all = cs_all

        return new_spec

    def coherence(self):
        """Compute Coherence function of the cross spectrum.

        Coherence is defined in Vaughan and Nowak, 1996 [#]_.
        It is a Fourier frequency dependent measure of the linear correlation
        between time series measured simultaneously in two energy channels.

        Returns
        -------
        coh : numpy.ndarray
            Coherence function

        References
        ----------
        .. [#] https://iopscience.iop.org/article/10.1086/310430
        """
        # this computes the averaged power spectrum, but using the
        # cross spectrum code to avoid circular imports

        return raw_coherence(
            self.unnorm_power, self.pds1.unnorm_power, self.pds2.unnorm_power, 0, 0, self.n
        )

    def phase_lag(self):
        """Calculate the fourier phase lag of the cross spectrum.

        This is defined as the argument of the complex cross spectrum, and gives
        the delay at all frequencies, in cycles, of one input light curve with respect
        to the other.
        """
        return np.angle(self.unnorm_power)

    def time_lag(self):
        r"""Calculate the fourier time lag of the cross spectrum.
        The time lag is calculated by taking the phase lag :math:`\phi` and

        ..math::

            \tau = \frac{\phi}{\two pi \nu}

        where :math:`\nu` is the center of the frequency bins.
        """
        if self.__class__ in [Crossspectrum, AveragedCrossspectrum]:
            ph_lag = self.phase_lag()

            return ph_lag / (2 * np.pi * self.freq)
        else:
            raise AttributeError("Object has no attribute named 'time_lag' !")

    def save_as_xspec(self, outroot):
        """Save the cross spectrum in a format that can be read by XSPEC.

        Parameters
        ----------
        outroot : str
            The root name of the output files. The method will produce two files,
            one with extension ``.pha`` and one with extension ``.rsp``.
        """
        from .io import save_as_xspec

        if self.type == "powerspectrum":
            save_as_xspec(self.freq, self.df, self.power.real, self.power_err.real, outroot)
        else:
            save_as_xspec(
                self.freq, self.df, self.power.real, self.power_err.real, outroot + "_real"
            )
            save_as_xspec(
                self.freq, self.df, self.power.imag, self.power_err.imag, outroot + "_imag"
            )

    def plot(
        self, labels=None, axis=None, title=None, marker="-", save=False, filename=None, ax=None
    ):
        """
        Plot the amplitude of the cross spectrum vs. the frequency using ``matplotlib``.

        Parameters
        ----------
        labels : iterable, default ``None``
            A list of tuple with ``xlabel`` and ``ylabel`` as strings.

        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of the ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for the``matplotlib.pyplot.axis()`` method.

        title : str, default ``None``
            The title of the plot.

        marker : str, default '-'
            Line style and color of the plot. Line styles and colors are
            combined in a single format string, as in ``'bo'`` for blue
            circles. See ``matplotlib.pyplot.plot`` for more options.

        save : boolean, optional, default ``False``
            If ``True``, save the figure with specified filename.

        filename : str
            File name of the image to save. Depends on the boolean ``save``.

        ax : ``matplotlib.Axes`` object
            An axes object to fill with the cross correlation plot.
        """

        if ax is None:
            fig = plt.figure("crossspectrum")
            ax = fig.add_subplot(1, 1, 1)

        ax2 = None
        if np.any(np.iscomplex(self.power)):
            ax.plot(self.freq, np.abs(self.power), marker, color="k", label="Amplitude")

            ax2 = ax.twinx()
            ax2.tick_params("y", colors="b")
            ax2.plot(
                self.freq, self.power.imag, marker, color="b", alpha=0.5, label="Imaginary Part"
            )

            ax.plot(self.freq, self.power.real, marker, color="r", alpha=0.5, label="Real Part")

            lines, line_labels = ax.get_legend_handles_labels()
            lines2, line_labels2 = ax2.get_legend_handles_labels()
            lines = lines + lines2
            line_labels = line_labels + line_labels2

        else:
            ax.plot(self.freq, np.abs(self.power), marker, color="b")
            lines, line_labels = ax.get_legend_handles_labels()

        xlabel = "Frequency (Hz)"
        ylabel = f"Power ({self.norm})"

        if labels is not None:
            try:
                xlabel = labels[0]
                ylabel = labels[1]

            except IndexError:
                simon("``labels`` must have two labels for x and y axes.")
                # Not raising here because in case of len(labels)==1, only
                # x-axis will be labelled.

        ax.set_xlabel(xlabel)
        if ax2 is not None:
            ax.set_ylabel(ylabel + "-Real")
            ax2.set_ylabel(ylabel + "-Imaginary")
        else:
            ax.set_ylabel(ylabel)

        ax.legend(lines, line_labels, loc="best")

        if axis is not None:
            ax.set_xlim(axis[0:2])
            ax.set_ylim(axis[2:4])
            if ax2 is not None:
                ax2.set_ylim(axis[2:4])
        if title is not None:
            ax.set_title(title)

        if save:
            if filename is None:
                plt.gcf().savefig("spec.png")
            else:
                plt.gcf().savefig(filename)

        return ax

    def classical_significances(self, threshold=1, trial_correction=False):
        """
        Compute the classical significances for the powers in the power
        spectrum, assuming an underlying noise distribution that follows a
        chi-square distributions with 2M degrees of freedom, where M is the
        number of powers averaged in each bin.

        Note that this function will *only* produce correct results when the
        following underlying assumptions are fulfilled:

        1. The power spectrum is Leahy-normalized
        2. There is no source of variability in the data other than the
           periodic signal to be determined with this method. This is important!
           If there are other sources of (aperiodic) variability in the data, this
           method will *not* produce correct results, but instead produce a large
           number of spurious false positive detections!
        3. There are no significant instrumental effects changing the
           statistical distribution of the powers (e.g. pile-up or dead time)

        By default, the method produces ``(index,p-values)`` for all powers in
        the power spectrum, where index is the numerical index of the power in
        question. If a ``threshold`` is set, then only powers with p-values
        *below* that threshold with their respective indices. If
        ``trial_correction`` is set to ``True``, then the threshold will be corrected
        for the number of trials (frequencies) in the power spectrum before
        being used.

        Parameters
        ----------
        threshold : float, optional, default ``1``
            The threshold to be used when reporting p-values of potentially
            significant powers. Must be between 0 and 1.
            Default is ``1`` (all p-values will be reported).

        trial_correction : bool, optional, default ``False``
            A Boolean flag that sets whether the ``threshold`` will be corrected
            by the number of frequencies before being applied. This decreases
            the ``threshold`` (p-values need to be lower to count as significant).
            Default is ``False`` (report all powers) though for any application
            where `threshold`` is set to something meaningful, this should also
            be applied!

        Returns
        -------
        pvals : iterable
            A list of ``(index, p-value)`` tuples for all powers that have p-values
            lower than the threshold specified in ``threshold``.

        """
        if not self.norm == "leahy":
            raise ValueError("This method only works on Leahy-normalized power spectra!")

        if np.size(self.m) == 1:
            # calculate p-values for all powers
            # leave out zeroth power since it just encodes the number of photons!
            pv = np.array([cospectra_pvalue(power, self.m) for power in self.power])
        else:
            pv = np.array([cospectra_pvalue(power, m) for power, m in zip(self.power, self.m)])

        # if trial correction is used, then correct the threshold for
        # the number of powers in the power spectrum
        if trial_correction:
            threshold /= self.power.shape[0]

        # need to add 1 to the indices to make up for the fact that
        # we left out the first power above!
        indices = np.where(pv < threshold)[0]

        pvals = np.vstack([pv[indices], indices])

        return pvals

    @staticmethod
    def from_time_array(
        times1,
        times2,
        dt,
        segment_size=None,
        gti=None,
        norm="none",
        power_type="all",
        silent=False,
        fullspec=False,
        use_common_mean=True,
        channels_overlap=False,
    ):
        """Calculate AveragedCrossspectrum from two arrays of event times.

        Parameters
        ----------
        times1 : `np.array`
            Event arrival times of channel 1
        times2 : `np.array`
            Event arrival times of channel 2
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency)

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be
            averaged. Only relevant (and required) for `AveragedCrossspectrum`.
        gti : [[gti0, gti1], ...]
            Good Time intervals. Defaults to the common GTIs from the two input
            objects. Could throw errors if these GTIs have overlaps with the
            input object GTIs! If you're getting errors regarding your GTIs,
            don't use this and only give GTIs to the input objects before
            making the cross spectrum.
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
        channels_overlap: bool, default False
            If True, the reference band contains all the photons of the subject band.
            This happens, for example, when calculating covariance spectra (see, e.g.,
            the docs for ``CovarianceSpectrum``). This will generally be false in a
            single cross spectrum.
        """

        return crossspectrum_from_time_array(
            times1,
            times2,
            dt,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            power_type=power_type,
            silent=silent,
            fullspec=fullspec,
            use_common_mean=use_common_mean,
            channels_overlap=channels_overlap,
        )

    @staticmethod
    def from_events(
        events1,
        events2,
        dt,
        segment_size=None,
        norm="none",
        power_type="all",
        silent=False,
        fullspec=False,
        use_common_mean=True,
        gti=None,
        channels_overlap=False,
    ):
        """Calculate AveragedCrossspectrum from two event lists

        Parameters
        ----------
        events1 : `stingray.EventList`
            Events from channel 1
        events2 : `stingray.EventList`
            Events from channel 2
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency)

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be averaged.
            Only relevant (and required) for AveragedCrossspectrum
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
        gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time intervals. Defaults to the common GTIs from the two input
            objects. Could throw errors if these GTIs have overlaps with the
            input object GTIs! If you're getting errors regarding your GTIs,
            don't use this and only give GTIs to the input objects before
            making the cross spectrum.
        channels_overlap: bool, default False
            If True, the reference band contains all the photons of the subject band.
            This happens, for example, when calculating covariance spectra (see, e.g.,
            the docs for ``CovarianceSpectrum``). This will generally be false in a
            single cross spectrum.
        """

        return crossspectrum_from_events(
            events1,
            events2,
            dt,
            segment_size=segment_size,
            norm=norm,
            power_type=power_type,
            silent=silent,
            fullspec=fullspec,
            use_common_mean=use_common_mean,
            gti=gti,
            channels_overlap=channels_overlap,
        )

    @staticmethod
    def from_lightcurve(
        lc1,
        lc2,
        segment_size=None,
        norm="none",
        power_type="all",
        silent=False,
        fullspec=False,
        use_common_mean=True,
        gti=None,
        channels_overlap=False,
    ):
        """Calculate AveragedCrossspectrum from two light curves

        Parameters
        ----------
        lc1 : `stingray.Lightcurve`
            Light curve from channel 1
        lc2 : `stingray.Lightcurve`
            Light curve from channel 2

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be averaged.
            Only relevant (and required) for AveragedCrossspectrum
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
        gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time intervals. Defaults to the common GTIs from the two input
            objects. Could throw errors if these GTIs have overlaps with the
            input object GTIs! If you're getting errors regarding your GTIs,
            don't  use this and only give GTIs to the input objects before
            making the cross spectrum.
        channels_overlap: bool, default False
            If True, the reference band contains all the photons of the subject band.
            This happens, for example, when calculating covariance spectra (see, e.g.,
            the docs for ``CovarianceSpectrum``). This will generally be false in a
            single cross spectrum.
        """
        return crossspectrum_from_lightcurve(
            lc1,
            lc2,
            segment_size=segment_size,
            norm=norm,
            power_type=power_type,
            silent=silent,
            fullspec=fullspec,
            use_common_mean=use_common_mean,
            gti=gti,
            channels_overlap=channels_overlap,
        )

    @staticmethod
    def from_stingray_timeseries(
        ts1,
        ts2,
        flux_attr,
        error_flux_attr=None,
        segment_size=None,
        norm="none",
        power_type="all",
        silent=False,
        fullspec=False,
        use_common_mean=True,
        gti=None,
        channels_overlap=False,
    ):
        """Calculate AveragedCrossspectrum from two light curves

        Parameters
        ----------
        ts1 : `stingray.Timeseries`
            Time series from channel 1
        ts2 : `stingray.Timeseries`
            Time series from channel 2
        flux_attr : `str`
            What attribute of the time series will be used.

        Other parameters
        ----------------
        error_flux_attr : `str`
            What attribute of the time series will be used as error bar.
        segment_size : float
            The length, in seconds, of the light curve segments that will be averaged.
            Only relevant (and required) for AveragedCrossspectrum
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
        gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time intervals. Defaults to the common GTIs from the two input
            objects. Could throw errors if these GTIs have overlaps with the
            input object GTIs! If you're getting errors regarding your GTIs,
            don't  use this and only give GTIs to the input objects before
            making the cross spectrum.
        channels_overlap: bool, default False
            If True, the reference band contains all the photons of the subject band.
            This happens, for example, when calculating covariance spectra (see, e.g.,
            the docs for ``CovarianceSpectrum``). This will generally be false in a
            single cross spectrum.
        """
        return crossspectrum_from_timeseries(
            ts1,
            ts2,
            flux_attr=flux_attr,
            error_flux_attr=error_flux_attr,
            segment_size=segment_size,
            norm=norm,
            power_type=power_type,
            silent=silent,
            fullspec=fullspec,
            use_common_mean=use_common_mean,
            gti=gti,
            channels_overlap=channels_overlap,
        )

    @staticmethod
    def from_lc_iterable(
        iter_lc1,
        iter_lc2,
        dt,
        segment_size,
        norm="none",
        power_type="all",
        silent=False,
        fullspec=False,
        use_common_mean=True,
        gti=None,
        channels_overlap=False,
    ):
        """Calculate AveragedCrossspectrum from two light curves

        Parameters
        ----------
        iter_lc1 : iterable of `stingray.Lightcurve` objects or `np.array`
            Light curves from channel 1. If arrays, use them as counts
        iter_lc1 : iterable of `stingray.Lightcurve` objects or `np.array`
            Light curves from channel 2. If arrays, use them as counts
        dt : float
            The time resolution of the light curves
            (sets the Nyquist frequency)

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be averaged.
            Only relevant (and required) for AveragedCrossspectrum
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
        gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time intervals. Defaults to the common GTIs from the two input
            objects. Could throw errors if these GTIs have overlaps with the
            input object GTIs! If you're getting errors regarding your GTIs,
            don't  use this and only give GTIs to the input objects before
            making the cross spectrum.
        save_all : bool, default False
            If True, save the cross spectrum of each segment in the ``cs_all``
            attribute of the output :class:`Crossspectrum` object.
        channels_overlap: bool, default False
            If True, the reference band contains all the photons of the subject band.
            This happens, for example, when calculating covariance spectra (see, e.g.,
            the docs for ``CovarianceSpectrum``). This will generally be false in a
            single cross spectrum.
        """

        return crossspectrum_from_lc_iterable(
            iter_lc1,
            iter_lc2,
            dt,
            segment_size,
            norm=norm,
            power_type=power_type,
            silent=silent,
            fullspec=fullspec,
            use_common_mean=use_common_mean,
            gti=gti,
            channels_overlap=channels_overlap,
        )

    def _initialize_from_any_input(
        self,
        data1,
        data2,
        dt=None,
        segment_size=None,
        norm="frac",
        power_type="all",
        silent=False,
        fullspec=False,
        gti=None,
        use_common_mean=True,
        save_all=False,
    ):
        """Initialize the class, trying to understand the input types.

        The input arguments are the same as ``__init__()``. Based on the type
        of ``data1``, this method will call the appropriate
        ``crossspectrum_from_XXXX`` function, and initialize ``self`` with
        the correct attributes.
        """
        if segment_size is None and isinstance(data1, StingrayObject):
            common_gti = cross_two_gtis(data1.gti, data2.gti)
            data1.gti = common_gti
            data2.gti = common_gti
            data1 = data1.apply_gtis(inplace=False)
            data2 = data2.apply_gtis(inplace=False)

            data1_is_binned = (
                "counts" in data1.array_attrs() or "_counts" in data1.internal_array_attrs()
            )
            data2_is_binned = (
                "counts" in data2.array_attrs() or "_counts" in data2.internal_array_attrs()
            )

            if data1_is_binned and data2_is_binned:
                assert data2.time.size == data1.time.size and np.allclose(
                    data2.time - data1.time, 0
                ), "Time arrays are not the same"
            elif data1_is_binned or data2_is_binned:
                raise ValueError("Please use input data of the same kind")

        if isinstance(data1, EventList):
            spec = crossspectrum_from_events(
                data1,
                data2,
                dt,
                segment_size,
                norm=norm,
                power_type=power_type,
                silent=silent,
                fullspec=fullspec,
                use_common_mean=use_common_mean,
                gti=gti,
                save_all=save_all,
                channels_overlap=self.channels_overlap,
            )
        elif isinstance(data1, Lightcurve):
            spec = crossspectrum_from_lightcurve(
                data1,
                data2,
                segment_size,
                norm=norm,
                power_type=power_type,
                silent=silent,
                fullspec=fullspec,
                use_common_mean=use_common_mean,
                gti=gti,
                save_all=save_all,
                channels_overlap=self.channels_overlap,
            )
            spec.lc1 = data1
            spec.lc2 = data2
        elif isinstance(data1, (tuple, list)):
            dt = data1[0].dt
            # This is a list of light curves.
            spec = crossspectrum_from_lc_iterable(
                data1,
                data2,
                dt,
                segment_size,
                norm=norm,
                power_type=power_type,
                silent=silent,
                fullspec=fullspec,
                gti=gti,
                use_common_mean=use_common_mean,
                save_all=save_all,
                channels_overlap=self.channels_overlap,
            )
        else:  # pragma: no cover
            raise TypeError(f"Bad inputs to Crosssspectrum: {type(data1)}")

        for key, val in spec.__dict__.items():
            setattr(self, key, val)
        return

    def _initialize_empty(self):
        """Set all attributes to None."""
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
        return

    def deadtime_correct(
        self, dead_time, rate, background_rate=0, limit_k=200, n_approx=None, paralyzable=False
    ):
        """
        Correct the power spectrum for dead time effects.

        This correction is based on the formula given in Zhang et al. 2015, assuming
        a constant dead time for all events.
        For more advanced dead time corrections, see the FAD method from `stingray.deadtime.fad`

        Parameters
        ----------
        dead_time: float
            The dead time of the detector.
        rate : float
            Detected source count rate

        Other Parameters
        ----------------
        background_rate : float, default 0
            Detected background count rate. This is important to estimate when deadtime is given by the
            combination of the source counts and background counts (e.g. in an imaging X-ray detector).
        paralyzable: bool, default False
            If True, the dead time correction is done assuming a paralyzable
            dead time. If False, the correction is done assuming a non-paralyzable
            (more common) dead time.
        limit_k : int, default 200
            Limit to this value the number of terms in the inner loops of
            calculations. Check the plots returned by  the `check_B` and
            `check_A` functions to test that this number is adequate.
        n_approx : int, default None
            Number of bins to calculate the model power spectrum. If None, it will use the size of
            the input frequency array. Relatively simple models (e.g., low count rates compared to
            dead time) can use a smaller number of bins to speed up the calculation, and the final
            power values will be interpolated.

        Returns
        -------
        spectrum: :class:`Crossspectrum` or derivative.
            The dead-time corrected spectrum.
        """
        # I put it here to avoid circular imports
        from stingray.deadtime import non_paralyzable_dead_time_model

        if paralyzable:
            raise NotImplementedError("Paralyzable dead time correction is not implemented yet.")

        model = non_paralyzable_dead_time_model(
            self.freq,
            dead_time,
            rate,
            bin_time=self.dt,
            limit_k=limit_k,
            background_rate=background_rate,
            n_approx=n_approx,
        )
        correction = 2 / model
        new_spec = copy.deepcopy(self)
        new_spec.power *= correction

        # Now correct internal attributes for both the spectrum and its possible
        # sub-spectra (PDSs from single channels, dynamical spectra, etc.)
        objects = [new_spec]
        if hasattr(new_spec, "pds1"):
            objects += [new_spec.pds1, new_spec.pds2]

        for obj in objects:
            for attr in ["unnorm_power", "power_err", "unnorm_power_err"]:
                if hasattr(obj, attr):
                    setattr(obj, attr, getattr(obj, attr) * correction)

        for attr in ["cs_all", "unnorm_cs_all"]:
            if hasattr(new_spec, attr):
                dynsp = getattr(new_spec, attr)
                for i, power in enumerate(dynsp):
                    dynsp[i] = power * correction

        return new_spec

    def _calculate_errors(self):
        """Calculate the errors on cross powers and lags.

        Uses different formulas if the reference band contains the photons of the subject band.
        This might happen, for example, when calculating covariance spectra using a large
        reference band.
        See the details in the documentation of
        :func:`stingray.fourier.error_on_averaged_cross_spectrum`.

        Please note that we have dedicated methods for covariance spectra and other variability
        versus energy spectra in `stingray.varenergyspectrum`, even though they only work for
        input event lists at the moment.
        """
        P1noise = poisson_level(norm="none", meanrate=self.countrate1, n_ph=self.nphots1)
        P2noise = poisson_level(norm="none", meanrate=self.countrate2, n_ph=self.nphots2)

        dRe, dIm, dphi, _ = error_on_averaged_cross_spectrum(
            self.unnorm_power,
            self.pds1.unnorm_power,
            self.pds2.unnorm_power,
            self.m,
            P1noise,
            P2noise,
            common_ref=self.channels_overlap,
        )
        bad = np.isnan(dRe) | np.isnan(dIm)

        if np.any(bad):
            warnings.warn(
                "Some error bars in the Averaged Crossspectrum are invalid."
                "Defaulting to sqrt(2 / M) in Leahy norm, rescaled to the appropriate norm."
            )

        Nph = np.sqrt(self.nphots1 * self.nphots2)

        assert self.nph == Nph

        default_err = np.sqrt(2 / self.m) * Nph / 2
        if isinstance(default_err, Iterable):
            default_err = np.array(default_err)[bad]

        dRe[bad] = default_err
        dIm[bad] = default_err

        power_err = dRe + 1.0j * dIm

        self.unnorm_power_err = power_err

        self.power_err = normalize_periodograms(
            power_err, self.dt, self.n, self.mean, n_ph=Nph, variance=self.variance, norm=self.norm
        )

        self.pds1.power_err = self.pds1.power / np.sqrt(self.pds1.m)
        self.pds2.power_err = self.pds2.power / np.sqrt(self.pds2.m)
        self.pds1.unnorm_power_err = self.pds1.unnorm_power / np.sqrt(self.pds1.m)
        self.pds2.unnorm_power_err = self.pds2.unnorm_power / np.sqrt(self.pds2.m)
        self.lag_err = dphi

        assert hasattr(self, "df")
        assert hasattr(self, "dt")


class AveragedCrossspectrum(Crossspectrum):
    type = "crossspectrum"
    """
    Make an averaged cross spectrum from a light curve by segmenting two
    light curves, Fourier-transforming each segment and then averaging the
    resulting cross spectra.

    Parameters
    ----------
    data1: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects OR :class:`stingray.EventList` object
        A light curve from which to compute the cross spectrum. In some cases,
        this would be the light curve of the wavelength/energy/frequency band
        of interest.

    data2: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects OR :class:`stingray.EventList` object
        A second light curve to use in the cross spectrum. In some cases, this
        would be the wavelength/energy/frequency reference band to compare the
        band of interest with.

    segment_size: float
        The size of each segment to average. Note that if the total duration of
        each :class:`Lightcurve` object in ``lc1`` or ``lc2`` is not an
        integer multiple of the ``segment_size``, then any fraction left-over
        at the end of the time series will be lost. Otherwise you introduce
        artifacts.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, default ``none``
        The normalization of the (real part of the) cross spectrum.

    Other Parameters
    ----------------
    gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time intervals. Defaults to the common GTIs from the two input
        objects. Could throw errors if these GTIs have overlaps with the
        input object GTIs! If you're getting errors regarding your GTIs,
        don't  use this and only give GTIs to the input objects before
        making the cross spectrum.

    dt : float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data1 or data2 are of :class:EventList

    power_type: string, optional, default ``all``
         Parameter to choose among complete, real part and magnitude of
         the cross spectrum.

    silent : bool, default False
         Do not show a progress bar when generating an averaged cross spectrum.
         Useful for the batch execution of many spectra

    lc1: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects
        For backwards compatibility only. Like ``data1``, but no
        :class:`stingray.events.EventList` objects allowed

    lc2: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects
        For backwards compatibility only. Like ``data2``, but no
        :class:`stingray.events.EventList` objects allowed

    fullspec: boolean, optional, default ``False``
        If True, return the full array of frequencies, otherwise return just the
        positive frequencies.

    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!)

    use_common_mean: bool
        Averaged cross spectra are normalized in two possible ways: one is by normalizing
        each of the single spectra that get averaged, the other is by normalizing after the
        averaging. If `use_common_mean` is selected, the spectrum will be normalized
        after the average.

    gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time intervals. Defaults to the common GTIs from the two input
        objects. Could throw errors if these GTIs have overlaps with the
        input object GTIs! If you're getting errors regarding your GTIs,
        don't  use this and only give GTIs to the input objects before
        making the cross spectrum.

    channels_overlap: bool, default False
        If True, the reference band contains all the photons of the subject band.
        This happens, for example, when calculating covariance spectra (see, e.g.,
        the docs for ``CovarianceSpectrum``). This will generally be false in a
        single cross spectrum.

    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples.

    power: numpy.ndarray
        The array of cross spectra.

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
        The number of averaged cross spectra.

    n: int
        The number of time bins per segment of light curve.

    nphots1: float
        The total number of photons in the first (interest) light curve.

    nphots2: float
        The total number of photons in the second (reference) light curve.

    gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time intervals.
    """

    def __init__(
        self,
        data1=None,
        data2=None,
        segment_size=None,
        norm="frac",
        gti=None,
        power_type="all",
        silent=False,
        lc1=None,
        lc2=None,
        dt=None,
        fullspec=False,
        save_all=False,
        use_common_mean=True,
        skip_checks=False,
        channels_overlap=False,
    ):
        self._type = None
        # for backwards compatibility
        if data1 is None:
            data1 = lc1
        if data2 is None:
            data2 = lc2

        empty = data1 is None and data2 is None
        good_input = not empty

        if not skip_checks:
            good_input = self.initial_checks(
                data1=data1,
                data2=data2,
                norm=norm,
                gti=gti,
                lc1=lc1,
                lc2=lc2,
                power_type=power_type,
                dt=dt,
                fullspec=fullspec,
                segment_size=segment_size,
            )
        norm = norm.lower()
        self.norm = norm
        self.dt = dt
        self.save_all = save_all
        self.segment_size = segment_size
        self.show_progress = not silent
        self.channels_overlap = channels_overlap

        if empty or not good_input:
            return self._initialize_empty()

        if isinstance(data1, Generator):
            warnings.warn(
                "The averaged Cross spectrum from a generator of "
                "light curves pre-allocates the full list of light "
                "curves, losing all advantage of lazy loading. If it "
                "is important for you, use the "
                "AveragedCrossspectrum.from_lc_iterable static "
                "method, specifying the sampling time `dt`."
            )
            data1 = list(data1)
            data2 = list(data2)

        return self._initialize_from_any_input(
            data1,
            data2,
            dt=dt,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            power_type=power_type,
            silent=silent,
            fullspec=fullspec,
            use_common_mean=use_common_mean,
            save_all=save_all,
        )

    def initial_checks(self, data1, segment_size=None, **kwargs):
        """

        Examples
        --------
        >>> times = np.arange(0, 10)
        >>> ev1 = EventList(times)
        >>> ev2 = EventList(times)
        >>> ac = AveragedCrossspectrum()

        If AveragedCrossspectrum, you need ``segment_size``
        >>> AveragedCrossspectrum.initial_checks(ac, data1=ev1, data2=ev2, dt=1)
        Traceback (most recent call last):
        ...
        ValueError: segment_size must be specified...

        And it needs to be finite!
        >>> AveragedCrossspectrum.initial_checks(ac, data1=ev1, data2=ev2, dt=1., segment_size=np.nan)
        Traceback (most recent call last):
        ...
        ValueError: segment_size must be finite!
        """
        good = Crossspectrum.initial_checks(self, data1, segment_size=segment_size, **kwargs)
        if not good:
            return False
        if isinstance(self, AveragedCrossspectrum) and segment_size is None and data1 is not None:
            raise ValueError("segment_size must be specified")

        if (
            isinstance(self, AveragedCrossspectrum)
            and segment_size is not None
            and not np.isfinite(segment_size)
        ):
            raise ValueError("segment_size must be finite!")
        return True

    def coherence(self):
        """Averaged Coherence function.


        Coherence is defined in Vaughan and Nowak, 1996 [#]_.
        It is a Fourier frequency dependent measure of the linear correlation
        between time series measured simultaneously in two energy channels.

        Compute an averaged Coherence function of cross spectrum by computing
        coherence function of each segment and averaging them. The return type
        is a tuple with first element as the coherence function and the second
        element as the corresponding uncertainty associated with it.

        Note : The uncertainty in coherence function is strictly valid for Gaussian \
               statistics only.

        Returns
        -------
        (coh, uncertainty) : tuple of np.ndarray
            Tuple comprising the coherence function and uncertainty.

        References
        ----------
        .. [#] https://iopscience.iop.org/article/10.1086/310430
        """
        if np.any(self.m < 50):
            simon(
                "Number of segments used in averaging is "
                "significantly low. The result might not follow the "
                "expected statistical distributions."
            )
        c = self.unnorm_power
        p1 = self.pds1.unnorm_power
        p2 = self.pds2.unnorm_power

        meanrate1 = self.nphots1 / self.n / self.dt
        meanrate2 = self.nphots2 / self.n / self.dt

        P1noise = poisson_level(norm="none", meanrate=meanrate1, n_ph=self.nphots1)
        P2noise = poisson_level(norm="none", meanrate=meanrate2, n_ph=self.nphots2)

        coh = raw_coherence(c, p1, p2, P1noise, P2noise, self.n)

        # Calculate uncertainty
        uncertainty = (2**0.5 * coh * (1 - coh)) / (np.sqrt(coh) * self.m**0.5)

        uncertainty[coh == 0] = 0.0

        return (coh, uncertainty)

    def phase_lag(self):
        """Return the fourier phase lag of the cross spectrum."""
        lag = np.angle(self.unnorm_power)
        if not hasattr(self, "lag_err") or self.lag_err.size != lag.size:
            self._calculate_errors()
        return lag, self.lag_err

    def time_lag(self):
        """Calculate time lag and uncertainty.

        Equation from Bendat & Piersol, 2011 [bendat-2011]__.

        Returns
        -------
        lag : np.ndarray
            The time lag

        lag_err : np.ndarray
            The uncertainty in the time lag
        """
        ph_lag, ph_lag_err = self.phase_lag()

        lag = ph_lag / (2 * np.pi * self.freq)
        lag_err = ph_lag_err / (2 * np.pi * self.freq)

        return lag, lag_err


class DynamicalCrossspectrum(AveragedCrossspectrum):
    type = "crossspectrum"
    """
    Create a dynamical cross spectrum, also often called a *spectrogram*.

    This class will divide two :class:`Lightcurve` objects into segments of
    length ``segment_size``, create a cross spectrum for each segment and store
    all powers in a matrix as a function of both time (using the mid-point of
    each segment) and frequency.

    This is often used to trace changes in period of a (quasi-)periodic signal
    over time.

    Parameters
    ----------
    data1 : :class:`stingray.Lightcurve` or :class:`stingray.EventList` object
        The time series or event list from the subject band, or channel, for which
        the dynamical cross spectrum is to be calculated. If :class:`stingray.EventList`, ``dt``
        must be specified as well.

    data2 : :class:`stingray.Lightcurve` or :class:`stingray.EventList` object
        The time series or event list from the reference band, or channel, of the dynamical
        cross spectrum. If :class:`stingray.EventList`, ``dt`` must be specified as well.

    segment_size : float, default 1
         Length of the segment of light curve, default value is 1 (in whatever
         units the ``time`` array in the :class:`Lightcurve`` object uses).

    norm: {"leahy" | "frac" | "abs" | "none" }, optional, default "frac"
        The normaliation of the periodogram to be used.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care, especially if these GTIs have overlaps with the input
        object GTIs! If you're getting errors regarding your GTIs, don't
        use this and only give GTIs to the input object before making
        the cross spectrum.

    sample_time: float
        Compulsory for input :class:`stingray.EventList` data. The time resolution of the
        lightcurve that is created internally from the input event lists. Drives the
        Nyquist frequency.

    Attributes
    ----------
    segment_size: float
        The size of each segment to average. Note that if the total
        duration of each input object in lc is not an integer multiple
        of the ``segment_size``, then any fraction left-over at the end of the
        time series will be lost.

    dyn_ps : np.ndarray
        The matrix of normalized squared absolute values of Fourier
        amplitudes. The axis are given by the ``freq``
        and ``time`` attributes.

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none``}
        The normalization of the periodogram.

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples.

    time: numpy.ndarray
        The array of mid-point times of each interval used for the dynamical
        power spectrum.

    df: float
        The frequency resolution.

    dt: float
        The time resolution of the dynamical spectrum. It is **not** the time resolution of the
        input light curve. It is the integration time of each line of the dynamical power
        spectrum (typically, an integer multiple of ``segment_size``).

    m: int
        The number of averaged powers in each spectral bin (initially 1, it changes after rebinning).
    """

    def __init__(
        self, data1=None, data2=None, segment_size=None, norm="frac", gti=None, sample_time=None
    ):
        self.segment_size = segment_size
        self.sample_time = sample_time
        self.gti = gti
        self.norm = norm

        if segment_size is None and data1 is None and data2 is None:
            self._initialize_empty()
            self.dyn_ps = None
            return

        if segment_size is None or data1 is None or data2 is None:
            raise TypeError("data1, data2, and segment_size must all be specified")

        if isinstance(data1, EventList) and sample_time is None:
            raise ValueError("To pass input event lists, please specify sample_time")
        elif isinstance(data1, Lightcurve):
            sample_time = data1.dt
            if segment_size > data1.tseg:
                raise ValueError(
                    "Length of the segment is too long to create "
                    "any segments of the light curve!"
                )
        if segment_size < 2 * sample_time:
            raise ValueError("Length of the segment is too short to form a light curve!")

        self._make_matrix(data1, data2)

    def _make_matrix(self, data1, data2):
        """
        Create a matrix of powers for each time step and each frequency step.

        Time increases with row index, frequency with column index.

        Parameters
        ----------
        data1 : :class:`Lightcurve` or :class:`EventList`
            The object providing the subject band/channel for the dynamical
            cross spectrum.
        data2 : :class:`Lightcurve` or :class:`EventList`
            The object providing the reference band for the dynamical
            cross spectrum.
        """
        avg = AveragedCrossspectrum(
            data1,
            data2,
            dt=self.sample_time,
            segment_size=self.segment_size,
            norm=self.norm,
            gti=self.gti,
            save_all=True,
        )
        self.dyn_ps = np.array(avg.cs_all).T
        conv = avg.cs_all / avg.unnorm_cs_all
        self.unnorm_conversion = np.nanmean(conv)
        self.freq = avg.freq
        current_gti = avg.gti
        self.nphots1 = avg.nphots1
        self.nphots2 = avg.nphots2

        tstart, _ = time_intervals_from_gtis(current_gti, self.segment_size)

        self.time = tstart + 0.5 * self.segment_size
        self.df = avg.df
        self.dt = self.segment_size
        self.m = 1

    def rebin_frequency(self, df_new, method="average"):
        """
        Rebin the Dynamic Power Spectrum to a new frequency resolution.
        Rebinning is an in-place operation, i.e. will replace the existing
        ``dyn_ps`` attribute.

        While the new resolution does not need to be an integer of the previous frequency
        resolution, be aware that if this is the case, the last frequency bin will be cut
        off by the fraction left over by the integer division

        Parameters
        ----------
        df_new: float
            The new frequency resolution of the dynamical power spectrum.
            Must be larger than the frequency resolution of the old dynamical
            power spectrum!

        method: {"sum" | "mean" | "average"}, optional, default "average"
            This keyword argument sets whether the powers in the new bins
            should be summed or averaged.
        """
        new_dynspec_object = copy.deepcopy(self)
        dynspec_new = []
        for data in self.dyn_ps.T:
            freq_new, bin_counts, bin_err, step = rebin_data(
                self.freq, data, dx_new=df_new, method=method
            )
            dynspec_new.append(bin_counts)

        new_dynspec_object.freq = freq_new
        new_dynspec_object.dyn_ps = np.array(dynspec_new).T
        new_dynspec_object.df = df_new
        new_dynspec_object.m = int(step) * self.m

        return new_dynspec_object

    def rebin_time(self, dt_new, method="average"):
        """
        Rebin the Dynamic Power Spectrum to a new time resolution.

        Note: this is *not* changing the time resolution of the input light
        curve! ``dt`` is the integration time of each line of the dynamical power
        spectrum (typically, an integer multiple of ``segment_size``).

        While the new resolution does not need to be an integer of the previous time
        resolution, be aware that if this is the case, the last time bin will be cut
        off by the fraction left over by the integer division

        Parameters
        ----------
        dt_new: float
            The new time resolution of the dynamical power spectrum.
            Must be larger than the time resolution of the old dynamical power
            spectrum!

        method: {"sum" | "mean" | "average"}, optional, default "average"
            This keyword argument sets whether the powers in the new bins
            should be summed or averaged.

        Returns
        -------
        time_new: numpy.ndarray
            Time axis with new rebinned time resolution.

        dynspec_new: numpy.ndarray
            New rebinned Dynamical Cross Spectrum.
        """
        if dt_new < self.dt:
            raise ValueError("New time resolution must be larger than old time resolution!")

        new_dynspec_object = copy.deepcopy(self)

        dynspec_new = []
        for data in self.dyn_ps:
            time_new, bin_counts, _, step = rebin_data(
                self.time, data, dt_new, method=method, dx=self.dt
            )
            dynspec_new.append(bin_counts)

        new_dynspec_object.time = time_new
        new_dynspec_object.dyn_ps = np.array(dynspec_new)
        new_dynspec_object.dt = dt_new
        new_dynspec_object.m = int(step) * self.m
        return new_dynspec_object

    def rebin_by_n_intervals(self, n, method="average"):
        """
        Rebin the Dynamic Power Spectrum to a new time resolution, by summing contiguous intervals.

        This is different from meth:`DynamicalPowerspectrum.rebin_time` in that it averages ``n``
        consecutive intervals regardless of their distance in time. ``rebin_time`` will instead
        average intervals that are separated at most by a time ``dt_new``.

        Parameters
        ----------
        n: int
            The number of intervals to be combined into one.

        method: {"sum" | "mean" | "average"}, optional, default "average"
            This keyword argument sets whether the powers in the new bins
            should be summed or averaged.

        Returns
        -------
        time_new: numpy.ndarray
            Time axis with new rebinned time resolution.

        dynspec_new: numpy.ndarray
            New rebinned Dynamical Cross Spectrum.
        """
        if not np.issubdtype(type(n), np.integer):
            warnings.warn("n must be an integer. Casting to int")

            n = int(n)

        if n == 1:
            return copy.deepcopy(self)
        if n < 1:
            raise ValueError("n must be >= 1")

        new_dynspec_object = copy.deepcopy(self)

        dynspec_new = []
        time_new = []
        for i, data in enumerate(self.dyn_ps.T):
            if i % n == 0:
                count = 1
                bin_counts = data
                bin_times = self.time[i]
            else:
                count += 1
                bin_counts += data
                bin_times += self.time[i]
            if count == n:
                if method in ["mean", "average"]:
                    bin_counts /= n
                dynspec_new.append(bin_counts)
                bin_times /= n
                time_new.append(bin_times)

        new_dynspec_object.time = time_new
        new_dynspec_object.dyn_ps = np.array(dynspec_new).T
        new_dynspec_object.dt *= n
        new_dynspec_object.m = n * self.m

        return new_dynspec_object

    def trace_maximum(self, min_freq=None, max_freq=None):
        """
        Return the indices of the maximum powers in each segment
        :class:`Powerspectrum` between specified frequencies.

        Parameters
        ----------
        min_freq: float, default ``None``
            The lower frequency bound.

        max_freq: float, default ``None``
            The upper frequency bound.

        Returns
        -------
        max_positions : np.array
            The array of indices of the maximum power in each segment having
            frequency between ``min_freq`` and ``max_freq``.
        """
        if min_freq is None:
            min_freq = np.min(self.freq)
        if max_freq is None:
            max_freq = np.max(self.freq)

        max_positions = []
        for ps in self.dyn_ps.T:
            indices = np.logical_and(self.freq <= max_freq, min_freq <= self.freq)
            max_power = np.max(ps[indices])
            max_positions.append(np.where(ps == max_power)[0][0])

        return np.array(max_positions)

    def shift_and_add(self, f0_list, nbins=100, output_obj_type=AveragedCrossspectrum, rebin=None):
        """Shift and add the dynamical cross spectrum.

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
        output_obj_type : class, default :class:`AveragedCrossspectrum`
            The type of the output object. Can be, e.g. :class:`AveragedCrossspectrum` or
            :class:`AveragedPowerspectrum`.

        Returns
        -------
        output: :class:`AveragedPowerspectrum` or :class:`AveragedCrossspectrum`
            The final averaged power spectrum.

        Examples
        --------
        >>> power_list = [[2, 5, 2, 2, 2], [1, 1, 5, 1, 1], [3, 3, 3, 5, 3]]
        >>> power_list = np.array(power_list).T
        >>> freqs = np.arange(5) * 0.1
        >>> f0_list = [0.1, 0.2, 0.3, 0.4]
        >>> dps = DynamicalCrossspectrum()
        >>> dps.dyn_ps = power_list
        >>> dps.freq = freqs
        >>> dps.df = 0.1
        >>> dps.m = 1
        >>> output = dps.shift_and_add(f0_list, nbins=5)
        >>> assert np.array_equal(output.m, [2, 3, 3, 3, 2])
        >>> assert np.array_equal(output.power, [2. , 2. , 5. , 2. , 1.5])
        >>> assert np.allclose(output.freq, [0.05, 0.15, 0.25, 0.35, 0.45])
        """
        from .fourier import shift_and_add

        final_freqs, final_powers, count = shift_and_add(
            self.freq, self.dyn_ps.T, f0_list, nbins=nbins, M=self.m, df=self.df, rebin=rebin
        )

        output = output_obj_type()
        good = ~np.isnan(final_powers)

        output.freq = final_freqs[good]
        output.power = final_powers[good]
        output.m = count[good]
        output.df = self.df
        output.norm = self.norm
        output.gti = self.gti
        output.segment_size = self.segment_size

        return output

    def power_colors(
        self,
        freq_edges=[1 / 256, 1 / 32, 0.25, 2.0, 16.0],
        freqs_to_exclude=None,
        poisson_power=0,
    ):
        """
        Compute the power colors of the dynamical power spectrum.

        See Heil et al. 2015, MNRAS, 448, 3348

        Parameters
        ----------
        freq_edges: iterable
            The edges of the frequency bins to be used for the power colors.

        freqs_to_exclude : 1-d or 2-d iterable, optional, default None
            The ranges of frequencies to exclude from the calculation of the power color.
            For example, the frequencies containing strong QPOs.
            A 1-d iterable should contain two values for the edges of a single range. (E.g.
            ``[0.1, 0.2]``). ``[[0.1, 0.2], [3, 4]]`` will exclude the ranges 0.1-0.2 Hz and
            3-4 Hz.

        poisson_power : float or iterable, optional, default 0
            The Poisson noise level of the power spectrum. If iterable, it should have the same
            length as ``freq``. (This might apply to the case of a power spectrum with a
            strong dead time distortion)

        Returns
        -------
        pc0: np.ndarray
        pc0_err: np.ndarray
        pc1: np.ndarray
        pc1_err: np.ndarray
            The power colors for each spectrum and their respective errors
        """
        power_colors = []
        if np.iscomplexobj(self.dyn_ps):
            warnings.warn("When using power_colors, complex powers will be cast to real.")

        for ps in self.dyn_ps.T:
            power_colors.append(
                power_color(
                    self.freq,
                    ps.real,
                    freq_edges=freq_edges,
                    freqs_to_exclude=freqs_to_exclude,
                    df=self.df,
                    poisson_power=poisson_power,
                    m=self.m,
                )
            )

        pc0, pc0_err, pc1, pc1_err = np.array(power_colors).T
        return pc0, pc0_err, pc1, pc1_err

    def compute_rms(self, min_freq, max_freq, poisson_noise_level=0):
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
            normalization as the PDS.

        Returns
        -------
        rms: float
            The fractional rms amplitude contained between ``min_freq`` and
            ``max_freq``.

        rms_err: float
            The error on the fractional rms amplitude.

        """
        dyn_ps_unnorm = self.dyn_ps / self.unnorm_conversion
        poisson_noise_unnrom = poisson_noise_level / self.unnorm_conversion
        if not hasattr(self, "nphots"):
            nphots = (self.nphots1 * self.nphots2) ** 0.5
        else:
            nphots = self.nphots

        good = (self.freq >= min_freq) & (self.freq <= max_freq)

        M_freq = self.m
        if isinstance(self.m, Iterable):
            M_freq = self.m[good]

        rmss = []
        rms_errs = []

        for unnorm_powers in dyn_ps_unnorm.T:
            r, re = get_rms_from_unnorm_periodogram(
                unnorm_powers[good],
                nphots,
                self.df,
                M=M_freq,
                poisson_noise_unnorm=poisson_noise_unnrom,
                segment_size=self.segment_size,
                kind="frac",
            )

            rmss.append(r)
            rms_errs.append(re)
        return rmss, rms_errs


def crossspectrum_from_time_array(
    times1,
    times2,
    dt,
    segment_size=None,
    gti=None,
    norm="none",
    power_type="all",
    silent=False,
    fullspec=False,
    use_common_mean=True,
    save_all=False,
    channels_overlap=False,
):
    """Calculate AveragedCrossspectrum from two arrays of event times.

    Parameters
    ----------
    times1 : `np.array`
        Event arrival times of channel 1
    times2 : `np.array`
        Event arrival times of channel 2
    dt : float
        The time resolution of the intermediate light curves
        (sets the Nyquist frequency)

    Other parameters
    ----------------
    segment_size : float
        The length, in seconds, of the light curve segments that will be averaged
    gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time intervals. Defaults to the common GTIs from the two input
        objects
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
    channels_overlap: bool, default False
        If True, the reference band contains all the photons of the subject band.
        This happens, for example, when calculating covariance spectra (see, e.g.,
        the docs for ``CovarianceSpectrum``). This will generally be false in a
        single cross spectrum.

    Returns
    -------
    spec : `AveragedCrossspectrum` or `Crossspectrum`
        The output cross spectrum.
    """
    force_averaged = segment_size is not None
    # Suppress progress bar for single periodogram
    silent = silent or (segment_size is None)
    results = avg_cs_from_timeseries(
        times1,
        times2,
        gti,
        segment_size,
        dt,
        norm=norm,
        use_common_mean=use_common_mean,
        fullspec=fullspec,
        silent=silent,
        power_type=power_type,
        return_auxil=True,
        return_subcs=save_all,
    )

    return _create_crossspectrum_from_result_table(
        results, force_averaged=force_averaged, channels_overlap=channels_overlap
    )


def crossspectrum_from_events(
    events1,
    events2,
    dt,
    segment_size=None,
    norm="none",
    power_type="all",
    silent=False,
    fullspec=False,
    use_common_mean=True,
    gti=None,
    save_all=False,
    channels_overlap=False,
):
    """Calculate AveragedCrossspectrum from two event lists

    Parameters
    ----------
    events1 : `stingray.EventList`
        Events from channel 1
    events2 : `stingray.EventList`
        Events from channel 2
    dt : float
        The time resolution of the intermediate light curves
        (sets the Nyquist frequency)

    Other parameters
    ----------------
    segment_size : float, default None
        The length, in seconds, of the light curve segments that will be averaged
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
    gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time intervals. Defaults to the common GTIs from the two input
        objects
    channels_overlap: bool, default False
        If True, the reference band contains all the photons of the subject band.
        This happens, for example, when calculating covariance spectra (see, e.g.,
        the docs for ``CovarianceSpectrum``). This will generally be false in a
        single cross spectrum.

    Returns
    -------
    spec : `AveragedCrossspectrum` or `Crossspectrum`
        The output cross spectrum.
    """

    if gti is None:
        gti = cross_two_gtis(events1.gti, events2.gti)

    dt = events1.suggest_compatible_dt(dt)

    return crossspectrum_from_time_array(
        events1.time,
        events2.time,
        dt,
        segment_size,
        gti,
        norm=norm,
        power_type=power_type,
        silent=silent,
        fullspec=fullspec,
        use_common_mean=use_common_mean,
        save_all=save_all,
        channels_overlap=channels_overlap,
    )


def crossspectrum_from_lightcurve(
    lc1,
    lc2,
    segment_size=None,
    norm="none",
    power_type="all",
    silent=False,
    fullspec=False,
    use_common_mean=True,
    gti=None,
    save_all=False,
    channels_overlap=False,
):
    """Calculate AveragedCrossspectrum from two light curves

    Parameters
    ----------
    lc1 : `stingray.Lightcurve`
        Light curve from channel 1
    lc2 : `stingray.Lightcurve`
        Light curve from channel 2

    Other parameters
    ----------------
    segment_size : float, default None
        The length, in seconds, of the light curve segments that will be averaged
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
    gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time intervals. Defaults to the common GTIs from the two input
        objects
    save_all : bool, default False
        Save all intermediate spectra used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.
    channels_overlap: bool, default False
        If True, the reference band contains all the photons of the subject band.
        This happens, for example, when calculating covariance spectra (see, e.g.,
        the docs for ``CovarianceSpectrum``). This will generally be false in a
        single cross spectrum.

    Returns
    -------
    spec : `AveragedCrossspectrum` or `Crossspectrum`
        The output cross spectrum.
    """
    error_flux_attr = None

    if lc1.err_dist == "gauss":
        error_flux_attr = "_counts_err"

    return crossspectrum_from_timeseries(
        lc1,
        lc2,
        "_counts",
        error_flux_attr=error_flux_attr,
        segment_size=segment_size,
        norm=norm,
        power_type=power_type,
        silent=silent,
        fullspec=fullspec,
        use_common_mean=use_common_mean,
        gti=gti,
        save_all=save_all,
        channels_overlap=channels_overlap,
    )


def crossspectrum_from_timeseries(
    ts1,
    ts2,
    flux_attr,
    error_flux_attr=None,
    segment_size=None,
    norm="none",
    power_type="all",
    silent=False,
    fullspec=False,
    use_common_mean=True,
    gti=None,
    save_all=False,
    channels_overlap=False,
):
    """Calculate AveragedCrossspectrum from two time series

    Parameters
    ----------
    ts1 : `stingray.StingrayTimeseries`
        Time series from channel 1
    ts2 : `stingray.StingrayTimeseries`
        Time series from channel 2
    flux_attr : `str`
        What attribute of the time series will be used.

    Other parameters
    ----------------
    error_flux_attr : `str`
        What attribute of the time series will be used as error bar.
    segment_size : float, default None
        The length, in seconds, of the light curve segments that will be averaged
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
    gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time intervals. Defaults to the common GTIs from the two input
        objects
    save_all : bool, default False
        Save all intermediate spectra used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.
    channels_overlap: bool, default False
        If True, the reference band contains all the photons of the subject band.
        This happens, for example, when calculating covariance spectra (see, e.g.,
        the docs for ``CovarianceSpectrum``). This will generally be false in a
        single cross spectrum.

    Returns
    -------
    spec : `AveragedCrossspectrum` or `Crossspectrum`
        The output cross spectrum.
    """
    force_averaged = segment_size is not None
    # Suppress progress bar for single periodogram
    silent = silent or (segment_size is None)
    if gti is None:
        gti = cross_two_gtis(ts1.gti, ts2.gti)

    err1 = err2 = None
    if error_flux_attr is not None:
        err1 = getattr(ts1, error_flux_attr)
        err2 = getattr(ts2, error_flux_attr)

    results = avg_cs_from_timeseries(
        ts1.time,
        ts2.time,
        gti,
        segment_size,
        ts1.dt,
        norm=norm,
        use_common_mean=use_common_mean,
        fullspec=fullspec,
        silent=silent,
        power_type=power_type,
        fluxes1=getattr(ts1, flux_attr),
        fluxes2=getattr(ts2, flux_attr),
        errors1=err1,
        errors2=err2,
        return_auxil=True,
        return_subcs=save_all,
    )

    return _create_crossspectrum_from_result_table(
        results, force_averaged=force_averaged, channels_overlap=channels_overlap
    )


def crossspectrum_from_lc_iterable(
    iter_lc1,
    iter_lc2,
    dt,
    segment_size,
    norm="none",
    power_type="all",
    silent=False,
    fullspec=False,
    use_common_mean=True,
    gti=None,
    save_all=False,
    channels_overlap=False,
):
    """Calculate AveragedCrossspectrum from two light curves

    Parameters
    ----------
    iter_lc1 : iterable of `stingray.Lightcurve` objects or `np.array`
        Light curves from channel 1. If arrays, use them as counts
    iter_lc1 : iterable of `stingray.Lightcurve` objects or `np.array`
        Light curves from channel 2. If arrays, use them as counts
    dt : float
        The time resolution of the light curves
        (sets the Nyquist frequency)
    segment_size : float
        The length, in seconds, of the light curve segments that will be averaged

    Other parameters
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
    gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time intervals. The GTIs of the input light curves are
        intersected with these.
    channels_overlap: bool, default False
        If True, the reference band contains all the photons of the subject band.
        This happens, for example, when calculating covariance spectra (see, e.g.,
        the docs for ``CovarianceSpectrum``). This will generally be false in a
        single cross spectrum.

    Returns
    -------
    spec : `AveragedCrossspectrum` or `Crossspectrum`
        The output cross spectrum.
    """

    force_averaged = segment_size is not None
    # Suppress progress bar for single periodogram
    silent = silent or (segment_size is None)

    common_gti = gti

    def iterate_lc_counts(iter_lc):
        for lc in iter_lc:
            if hasattr(lc, "counts"):
                n_bin = np.rint(segment_size / lc.dt).astype(int)

                gti = lc.gti
                if common_gti is not None:
                    gti = cross_two_gtis(common_gti, lc.gti)

                err = None
                if lc.err_dist == "gauss":
                    err = lc.counts_err

                flux_iterable = get_flux_iterable_from_segments(
                    lc.time, gti, segment_size, n_bin, fluxes=lc.counts, errors=err
                )
                for out in flux_iterable:
                    yield out
            else:
                yield lc

    results = avg_cs_from_iterables(
        iterate_lc_counts(iter_lc1),
        iterate_lc_counts(iter_lc2),
        dt,
        norm=norm,
        use_common_mean=use_common_mean,
        silent=silent,
        fullspec=fullspec,
        power_type=power_type,
        return_auxil=True,
        return_subcs=save_all,
    )
    return _create_crossspectrum_from_result_table(
        results, force_averaged=force_averaged, channels_overlap=channels_overlap
    )


def _create_crossspectrum_from_result_table(table, force_averaged=False, channels_overlap=False):
    """Copy the columns and metadata from the results of
    ``stingray.fourier.avg_cs_from_XX`` functions into
    `AveragedCrossspectrum` or `Crossspectrum` objects.

    By default, allocates a Crossspectrum object if the number of
    averaged spectra is 1, otherwise an AveragedCrossspectrum.
    If the user specifies ``force_averaged=True``, it always allocates
    an AveragedCrossspectrum.

    Parameters
    ----------
    table : `astropy.table.Table`
        results of `avg_cs_from_iterables` or `avg_cs_from_iterables_quick`

    Other parameters
    ----------------
    force_averaged : bool, default False

    channels_overlap: bool, default False
        If True, the reference band contains all the photons of the subject band.
        This happens, for example, when calculating covariance spectra (see, e.g.,
        the docs for ``CovarianceSpectrum``). This will generally be false in a
        single cross spectrum.

    Returns
    -------
    spec : `AveragedCrossspectrum` or `Crossspectrum`
        The output cross spectrum.
    """
    if table.meta["m"] > 1 or force_averaged:
        cs = AveragedCrossspectrum()
        cs.pds1 = AveragedCrossspectrum()
        cs.pds2 = AveragedCrossspectrum()
    else:
        cs = Crossspectrum()
        cs.pds1 = Crossspectrum()
        cs.pds2 = Crossspectrum()

    cs.freq = cs.pds1.freq = cs.pds2.freq = np.array(table["freq"])
    cs.norm = cs.pds1.norm = cs.pds2.norm = table.meta["norm"]

    cs.power = np.array(table["power"])
    cs.pds1.power = np.array(table["pds1"])
    cs.pds2.power = np.array(table["pds2"])
    cs.unnorm_power = np.array(table["unnorm_power"])
    cs.pds1.unnorm_power = np.array(table["unnorm_pds1"])
    cs.pds2.unnorm_power = np.array(table["unnorm_pds2"])

    cs.pds1.type = cs.pds2.type = "powerspectrum"

    if "subcs" in table.meta:
        cs.cs_all = np.array(table.meta["subcs"])
        cs.unnorm_cs_all = np.array(table.meta["unnorm_subcs"])

    for attr, val in table.meta.items():
        if not attr.endswith("subcs"):
            setattr(cs, attr, val)
            setattr(cs.pds1, attr, val)
            setattr(cs.pds2, attr, val)

    cs.err_dist = "poisson"
    if cs.variance is not None:
        cs.err_dist = cs.pds1.err_dist = cs.pds2.err_dist = "gauss"

    # Transform nphods1 in nphots for pds1, etc.
    for attr, val in table.meta.items():
        if attr.endswith("1"):
            setattr(cs.pds1, attr[:-1], val)
        if attr.endswith("2"):
            setattr(cs.pds2, attr[:-1], val)

    mean = table.meta["mean"]
    nph = table.meta["nphots"]
    cs.mean = mean
    cs.channels_overlap = channels_overlap

    cs.nph = nph
    cs._calculate_errors()
    return cs

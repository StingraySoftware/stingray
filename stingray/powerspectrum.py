import copy
import warnings
from collections.abc import Generator, Iterable

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

from stingray.crossspectrum import AveragedCrossspectrum, Crossspectrum, DynamicalCrossspectrum
from stingray.stats import pds_probability, amplitude_upper_limit, pds_detection_level

from .events import EventList
from .gti import cross_two_gtis, time_intervals_from_gtis, create_gti_mask

from .lightcurve import Lightcurve
from .fourier import avg_pds_from_iterable, unnormalize_periodograms
from .fourier import avg_pds_from_timeseries
from .fourier import get_flux_iterable_from_segments
from .fourier import poisson_level
from .fourier import get_rms_from_unnorm_periodogram


__all__ = ["Powerspectrum", "AveragedPowerspectrum", "DynamicalPowerspectrum"]


class Powerspectrum(Crossspectrum):
    type = "powerspectrum"
    """
    Make a :class:`Powerspectrum` (also called periodogram) from a (binned)
    light curve. Periodograms can be normalized by either Leahy normalization,
    fractional rms normalization, absolute rms normalization, or not at all.

    You can also make an empty :class:`Powerspectrum` object to populate with
    your own fourier-transformed data (this can sometimes be useful when making
    binned power spectra).

    Parameters
    ----------
    data: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data to be Fourier-transformed.

    norm: {"leahy" | "frac" | "abs" | "none" }, optional, default "frac"
        The normaliation of the power spectrum to be used. Options are
        "leahy", "frac", "abs" and "none", default is "frac".

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care, especially if these GTIs have overlaps with the input
        object GTIs! If you're getting errors regarding your GTIs, don't
        use this and only give GTIs to the input object before making
        the power spectrum.

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!).

    Attributes
    ----------
    norm: {"leahy" | "frac" | "abs" | "none" }
        The normalization of the power spectrum.

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

    unnorm_power: numpy.ndarray
        The array of unnormalized powers

    unnorm_power_err: numpy.ndarray
        The uncertainties of ``unnorm_power``.

    df: float
        The frequency resolution.

    m: int
        The number of averaged powers in each bin.

    n: int
        The number of data points in the light curve.

    nphots: float
        The total number of photons in the light curve.

    """

    def __init__(self, data=None, norm="frac", gti=None, dt=None, lc=None, skip_checks=False):
        self._type = None
        if lc is not None:
            warnings.warn(
                "The lc keyword is now deprecated. Use data " "instead", DeprecationWarning
            )
        if data is None:
            data = lc

        good_input = data is not None
        if good_input and not skip_checks:
            good_input = self.initial_checks(
                data1=data, data2=data, norm=norm, gti=gti, lc1=lc, lc2=lc, dt=dt
            )

        norm = norm.lower()
        self.norm = norm
        self.dt = dt

        if not good_input:
            return self._initialize_empty()

        return self._initialize_from_any_input(data, dt=dt, norm=norm)

    def rebin(self, df=None, f=None, method="mean"):
        """
        Rebin the power spectrum.

        Parameters
        ----------
        df: float
            The new frequency resolution.

        Other Parameters
        ----------------
        f: float
            The rebin factor. If specified, it substitutes ``df`` with
            ``f*self.df``, so ``f>1`` is recommended.

        Returns
        -------
        bin_cs = :class:`Powerspectrum` object
            The newly binned power spectrum.
        """
        bin_ps = Crossspectrum.rebin(self, df=df, f=f, method=method)
        bin_ps.nphots = bin_ps.nphots1

        return bin_ps

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
            segment_size=self.segment_size,
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
           periodic signal to be determined with this method. This is
           important! If there are other sources of (aperiodic) variability in
           the data, this method will *not* produce correct results, but
           instead produce a large number of spurious false positive
           detections!
        3. There are no significant instrumental effects changing the
           statistical distribution of the powers (e.g. pile-up or dead time)

        By default, the method produces ``(index,p-values)`` for all powers in
        the power spectrum, where index is the numerical index of the power in
        question. If a ``threshold`` is set, then only powers with p-values
        *below* that threshold with their respective indices. If
        ``trial_correction`` is set to ``True``, then the threshold will be
        corrected for the number of trials (frequencies) in the power spectrum
        before being used.

        Parameters
        ----------
        threshold : float, optional, default ``1``
            The threshold to be used when reporting p-values of potentially
            significant powers. Must be between 0 and 1.
            Default is ``1`` (all p-values will be reported).

        trial_correction : bool, optional, default ``False``
            A Boolean flag that sets whether the ``threshold`` will be
            corrected by the number of frequencies before being applied. This
            decreases the ``threshold`` (p-values need to be lower to count as
            significant). Default is ``False`` (report all powers) though for
            any application where `threshold`` is set to something meaningful,
            this should also be applied!

        Returns
        -------
        pvals : iterable
            A list of ``(p-value, index)`` tuples for all powers that have
            p-values lower than the threshold specified in ``threshold``.

        """
        if not self.norm == "leahy":
            raise ValueError("This method only works on " "Leahy-normalized power spectra!")

        if trial_correction:
            ntrial = self.power.shape[0]
        else:
            ntrial = 1

        if np.size(self.m) == 1:
            # calculate p-values for all powers
            # leave out zeroth power since it just encodes the number of
            # photons!
            pv = pds_probability(self.power, n_summed_spectra=self.m, ntrial=ntrial)
        else:
            pv = np.array(
                [
                    pds_probability(power, n_summed_spectra=m, ntrial=ntrial)
                    for power, m in zip(self.power, self.m)
                ]
            )

        # need to add 1 to the indices to make up for the fact that
        # we left out the first power above!
        indices = np.where(pv < threshold)[0]

        pvals = np.vstack([pv[indices], indices])

        return pvals

    def modulation_upper_limit(self, fmin=None, fmax=None, c=0.95):
        r"""
        Upper limit on a sinusoidal modulation.

        To understand the meaning of this amplitude: if the modulation is
        described by:

        ..math:: p = \overline{p} (1 + a * \sin(x))

        this function returns a.

        If it is a sum of sinusoidal harmonics instead
        ..math:: p = \overline{p} (1 + \sum_l a_l * \sin(lx))
        a is equivalent to :math:`\sqrt(\sum_l a_l^2)`.

        See `stingray.stats.power_upper_limit`,
        `stingray.stats.amplitude_upper_limit`
        for more information.

        The formula used to calculate the upper limit assumes the Leahy
        normalization.
        If the periodogram is in another normalization, we will internally
        convert it to Leahy before calculating the upper limit.

        Parameters
        ----------
        fmin: float
            The minimum frequency to search (defaults to the first nonzero bin)

        fmax: float
            The maximum frequency to search (defaults to the Nyquist frequency)

        Other Parameters
        ----------------
        c: float
            The confidence value for the upper limit (e.g. 0.95 = 95%)

        Returns
        -------
        a: float
            The modulation amplitude that could produce P>pmeas with 1 - c
            probability.

        Examples
        --------
        >>> pds = Powerspectrum()
        >>> pds.norm = "leahy"
        >>> pds.freq = np.arange(0., 5.)
        >>> # Note: this pds has 40 as maximum value between 2 and 5 Hz
        >>> pds.power = np.array([100000, 1, 1, 40, 1])
        >>> pds.m = 1
        >>> pds.nphots = 30000
        >>> assert np.isclose(
        ...     pds.modulation_upper_limit(fmin=2, fmax=5, c=0.99),
        ...     0.10164,
        ...     atol=0.0001)
        """

        pds = self
        if self.norm != "leahy":
            pds = self.to_norm("leahy")

        freq = pds.freq
        fnyq = np.max(freq)
        power = pds.power
        freq_mask = freq > 0
        if fmin is not None or fmax is not None:
            if fmin is not None:
                freq_mask[freq < fmin] = 0
            if fmax is not None:
                freq_mask[freq > fmax] = 0
        freq = freq[freq_mask]
        power = power[freq_mask]

        maximum_val = np.argmax(power)
        nyq_ratio = freq[maximum_val] / fnyq

        # Since we have averaged powerspectra of segments, we need to set summed_flag to False.
        # Check power_upper_limit/amplitude_upper_limit functions for more details.
        return amplitude_upper_limit(
            power[maximum_val],
            pds.nphots,
            n=pds.m,
            c=c,
            nyq_ratio=nyq_ratio,
            fft_corr=True,
            summed_flag=False,
        )

    @staticmethod
    def from_time_array(
        times, dt, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True
    ):
        """
        Calculate an average power spectrum from an array of event times.

        Parameters
        ----------
        times : `np.array`
            Event arrival times.
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency).

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be
            averaged. Only relevant (and required) for
            ``AveragedPowerspectrum``.
        gti: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
            Additional, optional Good Time intervals that get intersected with
            the GTIs of the input object. Can cause errors if there are
            overlaps between these GTIs and the input object GTIs. If that
            happens, assign the desired GTIs to the input object.
        norm : str, default "frac"
            The normalization of the periodogram. `abs` is absolute rms, `frac`
            is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
            the unnormalized periodogram.
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or
            on the full light curve. This gives different results
            (Alston+2013). By default, we assume the mean is calculated on the
            full light curve, but the user can set ``use_common_mean`` to False
            to calculate it on a per-segment basis.
        silent : bool, default False
            Silence the progress bars.
        """

        return powerspectrum_from_time_array(
            times,
            dt,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
        )

    @staticmethod
    def from_events(
        events,
        dt,
        segment_size=None,
        gti=None,
        norm="frac",
        silent=False,
        use_common_mean=True,
        save_all=False,
    ):
        """
        Calculate an average power spectrum from an event list.

        Parameters
        ----------
        events : `stingray.EventList`
            Event list to be analyzed.
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency).

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be
            averaged. Only relevant (and required) for
            ``AveragedPowerspectrum``.
        gti: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
            Additional, optional Good Time intervals that get intersected with
            the GTIs of the input object. Can cause errors if there are
            overlaps between these GTIs and the input object GTIs. If that
            happens, assign the desired GTIs to the input object.
        norm : str, default "frac"
            The normalization of the periodogram. `abs` is absolute rms, `frac`
            is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
            the unnormalized periodogram.
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or
            on the full light curve. This gives different results
            (Alston+2013). By default, we assume the mean is calculated on the
            full light curve, but the user can set ``use_common_mean`` to False
            to calculate it on a per-segment basis.
        silent : bool, default False
            Silence the progress bars.
        save_all : bool, default False
            Save all intermediate PDSs used for the final average.
        """
        if gti is None:
            gti = events.gti
        return powerspectrum_from_events(
            events,
            dt,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
            save_all=save_all,
        )

    @staticmethod
    def from_stingray_timeseries(
        ts,
        flux_attr,
        error_flux_attr=None,
        segment_size=None,
        norm="none",
        silent=False,
        use_common_mean=True,
        gti=None,
        save_all=False,
    ):
        """Calculate AveragedPowerspectrum from a time series.

        Parameters
        ----------
        ts : `stingray.Timeseries`
            Input Time Series
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
        silent : bool, default False
            Silence the progress bars
        gti: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time intervals. Defaults to the common GTIs from the two input
            objects. Could throw errors if these GTIs have overlaps with the
            input object GTIs! If you're getting errors regarding your GTIs,
            don't  use this and only give GTIs to the input objects before
            making the cross spectrum.
        save_all : bool, default False
            Save all intermediate PDSs used for the final average.
        """
        return powerspectrum_from_timeseries(
            ts,
            flux_attr=flux_attr,
            error_flux_attr=error_flux_attr,
            segment_size=segment_size,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
            gti=gti,
            save_all=save_all,
        )

    @staticmethod
    def from_lightcurve(
        lc,
        segment_size=None,
        gti=None,
        norm="frac",
        silent=False,
        use_common_mean=True,
        save_all=False,
    ):
        """
        Calculate a power spectrum from a light curve.

        Parameters
        ----------
        events : `stingray.Lightcurve`
            Light curve to be analyzed.
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency).

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be
            averaged. Only relevant (and required) for
            ``AveragedPowerspectrum``.
        gti: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
            Additional, optional Good Time intervals that get intersected with
            the GTIs of the input object. Can cause errors if there are
            overlaps between these GTIs and the input object GTIs. If that
            happens, assign the desired GTIs to the input object.
        norm : str, default "frac"
            The normalization of the periodogram. `abs` is absolute rms, `frac`
            is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
            the unnormalized periodogram.
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or
            on the full light curve. This gives different results
            (Alston+2013). By default, we assume the mean is calculated on the
            full light curve, but the user can set ``use_common_mean`` to False
            to calculate it on a per-segment basis.
        silent : bool, default False
            Silence the progress bars.
        save_all : bool, default False
            Save all intermediate PDSs used for the final average.
        """
        if gti is None:
            gti = lc.gti
        return powerspectrum_from_lightcurve(
            lc,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
            save_all=save_all,
        )

    @staticmethod
    def from_lc_iterable(
        iter_lc,
        dt,
        segment_size=None,
        gti=None,
        norm="frac",
        silent=False,
        use_common_mean=True,
        save_all=False,
    ):
        """
        Calculate the average power spectrum of an iterable collection of
        light curves.

        Parameters
        ----------
        iter_lc : iterable of `stingray.Lightcurve` objects or `np.array`
            Light curves. If arrays, use them as counts.
        dt : float
            The time resolution of the light curves
            (sets the Nyquist frequency)

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be
            averaged. Only relevant (and required) for
            ``AveragedPowerspectrum``.
        gti: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
            Additional, optional Good Time intervals that get intersected with
            the GTIs of the input object. Can cause errors if there are
            overlaps between these GTIs and the input object GTIs. If that
            happens, assign the desired GTIs to the input object.
        norm : str, default "frac"
            The normalization of the periodogram. `abs` is absolute rms, `frac`
            is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
            the unnormalized periodogram.
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or
            on the full light curve. This gives different results
            (Alston+2013). By default, we assume the mean is calculated on the
            full light curve, but the user can set ``use_common_mean`` to False
            to calculate it on a per-segment basis.
        silent : bool, default False
            Silence the progress bars.
        save_all : bool, default False
            Save all intermediate PDSs used for the final average.
        """

        return powerspectrum_from_lc_iterable(
            iter_lc,
            dt,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
            save_all=save_all,
        )

    def _initialize_from_any_input(
        self,
        data,
        dt=None,
        segment_size=None,
        gti=None,
        norm="frac",
        silent=False,
        use_common_mean=True,
        save_all=False,
    ):
        """
        Initialize the class, trying to understand the input types.

        The input arguments are the same as ``__init__()``. Based on the type
        of ``data``, this method will call the appropriate
        ``powerspectrum_from_XXXX`` function, and initialize ``self`` with
        the correct attributes.
        """
        if isinstance(data, EventList):
            spec = powerspectrum_from_events(
                data,
                dt,
                segment_size,
                norm=norm.lower(),
                silent=silent,
                use_common_mean=use_common_mean,
                gti=gti,
                save_all=save_all,
            )
        elif isinstance(data, Lightcurve):
            spec = powerspectrum_from_lightcurve(
                data,
                segment_size,
                norm=norm,
                silent=silent,
                use_common_mean=use_common_mean,
                gti=gti,
                save_all=save_all,
            )
            spec.lc1 = data
        elif isinstance(data, (tuple, list)):
            if not isinstance(data[0], Lightcurve):  # pragma: no cover
                raise TypeError(f"Bad inputs to Powerspectrum: {type(data[0])}")
            dt = data[0].dt
            # This is a list of light curves.
            spec = powerspectrum_from_lc_iterable(
                data,
                dt,
                segment_size,
                norm=norm,
                silent=silent,
                use_common_mean=use_common_mean,
                gti=gti,
                save_all=save_all,
            )
        else:  # pragma: no cover
            raise TypeError(f"Bad inputs to Powerspectrum: {type(data)}")

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
        self.m = 1
        self.n = None
        self.k = 1
        return


class AveragedPowerspectrum(AveragedCrossspectrum, Powerspectrum):
    type = "powerspectrum"
    """
    Make an averaged periodogram from a light curve by segmenting the light
    curve, Fourier-transforming each segment and then averaging the
    resulting periodograms.

    Parameters
    ----------
    data: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects OR :class:`stingray.EventList` object
        The light curve data to be Fourier-transformed.

    segment_size: float
        The size of each segment to average. Note that if the total
        duration of each :class:`Lightcurve` object in lc is not an integer
        multiple of the ``segment_size``, then any fraction left-over at the
        end of the time series will be lost.

    norm: {"leahy" | "frac" | "abs" | "none" }, optional, default "frac"
        The normalization of the periodogram to be used.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care, especially if these GTIs have overlaps with the input
        object GTIs! If you're getting errors regarding your GTIs, don't
        use this and only give GTIs to the input object before making
        the power spectrum.

    silent : bool, default False
         Do not show a progress bar when generating an averaged cross spectrum.
         Useful for the batch execution of many spectra.

    dt: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data is of :class:EventList.

    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!).

    Attributes
    ----------
    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }
        The normalization of the periodogram.

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples.

    power: numpy.ndarray
        The array of normalized powers

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectra of segments of a light curve).
        Note that for a single realization (``m=1``) the error is equal to the
        power.

    unnorm_power: numpy.ndarray
        The array of unnormalized powers

    unnorm_power_err: numpy.ndarray
        The uncertainties of ``unnorm_power``.

    df: float
        The frequency resolution.

    m: int
        The number of averaged periodograms.

    n: int
        The number of data points in the light curve.

    nphots: float
        The total number of photons in the light curve.

    cs_all: list of :class:`Powerspectrum` objects
        The list of all periodograms used to calculate the average periodogram.
    """

    def __init__(
        self,
        data=None,
        segment_size=None,
        norm="frac",
        gti=None,
        silent=False,
        dt=None,
        lc=None,
        large_data=False,
        save_all=False,
        skip_checks=False,
        use_common_mean=True,
    ):
        self._type = None
        if lc is not None:
            warnings.warn(
                "The lc keyword is now deprecated. Use data " "instead", DeprecationWarning
            )
        # Backwards compatibility: user might have supplied lc instead
        if data is None:
            data = lc

        good_input = data is not None
        if good_input and not skip_checks:
            good_input = self.initial_checks(
                data1=data,
                data2=data,
                norm=norm,
                gti=gti,
                lc1=lc,
                lc2=lc,
                dt=dt,
                segment_size=segment_size,
            )

        norm = norm.lower()
        self.norm = norm
        self.dt = dt
        self.save_all = save_all
        self.segment_size = segment_size
        self.show_progress = not silent
        self.k = 1

        if not good_input:
            return self._initialize_empty()

        if isinstance(data, Generator):
            warnings.warn(
                "The averaged power spectrum from a generator of "
                "light curves pre-allocates the full list of light "
                "curves, losing all advantage of lazy loading. If it "
                "is important for you, use the "
                "AveragedPowerspectrum.from_lc_iterable static "
                "method, specifying the sampling time `dt`."
            )
            data = list(data)

        return self._initialize_from_any_input(
            data,
            dt=dt,
            segment_size=segment_size,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
            save_all=save_all,
        )

    def initial_checks(self, *args, **kwargs):
        return AveragedCrossspectrum.initial_checks(self, *args, **kwargs)


class DynamicalPowerspectrum(DynamicalCrossspectrum):
    type = "powerspectrum"
    """
    Create a dynamical power spectrum, also often called a *spectrogram*.

    This class will divide a :class:`Lightcurve` object into segments of
    length ``segment_size``, create a power spectrum for each segment and store
    all powers in a matrix as a function of both time (using the mid-point of
    each segment) and frequency.

    This is often used to trace changes in period of a (quasi-)periodic signal
    over time.

    Parameters
    ----------
    lc : :class:`stingray.Lightcurve` or :class:`stingray.EventList` object
        The time series or event list of which the dynamical power spectrum is
        to be calculated. If :class:`stingray.EventList`, ``dt`` must be specified as well.

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
        the power spectrum.

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
        The number of averaged cross spectra.
    """

    def __init__(self, lc=None, segment_size=None, norm="frac", gti=None, sample_time=None):
        self.segment_size = segment_size
        self.sample_time = sample_time
        self.gti = gti
        self.norm = norm

        if segment_size is None and lc is None:
            self._initialize_empty()
            self.dyn_ps = None
            return

        if segment_size is None or lc is None:
            raise TypeError("lc and segment_size must all be specified")

        if isinstance(lc, EventList) and sample_time is None:
            raise ValueError("To pass an input event lists, please specify sample_time")
        elif isinstance(lc, Lightcurve):
            sample_time = lc.dt
            if segment_size > lc.tseg:
                raise ValueError(
                    "Length of the segment is too long to create "
                    "any segments of the light curve!"
                )
        if segment_size < 2 * sample_time:
            raise ValueError("Length of the segment is too short to form a light curve!")

        self._make_matrix(lc)

    def shift_and_add(self, f0_list, nbins=100, rebin=None):
        """Shift-and-add the dynamical power spectrum.

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

        Returns
        -------
        output: :class:`AveragedPowerspectrum`
            The final averaged power spectrum.

        Examples
        --------
        >>> power_list = [[2, 5, 2, 2, 2], [1, 1, 5, 1, 1], [3, 3, 3, 5, 3]]
        >>> power_list = np.array(power_list).T
        >>> freqs = np.arange(5) * 0.1
        >>> f0_list = [0.1, 0.2, 0.3, 0.4]
        >>> dps = DynamicalPowerspectrum()
        >>> dps.dyn_ps = power_list
        >>> dps.freq = freqs
        >>> dps.df = 0.1
        >>> dps.m = 1
        >>> output = dps.shift_and_add(f0_list, nbins=5)
        >>> assert isinstance(output, AveragedPowerspectrum)
        >>> assert np.array_equal(output.m, [2, 3, 3, 3, 2])
        >>> assert np.array_equal(output.power, [2. , 2. , 5. , 2. , 1.5])
        >>> assert np.allclose(output.freq, [0.05, 0.15, 0.25, 0.35, 0.45])
        """
        return super().shift_and_add(
            f0_list, nbins=nbins, output_obj_type=AveragedPowerspectrum, rebin=rebin
        )

    def _make_matrix(self, lc):
        """
        Create a matrix of powers for each time step and each frequency step.

        Time increases with row index, frequency with column index.

        Parameters
        ----------
        lc : :class:`Lightcurve` object
            The :class:`Lightcurve` object from which to generate the dynamical
            power spectrum.
        """
        avg = AveragedPowerspectrum(
            lc,
            dt=self.sample_time,
            segment_size=self.segment_size,
            norm=self.norm,
            gti=self.gti,
            save_all=True,
        )
        conv = avg.cs_all / avg.unnorm_cs_all
        self.unnorm_conversion = np.nanmean(conv)
        self.dyn_ps = np.array(avg.cs_all).T
        self.freq = avg.freq
        current_gti = avg.gti

        tstart, _ = time_intervals_from_gtis(current_gti, self.segment_size)

        self.time = tstart + 0.5 * self.segment_size
        self.df = avg.df
        self.dt = self.segment_size
        self.meanrate = avg.nphots / avg.n / avg.dt
        self.nphots = avg.nphots
        self.m = 1

    def power_colors(
        self,
        freq_edges=[1 / 256, 1 / 32, 0.25, 2.0, 16.0],
        freqs_to_exclude=None,
        poisson_power=None,
    ):
        """
        Return the power colors of the dynamical power spectrum.

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

        poisson_level : float or iterable, optional
            Defaults to the theoretical Poisson noise level (e.g. 2 for Leahy normalization).
            The Poisson noise level of the power spectrum. If iterable, it should have the same
            length as ``frequency``. (This might apply to the case of a power spectrum with a
            strong dead time distortion)

        Returns
        -------
        pc0: np.ndarray
        pc0_err: np.ndarray
        pc1: np.ndarray
        pc1_err: np.ndarray
            The power colors for each spectrum and their respective errors
        """
        if poisson_power is None:
            poisson_power = poisson_level(
                norm=self.norm,
                meanrate=self.meanrate,
                n_ph=self.nphots,
            )

        return super().power_colors(
            freq_edges=freq_edges,
            freqs_to_exclude=freqs_to_exclude,
            poisson_power=poisson_power,
        )


class GtiCorrPowerspectrum(Powerspectrum):
    main_array_attr = "freq"
    type = "powerspectrum"

    """Calculate the power spectrum of gappy light curves.

    GtiCorrPowerspectrum computes the power spectrum of gappy light curves,
    cleaning up the frequencies that are more affected by gaps.
    Optionally, it fills bad time intervals (BTIs) with the mean count rate from
    good time intervals (GTIs), mitigating window-induced features in the periodogram.
    By analyzing the visibility light curve (synthetic light curve with constant mean
    counts in GTIs), the class identifies strong peaks in the periodogram that correspond to
    the missing data and applies notch filtering to the power spectrum to remove these
    frequencies.
    Additionally, it rescales the power spectrum to account for the number of bins in and out GTIs,
    ensuring accurate white noise and rms estimation.

    The detailed explanation of the method is given in
    `El Byad et al. 2025 <https://arxiv.org/pdf/2505.16921>`__

    Parameters
    ----------
    *args:
        Any arguments that can be passed to ``Powerspectrum``

    Other Parameters
    ----------------
    fill_lc: boolean, optional, default ``True``
        If True, fill the BTIs of the light curve with the mean value of the counts in the GTIs.
        Recommended.

    sigma_threshold: float, optional, default ``3``
        The sigma threshold for the detection of features in the power spectrum of the observing
        window.

    **kwargs:
        Any other keyword arguments that can be passed to ``Powerspectrum``

    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of power values

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

    nphots: float
        The total number of photons in light curve

    """

    def __init__(self, *args, fill_lc=True, sigma_threshold=3, **kwargs):
        dt = kwargs.pop("dt", None)
        skip_checks = kwargs.pop("skip_checks", False)

        if len(args) == 0:
            self.lc = None
        elif isinstance(args[0], EventList):
            self.lc = args[0].to_lc(dt)
        else:
            self.lc = args[0]
            if dt is None:
                dt = self.lc.dt

        if fill_lc:
            self.fill_lc_with_mean()

        self.sigma_threshold = sigma_threshold
        if not skip_checks:
            self.initial_checks(*args, dt=dt, **kwargs)

        super().__init__(self.lc, *args[1:], dt=dt, **kwargs, skip_checks=True)

        self.mjdref = None
        if hasattr(self.lc, "mjdref"):
            self.mjdref = self.lc.mjdref

        if len(args) == 0:
            self.mask = None
            return

        if fill_lc:
            lc_mask = create_gti_mask(self.lc.time, self.lc.gti)
            self.power *= lc_mask.size / np.count_nonzero(lc_mask)

        self.mask = np.ones(self.power.size, dtype=bool)

    def initial_checks(self, *args, **kwargs):
        if self.lc is not None and not np.allclose(np.diff(self.lc.time), self.lc.dt):
            raise ValueError(
                "The time array in the light curve is not evenly spaced. "
                "This is not supported by GtiCorrPowerspectrum."
            )

    def fill_lc_with_mean(self):
        if self.lc is None:
            return

        mask = create_gti_mask(self.lc.time, self.lc.gti)
        self.lc.counts = self.lc.counts.astype(float)
        self.lc.counts[~mask] = np.mean(self.lc.counts[mask])

    def clean_gti_features(self, plot=False, figname="gti_features"):
        gti = getattr(self, "gti", None)
        if gti is None:
            raise AttributeError("GTI attribute is not set for this object.")
        exposure = np.sum(gti[:, 1] - gti[:, 0])
        ref_ctrate = self.nphots / exposure
        self.exposure = exposure

        lc = copy.deepcopy(self.lc)
        lc.gti = np.array([[gti[0, 0], gti[-1, 1]]])
        mask = create_gti_mask(lc.time, gti)
        lc.counts = lc.counts.astype(float)
        lc.counts[~mask] = 0
        lc.counts[mask] = ref_ctrate * lc.dt

        ps_gti = Powerspectrum(lc, norm="leahy")

        # Correct the PS level to overcome the Nph from filling the lc with mean
        prob = scipy.stats.norm.cdf(-self.sigma_threshold)
        thresh = pds_detection_level(prob)

        bad = ps_gti.power > thresh
        if plot:
            self._plot_gti_features(ps_gti, thresh, figname)
        self.mask = self.mask & ~bad
        newpow = self.apply_mask(self.mask)
        return newpow

    def _plot_gti_features(self, ps_gti, thresh, figname):
        """Plot the features in the power spectrum of the GTI-corrected light curve.

        This method generates a log-log plot of the power spectrum and saves it as a JPEG file.

        Parameters
        ----------
        ps_gti : Powerspectrum
            The power spectrum of the synthetic light curve having the mean counts inside GTIs
            and 0 outside.
        thresh : float
            The threshold value for the power spectrum, above which features are considered significant.
        figname : str
            The name of the figure file to be saved (without extension).
        """
        fig = plt.figure(figname)
        plt.loglog(ps_gti.freq, ps_gti.power)
        plt.axhline(thresh)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(f"Power {ps_gti.norm}")
        plt.savefig(figname + ".jpg")
        plt.close(fig)

    def rebin_log(self, *args, **kwargs):
        """Rebin the power spectrum logarithmically and filter out NaN values.

        This method overrides the parent class's `rebin_log` method by applying a mask
        to remove any bins where the rebinned power is NaN.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the parent class's `rebin_log` method.
        **kwargs : dict
            Keyword arguments passed to the parent class's `rebin_log` method.

        Returns
        -------
        GtiCorrPowerspectrum
            A new power spectrum object with rebinned frequencies and powers,
            with NaN values filtered out.
        """
        new_ps = Powerspectrum.rebin_log(self, *args, **kwargs)
        mask = ~np.isnan(new_ps.power)
        return new_ps.apply_mask(mask)


def powerspectrum_from_time_array(
    times,
    dt,
    segment_size=None,
    gti=None,
    norm="frac",
    silent=False,
    use_common_mean=True,
    save_all=False,
):
    """
    Calculate a power spectrum from an array of event times.

    Parameters
    ----------
    times : `np.array`
        Event arrival times.
    dt : float
        The time resolution of the intermediate light curves
        (sets the Nyquist frequency).

    Other parameters
    ----------------
    segment_size : float
        The length, in seconds, of the light curve segments that will be
        averaged. Only required (and used) for ``AveragedPowerspectrum``.
    gti : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Additional, optional Good Time intervals that get intersected with
        the GTIs of the input object. Can cause errors if there are
        overlaps between these GTIs and the input object GTIs. If that
        happens, assign the desired GTIs to the input object.
    norm : str, default "frac"
        The normalization of the periodogram. `abs` is absolute rms, `frac`
        is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
        the unnormalized periodogram.
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or
        on the full light curve. This gives different results
        (Alston+2013). By default, we assume the mean is calculated on the
        full light curve, but the user can set ``use_common_mean`` to False
        to calculate it on a per-segment basis.
    silent : bool, default False
        Silence the progress bars.
    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
    """
    force_averaged = segment_size is not None
    # Suppress progress bar for single periodogram
    silent = silent or (segment_size is None)
    table = avg_pds_from_timeseries(
        times,
        gti,
        segment_size,
        dt,
        norm=norm,
        use_common_mean=use_common_mean,
        silent=silent,
        return_subcs=save_all,
    )

    return _create_powerspectrum_from_result_table(table, force_averaged=force_averaged)


def powerspectrum_from_events(
    events,
    dt,
    segment_size=None,
    gti=None,
    norm="frac",
    silent=False,
    use_common_mean=True,
    save_all=False,
):
    """
    Calculate a power spectrum from an event list.

    Parameters
    ----------
    events : `stingray.EventList`
        Event list to be analyzed.
    dt : float
        The time resolution of the intermediate light curves
        (sets the Nyquist frequency)

    Other parameters
    ----------------
    segment_size : float
        The length, in seconds, of the light curve segments that will be
        averaged. Only required (and used) for ``AveragedPowerspectrum``.
    gti : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Additional, optional Good Time intervals that get intersected with
        the GTIs of the input object. Can cause errors if there are
        overlaps between these GTIs and the input object GTIs. If that
        happens, assign the desired GTIs to the input object.
    norm : str, default "frac"
        The normalization of the periodogram. `abs` is absolute rms, `frac`
        is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
        the unnormalized periodogram.
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or
        on the full light curve. This gives different results
        (Alston+2013). By default, we assume the mean is calculated on the
        full light curve, but the user can set ``use_common_mean`` to False
        to calculate it on a per-segment basis.
    silent : bool, default False
        Silence the progress bars.
    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
    """
    if gti is None:
        gti = events.gti

    dt = events.suggest_compatible_dt(dt)

    return powerspectrum_from_time_array(
        events.time,
        dt,
        segment_size,
        gti,
        norm=norm,
        silent=silent,
        use_common_mean=use_common_mean,
        save_all=save_all,
    )


def powerspectrum_from_lightcurve(
    lc, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True, save_all=False
):
    """
    Calculate a power spectrum from a light curve

    Parameters
    ----------
    lc : `stingray.Lightcurve`
        Light curve to be analyzed.

    Other parameters
    ----------------
    segment_size : float
        The length, in seconds, of the light curve segments that will be
        averaged. Only required (and used) for ``AveragedPowerspectrum``.
    gti : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Additional, optional Good Time intervals that get intersected with
        the GTIs of the input object. Can cause errors if there are
        overlaps between these GTIs and the input object GTIs. If that
        happens, assign the desired GTIs to the input object.
    norm : str, default "frac"
        The normalization of the periodogram. `abs` is absolute rms, `frac`
        is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
        the unnormalized periodogram.
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or
        on the full light curve. This gives different results
        (Alston+2013). By default, we assume the mean is calculated on the
        full light curve, but the user can set ``use_common_mean`` to False
        to calculate it on a per-segment basis.
    silent : bool, default False
        Silence the progress bars.
    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
    """
    force_averaged = segment_size is not None
    # Suppress progress bar for single periodogram
    silent = silent or (segment_size is None)
    err = None
    if lc.err_dist == "gauss":
        err = lc.counts_err
    if gti is None:
        gti = lc.gti

    table = avg_pds_from_timeseries(
        lc.time,
        gti,
        segment_size,
        lc.dt,
        norm=norm,
        use_common_mean=use_common_mean,
        silent=silent,
        fluxes=lc.counts,
        errors=err,
        return_subcs=save_all,
    )

    return _create_powerspectrum_from_result_table(table, force_averaged=force_averaged)


def powerspectrum_from_timeseries(
    ts,
    flux_attr,
    error_flux_attr=None,
    segment_size=None,
    norm="none",
    silent=False,
    use_common_mean=True,
    gti=None,
    save_all=False,
):
    """Calculate power spectrum from a time series

    Parameters
    ----------
    ts : `stingray.StingrayTimeseries`
        Input time series
    flux_attr : `str`
        What attribute of the time series will be used.

    Other parameters
    ----------------
    error_flux_attr : `str`
        What attribute of the time series will be used as error bar.
    segment_size : float
        The length, in seconds, of the light curve segments that will be
        averaged. Only required (and used) for ``AveragedPowerspectrum``.
    gti : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Additional, optional Good Time intervals that get intersected with
        the GTIs of the input object. Can cause errors if there are
        overlaps between these GTIs and the input object GTIs. If that
        happens, assign the desired GTIs to the input object.
    norm : str, default "frac"
        The normalization of the periodogram. `abs` is absolute rms, `frac`
        is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
        the unnormalized periodogram.
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or
        on the full light curve. This gives different results
        (Alston+2013). By default, we assume the mean is calculated on the
        full light curve, but the user can set ``use_common_mean`` to False
        to calculate it on a per-segment basis.
    silent : bool, default False
        Silence the progress bars.
    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.

    Returns
    -------
    spec : `AveragedCrossspectrum` or `Crossspectrum`
        The output cross spectrum.
    """
    force_averaged = segment_size is not None
    # Suppress progress bar for single periodogram
    silent = silent or (segment_size is None)
    if gti is None:
        gti = ts.gti

    err = None
    if error_flux_attr is not None:
        err = getattr(ts, error_flux_attr)

    results = avg_pds_from_timeseries(
        ts.time,
        gti,
        segment_size,
        ts.dt,
        norm=norm,
        use_common_mean=use_common_mean,
        silent=silent,
        fluxes=getattr(ts, flux_attr),
        errors=err,
        return_subcs=save_all,
    )

    return _create_powerspectrum_from_result_table(results, force_averaged=force_averaged)


def powerspectrum_from_lc_iterable(
    iter_lc,
    dt,
    segment_size=None,
    gti=None,
    norm="frac",
    silent=False,
    use_common_mean=True,
    save_all=False,
):
    """
    Calculate an average power spectrum from an iterable collection of light
    curves.

    Parameters
    ----------
    iter_lc : iterable of `stingray.Lightcurve` objects or `np.array`
        Light curves. If arrays, use them as counts.
    dt : float
        The time resolution of the light curves
        (sets the Nyquist frequency).

    Other parameters
    ----------------
    segment_size : float, default None
        The length, in seconds, of the light curve segments that will be
        averaged. If not ``None``, it will be used to check the segment size of
         the output.
    gti : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Additional, optional Good Time intervals that get intersected with
        the GTIs of the input object. Can cause errors if there are
        overlaps between these GTIs and the input object GTIs. If that
        happens, assign the desired GTIs to the input object.
    norm : str, default "frac"
        The normalization of the periodogram. `abs` is absolute rms, `frac`
        is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
        the unnormalized periodogram.
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or
        on the full light curve. This gives different results
        (Alston+2013). By default, we assume the mean is calculated on the
        full light curve, but the user can set ``use_common_mean`` to False
        to calculate it on a per-segment basis.
    silent : bool, default False
        Silence the progress bars.
    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
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
            elif isinstance(lc, Iterable):
                yield lc
            else:
                raise TypeError(
                    "The inputs to `powerspectrum_from_lc_iterable`"
                    " must be Lightcurve objects or arrays"
                )

    table = avg_pds_from_iterable(
        iterate_lc_counts(iter_lc),
        dt,
        norm=norm,
        use_common_mean=use_common_mean,
        silent=silent,
        return_subcs=save_all,
    )
    return _create_powerspectrum_from_result_table(table, force_averaged=force_averaged)


def _create_powerspectrum_from_result_table(table, force_averaged=False):
    """
    Copy the columns and metadata from the results of
    ``stingray.fourier.avg_pds_from_XX`` functions into
    `AveragedPowerspectrum` or `Powerspectrum` objects.

    By default, allocates a Powerspectrum object if the number of
    averaged spectra is 1, otherwise an AveragedPowerspectrum.
    If the user specifies ``force_averaged=True``, it always allocates
    an AveragedPowerspectrum.

    Parameters
    ----------
    table : `astropy.table.Table`
        results of `avg_cs_from_iterables` or `avg_cs_from_iterables_quick`

    Other parameters
    ----------------
    force_averaged : bool, default False

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
    """
    if table.meta["m"] > 1 or force_averaged:
        cs = AveragedPowerspectrum()
    else:
        cs = Powerspectrum()

    cs.freq = np.array(table["freq"])
    cs.power = np.array(table["power"])
    cs.unnorm_power = np.array(table["unnorm_power"])

    for attr, val in table.meta.items():
        setattr(cs, attr, val)

    if "subcs" in table.meta:
        cs.cs_all = np.array(table.meta["subcs"])
        cs.unnorm_cs_all = np.array(table.meta["unnorm_subcs"])

    cs.err_dist = "poisson"
    if hasattr(cs, "variance") and cs.variance is not None:
        cs.err_dist = "gauss"

    cs.power_err = cs.power / np.sqrt(cs.m)
    cs.unnorm_power_err = cs.unnorm_power / np.sqrt(cs.m)
    cs.nphots1 = cs.nphots
    return cs

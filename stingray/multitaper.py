import copy

from stingray.gti import check_gtis, cross_two_gtis
from stingray.fourier import normalize_periodograms
from stingray.powerspectrum import Powerspectrum
import warnings

import numpy as np
import scipy.optimize
import scipy.stats
from scipy import signal, interpolate

from astropy.timeseries import LombScargle

from .events import EventList
from .lightcurve import Lightcurve
from .utils import rebin_data, simon, fft, rfft, rfftfreq

try:
    integration_func = np.trapezoid
except AttributeError:
    # < Numpy 2.0.0
    integration_func = np.trapz

__all__ = ["Multitaper"]

# Inspired from nitime (https://nipy.org/nitime/)


class Multitaper(Powerspectrum):
    """
    Class to calculate the multitaper periodogram from lightcurve data.

    Parameters
    ----------
    data: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data to be Fourier-transformed.

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }, optional, default ``frac``
        The normaliation of the power spectrum to be used. Options are
        ``leahy``, ``frac``, ``abs`` and ``none``, default is ``frac``.

    NW: float, optional, default ``4``
        The normalized half-bandwidth of the data tapers, indicating a
        multiple of the fundamental frequency of the DFT (Fs/N).
        Common choices are n/2, for n >= 4.

    adaptive: boolean, optional, default ``False``
        Use an adaptive weighting routine to combine the PSD estimates of
        different tapers.

    jackknife: boolean, optional, default ``True``
        Use the jackknife method to make an estimate of the PSD variance
        at each point.

    low_bias: boolean, optional, default ``True``
        Rather than use 2NW tapers, only use the tapers that have better than
        90% spectral concentration within the bandwidth (still using
        a maximum of 2NW tapers)

    lombscargle: boolean, optional, default ``False``
        Whether to use the Lomb (1976) Scargle (1982) periodogram when
        calculating the Multitaper spectral estimate. Highly recommended for
        unevenly sampled time-series. Adaptive weighting and jack-knife
        estimated variance are yet not supported.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    Attributes
    ----------
    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }
        the normalization of the power spectrun

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes

    unnorm_power: numpy.ndarray
        The array of unnormalized values of Fourier amplitudes

    multitaper_norm_power: numpy.ndarray
        The array of normalized values of Fourier amplitudes, normalized
        according to the scheme followed in nitime, that is, by the length and
        the sampling frequency.

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectrum). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged powers in each bin

    n: int
        The number of data points in the light curve

    k: array of int
        The rebinning scheme if the object has been rebinned otherwise is set to 1.

    nphots: float
        The total number of photons in the light curve

    jk_var_deg_freedom: numpy.ndarray
        Array differs depending on whether
        the jackknife was used. It is either
        * The jackknife estimated variance of the log-psd, OR
        * The degrees of freedom in a chi2 model of how the estimated
          PSD is distributed about the true log-PSD (this is either
          2*floor(2*NW), or calculated from adaptive weights)

    Notes
    -----
    The bandwidth of the windowing function will determine the number of
    tapers to use. This parameter (NW) represents trade-off between frequency
    resolution (lower main lobe BW for the taper) and variance reduction
    (higher BW and number of averaged estimates). Typically, the number of
    tapers is calculated as 2x the bandwidth-to-fundamental-frequency
    ratio (NW), as these eigenfunctions have the best energy concentration.

    """

    def __init__(
        self,
        data=None,
        norm="frac",
        gti=None,
        dt=None,
        lc=None,
        NW=4,
        adaptive=False,
        jackknife=True,
        low_bias=True,
        lombscargle=False,
    ):
        if lc is not None:
            warnings.warn(
                "The lc keyword is now deprecated. Use data " "instead", DeprecationWarning
            )
        if data is None:
            data = lc

        if isinstance(norm, str) is False:
            raise TypeError("norm must be a string")

        if norm.lower() not in ["frac", "abs", "leahy", "none"]:
            raise ValueError("norm must be 'frac', 'abs', 'leahy', or 'none'!")

        self.norm = norm.lower()

        if isinstance(data, EventList) and dt is None:
            raise ValueError(
                "If using event lists, please specify " "the bin time to generate lightcurves."
            )

        if data is None:
            self.freq = None
            self.power = None
            self.multitaper_norm_power = None
            self.eigvals = None
            self.power_err = None
            self.df = None
            self.m = 1
            self.n = None
            self.nphots = None
            self.k = 1
            self.jk_var_deg_freedom = None
            self.segment_size = None
            return
        elif not isinstance(data, EventList):
            lc = data
        else:
            lc = data.to_lc(dt)

        self.gti = gti
        self.lc = lc
        self.power_type = "real"
        self.fullspec = False
        self.k = 1
        self.segment_size = None

        self._make_multitaper_periodogram(
            lc,
            NW=NW,
            adaptive=adaptive,
            jackknife=jackknife,
            low_bias=low_bias,
            lombscargle=lombscargle,
        )

    def _make_multitaper_periodogram(
        self, lc, NW=4, adaptive=False, jackknife=True, low_bias=True, lombscargle=False
    ):
        """Compute the normalized multitaper spectral estimate.

        This includes checking for the presence of and applying Good Time Intervals,
        computing the a nitime inspired normalized power spectrum, unnormalizing it,
        and then renormalizing it using the required normalization.

        Parameters
        ----------
        lc : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        NW: float, optional, default ``4``
            The normalized half-bandwidth of the data tapers, indicating a
            multiple of the fundamental frequency of the DFT (Fs/N).
        Common choices are n/2, for n >= 4.

        adaptive: boolean, optional, default ``False``
            Use an adaptive weighting routine to combine the PSD estimates of
            different tapers.

        jackknife: boolean, optional, default ``True``
            Use the jackknife method to make an estimate of the PSD variance
            at each point.

        low_bias: boolean, optional, default ``True``
            Rather than use 2NW tapers, only use the tapers that have better than
            90% spectral concentration within the bandwidth (still using
            a maximum of 2NW tapers)
        """

        if not isinstance(lc, Lightcurve):
            raise TypeError("lc must be a lightcurve.Lightcurve object")

        if self.gti is None:
            self.gti = cross_two_gtis(lc.gti, lc.gti)

        check_gtis(self.gti)

        if self.gti.shape[0] != 1:
            raise TypeError("Non-averaged Spectra need " "a single Good Time Interval")

        lc = lc.split_by_gti()[0]

        self.meancounts = lc.meancounts
        self.nphots = np.float64(np.sum(lc.counts))

        self.err_dist = "poisson"
        if lc.err_dist == "poisson":
            self.var = lc.meancounts
        else:
            self.var = np.mean(lc.counts_err) ** 2
            self.err_dist = "gauss"

        self.dt = lc.dt
        self.n = lc.n

        # the frequency resolution
        self.df = 1.0 / lc.tseg

        # the number of averaged periodograms in the final output
        # This should *always* be 1 here
        self.m = 1

        if lombscargle:
            self.freq, self.multitaper_norm_power = self._fourier_multitaper_lomb_scargle(
                lc, NW=NW, low_bias=low_bias
            )

            self.unnorm_power = self.multitaper_norm_power * lc.n * 2

        else:
            self.freq, self.multitaper_norm_power = self._fourier_multitaper(
                lc, NW=NW, adaptive=adaptive, jackknife=jackknife, low_bias=low_bias
            )

            self.unnorm_power = self.multitaper_norm_power * lc.n / lc.dt

        self.power = self._normalize_multitaper(self.unnorm_power)

        if lc.err_dist.lower() != "poisson":
            simon(
                "Looks like your lightcurve statistic is not poisson."
                "The errors in the Powerspectrum will be incorrect."
            )

        self.power_err = self.power / np.sqrt(self.m)

    def _fourier_multitaper(self, lc, NW=4, adaptive=False, jackknife=True, low_bias=True):
        """Compute the multitaper periodogram.

        Auxiliary method to apply the multitaper algorithm by first computing
        the DPSS windows, then fourier transforming the data tapered with these
        DPSS windows, and then calculating the power spectral density using a
        desired scheme (a combination of adaptive and low_bias parameters).
        Also set some of the multitaper specific attributes.

        Parameters
        ----------
        lc : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the spectrum estimate.

        NW: float, optional, default ``4``
            The normalized half-bandwidth of the data tapers, indicating a
            multiple of the fundamental frequency of the DFT (Fs/N).
            Common choices are n/2, for n >= 4.

        adaptive: boolean, optional, default ``False``
            Use an adaptive weighting routine to combine the PSD estimates of
            different tapers.

        jackknife: boolean, optional, default ``True``
            Use the jackknife method to make an estimate of the PSD variance
            at each point.

        low_bias: boolean, optional, default ``True``
            Rather than use 2NW tapers, only use the tapers that have better than
            90% spectral concentration within the bandwidth (still using
            a maximum of 2NW tapers)

        Returns
        -------
        freq_multitaper: numpy.ndarray
            The frequency mid-bins of the PSD amplitudes

        psd_multitaper: numpy.ndarray
            The value of the PSD amplitudes at the given frequency mid-bins
        """

        if NW < 0.5:
            raise ValueError("The value of normalized half-bandwidth " "should be greater than 0.5")

        Kmax = int(2 * NW)

        dpss_tapers, eigvals = signal.windows.dpss(
            M=lc.n, NW=NW, Kmax=Kmax, sym=False, return_ratios=True
        )

        if low_bias:
            selected_tapers = eigvals > 0.9
            if not selected_tapers.any():
                simon("Could not properly use low_bias, " "keeping the lowest-bias taper")
                selected_tapers = [np.argmax(eigvals)]

            eigvals = eigvals[selected_tapers]
            dpss_tapers = dpss_tapers[selected_tapers, :]

        print(f"Using {len(eigvals)} DPSS windows for " "multitaper spectrum estimator")

        data_multitaper = lc.counts - np.mean(lc.counts)  # De-mean
        data_multitaper = np.tile(data_multitaper, (len(eigvals), 1))
        data_multitaper = np.multiply(data_multitaper, dpss_tapers)

        freq_response = rfft(data_multitaper, n=lc.n)

        # Adjust DC and maybe Nyquist, depending on one-sided transform
        freq_response[..., 0] /= np.sqrt(2.0)
        if lc.n % 2 == 0:
            freq_response[..., -1] /= np.sqrt(2.0)

        freq_response = freq_response[..., 1:-1]
        freq_multitaper = rfftfreq(lc.n, d=lc.dt)[1:-1]

        if adaptive:
            psd_multitaper, weights_multitaper = self._get_adaptive_psd(freq_response, eigvals)
        else:
            weights_multitaper = np.sqrt(eigvals)[:, np.newaxis]
            psd_multitaper = self.psd_from_freq_response(freq_response, weights_multitaper)

        psd_multitaper *= lc.dt  # /= sampling_freq
        self.eigvals = eigvals

        if jackknife:
            self.jk_var_deg_freedom = self.jackknifed_sdf_variance(freq_response, eigvals, adaptive)
        else:
            if adaptive:
                self.jk_var_deg_freedom = 2 * (weights_multitaper**2).sum(axis=-2)
            else:
                self.jk_var_deg_freedom = np.full((len(freq_multitaper)), 2 * len(eigvals))

        return freq_multitaper, psd_multitaper

    def psd_from_freq_response(self, freq_response, weights):
        """Calculate the weighted PSD from the Fourier transformed data.

        Calculate the weighted PSD from the Fourier transformed data by
        combining the frequencies responses of these tapered data using the given
        weights.

        Parameters
        ----------
        freq_response: numpy.ndarray
            The frequency responses or the Fourier transforms of the tapered data

        weights: numpy.ndarray
            The weights to combine the different Fourier transforms of the data
            tapered with different windows to produce the weighted PSD.

        Returns
        -------
        psd: numpy.ndarray
            The weighted power spectral density of the provided Fourier
            transformed data.

        Notes
        -----
        The shape of ``freq_response`` and ``weights`` must either be same or
        broadcastable.
        """

        psd = freq_response * weights
        psd *= psd.conj()
        psd = psd.real.sum(axis=-2)  # Sum all rows
        psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
        return psd

    def _get_adaptive_psd(self, freq_response, eigvals, max_iter=150):
        r"""Compute the PSD from tapered data.

        Performs an iterative procedure to compute the PSD from tapered data
        by finding the optimal weights for the direct spectral estimators
        of the DPSS tapered signals.

        Parameters
        ----------
        freq_response: numpy.ndarray
            The frequency responses or the Fourier transforms of the tapered data

        eigvals: numpy.ndarray
            The eigenvalues of the DPSS tapers

        max_iter: int, optional, default ``150``
            Maximum number of iterations for weight computation

        Returns
        -------
        psd : numpy.ndarray
            The adaptively computed PSD

        weights : numpy.ndarray
            The weights used to combine the tapered spectra

        Notes
        -----
        The weights to use for making the multitaper estimate, such that
        :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`

        If the number of tapers are less than 3, no adaptive weights are
        calculated and the square roots of eigenvalues are used as weights
        to produce the weighted PSD
        """

        n_tapers = len(eigvals)
        n_freqs = freq_response.shape[-1]

        sqrt_eigvals = np.sqrt(eigvals)

        if n_tapers < 3:
            simon("Not adaptively combining, number of tapers < 3")
            weights = sqrt_eigvals[:, np.newaxis]
            return self.psd_from_freq_response(freq_response, weights), weights

        psd_est = self.psd_from_freq_response(freq_response, sqrt_eigvals[:, np.newaxis])

        var = integration_func(psd_est, dx=np.pi / n_freqs) / (2 * np.pi)
        del psd_est

        psd = np.empty(n_freqs)

        weights = np.empty((n_tapers, n_freqs))

        # combine the SDFs in the traditional way in order to estimate
        # the variance of the timeseries

        # The process is to iteratively switch solving for the following
        # two expressions:
        # (1) Adaptive Multitaper SDF:
        # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
        #
        # (2) Weights
        # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
        #
        # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
        # and the expected value of the broadband bias function
        # E{B_k(f)} is replaced by its full-band integration
        # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

        # start with an estimate from incomplete data--the first 2 tapers
        psd_iter = self.psd_from_freq_response(freq_response[:2], sqrt_eigvals[:2, np.newaxis])

        err = np.zeros_like(freq_response)

        for ite in range(max_iter):
            d_k = psd_iter / (
                eigvals[:, np.newaxis] * psd_iter + (1 - eigvals[:, np.newaxis]) * var
            )
            d_k *= sqrt_eigvals[:, np.newaxis]
            # Test for convergence -- this is overly conservative, since
            # iteration only stops when all frequencies have converged.
            # A better approach is to iterate separately for each freq, but
            # that is a nonvectorized algorithm.
            # Take the RMS difference in weights from the previous iterate
            # across frequencies. If the maximum RMS error across freqs is
            # less than 1e-10, then we're converged

            err -= d_k
            if np.max(np.mean(err**2, axis=0)) < 1e-10:
                break

            # update the iterative estimate with this d_k
            psd_iter = self.psd_from_freq_response(freq_response, d_k)
            err = d_k
        if ite == max_iter - 1:
            simon("Iterative multi-taper PSD computation did not converge.")

        return psd_iter, d_k

    def _normalize_multitaper(self, unnorm_power):
        """
        Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
        fractional rms^2 normalization, or not at all.

        Parameters
        ----------
        unnorm_power: numpy.ndarray
            The unnormalized spectrum estimate.

        Returns
        -------
        power: numpy.nd.array
            The normalized spectrum estimate (real part of the spectrum). For
            'none' normalization, imaginary part is returned as well.
        """

        mean = self.nphots / self.n
        variance = None
        if self.err_dist != "poisson":
            variance = self.var
        return normalize_periodograms(
            unnorm_power,
            self.dt,
            self.n,
            mean,
            n_ph=self.nphots,
            variance=variance,
            norm=self.norm,
            power_type=self.power_type,
        )

    def jackknifed_sdf_variance(self, freq_response, eigvals, adaptive):
        r"""Compute the variance of the spectrum through jack-knife.

        Returns the variance of the log-sdf estimated through jack-knifing
        a group of independent sdf estimates.

        Parameters
        ----------

        freq_response: numpy.ndarray
            The frequency responses or the Fourier transforms of the tapered data

        eigvals: numpy.ndarray
            The eigenvalues of the DPSS tapers

        adaptive: boolean
            Whether or not to use the adaptive weighting method to calculate
            the log-sdf through jack-knife method

        Returns
        -------

        var : The estimate for log-sdf variance

        Notes
        -----

        The jackknifed mean estimate is distributed about the true mean as
        a Student's t-distribution with (K-1) degrees of freedom, and
        standard error equal to sqrt(var). However, Thompson and Chave [1]
        point out that this variance better describes the sample mean.
        [1] Thomson D J, Chave A D (1991) Advances in Spectrum Analysis and Array
        Processing (Prentice-Hall, Englewood Cliffs, NJ), 1, pp 58-113.
        """

        K = len(eigvals)

        all_indices = set(range(K))
        jk_sdk = []

        for i in range(K):
            selected_indices = list(all_indices.difference([i]))
            selected_data = np.take(freq_response, selected_indices, axis=0)
            selected_eigvals = np.take(eigvals, selected_indices)

            if adaptive:
                selected_psd, selected_weights = self._get_adaptive_psd(
                    selected_data, selected_eigvals
                )

            else:
                selected_weights = np.sqrt(selected_eigvals)[:, np.newaxis]
                selected_psd = self.psd_from_freq_response(selected_data, selected_weights)

            jk_sdk.append(selected_psd)

        jk_sdk = np.log(jk_sdk)
        jk_var = jk_sdk - jk_sdk.mean(axis=0)

        np.power(jk_var, 2, jk_var)
        jk_var = jk_var.sum(axis=0)

        # Thompson's recommended factor, eq 18
        # Jackknifing Multitaper Spectrum Estimates
        # IEEE SIGNAL PROCESSING MAGAZINE [20] JULY 2007
        K = float(K)
        f = (K - 1) ** 2 / K / (K - 0.5)
        jk_var *= f
        return jk_var

    def rebin(self, df=None, f=None, method="mean"):
        """Rebin the multitaper spectrum to a new frequency resolution ``df``.

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
        bin_cs = :class:`Multitaper` (or one of its subclasses) object
            The newly binned multitaper spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from :class:`AveragedPowerspectrum`, it will return an object of class
            :class:`AveragedPowerspectrum`, too.
        """

        if f is None and df is None:
            raise ValueError("You need to specify at least one between f and " "df")
        elif f is not None:
            df = f * self.df

        # rebin cross spectrum to new resolution
        binfreq, binmtp, binerr, step_size = rebin_data(
            self.freq, self.power, df, self.power_err, method=method, dx=self.df
        )
        # make an empty cross spectrum object
        # note: syntax deliberate to work with subclass Powerspectrum
        bin_mtp = copy.copy(self)

        # store the binned periodogram in the new object
        bin_mtp.freq = binfreq
        bin_mtp.power = binmtp
        bin_mtp.eigvals = self.eigvals
        bin_mtp.df = df
        bin_mtp.n = self.n
        bin_mtp.norm = self.norm
        bin_mtp.nphots = self.nphots
        bin_mtp.power_err = binerr

        if hasattr(self, "unnorm_power"):
            _, binpower_unnorm, _, _ = rebin_data(
                self.freq, self.unnorm_power, df, method=method, dx=self.df
            )

            bin_mtp.unnorm_power = binpower_unnorm

        if hasattr(self, "multitaper_norm_power"):
            _, bin_multitaper_norm_power, _, _ = rebin_data(
                self.freq, self.multitaper_norm_power, df, method=method, dx=self.df
            )

            bin_mtp.multitaper_norm_power = bin_multitaper_norm_power

        if hasattr(self, "jk_var_deg_freedom"):
            _, bin_jk_var_deg_freedom, _, _ = rebin_data(
                self.freq, self.jk_var_deg_freedom, df, method=method, dx=self.df
            )

            bin_mtp.jk_var_deg_freedom = bin_jk_var_deg_freedom

        bin_mtp.m = np.rint(step_size * self.m)

        return bin_mtp

    def _fourier_multitaper_lomb_scargle(self, lc, NW=4, low_bias=True):
        """Compute the multitaper lomb-scargle spectral estimate.

        Use the multitapering concept to calculate the spectral estimate of
        an unevenly sampled time-series, using the Lomb (1976) - Scargle (1982)
        periodogram. (Springford et al. 2020)


        Parameters
        ----------
        lc : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the spectrum estimate.

        NW: float, optional, default ``4``
            The normalized half-bandwidth of the data tapers, indicating a
            multiple of the fundamental frequency of the DFT (Fs/N).
            Common choices are n/2, for n >= 4.

        low_bias: boolean, optional, default ``True``
            Rather than use 2NW tapers, only use the tapers that have better than
            90% spectral concentration within the bandwidth (still using
            a maximum of 2NW tapers)

        Returns
        -------
        freq_multitaper: numpy.ndarray
            The frequency mid-bins of the PSD amplitudes

        psd_multitaper: numpy.ndarray
            The value of the PSD amplitudes at the given frequency mid-bins

        Notes
        -----
        Does not currently support adaptive weighting or jack-knife estimates.
        """

        lc.apply_gtis()  # Remove bins with missing data

        if NW < 0.5:
            raise ValueError("The value of normalized half-bandwidth " "should be greater than 0.5")

        Kmax = int(2 * NW)

        dpss_tapers, eigvals = signal.windows.dpss(
            M=lc.n, NW=NW, Kmax=Kmax, sym=False, return_ratios=True
        )

        if low_bias:
            selected_tapers = eigvals > 0.9
            if not selected_tapers.any():
                simon("Could not properly use low_bias, " "keeping the lowest-bias taper")
                selected_tapers = [np.argmax(eigvals)]

            eigvals = eigvals[selected_tapers]
            dpss_tapers = dpss_tapers[selected_tapers, :]

        print(f"Using {len(eigvals)} DPSS windows for " "multitaper spectrum estimator")

        dpss_data_interpolated = []

        data_irregular = lc.counts - np.mean(lc.counts)
        times_regular = np.linspace(lc.time[0], lc.time[-1], lc.n)

        for dpss_taper in dpss_tapers:
            cubic_spline_interp = interpolate.InterpolatedUnivariateSpline(
                times_regular, dpss_taper, k=3
            )
            # Interpolating DPSS tapers to IRREGULAR times
            dpss_interpolated = cubic_spline_interp(lc.time)
            dpss_interpolated /= np.sum(dpss_interpolated**2)  # Re normalizing
            # From Springford R implementation
            dpss_interpolated *= np.sqrt(lc.n)
            dpss_interpolated *= data_irregular
            dpss_data_interpolated.append(dpss_interpolated)

        psd_multitaper_ls = []

        tseg = lc.time[-1] - lc.time[0]

        # These are the frequencies given in Springford et al. (2020)
        # freq_mtls = np.arange(0, lc.times.shape[0])*(1/tseg) # This works
        # freq_mtls = freq_mtls[1:]

        # The frequencies rest of Stingray uses
        freq_multitaper_ls = rfftfreq(n=lc.n, d=lc.dt)[1:-1]  # Avoiding zero

        for values in dpss_data_interpolated:
            psd = LombScargle(lc.time, values).power(
                frequency=freq_multitaper_ls, normalization="psd"
            )

            # Normalization in Springford et al. (2020)
            # psd *= 0.5 * tseg / lc.time.shape[0]

            psd_multitaper_ls.append(psd)

        psd_multitaper_ls = np.array(psd_multitaper_ls)
        psd_multitaper_ls = np.mean(psd_multitaper_ls, axis=-2)

        self.jk_var_deg_freedom = None
        self.eigvals = eigvals

        return freq_multitaper_ls, psd_multitaper_ls

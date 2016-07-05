from __future__ import division
import numpy as np
import scipy
import scipy.stats
import scipy.fftpack
import scipy.optimize
import logging

import stingray.lightcurve as lightcurve
import stingray.utils as utils
from stingray.utils import simon

__all__ = ["Powerspectrum", "AveragedPowerspectrum"]


def classical_pvalue(power, nspec):
    """
    Compute the probability of detecting the current power under
    the assumption that there is no periodic oscillation in the data.

    This computes the single-trial p-value that the power was
    observed under the null hypothesis that there is no signal in
    the data.

    Important: the underlying assumptions that make this calculation valid
    are:
    (1) the powers in the power spectrum follow a chi-square distribution
    (2) the power spectrum is normalized according to Leahy (1984), such
    that the powers have a mean of 2 and a variance of 4
    (3) there is only white noise in the light curve. That is, there is no
    aperiodic variability that would change the overall shape of the power
    spectrum.

    Also note that the p-value is for a *single trial*, i.e. the power
    currently being tested. If more than one power or more than one power
    spectrum are being tested, the resulting p-value must be corrected for the
    number of trials (Bonferroni correction).

    Mathematical formulation in Groth, 1975.
    Original implementation in IDL by Anna L. Watts.

    Parameters
    ----------
    power :  float
        The squared Fourier amplitude of a spectrum to be evaluated

    nspec : int
        The number of spectra or frequency bins averaged in `power`.
        This matters because averaging spectra or frequency bins increases
        the signal-to-noise ratio, i.e. makes the statistical distributions
        of the noise narrower, such that a smaller power might be very
        significant in averaged spectra even though it would not be in a single
        power spectrum.

    """

    assert np.isfinite(power), "power must be a finite floating point number!"
    assert power > 0, "power must be a positive real number!"
    assert np.isfinite(nspec), "nspec must be a finite integer number"
    assert nspec >= 1, "nspec must be larger or equal to 1"
    assert np.isclose(nspec % 1, 0), "nspec must be an integer number!"

    # If the power is really big, it's safe to say it's significant,
    # and the p-value will be nearly zero
    if (power*nspec) > 30000:
        simon("Probability of no signal too miniscule to calculate.")
        return 0.0

    else:
        pval = _pavnosigfun(power, nspec)
        return pval


def _pavnosigfun(power, nspec):
    """
    Helper function doing the actual calculation of the p-value.
    """
    sum = 0.0
    m = nspec - 1

    pn = power * nspec

    while m >= 0:

        s = 0.0
        for i in range(int(m)-1):
            s += np.log(float(m-i))

        logterm = m*np.log(pn/2) - pn/2 - s
        term = np.exp(logterm)
        ratio = sum / term

        if ratio > 1.0e15:
            return sum

        sum += term
        m -= 1

    return sum


class Powerspectrum(object):

    def __init__(self, lc=None, norm='rms'):
        """
        Make a Periodogram (power spectrum) from a (binned) light curve.
        Periodograms can be Leahy normalized or fractional rms normalized.
        You can also make an empty Periodogram object to populate with your
        own fourier-transformed data (this can sometimes be useful when making
        binned periodograms).

        Parameters
        ----------
        lc: lightcurve.Lightcurve object, optional, default None
            The light curve data to be Fourier-transformed.

        norm: {"leahy" | "rms"}, optional, default "rms"
            The normaliation of the periodogram to be used. Options are
            "leahy" or "rms", default is "rms".


        Attributes
        ----------
        norm: {"leahy" | "rms"}
            the normalization of the periodogram

        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        ps: numpy.ndarray
            The array of normalized squared absolute values of Fourier
            amplitudes

        df: float
            The frequency resolution

        m: int
            The number of averaged powers in each bin

        n: int
            The number of data points in the light curve

        nphots: float
            The total number of photons in the light curve


        """

        # TODO: One should be able to convert from rms to Leahy and do this
        # anyway!
        assert isinstance(norm, str), "norm is not a string!"

        assert norm.lower() in ["rms", "leahy"], \
            "norm must be either 'rms' or 'leahy'!"

        self.norm = norm.lower()

        # check if input data is a Lightcurve object, if not make one or
        # make an empty Periodogram object if lc == time == counts == None
        if lc is not None:
            pass
        else:
            self.freq = None
            self.ps = None
            self.df = None
            self.nphots = None
            self.m = 1
            self.n = None
            return

        self._make_powerspectrum(lc)

    def _make_powerspectrum(self, lc):

        # make sure my inputs work!
        assert isinstance(lc, lightcurve.Lightcurve), \
            "lc must be a lightcurve.Lightcurve object!"

        # total number of photons is the sum of the
        # counts in the light curve
        self.nphots = np.sum(lc.counts)

        # the number of data points in the light curve
        self.n = lc.counts.shape[0]

        # the frequency resolution
        self.df = 1 / lc.tseg

        # the number of averaged periodograms in the final output
        # This should *always* be 1 here
        self.m = 1

        # make the actual Fourier transform
        self.freq, self.unnorm_powers = self._fourier_modulus(lc)

        # normalize to either Leahy or rms normalization
        self.ps = self._normalize_periodogram(self.unnorm_powers, lc)

    def _fourier_modulus(self, lc):
        """
        Fourier transform the light curve, then square the
        absolute value of the Fourier amplitudes.

        Parameters
        ----------
        lc: lightcurve.Lightcurve object
            The light curve to be Fourier transformed

        Returns
        -------
        fr: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        """
        fourier = scipy.fftpack.fft(lc.counts)  # do Fourier transform
        freqs = scipy.fftpack.fftfreq(lc.counts.shape[0], lc.dt)
        fr = np.abs(fourier[freqs > 0])**2.
        return freqs[freqs > 0], fr

    def _normalize_periodogram(self, unnorm_powers, lc):
        """
        Normalize the periodogram to either Leahy or RMS normalization.
        In Leahy normalization, the periodogram is normalized in such a way
        that a flat light curve of Poissonian data will make a realization of
        the power spectrum in which the powers are distributed as Chi^2 with
        two degrees of freedom (with a mean of 2 and a variance of 4).

        In rms normalization, the periodogram will be normalized such that
        the integral of the periodogram will equal the total variance in the
        light curve divided by the mean of the light curve squared.

        Parameters
        ----------
        unnorm_powers: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        lc: lightcurve.Lightcurve object
            The input light curve


        Returns
        -------
        ps: numpy.nd.array
            The normalized periodogram
        """
        if self.norm.lower() == 'leahy':
            p = unnorm_powers
            ps = 2 * p / self.nphots

        elif self.norm.lower() == 'rms':
            p = unnorm_powers / np.float(self.n**2)
            ps = (p*2*lc.tseg) / (np.mean(lc.counts)**2)

        else:
            raise Exception("Normalization not recognized!")

        return ps

    def rebin(self, df, method="mean"):
        """
        Rebin the periodogram to a new frequency resolution df.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Returns
        -------
        bin_ps = Periodogram object
            The newly binned periodogram
        """

        # rebin power spectrum to new resolution
        binfreq, binps, step_size = utils.rebin_data(self.freq[1:],
                                                     self.ps[1:], df,
                                                     method=method)

        # make an empty periodogram object
        bin_ps = Powerspectrum()

        # store the binned periodogram in the new object
        bin_ps.norm = self.norm
        bin_ps.freq = np.hstack([binfreq[0] - self.df, binfreq])
        bin_ps.ps = np.hstack([self.ps[0], binps])
        bin_ps.df = df
        bin_ps.n = self.n
        bin_ps.nphots = self.nphots
        bin_ps.m = int(step_size)

        return bin_ps

    def rebin_log(self, f=0.01):
        """
        Logarithmic rebin of the periodogram.
        The new frequency depends on the previous frequency
        modified by a factor f:

        dnu_j = dnu_{j-1}*(1+f)

        Parameters
        ----------
        f: float, optional, default 0.01
            parameter that steers the frequency resolution


        Returns
        -------
        binfreq: numpy.ndarray
            the binned frequencies

        binps: numpy.ndarray
            the binned powers

        nsamples: numpy.ndarray
            the samples of the original periodogramincluded in each
            frequency bin
        """

        minfreq = self.freq[1] * 0.5  # frequency to start from
        maxfreq = self.freq[-1]  # maximum frequency to end
        binfreq = [minfreq, minfreq + self.df]  # first
        df = self.freq[1]  # the frequency resolution of the first bin

        # until we reach the maximum frequency, increase the width of each
        # frequency bin by f
        while binfreq[-1] <= maxfreq:
            binfreq.append(binfreq[-1] + df*(1.0+f))
            df = binfreq[-1] - binfreq[-2]

        # compute the mean of the powers that fall into each new frequency bin
        binps, bin_edges, binno = scipy.stats.binned_statistic(
            self.freq, self.ps, statistic="mean", bins=binfreq)

        # compute the number of powers in each frequency bin
        nsamples = np.array([len(binno[np.where(binno == i)[0]])
                             for i in range(np.max(binno))])

        # the frequency resolution
        df = np.diff(binfreq)

        # shift the lower bin edges to the middle of the bin and drop the
        # last right bin edge
        binfreq = binfreq[:-1] + df/2

        return binfreq, binps, nsamples

    def compute_rms(self, min_freq, max_freq):
        """
        Compute the fractional rms amplitude in the periodgram
        between two frequencies.

        Parameters
        ----------
        min_freq: float
            The lower frequency bound for the calculation

        max_freq: float
            The upper frequency bound for the calculation

        Returns
        -------
        rms: float
            The fractional rms amplitude contained between min_freq and
            max_freq

        """
        # assert min_freq >= self.freq[0], "Lower frequency bound must be " \
        #                                 "larger or equal the minimum " \
        #                                 "frequency in the periodogram!"

        # assert max_freq <= self.freq[-1], "Upper frequency bound must be " \
        #                                 "smaller or equal the maximum " \
        #                                 "frequency in the periodogram!"

        minind = self.freq.searchsorted(min_freq)
        maxind = self.freq.searchsorted(max_freq)
        powers = self.ps[minind:maxind]
        if self.norm.lower() == 'leahy':
            rms = np.sqrt(np.sum(powers)/self.nphots)
        elif self.norm.lower() == "rms":
            rms = np.sqrt(np.sum(powers*self.df))
        else:
            raise Exception("Normalization not recognized!")

        rms_err = self._rms_error(powers)

        return rms, rms_err

    def _rms_error(self, powers):
        """
        Compute the error on the fractional rms amplitude using error
        propagation.
        Note: this uses the actual measured powers, which is not
        strictly correct. We should be using the underlying power spectrum,
        but in the absence of an estimate of that, this will have to do.

        Parameters
        ----------
        powers: iterable
            The list of powers used to compute the fractional rms amplitude.


        Returns
        -------
        delta_rms: float
            the error on the fractional rms amplitude
        """
        p_err = scipy.stats.chi2(2.0*self.m).var() * powers / self.m
        drms_dp = 1 / (2*np.sqrt(np.sum(powers)*self.df))
        delta_rms = np.sum(p_err*drms_dp*self.df)
        return delta_rms

    def classical_significances(self, threshold=1, trial_correction=False):
        """
        Compute the classical significances for the powers in the power
        spectrum, assuming an underlying noise distribution that follows a
        chi-square distributions with 2M degrees of freedom, where M is the
        number of powers averaged in each bin.

        Note that this function will *only* produce correct results when the
        following underlying assumptions are fulfilled:
        (1) The power spectrum is Leahy-normalized
        (2) There is no source of variability in the data other than the
        periodic signal to be determined with this method. This is important!
        If there are other sources of (aperiodic) variability in the data, this
        method will *not* produce correct results, but instead produce a large
        number of spurious false positive detections!
        (3) There are no significant instrumental effects changing the
        statistical distribution of the powers (e.g. pile-up or dead time)

        By default, the method produces (index,p-values) for all powers in
        the power spectrum, where index is the numerical index of the power in
        question. If a `threshold` is set, then only powers with p-values
        *below* that threshold with their respective indices. If
        `trial_correction` is set to True, then the threshold will be corrected
        for the number of trials (frequencies) in the power spectrum before
        being used.

        Parameters
        ----------
        threshold : float
            The threshold to be used when reporting p-values of potentially
            significant powers. Must be between 0 and 1.
            Default is 1 (all p-values will be reported).

        trial_correction : bool
            A Boolean flag that sets whether the `threshold` will be correted
            by the number of frequencies before being applied. This decreases
            the threshold (p-values need to be lower to count as significant).
            Default is False (report all powers) though for any application
            where `threshold` is set to something meaningful, this should also
            be applied!

        Returns
        -------
        pvals : iterable
            A list of (index, p-value) tuples for all powers that have p-values
            lower than the threshold specified in `threshold`.

        """
        assert self.norm == "leahy", "This method only works on " \
                                     "Leahy-normalized power spectra!"

        # calculate p-values for all powers
        # leave out zeroth power since it just encodes the number of photons!
        pv = np.array([classical_pvalue(power, self.m)
                      for power in self.ps])

        # if trial correction is used, then correct the threshold for
        # the number of powers in the power spectrum
        if trial_correction:
            threshold /= self.ps.shape[0]

        # need to add 1 to the indices to make up for the fact that
        # we left out the first power above!
        indices = np.where(pv < threshold)[0]

        pvals = np.vstack([pv[indices], indices])

        return pvals


class AveragedPowerspectrum(Powerspectrum):

    def __init__(self, lc, segment_size, norm="rms"):
        """
        Make an averaged periodogram from a light curve by segmenting the light
        curve, Fourier-transforming each segment and then averaging the
        resulting periodograms.

        Parameters
        ----------
        lc: lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            The light curve data to be Fourier-transformed.

        segment_size: float
            The size of each segment to average. Note that if the total
            duration of each Lightcurve object in lc is not an integer multiple
            of the segment_size, then any fraction left-over at the end of the
            time series will be lost.

        norm: {"leahy" | "rms"}, optional, default "rms"
            The normaliation of the periodogram to be used. Options are
            "leahy" or "rms", default is "rms".


        Attributes
        ----------
        norm: {"leahy" | "rms"}
            the normalization of the periodogram

        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        ps: numpy.ndarray
            The array of normalized squared absolute values of Fourier
            amplitudes

        df: float
            The frequency resolution

        m: int
            The number of averaged periodograms

        n: int
            The number of data points in the light curve

        nphots: float
            The total number of photons in the light curve


        """
        assert np.isfinite(segment_size), "segment_size must be finite!"

        self.norm = norm.lower()
        self.segment_size = segment_size

        Powerspectrum.__init__(self, lc, norm)

        return

    def _make_segment_psd(self, lc, segment_size):
        assert isinstance(lc, lightcurve.Lightcurve)

        # number of bins per segment
        nbins = int(segment_size/lc.dt)

        start_ind = 0
        end_ind = nbins

        ps_all = []
        nphots_all = []
        while end_ind <= lc.counts.shape[0]:
            time = lc.times[start_ind:end_ind]
            counts = lc.counts[start_ind:end_ind]
            lc_seg = lightcurve.Lightcurve(time, counts)
            ps_seg = Powerspectrum(lc_seg, norm=self.norm)
            ps_all.append(ps_seg)
            nphots_all.append(np.sum(lc_seg.counts))
            start_ind += nbins
            end_ind += nbins

        return ps_all, nphots_all

    def _make_powerspectrum(self, lc):

        # chop light curves into segments
        if isinstance(lc, lightcurve.Lightcurve):
            self.ps_all, nphots_all = self._make_segment_psd(lc,
                                                        self.segment_size)
        else:
            self.ps_all, nphots_all = [], []
            for lc_seg in lc:
                ps_sep, nphots_sep = self._make_segment_psd(lc_seg,
                                                            self.segment_size)

                self.ps_all.append(ps_sep)
                nphots_all.append(nphots_sep)

            self.ps_all = np.hstack(self.ps_all)
            nphots_all = np.hstack(nphots_all)

        m = len(self.ps_all)
        nphots = np.mean(nphots_all)
        ps_avg = np.zeros_like(self.ps_all[0].ps)
        for ps in self.ps_all:
            ps_avg += ps.ps

        ps_avg /= np.float(m)

        self.freq = self.ps_all[0].freq
        self.ps = ps_avg
        self.m = m
        self.df = self.ps_all[0].df
        self.n = self.ps_all[0].n
        self.nphots = nphots

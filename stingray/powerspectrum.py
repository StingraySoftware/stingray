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
from stingray.crossspectrum import Crossspectrum, AveragedCrossspectrum


__all__ = ["Powerspectrum", "AveragedPowerspectrum", "DynamicalPowerspectrum"]


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

    Returns
    -------
    pval : float
        The classical p-value of the observed power being consistent with
        the null hypothesis of white noise

    """
    if not np.isfinite(power):
        raise ValueError("power must be a finite floating point number!")

    if power < 0:
        raise ValueError("power must be a positive real number!")

    if not np.isfinite(nspec):
        raise ValueError("nspec must be a finite integer number")

    if nspec < 1:
        raise ValueError("nspec must be larger or equal to 1")

    if not np.isclose(nspec % 1, 0):
        raise ValueError("nspec must be an integer number!")

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


class Powerspectrum(Crossspectrum):

    def __init__(self, lc=None, norm='frac'):
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

        power: numpy.ndarray
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
        Crossspectrum.__init__(self, lc1=lc, lc2=lc, norm=norm)
        self.nphots = self.nphots1

    def rebin(self, df, method="mean"):
        bin_ps = Crossspectrum.rebin(self, df=df, method=method)
        bin_ps.nphots = bin_ps.nphots1

        return bin_ps

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
        minind = self.freq.searchsorted(min_freq)
        maxind = self.freq.searchsorted(max_freq)
        powers = self.power[minind:maxind]

        if self.norm.lower() == 'leahy':
            rms = np.sqrt(np.sum(powers)/self.nphots)

        elif self.norm.lower() == "frac":
            rms = np.sqrt(np.sum(powers*self.df))
        else:
            raise TypeError("Normalization not recognized!")

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
        if not self.norm == "leahy":
            raise ValueError("This method only works on "
                             "Leahy-normalized power spectra!")

        # calculate p-values for all powers
        # leave out zeroth power since it just encodes the number of photons!
        pv = np.array([classical_pvalue(power, self.m)
                      for power in self.power])

        # if trial correction is used, then correct the threshold for
        # the number of powers in the power spectrum
        if trial_correction:
            threshold /= self.power.shape[0]

        # need to add 1 to the indices to make up for the fact that
        # we left out the first power above!
        indices = np.where(pv < threshold)[0]

        pvals = np.vstack([pv[indices], indices])

        return pvals


class AveragedPowerspectrum(AveragedCrossspectrum, Powerspectrum):

    def __init__(self, lc, segment_size, norm="frac"):
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

        power: numpy.ndarray
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

        self.type = "powerspectrum"

        assert isinstance(norm, str), "norm is not a string!"

        assert norm.lower() in ["frac", "abs", "leahy", "none"], \
                "norm must be 'frac', 'abs', 'leahy', or 'none'!"

        self.norm = norm.lower()

        assert np.isfinite(segment_size), "segment_size must be finite!"

        self.segment_size = segment_size

        Powerspectrum.__init__(self, lc, norm)

        return

    def _make_segment_spectrum(self, lc, segment_size):

        if not isinstance(lc, lightcurve.Lightcurve):
            raise TypeError("lc must be a lightcurve.Lightcurve object")

        # number of bins per segment
        nbins = int(segment_size/lc.dt)

        start_ind = 0
        end_ind = nbins

        power_all = []
        nphots_all = []
        while end_ind <= lc.counts.shape[0]:
            time = lc.time[start_ind:end_ind]
            counts = lc.counts[start_ind:end_ind]
            lc_seg = lightcurve.Lightcurve(time, counts)
            power_seg = Powerspectrum(lc_seg, norm=self.norm)
            power_all.append(power_seg)
            nphots_all.append(np.sum(lc_seg.counts))
            start_ind += nbins
            end_ind += nbins

        return power_all, nphots_all


class DynamicalPowerspectrum(AveragedPowerspectrum):
    def __init__(self, lc, segment_size, norm="frac"):
        """
        Parameters
        ----------
        lc : lightcurve.Lightcurve object
            The time series of which the Dynamical powerspectrum is
            to be calculated.

        segment_size : float, default 1
             Length of the segment of light curve, default value is 1 second.

        norm: {"leahy" | "rms"}, optional, default "rms"
            The normaliation of the periodogram to be used. Options are
            "leahy" or "rms", default is "rms".

        Attributes
        ----------
        matrix : np.ndarray
            The matrix with fourier frequencies in the first column and time
            in the second column.
        """
        if segment_size < 2*lc.dt:
            raise ValueError("Length of the segment is too short to form a "
                             "light curve!")

        AveragedPowerspectrum.__init__(self, lc, segment_size, norm)
        power_all, _ = AveragedPowerspectrum._make_segment_spectrum(
            self, lc, segment_size)
        self._make_matrix(power_all)

    def _make_matrix(self, power_all):
        """Create the matrix with freq and time columns."""
        row1 = power_all[0].freq
        row2 = list(range(0, self.segment_size*len(row1), self.segment_size))
        self.matrix = np.array([row1, row2]).T

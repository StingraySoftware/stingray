# -*- coding: utf-8 -*-
from __future__ import division
import collections
import numpy as np

from stingray import Lightcurve
import stingray.utils as utils

__all__ = ['Covariancespectrum', 'AveragedCovariancespectrum']


class Covariancespectrum(object):

    def __init__(self, event_list, dt, band_interest=None,
                 ref_band_interest=None, std=None):
        """
        Parameters
        ----------
        event_list : numpy 2D array
            A numpy 2D array with first column as time of arrival and second
            column as photon energies associated.
            Note : The event list must be in sorted order with respect to the
            times of arrivals.

        dt : float
            The time resolution of the Lightcurve formed from the energy bin.

        band_interest : iterable of tuples, default All
            An iterable of tuples with minimum and maximum values of the range
            in the band of interest. e.g list of tuples, tuple of tuples.

        ref_band_interest : tuple of reference band range, default All
            A tuple with minimum and maximum values of the range in the band
            of interest in reference channel.

        std : float or np.array or list of numbers
            The term std is used to cacluate the excess variance of a band.
            If std is set to None, default Poisson case is taken and the
            std is calculated as `mean(lc)**0.5`. In the case of a single
            float as input, the same is used as the standard deviation which
            is also used as the std. And if the std is an iterable of
            numbers, their mean is used for the same purpose.


        Attributes
        ----------
        energy_events : dictionary
            A dictionary with energy bins as keys and time of arrivals of
            photons with the same energy as value.

        energy_covar : dictionary
            A dictionary with mid point of band_interest and their covariance
            computed with their individual reference band. The covaraince
            values are normalized.

        unnorm_covar : np.ndarray
            An array of arrays with mid point band_interest and their
            covariance. It is the array-form of the dictionary `energy_covar`.
            The covariance values are unnormalized.

        covar : np.ndarray
            Normalized covaraiance spectrum.

        covar_error : np.ndarray
            Errors of the normalized covariance spectrum.

        min_time : int
            Time of arrival of the earliest photon.

        max_time : int
            Time of arrival of the last photon.

        min_energy : float
            Energy of the photon with the minimum energy.

        max_energy : float
            Energy of the photon with the maximum energy.

        Reference
        ---------
        [1] Wilkinson, T. and Uttley, P. (2009), Accretion disc variability
            in the hard state of black hole X-ray binaries. Monthly Notices
            of the Royal Astronomical Society, 397: 666–676.
            doi: 10.1111/j.1365-2966.2009.15008.x

        Examples
        --------
        See https://github.com/StingraySoftware/notebooks repository for
        detailed notebooks on the code.
        """

        # This parameter is used to identify whether the current object is
        # an instance of Covariancespectrum or AveragedCovariancespectrum.
        self.avg_covar = False

        self._init_vars(event_list, dt, band_interest,
                        ref_band_interest, std)

        # A dictionary with energy bin as key and events as value of the key
        self.energy_events = {}

        self._construct_energy_events(self.energy_events)

        self._update_energy_events(self.energy_events)

        # The dictionary with covariance spectrum for each energy bin
        self.energy_covar = {}

        self._construct_energy_covar(self.energy_events, self.energy_covar)

    def _init_vars(self, event_list, dt, band_interest,
                   ref_band_interest, std):
        """
        Check for consistency with input variables and declare public ones.
        """
        if not np.all(np.diff(event_list, axis=0).T[0] >= 0):
            utils.simon("The event list must be sorted with respect to "
                        "times of arrivals.")
            event_list = event_list[event_list[:, 0].argsort()]

        self.event_list = event_list

        self.event_list_T = event_list.T

        self._init_special_vars()

        if ref_band_interest is None:
            ref_band_interest = (self.min_energy, self.max_energy)

        assert type(ref_band_interest) in (list, tuple), "Ref Band interest " \
                                                         "should be either " \
                                                         "tuple or list."

        assert len(ref_band_interest) == 2, "Band interest should be a tuple" \
                                            " with min and max energy value " \
                                            "for the reference band."
        self.ref_band_interest = ref_band_interest

        if band_interest is not None:
            for element in list(band_interest):
                assert type(element) in (list, tuple), \
                    "band_interest should be iterable of either tuple or list."
                assert len(element) == 2, "Band interest should be a tuple " \
                                          "with min and max energy values."

        self.band_interest = band_interest
        self.dt = dt

        self.std = std

    def _init_special_vars(self, T_start=None, T_end=None):
        """
        Method to set mininum and maximum time and energy parameters. It has
        been separated from the main init method due to multiple calls from
        AveragedCovariancespectrum.
        """
        self.min_energy = np.min(self.event_list_T[1][T_start:T_end])
        self.max_energy = np.max(self.event_list_T[1][T_start:T_end])
        self.min_time = np.min(self.event_list_T[0][T_start:T_end])
        self.max_time = np.max(self.event_list_T[0][T_start:T_end])

    def _construct_energy_events(self, energy_events, T_start=None, T_end=None):
        # The T_start and T_end parameters are for the purpose of
        # AveragedCovariancespectrum where the range of consideration
        # is defined.
        event_list_T = np.array([self.event_list_T[0][T_start: T_end],
                                 self.event_list_T[1][T_start: T_end]])
        least_count = np.diff(np.unique(event_list_T[1])).min()

        # An array of unique energy values
        unique_energy = np.unique(event_list_T[1])

        for i in range(len(unique_energy) - 1):
            energy_events[unique_energy[i] + least_count*0.5] = []

        # Add time of arrivals to corresponding energy bins
        # For each bin except the last one, the lower bound is included and
        # the upper bound is excluded.
        for energy in energy_events.keys():
            # The last energy bin
            if energy == self.max_energy - least_count*0.5:
                toa = event_list_T[0][np.logical_and(
                    event_list_T[1] >= energy - least_count*0.5,
                    event_list_T[1] <= energy + least_count*0.5)]
                energy_events[energy] = sorted(toa)
            else:
                toa = event_list_T[0][np.logical_and(
                    event_list_T[1] >= energy - least_count*0.5,
                    event_list_T[1] < energy + least_count*0.5)]
                energy_events[energy] = sorted(toa)

    def _update_energy_events(self, energy_events):
        """
        In case of a specific band interest, merge the required energy bins
        into one with the new key as the mid-point of the band interest.
        """
        if self.band_interest is not None:
            energy_events_ = {}
            for band in list(self.band_interest):
                mid_bin = (band[0] + band[1]) / 2
                energy_events_[mid_bin] = []

                # Modify self.energy_events to form a band with one key
                for key in list(self.energy_events.keys()):
                    if key >= band[0] and key <= band[1]:
                        energy_events_[mid_bin] += energy_events[key]
                        del energy_events[key]

            energy_events.update(energy_events_)

    def _init_energy_covar(self, energy_events, energy_covar):
        """
        Initialize the energy_covar dictionary for further computations.
        """
        # Initialize it with empty mapping
        if self.band_interest is None:
            for key in energy_events.keys():
                energy_covar[key] = []
        else:
            for band in list(self.band_interest):
                mid_bin = (band[0] + band[1]) / 2
                energy_covar[mid_bin] = []

        if not self.avg_covar:
            # Error in covariance
            self.covar_error = {}

    def _construct_energy_covar(self, energy_events, energy_covar,
                                covar_error=None, xs_var=None):
        """Form the actual output covaraince dictionary and array."""
        self._init_energy_covar(energy_events, energy_covar)

        if not self.avg_covar:
            xs_var = dict()

        for energy in energy_covar.keys():
            lc = Lightcurve.make_lightcurve(
                    energy_events[energy], self.dt, tstart=self.min_time,
                    tseg=self.max_time - self.min_time)

            # Calculating timestamps for lc_ref
            toa_ref = []
            for key, value in energy_events.items():
                if key >= self.ref_band_interest[0] and \
                        key <= self.ref_band_interest[1]:
                    if key != energy:
                        toa_ref.extend(value)

            toa_ref = np.array(sorted(toa_ref))

            lc_ref = Lightcurve.make_lightcurve(
                    toa_ref, self.dt, tstart=self.min_time,
                    tseg=self.max_time - self.min_time)

            assert len(lc.time) == len(lc_ref.time)

            covar = self._compute_covariance(lc, lc_ref)

            energy_covar[energy] = covar
            if not self.avg_covar:
                self.covar_error[energy] = self._calculate_covariance_error(
                                                lc, lc_ref)
            else:
                covar_error[energy] = self._calculate_covariance_error(
                                           lc, lc_ref)
            # Excess variance in ref band
            xs_var[energy] = self._calculate_excess_variance(lc_ref)

        for key, value in energy_covar.items():
            if not xs_var[key] > 0:
                utils.simon("The excess variance in the reference band is "
                            "negative. This implies that the reference "
                            "band was badly chosen. Beware that the "
                            "covariance spectra will have NaNs!")

        if not self.avg_covar:
            self.unnorm_covar = np.vstack(energy_covar.items())
            energy_covar[key] = value / (xs_var[key])**0.5

            self.covar = np.vstack(energy_covar.items())

            self.covar_error = np.vstack(self.covar_error.items())

    def _calculate_excess_variance(self, lc):
        """Calculate excess variance in a band with the standard deviation."""
        std = self._calculate_std(lc)
        return np.var(lc) - std**2

    def _calculate_std(self, lc):
        """Return std calculated for the possible types of `std`"""
        if self.std is None:
            std = np.mean(lc)**0.5
        elif isinstance(self.std, collections.Iterable):
            std = np.mean(self.std)  # Iterable of numbers
        else:  # Single float number
            std = self.std

        return std

    def _compute_covariance(self, lc1, lc2):
        """Calculate and return the covariance between two time series."""
        return np.cov(lc1.counts, lc2.counts)[0][1]

    def _calculate_covariance_error(self, lc_x, lc_y):
        """Calculate the error of the normalized covariance spectrum."""
        # Excess Variance of reference band
        xs_x = self._calculate_excess_variance(lc_x)
        # Standard deviation of light curve
        err_y = self._calculate_std(lc_y)
        # Excess Variance of reference band
        xs_y = self._calculate_excess_variance(lc_y)
        # Standard deviation of light curve
        err_x = self._calculate_std(lc_x)
        # Number of time bins in lightcurve
        N = lc_x.ncounts
        # Number of segments averaged
        if not self.avg_covar:
            M = 1
        else:
            M = self.segment_size

        num = xs_x*err_y + xs_y*err_x + err_x*err_y
        denom = N * M * xs_y

        return (num / denom)**0.5


class AveragedCovariancespectrum(Covariancespectrum):
    def __init__(self, event_list, dt, segment_size, band_interest=None,
                 ref_band_interest=None, std=None):
        """
        Make an averaged covariance spectrum by segmenting the light curve
        formed, calculating covariance for each segment and then averaging
        the resulting covariance spectra.

        Parameters
        ----------
        event_list : numpy 2D array
            A numpy 2D array with first column as time of arrival and second
            column as photon energies associated.
            Note : The event list must be in sorted order with respect to the
            times of arrivals.

        dt : float
            The time resolution of the Lightcurve formed from the energy bin.

        segment_size : float
            The size of each segment to average. Note that if the total
            duration of each Lightcurve object formed is not an integer
            multiple of the segment_size, then any fraction left-over at the
            end of the time series will be lost.


        band_interest : iterable of tuples, default All
            An iterable of tuples with minimum and maximum values of the range
            in the band of interest. e.g list of tuples, tuple of tuples.

        ref_band_interest : tuple of reference band range, default All
            A tuple with minimum and maximum values of the range in the band
            of interest in reference channel.

        std : float or np.array or list of numbers
            The term std is used to cacluate the excess variance of a band.
            If std is set to None, default Poisson case is taken and the
            std is calculated as `mean(lc)**0.5`. In the case of a single
            float as input, the same is used as the standard deviation which
            is also used as the std. And if the std is an iterable of
            numbers, their mean is used for the same purpose.


        Attributes
        ----------
        energy_events : dictionary
            A dictionary with energy bins as keys and time of arrivals of
            photons with the same energy as value.

        energy_covar : dictionary
            A dictionary with mid point of band_interest and their covariance
            computed with their individual reference band. The covaraince
            values are normalized.

        unnorm_covar : np.ndarray
            An array of arrays with mid point band_interest and their
            covariance. It is the array-form of the dictionary `energy_covar`.
            The covariance values are unnormalized.

        covar : np.ndarray
            Normalized covaraiance spectrum.

        covar_error : np.ndarray
            Errors of the normalized covariance spectrum.

        min_time : int
            Time of arrival of the earliest photon.

        max_time : int
            Time of arrival of the last photon.

        min_energy : float
            Energy of the photon with the minimum energy.

        max_energy : float
            Energy of the photon with the maximum energy.

        """
        # Set parameter to distinguish between parent class and derived class.
        self.avg_covar = True

        self._init_vars(event_list, dt, band_interest, ref_band_interest, std)
        self.segment_size = segment_size

        self._make_averaged_covar_spectrum()

    def _make_averaged_covar_spectrum(self):
        """
        Calls methods from base class for every segment and calculates averaged
        covariance and error.
        """
        self.nbins = int((self.max_time - self.min_time + 1) / self.segment_size)

        for n in range(self.nbins):
            tstart = self.min_time + n*self.segment_size
            tend = self.min_time + self.segment_size*(n+1) - 1
            indices = np.intersect1d(np.where(self.event_list_T[0] >= tstart),
                                     np.where(self.event_list_T[0] <= tend))

            # Set minimum and maximum values for the specified indices value
            self._init_special_vars(T_start=indices[0], T_end=indices[-1]+1)

            energy_events = {}

            self._construct_energy_events(energy_events, T_start=indices[0],
                                          T_end=indices[-1]+1)

            self._update_energy_events(energy_events)

            energy_covar = {}
            xs_var = {}
            covar_error = {}
            self._construct_energy_covar(energy_events, energy_covar,
                                         covar_error, xs_var)

            if n == 0:  # Declare
                self.energy_covar = energy_covar
                self.xs_var = xs_var
                self.covar_error = covar_error
            else:  # Sum up
                for key in energy_covar.keys():
                    self.energy_covar[key] = self.energy_covar.get(key, 0) + \
                                             energy_covar[key]
                    self.xs_var[key] = self.xs_var.get(key, 0) + xs_var[key]
                    self.covar_error[key] = self.covar_error.get(key, 0) + \
                                            covar_error[key]

        # Now divide with total number of bins for averaging
        for key in self.energy_covar.keys():
            self.energy_covar[key] /= self.nbins
            self.xs_var[key] /= self.nbins
            self.covar_error[key] /= self.nbins

        self.unnorm_covar = np.vstack(self.energy_covar.items())

        for key, value in self.energy_covar.items():
            self.energy_covar[key] = value / (self.xs_var[key])**0.5

        self.covar = np.vstack(self.energy_covar.items())
        self.covar_error = np.vstack(self.covar_error.items())

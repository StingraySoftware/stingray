from __future__ import division

import numpy as np

from stingray import Lightcurve
import stingray.utils as utils


class Covariancespectrum(object):

    def __init__(self, event_list, dt, band_interest=None,
                 ref_band_interest=None):
        """
        Parameters
        ----------
        event_list : numpy 2D array
            A numpy 2D array with first column as time of arrival and second
            column as photon energies associated.

        dt : float
            The time resolution of the Lightcurve formed from the energy bin.

        band_interest : iterable of tuples, default All
            An iterable of tuples with minimum and maximum values of the range
            in the band of interest.

        ref_band_interest : tuple of reference band range, default All
            A tuple with minimum and maximum values of the range in the band
            of interest in reference channel.
        """
        min_energy, max_energy = np.min(event_list.T[1]), np.max(event_list.T[1])
        min_time, max_time = np.min(event_list.T[0]), np.max(event_list.T[0])

        if band_interest is None:
            band_interest = (min_energy, max_energy)

        if ref_band_interest is None:
            ref_band_interest = (min_energy, max_energy)

        assert len(ref_band_interest) == 2, "Band interest should be a tuple " \
                                            "with min and max energy value " \
                                            "for the reference band."

        # Sorted by energy values as second row
        event_list_T = event_list[event_list[:, 1].argsort()].T

        least_count = np.diff(np.unique(event_list_T[1])).min()

        # An array of unique energy values
        unique_energy = np.unique(event_list_T[1])

        # A dictionary with energy bin as key and events as value of the key
        energy_events = {}

        for i in range(len(unique_energy) - 1):
            energy_events[unique_energy[i] + least_count*0.5] = []

        # Add time of arrivals to corresponding energy bins
        for energy in energy_events.keys():
            if energy == max_energy - least_count*0.5:  # The last energy bin
                toa = event_list_T[0][np.logical_and(
                    event_list_T[1] >= energy - least_count*0.5,
                    event_list_T[1] <= energy + least_count*0.5)]
                energy_events[energy] = sorted(toa)
            else:
                toa = event_list_T[0][np.logical_and(
                    event_list_T[1] >= energy - least_count*0.5,
                    event_list_T[1] < energy + least_count*0.5)]
                energy_events[energy] = sorted(toa)

        # The dictionary with covariance spectrum for each energy bin
        self.energy_covar = {}

        # Initialize it with empty mapping
        for key in energy_events.keys():
            self.energy_covar[key] = []

        for energy in energy_events.keys():
            lc = Lightcurve.make_lightcurve(energy_events[energy], dt, tstart=min_time, tseg=max_time - min_time)

            # Calculating timestamps for lc_ref
            toa_ref = []
            for key, value in energy_events.items():
                if key >= ref_band_interest[0] and key <= ref_band_interest[1]:
                    if key != energy:
                        toa_ref.extend(value)

            toa_ref = np.array(sorted(toa_ref))

            lc_ref = Lightcurve.make_lightcurve(toa_ref, dt, tstart=min_time, tseg=max_time - min_time)

            assert len(lc.time) == len(lc_ref.time)

            covar = self._compute_covariance(lc, lc_ref)

            self.energy_covar[energy] = covar

        self.covar = np.vstack(self.energy_covar.items())

    def _compute_covariance(self, lc1, lc2):
        return np.cov(lc1.counts, lc2.counts)[0][1]

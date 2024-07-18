# -*- coding: utf-8 -*-

from collections.abc import Iterable

import numpy as np

from stingray import Lightcurve
from stingray.events import EventList
import stingray.utils as utils

__all__ = ["Covariancespectrum", "AveragedCovariancespectrum"]


class Covariancespectrum(object):
    """
          Compute a covariance spectrum for the data. The input data can be
          either in event data or pre-made light curves. Event data can either
          be in the form of a ``numpy.ndarray`` with ``(time stamp, energy)`` pairs or
          a :class:`stingray.events.EventList` object. If light curves are formed ahead
          of time, then a list of :class:`stingray.Lightcurve` objects should be passed to the
          object, ideally one light curve for each band of interest.

          For the case where the data is input as a list of :class:`stingray.Lightcurve` objects,
          the reference band(s) should either be

          1. a single :class:`stingray.Lightcurve` object,
          2. a list of :class:`stingray.Lightcurve` objects with the reference band for each band
             of interest pre-made, or
          3. ``None``, in which case reference bands will
             formed by combining all light curves *except* for the band of interest.

          In the case of event data, ``band_interest`` and ``ref_band_interest`` can
          be (multiple) pairs of energies, and the light curves for the bands of
          interest and reference bands will be produced dynamically.


          Parameters
          ----------
          data : {``numpy.ndarray`` | :class:`stingray.events.EventList` object | list of :class:`stingray.Lightcurve` objects}
              ``data`` contains the time series data, either in the form of a
              2-D array of ``(time stamp, energy)`` pairs for event data, or as a
              list of light curves.
              Note : The event list must be in sorted order with respect to the
              times of arrivals.

          dt : float
              The time resolution of the :class:`stingray.Lightcurve` formed from the energy bin.
              Only used if ``data`` is an event list.

          band_interest : {``None``, iterable of tuples}
              If ``None``, all possible energy values will be assumed to be of
              interest, and a covariance spectrum in the highest resolution
              will be produced.
              Note: if the input is a list of :class:`stingray.Lightcurve` objects, then the user may
              supply their energy values here, for construction of a
              reference band.

          ref_band_interest : {``None``, tuple, :class:`stingray.Lightcurve`, list of :class:`stingray.Lightcurve` objects}
              Defines the reference band to be used for comparison with the
              bands of interest. If ``None``, all bands *except* the band of
              interest will be used for each band of interest, respectively.
              Alternatively, a tuple can be given for event list data, which will
              extract the reference band (always excluding the band of interest),
              or one may put in a single :class:`stingray.Lightcurve` object to be used (the same
              for each band of interest) or a list of :class:`stingray.Lightcurve` objects, one for
              each band of interest.

          std : float or np.array or list of numbers
              The term ``std`` is used to calculate the excess variance of a band.
              If ``std`` is set to ``None``, default Poisson case is taken and the
              std is calculated as ``mean(lc)**0.5``. In the case of a single
              float as input, the same is used as the standard deviation which
              is also used as the std. And if the std is an iterable of
              numbers, their mean is used for the same purpose.

          Attributes
          ----------
          unnorm_covar : np.ndarray
              An array of arrays with mid point ``band_interest`` and their
              covariance. It is the array-form of the dictionary ``energy_covar``.
              The covariance values are unnormalized.

          covar : np.ndarray
              Normalized covariance spectrum.

          covar_error : np.ndarray
              Errors of the normalized covariance spectrum.

          References
          ----------
          [1] Wilkinson, T. and Uttley, P. (2009), Accretion disc variability\
              in the hard state of black hole X-ray binaries. Monthly Notices\
              of the Royal Astronomical Society, 397: 666–676.\
              doi: 10.1111/j.1365-2966.2009.15008.x

          Examples
          --------
          See the `notebooks repository <https://github.com/StingraySoftware/notebooks>`_ for
          detailed notebooks on the code.

    """

    def __init__(self, data, dt=None, band_interest=None, ref_band_interest=None, std=None):
        self.dt = dt
        self.std = std

        # check whether data is an EventList object:
        if isinstance(data, EventList):
            data = np.vstack([data.time, data.energy]).T

        # check whether the data contains a list of Lightcurve objects
        if isinstance(data[0], Lightcurve):
            self.use_lc = True
            self.lcs = data
        else:
            self.use_lc = False

        # if band_interest is None, extract the energy bins and make an array
        # with the lower and upper bounds of the energy bins
        if band_interest is None:
            if not self.use_lc:
                self._create_band_interest(data)
            else:
                self.band_interest = np.vstack(
                    [np.arange(len(data)), np.arange(1, len(data) + 1, 1)]
                ).T
        else:
            if np.size(band_interest) < 2:
                raise ValueError(
                    "band_interest must contain at least 2 values "
                    "(minimum and maximum values for each band) "
                    "and be a 2D array!"
                )

            self.band_interest = np.atleast_2d(band_interest)

        if self.use_lc is False and not dt:
            raise ValueError(
                "If the input data is event data, the dt keyword "
                "must be set and supply a time resolution for "
                "creating light curves!"
            )

        # if we don't have light curves already, make them:
        if not self.use_lc:
            if not np.all(np.diff(data, axis=0).T[0] >= 0):
                utils.simon("The event list must be sorted with respect to " "times of arrivals.")
                data = data[data[:, 0].argsort()]

            self.lcs = self._make_lightcurves(data)

        # check whether band of interest contains a Lightcurve object:
        if np.size(ref_band_interest) == 1 or isinstance(ref_band_interest, Lightcurve):
            if isinstance(ref_band_interest, Lightcurve):
                self.ref_band_lcs = ref_band_interest
            # ref_band_interest must either be a Lightcurve, or must have
            # multiple entries

            elif ref_band_interest is None:
                if self.use_lc:
                    self.ref_band_lcs = self._make_reference_bands_from_lightcurves(
                        ref_band_interest
                    )
                else:
                    self.ref_band_lcs = self._make_reference_bands_from_event_data(data)
            else:
                raise ValueError(
                    "ref_band_interest must contain either "
                    "a Lightcurve object, a list of Lightcurve "
                    "objects or a tuple of length 2."
                )
        else:
            # check whether ref_band_interest is a list of light curves
            if isinstance(ref_band_interest[0], Lightcurve):
                self.ref_band_lcs = ref_band_interest
                assert len(ref_band_interest) == len(self.lcs), (
                    "The list of "
                    "reference light "
                    "curves must have "
                    "the same length as "
                    "the list of light curves"
                    "of interest."
                )
            # if not, it must be a tuple, so we're going to make a list of light
            # curves
            else:
                if self.use_lc:
                    self.ref_band_lcs = self._make_reference_bands_from_lightcurves(
                        bounds=ref_band_interest
                    )
                else:
                    self.ref_band_lcs = self._make_reference_bands_from_event_data(data)

        self._construct_covar()

    def _make_reference_bands_from_event_data(self, data, bounds=None):
        """
        Helper method constructing reference bands for each band of interest, and constructing
        light curves from these reference bands. This operates only if the data given to
        :class:`Covariancespectrum` is event list data (i.e. photon arrival times and energies).

        Parameters
        ----------
        data : numpy.ndarray
            Array of shape ``(N, 2)``, where N is the number of photons. First column contains the
            times of arrivals, second column the corresponding photon energies.

        bounds : iterable
            The energy bounds to use for the reference band. Must be of type ``(elow, ehigh)``.

        Returns
        -------

        lc_all: list of :class:`stingray.Lightcurve` objects.
            The list of `:class:`stingray.Lightcurve` objects containing all reference
            bands, between the values given in ``bounds``.

        """

        if not bounds:
            bounds = [np.min(data[:, 1]), np.max(data[:, 1])]

        if bounds[1] <= np.min(self.band_interest[:, 0]) or bounds[0] >= np.max(
            self.band_interest[:, 1]
        ):
            elow = bounds[0]
            ehigh = bounds[1]

            toa = data[np.logical_and(data[:, 1] >= elow, data[:, 1] <= ehigh)]

            lc_all = Lightcurve.make_lightcurve(toa, self.dt, tstart=self.tstart, tseg=self.tseg)

        else:
            lc_all = []
            for i, b in enumerate(self.band_interest):
                elow = b[0]
                ehigh = b[1]

                emask1 = data[np.logical_and(data[:, 1] <= elow, data[:, 1] >= bounds[0])]

                emask2 = data[np.logical_and(data[:, 1] <= bounds[1], data[:, 1] >= ehigh)]

                toa = np.vstack([emask1, emask2])
                lc = Lightcurve.make_lightcurve(toa, self.dt, tstart=self.tstart, tseg=self.tseg)
                lc_all.append(lc)

        return lc_all

    def _make_reference_bands_from_lightcurves(self, bounds=None):
        """
        Helper class to construct reference bands for all light curves in ``band_interest``, assuming the
        data is given to the class :class:`Covariancespectrum` as a (set of) lightcurve(s). Generally
        sums up all other light curves within ``bounds`` that are *not* the band of interest.

        Parameters
        ----------
        bounds : iterable
            The energy bounds to use for the reference band. Must be of type ``(elow, ehigh)``.

        Returns
        -------
        lc_all: list of :class:`stingray.Lightcurve` objects.
            The list of :class:`stingray.Lightcurve` objects containing all reference bands,
            between the values given in ``bounds``.

        """

        if not bounds:
            bounds_idx = [0, len(self.band_interest)]

        else:
            low_bound = self.band_interest.searchsorted(bounds[0])
            high_bound = self.band_interest.searchsorted(bounds[1])

            bounds_idx = [low_bound, high_bound]

        lc_all = []
        for i, b in enumerate(self.band_interest):
            # initialize empty counts array
            counts = np.zeros_like(self.lcs[0].counts)
            for j in range(bounds_idx[0], bounds_idx[1], 1):
                if i == j:
                    continue
                else:
                    counts += self.lcs[j].counts

            # make a combined light curve
            lc = Lightcurve(self.lcs[0].time, counts, skip_checks=True)

            # add to list of reference light curves
            lc_all.append(lc)

        return lc_all

    def _construct_covar(self):
        """
        Helper method to construct the covariance attribute and fill it with values.
        """

        self.avg_covar = False
        covar = np.zeros(len(self.lcs))
        covar_err = np.zeros(len(self.lcs))
        xs_var = np.zeros(len(self.lcs))

        for i in range(len(self.lcs)):
            lc = self.lcs[i]

            if np.size(self.ref_band_lcs) == 1 or isinstance(self.ref_band_lcs, Lightcurve):
                lc_ref = self.ref_band_lcs
            else:
                lc_ref = self.ref_band_lcs[i]

            cv = self._compute_covariance(lc, lc_ref)
            cv_err = self._calculate_covariance_error(lc, lc_ref)

            covar[i] = cv
            covar_err[i] = cv_err

            xs = self._calculate_excess_variance(lc_ref)
            if not xs > 0:
                utils.simon(
                    "The excess variance in the reference band is "
                    "negative. This implies that the reference "
                    "band was badly chosen. Beware that the "
                    "covariance spectra will have NaNs!"
                )

            xs_var[i] = xs

        self.unnorm_covar = covar
        energy_covar = covar / xs_var**0.5

        self.covar = energy_covar

        self.covar_error = covar_err

        return

    def _make_lightcurves(self, data):
        """
        Create light curves for all bands of interest from ``data``. Takes the information the
        ``band_interest`` attribute and event data in ``data``, and produces a list of
        :class:`stingray.Lightcurve` objects.

        Parameters
        ----------
        data : numpy.ndarray
            Array of shape ``(N, 2)``, where ``N`` is the number of photons. First column contains the
            times of arrivals, second column the corresponding photon energies.

        Returns
        -------
        lc_all : iterable of :class:`stingray.Lightcurve` objects
            A list of :class:`stingray.Lightcurve` objects of all bands of interest.
        """

        self.tstart = np.min(data[:, 0])
        self.tend = np.max(data[:, 0])

        self.tseg = self.tend - self.tstart

        lc_all = []

        for i, b in enumerate(self.band_interest):
            elow = b[0]
            ehigh = b[1]

            toa = data[np.logical_and(data[:, 1] >= elow, data[:, 1] <= ehigh)]

            lc = Lightcurve.make_lightcurve(toa, self.dt, tstart=self.tstart, tseg=self.tseg)
            lc_all.append(lc)

        return lc_all

    def _create_band_interest(self, data):
        """
        If no bands of interest are given, but event data is, create bands of interest for each
        discrete energy value in the second column of ``data``.

        Parameters
        ----------
        data : numpy.ndarray
            Array of shape (N, 2), where N is the number of photons. First column contains the
            times of arrivals, second column the corresponding photon energies.

        """

        unique_energy = np.unique(data[:, 1])
        energ_diff = np.diff(unique_energy)

        energy_low = np.zeros_like(unique_energy)
        energy_high = np.zeros_like(unique_energy)

        energy_low[:-1] = unique_energy[:-1] - 0.5 * energ_diff
        energy_high[:-1] = unique_energy[:-1] + 0.5 * energ_diff

        energy_low[-1] = unique_energy[-1] - 0.5 * energ_diff[-1]
        energy_high[-1] = unique_energy[-1] + 0.5 * energ_diff[-1]

        energy_list = np.vstack([energy_low, energy_high]).T

        self.band_interest = energy_list

    def _calculate_excess_variance(self, lc):
        """Calculate excess variance in a band with the standard deviation."""
        std = self._calculate_std(lc)
        return np.var(lc) - std**2

    def _calculate_std(self, lc):
        """Return std calculated for the possible types of `std`"""
        if self.std is None:
            std = np.mean(lc) ** 0.5
        elif isinstance(self.std, Iterable):
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
        nn = lc_x.n
        # Number of segments averaged
        if not self.avg_covar:
            mm = 1
        else:
            mm = self.nbins

        num = xs_x * err_y + xs_y * err_x + err_x * err_y
        denom = nn * mm * xs_y

        return (num / denom) ** 0.5


class AveragedCovariancespectrum(Covariancespectrum):
    """
    Compute a covariance spectrum for the data, defined in [covar spectrum]_ Equation 15.

    Parameters
    ----------
    data : {numpy.ndarray | list of :class:`stingray.Lightcurve` objects}
        ``data`` contains the time series data, either in the form of a
        2-D array of ``(time stamp, energy)`` pairs for event data, or as a
        list of :class:`stingray.Lightcurve` objects.
        Note : The event list must be in sorted order with respect to the
        times of arrivals.

    segment_size : float
        The length of each segment in the averaged covariance spectrum.
        The number of segments will be calculated automatically using the
        total length of the data set and the segment_size defined here.

    dt : float
        The time resolution of the :class:`stingray.Lightcurve` formed
        from the energy bin. Only used if `data` is an event list.

    band_interest : {``None``, iterable of tuples}
        If ``None``, all possible energy values will be assumed to be of
        interest, and a covariance spectrum in the highest resolution
        will be produced.
        Note: if the input is a list of :class:`stingray.Lightcurve` objects,
        then the user may supply their energy values here, for construction of a
        reference band.

    ref_band_interest : {None, tuple, :class:`stingray.Lightcurve`, list of :class:`stingray.Lightcurve` objects}
        Defines the reference band to be used for comparison with the
        bands of interest. If None, all bands *except* the band of
        interest will be used for each band of interest, respectively.
        Alternatively, a tuple can be given for event list data, which will
        extract the reference band (always excluding the band of interest),
        or one may put in a single :class:`stingray.Lightcurve` object to be used (the same
        for each band of interest) or a list of :class:`stingray.Lightcurve` objects, one for
        each band of interest.

    std : float or np.array or list of numbers
        The term ``std`` is used to calculate the excess variance of a band.
        If ``std`` is set to ``None``, default Poisson case is taken and the
        ``std`` is calculated as ``mean(lc)**0.5``. In the case of a single
        float as input, the same is used as the standard deviation which
        is also used as the std. And if the std is an iterable of
        numbers, their mean is used for the same purpose.

    Attributes
    ----------
    unnorm_covar : np.ndarray
        An array of arrays with mid point band_interest and their
        covariance. It is the array-form of the dictionary ``energy_covar``.
        The covariance values are unnormalized.

    covar : np.ndarray
        Normalized covariance spectrum.

    covar_error : np.ndarray
        Errors of the normalized covariance spectrum.

    References
    ----------
    .. [covar spectrum] http://arxiv.org/pdf/1405.6575v2.pdf
    """

    def __init__(
        self, data, segment_size, dt=None, band_interest=None, ref_band_interest=None, std=None
    ):
        self.segment_size = segment_size

        Covariancespectrum.__init__(
            self,
            data,
            dt=dt,
            band_interest=band_interest,
            ref_band_interest=ref_band_interest,
            std=std,
        )

    def _construct_covar(self):
        """
        Helper method to construct the covariance attribute and fill it with values.
        """
        self.avg_covar = True

        start_time = self.lcs[0].time[0]

        covar = np.zeros(len(self.lcs))
        covar_err = np.zeros(len(self.lcs))
        xs_var = np.zeros(len(self.lcs))

        for i in range(len(self.lcs)):
            lc = self.lcs[i]

            if np.size(self.ref_band_lcs) == 1:
                lc_ref = self.ref_band_lcs
            else:
                lc_ref = self.ref_band_lcs[i]

            tstart = start_time
            tend = start_time + self.segment_size
            cv = 0.0
            cv_err = 0.0
            xs = 0.0

            self.nbins = int((tend - tstart) / self.segment_size)
            for k in range(self.nbins):
                start_ind = lc.time.searchsorted(tstart)
                end_ind = lc.time.searchsorted(tend)

                lc_seg = lc.truncate(start=start_ind, stop=end_ind)
                lc_ref_seg = lc_ref.truncate(start=start_ind, stop=end_ind)

                cv += self._compute_covariance(lc_seg, lc_ref_seg)
                cv_err += self._calculate_covariance_error(lc_seg, lc_ref_seg)
                xs += self._calculate_excess_variance(lc_ref_seg)
                if not xs > 0:
                    utils.simon(
                        "The excess variance in the reference band is "
                        "negative. This implies that the reference "
                        "band was badly chosen. Beware that the "
                        "covariance spectra will have NaNs!"
                    )

                tstart += self.segment_size
                tend += self.segment_size

            covar[i] = cv / self.nbins
            covar_err[i] = cv_err / self.nbins
            xs_var[i] = xs / self.nbins

        self.unnorm_covar = covar
        energy_covar = covar / xs_var**0.5

        self.covar = energy_covar

        self.covar_error = covar_err

        return

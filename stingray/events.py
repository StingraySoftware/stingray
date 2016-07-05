"""
Definition of :class:`EventList`.

:class:`EventList` is used to handle photon arrival times.
"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from stingray.simulator.events import events_to_lc, lc_to_events, assign_energies

from .lightcurve import Lightcurve
import stingray.io as io
import stingray.utils as utils

import numpy as np


class EventList(object):
    def __init__(self, times, energies=None, mjdref=0, dt=0, notes="", gti=None, pi=None):
        """
        Make an event list object from an array of time stamps

        Parameters
        ----------
        times: iterable
            A list or array of time stamps

        Other Parameters
        ----------------
        dt: float
            The time resolution of the events. Only relevant when using events
            to produce light curves with similar bin time.

        energies: iterable
            A list of array of photon energy values

        mjdref : float
            The MJD used as a reference for the time array.

        gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

        pi : integer, numpy.ndarray
            PI channels

        Attributes
        ----------
        times: numpy.ndarray
            The array of event arrival times, in seconds from the reference
            MJD (self.mjdref)

        energies: numpy.ndarray
            The array of photon energy values

        ncounts: int
            The number of data points in the event list

        dt: float
            The time resolution of the events. Only relevant when using events
            to produce light curves with similar bin time.

        mjdref : float
            The MJD used as a reference for the time array.

        gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

        pi : integer, numpy.ndarray
            PI channels

        """
        self.times = np.array(times, dtype=np.longdouble)
        self.energies = np.array(energies)
        self.ncounts = len(times)
        self.notes = notes
        self.dt = dt
        self.mjdref = mjdref
        self.gti = gti
        self.pi = pi

    @staticmethod
    def from_fits(fname, **kwargs):
        from .utils import _assign_value_if_none
        from .io import load_events_and_gtis

        ret = load_events_and_gtis(fname, **kwargs)

        times = ret.ev_list
        gti = ret.gtis

        pi = _assign_value_if_none(ret.additional_data, "PI", None)
        pha = _assign_value_if_none(ret.additional_data, "PHA", None)
        dt = ret.dt
        mjdref = ret.mjdref

        return EventList(times, gti=gti, pi=pi, pha=pha, dt=dt, mjdref=mjdref)

    def to_lc(self, bin_time, start_time=None):
        """
        Convert event list to a light curve object.

        Parameters
        ----------
        bin_time : float
            Binning time of the light curve

        Other Parameters
        ----------------
        start_time : float
            Initial time of the light curve

        Returns
        -------
        lc: `Lightcurve` object
        """

        return Lightcurve.make_lightcurve(self.times, bin_time, tstart=start_time)

    def set_times(self, lc):
        """
        Assign photon arrival times to event list, using acception-rejection
        method.

        Parameters
        ----------
        lc: `Lightcurve` object
        """ 

        times = lc_to_events(lc.times, lc.counts)
        self.times = EventList(times)

    def set_energies(self, spectrum):
        """
        Assign energies to event list.

        Parameters
        ----------
        spectrum: 2-d array or list
            Energies versus corresponding fluxes. The 2-d array or list must
            have energies across the first dimension and fluxes across the
            second one.
        """

        self.energies = assign_energies(self.ncounts, spectrum)

    def read(self, filename, format_='pickle'):
        """
        Imports EventList object.

        Parameters
        ----------
        filename: str
            Name of the EventList object to be read.

        format_: str
            Available options are 'pickle', 'hdf5', 'ascii'

        Returns
        --------
        If format_ is 'ascii': astropy.table is returned.
        If format_ is 'hdf5': dictionary with key-value pairs is returned.
        If format_ is 'pickle': class object is set.
        """

        if format_ == 'ascii' or format_ == 'hdf5':
            return io.read(filename, format_)

        elif format_ == 'pickle':
            self = io.read(filename, format_)

        else:
            utils.simon("Format not understood.")

    def write(self, filename, format_='pickle', **kwargs):
        """
        Exports EventList object.

        Parameters
        ----------
        filename: str
            Name of the LightCurve object to be created.

        format_: str
            Available options are 'pickle', 'hdf5', 'ascii'
        """

        if format_ == 'ascii':
            io.write(np.array([self.times]).T,
              filename, format_, fmt=["%s"])

        elif format_ == 'pickle':
            io.write(self, filename, format_)

        elif format_ == 'hdf5':
            io.write(self, filename, format_)

        else:
            utils.simon("Format not understood.")


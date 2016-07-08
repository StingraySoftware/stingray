"""
Definition of :class:`EventList`.

:class:`EventList` is used to handle photon arrival times.
"""
from __future__ import absolute_import, division, print_function

from .lightcurve import Lightcurve
import stingray.io as io
import stingray.utils as utils

import numpy as np
import numpy.random as ra


class EventList(object):
    def __init__(self, time=None, energies=None, ncounts=None, mjdref=0, dt=0, notes="", 
            gti=None, pi=None):
        """
        Make an event list object from an array of time stamps

        Parameters
        ----------
        time: iterable
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

        ncounts: int
            Number of desired data points in event list.

        gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

        pi : integer, numpy.ndarray
            PI channels

        Attributes
        ----------
        time: numpy.ndarray
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
        self.time = np.array(time, dtype=np.longdouble)
        self.energies = np.array(energies)
        self.notes = notes
        self.dt = dt
        self.mjdref = mjdref
        self.gti = gti
        self.pi = pi
        self.ncounts = ncounts

        if self.ncounts is None:
            
            try:
                self.ncounts = len(time)
            except:
                # In case of a 0-d array, pass
                pass

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

    def to_lc(self, dt, tstart=None, tend=None):
        """
        Convert event list to a light curve object.

        Parameters
        ----------
        dt: float
            Binning time of the light curve

        Other Parameters
        ----------------
        tstart : float
            Initial time of the light curve

        tend: float
            End time of the light curve
        
        Returns
        -------
        lc: `Lightcurve` object
        """

        return Lightcurve.make_lightcurve(self.time, dt, tstart=tstart, 
            tend=tend)

    def set_times(self, lc, use_spline=False, bin_time=None):
        """
        Assign photon arrival times to event list, using acception-rejection
        method.

        Parameters
        ----------
        lc: `Lightcurve` object
        """ 
        
        try:
            import scipy.interpolate as sci
        
        except:
            if use_spline:
                utils.simon("Scipy not available. Cannot use spline.")
                use_spline = False

        times = lc.time
        counts = lc.counts

        bin_time = utils.assign_value_if_none(bin_time, times[1] - times[0])
        n_bin = len(counts)
        bin_start = 0
        maxlc = np.max(counts)
        intlc = maxlc * n_bin
        n_events_predict = int(intlc + 10 * np.sqrt(intlc))

        # Max number of events per chunk must be < 100000
        events_per_bin_predict = n_events_predict / n_bin
        
        if use_spline:
            max_bin = np.max([4, 1000000 / events_per_bin_predict])
            
        else:
            max_bin = np.max([4, 5000000 / events_per_bin_predict])

        ev_list = np.zeros(n_events_predict)
        nev = 0

        while bin_start < n_bin:
            
            t0 = times[bin_start]
            bin_stop = min([bin_start + max_bin, n_bin + 1])
            lc_filt = counts[bin_start:bin_stop]
            t_filt = times[bin_start:bin_stop]
            length = t_filt[-1] - t_filt[0]
            n_bin_filt = len(lc_filt)
            n_to_simulate = n_bin_filt * max(lc_filt)
            safety_factor = 10

            if n_to_simulate > 10000:
                safety_factor = 4.

            n_to_simulate += safety_factor * np.sqrt(n_to_simulate)
            n_to_simulate = int(np.ceil(n_to_simulate))
            n_predict = ra.poisson(np.sum(lc_filt))

            random_ts = ra.uniform(t_filt[0] - bin_time / 2,
                                   t_filt[-1] + bin_time / 2, n_to_simulate)

            random_amps = ra.uniform(0, max(lc_filt), n_to_simulate)
            
            if use_spline:
                lc_spl = sci.splrep(t_filt, lc_filt, s=np.longdouble(0), k=1)
                pts = sci.splev(random_ts, lc_spl)
            
            else:
                rough_bins = np.rint((random_ts - t0) / bin_time)
                rough_bins = rough_bins.astype(int)
                pts = lc_filt[rough_bins]
            
            good = random_amps < pts
            len1 = len(random_ts)
            random_ts = random_ts[good]
            len2 = len(random_ts)
            random_ts = random_ts[:n_predict]
            random_ts.sort()
            new_nev = len(random_ts)
            ev_list[nev:nev + new_nev] = random_ts[:]
            nev += new_nev
            bin_start += max_bin

        # Discard all zero entries at the end
        time = ev_list[:nev]
        time.sort()
        
        self.time = EventList(time).time

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

        if self.ncounts is None:
            utils.simon("Either set time values or explicity provide counts.")
            return

        if isinstance(spectrum, list):
            try:
                energies = np.array(spectrum[0])
                fluxes = np.array(spectrum[1])
            
            except TypeError:
                utils.simon("Spectrum must be a 2-d array or list")

        else:
            try:
                energies = spectrum[0]
                fluxes = spectrum[1]

            except IndexError:
                utils.simon("Spectrum must be a 2-d array or list")
        
        # Create a set of probability values
        prob = fluxes / float(sum(fluxes))

        # Calculate cumulative probability
        cum_prob = np.cumsum(prob)

        # Draw N random numbers between 0 and 1, where N is the size of event list
        R = ra.uniform(0, 1, self.ncounts)

        # Assign energies to events corresponding to the random numbers drawn
        self.energies = np.array([energies[np.argwhere(cum_prob == 
            min(cum_prob[(cum_prob - r) > 0]))] for r in R])

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
            io.write(np.array([self.time]).T,
              filename, format_, fmt=["%s"])

        elif format_ == 'pickle':
            io.write(self, filename, format_)

        elif format_ == 'hdf5':
            io.write(self, filename, format_)

        else:
            utils.simon("Format not understood.")


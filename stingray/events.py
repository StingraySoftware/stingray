"""
Definition of :class:`EventList`.

:class:`EventList` is used to handle photon arrival times.
"""

import warnings

import numpy as np

from stingray.utils import _int_sum_non_zero
from stingray.loggingconfig import setup_logger

from .base import StingrayTimeseries
from .filters import get_deadtime_mask
from .gti import generate_indices_of_boundaries
from .io import pi_to_energy, get_file_extension
from .io import FITSTimeseriesReader
from .lightcurve import Lightcurve
from .utils import simon, njit
from .utils import histogram

__all__ = ["EventList"]

logger = setup_logger()


@njit
def _from_lc_numba(times, counts, empty_times):
    """Create a rough event list from a light curve.

    This function creates as many events as the counts in each time bin of the light curve,
    with event times equal to the light curve time stamps.

    Parameters
    ----------
    times : array-like
        Array of time stamps
    counts : array-like
        Array of counts
    empty_times : array-like
        Empty array to be filled with time stamps
    """
    last = 0
    for t, c in zip(times, counts):
        if c <= 0:
            continue
        val = c + last
        empty_times[last:val] = t
        last = val
    # If c < 0 in some cases, some times will be empty
    return empty_times[:val]


def simple_events_from_lc(lc):
    """
    Create an :class:`EventList` from a :class:`stingray.Lightcurve` object. Note that all
    events in a given time bin will have the same time stamp.

    Bins with negative counts will be ignored.

    Parameters
    ----------
    lc: :class:`stingray.Lightcurve` object
        Light curve to use for creation of the event list.

    Returns
    -------
    ev: :class:`EventList` object
        The resulting list of photon arrival times generated from the light curve.

    Examples
    --------
    >>> from stingray import Lightcurve
    >>> lc = Lightcurve([0, 1, 2], [2, 3, -1], dt=1)
    >>> ev = simple_events_from_lc(lc)
    >>> assert np.allclose(ev.time, [0, 0, 1, 1, 1])
    """
    counts = lc.counts.astype(int)
    allcounts = _int_sum_non_zero(counts)
    times = _from_lc_numba(lc.time, counts, np.zeros(allcounts, dtype=float))
    return EventList(time=times, gti=lc.gti)


class EventList(StingrayTimeseries):
    """
    Basic class for event list data. Event lists generally correspond to individual events (e.g. photons)
    recorded by the detector, and their associated properties. For X-ray data where this type commonly occurs,
    events are time stamps of when a photon arrived in the detector, and (optionally) the photon energy associated
    with the event.

    Parameters
    ----------
    time: iterable
        A list or array of time stamps

    Other Parameters
    ----------------
    dt: float
        The time resolution of the events. Only relevant when using events
        to produce light curves with similar bin time.

    energy: iterable
        A list of array of photon energy values in keV

    mjdref : float
        The MJD used as a reference for the time array.

    ncounts: int
        Number of desired data points in event list. Deprecated

    gtis: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Good Time Intervals

    pi : integer, numpy.ndarray
        PI channels

    notes : str
        Any useful annotations

    high_precision : bool
        Change the precision of self.time to float128. Useful while dealing with fast pulsars.

    mission : str
        Mission that recorded the data (e.g. NICER)

    instr : str
        Instrument onboard the mission

    header : str
        The full header of the original FITS file, if relevant

    detector_id : iterable
        The detector that recorded each photon (if the instrument has more than
        one, e.g. XMM/EPIC-pn)

    timeref : str
        The time reference, as recorded in the FITS file (e.g. SOLARSYSTEM)

    timesys : str
        The time system, as recorded in the FITS file (e.g. TDB)

    ephem : str
        The JPL ephemeris used to barycenter the data, if any (e.g. DE430)

    rmf_file : str, default None
        The file name of the RMF file to use for calibration.

    skip_checks : bool, default False
        Skip checks for the validity of the event list. Use with caution.

    **other_kw :
        Used internally. Any other keyword arguments will be ignored

    Attributes
    ----------
    time: numpy.ndarray
        The array of event arrival times, in seconds from the reference
        MJD defined in ``mjdref``

    energy: numpy.ndarray
        The array of photon energy values

    ncounts: int
        The number of data points in the event list

    dt: float
        The time resolution of the events. Only relevant when using events
        to produce light curves with similar bin time.

    mjdref : float
        The MJD used as a reference for the time array.

    gtis: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Good Time Intervals

    pi : integer, numpy.ndarray
        PI channels

    high_precision : bool
        Change the precision of self.time to float128. Useful while dealing with fast pulsars.

    mission : str
        Mission that recorded the data (e.g. NICER)

    instr : str
        Instrument onboard the mission

    detector_id : iterable
        The detector that recoded each photon, if relevant (e.g. XMM, Chandra)

    header : str
        The full header of the original FITS file, if relevant

    """

    main_array_attr = "time"

    def __init__(
        self,
        time=None,
        energy=None,
        ncounts=None,
        mjdref=0,
        dt=0,
        notes="",
        gti=None,
        pi=None,
        high_precision=False,
        mission=None,
        instr=None,
        header=None,
        detector_id=None,
        ephem=None,
        timeref=None,
        timesys=None,
        rmf_file=None,
        skip_checks=False,
        **other_kw,
    ):
        if ncounts is not None:
            warnings.warn(
                "The ncounts keyword does nothing, and is maintained for backwards compatibility.",
                DeprecationWarning,
            )

        if rmf_file is not None:
            if pi is None:
                warnings.warn("PI channels must be provided to calibrate the energy")
            else:
                energy = pi_to_energy(pi, rmf_file)

        StingrayTimeseries.__init__(
            self,
            time=time,
            energy=None if energy is None else np.asanyarray(energy),
            mjdref=mjdref,
            dt=dt,
            notes=notes,
            gti=np.asanyarray(gti) if gti is not None else None,
            pi=None if pi is None else np.asanyarray(pi),
            high_precision=high_precision,
            mission=mission,
            instr=instr,
            header=header,
            detector_id=detector_id,
            ephem=ephem,
            timeref=timeref,
            timesys=timesys,
            rmf_file=rmf_file,
            skip_checks=skip_checks,
            **other_kw,
        )

        if other_kw != {}:
            warnings.warn(f"Unrecognized keywords: {list(other_kw.keys())}")

        if (self.time is not None) and (self.energy is not None):
            if np.size(self.time) != np.size(self.energy):
                raise ValueError("Lengths of time and energy must be equal.")

    @property
    def ncounts(self):
        """Number of events in the event list."""
        return self.n

    def to_lc(self, dt, tstart=None, tseg=None):
        """
        Convert event list to a :class:`stingray.Lightcurve` object.

        Parameters
        ----------
        dt: float
            Binning time of the light curve

        Other Parameters
        ----------------
        tstart : float
            Start time of the light curve

        tseg: float
            Total duration of light curve

        Returns
        -------
        lc: :class:`stingray.Lightcurve` object
        """
        dt = self.suggest_compatible_dt(dt)
        return Lightcurve.make_lightcurve(
            self.time, dt, tstart=tstart, gti=self._gti, tseg=tseg, mjdref=self.mjdref
        )

    def to_binned_timeseries(self, dt, array_attrs=None):
        """Convert the event list to a binned :class:`stingray.StingrayTimeseries` object.

        The result will be something similar to a light curve, but with arbitrary
        attributes corresponding to a weighted sum of each specified attribute of
        the event list.

        E.g. if the event list has a ``q`` attribute, the final time series will
        have a ``q`` attribute, which is the sum of all ``q`` values in each time bin.

        Parameters
        ----------
        dt: float
            Binning time of the light curve

        Other Parameters
        ----------------
        array_attrs: list of str
            List of attributes to be converted to light curve arrays. If None,
            all array attributes will be converted.

        Returns
        -------
        lc: :class:`stingray.Lightcurve` object
        """
        if array_attrs is None:
            array_attrs = self.array_attrs()

        ranges = [self.gti[0, 0], self.gti[-1, 1]]
        nbins = int((ranges[1] - ranges[0]) / dt)
        ranges = [ranges[0], ranges[0] + nbins * dt]
        times = np.arange(ranges[0] + dt * 0.5, ranges[1], dt)

        counts = histogram(self.time, range=ranges, bins=nbins)

        attr_dict = dict(counts=counts)

        for attr in array_attrs:
            if getattr(self, attr, None) is not None:
                logger.info(f"Creating the {attr} array")

                attr_dict[attr] = histogram(
                    self.time, bins=nbins, weights=getattr(self, attr), range=ranges
                )
        meta_attrs = dict((attr, getattr(self, attr)) for attr in self.meta_attrs())
        new_ts = StingrayTimeseries(times, array_attrs=attr_dict, **meta_attrs)
        new_ts.dt = dt
        return new_ts

    def to_lc_iter(self, dt, segment_size=None):
        """Convert event list to a generator of Lightcurves.

        Parameters
        ----------
        dt: float
            Binning time of the light curves

        Other parameters
        ----------------
        segment_size : float, default None
            Optional segment size. If None, use the GTI boundaries

        Returns
        -------
        lc_gen: `generator`
            Generates one :class:`stingray.Lightcurve` object for each GTI or segment
        """

        segment_iter = generate_indices_of_boundaries(
            self.time, self.gti, segment_size=segment_size, dt=0
        )

        for st, end, idx_st, idx_end in segment_iter:
            tseg = end - st

            lc = Lightcurve.make_lightcurve(
                self.time[idx_st : idx_end + 1],
                dt,
                tstart=st,
                gti=np.asanyarray([[st, end]]),
                tseg=tseg,
                mjdref=self.mjdref,
                use_hist=True,
            )
            yield lc

    def to_lc_list(self, dt, segment_size=None):
        """Convert event list to a list of Lightcurves.

        Parameters
        ----------
        dt: float
            Binning time of the light curves

        Other parameters
        ----------------
        segment_size : float, default None
            Optional segment size. If None, use the GTI boundaries

        Returns
        -------
        lc_list: `List`
            List containing one :class:`stingray.Lightcurve` object for each GTI or segment
        """
        return list(self.to_lc_iter(dt, segment_size))

    @staticmethod
    def from_lc(lc):
        """
        Create an :class:`EventList` from a :class:`stingray.Lightcurve` object. Note that all
        events in a given time bin will have the same time stamp.

        Bins with negative counts will be ignored.

        Parameters
        ----------
        lc: :class:`stingray.Lightcurve` object
            Light curve to use for creation of the event list.

        Returns
        -------
        ev: :class:`EventList` object
            The resulting list of photon arrival times generated from the light curve.
        """
        return simple_events_from_lc(lc)

    def simulate_times(self, lc, use_spline=False, bin_time=None):
        """Simulate times from an input light curve.

        Randomly simulate photon arrival times to an :class:`EventList` from a
        :class:`stingray.Lightcurve` object, using the inverse CDF method.

        ..note::
            Preferably use model light curves containing **no Poisson noise**,
            as this method will intrinsically add Poisson noise to them.

        Parameters
        ----------
        lc: :class:`stingray.Lightcurve` object

        Other Parameters
        ----------------
        use_spline : bool
            Approximate the light curve with a spline to avoid binning effects

        bin_time : float default None
            Ignored and deprecated, maintained for backwards compatibility.

        Returns
        -------
        times : array-like
            Simulated photon arrival times
        """
        # Need import here, or there will be a circular import
        from .simulator.base import simulate_times

        if bin_time is not None:
            warnings.warn("Bin time will be ignored in simulate_times", DeprecationWarning)

        vals = simulate_times(lc, use_spline=use_spline)
        self.time = vals
        self._gti = lc.gti

    def simulate_energies(self, spectrum, use_spline=False):
        """
        Assign (simulate) energies to event list from a spectrum.

        Parameters
        ----------
        spectrum: 2-d array or list [energies, spectrum]
            Energies versus corresponding fluxes. The 2-d array or list must
            have energies across the first dimension and fluxes across the
            second one. If the dimension of the energies is the same as
            spectrum, they are interpreted as bin centers.
            If it is longer by one, they are interpreted as proper bin edges
            (similarly to the bins of `np.histogram`).
            Note that for non-uniformly binned spectra, it is advisable to pass
            the exact edges.
        """
        from .simulator.base import simulate_with_inverse_cdf

        if self.ncounts is None or self.ncounts == 0:
            simon("Simulating on an empty event list")
            return

        if isinstance(spectrum, list) or isinstance(spectrum, np.ndarray):
            energy = np.asanyarray(spectrum)[0]
            fluxes = np.asanyarray(spectrum)[1]

            if not isinstance(energy, np.ndarray):
                raise IndexError("Spectrum must be a 2-d array or list")

        else:
            raise TypeError("Spectrum must be a 2-d array or list")

        if energy.size == fluxes.size:
            de = energy[1] - energy[0]
            energy = np.concatenate([energy - de / 2, [energy[-1] + de / 2]])

        self.energy = simulate_with_inverse_cdf(
            fluxes, self.ncounts, edges=energy, sorted=False, interp_kind="linear"
        )

    def sort(self, inplace=False):
        """Sort the event list in time.

        Other parameters
        ----------------
        inplace : bool, default False
            Sort in place. If False, return a new event list.

        Returns
        -------
        eventlist : `EventList`
            The sorted event list. If ``inplace=True``, it will be a shallow copy
            of ``self``.

        Examples
        --------
        >>> events = EventList(time=[0, 2, 1], energy=[0.3, 2, 0.5], pi=[3, 20, 5],
        ...                    skip_checks=True)
        >>> e1 = events.sort()
        >>> assert np.allclose(e1.time, [0, 1, 2])
        >>> assert np.allclose(e1.energy, [0.3, 0.5, 2])
        >>> assert np.allclose(e1.pi, [3, 5, 20])

        But the original event list has not been altered (``inplace=False`` by
        default):
        >>> assert np.allclose(events.time, [0, 2, 1])

        Let's do it in place instead
        >>> e2 = events.sort(inplace=True)
        >>> assert np.allclose(e2.time, [0, 1, 2])

        In this case, the original event list has been altered.
        >>> assert np.allclose(events.time, [0, 1, 2])

        """
        order = np.argsort(self.time)
        return self.apply_mask(order, inplace=inplace)

    def join(self, other, strategy="infer"):
        """
        Join two :class:`EventList` objects into one.

        If both are empty, an empty :class:`EventList` is returned.

        GTIs are crossed if the event lists are over a common time interval,
        and appended otherwise.

        Standard attributes such as ``pi`` and ``energy`` remain ``None`` if they are ``None``
        in both. Otherwise, ``np.nan`` is used as a default value for the :class:`EventList` where
        they were None. Arbitrary attributes (e.g., Stokes parameters in polarimetric data) are
        created and joined using the same convention.

        Multiple checks are done on the joined event lists. If the time array of the event list
        being joined is empty, it is ignored. If the time resolution is different, the final
        event list will have the rougher time resolution. If the MJDREF is different, the time
        reference will be changed to the one of the first event list. An empty event list will
        be ignored.

        Parameters
        ----------
        other : :class:`EventList` object or class:`list` of :class:`EventList` objects
            The other :class:`EventList` object which is supposed to be joined with.
            If ``other`` is a list, it is assumed to be a list of :class:`EventList` objects
            and they are all joined, one by one.

        Other parameters
        ----------------
        strategy : {"intersection", "union", "append", "infer", "none"}
            Method to use to merge the GTIs. If "intersection", the GTIs are merged
            using the intersection of the GTIs. If "union", the GTIs are merged
            using the union of the GTIs. If "none", a single GTI with the minimum and
            the maximum time stamps of all GTIs is returned. If "infer", the strategy
            is decided based on the GTIs. If there are no overlaps, "union" is used,
            otherwise "intersection" is used. If "append", the GTIs are simply appended
            but they must be mutually exclusive.

        Returns
        -------
        `ev_new` : :class:`EventList` object
            The resulting :class:`EventList` object.
        """

        return self._join_timeseries(other, strategy=strategy, ignore_meta=["header", "ncounts"])

    @classmethod
    def read(cls, filename, fmt=None, rmf_file=None, **kwargs):
        r"""Read a :class:`EventList` object from file.

        Currently supported formats are

        * pickle (not recommended for long-term storage)
        * hea or ogip : FITS Event files from (well, some) HEASARC-supported missions.
        * any other formats compatible with the writers in
          :class:`astropy.table.Table` (ascii.ecsv, hdf5, etc.)

        Files that need the :class:`astropy.table.Table` interface MUST contain
        at least a ``time`` column. Other recognized columns are ``energy`` and
        ``pi``.
        The default ascii format is enhanced CSV (ECSV). Data formats
        supporting the serialization of metadata (such as ECSV and HDF5) can
        contain all eventlist attributes such as ``mission``, ``gti``, etc with
        no significant loss of information. Other file formats might lose part
        of the metadata, so must be used with care.

        Parameters
        ----------
        filename: str
            Path and file name for the file to be read.

        fmt: str
            Available options are 'pickle', 'hea', and any `Table`-supported
            format such as 'hdf5', 'ascii.ecsv', etc.

        Other parameters
        ----------------
        rmf_file : str, default None
            The file name of the RMF file to use for energy calibration. Defaults to
            None, which implies no channel->energy conversion at this stage (or a default
            calibration applied to selected missions).

        kwargs : dict
            Any further keyword arguments to be passed to `load_events_and_gtis`
            for reading in event lists in OGIP/HEASOFT format

        Returns
        -------
        ev: :class:`EventList` object
            The :class:`EventList` object reconstructed from file
        """
        if fmt is None:
            for fits_ext in ["fits", "evt"]:
                if fits_ext in get_file_extension(filename).lower():
                    fmt = "hea"
                    break
        if fmt is not None and fmt.lower() in ("hea", "ogip"):
            additional_columns = kwargs.pop("additional_columns", None)

            evt = FITSTimeseriesReader(
                filename, output_class=EventList, additional_columns=additional_columns, **kwargs
            )[:]

            if rmf_file is not None:
                evt.convert_pi_to_energy(rmf_file)
            return evt

        return super().read(filename=filename, fmt=fmt, **kwargs)

    def convert_pi_to_energy(self, rmf_file):
        """Calibrate the energy column of the event list.

        Defines the ``energy`` attribute of the event list by converting the
        PI channels to energy using the provided RMF file.

        Parameters
        ----------
        rmf_file : str
            The file name of the RMF file to use for calibration.
        """

        self.energy = pi_to_energy(self.pi, rmf_file)

    def get_energy_mask(self, energy_range, use_pi=False):
        """Get a mask corresponding to events with a given energy range.

        Parameters
        ----------
        energy_range: [float, float]
            Energy range in keV, or in PI channel (if ``use_pi`` is True)

        Other Parameters
        ----------------
        use_pi : bool, default False
            Use PI channel instead of energy in keV
        """
        if use_pi:
            energies = self.pi
        else:
            energies = self.energy
        return (energies >= energy_range[0]) & (energies < energy_range[1])

    def suggest_compatible_dt(self, dt, warn=True):
        """Suggest a compatible time resolution for the event list.

        If the event list has a time resolution, it is better to avoid
        using a time bin smaller than that for sure, but also a time bin which is
        not a multiple of it, as it will create beats that appear in the power spectrum
        as a comb of peaks. This method suggests a compatible time resolution.

        Parameters
        ----------
        dt : float
            Desired time resolution

        Other Parameters
        ----------------
        warn : bool, default True
            Issue a warning if the time resolution is changed
        """
        if hasattr(self, "dt") and self.dt > 0 and not np.isclose(self.dt, dt, rtol=1e-4):
            dt = self.dt * max(np.rint(dt / self.dt), 1)
            warnings.warn(
                f"The input event list has a time resolution of {self.dt}. "
                f"Using a multiple of that as dt ({dt})."
            )
            return dt
        return dt

    def filter_energy_range(self, energy_range, inplace=False, use_pi=False):
        """Filter the event list from a given energy range.

        Parameters
        ----------
        energy_range: [float, float]
            Energy range in keV, or in PI channel (if ``use_pi`` is True)

        Other Parameters
        ----------------
        inplace : bool, default False
            Do the change in place (modify current event list). Otherwise, copy
            to a new event list.
        use_pi : bool, default False
            Use PI channel instead of energy in keV

        Examples
        --------
        >>> events = EventList(time=[0, 1, 2], energy=[0.3, 0.5, 2], pi=[3, 5, 20])
        >>> e1 = events.filter_energy_range([0, 1])
        >>> assert np.allclose(e1.time, [0, 1])
        >>> assert np.allclose(events.time, [0, 1, 2])
        >>> e2 = events.filter_energy_range([0, 10], use_pi=True, inplace=True)
        >>> assert np.allclose(e2.time, [0, 1])
        >>> assert np.allclose(events.time, [0, 1])

        """
        mask = self.get_energy_mask(energy_range, use_pi=use_pi)
        return self.apply_mask(mask, inplace=inplace)

    def apply_deadtime(self, deadtime, inplace=False, **kwargs):
        """Apply deadtime filter to this event list.

        Additional arguments in ``kwargs`` are passed to `get_deadtime_mask`

        Parameters
        ----------
        deadtime : float
            Value of dead time to apply to data
        inplace : bool, default False
            If True, apply the deadtime to the current event list. Otherwise,
            return a new event list.

        Returns
        -------
        new_event_list : `EventList` object
            Filtered event list. if `inplace` is True, this is the input object
            filtered for deadtime, otherwise this is a new object.
        additional_output : object
            Only returned if `return_all` is True. See `get_deadtime_mask` for
            more details.

        Examples
        --------
        >>> events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
        >>> events = EventList(events, gti=[[0, 3.3]])
        >>> events.pi=np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
        >>> events.energy=np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
        >>> events.mjdref = 10
        >>> filt_events, retval = events.apply_deadtime(0.11, inplace=False,
        ...                                             verbose=False,
        ...                                             return_all=True)
        >>> assert filt_events is not events
        >>> expected = np.array([1, 2, 2.2, 3, 3.2])
        >>> assert np.allclose(filt_events.time, expected)
        >>> assert np.allclose(filt_events.pi, 1)
        >>> assert np.allclose(filt_events.energy, 1)
        >>> assert not np.allclose(events.pi, 1)
        >>> filt_events = events.apply_deadtime(0.11, inplace=True,
        ...                                     verbose=False)
        >>> assert filt_events is events
        """
        local_retall = kwargs.pop("return_all", False)

        mask, retall = get_deadtime_mask(self.time, deadtime, return_all=True, **kwargs)

        new_ev = self.apply_mask(mask, inplace=inplace)

        if local_retall:
            new_ev = [new_ev, retall]

        return new_ev

    def get_color_evolution(self, energy_ranges, segment_size=None, use_pi=False):
        """Compute the color in equal-length segments of the event list.

        Parameters
        ----------
        energy_ranges : 2x2 list
            List of energy ranges to compute the color:
            ``[[en1_min, en1_max], [en2_min, en2_max]]``
        segment_size : float
            Segment size in seconds. If None, the full GTIs are considered
            instead as segments.

        Other Parameters
        ----------------
        use_pi : bool, default False
            Use PI channel instead of energy in keV

        Returns
        -------
        color : array-like
            Array of colors, computed in each segment as the ratio of the
            counts in the second energy range to the counts in the first energy
            range.
        """
        if energy_ranges is None or np.shape(energy_ranges) != (2, 2):
            raise ValueError("Energy ranges must be specified as a 2x2 array")

        def color(ev):
            mask1 = ev.get_energy_mask(energy_ranges[0], use_pi=use_pi)
            mask2 = ev.get_energy_mask(energy_ranges[1], use_pi=use_pi)
            en1_ct = np.count_nonzero(mask1)
            en2_ct = np.count_nonzero(mask2)

            if en1_ct == 0 or en2_ct == 0:
                warnings.warn("No counts in one of the energy ranges. Returning NaN")
                return np.nan, np.nan
            color = en2_ct / en1_ct
            color_err = color * (np.sqrt(en1_ct) / en1_ct + np.sqrt(en2_ct) / en2_ct)
            return color, color_err

        starts, stops, (colors, color_errs) = self.analyze_segments(color, segment_size)

        return starts, stops, colors, color_errs

    def get_intensity_evolution(self, energy_range, segment_size=None, use_pi=False):
        """Compute the intensity in equal-length segments (or full GTIs) of the event list.

        Parameters
        ----------
        energy_range : ``[en1_min, en1_max]``
            Energy range to compute the intensity
        segment_size : float
            Segment size in seconds. If None, the full GTIs are considered
            instead as segments.

        Other Parameters
        ----------------
        use_pi : bool, default False
            Use PI channel instead of energy in keV

        Returns
        -------
        intensity : array-like
            Array of intensities (in counts/s), computed in each segment.
        """
        if energy_range is None or np.shape(energy_range) != (2,):
            raise ValueError("Energy ranges must be specified as a 2-element list")

        def intensity(ev):
            mask1 = ev.get_energy_mask(energy_range, use_pi=use_pi)
            en1_ct = np.count_nonzero(mask1)
            segment_size = ev.gti[0, 1] - ev.gti[0, 0]
            rate = en1_ct / segment_size
            rate_err = np.sqrt(en1_ct) / segment_size
            return rate, rate_err

        starts, stops, (rate, rate_err) = self.analyze_segments(intensity, segment_size)

        return starts, stops, rate, rate_err

    def get_mask_for_filter_by_detector_id(self, value):
        """Supplies a mask to filter the data for a particular detector(s) for a event file.
        For example can be used to filter the data for a single unit of LAXPC or a selected NICER detector ID


        Args:
            value: The value for which the mask returns 1
        """
        if isinstance(value, int):
            col = self.detector_id
            return col == value
        elif isinstance(value, list) and all(isinstance(i, int) for i in value):
            col = self.detector_id
            full_mask = col == value[0]
            for i in range(1, len(value)):
                mask = col == value[i]
                full_mask = full_mask | mask
            return full_mask

    def filter_detector_id(self, detector_id, inplace=False):
        """Filters the data selecting only a selected detector or a list of detectors.

        Args:
            detector_id (int, optional): The detector id on which the filtering needs to be applied. It can be a list.
        """

        mask = self.get_mask_for_filter_by_detector_id(detector_id)
        new_ev = self.apply_mask(mask, inplace=inplace)
        return new_ev

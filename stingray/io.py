import math
import copy
import os
import sys
import traceback
import warnings
import subprocess as sp
from collections.abc import Iterable

import numpy as np
from astropy.table import Table
from astropy.logger import AstropyUserWarning
import matplotlib.pyplot as plt
from astropy.io import fits as pf
from astropy import units as u

import stingray.utils as utils
from stingray.loggingconfig import setup_logger
from stingray.utils import fits_open_including_remote


from .utils import (
    assign_value_if_none,
    is_string,
    order_list_of_arrays,
    is_sorted,
    make_dictionary_lowercase,
)
from .gti import (
    get_gti_from_all_extensions,
    load_gtis,
    get_total_gti_length,
    split_gtis_by_exposure,
    cross_two_gtis,
)

from .mission_support import (
    read_mission_info,
    rough_calibration,
    get_rough_conversion_function,
    mission_specific_event_interpretation,
)

# Python 3
import pickle

_H5PY_INSTALLED = True
DEFAULT_FORMAT = "hdf5"

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False
    DEFAULT_FORMAT = "pickle"

HAS_128 = True
try:
    np.float128
except AttributeError:  # pragma: no cover
    HAS_128 = False

logger = setup_logger()


def read_rmf(rmf_file):
    """Load RMF info.

    .. note:: Preliminary: only EBOUNDS are read.

    Parameters
    ----------
    rmf_file : str
        The rmf file used to read the calibration.

    Returns
    -------
    pis : array-like
        the PI channels
    e_mins : array-like
        the lower energy bound of each PI channel
    e_maxs : array-like
        the upper energy bound of each PI channel
    """

    with pf.open(rmf_file, checksum=True, memmap=False) as lchdulist:
        lchdulist.verify("warn")
        lctable = lchdulist["EBOUNDS"].data
        pis = np.array(lctable.field("CHANNEL"))
        e_mins = np.array(lctable.field("E_MIN"))
        e_maxs = np.array(lctable.field("E_MAX"))

    return pis, e_mins, e_maxs


def pi_to_energy(pis, rmf_file):
    """Read the energy channels corresponding to the given PI channels.

    Parameters
    ----------
    pis : array-like
        The channels to lookup in the rmf

    Other Parameters
    ----------------
    rmf_file : str
        The rmf file used to read the calibration.
    """
    calp, cal_emin, cal_emax = read_rmf(rmf_file)
    es = np.zeros(len(pis), dtype=float)
    for ic, c in enumerate(calp):
        good = pis == c
        if not np.any(good):
            continue
        es[good] = (cal_emin[ic] + cal_emax[ic]) / 2

    return es


def get_file_extension(fname):
    """Get the extension from the file name.

    If g-zipped, add '.gz' to extension.

    Examples
    --------
    >>> get_file_extension('ciao.tar')
    '.tar'
    >>> get_file_extension('ciao.tar.gz')
    '.tar.gz'
    >>> get_file_extension('ciao.evt.gz')
    '.evt.gz'
    >>> get_file_extension('ciao.a.tutti.evt.gz')
    '.evt.gz'
    """
    fname_root = fname.replace(".gz", "")
    fname_root = os.path.splitext(fname_root)[0]

    return fname.replace(fname_root, "")


def high_precision_keyword_read(hdr, keyword):
    """Read FITS header keywords, also if split in two.

    In the case where the keyword is split in two, like

        MJDREF = MJDREFI + MJDREFF

    in some missions, this function returns the summed value. Otherwise, the
    content of the single keyword

    Parameters
    ----------
    hdr : dict_like
        The FITS header structure, or a dictionary

    keyword : str
        The key to read in the header

    Returns
    -------
    value : long double
        The value of the key, or ``None`` if something went wrong

    """
    try:
        value = np.longdouble(hdr[keyword])
        return value
    except KeyError:
        pass
    try:
        if len(keyword) == 8:
            keyword = keyword[:7]
        value = np.longdouble(hdr[keyword + "I"])
        value += np.longdouble(hdr[keyword + "F"])
        return value
    except KeyError:
        return None


def _case_insensitive_search_in_list(string, list_of_strings):
    """Search for a string in a list of strings, in a case-insensitive way.

    Example
    -------
    >>> _case_insensitive_search_in_list("a", ["A", "b"])
    'A'
    >>> assert _case_insensitive_search_in_list("a", ["c", "b"]) is None
    """
    for s in list_of_strings:
        if string.lower() == s.lower():
            return s
    return None


def _get_additional_data(lctable, additional_columns, warn_if_missing=True):
    """Get additional data from a FITS data table.

    Parameters
    ----------
    lctable: `astropy.io.fits.fitsrec.FITS_rec`
        Data table
    additional_columns: list of str
        List of column names to retrieve from the table

    Other parameters
    ----------------
    warn_if_missing: bool, default True
        Warn if a column is not found

    Returns
    -------
    additional_data: dict
        Dictionary associating to each additional column the content of the
        table.
    """
    additional_data = {}
    if additional_columns is not None:
        for a in additional_columns:
            key = _case_insensitive_search_in_list(a, lctable._coldefs.names)
            if key is not None:
                conversion = 1
                if (
                    key.lower() == "energy"
                    and hasattr(lctable.columns[key], "unit")
                    and (unit := lctable.columns[key].unit) is not None
                ):
                    conversion = (1 * u.Unit(unit)).to(u.keV).value
                additional_data[a] = np.array(lctable.field(key)) * conversion
            else:
                if warn_if_missing:
                    warnings.warn("Column " + a + " not found")
                additional_data[a] = np.zeros(len(lctable))

    return additional_data


def get_key_from_mission_info(info, key, default, inst=None, mode=None):
    """Get the name of a header key or table column from the mission database.

    Many entries in the mission database have default values that can be
    altered for specific instruments or observing modes. Here, if there is a
    definition for a given instrument and mode, we take that, otherwise we use
    the default).

    Parameters
    ----------
    info : dict
        Nested dictionary containing all the information for a given mission.
        It can be nested, e.g. contain some info for a given instrument, and
        for each observing mode of that instrument.
    key : str
        The key to read from the info dictionary
    default : object
        The default value. It can be of any type, depending on the expected
        type for the entry.

    Other parameters
    ----------------
    inst : str
        Instrument
    mode : str
        Observing mode

    Returns
    -------
    retval : object
        The wanted entry from the info dictionary

    Examples
    --------
    >>> info = {'ecol': 'PI', "A": {"ecol": "BLA"}, "C": {"M1": {"ecol": "X"}}}
    >>> get_key_from_mission_info(info, "ecol", "BU", inst="A", mode=None)
    'BLA'
    >>> get_key_from_mission_info(info, "ecol", "BU", inst="B", mode=None)
    'PI'
    >>> get_key_from_mission_info(info, "ecol", "BU", inst="A", mode="M1")
    'BLA'
    >>> get_key_from_mission_info(info, "ecol", "BU", inst="C", mode="M1")
    'X'
    >>> get_key_from_mission_info(info, "ghghg", "BU", inst="C", mode="M1")
    'BU'
    """
    filt_info = make_dictionary_lowercase(info, recursive=True)
    key = key.lower()
    if inst is not None:
        inst = inst.lower()
        if inst in filt_info:
            filt_info.update(filt_info[inst])
            filt_info.pop(inst)
    if mode is not None:
        mode = mode.lower()
        if mode in filt_info:
            filt_info.update(filt_info[mode])
            filt_info.pop(mode)

    if key in filt_info:
        return filt_info[key]
    return default


def lcurve_from_fits(
    fits_file,
    gtistring="GTI",
    timecolumn="TIME",
    ratecolumn=None,
    ratehdu=1,
    fracexp_limit=0.9,
    outfile=None,
    noclobber=False,
    outdir=None,
):
    """Load a lightcurve from a fits file.

    .. note ::
        FITS light curve handling is still under testing.
        Absolute times might be incorrect depending on the light curve format.

    Parameters
    ----------
    fits_file : str
        File name of the input light curve in FITS format

    Returns
    -------
    data : dict
        Dictionary containing all information needed to create a
        :class:`stingray.Lightcurve` object

    Other Parameters
    ----------------
    gtistring : str
        Name of the GTI extension in the FITS file
    timecolumn : str
        Name of the column containing times in the FITS file
    ratecolumn : str
        Name of the column containing rates in the FITS file
    ratehdu : str or int
        Name or index of the FITS extension containing the light curve
    fracexp_limit : float
        Minimum exposure fraction allowed
    noclobber : bool
        If True, do not overwrite existing files
    """
    warnings.warn(
        """WARNING! FITS light curve handling is still under testing.
        Absolute times might be incorrect."""
    )
    # TODO:
    # treat consistently TDB, UTC, TAI, etc. This requires some documentation
    # reading. For now, we assume TDB
    from astropy.io import fits as pf
    from astropy.time import Time
    import numpy as np
    from stingray.gti import create_gti_from_condition

    lchdulist = pf.open(fits_file)
    lctable = lchdulist[ratehdu].data

    # Units of header keywords
    tunit = lchdulist[ratehdu].header["TIMEUNIT"]

    try:
        mjdref = high_precision_keyword_read(lchdulist[ratehdu].header, "MJDREF")
        mjdref = Time(mjdref, scale="tdb", format="mjd")
    except Exception:
        mjdref = None

    try:
        instr = lchdulist[ratehdu].header["INSTRUME"]
    except Exception:
        instr = "EXTERN"

    # ----------------------------------------------------------------
    # Trying to comply with all different formats of fits light curves.
    # It's a madness...
    try:
        tstart = high_precision_keyword_read(lchdulist[ratehdu].header, "TSTART")
        tstop = high_precision_keyword_read(lchdulist[ratehdu].header, "TSTOP")
    except Exception:  # pragma: no cover
        raise (Exception("TSTART and TSTOP need to be specified"))

    # For nulccorr lcs this would work

    timezero = high_precision_keyword_read(lchdulist[ratehdu].header, "TIMEZERO")
    # Sometimes timezero is "from tstart", sometimes it's an absolute time.
    # This tries to detect which case is this, and always consider it
    # referred to tstart
    timezero = assign_value_if_none(timezero, 0)

    # for lcurve light curves this should instead work
    if tunit == "d":
        # TODO:
        # Check this. For now, I assume TD (JD - 2440000.5).
        # This is likely wrong
        timezero = Time(2440000.5 + timezero, scale="tdb", format="jd")
        tstart = Time(2440000.5 + tstart, scale="tdb", format="jd")
        tstop = Time(2440000.5 + tstop, scale="tdb", format="jd")
        # if None, use NuSTAR default MJDREF
        mjdref = assign_value_if_none(
            mjdref,
            Time(np.longdouble("55197.00076601852"), scale="tdb", format="mjd"),
        )

        timezero = (timezero - mjdref).to("s").value
        tstart = (tstart - mjdref).to("s").value
        tstop = (tstop - mjdref).to("s").value

    if timezero > tstart:
        timezero -= tstart

    time = np.array(lctable.field(timecolumn), dtype=np.longdouble)
    if time[-1] < tstart:
        time += timezero + tstart
    else:
        time += timezero

    try:
        dt = high_precision_keyword_read(lchdulist[ratehdu].header, "TIMEDEL")
        if tunit == "d":
            dt *= 86400
    except Exception:
        warnings.warn(
            "Assuming that TIMEDEL is the median difference between the" " light curve times",
            AstropyUserWarning,
        )
        # Avoid NaNs
        good = time == time
        dt = np.median(np.diff(time[good]))

    # ----------------------------------------------------------------
    if ratecolumn is None:
        for name in ["RATE", "RATE1", "COUNTS"]:
            if name in lctable.names:
                ratecolumn = name
                break
        else:  # pragma: no cover
            raise ValueError("None of the accepted rate columns were found in the file")

    rate = np.array(lctable.field(ratecolumn), dtype=float)

    errorcolumn = "ERROR"
    if ratecolumn == "RATE1":
        errorcolumn = "ERROR1"

    try:
        rate_e = np.array(lctable.field(errorcolumn), dtype=np.longdouble)
    except Exception:
        rate_e = np.zeros_like(rate)

    if "RATE" in ratecolumn:
        rate *= dt
        rate_e *= dt

    try:
        fracexp = np.array(lctable.field("FRACEXP"), dtype=np.longdouble)
    except Exception:
        fracexp = np.ones_like(rate)

    good_intervals = (rate == rate) * (fracexp >= fracexp_limit)

    rate[good_intervals] /= fracexp[good_intervals]
    rate_e[good_intervals] /= fracexp[good_intervals]

    rate[~good_intervals] = 0

    try:
        gtitable = lchdulist[gtistring].data
        gti_list = np.array(
            [[a, b] for a, b in zip(gtitable.field("START"), gtitable.field("STOP"))],
            dtype=np.longdouble,
        )
    except Exception:
        gti_list = create_gti_from_condition(time, good_intervals)

    lchdulist.close()

    res = {
        "time": time,
        "counts": rate,
        "err": rate_e,
        "gti": gti_list,
        "mjdref": mjdref.mjd,
        "dt": dt,
        "instr": instr,
        "header": lchdulist[ratehdu].header.tostring(),
    }
    return res


def load_events_and_gtis(
    fits_file,
    additional_columns=None,
    gtistring=None,
    gti_file=None,
    hduname=None,
    column=None,
):
    """Load event lists and GTIs from one or more files.

    Loads event list from HDU EVENTS of file fits_file, with Good Time
    intervals. Optionally, returns additional columns of data from the same
    HDU of the events.

    Parameters
    ----------
    fits_file : str

    Other parameters
    ----------------
    additional_columns: list of str, optional
        A list of keys corresponding to the additional columns to extract from
        the event HDU (ex.: ['PI', 'X'])
    gtistring : str
        Comma-separated list of accepted GTI extensions (default GTI,STDGTI),
        with or without appended integer number denoting the detector
    gti_file : str, default None
        External GTI file
    hduname : str or int, default 1
        Name of the HDU containing the event list
    column : str, default None
        The column containing the time values. If None, we use the name
        specified in the mission database, and if there is nothing there,
        "TIME"
    return_limits: bool, optional
        Return the TSTART and TSTOP keyword values

    Returns
    -------
    retvals : Object with the following attributes:
        ev_list : array-like
            Event times in Mission Epoch Time
        gti_list: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            GTIs in Mission Epoch Time
        additional_data: dict
            A dictionary, where each key is the one specified in additional_colums.
            The data are an array with the values of the specified column in the
            fits file.
        t_start : float
            Start time in Mission Epoch Time
        t_stop : float
            Stop time in Mission Epoch Time
        pi_list : array-like
            Raw Instrument energy channels
        cal_pi_list : array-like
            Calibrated PI channels (those that can be easily converted to energy
            values, regardless of the instrument setup.)
        energy_list : array-like
            Energy of each photon in keV (only for NuSTAR, NICER, XMM)
        instr : str
            Name of the instrument (e.g. EPIC-pn or FPMA)
        mission : str
            Name of the instrument (e.g. XMM or NuSTAR)
        mjdref : float
            MJD reference time for the mission
        header : str
            Full header of the FITS file, for debugging purposes
        detector_id : array-like, int
            Detector id for each photon (e.g. each of the CCDs composing XMM's or
            Chandra's instruments)
    """
    from astropy.io import fits as pf

    hdulist = pf.open(fits_file)
    probe_header = hdulist[0].header
    # Let's look for TELESCOP here. This is the most common keyword to be
    # found in well-behaved headers. If it is not in header 0, I take this key
    # and the remaining information from header 1.
    if "TELESCOP" not in probe_header:
        probe_header = hdulist[1].header
    mission_key = "MISSION"
    if mission_key not in probe_header:
        mission_key = "TELESCOP"
    mission = probe_header[mission_key].lower()

    mission_specific_processing = mission_specific_event_interpretation(mission)
    if mission_specific_processing is not None:
        mission_specific_processing(hdulist)

    db = read_mission_info(mission)
    instkey = get_key_from_mission_info(db, "instkey", "INSTRUME")
    instr = mode = None
    if instkey in probe_header:
        instr = probe_header[instkey].strip()

    modekey = get_key_from_mission_info(db, "dmodekey", None, instr)
    if modekey is not None and modekey in probe_header:
        mode = probe_header[modekey].strip()

    if gtistring is None:
        gtistring = get_key_from_mission_info(db, "gti", "GTI,STDGTI", instr, mode)
    if hduname is None:
        hduname = get_key_from_mission_info(db, "events", "EVENTS", instr, mode)

    if hduname not in hdulist:
        warnings.warn(f"HDU {hduname} not found. Trying first extension")
        hduname = 1

    datatable = hdulist[hduname].data
    header = hdulist[hduname].header

    ephem = timeref = timesys = None

    if "PLEPHEM" in header:
        # For the rare cases where this is a number, e.g. 200, I add `str`
        # It's supposed to be a string.
        ephem = str(header["PLEPHEM"]).strip().lstrip("JPL-").lower()
    if "TIMEREF" in header:
        timeref = header["TIMEREF"].strip().lower()
    if "TIMESYS" in header:
        timesys = header["TIMESYS"].strip().lower()

    if column is None:
        column = get_key_from_mission_info(db, "time", "TIME", instr, mode)
    ev_list = np.array(datatable.field(column), dtype=np.longdouble)

    detector_id = None
    ckey = get_key_from_mission_info(db, "ccol", "NONE", instr, mode)
    if ckey != "NONE" and ckey in datatable.columns.names:
        detector_id = datatable.field(ckey)

    det_number = None if detector_id is None else list(set(detector_id))

    timedel = np.longdouble(0.0)
    if "TIMEDEL" in header:
        timedel = np.longdouble(header["TIMEDEL"])
    timezero = np.longdouble(0.0)
    if "TIMEZERO" in header:
        timezero = np.longdouble(header["TIMEZERO"])

    if "TIMEPIXR" in header:
        timezero += (0.5 - np.longdouble(header["TIMEPIXR"])) * timedel

    ev_list += timezero

    t_start = ev_list[0]
    t_stop = ev_list[-1]
    if "TSTART" in header:
        t_start = np.longdouble(header["TSTART"])
    if "TSTOP" in header:
        t_stop = np.longdouble(header["TSTOP"])

    mjdref = np.longdouble(high_precision_keyword_read(header, "MJDREF"))

    # Read and handle GTI extension
    accepted_gtistrings = gtistring.split(",")

    if gti_file is None:
        # Select first GTI with accepted name
        try:
            gti_list = get_gti_from_all_extensions(
                hdulist,
                accepted_gtistrings=accepted_gtistrings,
                det_numbers=det_number,
            )
        except Exception as e:  # pragma: no cover
            warnings.warn(
                (
                    f"No valid GTI extensions found. \nError: {str(e)}\n"
                    "GTIs will be set to the entire time series."
                ),
                AstropyUserWarning,
            )
            gti_list = np.array([[t_start, t_stop]], dtype=np.longdouble)
    else:
        gti_list = load_gtis(gti_file, gtistring)

    pi_col = get_key_from_mission_info(db, "ecol", "PI", instr, mode)
    if additional_columns is None:
        additional_columns = [pi_col]
    if pi_col not in additional_columns:
        additional_columns.append(pi_col)
    # If data were already calibrated, use this!
    additional_data = _get_additional_data(datatable, additional_columns)
    if "energy" not in additional_columns:
        additional_data.update(_get_additional_data(datatable, ["energy"], warn_if_missing=False))
    del additional_columns

    hdulist.close()
    # Sort event list
    if not is_sorted(ev_list):
        warnings.warn("Warning: input data are not sorted. Sorting them for you.")
        order = np.argsort(ev_list)
        ev_list = ev_list[order]
        if detector_id is not None:
            detector_id = detector_id[order]

        additional_data = order_list_of_arrays(additional_data, order)

    pi = additional_data[pi_col].astype(np.float32)
    cal_pi = pi

    # EventReadOutput() is an empty class. We will assign a number of attributes to
    # it, like the arrival times of photons, the energies, and some information
    # from the header.
    returns = EventReadOutput()

    returns.ev_list = ev_list
    returns.gti_list = gti_list
    returns.pi_list = pi
    returns.cal_pi_list = cal_pi

    if "energy" in additional_data and np.any(additional_data["energy"] > 0.0):
        returns.energy_list = additional_data["energy"]
    else:
        try:
            func = get_rough_conversion_function(
                mission, instrument=instr, epoch=t_start / 86400 + mjdref
            )
            returns.energy_list = func(cal_pi, detector_id=detector_id)
            logger.info(
                f"A default calibration was applied to the {mission} data. "
                "See io.rough_calibration for details. "
                "Use the `rmf_file` argument in `EventList.read`, or calibrate with "
                "`EventList.convert_pi_to_energy(rmf_file)`, if you want to apply a specific "
                "response matrix"
            )
        except ValueError:
            returns.energy_list = None
    returns.instr = instr.lower()
    returns.mission = mission.lower()
    returns.mjdref = mjdref
    returns.header = header.tostring()
    returns.additional_data = additional_data
    returns.t_start = t_start
    returns.t_stop = t_stop
    returns.detector_id = detector_id
    returns.ephem = ephem
    returns.timeref = timeref
    returns.timesys = timesys

    return returns


class EventReadOutput:
    def __init__(self):
        pass


class FITSTimeseriesReader(object):
    main_array_attr = "time"

    def __init__(
        self,
        fname,
        output_class=None,
        force_hduname=None,
        gti_file=None,
        gtistring=None,
        additional_columns=None,
        data_kind="events",
    ):
        self.fname = fname
        self._meta_attrs = []
        self.gtistring = gtistring
        self.output_class = output_class
        self.additional_columns = additional_columns
        if "EventList" in str(output_class) or data_kind.lower() in ["events", "times"]:
            self._initialize_header_events(fname, force_hduname=force_hduname)
        else:
            raise NotImplementedError(
                "Only events are supported by FITSTimeseriesReader at the moment. "
                f"{data_kind} is an unknown data kind."
            )
        self.data_kind = data_kind
        if additional_columns is None and self.detector_key != "NONE":
            additional_columns = [self.detector_key]
        elif self.detector_key != "NONE":
            additional_columns.append(self.detector_key)
        self.data_hdu = self.data_hdus[self.hduname]
        self.gti_file = gti_file
        self._read_gtis(self.gti_file)

    @property
    def time(self):
        return self[:].time

    def meta_attrs(self):
        return self._meta_attrs

    def _add_meta_attr(self, name, value):
        """Add a meta attribute to the object."""
        if name not in self._meta_attrs:
            self._meta_attrs.append(name)
        setattr(self, name, value)

    @property
    def exposure(self):
        """
        Return the total exposure of the time series, i.e. the sum of the GTIs.

        Returns
        -------
        total_exposure : float
            The total exposure of the time series, in seconds.
        """

        return get_total_gti_length(self.gti)

    def __getitem__(self, index):
        """Return an element or a slice of the object, e.g. ``ts[1]`` or ``ts[1:2]."""

        data = self.data_hdu.data[index]

        return self.transform_slice(data)

    def transform_slice(self, data):
        # Here there will be some logic to understand whether transfomring to events or something else

        if self.data_kind == "times":
            return data[self.time_column][:] + self.timezero
        if self.output_class is None:
            return data
        if self.data_kind == "events":
            return self._transform_slice_into_events(data)

    def _transform_slice_into_events(self, data):
        """Take a slice of data from a FITS event file and make it a StingrayTimeseries.

        Data taken from a FITS file will typically be a Numpy record array. This method
        tries to interpret the information contained in the record array based on what
        we know of the mission and the instrument. For sure, there will be a TIME column
        that will become the ``time`` array of the timeseries object. If there is a PI/PHA
        column, it will become the ``pi`` array, and if we know the conversion law for the mission,
        this will also be converted to energy. If there is an ENERGY column, it will directly
        be loaded into the energy column.
        Additional meta (e.g. GTIs, MJDREF, etc.) information will also be added to the object.

        Parameters
        ----------
        data : np.recarray
            The slice of data to transform

        Returns
        -------
        new_ts : any StingrayTimeseries subclass
            The transformed timeseries object. It will typically be an ``EventList`` object,
            but the user can change this by specifying the ``output_class`` parameter in the
            constructor of the reader.

        """
        columns = [self.time_column]
        for col in self.pi_column, self.energy_column:
            if col is not None:
                columns.append(col)
        new_ts = self.output_class()
        if self._mission_specific_processing is not None:
            data = self._mission_specific_processing(data, header=self.header, hduname=self.hduname)

        # Set the times
        setattr(
            new_ts,
            self.main_array_attr,
            data[self.time_column][:] + self.timezero,
        )
        # Get conversion function PI->Energy
        try:
            pi_energy_func = get_rough_conversion_function(
                self.mission,
                instrument=self.instr,
                epoch=self.t_start / 86400 + self.mjdref,
            )
        except ValueError:
            pi_energy_func = None
        if self.energy_column in data.dtype.names:
            conversion = 1
            if (
                hasattr(data.columns[self.energy_column], "unit")
                and (unit := data.columns[self.energy_column].unit) is not None
            ):
                conversion = (1 * u.Unit(unit)).to(u.keV).value
            new_ts.energy = data[self.energy_column] * conversion
        elif self.pi_column is not None and self.pi_column.lower() in [
            col.lower() for col in data.dtype.names
        ]:
            new_ts.pi = data[self.pi_column]
            if pi_energy_func is not None:
                new_ts.energy = pi_energy_func(new_ts.pi)

        det_numbers = None
        if self.detector_key is not None and self.detector_key in data.dtype.names:
            new_ts.detector_id = data[self.detector_key]
            det_numbers = list(set(new_ts.detector_id))
            self._read_gtis(self.gti_file, det_numbers=det_numbers)

        if self.additional_columns is not None:
            for col in self.additional_columns:
                if col == self.detector_key:
                    continue
                if col in data.dtype.names:
                    setattr(new_ts, col.replace(".", "_").lower(), data[col])

        for attr in self.meta_attrs():
            local_value = getattr(self, attr)
            if attr in ["t_start", "t_stop", "gti"] and local_value is not None:
                setattr(new_ts, attr, local_value + self.timezero)
            else:
                setattr(new_ts, attr, local_value)

        return new_ts

    def _initialize_header_events(self, fname, force_hduname=None):
        """Read the header of the FITS file and set the relevant attributes.

        When possibile, some mission-specific information is read from the keywords and
        extension names found in ``xselect.mdb``.

        Parameters
        ----------
        fname : str
            The name of the FITS file to read

        Other parameters
        ----------------
        force_hduname : str or int, default None
            If not None, the name of the HDU to read. If None, an extension called
            EVENTS or the first extension.
        """
        hdulist = fits_open_including_remote(fname)
        self.data_hdus = hdulist
        if not force_hduname:
            for hdu in hdulist:
                if "TELESCOP" in hdu.header or "MISSION" in hdu.header:
                    probe_header = hdu.header
                    break
        else:
            probe_header = hdulist[force_hduname].header

        # We need the minimal information to read the mission database.
        # That is, the name of the mission/telescope, the instrument and,
        # if available, the observing mode.
        mission_key = "MISSION"
        if mission_key not in probe_header:
            mission_key = "TELESCOP"
        self._add_meta_attr("mission", probe_header[mission_key].lower())
        self._add_meta_attr(
            "_mission_specific_processing",
            mission_specific_event_interpretation(self.mission),
        )

        # Now, we read the mission info, and we try to get the relevant
        # information from the header using the mission-specific keywords.
        db = read_mission_info(self.mission)
        instkey = get_key_from_mission_info(db, "instkey", "INSTRUME")
        instr = mode = None
        if instkey in probe_header:
            instr = probe_header[instkey].strip()

        modekey = get_key_from_mission_info(db, "dmodekey", None, instr)
        if modekey is not None and modekey in probe_header:
            mode = probe_header[modekey].strip()
        self._add_meta_attr("instr", instr)
        self._add_meta_attr("mode", mode)

        gtistring = self.gtistring

        if self.gtistring is None:
            gtistring = get_key_from_mission_info(db, "gti", "GTI,STDGTI", instr, self.mode)
        self._add_meta_attr("gtistring", gtistring)

        if force_hduname is None:
            hduname = get_key_from_mission_info(db, "events", "EVENTS", instr, self.mode)
        else:
            hduname = force_hduname

        # If the EVENT/``force_hduname`` extension is not found, try the first extension
        # which is usually the one containing the data
        if hduname not in hdulist:
            warnings.warn(f"HDU {hduname} not found. Trying first extension")
            hduname = 1
        self._add_meta_attr("hduname", hduname)

        header = hdulist[hduname].header
        if "OBS_ID" in header:
            self._add_meta_attr("obsid", header["OBS_ID"])

        timedel = np.longdouble(0.0)
        if "TIMEDEL" in header:
            timedel = np.longdouble(header["TIMEDEL"])

        self._add_meta_attr("dt", timedel)

        # self.header has to be a string, for backwards compatibility and... for convenience!
        # No need to cope with dicts working badly with Netcdf, for example. The header
        # can be saved back and forth to files and be interpreted through
        # fits.Header.fromstring(self.header) when needed.
        self._add_meta_attr("header", hdulist[self.hduname].header.tostring())
        self._add_meta_attr("nphot", header["NAXIS2"])

        # These are the important keywords for timing.
        ephem = timeref = timesys = None
        if "PLEPHEM" in header:
            # For the rare cases where this is a number, e.g. 200, I add `str`
            # It's supposed to be a string.
            ephem = str(header["PLEPHEM"]).strip().lstrip("JPL-").lower()
        if "TIMEREF" in header:
            timeref = header["TIMEREF"].strip().lower()
        if "TIMESYS" in header:
            timesys = header["TIMESYS"].strip().lower()
        self._add_meta_attr("ephem", ephem)
        self._add_meta_attr("timeref", timeref)
        self._add_meta_attr("timesys", timesys)

        timezero = np.longdouble(0.0)
        if "TIMEZERO" in header:
            timezero = np.longdouble(header["TIMEZERO"])

        if "TIMEPIXR" in header:
            timezero += (0.5 - np.longdouble(header["TIMEPIXR"])) * timedel

        t_start = t_stop = None
        if "TSTART" in header:
            t_start = np.longdouble(header["TSTART"])
        if "TSTOP" in header:
            t_stop = np.longdouble(header["TSTOP"])
        self._add_meta_attr("timezero", timezero)

        self._add_meta_attr("t_start", t_start)
        self._add_meta_attr("t_stop", t_stop)

        self._add_meta_attr(
            "time_column",
            get_key_from_mission_info(db, "time", "TIME", instr, mode),
        )

        self._add_meta_attr(
            "detector_key",
            get_key_from_mission_info(db, "ccol", "NONE", instr, mode),
        )

        self._add_meta_attr("mjdref", np.longdouble(high_precision_keyword_read(header, "MJDREF")))

        # Try to get the information needed to calculate the event energy. We start from the
        # PI column
        default_pi_column = get_key_from_mission_info(db, "ecol", "PI", instr, self.mode)
        if isinstance(default_pi_column, str):
            default_pi_column = _case_insensitive_search_in_list(
                default_pi_column, hdulist[self.hduname].data.columns.names
            )

        self._add_meta_attr("pi_column", default_pi_column)

        # If a column named "energy" is found, we read it and assume the energy conversion
        # is already done.
        energy_column = _case_insensitive_search_in_list(
            "energy", hdulist[self.hduname].data.columns.names
        )
        self._add_meta_attr("energy_column", energy_column)

    def _read_gtis(self, gti_file=None, det_numbers=None):
        """Read GTIs from the FITS file."""
        # This is ugly, but if, e.g., we are reading XMM data, we *need* the
        # detector number to access GTIs.
        # So, here I'm reading a bunch of rows hoping that they represent the
        # detector number population
        if self.detector_key is not None:
            hdul = self.data_hdus
            data = hdul[self.hduname].data
            if self.detector_key in data.dtype.names:
                probe_vals = data[:100][self.detector_key]
                det_numbers = list(set(probe_vals))
            del hdul

        accepted_gtistrings = self.gtistring.split(",")
        gti_list = None

        if gti_file is not None:
            self._add_meta_attr("gti", load_gtis(gti_file, self.gtistring))
            return

        # Select first GTI with accepted name
        try:
            gti_list = get_gti_from_all_extensions(
                self.fname,
                accepted_gtistrings=accepted_gtistrings,
                det_numbers=det_numbers,
            )
        except Exception as e:  # pragma: no cover
            warnings.warn(
                (
                    f"No valid GTI extensions found. \nError: {str(e)}\n"
                    "GTIs will be set to the entire time series."
                ),
            )

        self._add_meta_attr("gti", gti_list)

    def _get_idx_from_time_range(self, start, stop):
        """Get the index of the times in the event list that fall within the given time range.

        Instead of reading all the data from the file and doing ``np.searchsorted``, which could
        easily fill up the memory, this function does a two-step procedure. It first uses
        ``self._trace_nphots_in_file`` to get a grid of times and their corresponding
        indices in the file. Then, it reads only the data that strictly include the requested time
        range, and on those data it performs a searchsorted operation. The final indices will be
        summed to the lower index of the data that was read.

        Parameters
        ----------
        start : float
            Start time of the interval
        stop : float
            Stop time of the interval

        Returns
        -------
        lower_edge : int
            Index of the first photon in the requested time range
        upper_edge : int
            Index of the last photon in the requested time range
        """
        time_edges, idx_edges = self._trace_nphots_in_file(
            nedges=int(self.exposure // (stop - start)) + 2
        )

        raw_min_idx = np.searchsorted(time_edges, start, side="left")
        raw_max_idx = np.searchsorted(time_edges, stop, side="right")

        raw_min_idx = max(0, raw_min_idx - 2)
        raw_max_idx = min(time_edges.size - 1, raw_max_idx + 2)

        raw_lower_edge = idx_edges[raw_min_idx]
        raw_upper_edge = idx_edges[raw_max_idx]

        assert (
            start - time_edges[raw_min_idx] >= 0
        ), f"Start: {start}; {start - time_edges[raw_min_idx]} > 0"
        assert (
            time_edges[raw_max_idx] - stop >= 0
        ), f"Stop: {stop}; {time_edges[raw_max_idx] - stop} < 0"

        with fits_open_including_remote(self.fname) as hdulist:
            filtered_times = hdulist[self.hduname].data[self.time_column][
                raw_lower_edge : raw_upper_edge + 1
            ]
            # lower_edge = np.searchsorted(filtered_times, [start, stop])
            lower_edge, upper_edge = np.searchsorted(filtered_times, [start, stop])
            # Searchsorted will find the first number above stop. We want the last number below stop!
            upper_edge -= 1

        return lower_edge + raw_lower_edge, upper_edge + raw_lower_edge

    def apply_gti_lists(self, new_gti_lists, root_file_name=None, fmt=DEFAULT_FORMAT):
        """Split the event list into different files, each with a different GTI.

        Parameters
        ----------
        new_gti_lists : list of lists
            A list of lists of GTIs. Each sublist should contain a list of GTIs
            for a new file.

        Other Parameters
        ----------------
        root_file_name : str, default None
            The root name of the output files. The file name will be appended with
            "_00", "_01", etc.
            If None, a generator is returned instead of writing the files.
        fmt : str
            The format of the output files. Default is 'hdf5'.

        Returns
        -------
        output_files : list of str
            A list of the output file names.

        """

        if len(new_gti_lists[0]) == len(self.gti) and np.all(
            np.abs(np.asanyarray(new_gti_lists[0]).flatten() - self.gti.flatten()) < 1e-3
        ):
            ev = self[:]
            if root_file_name is None:
                yield ev
            else:
                output_file = root_file_name + f"_00." + fmt.lstrip(".")
                ev.write(output_file, fmt=fmt)
                yield output_file

        else:
            for i, gti in enumerate(new_gti_lists):
                if len(gti) == 0:
                    continue
                gti = np.asarray(gti)
                lower_edge, upper_edge = self._get_idx_from_time_range(gti[0, 0], gti[-1, 1])

                ev = self[lower_edge : upper_edge + 1]
                if hasattr(ev, "gti"):
                    ev.gti = gti

                if root_file_name is not None:
                    new_file = root_file_name + f"_{i:002d}." + fmt.lstrip(".")
                    logger.info(f"Writing {new_file}")
                    ev.write(new_file, fmt=fmt)
                    yield new_file
                else:
                    yield ev

    def _trace_nphots_in_file(self, nedges=1001):
        """Trace the number of photons as time advances in the file.

        This function traces the number of photons as time advances in the file.
        This is a way to quickly map the distribution of photons in time, without
        reading the entire file. This map can be useful to then access the wanted
        data without loading all the file in memory.

        Other Parameters
        ----------------
        nedges : int
            The number of time edges to trace. Default is 1001.

        Returns
        -------
        time_edges : np.ndarray
            The time edges
        idx_edges : np.ndarray
            The index edges
        """

        if hasattr(self, "_time_edges") and len(self._time_edges) >= nedges:
            return self._time_edges, self._idx_edges

        fname = self.fname

        with fits_open_including_remote(fname) as hdul:
            size = hdul[1].header["NAXIS2"]
            nedges = min(nedges, size // 10 + 2)

            time_edges = np.zeros(nedges)
            idx_edges = np.zeros(nedges, dtype=int)
            for i, edge_idx in enumerate(np.linspace(0, size - 1, nedges).astype(int)):
                idx_edges[i] = edge_idx
                time_edges[i] = hdul[1].data["TIME"][edge_idx]

            mingti, maxgti = np.min(self.gti), np.max(self.gti)
            if time_edges[0] > mingti:
                time_edges[0] = mingti
            if time_edges[-1] < maxgti:
                time_edges[-1] = maxgti

        self._time_edges, self._idx_edges = time_edges, idx_edges

        return time_edges, idx_edges

    def split_by_number_of_samples(self, nsamples, root_file_name=None, fmt=DEFAULT_FORMAT):
        """Split the event list into different files, each with approx. the given no. of photons.

        Parameters
        ----------
        nsamples : int
            The number of photons in each output file.

        Other Parameters
        ----------------
        root_file_name : str, default None
            The root name of the output files. The file name will be appended with
            "_00", "_01", etc.
            If None, a generator is returned instead of writing the files.
        fmt : str
            The format of the output files. Default is 'hdf5'.

        Returns
        -------
        output_files : list of str
            A list of the output file names.
        """
        n_intervals = int(np.rint(self.nphot / nsamples))
        exposure_per_interval = self.exposure / n_intervals
        new_gti_lists = split_gtis_by_exposure(self.gti, exposure_per_interval)

        return self.apply_gti_lists(new_gti_lists, root_file_name=root_file_name, fmt=fmt)

    def filter_at_time_intervals(
        self, time_intervals, root_file_name=None, fmt=DEFAULT_FORMAT, check_gtis=True
    ):
        """Filter the event list at the given time intervals.

        Parameters
        ----------
        time_intervals : 2-d float array
            List of time intervals of the form ``[[time0_0, time0_1], [time1_0, time1_1], ...]``

        Other Parameters
        ----------------
        root_file_name : str, default None
            The root name of the output files. The file name will be appended with
            "_00", "_01", etc.
            If None, a generator is returned instead of writing the files.
        fmt : str
            The format of the output files. Default is 'hdf5'.

        Returns
        -------
        output_files : list of str
            A list of the output file names.
        """
        if len(np.shape(time_intervals)) == 1:
            time_intervals = [time_intervals]
        if check_gtis:
            new_gti = [cross_two_gtis(self.gti, [t_int]) for t_int in time_intervals]
        else:
            new_gti = [np.asarray([t_int]) for t_int in time_intervals]
        return self.apply_gti_lists(new_gti, root_file_name=root_file_name, fmt=fmt)


def mkdir_p(path):  # pragma: no cover
    """Safe ``mkdir`` function

    Parameters
    ----------
    path : str
        The absolute path to the directory to be created
    """
    import os

    os.makedirs(path, exist_ok=True)


def read_header_key(fits_file, key, hdu=1):
    """Read the header key key from HDU hdu of the file ``fits_file``.

    Parameters
    ----------
    fits_file: str
        The file name and absolute path to the event file.

    key: str
        The keyword to be read

    Other Parameters
    ----------------
    hdu : int
        Index of the HDU extension from which the header key to be read.

    Returns
    -------
    value : object
        The value stored under ``key`` in ``fits_file``
    """

    hdulist = fits_open_including_remote(fits_file, ignore_missing_end=True)
    try:
        value = hdulist[hdu].header[key]
    except KeyError:  # pragma: no cover
        value = ""
    hdulist.close()
    return value


def ref_mjd(fits_file, hdu=1):
    """Read ``MJDREFF``, ``MJDREFI`` or, if failed, ``MJDREF``, from the FITS header.

    Parameters
    ----------
    fits_file : str
        The file name and absolute path to the event file.

    Other Parameters
    ----------------
    hdu : int
        Index of the HDU extension from which the header key to be read.

    Returns
    -------
    mjdref : numpy.longdouble
        the reference MJD
    """

    if isinstance(fits_file, Iterable) and not is_string(fits_file):  # pragma: no cover
        fits_file = fits_file[0]
        logger.info("opening %s" % fits_file)

    hdulist = fits_open_including_remote(fits_file, ignore_missing_end=True)

    ref_mjd_val = high_precision_keyword_read(hdulist[hdu].header, "MJDREF")

    hdulist.close()
    return ref_mjd_val


def common_name(str1, str2, default="common"):
    """Strip two strings of the letters not in common.

    Filenames must be of same length and only differ by a few letters.

    Parameters
    ----------
    str1 : str
    str2 : str

    Other Parameters
    ----------------
    default : str
        The string to return if ``common_str`` is empty

    Returns
    -------
    common_str : str
        A string containing the parts of the two names in common

    """
    if not len(str1) == len(str2):
        return default
    common_str = ""
    # Extract the MP root of the name (in case they're event files)

    for i, letter in enumerate(str1):
        if str2[i] == letter:
            common_str += letter
    # Remove leading and trailing underscores and dashes
    common_str = common_str.rstrip("_").rstrip("-")
    common_str = common_str.lstrip("_").lstrip("-")
    if common_str == "":
        common_str = default
    logger.debug("common_name: %s %s -> %s" % (str1, str2, common_str))
    return common_str


def split_numbers(number, shift=0):
    """
    Split high precision number(s) into doubles.

    You can specify the number of shifts to move the decimal point.

    Parameters
    ----------
    number: long double
        The input high precision number which is to be split

    Other parameters
    ----------------
    shift: integer
        Move the cut by `shift` decimal points to the right (left if negative)

    Returns
    -------
    number_I: double
        First part of high precision number

    number_F: double
        Second part of high precision number

    Examples
    --------
    >>> n = 12.34
    >>> i, f = split_numbers(n)
    >>> assert i == 12
    >>> assert np.isclose(f, 0.34)
    >>> assert np.allclose(split_numbers(n, 2), (12.34, 0.0))
    >>> assert np.allclose(split_numbers(n, -1), (10.0, 2.34))
    """
    if isinstance(number, Iterable):
        number = np.asanyarray(number)
        number *= 10**shift
        mods = [math.modf(n) for n in number]
        number_F = [f for f, _ in mods]
        number_I = [i for _, i in mods]
    else:
        number *= 10**shift
        number_F, number_I = math.modf(number)

    return np.double(number_I) / 10**shift, np.double(number_F) / 10**shift


def savefig(filename, **kwargs):
    """
    Save a figure plotted by ``matplotlib``.

    Note : This function is supposed to be used after the ``plot``
    function. Otherwise it will save a blank image with no plot.

    Parameters
    ----------
    filename : str
        The name of the image file. Extension must be specified in the
        file name. For example filename with `.png` extension will give a
        rasterized image while ``.pdf`` extension will give a vectorized
        output.

    kwargs : keyword arguments
        Keyword arguments to be passed to ``savefig`` function of
        ``matplotlib.pyplot``. For example use `bbox_inches='tight'` to
        remove the undesirable whitespace around the image.
    """

    if not plt.fignum_exists(1):
        utils.simon(
            "use ``plot`` function to plot the image first and "
            "then use ``savefig`` to save the figure."
        )

    plt.savefig(filename, **kwargs)


def _can_save_longdouble(probe_file: str, fmt: str) -> bool:
    """Check if a given file format can save tables with longdoubles.

    Try to save a table with a longdouble column, and if it doesn't work, catch the exception.
    If the exception is related to longdouble, return False (otherwise just raise it, this
    would mean there are larger problems that need to be solved). In this case, also warn that
    probably part of the data will not be saved.

    If no exception is raised, return True.

    Parameters
    ----------
    probe_file : str
        The name of the file to be used for probing
    fmt : str
        The format to be used for probing, in the ``format`` argument of ``Table.write``

    Returns
    -------
    yes_it_can : bool
        Whether the format can serialize the metadata
    """
    if not HAS_128:  # pragma: no cover
        # There are no known issues with saving longdoubles where numpy.float128 is not defined
        return True

    try:
        Table({"a": np.arange(0, 3, 1.212314).astype(np.float128)}).write(
            probe_file, format=fmt, overwrite=True
        )
        yes_it_can = True
        os.unlink(probe_file)
    except ValueError as e:
        if "float128" not in str(e):  # pragma: no cover
            raise
        warnings.warn(
            f"{fmt} output does not allow saving metadata at maximum precision. "
            "Converting to lower precision"
        )
        yes_it_can = False
    return yes_it_can


def _can_serialize_meta(probe_file: str, fmt: str) -> bool:
    """
    Try to save a table with meta to be serialized, and if it doesn't work, catch the exception.
    If the exception is related to serialization, return False (otherwise just raise it, this
    would mean there are larger problems that need to be solved). In this case, also warn that
    probably part of the data will not be saved.

    If no exception is raised, return True.

    Parameters
    ----------
    probe_file : str
        The name of the file to be used for probing
    fmt : str
        The format to be used for probing, in the ``format`` argument of ``Table.write``

    Returns
    -------
    yes_it_can : bool
        Whether the format can serialize the metadata
    """
    try:
        Table({"a": [3]}).write(probe_file, overwrite=True, format=fmt, serialize_meta=True)

        os.unlink(probe_file)
        yes_it_can = True
    except TypeError as e:
        if "serialize_meta" not in str(e):  # pragma: no cover
            raise
        warnings.warn(
            f"{fmt} output does not serialize the metadata at the moment. "
            "Some attributes will be lost."
        )
        yes_it_can = False
    return yes_it_can


def run_flx2xsp(infile, outroot):
    """Wrapper around the HEASOFT tool ``flx2xsp``.

    Converts a spectrum in text format to Xspec format.

    Parameters
    ----------
    infile: str
        The name of the input file, containing the spectrum in text format. The file should have
        four columns: the lower energy of the bin, the higher energy of the bin, the power in the bin and the error on the power in the bin.
    outroot: str
        The root name of the output files. The file name will be appended with
        ".pha" and ".rsp" for the different files that will be created. The output files will be in Xspec format.
    """
    try:
        sp.check_call(f"flx2xsp {infile} {outroot}.pha {outroot}.rsp".split())
    except FileNotFoundError:
        raise RuntimeError("You need to install and initialize HEASOFT to save in Xspec format")


def save_as_xspec(x, dx, y, yerr, outroot):
    """Save frequency spectra in a format readable to FTOOLS and Xspec.

    Parameters
    ----------
    x: float array
        The energies of the spectrum
    dx: float array
        The width of the energy bins
    y: float array
        The power of the spectrum in each bin
    yerr: float array
        The error on the power in each bin
    outroot: str
        The root name of the output files. The file name will be appended with
        ".txt", ".pha" and ".rsp" for the different files that will be created

    Notes
    -----
    Uses method described by Ingram and Done in Appendix A of
    `this paper<https://arxiv.org/pdf/1108.0789>__`
    """

    flo = x - dx / 2
    fhi = x + dx / 2
    power = y * dx
    power_err = yerr * dx
    outname = outroot + ".txt"

    np.savetxt(outname, np.transpose([flo, fhi, power, power_err]))
    run_flx2xsp(outname, outroot)

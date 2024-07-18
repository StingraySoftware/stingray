import numpy as np
import warnings
import copy

import numpy as np
import numpy.random as ra

from .utils import njit
from .loggingconfig import setup_logger

__all__ = ["Window1D", "Optimal1D"]

logger = setup_logger()


class Window1D(object):
    """
    Make a top hat filter (window function) for power spectrum or cross
    spectrum. It assumes that the first model is the QPO component
    Lorentzian model.

    Parameters
    ----------
    model: astropy.modeling.models class instance
        The compound model fit to the spectrum.

    Attributes
    ----------
    model:  astropy.modeling.models class instance
        The compound model fit to the spectrum.

    x_o: Parameter class instance
        Centroid of Lorentzian model.

    fwhm: Parameter class instance
        Full width at half maximum of Lorentzian model.
    """

    def __init__(self, model):
        self.model = model
        self.x_0 = model[0].x_0
        self.fwhm = model[0].fwhm

    def __call__(self, x):
        y = np.zeros((len(x),), dtype=np.float64)
        for i in range(len(x)):
            if np.abs(x[i] - self.x_0[0]) <= self.fwhm[0] / 2:
                y[i] = 1.0
        return y


class Optimal1D(object):
    """
    Make a optimal filter for power spectrum or cross spectrum.
    It assumes that the first model is the QPO component.

    Parameters
    ----------
    model: astropy.modeling.models class instance
        The compound model fit to the spectrum.

    Attributes
    ----------
    model:  astropy.modeling.models class instance
        The compound model fit to the spectrum.

    filter: astropy.modeling.models class instance
        It s the ratio of QPO component to the model fit to the spectrum.
    """

    def __init__(self, model):
        self.model = model
        qpo_component_model = self.model[0]
        all_components_model = self.model
        self.filter = qpo_component_model / all_components_model

    def __call__(self, x):
        return self.filter(x)


def _paralyzable_dead_time(event_list, dead_time):
    """Apply paralyzable dead time to an event list.

    Parameters
    ----------
    event_list : array of floats
        Event times of arrival
    dead_time: float
        Dead time (single value)

    Returns
    -------
    output_event_list : array of floats
        Filtered event times
    mask : array of bools
        Final mask, showing all good events in the original event list.
    """
    mask = np.ones(len(event_list), dtype=bool)
    dead_time_end = event_list + dead_time
    bad = dead_time_end[:-1] > event_list[1:]
    # Easy: paralyzable case. Here, events coming during dead time produce
    # more dead time. So...
    mask[1:][bad] = False

    return event_list[mask], mask


@njit()
def _nonpar_core(event_list, dead_time_end, mask):
    """Numba-compiled core of the non-paralyzable dead time calculation.

    Parameters
    ----------
    event_list : array of floats
        Event times of arrival
    dead_time_end : array of floats
        End of the dead time of each event
    mask : array of bools
        Final mask of good events. Initially, all entries must be ``True``

    Return
    ------
    mask : array of bools
        Final mask of good events
    """
    for i in range(1, len(event_list)):
        if event_list[i] < dead_time_end[i - 1]:
            dead_time_end[i] = dead_time_end[i - 1]
            mask[i] = False
    return mask


def _non_paralyzable_dead_time(event_list, dead_time):
    """Apply non-paralyzable dead time to an event list.

    Parameters
    ----------
    event_list : array of floats
        Event times of arrival
    dead_time: float
        Dead time (single value)

    Returns
    -------
    output_event_list : array of floats
        Filtered event times
    mask : array of bools
        Final mask, showing all good events in the original event list.
    """
    event_list_dbl = (event_list - event_list[0]).astype(np.double)
    dead_time_end = event_list_dbl + np.double(dead_time)
    mask = np.ones(event_list_dbl.size, dtype=bool)
    mask = _nonpar_core(event_list_dbl, dead_time_end, mask)
    return event_list[mask], mask


class DeadtimeFilterOutput(object):
    uf_events = None
    is_event = None
    deadtime = None
    mask = None
    bkg = None


def get_deadtime_mask(
    ev_list,
    deadtime,
    bkg_ev_list=None,
    dt_sigma=None,
    paralyzable=False,
    return_all=False,
    verbose=False,
):
    """Filter an event list for a given dead time.

    Parameters
    ----------
    ev_list : array-like
        The event list
    deadtime: float
        The (mean, if not constant) value of deadtime

    Other Parameters
    ----------------
    bkg_ev_list : array-like
        A background event list that affects dead time
    dt_sigma : float
        If specified, dead time will not have a single value but it will have
        a normal distribution with mean ``deadtime`` and standard deviation
        ``dt_sigma``.

    Returns
    -------
    mask : array-like, optional
        The mask that filters the input event list and produces the output
        event list.
    additional_output : object
        Object with all the following attributes. Only returned if
        `return_all` is True
        uf_events : array-like
            Unfiltered event list (events + background)
        is_event : array-like
            Boolean values; True if event, False if background
        deadtime : array-like
            Dead time values
        bkg : array-like
            The filtered background event list

    """
    additional_output = DeadtimeFilterOutput()

    # Create the total lightcurve, and a "kind" array that keeps track
    # of the events classified as "signal" (True) and "background" (False)
    if bkg_ev_list is not None:
        tot_ev_list = np.append(ev_list, bkg_ev_list)
        ev_kind = np.append(
            np.ones(len(ev_list), dtype=bool), np.zeros(len(bkg_ev_list), dtype=bool)
        )
        order = np.argsort(tot_ev_list)
        tot_ev_list = tot_ev_list[order]
        ev_kind = ev_kind[order]
        del order
    else:
        tot_ev_list = ev_list
        ev_kind = np.ones(len(ev_list), dtype=bool)

    additional_output.uf_events = tot_ev_list
    additional_output.is_event = ev_kind
    additional_output.deadtime = deadtime
    additional_output.uf_mask = np.ones(tot_ev_list.size, dtype=bool)
    additional_output.bkg = tot_ev_list[np.logical_not(ev_kind)]

    if deadtime <= 0.0:
        if deadtime < 0:
            raise ValueError("Dead time is less than 0. Please check.")
        retval = [np.ones(ev_list.size, dtype=bool), additional_output]
        return retval

    nevents = len(tot_ev_list)
    all_ev_kind = ev_kind.copy()

    if dt_sigma is not None:
        deadtime_values = ra.normal(deadtime, dt_sigma, nevents)
        deadtime_values[deadtime_values < 0] = 0.0
    else:
        deadtime_values = np.zeros(nevents) + deadtime

    initial_len = len(tot_ev_list)

    # Note: saved_mask gives the mask that produces tot_ev_list_filt from
    # tot_ev_list. The same mask can be used to also filter all other arrays.
    if paralyzable:
        tot_ev_list_filt, saved_mask = _paralyzable_dead_time(tot_ev_list, deadtime_values)

    else:
        tot_ev_list_filt, saved_mask = _non_paralyzable_dead_time(tot_ev_list, deadtime_values)
    del tot_ev_list

    ev_kind = ev_kind[saved_mask]
    deadtime_values = deadtime_values[saved_mask]
    final_len = tot_ev_list_filt.size
    if verbose:
        logger.info(
            "filter_for_deadtime: "
            "{0}/{1} events rejected".format(initial_len - final_len, initial_len)
        )

    retval = saved_mask[all_ev_kind]

    if return_all:
        # uf_events: source and background events together
        # ev_kind : kind of each event in uf_events.
        # bkg : Background events
        additional_output.uf_events = tot_ev_list_filt
        additional_output.is_event = ev_kind
        additional_output.deadtime = deadtime_values
        additional_output.bkg = tot_ev_list_filt[np.logical_not(ev_kind)]
        retval = [retval, additional_output]

    return retval


def filter_for_deadtime(event_list, deadtime, **kwargs):
    """Filter an event list for a given dead time.

    This function accepts either a list of times or a
    `stingray.events.EventList` object.

    For the complete optional parameter list, see `get_deadtime_mask`

    Parameters
    ----------
    ev_list : array-like or class:`stingray.events.EventList`
        The event list
    deadtime: float
        The (mean, if not constant) value of deadtime

    Returns
    -------
    new_ev_list : class:`stingray.events.EventList` object
        The filtered event list
    additional_output : dict
        See `get_deadtime_mask`

    """
    # Need to import here to avoid circular imports in the top module.
    from stingray.events import EventList

    local_retall = kwargs.pop("return_all", False)

    if isinstance(event_list, EventList):
        retval = event_list.apply_deadtime(deadtime, return_all=local_retall, **kwargs)
    else:
        mask, retall = get_deadtime_mask(event_list, deadtime, return_all=True, **kwargs)
        retval = event_list[mask]
        if local_retall:
            retval = [retval, retall]

    return retval

import numpy as np
import numpy.random as ra
import scipy.interpolate as sci
from stingray import Lightcurve
from stingray.gti import gti_border_bins


def simulate_times(lc, use_spline=False):
    """
    Simulate an event list, by using the inverse CDF method:

    + Assume that the light curve is a probability density (must be positive
      definite)

    + Calculate the CDF from the cumulative sum, and normalize it from 0 to 1

    + Extract N random probability values from 0 to 1

    + Find the CDF values corresponding to these N values

    + Find the times corresponding to these N CDF values

    Parameters
    ----------
    lc: `Lightcurve` object
        The `counts` array of the light curve should be give the expected
        number of photons in that bin. For this reason, please note that the
        light curve should not contain any negative values, or this method will
        raise an exception.


    Other Parameters
    ----------------
    use_spline : bool
        Approximate the light curve with a spline to avoid binning effects

    Returns
    -------
    times : array-like
        Simulated photon arrival times

    Examples
    --------
    >>> t = [0.5, 1.5, 3.5]
    >>> c = [100] * 3
    >>> lc = Lightcurve(t, c, gti=[[0, 2], [3, 4]], dt=1.)
    >>> times = simulate_times(lc, use_spline=True)
    >>> assert np.all(np.diff(times) > 0)  # Output array is sorted
    >>> assert np.all(times >= 0.)  # All times inside GTIs
    >>> assert np.all(times <= 4.)
    >>> assert np.any(times > 3.)
    >>> assert np.any(times < 2.)
    >>> assert not np.any((times > 2.) & (times < 3.))  # No times outside GTIs
    >>> lc.counts[0] = -3.
    >>> simulate_times(lc)  # Test with one negative value in the lc
    Traceback (most recent call last):
        ...
    ValueError: simulate_times can only work with...
    """
    return simulate_times_from_count_array(lc.time, lc.counts, lc.gti, lc.dt, use_spline=use_spline)


def simulate_times_from_count_array(time, counts, gti, dt, use_spline=False):
    """Simulate an event list, by using the inverse CDF method.

    + Assume that the light curve is a probability density (must be positive
      definite)

    + Calculate the CDF from the cumulative sum, and normalize it from 0 to 1

    + Extract N random probability values from 0 to 1

    + Find the CDF values corresponding to these N values

    + Find the times corresponding to these N CDF values

    Parameters
    ----------
    time: array-like
    counts: array-like
    gti: [[gti00, gti01], ..., [gtin0, gtin1]]
    dt: float

    Other Parameters
    ----------------
    use_spline : bool
        Approximate the light curve with a spline to avoid binning effects

    Returns
    -------
    times : array-like
        Simulated photon arrival times

    Examples
    --------
    >>> t = [0., 1., 2., 3., 5.]
    >>> c = [100] * 5
    >>> gti = [[-0.5, 3.5], [4.5, 5.5]]
    >>> times = simulate_times_from_count_array(t, c, gti, 1, use_spline=True)
    >>> assert np.all(np.diff(times) > 0)  # Output array is sorted
    >>> assert np.all(times >= -0.5)  # All times inside GTIs
    >>> assert np.all(times <= 5.5)
    >>> assert np.any(times > 4.5)
    >>> assert np.any(times < 3.5)
    >>> assert not np.any((times > 3.5) & (times < 4.5))  # No times outside GTIs
    >>> # test that it works with integer times (former bug)
    >>> times = simulate_times_from_count_array([0, 1, 2, 3, 5], c, gti, 1, use_spline=True)
    >>> c[0] = -3.
    >>> simulate_times_from_count_array(t, c, gti, 1)  # Test with one negative value in the lc
    Traceback (most recent call last):
        ...
    ValueError: simulate_times can only work with...
    """
    time = np.asanyarray(time)
    counts = np.asanyarray(counts).astype(float)
    gti = np.asanyarray(gti)
    kind = "linear"
    if use_spline and time.size > 2:
        kind = "cubic"

    if np.any(counts < 0):
        raise ValueError("simulate_times can only work with positive-definite light curves")

    if len(gti) > 1:  # Work GTI by GTI, to avoid the spillover of events
        all_events = []
        start_bins, stop_bins = gti_border_bins(gti, time, dt=dt)
        for i, (start, stop) in enumerate(zip(start_bins, stop_bins)):
            new_events = simulate_times_from_count_array(
                time[start:stop], counts[start:stop], [gti[i]], dt, use_spline=use_spline
            )
            all_events.append(new_events)
        return np.concatenate(all_events)

    n_events_predict = np.random.poisson(np.sum(counts))
    tmin = gti[0, 0]
    tmax = gti[-1, 1]
    times = simulate_with_inverse_cdf(
        counts, n_events_predict, sorted=True, x_range=[tmin, tmax], interp_kind=kind
    )

    return times


def simulate_with_inverse_cdf(
    binned_pdf, N, x_range=None, interp_kind="linear", sorted=False, edges=None
):
    """Simulate single values from a binned probability distribution.

    Parameters
    ----------
    binned_pdf : `np.array`
        The input "probability distribution". It does not need to be an actual
        probability distribution, but it can be a light curve, a spectrum, or
        any other histogram-like curve. It just needs to be positive definite!
    N : int
        The number of values to generate.

    Other parameters
    ----------------
    edges : `np.array`
        The edges of the PDF bins. The array is longer than binned_pdf by 1
        bin, a la `np.histogram`
    x_range : [float, float], default None
        The range of the values to be generated. Defaults to [0, 1].
    interp_kind : str
        Any valid interpolation kind accepted from `sci.interp1d`.
    sorted : bool, default False
        If true, sort the values.

    Raises
    ------
    ValueError
        If the input probability distribution has negative values

    Examples
    --------
    Let us simulate values between 0 and 1, with none between 0.25 and 0.5,
    sorted.
    >>> vals = simulate_with_inverse_cdf([2, 0, 4, 3], 103,
    ...                                  interp_kind="linear", sorted=True)

    103 values were simulated
    >>> vals.size
    103

    No values were simulated between 0.25 and 0.5
    >>> assert np.count_nonzero((vals > 0.25)&(vals < 0.5)) == 0

    All values are between 0 and 1
    >>> assert np.all((vals >= 0)&(vals < 1))

    Values are sorted
    >>> assert np.all(np.diff(vals)) >= 0

    We should get exactly the same result by passing the edges.
    >>> vals = simulate_with_inverse_cdf([2, 0, 4, 3], 103,
    ...                                  edges=[0, 0.25, 0.5, 0.75, 1],
    ...                                  interp_kind="linear", sorted=True)

    No values were simulated between 0.25 and 0.5
    >>> assert np.count_nonzero((vals > 0.25)&(vals < 0.5)) == 0

    All values are between 0 and 1
    >>> assert np.all((vals >= 0)&(vals < 1))

    Do not pass negative values in the binned PDF!
    >>> simulate_with_inverse_cdf([2, -1., 4], 10)
    Traceback (most recent call last):
        ...
    ValueError: simulate_with_inverse_cdf only works on...

    A single bin is interpreted as a uniform distribution.
    >>> vals = simulate_with_inverse_cdf([2], 14, sorted=True)

    14 values were simulated
    >>> vals.size
    14

    Values are sorted
    >>> assert np.all(np.diff(vals)) >= 0

    """
    binned_pdf = np.asanyarray(binned_pdf).astype(float)

    if x_range is None:
        x_range = [0, 1]

    if edges is None:
        edges = np.linspace(x_range[0], x_range[1], binned_pdf.size + 1)

    if np.any(binned_pdf < 0):
        raise ValueError(
            "simulate_with_inverse_cdf only works on positive-definite input " "curves"
        )

    if len(binned_pdf) == 1:  # Corner case: a single bin
        vals = np.random.uniform(x_range[0], x_range[1], N)
        if sorted:
            vals = np.sort(vals)
        return vals

    binned_pdf = np.concatenate([[0], binned_pdf])
    cdf = np.cumsum(binned_pdf)
    cdf /= cdf[-1]

    inv_cdf_func = sci.interp1d(cdf, edges, kind=interp_kind)

    cdf_vals = np.random.uniform(0, 1, N)
    if sorted:
        cdf_vals = np.sort(cdf_vals)

    return inv_cdf_func(cdf_vals)

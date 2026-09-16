import os
import numpy as np

import stingray.lightcurve as lightcurve


def sample_data():
    """
    Import data from .txt file and return a light curve object.

    Returns
    -------
    sample: :class:`Lightcurve` object
        The :class:`Lightcurve` object with the desired time stamps
        and counts.
    """

    lc_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets", "lc_sample.txt")
    data = np.loadtxt(lc_file)

    # Extract first and second columns to indicate dates and counts respectively
    dates = data[0 : len(data), 0]
    dt = dates[1] - dates[0]
    counts = data[0 : len(data), 1]

    # Return class:`Lightcurve` object
    return lightcurve.Lightcurve(dates, counts, dt=dt, skip_checks=True, err_dist="poisson")

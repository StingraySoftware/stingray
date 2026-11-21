import pickle
import numpy as np

__all__ = [
    "TransferFunction",
    "simple_ir",
    "relativistic_ir",
]

"""
Implementation of time and energy averaged responses from 2-d
transfer functions.
"""


class TransferFunction(object):
    """
    Create or retrieve a transfer function, and form
    time and energy averaged responses.

    Parameters
    ----------
    data : numpy 2-d array / 2-d list
        inner/first dimension (or column counts) represents time
        and outer/second dimension (or rows counts) represents energy.

        As an example, if you have you 2-d model defined by 'arr', then
        arr[1][5] defines a time of 5(s) and energy of 1(keV) [assuming
        'dt' and 'de' are 1 and 'tstart' and 'estart' are 0.]

        Note that each row is a different energy channel starting from
        the lowest to the highest.

    dt : float, default 1
        time interval

    de : float, default 1
        energy interval

    tstart : float, default 0
        initial time value across time axis

    estart : float, default 0
        initial energy value across energy axis

    Attributes
    ----------
    time : numpy.ndarray
        energy-averaged/time-resolved response of 2-d transfer
        function

    energy : numpy.ndarray
        time-averaged/energy-resolved response of 2-d transfer
        function
    """

    def __init__(self, data, dt=1, de=1, tstart=0, estart=0, time=None, energy=None):
        self.data = np.asanyarray(data)
        self.dt = dt
        self.de = de
        self.tstart = tstart
        self.estart = estart
        self.time = None
        self.energy = None

        if len(data[0]) < 2:
            raise ValueError("Number of columns should be greater than 1.")

        if len(data[:]) < 2:
            raise ValueError("Number of rows should be greater than 1.")

    def time_response(self, e0=None, e1=None):
        """
        Form an energy-averaged/time-resolved response of 2-d transfer
        function.

        Returns
        -------
        energy : numpy.ndarray
            energy-averaged/time-resolved response of 2-d transfer function

        e0: int
            start value of energy interval to be averaged

        e1: int
            end value of energy interval to be averaged
        """

        # Set start and stop values
        if e0 is None:
            start = 0
        else:
            start = int(self.estart + e0 / self.de)

        if e1 is None:
            stop = len(self.data[:][0]) - 1
        else:
            stop = int(self.estart + e1 / self.de)

        # Ensure start and stop values are legal
        if (start < 0) or (stop < 0):
            raise ValueError("e0 and e1 must be positive.")

        if (start > len(self.data[:][0])) or (stop > len(self.data[:][0])):
            raise ValueError("One or both energy values are out of range.")

        if start == stop:
            raise ValueError("e0 and e1 must be separated by at least de.")

        self.time = np.mean(self.data[start:stop, :], axis=0)

    def energy_response(self):
        """
        Form a time-averaged/energy-resolved response of 2-d transfer function.

        Returns
        -------
        time : numpy.ndarray
            time-averaged/energy-resolved response of 2-d transfer function
        """

        self.energy = np.mean(self.data, axis=1)

    def plot(self, response="2d", save=False, filename=None, show=False):
        """
        Plot 'time', 'energy' or 2-d response using matplotlib.

        In case of 1-d response, 'time' and 'energy' would appear
        along x-axis and corresponding flux across y-axis. In case
        of 2-d response, a spectrograph would be formed with 'time'
        along x-axis and 'energy' along y-axis.

        Parameters
        ----------
        response : str
            type of response - accepts 'time', 'energy', '2d'

        filename : str
            the name of file to save plot to. If a default of 'None' is
            picked, plot is not saved.
        """

        import matplotlib.pyplot as plt

        fig = plt.figure()

        if response == "time":
            t = np.linspace(self.tstart, len(self.data[0]) * self.dt, len(self.data[0]))
            figure = plt.plot(t, self.time)
            plt.xlabel("Time")
            plt.ylabel("Flux")
            plt.title("Time-resolved Response")

        elif response == "energy":
            e = np.linspace(self.estart, len(self.data[:]) * self.de, len(self.data[:]))
            figure = plt.plot(e, self.energy)
            plt.xlabel("Energy")
            plt.ylabel("Flux")
            plt.title("Energy-resolved Response")

        elif response == "2d":
            figure = plt.imshow(self.data, interpolation="nearest", cmap="Oranges", origin="lower")
            plt.xlabel("Time")
            plt.ylabel("Energy")
            plt.title("2-d Transfer Function")
            plt.colorbar()

        else:
            raise ValueError(
                "Response value is not recognized. Available"
                "response types are 'time', 'energy', and '2d'."
            )

        if save:
            if filename is None:
                plt.savefig("out.png")
            else:
                plt.savefig(filename)

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def read(filename, fmt="pickle"):
        """
        Reads transfer function from a 'pickle' file.

        Parameters
        ----------
        fmt : str
            the format of the file to be retrieved - accepts 'pickle'.

        Returns
        -------
        data : class instance
            :class:`~stingray.simulator.transfer.TransferFunction` object
        """

        if fmt == "pickle":
            with open(filename, "rb") as fobj:
                return pickle.load(fobj)

        else:
            raise KeyError("Format not understood.")

    def write(self, filename, fmt="pickle"):
        """
        Writes a transfer function to 'pickle' file.

        Parameters
        ----------
        fmt : str
            the format of the file to be saved - accepts 'pickle'
        """

        if fmt == "pickle":
            with open(filename, "wb") as fobj:
                pickle.dump(self, fobj)
        else:
            raise KeyError("Format not understood.")


"""
Implementation of artificial methods to create energy-averaged
responses for quick testing.
"""


def simple_ir(dt=0.125, start=0, width=1000, intensity=1):
    """
    Construct a simple impulse response using start time,
    width and scaling intensity.
    To create a delta impulse response, set width to ``dt``.

    The response is a box from ``start`` to ``start + width``. Each bin
    holds the response integrated over the bin, so that ``h.sum()``, the
    gain, is ``intensity`` at any ``dt``. A bin that the box only partly
    covers gets the covered fraction, so a ``start`` or ``width`` that is
    not a multiple of ``dt`` is spread over two bins and keeps the mean
    delay. A box that fits in one bin gives a single bin of ``intensity``.

    Parameters
    ----------
    dt : float
        Time resolution, in seconds

    start : float
        start time of impulse response, in seconds

    width : float
        width of impulse response, in seconds

    intensity : float
        scaling parameter to set the intensity of delayed emission
        corresponding to direct emission. It is the gain of the response.

    Returns
    -------
    h : numpy.ndarray
        Constructed impulse response
    """
    if start < 0 or width <= 0:
        raise ValueError("The impulse response needs start >= 0 and width > 0.")
    h = _integrate_over_bins(lambda t: t, dt, start, start + width)
    return intensity * h / h.sum()


def relativistic_ir(dt=0.125, t1=3, t2=4, t3=10, p1=1, p2=1.4, rise=4.8, decay=0.8, intensity=1):
    """
    Construct a realistic impulse response considering the relativistic
    effects.

    The response is a function of time: zero before ``t1``,
    ``p1 + (p2 - p1) * exp(rise * (t - t2))`` from ``t1`` to ``t2``, and
    ``p2 * exp(-decay * (t - t2))`` from ``t2`` to ``t3``. Each bin holds
    the response integrated over the bin, scaled so that ``h.sum()``, the
    gain, is ``intensity``. The shape is then the same at any ``dt``, and
    ``p1`` and ``p2`` only set its proportions.

    Parameters
    ----------
    dt : float
        Time resolution, in seconds

    t1 : float
        primary peak time, in seconds

    t2 : float
        secondary peak time, in seconds

    t3 : float
        end time, in seconds

    p1 : float
        value of primary peak

    p2 : float
        value of secondary peak

    rise : float
        rate of the rising exponential from primary peak to secondary peak,
        in 1 / s

    decay : float
        rate of the decaying exponential from secondary peak to end time,
        in 1 / s

    intensity : float
        the gain of the response, that is, the area under it. In line
        reverberation it is the responsivity, and in X-ray reverberation
        roughly the reflection fraction.

    Returns
    -------
    h: numpy.ndarray
        Constructed impulse response
    """

    assert t2 > t1, "Secondary peak must be after primary peak."
    assert t3 > t2, "End time must be after secondary peak."
    assert p2 > p1, "Secondary peak must be greater than primary peak."

    def rise_primitive(t):
        if rise == 0:
            return p2 * t
        return p1 * t + (p2 - p1) / rise * np.exp(rise * (t - t2))

    def decay_primitive(t):
        if decay == 0:
            return p2 * t
        return -p2 / decay * np.exp(-decay * (t - t2))

    h = _integrate_over_bins(decay_primitive, dt, t2, t3)
    h_rise = _integrate_over_bins(rise_primitive, dt, t1, t2)
    h[: h_rise.size] += h_rise
    return intensity * h / h.sum()


def _integrate_over_bins(primitive, dt, t_start, t_end):
    """
    Integrate a response that is zero outside ``[t_start, t_end]`` over
    each time bin, from time zero to the bin that contains ``t_end``.

    Parameters
    ----------
    primitive : callable
        An antiderivative of the response, taking times in seconds.
    dt : float
        Time resolution, in seconds.
    t_start, t_end : float
        Start and end time of the response.

    Returns
    -------
    h : numpy.ndarray
        The integral of the response over each bin.
    """
    # Bin units, snapped to a bin edge within rounding error of it, so that
    # no bin gets a spurious tiny weight
    x_start, x_end = (
        np.round(x) if abs(x - np.round(x)) < 1e-9 else x for x in (t_start / dt, t_end / dt)
    )
    edges = np.arange(int(np.ceil(x_end)) + 1)
    lower = np.clip(edges[:-1], x_start, x_end) * dt
    upper = np.clip(edges[1:], x_start, x_end) * dt
    return primitive(upper) - primitive(lower)

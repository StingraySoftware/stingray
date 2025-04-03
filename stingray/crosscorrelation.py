import warnings
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from stingray.utils import ifft, fftfreq

from stingray.lightcurve import Lightcurve
from stingray.crossspectrum import Crossspectrum, AveragedCrossspectrum
from stingray.exceptions import StingrayError
import stingray.utils as utils

__all__ = ["CrossCorrelation", "AutoCorrelation"]


class CrossCorrelation(object):
    r"""Make a cross-correlation from light curves or a cross spectrum.

    You can also make an empty :class:`Crosscorrelation` object to populate
    with your own cross-correlation data.

    Parameters
    ----------
    lc1: :class:`stingray.Lightcurve` object, optional, default ``None``
        The first light curve data for correlation calculations.

    lc2: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data for the correlation calculations.

    cross: :class: `stingray.Crossspectrum` object, default ``None``
        The cross spectrum data for the correlation calculations.

    mode: {``full``, ``valid``, ``same``}, optional, default ``same``
        A string indicating the size of the correlation output.
        See the relevant ``scipy`` documentation [scipy-docs]_
        for more details.

    norm: {``none``, ``variance``}
        if "variance", the cross correlation is normalized so that perfect
        correlation gives 1, and perfect anticorrelation gives -1. See
        Gaskell \& Peterson 1987, Gardner \& Done 2017

    Attributes
    ----------
    lc1: :class:`stingray.Lightcurve`
        The first light curve data for correlation calculations.

    lc2: :class:`stingray.Lightcurve`
        The light curve data for the correlation calculations.

    cross: :class: `stingray.Crossspectrum`
        The cross spectrum data for the correlation calculations.

    corr: numpy.ndarray
         An array of correlation data calculated from two light curves

    time_lags: numpy.ndarray
         An array of all possible time lags against which each point in corr is calculated

    dt: float
         The time resolution of each light curve (used in ``time_lag`` calculations)

    time_shift: float
         Time lag that gives maximum value of correlation between two light curves.
         There will be maximum correlation between light curves if one of the light curve
         is shifted by ``time_shift``.

    n: int
         Number of points in ``self.corr`` (length of cross-correlation data)

    auto: bool
        An internal flag to indicate whether this is a cross-correlation or an auto-correlation.

    norm: {``none``, ``variance``}
        The normalization specified in input

    References
    ----------
    .. [scipy-docs] https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.correlate.html
    """

    def __init__(self, lc1=None, lc2=None, cross=None, mode="same", norm="none"):
        self.auto = False
        self.norm = norm
        if isinstance(mode, str) is False:
            raise TypeError("mode must be a string")

        if mode.lower() not in ["full", "valid", "same"]:
            raise ValueError("mode must be 'full', 'valid' or 'same'!")

        self.mode = mode.lower()
        self.lc1 = None
        self.lc2 = None
        self.cross = None

        # Populate all attributes by ``None` if user passes no lightcurve data
        if lc1 is None or lc2 is None:
            if lc1 is not None or lc2 is not None:
                raise TypeError("You can't do a cross correlation with just one " "light curve!")

            else:
                if cross is None:
                    # all object input params are ``None``
                    self.corr = None
                    self.time_shift = None
                    self.time_lags = None
                    self.dt = None
                    self.n = None
                else:
                    self._make_cross_corr(cross)
                    return
        else:
            self._make_corr(lc1, lc2)

    def _make_cross_corr(self, cross):
        """
        Do some checks on the cross spectrum supplied to the method,
        and then calculate the time shifts, time lags and cross correlation.

        Parameters
        ----------
        cross: :class:`stingray.Crossspectrum` object
            The crossspectrum, averaged or not.

        """

        if not isinstance(cross, Crossspectrum):
            if not isinstance(cross, AveragedCrossspectrum):
                raise TypeError(
                    "cross must be a crossspectrum.Crossspectrum \
                        or crossspectrum.AveragedCrossspectrum object"
                )

        if self.cross is None:
            self.cross = cross
            self.dt = 1 / (cross.df * cross.n)
        if self.dt is None:
            self.dt = 1 / (cross.df * cross.n)

        prelim_corr = abs(ifft(cross.power).real)  # keep only the real
        self.n = len(prelim_corr)

        # ifft spits out an array that looks like [0,1,...n,-n,...-1]
        # where n is the last positive frequency
        # correcting for this by putting them in order

        times = fftfreq(self.n, cross.df)
        time, corr = np.array(sorted(zip(times, prelim_corr))).T
        self.corr = corr
        self.time_shift, self.time_lags, self.n = self.cal_timeshift(dt=self.dt)

    def _make_corr(self, lc1, lc2):
        """
        Do some checks on the light curves supplied to the method, and then calculate the time
        shifts, time lags and cross correlation.

        Parameters
        ----------
        lc1::class:`stingray.Lightcurve` object
            The first light curve data.

        lc2::class:`stingray.Lightcurve` object
            The second light curve data.

        """

        if not isinstance(lc1, Lightcurve):
            raise TypeError("lc1 must be a lightcurve.Lightcurve object")
        if not isinstance(lc2, Lightcurve):
            raise TypeError("lc2 must be a lightcurve.Lightcurve object")

        if not np.isclose(lc1.dt, lc2.dt):
            raise StingrayError("Light curves do not have " "same time binning dt.")
        else:
            # ignore very small differences in dt neglected by np.isclose()
            lc1.dt = lc2.dt
            self.dt = lc1.dt

        # self.lc1 and self.lc2 may get assigned values explicitly in which case there is no need to copy data
        if self.lc1 is None:
            self.lc1 = lc1
        if self.lc2 is None:
            self.lc2 = lc2

        # Subtract means before passing scipy.signal.correlate into correlation
        lc1_counts = self.lc1.counts - np.mean(self.lc1.counts)
        lc2_counts = self.lc2.counts - np.mean(self.lc2.counts)

        # Calculates cross-correlation of two lightcurves
        self.corr = signal.correlate(lc1_counts, lc2_counts, self.mode)

        self.n = np.size(self.corr)
        self.time_shift, self.time_lags, self.n = self.cal_timeshift(dt=self.dt)

        # Normalization that makes the maximum correlation equal to 1, and
        # maximum anticorrelation -1.
        if self.norm == "variance":
            # Note that self.corr is normalized so that the maximum is
            # proportional to the number of bins in the first input
            # light curve. Hence, the division by the lc size
            variance1 = np.var(lc1.counts) - np.var(lc1.counts_err)
            variance2 = np.var(lc2.counts) - np.var(lc2.counts_err)
            self.corr = self.corr / np.sqrt(variance1 * variance2) / lc1_counts.size

    def cal_timeshift(self, dt=1.0):
        """
        Calculate the cross correlation against all possible time lags, both positive and negative.

        The method signal.correlation_lags() uses SciPy versions >= 1.6.1 ([scipy-docs-lag]_)

        References
        ----------
        .. [scipy-docs-lag] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlation_lags.html

        Parameters
        ----------
        dt: float, optional, default ``1.0``
            Time resolution of the light curve, should be passed when object is populated with
            correlation data and no information about light curve can be extracted. Used to
            calculate ``time_lags``.

        Returns
        -------
        self.time_shift: float
             Value of the time lag that gives maximum value of correlation between two light curves.

        self.time_lags: numpy.ndarray
             An array of ``time_lags`` calculated from correlation data
        """
        if self.dt is None:
            self.dt = dt
        if self.corr is None:
            if (self.lc1 is None or self.lc2 is None) and (self.cross is None):
                raise StingrayError(
                    "Please provide either two lightcurve objects or \
                 a [average]crossspectrum object to calculate correlation and time_shift"
                )
            else:
                # This will cover very rare case of assigning self.lc1 and lc2
                # or self.cross and also self.corr = ``None``.
                # In this case, correlation is calculated using self.lc1
                # and self.lc2 and using that correlation data,
                # time_shift is calculated.
                if self.cross is not None:
                    self._make_cross_corr(self.cross)
                else:
                    self._make_corr(self.lc1, self.lc2)

        self.n = len(self.corr)
        n1 = n2 = self.n
        if self.lc1 is not None:
            n1 = np.size(self.lc1.counts)
        if self.lc2 is not None:
            n2 = np.size(self.lc2.counts)

        if self.cross is not None:
            # Obtains correlation lags if a cross spectrum object is given
            # Correlation against all possible lags, positive as well as negative lags are stored
            # signal.correlation_lags() method uses SciPy versions >= 1.6.1
            x_lags = signal.correlation_lags(self.n, self.n, self.mode)

        else:
            # Obtains correlation lags if two light curves are provided
            # Correlation against all possible lags, positive as well as negative lags are stored
            # signal.correlation_lags() method uses SciPy versions >= 1.6.1
            x_lags = signal.correlation_lags(n1, n2, self.mode)

        self.time_lags = x_lags * self.dt
        # time_shift is the time lag for max. correlation
        self.time_shift = self.time_lags[np.argmax(self.corr)]

        return self.time_shift, self.time_lags, self.n

    def plot(
        self, labels=None, axis=None, title=None, marker="-", save=False, filename=None, ax=None
    ):
        """
        Plot the :class:`Crosscorrelation` as function using Matplotlib.
        Plot the Crosscorrelation object on a graph ``self.time_lags`` on x-axis and
        ``self.corr`` on y-axis

        Parameters
        ----------
        labels : iterable, default ``None``
            A list of tuple with ``xlabel`` and ``ylabel`` as strings.

        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for ``matplotlib.pyplot.axis()`` function.

        title : str, default ``None``
            The title of the plot.

        marker : str, default ``-``
            Line style and color of the plot. Line styles and colors are
            combined in a single format string, as in ``'bo'`` for blue
            circles. See ``matplotlib.pyplot.plot`` for more options.

        save : boolean, optional (default=False)
            If True, save the figure with specified filename.

        filename : str
            File name of the image to save. Depends on the boolean ``save``.

        ax : ``matplotlib.Axes`` object
            An axes object to fill with the cross correlation plot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.plot(self.time_lags, self.corr, marker)
        if labels is not None:
            try:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
            except TypeError:
                utils.simon("``labels`` must be either a list or tuple with " "x and y labels.")
                raise
            except IndexError:
                utils.simon("``labels`` must have two labels for x and y " "axes.")
                # Not raising here because in case of len(labels)==1, only
                # x-axis will be labelled.

        # axis is a tuple containing formatting information
        if axis is not None:
            ax.axis(axis)

        if title is not None:
            ax.set_title(title)

        if save:
            if filename is None:
                plt.savefig("corr.pdf", format="pdf")
            else:
                plt.savefig(filename)
        else:
            plt.show(block=False)

        return ax


class AutoCorrelation(CrossCorrelation):
    """
    Make an auto-correlation from a light curve.
    You can also make an empty Autocorrelation object to populate with your
    own auto-correlation data.

    Parameters
    ----------
    lc: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data for correlation calculations.

    mode: {``full``, ``valid``, ``same``}, optional, default ``same``
        A string indicating the size of the correlation output.
        See the relevant ``scipy`` documentation [scipy-docs]
        for more details.

    Attributes
    ----------
    lc1, lc2::class:`stingray.Lightcurve`
        The light curve data for correlation calculations.

    corr: numpy.ndarray
         An array of correlation data calculated from lightcurve data

    time_lags: numpy.ndarray
         An array of all possible time lags against which each point in corr is calculated

    dt: float
         The time resolution of each lightcurve (used in time_lag calculations)

    time_shift: float, zero
         Max. Value of AutoCorrelation is always at zero lag.

    n: int
         Number of points in self.corr(Length of auto-correlation data)
    """

    def __init__(self, lc=None, mode="same"):
        CrossCorrelation.__init__(self, lc1=lc, lc2=lc, mode=mode)
        self.auto = True

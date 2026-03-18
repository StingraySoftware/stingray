import numpy as np
from scipy.linalg import toeplitz
import warnings
import matplotlib.pyplot as plt

from scipy.linalg import hankel

from stingray import lightcurve
import stingray.utils as utils
from stingray.utils import fftshift, fft2, ifftshift, fft

__all__ = ["Bispectrum"]


class Bispectrum(object):
    """Makes a :class:`Bispectrum` object from a :class:`stingray.Lightcurve`.

    :class:`Bispectrum` is a higher order time series analysis method and is calculated by
    indirect method as Fourier transform of triple auto-correlation function also called as
    3rd order cumulant.

    Parameters
    ----------
    lc : :class:`stingray.Lightcurve` object
        The light curve data for bispectrum calculation.

    maxlag : int, optional, default ``None``
        Maximum lag on both positive and negative sides of
        3rd order cumulant (Similar to lags in correlation).
        if ``None``, max lag is set to one-half of length of light curve.

    window : {``uniform``, ``parzen``, ``hamming``, ``hanning``, ``triangular``, ``welch``, ``blackman``, ``flat-top``}, optional, default 'uniform'
        Type of window function to apply to the data.

    scale : {``biased``, ``unbiased``}, optional, default ``biased``
        Flag to decide biased or unbiased normalization for 3rd order cumulant function.


    Attributes
    ----------
    lc : :class:`stingray.Lightcurve` object
        The light curve data to compute the :class:`Bispectrum`.

    fs : float
        Sampling frequencies

    n : int
        Total Number of samples of light curve observations.

    maxlag : int
        Maximum lag on both positive and negative sides of
        3rd order cumulant (similar to lags in correlation)

    signal : numpy.ndarray
        Row vector of light curve counts for matrix operations

    scale : {``biased``, ``unbiased``}
        Flag to decide biased or unbiased normalization for 3rd order cumulant function.

    lags : numpy.ndarray
        An array of time lags for which 3rd order cumulant is calculated

    freq : numpy.ndarray
        An array of freq values for :class:`Bispectrum`.

    cum3 : numpy.ndarray
        A ``maxlag*2+1 x maxlag*2+1`` matrix containing 3rd order cumulant data for different lags.

    bispec : numpy.ndarray
        A`` maxlag*2+1 x maxlag*2+1`` matrix containing bispectrum data for different frequencies.

    bispec_mag : numpy.ndarray
        Magnitude of the bispectrum

    bispec_phase : numpy.ndarray
        Phase of the bispectrum

    References
    ----------
    1) The biphase explained: understanding the asymmetries invcoupled Fourier components of astronomical timeseries
    by Thomas J. Maccarone Department of Physics, Box 41051, Science Building, Texas Tech University, Lubbock TX 79409-1051
    School of Physics and Astronomy, University of Southampton, SO16 4ES

    2) T. S. Rao, M. M. Gabr, An Introduction to Bispectral Analysis and Bilinear Time
    Series Models, Lecture Notes in Statistics, Volume 24, D. Brillinger, S. Fienberg,
    J. Gani, J. Hartigan, K. Krickeberg, Editors, Springer-Verlag, New York, NY, 1984.

    3) Matlab version of bispectrum under following link.
    https://www.mathworks.com/matlabcentral/fileexchange/60-bisp3cum

    Examples
    --------

    ::

       >> from stingray.lightcurve import Lightcurve
       >> from stingray.bispectrum import Bispectrum
       >> lc = Lightcurve([1,2,3,4,5],[2,3,1,1,2])
       >> bs = Bispectrum(lc,maxlag=1)
       >> bs.lags
       array([-1.,  0.,  1.])
       >> bs.freq
       array([-0.5,  0.,  0.5])
       >> bs.cum3
       array([[-0.2976,  0.1024,  0.1408],
           [ 0.1024,  0.144, -0.2976],
           [ 0.1408, -0.2976,  0.1024]])
       >> bs.bispec_mag
       array([[ 1.26336794,  0.0032   ,  0.0032    ],
           [ 0.0032   ,  0.16     ,  0.0032    ],
           [ 0.0032   ,  0.0032   ,  1.26336794]])
       >> bs.bispec_phase
       array([[ -9.65946229e-01,   2.25347190e-14,   3.46944695e-14],
           [  0.00000000e+00,   3.14159265e+00,   0.00000000e+00],
           [ -3.46944695e-14,  -2.25347190e-14,   9.65946229e-01]])
    """

    def __init__(self, lc, maxlag=None, window=None, scale="biased"):
        # Function call to create Bispectrum Object
        self._make_bispetrum(lc, maxlag, window, scale)

    def _make_bispetrum(self, lc, maxlag, window, scale):
        """
        Makes a Bispectrum Object with given lighcurve, maxlag and scale.

        Helper method.
        """

        if not isinstance(lc, lightcurve.Lightcurve):
            raise TypeError("lc must be a lightcurve.ightcurve object")

        # Available Windows. Used to resolve window paramneter
        WINDOWS = [
            "uniform",
            "parzen",
            "hamming",
            "hanning",
            "triangular",
            "welch",
            "blackmann",
            "flat-top",
        ]

        if window:
            if not isinstance(window, str):
                raise TypeError("Window must be specified as string!")
            window = window.lower()
            if window not in WINDOWS:
                raise ValueError("Wrong window specified or window function is not available")

        self.lc = lc
        self.fs = 1 / lc.dt
        self.n = self.lc.n

        if maxlag is None:
            # if maxlag is not specified, it is set to half of length of lightcurve
            self.maxlag = int(self.lc.n / 2)
        else:
            if not (isinstance(maxlag, int)):
                raise ValueError("maxlag must be an integer")

            # if negative maxlag is entered, convert it to +ve
            if maxlag < 0:
                self.maxlag = -maxlag
            else:
                self.maxlag = maxlag

        if isinstance(scale, str) is False:
            raise TypeError("scale must be a string")

        if scale.lower() not in ["biased", "unbiased"]:
            raise ValueError("scale can only be either 'biased' or 'unbiased'.")
        self.scale = scale.lower()

        if window is None:
            self.window_name = "No Window"
            self.window = None
        else:
            self.window_name = window
            self.window = self._get_window()

        # Other Attributes
        self.lags = None
        self.cum3 = None
        self.freq = None
        self.bispec = None
        self.bispec_mag = None
        self.bispec_phase = None

        # converting to a row vector to apply matrix operations
        self.signal = np.reshape(lc, (1, len(self.lc.counts)))

        # Mean subtraction before bispecrum calculation
        self.signal = self.signal - np.mean(lc.counts)

        self._cumulant3()
        self._normalize_cumulant3()
        self._cal_bispec()

    def _get_window(self):
        """
        Returns a window function of self.window_name type
        """
        N = 2 * self.maxlag + 1
        window_even = utils.create_window(N, self.window_name)

        # 2d even window
        window2d = np.array(
            [
                window_even,
            ]
            * N
        )

        ## One-sided window with zero padding
        window = np.zeros(N)
        window[: self.maxlag + 1] = window_even[self.maxlag :]
        window[self.maxlag :] = 0

        # 2d window function to apply to bispectrum
        row = np.concatenate(([window[0]], np.zeros(2 * self.maxlag)))
        toep_matrix = toeplitz(np.ravel(window), np.ravel(row))
        toep_matrix += np.tril(toep_matrix, -1).transpose()
        window = toep_matrix[..., ::-1] * window2d * window2d.transpose()
        return window

    def _cumulant3(self):
        """
        Calculates the 3rd Order cummulant of the lightcurve.

        Assigns
        -------
        self.cum3,
        self.lags
        """
        # Initialize square cumulant matrix if zeros
        cum3_dim = 2 * self.maxlag + 1
        self.cum3 = np.zeros((cum3_dim, cum3_dim))

        # calculate lags for different values of 3rd order cumulant
        lagindex = np.arange(-self.maxlag, self.maxlag + 1)
        self.lags = lagindex * self.lc.dt

        # Defines indices for matrices
        ind = np.arange((self.n - self.maxlag) - 1, self.n)
        ind_t = np.arange(self.maxlag, self.n)
        zero_maxlag = np.zeros((1, self.maxlag))
        zero_maxlag_t = zero_maxlag.transpose()

        sig = self.signal.transpose()

        rev_signal = np.array([self.signal[0][::-1]])
        col = np.concatenate((sig[ind], zero_maxlag_t), axis=0)
        row = np.concatenate((rev_signal[0][ind_t], zero_maxlag[0]), axis=0)

        # converts row and column into a toeplitz matrix
        toep = toeplitz(np.ravel(col), np.ravel(row))
        rev_signal = np.repeat(rev_signal, [2 * self.maxlag + 1], axis=0)

        # Calculates Cummulant of 1D signal i.e. Lightcurve counts
        self.cum3 = self.cum3 + np.matmul(np.multiply(toep, rev_signal), toep.transpose())

    def _normalize_cumulant3(self):
        """
        Scales (biased or ubiased) the 3rd Order cumulant of the lightcurve .

        Updates
        -------
        seff.cum3
        """

        # Biased scaling of cummulant
        if self.scale == "biased":
            self.cum3 = self.cum3 / self.n
        else:
            # unbiased Scaling of cummulant
            maxlag1 = self.maxlag + 1

            # Scaling matrix initialized used to do unbiased normalization of cumulant
            scal_matrix = np.zeros((maxlag1, maxlag1), dtype="int64")

            # Calculate scaling matrix for unbiased normalization
            for k in range(maxlag1):
                maxlag1k = maxlag1 - (k + 1)
                scal_matrix[k, k:maxlag1] = np.tile(self.n - maxlag1k, (1, maxlag1k + 1))
            scal_matrix += np.triu(scal_matrix, k=1).transpose()

            maxlag1ind = np.arange(self.maxlag - 1, -1, -1)
            lagdiff = self.n - maxlag1

            # Rows and columns for Toeplitz matrix
            col = np.arange(lagdiff, self.n - 1)
            col = np.reshape(col, (1, len(col))).transpose()
            row = np.arange(lagdiff, (self.n - 2 * self.maxlag) - 1, -1)
            row = np.reshape(row, (1, len(row)))

            # Toeplitz matrix
            toep_matrix = toeplitz(np.ravel(col), np.ravel(row))
            # Matrix used to concatenate with scaling matrix
            conc_mat = np.array([scal_matrix[self.maxlag, maxlag1ind]])
            join_matrix = np.concatenate((toep_matrix, conc_mat), axis=0)
            scal_matrix = np.concatenate((scal_matrix, join_matrix), axis=1)
            co_mat = scal_matrix[maxlag1ind, :]
            co_mat = co_mat[:, np.arange(2 * self.maxlag, -1, -1)]

            # Scaling matrix calculated
            scal_matrix = np.concatenate((scal_matrix, co_mat), axis=0)
            # Set numbers less than 1 to be equal to 1
            scal_matrix[scal_matrix < 1] = 1
            self.cum3 = np.divide(self.cum3, scal_matrix)

    def _cal_bispec(self):
        """
        Calculates bispectrum as a fourier transform of 3rd Order Cumulant.

        Attributes
        ----------
        self.freq
        self.bispec
        self.bispec_mag
        self.bispec_phase
        """
        self.freq = (1 / 2) * self.fs * (self.lags / self.lc.dt) / self.maxlag

        # Apply window if specified otherwise calculate with applying window
        if self.window is None:
            self.bispec = fftshift(fft2(ifftshift(self.cum3)))
        else:
            self.bispec = fftshift(fft2(ifftshift(self.cum3 * self.window)))

        self.bispec_mag = np.abs(self.bispec)
        self.bispec_phase = np.angle((self.bispec))

    def plot_cum3(self, axis=None, save=False, filename=None):
        """
        Plot the 3rd order cumulant as function of time lags using ``matplotlib``.
        Plot the ``cum3`` attribute on a graph with the ``lags`` attribute on x-axis and y-axis and
        ``cum3`` on z-axis

        Parameters
        ----------
        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for ``matplotlib.pyplot.axis()`` method.

        save : bool, optionalm, default ``False``
            If ``True``, save the figure with specified filename.

        filename : str
            File name and path of the image to save. Depends on the boolean ``save``.

        Returns
        -------
        plt : ``matplotlib.pyplot`` object
            Reference to plot, call ``show()`` to display it
        """
        cont = plt.contourf(self.lags, self.lags, self.cum3, 100, cmap=plt.cm.Spectral_r)
        plt.colorbar(cont)
        plt.title("3rd Order Cumulant")
        plt.xlabel("lags 1")
        plt.ylabel("lags 2")

        if axis is not None:
            plt.axis(axis)

        if save:
            if filename is None:
                plt.savefig("bispec_cum3.png")
            else:
                plt.savefig(filename)
        return plt

    def plot_mag(self, axis=None, save=False, filename=None):
        """
        Plot the magnitude of bispectrum as function of freq using ``matplotlib``.
        Plot the ``bispec_mag`` attribute on a graph with ``freq`` attribute on the x-axis and y-axis and
        the ``bispec_mag`` attribute on the z-axis.

        Parameters
        ----------
        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for ``matplotlib.pyplot.axis()`` method.

        save : bool, optional, default ``False``
            If ``True``, save the figure with specified filename and path.

        filename : str
            File name and path of the image to save. Depends on the bool ``save``.

        Returns
        -------
        plt : ``matplotlib.pyplot`` object
            Reference to plot, call ``show()`` to display it
        """

        cont = plt.contourf(self.freq, self.freq, self.bispec_mag, 100, cmap=plt.cm.Spectral_r)
        plt.colorbar(cont)
        plt.title("Bispectrum Magnitude")
        plt.xlabel("freq 1")
        plt.ylabel("freq 2")

        if axis is not None:
            plt.axis(axis)

        if save:
            if filename is None:
                plt.savefig("bispec_mag.png")
            else:
                plt.savefig(filename)
        return plt

    def plot_phase(self, axis=None, save=False, filename=None):
        """
        Plot the phase of bispectrum as function of freq using ``matplotlib``.
        Plot the ``bispec_phase`` attribute on a graph with ``phase`` attribute on the x-axis and
        y-axis and the ``bispec_phase`` attribute on the z-axis.

        Parameters
        ----------
        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for ``matplotlib.pyplot.axis()`` function.

        save : bool, optional, default ``False``
            If ``True``, save the figure with specified filename and path.

        filename : str
            File name and path of the image to save. Depends on the bool ``save``.

        Returns
        -------
        plt : ``matplotlib.pyplot`` object
            Reference to plot, call ``show()`` to display it
        """

        cont = plt.contourf(self.freq, self.freq, self.bispec_phase, 100, cmap=plt.cm.Spectral_r)
        plt.colorbar(cont)
        plt.title("Bispectrum Phase")
        plt.xlabel("freq 1")
        plt.ylabel("freq 2")

        if axis is not None:
            plt.axis(axis)

        # Save figure
        if save:
            if filename is None:
                plt.savefig("bispec_phase.png")
            else:
                plt.savefig(filename)
        return plt

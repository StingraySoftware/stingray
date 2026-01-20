import pickle
from os import error
import numpy as np
import numbers
import warnings
from scipy import signal
import astropy.modeling.models
from stingray import utils
from stingray import Lightcurve
from stingray import AveragedPowerspectrum

__all__ = ["Simulator"]


class Simulator(object):
    """
    Framework to simulate light curves with given variability distributions.
    The simulator module provides tools to simulate time series and spectral data. This can be useful, e.g.,
    to access the uncertainty of a previous analysis or to develop an intuition of the detectability of a given phenomenon

    Stingray simulator supports multiple methods to carry out these simulations.
    Light curves can be simulated through power-law spectrum, through a user-defined or pre-defined model, or through impulse responses.
    The module is designed in a way such that all these methods can be accessed using a common interface.

    Parameters
    ----------
    dt : float, default 1
        Time resolution (sampling interval) of the simulated light curve in seconds.
    N : int, default 1024
        Number of time bins in the simulated light curve.
    mean : float, default 0
        mean value of the simulated light curve.
    rms : float, default 1
        Fractional RMS amplitude of the light curve; actual RMS is `mean * rms`.
    err : float, default 0
        the errorbars on the final light curves.
    red_noise : int, default 1
        Factor by which to extend the light curve length to mitigate red noise leakage.
    random_state : int or numpy.random.RandomState, optional
        Seed or random state for reproducible random number generation.
    tstart : float, default 0
        Start time of the light curve in seconds.
    poisson : bool, default False
        If True, simulates Poisson-distributed counts; otherwise, assumes Gaussian noise.

    """

    def __init__(
        self, dt, N, mean, rms, err=0.0, red_noise=1, random_state=None, tstart=0.0, poisson=False
    ):
        self.dt = dt

        if not isinstance(N, (int, np.integer)):
            raise ValueError("N must be integer!")

        self.N = N

        if mean == 0:
            warnings.warn(
                "Careful! A mean of zero is unphysical!" + "This may have unintended consequences!"
            )
        self.mean = mean
        self.nphot = self.mean * self.N
        self.rms = rms
        self.red_noise = red_noise
        self.tstart = tstart
        self.time = dt * np.arange(N) + self.tstart
        self.nphot_factor = 1000_000
        self.err = err
        self.poisson = poisson

        # Initialize a tuple of energy ranges with corresponding light curves
        self.channels = []

        self.random_state = utils.get_random_state(random_state)

        assert rms <= 1, "Fractional rms must be less than 1."
        assert dt > 0, "Time resolution must be greater than 0"

    def simulate(self, *args):
        """
        Simulate light curve generation using power spectrum or
        impulse response.

        Examples
        --------
        * x = simulate(beta):
           For generating a light curve using power law spectrum.

              Parameters:
                * beta : float
                  Defines the shape of spectrum

        * x = simulate(s):
           For generating a light curve from user-provided spectrum.
            **Note**: In this case, the `red_noise` parameter is provided.
            You can generate a longer light curve by providing a higher
            frequency resolution on the input power spectrum.

              Parameters:
                * s : array-like
                  power spectrum

        * x = simulate(model):
           For generating a light curve from pre-defined model

              Parameters:
                * model : astropy.modeling.Model
                  the pre-defined model

        * x = simulate('model', params):
           For generating a light curve from pre-defined model

              Parameters:
                * model : string
                  the pre-defined model
                * params : list iterable or dict
                  the parameters for the pre-defined model

        * x = simulate(s, h):
           For generating a light curve using impulse response.

              Parameters:
                * s : array-like
                  Underlying variability signal
                * h : array-like
                  Impulse response

        * x = simulate(s, h, 'same'):
           For generating a light curve of same length as input signal,
           using impulse response.

              Parameters:
                * s : array-like
                  Underlying variability signal
                * h : array-like
                  Impulse response
                * mode : str
                  mode can be 'same', 'filtered, or 'full'.
                  'same' indicates that the length of output light
                  curve is same as that of input signal.
                  'filtered' means that length of output light curve
                  is len(s) - lag_delay
                  'full' indicates that the length of output light
                  curve is len(s) + len(h) -1

        Parameters
        ----------
        args
            See examples below.

        Returns
        -------
        lightCurve : `LightCurve` object

        """
        arg_len = len(args)

        if arg_len == 1:
            if isinstance(args[0], (numbers.Integral, float)):
                return self._simulate_power_law(args[0])
            
            elif isinstance(args[0], astropy.modeling.Model):
                return self._simulate_model(args[0])
            
            return self._simulate_power_spectrum(args[0])

        elif arg_len == 2:
            if utils.is_string(args[0]):
                return self._simulate_model_string(args[0], args[1])
            
            return self._simulate_impulse_response(args[0], args[1])

        elif arg_len == 3:
            return self._simulate_impulse_response(args[0], args[1], args[2])

        else:
            raise ValueError("Length of arguments must be 1, 2, or 3.")


    def simulate_channel(self, channel, *args):
        """
        Simulate a lightcurve and add it to corresponding energy
        channel.

        Parameters
        ----------
        channel : str
            range of energy channel (e.g., 3.5-4.5)

        *args
            see description of simulate() for details

        Returns
        -------
            lightCurve : `LightCurve` object
        """

        # Check that channel name does not already exist.
        if channel not in [lc[0] for lc in self.channels]:
            self.channels.append((channel, self.simulate(*args)))

        else:
            raise KeyError("A channel with this name already exists.")

    def get_channel(self, channel):
        """
        Get lightcurve belonging to the energy channel.
        """

        return [lc[1] for lc in self.channels if lc[0] == channel][0]

    def get_channels(self, channels):
        """
        Get multiple light curves belonging to the energy channels.
        """

        return [lc[1] for lc in self.channels if lc[0] in channels]

    def get_all_channels(self):
        """
        Get lightcurves belonging to all channels.
        """

        return [lc[1] for lc in self.channels]

    def delete_channel(self, channel):
        """
        Delete an energy channel.
        """

        channel = [lc for lc in self.channels if lc[0] == channel]

        if len(channel) == 0:
            raise KeyError("This channel does not exist or has already been " "deleted.")
        else:
            index = self.channels.index(channel[0])
            del self.channels[index]

    def delete_channels(self, channels):
        """
        Delete multiple energy channels.
        """
        n = len(channels)
        channels = [lc for lc in self.channels if lc[0] in channels]

        if len(channels) != n:
            raise KeyError(
                "One of more of the channels do not exist or have " "already been deleted."
            )
        else:
            indices = [self.channels.index(channel) for channel in channels]
            for i in sorted(indices, reverse=True):
                del self.channels[i]

    def count_channels(self):
        """
        Return total number of energy channels.
        """

        return len(self.channels)

    def simple_ir(self, start=0, width=1000, intensity=1):
        """
        Construct a simple impulse response using start time,
        width and scaling intensity.
        To create a delta impulse response, set width to 1.

        Parameters
        ----------
        start : int
            start time of impulse response
        width : int
            width of impulse response
        intensity : float
            scaling parameter to set the intensity of delayed emission
            corresponding to direct emission.

        Returns
        -------
        h : numpy.ndarray
            Constructed impulse response
        """

        # Fill in 0 entries until the start time
        h_zeros = np.zeros(int(start / self.dt))

        # Define constant impulse response
        h_ones = np.ones(int(width / self.dt)) * intensity

        return np.append(h_zeros, h_ones)

    def relativistic_ir(self, t1=3, t2=4, t3=10, p1=1, p2=1.4, rise=0.6, decay=0.1):
        """
        Construct a realistic impulse response considering the relativistic
        effects.

        Parameters
        ----------
        t1 : int
            primary peak time
        t2 : int
            secondary peak time
        t3 : int
            end time
        p1 : float
            value of primary peak
        p2 : float
            value of secondary peak
        rise : float
            slope of rising exponential from primary peak to secondary peak
        decay : float
            slope of decaying exponential from secondary peak to end time

        Returns
        -------
        h : numpy.ndarray
            Constructed impulse response
        """

        dt = self.dt

        assert t2 > t1, "Secondary peak must be after primary peak."
        assert t3 > t2, "End time must be after secondary peak."
        assert p2 > p1, "Secondary peak must be greater than primary peak."

        # Append zeros before start time
        h_primary = np.append(np.zeros(int(t1 / dt)), p1)

        # Create a rising exponential of user-provided slope
        x = np.linspace(t1 / dt, t2 / dt, int((t2 - t1) / dt))
        h_rise = np.exp(rise * x)

        # Evaluate a factor for scaling exponential
        factor = np.max(h_rise) / (p2 - p1)
        h_secondary = (h_rise / factor) + p1

        # Create a decaying exponential until the end time
        x = np.linspace(t2 / dt, t3 / dt, int((t3 - t2) / dt))
        h_decay = np.exp((-decay) * (x - 4 / dt))

        # Add the three responses
        h = np.append(h_primary, h_secondary)
        h = np.append(h, h_decay)

        return h

    def _find_inverse(self, real, imaginary):
        """
        Forms complex numbers corresponding to real and imaginary
        parts and finds inverse series.

        Parameters
        ----------
        real : numpy.ndarray
            Co-effients corresponding to real parts of complex numbers
        imaginary : numpy.ndarray
            Co-efficients correspondong to imaginary parts of complex
            numbers

        Returns
        -------
        ifft : numpy.ndarray
            Real inverse fourier transform of complex numbers
        """

        # Form complex numbers corresponding to each frequency
        f = [complex(r, i) for r, i in zip(real, imaginary)]

        f = np.hstack([self.mean * self.N * self.red_noise, f])

        # Obtain time series
        return np.fft.irfft(f, n=self.N * self.red_noise)

    def _timmerkoenig(self, pds_shape):
        """Straight application of T&K method to a PDS shape."""
        pds_size = pds_shape.size

        real = np.random.normal(size=pds_size) * np.sqrt(0.5 * pds_shape)
        imaginary = np.random.normal(size=pds_size) * np.sqrt(0.5 * pds_shape)
        imaginary[-1] = 0

        counts = self._find_inverse(real, imaginary)

        self.std = counts.std()

        rescaled_counts = self._extract_and_scale(counts)
        err = np.zeros_like(rescaled_counts)

        if self.poisson:
            bad = rescaled_counts < 0
            if np.any(bad):
                warnings.warn("Some bins of the light curve have counts < 0. Setting to 0")
                rescaled_counts[bad] = 0
            lc = Lightcurve(
                self.time,
                np.random.poisson(rescaled_counts),
                err_dist="poisson",
                dt=self.dt,
                skip_checks=True,
            )
            lc.smooth_counts = rescaled_counts
        else:
            lc = Lightcurve(
                self.time, rescaled_counts, err=err, err_dist="gauss", dt=self.dt, skip_checks=True
            )

        return lc

    def _simulate_power_law(self, B):
        """
        Generate LightCurve from a power law spectrum.

        Parameters
        ----------
        B : int
            Defines the shape of power law spectrum.

        Returns
        -------
        lightCurve : array-like
        """
        # Define frequencies at which to compute PSD
        w = np.fft.rfftfreq(self.red_noise * self.N, d=self.dt)[1:]

        pds_shape = np.power((1 / w), B)

        return self._timmerkoenig(pds_shape)

    def _simulate_power_spectrum(self, s):
        """
        Generate a light curve from user-provided spectrum.

        Parameters
        ----------
        s : array-like
            power spectrum

        Returns
        -------
        lightCurve : `LightCurve` object
        """
        # Cast spectrum as numpy array
        pds_shape = np.zeros(s.size * self.red_noise)
        pds_shape[: s.size] = s

        return self._timmerkoenig(pds_shape)

    def _simulate_model(self, model):
        """
        For generating a light curve from a pre-defined model

        Parameters
        ----------
        model : astropy.modeling.Model derived function
            the pre-defined model
            (library-based, available in astropy.modeling.models or
            custom-defined)

        Returns
        -------
        lightCurve : :class:`stingray.lightcurve.LightCurve` object
        """
        # Frequencies at which the PSD is to be computed
        # (only positive frequencies, since the signal is real)
        nbins = self.red_noise * self.N
        simfreq = np.fft.rfftfreq(nbins, d=self.dt)[1:]

        # Compute PSD from model
        simpsd = model(simfreq)

        return self._timmerkoenig(simpsd)

    def _simulate_model_string(self, model_str, params):
        """
        For generating a light curve from a pre-defined model

        Parameters
        ----------
        model_str : string
            name of the pre-defined model
        params : list or dictionary
            parameters of the pre-defined model

        Returns
        -------
        lightCurve : :class:`stingray.lightcurve.LightCurve` object
        """
        from . import models

        # Frequencies at which the PSD is to be computed
        # (only positive frequencies, since the signal is real)
        nbins = self.red_noise * self.N
        simfreq = np.fft.rfftfreq(nbins, d=self.dt)[1:]

        if model_str not in dir(models):
            raise ValueError("Model is not defined!")

        if isinstance(params, dict):
            model = eval("models." + model_str + "(**params)")
            # Compute PSD from model
            simpsd = model(simfreq)
        elif isinstance(params, list):
            simpsd = eval("models." + model_str + "(simfreq, params)")
        else:
            raise ValueError("Params should be list or dictionary!")

        return self._timmerkoenig(simpsd)

    def _simulate_impulse_response(self, s, h, mode="same"):
        """
        Generate LightCurve from impulse response. To get
        accurate results, binning intervals (dt) of variability
        signal 's' and impulse response 'h' must be equal.

        Parameters
        ----------
        s : array-like
            Underlying variability signal
        h : array-like
            Impulse response
        mode : str
            mode can be 'same', 'filtered, or 'full'.
            'same' indicates that the length of output light
            curve is same as that of input signal.
            'filtered' means that length of output light curve
            is len(s) - lag_delay
            'full' indicates that the length of output light
            curve is len(s) + len(h) -1

        Returns
        -------
        lightCurve : :class:`stingray.lightcurve.LightCurve` object
        """
        lc = signal.fftconvolve(s, h)

        if mode == "same":
            lc = lc[: -(len(h) - 1)]

        elif mode == "filtered":
            lc = lc[(len(h) - 1) : -(len(h) - 1)]

        time = self.dt * np.arange(0.5, len(lc)) + self.tstart
        err = np.zeros_like(time)
        return Lightcurve(time, lc, err_dist="gauss", dt=self.dt, err=err, skip_checks=True)

    def _extract_and_scale(self, long_lc):
        """
        i) Make a random cut and extract a light curve of required
        length.

        ii) Rescale light curve i) with zero mean and unit standard
        deviation, and ii) user provided mean and rms (fractional
        rms * mean)

        Parameters
        ----------
        long_lc : numpy.ndarray
            Simulated lightcurve of length 'N' times 'red_noise'

        Returns
        -------
        lc : numpy.ndarray
            Normalized and extracted lightcurve of length 'N'
        """
        if self.red_noise == 1:
            lc = long_lc
        else:
            # Make random cut and extract light curve of length 'N'
            extract = self.random_state.randint(self.N - 1, self.red_noise * self.N - self.N + 1)
            lc = np.take(long_lc, range(extract, extract + self.N))

        mean_lc = np.mean(lc)

        if self.mean == 0:
            return (lc - mean_lc) / self.std * self.rms
        else:
            return (lc - mean_lc) / self.std * self.mean * self.rms + self.mean

    def powerspectrum(self, lc, seg_size=None):
        """
        Make a powerspectrum of the simulated light curve.

        Parameters
        ----------
        lc : lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            The light curve data to be Fourier-transformed.

        Returns
        -------
        power : numpy.ndarray
            The array of normalized squared absolute values of Fourier
            amplitudes

        """
        if seg_size is None:
            seg_size = lc.tseg

        return AveragedPowerspectrum(lc, seg_size).power

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
            `TransferFunction` object
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

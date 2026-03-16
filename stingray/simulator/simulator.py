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
from stingray import AveragedCrossspectrum
from typing import Iterable, Tuple, Optional, Union, Callable, Literal

__all__ = ["Simulator", "CrossSpectrumSimulator"]


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
        if isinstance(args[0], (numbers.Integral, float)) and len(args) == 1:
            return self._simulate_power_law(args[0])

        elif isinstance(args[0], astropy.modeling.Model) and len(args) == 1:
            return self._simulate_model(args[0])

        elif utils.is_string(args[0]) and len(args) == 2:
            return self._simulate_model_string(args[0], args[1])

        elif len(args) == 1:
            return self._simulate_power_spectrum(args[0])

        elif len(args) == 2:
            return self._simulate_impulse_response(args[0], args[1])

        elif len(args) == 3:
            return self._simulate_impulse_response(args[0], args[1], args[2])

        else:
            raise ValueError("Length of arguments must be 1, 2 or 3.")

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

    def get_refftfreq(self) -> np.ndarray:
        """
        Calculate the positive frequencies for the real-valued FFT of a signal.

        This method computes the frequencies corresponding to the positive
        half of the discrete Fourier Transform (DFT) of a signal, based on
        the parameters of the red noise and the number of samples.

        Returns:
            np.ndarray: An array of positive frequency values corresponding
            to the real FFT of the signal.
        """

        return np.fft.rfftfreq(self.red_noise * self.N, d=self.dt)[1:]


class CrossSpectrumSimulator(Simulator):
    r"""Simulate pairs of correlated light curves with arbitrary coherence and
    phase lag distributions.

    Extends :class:`Simulator` with the correlated Timmer-Koenig method from
    Larner, Nowak, & Wilms (2026), which generates two light curves whose
    cross-spectral properties (coherence :math:`\gamma^2` and phase lag
    :math:`\phi` or cospectrum :math:`\mathrm{Re}[C]` and quadrature spectrum 
    :math:`\mathrm{Im}[C]`) match user-supplied models at every Fourier frequency.

    Parameters
    ----------
    dt : float
        Time resolution (sampling interval) of the simulated light curves in
        seconds.
    N : int
        Number of time bins in each simulated light curve.
    mean : float
        Mean count rate of the simulated light curves in counts/s.
    rms : float or tuple of float
        Fractional root-mean-square variability of the output light curves.
        Must satisfy ``rms <= 1``. If a tuple ``(rms1, rms2)`` is given,
        independent fractional RMS values are applied to the reference and
        dependent light curves, respectively.
    err : float, optional
        Constant error bar to attach to every time bin. Default is 0.
    red_noise : int, optional
        Oversampling factor used to mitigate red-noise leakage: the internal
        time series is generated at ``red_noise * N`` bins and a random
        segment of length ``N`` is extracted. Must be 1 for
        ``CrossSpectrumSimulator`` (values > 1 raise
        :exc:`NotImplementedError`). Default is 1.
    random_state : int or numpy.random.RandomState, optional
        Seed or random-state object for reproducible simulations.
        Default is ``None`` (unseeded).
    tstart : float, optional
        Start time of the output light curves in seconds. Default is 0.
    poisson : bool, optional
        If ``True``, draw final counts from a Poisson distribution so that
        the light curves contain non-negative integer counts. Default is
        ``False``.

    Attributes
    ----------
    dt : float
        Time resolution of the simulated light curves in seconds.
    N : int
        Number of time bins in each simulated light curve.
    mean : float
        Mean count rate in counts/s.
    rms : float or tuple of (float, float)
        Fractional RMS variability. Stored as a tuple when a per-band pair
        was supplied at construction time.
    red_noise : int
        Oversampling factor (always 1 for this class).
    tstart : float
        Start time of the output light curves in seconds.
    time : numpy.ndarray
        Array of time bin centres, ``dt * arange(N) + tstart``.
    err : float
        Constant error bar on each time bin.
    poisson : bool
        Whether Poisson noise is applied to the output counts.
    channels : list of tuple
        Energy-channel store inherited from :class:`Simulator`. Each entry is
        a ``(channel_name, Lightcurve)`` pair; managed via
        ``simulate_channel`` / ``get_channel`` / ``delete_channel``.
    random_state : numpy.random.RandomState
        Random-state object used for all stochastic draws.

    Methods
    -------
    CS_simulate(pds1, pds2[, params, lag, coh, cospec, quadspec, ...])
        Simulate a correlated pair of light curves from power-spectral and
        cross-spectral model inputs.
    crossspectrum(lc1, lc2[, seg_size, norm])
        Compute the averaged cross spectrum of two light curves (static
        method; can also be called on an instance).
    get_refftfreq()
        Return the positive FFT frequencies at which PSD and lag models are
        evaluated, of length ``N // 2``.
    simulate(\*args)
        Inherited single-band Timmer-Koenig simulator; the
        ``_extract_and_scale`` override ensures compatibility with the tuple
        ``rms`` attribute.

    Raises
    ------
    NotImplementedError
        If ``red_noise > 1`` is requested.
    AssertionError
        If any value in a tuple ``rms`` exceeds 1.

    See Also
    --------
    Simulator : Parent class providing single-band simulation and
        channel-management utilities.

    References
    ----------
    Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (under review)
    Timmer, J., & Koenig, M. 1995, A&A, 300, 707

    Examples
    --------
    Simulate two light curves with 80 % coherence and a 0.5 rad phase lag:

    >>> import numpy as np
    >>> from stingray.simulator import CrossSpectrumSimulator
    >>> sim = CrossSpectrumSimulator(N=1024, mean=100.0, dt=0.1, rms=0.1,
    ...                              random_state=42)
    >>> lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)
    >>> lc1.n == 1024
    True
    """

    def __init__(self, *args, **kwargs):
        # rms may be a tuple (rms1, rms2); parent only accepts a scalar.
        # Extract it, validate, then pass the max to satisfy the parent assertion.
        rms_tuple = None
        if "rms" in kwargs and isinstance(kwargs["rms"], tuple):
            rms_tuple = kwargs["rms"]
            rms1, rms2 = rms_tuple
            assert rms1 <= 1 and rms2 <= 1, "Fractional rms must be less than 1."
            kwargs = {**kwargs, "rms": max(rms1, rms2)}
        elif len(args) > 3 and isinstance(args[3], tuple):
            rms_tuple = args[3]
            rms1, rms2 = rms_tuple
            assert rms1 <= 1 and rms2 <= 1, "Fractional rms must be less than 1."
            args = args[:3] + (max(rms1, rms2),) + args[4:]

        super().__init__(*args, **kwargs)

        if rms_tuple is not None:
            self.rms = rms_tuple  # Restore original tuple after parent sets scalar

        if self.red_noise != 1:
            raise NotImplementedError("Red noise > 1 not implemented for cross spectral fitting")

    def CS_simulate(
        self,
        pds1: Union[str, float, astropy.modeling.Model, Callable[[Iterable], Iterable]],
        pds2: Optional[Union[str, float, astropy.modeling.Model, Callable[[Iterable], Iterable]]],
        params: Optional[Union[list, dict]] = None,
        lag: Optional[
            Union[str, float, astropy.modeling.Model, Callable[[Iterable], Iterable], Iterable]
        ] = None,
        coh: Optional[
            Union[str, float, astropy.modeling.Model, Callable[[Iterable], Iterable], Iterable]
        ] = None,
        cospec: Optional[Union[str, float, Callable[[Iterable], Iterable], Iterable]] = None,
        quadspec: Optional[Union[str, float, Callable[[Iterable], Iterable], Iterable]] = None,
        lag_params: Optional[Union[list, dict]] = None,
        coh_params: Optional[Union[list, dict]] = None,
        cospec_params: Optional[Union[list, dict]] = None,
        quadspec_params: Optional[Union[list, dict]] = None,
    ) -> Tuple[Lightcurve, Lightcurve]:
        r"""Simulate two LightCurves from a power spectrum with a specified
        phase lag and/or coherence distribution.

        Parameters
        ----------
        pds1 : str or float or astropy.modeling.Model or callable or array-like
            Shape of the power spectrum for the reference time series.
            If str, a model name from ``stingray.simulator.models``.
            If float, the index of a power-law power spectrum.
            If ``astropy.modeling.Model``, the model is evaluated at each frequency.
            If other callable, must have signature ``f(frequency) -> pds``.
            If array-like, must give the PSD at each frequency in ``self.get_refftfreq()``.
        pds2 : str or float or astropy.modeling.Model or callable or array-like, optional
            Shape of the power spectrum for the dependent time series.
            Accepts the same types as ``pds1``. If not given, defaults to ``pds1``.
        params : list or dict, optional
            Parameters for the predefined model when ``pds1`` or ``pds2`` is a string.
        lag : str or float or astropy.modeling.Model or callable or array-like, optional
            Phase lag spectrum in radians. If omitted, no phase lag is applied.
            If float, a constant value in :math:`[-\pi, \pi]`.
            If str, a model name from ``stingray.simulator.models``; supply
            parameters via ``lag_params``.
            If callable, must have signature ``f(frequency) -> lag``.
            If array-like, must give the lag at each frequency in ``self.get_refftfreq()``.
        coh : str or float or astropy.modeling.Model or callable or array-like, optional
            Coherence spectrum. If omitted, no coherence modification is applied.
            If float, a constant value in [0, 1].
            If str, a model name from ``stingray.simulator.models``; supply
            parameters via ``coh_params``.
            If callable, must have signature ``f(frequency) -> coherence``.
            If array-like, must give the coherence at each frequency in ``self.get_refftfreq()``.
        cospec : str or float or callable or array-like, optional
            Real part of the cross spectrum (co-spectrum). Cannot be specified
            together with ``coh`` or ``lag``.
            If float, constant across all frequencies.
            If str, a model name from ``stingray.simulator.models``; supply
            parameters via ``cospec_params``.
            If callable, must have signature ``f(frequency) -> cospec``.
            If array-like, must give the value at each frequency in ``self.get_refftfreq()``.
        quadspec : str or float or callable or array-like, optional
            Imaginary part of the cross spectrum (quadrature spectrum). Cannot be
            specified together with ``coh`` or ``lag``.
            If float, constant across all frequencies.
            If str, a model name from ``stingray.simulator.models``; supply
            parameters via ``quadspec_params``.
            If callable, must have signature ``f(frequency) -> quadspec``.
            If array-like, must give the value at each frequency in ``self.get_refftfreq()``.
        lag_params : list or dict, optional
            Parameters for the predefined model when ``lag`` is a string.
        coh_params : list or dict, optional
            Parameters for the predefined model when ``coh`` is a string.
        cospec_params : list or dict, optional
            Parameters for the predefined model when ``cospec`` is a string.
        quadspec_params : list or dict, optional
            Parameters for the predefined model when ``quadspec`` is a string.

        Returns
        -------
        lc1, lc2 : tuple of `~stingray.Lightcurve`
            Reference and dependent light curves, respectively.

        Raises
        ------
        ValueError
            If both ``cospec``/``quadspec`` and ``coh``/``lag`` are specified,
            or if a model string cannot be parsed.
        """

        if pds2 is None:
            pds2 = pds1

        use_coh = coh is not None
        use_lag = lag is not None
        use_cross_spectra = cospec is not None or quadspec is not None

        if use_cross_spectra and (use_coh or use_lag):
            raise ValueError(
                "Cannot specify both cospec/quadspec and coh/lag. " "Use one pair or the other."
            )

        w = self.get_refftfreq()

        # Inspect the input and generate...
        # pds_distribution 1
        if isinstance(pds1, (float, int)):
            pds_shape1 = self._make_powerlaw_pds(pds1, self.dt, self.red_noise * self.N)

        elif isinstance(pds1, str):
            from stingray.simulator import models

            if not hasattr(models, pds1):
                raise ValueError("Model string not defined")

            if isinstance(params, dict):
                model = getattr(models, pds1)(**params)
                pds_shape1 = model(w)
            elif isinstance(params, list):
                model_func = getattr(models, pds1)
                pds_shape1 = model_func(w, params)
            else:
                raise ValueError("Params should be list or dictionary!")

        elif callable(pds1):
            pds_shape1 = pds1(w)
        else:
            pds_shape1 = np.asarray(pds1)

        # pds distribution 2
        if isinstance(pds2, (float, int)):
            pds_shape2 = self._make_powerlaw_pds(pds2, self.dt, self.red_noise * self.N)

        elif isinstance(pds2, str):
            from stingray.simulator import models

            if not hasattr(models, pds2):
                raise ValueError("Model string not defined")

            if isinstance(params, dict):
                model = getattr(models, pds2)(**params)
                pds_shape2 = model(w)
            elif isinstance(params, list):
                model_func = getattr(models, pds2)
                pds_shape2 = model_func(w, params)
            else:
                raise ValueError("Params should be list or dictionary!")

        elif callable(pds2):
            pds_shape2 = pds2(w)
        else:
            pds_shape2 = np.asarray(pds2)

        # Parse cospec distribution
        cospec_shape = None
        if cospec is not None:
            if isinstance(cospec, (float, int)):
                cospec_shape = np.ones_like(w) * cospec
            elif isinstance(cospec, str):
                from stingray.simulator import models

                if not hasattr(models, cospec):
                    raise ValueError("Model string not defined")
                if isinstance(cospec_params, dict):
                    model = getattr(models, cospec)(**cospec_params)
                    cospec_shape = model(w)
                elif isinstance(cospec_params, list):
                    model_func = getattr(models, cospec)
                    cospec_shape = model_func(w, cospec_params)
                else:
                    raise ValueError("Params should be list or dictionary!")
            elif isinstance(cospec, Iterable):
                cospec_shape = cospec
            else:
                cospec_shape = cospec(w)

        # Parse quadspec distribution
        quadspec_shape = None
        if quadspec is not None:
            if isinstance(quadspec, (float, int)):
                quadspec_shape = np.ones_like(w) * quadspec
            elif isinstance(quadspec, str):
                from stingray.simulator import models

                if not hasattr(models, quadspec):
                    raise ValueError("Model string not defined")
                if isinstance(quadspec_params, dict):
                    model = getattr(models, quadspec)(**quadspec_params)
                    quadspec_shape = model(w)
                elif isinstance(quadspec_params, list):
                    model_func = getattr(models, quadspec)
                    quadspec_shape = model_func(w, quadspec_params)
                else:
                    raise ValueError("Params should be list or dictionary!")
            elif isinstance(quadspec, Iterable):
                quadspec_shape = quadspec
            else:
                quadspec_shape = quadspec(w)

        # If neither lag nor coh nor cross spectra specified, simulate two independent light curves
        if not use_lag and not use_coh and not use_cross_spectra:
            if params is not None:
                return (self.simulate(pds1, params), self.simulate(pds2, params))
            else:
                return (self.simulate(pds1), self.simulate(pds2))

        # Parse lag distribution
        lag_shape = None
        if use_lag:
            if isinstance(lag, (float, int)):
                lag_shape = np.ones_like(w) * lag
            elif isinstance(lag, str):
                from stingray.simulator import models

                if not hasattr(models, lag):
                    raise ValueError("Model string not defined")
                if isinstance(lag_params, dict):
                    model = getattr(models, lag)(**lag_params)
                    lag_shape = model(w)
                elif isinstance(lag_params, list):
                    model_func = getattr(models, lag)
                    lag_shape = model_func(w, lag_params)
                else:
                    raise ValueError("Params should be list or dictionary!")
            elif isinstance(lag, Iterable):
                lag_shape = lag
            else:
                lag_shape = lag(w)

        # Parse coh distribution
        coh_shape = None
        if use_coh:
            if isinstance(coh, (float, int)):
                coh_shape = np.ones_like(w) * coh
            elif isinstance(coh, str):
                from stingray.simulator import models

                if not hasattr(models, coh):
                    raise ValueError("Model string not defined")
                if isinstance(coh_params, dict):
                    model = getattr(models, coh)(**coh_params)
                    coh_shape = model(w)
                elif isinstance(coh_params, list):
                    model_func = getattr(models, coh)
                    coh_shape = model_func(w, coh_params)
                else:
                    raise ValueError("Params should be list or dictionary!")
            elif isinstance(coh, Iterable):
                coh_shape = coh
            else:
                coh_shape = coh(w)

        c1, c2 = self._correlated_timmerkoenig(
            P1=pds_shape1,
            P2=pds_shape2,
            output_length=self.N * self.dt,
            dt=self.dt,
            mean=self.mean,
            lag=lag_shape,
            gamma=coh_shape,
            red_noise=self.red_noise,
            rms=self.rms,
            poisson=self.poisson,
            cospec=cospec_shape,
            quadspec=quadspec_shape,
        )

        t = np.arange(len(c1)) * self.dt

        lc1 = Lightcurve(time=t, counts=c1, err=np.zeros_like(c1), dt=self.dt, skip_checks=True)
        lc2 = Lightcurve(time=t, counts=c2, err=np.zeros_like(c2), dt=self.dt, skip_checks=True)

        return (lc1, lc2)

    @staticmethod
    def crossspectrum(
        lc1: Lightcurve,
        lc2: Lightcurve,
        seg_size: Optional[float] = None,
        norm: Optional[Literal["frac", "abs", "leahy", "none"]] = "frac",
    ) -> np.ndarray:
        """
        Make a cross spectrum of the simulated light curves.

        Parameters
        ----------
        lc1 : `~stingray.Lightcurve` or iterable of `~stingray.Lightcurve`
            The reference light curve data to be Fourier-transformed.
        lc2 : `~stingray.Lightcurve` or iterable of `~stingray.Lightcurve`
            The dependent light curve data to be Fourier-transformed.
        seg_size : float, optional
            Segment size in seconds. Defaults to the full light curve length.
        norm : str, optional
            Normalization of the cross spectrum. One of ``'frac'``, ``'abs'``,
            ``'leahy'``, or ``'none'``. Default is ``'frac'``.

        Returns
        -------
        power : numpy.ndarray
            Array of complex cross-spectral powers.

        Notes
        -----
        ``lc1`` and ``lc2`` must have the same length.
        """

        # Following stingray convention by including this method

        if seg_size is None:
            seg_size = lc1.tseg

        return AveragedCrossspectrum(lc1, lc2, seg_size, silent=True, norm=norm).power

    def _compute_transfer_function(
        self,
        gamma2: Union[float, np.ndarray[float]],
        phi: Union[float, np.ndarray[float]],
        P_X: Union[float, np.ndarray[float]],
        P_Y: Union[float, np.ndarray[float]],
    ) -> Union[complex, np.ndarray[complex]]:
        r"""Compute the transfer function T from coherence and phase lag.

        Implements Equation 15 from Larner, Nowak, & Wilms (2026):

        .. math::

            T = \sqrt{\frac{P_Y \gamma^2}{P_X}} \, e^{i\phi}

        Parameters
        ----------
        gamma2 : float or numpy.ndarray
            Coherence squared, :math:`\gamma^2`. Range: [0, 1].
        phi : float or numpy.ndarray
            Phase lag in radians. Range: :math:`[-\pi, \pi]`.
        P_X : float or numpy.ndarray
            Power spectrum of the reference time series.
        P_Y : float or numpy.ndarray
            Power spectrum of the dependent time series.

        Returns
        -------
        T : complex or numpy.ndarray
            Complex transfer function relating the two time series in Fourier space.

        Notes
        -----
        The transfer function encodes both the coherence (in its magnitude) and
        the phase lag (in its argument). This formulation ensures proper
        normalization of the power spectra.

        References
        ----------
        Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (under review)
        """

        magnitude = np.sqrt(P_Y * gamma2 / P_X)
        return magnitude * np.exp(1j * phi)

    def _compute_normalization_constant(
        self,
        P_X: Union[float, np.ndarray[float]],
        P_Y: Union[float, np.ndarray[float]],
        T: Union[complex, np.ndarray[complex]],
    ) -> Union[float, np.ndarray[float]]:
        r"""Compute the normalization constant K for the incoherent component.

        Implements Equation 12 from Larner, Nowak, & Wilms (2026):

        .. math::

            K = \sqrt{\frac{P_Y - P_X |T|^2}{2}}

        Parameters
        ----------
        P_X : float or numpy.ndarray
            Power spectrum of the reference time series.
        P_Y : float or numpy.ndarray
            Power spectrum of the dependent time series.
        T : complex or numpy.ndarray
            Transfer function (from ``_compute_transfer_function``).

        Returns
        -------
        K : float or numpy.ndarray
            Normalization constant for the incoherent component.

        Notes
        -----
        This constant ensures that the dependent time series Y has the correct
        power spectrum :math:`P_Y`. The factor of 2 accounts for the variance of
        complex Gaussian random variables.

        References
        ----------
        Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (under review)
        """

        T_mag_squared = np.abs(T) ** 2

        return np.sqrt(np.maximum(P_Y - P_X * T_mag_squared, 0.0) / 2)

    def _cross_spectra_to_coh_lag(
        self,
        cospec: Union[float, np.ndarray],
        quadspec: Union[float, np.ndarray],
        P_X: Union[float, np.ndarray],
        P_Y: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        r"""Convert co-spectrum and quadrature spectrum to coherence and phase lag.

        Parameters
        ----------
        cospec : float or numpy.ndarray
            Real part of the cross spectrum, :math:`\mathrm{Re}[C]`.
        quadspec : float or numpy.ndarray
            Imaginary part of the cross spectrum, :math:`\mathrm{Im}[C]`.
        P_X : float or numpy.ndarray
            Power spectrum of the reference time series.
        P_Y : float or numpy.ndarray
            Power spectrum of the dependent time series.

        Returns
        -------
        gamma2 : float or numpy.ndarray
            Coherence squared,
            :math:`\gamma^2 = (\mathrm{Re}[C]^2 + \mathrm{Im}[C]^2) / (P_X P_Y)`.
        phi : float or numpy.ndarray
            Phase lag in radians,
            :math:`\phi = \arctan2(\mathrm{Im}[C],\, \mathrm{Re}[C])`.
        """

        gamma2 = (cospec**2 + quadspec**2) / (P_X * P_Y)
        phi = np.arctan2(quadspec, cospec)
        return gamma2, phi

    def _invert_fft(self, x: Iterable[complex], mean: float, nbins: int) -> np.ndarray:
        """Prepend the DC component and compute the inverse real FFT.

        Parameters
        ----------
        x : array-like of complex
            One-sided FFT coefficients at positive frequencies (excluding DC).
        mean : float
            Mean count rate; used to set the DC (zero-frequency) component.
        nbins : int
            Length of the output time series passed to ``numpy.fft.irfft``.

        Returns
        -------
        counts : numpy.ndarray
            Real-valued time series of length ``nbins``.
        """
        f = np.hstack([mean * nbins, x])

        return np.fft.irfft(f, n=nbins)

    def _extract_and_scale(
        self,
        long_lc: np.ndarray,
        N: Optional[int] = None,
        mean: Optional[float] = None,
        red_noise: Optional[int] = None,
        random_state=None,
        rms: Optional[float] = None,
    ):
        """
        i) Make a random cut and extract a light curve of required
        length.

        ii) Rescale light curve i) with zero mean and unit standard
        deviation, and ii) user provided mean and rms (fractional
        rms * mean)

        Overrides the parent method to accept explicit parameters
        instead of relying solely on instance attributes. This allows
        both light curves in a correlated pair to use the same random
        extraction cut (via the same ``random_state`` seed) while
        supporting per-band rms.

        Falls back to instance attributes when called via the parent's
        ``_timmerkoenig`` path (i.e. when extra arguments are omitted).

        Parameters
        ----------
        long_lc : numpy.ndarray
            Simulated lightcurve of length 'N' times 'red_noise'
        N : int, optional
            Number of bins in output lightcurve. Defaults to ``self.N``.
        mean : float, optional
            Mean count rate of the output lightcurve. Defaults to ``self.mean``.
        red_noise : int, optional
            Red noise oversampling factor. Defaults to ``self.red_noise``.
        random_state : int or numpy.random.RandomState, optional
            Seed or state for the random extraction cut. Defaults to
            ``self.random_state``.
        rms : float, optional
            Fractional RMS. Defaults to ``self.rms`` (scalar only).

        Returns
        -------
        lc : numpy.ndarray
            Normalized and extracted lightcurve of length 'N'
        """
        if N is None:
            N = self.N
        if mean is None:
            mean = self.mean
        if red_noise is None:
            red_noise = self.red_noise
        if rms is None:
            rms = self.rms if not isinstance(self.rms, tuple) else self.rms[0]

        random_state = utils.get_random_state(random_state)

        std = long_lc.std()

        if red_noise == 1:
            lc = long_lc
        else:
            extract = random_state.randint(0, red_noise * N - N + 1)
            lc = np.take(long_lc, range(extract, extract + N))

        mean_lc = np.mean(lc)

        if mean == 0:
            return (lc - mean_lc) / std * rms
        else:
            return (lc - mean_lc) / std * mean * rms + mean

    def _make_powerlaw_pds(self, index: float, dt: float, bins: int) -> np.ndarray:
        """Construct a power-law PDS from a given spectral index.

        Parameters
        ----------
        index : float
            Spectral index of the power law (negative slope in log-log space).
        dt : float
            Time resolution in seconds.
        bins : int
            Total number of time bins (used to compute FFT frequencies).

        Returns
        -------
        p : numpy.ndarray
            Power-law PDS evaluated at the positive FFT frequencies.
        """

        w = np.fft.rfftfreq(bins, d=dt)[1:]

        p = np.power((1 / w), index)

        return p

    def _correlated_timmerkoenig(
        self,
        P1: np.ndarray,
        output_length: float,
        dt: float,
        mean: Union[float, Tuple[float, float]],
        lag: Optional[Union[float, np.ndarray[float]]] = None,
        gamma: Optional[Union[float, np.ndarray[float]]] = None,
        red_noise: Optional[int] = 1,
        rms: Optional[Union[float, Tuple[float, float]]] = 0.1,
        P2: Optional[np.ndarray] = None,
        poisson: bool = False,
        cospec: Optional[Union[float, np.ndarray[float]]] = None,
        quadspec: Optional[Union[float, np.ndarray[float]]] = None,
    ) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        r"""Simulate two correlated time series with arbitrary coherence and phase lag.

        Implements the method from Larner, Nowak, & Wilms (2026) using Equations 9, 10,
        12, and 15. The reference time series is generated using the Timmer-Koenig method,
        and the dependent time series is constructed as a weighted sum of a coherent
        component (derived from the reference via a transfer function) and an incoherent
        component.

        Parameters
        ----------
        P1 : numpy.ndarray
            Power spectrum of the reference time series (P_X in paper notation).
        output_length : float
            Length of the output light curve in seconds.
        dt : float
            Time resolution of the output light curve in seconds.
        mean : float
            Mean count rate of the output light curves in counts/s.
        lag : float or numpy.ndarray, optional
            Phase lag spectrum in radians (:math:`\phi`). Range: :math:`[-\pi, \pi]`.
            If float, constant across all frequencies.
            If array, must have the same length as ``P1``. Default: 0.
        gamma : float or numpy.ndarray, optional
            Coherence squared spectrum (:math:`\gamma^2`). Range: [0, 1].
            If float, constant across all frequencies.
            If array, must have the same length as ``P1``. Default: 1 (perfect coherence).
        red_noise : int, optional
            Red noise oversampling factor. Default: 1.
        rms : float or tuple of float, optional
            Fractional RMS variability. If tuple ``(rms1, rms2)``, different values
            are used for the reference and dependent series. Default: 0.1.
        P2 : numpy.ndarray, optional
            Power spectrum of the dependent time series (P_Y). If ``None``, ``P2 = P1``.
            Must have the same shape as ``P1`` if provided.
        poisson : bool, optional
            If ``True``, draw final counts from a Poisson distribution. Default: ``False``.
        cospec : float or numpy.ndarray, optional
            Real part of the cross spectrum, :math:`\mathrm{Re}[C]`. Cannot be specified
            together with ``gamma``/``lag``. If float, constant across all frequencies.
        quadspec : float or numpy.ndarray, optional
            Imaginary part of the cross spectrum, :math:`\mathrm{Im}[C]`. Cannot be
            specified together with ``gamma``/``lag``. If float, constant across all
            frequencies.

        Returns
        -------
        counts1 : numpy.ndarray
            Reference time series.
        counts2 : numpy.ndarray
            Dependent time series, correlated with ``counts1``.

        Raises
        ------
        ValueError
            If ``P1`` and ``P2`` have different shapes, or if both
            ``cospec``/``quadspec`` and ``gamma``/``lag`` are specified.

        Notes
        -----
        The method follows these steps at each Fourier frequency:

        1. Reference transform [Eq. 9]:

           .. math:: X = \sqrt{P_X / 2} \, (A_r + i B_r)

        2. Transfer function [Eq. 15]:

           .. math:: T = \sqrt{\frac{P_Y \gamma^2}{P_X}} \, e^{i\phi}

        3. Normalization constant [Eq. 12]:

           .. math:: K = \sqrt{\frac{P_Y - P_X |T|^2}{2}}

        4. Dependent transform [Eq. 10]:

           .. math:: Y = K (H_r + i J_r) + T X

        where :math:`A_r, B_r, H_r, J_r` are independent standard normal random
        variables. When :math:`\gamma^2 = 1` the incoherent component vanishes and
        :math:`Y = T X`.

        References
        ----------
        Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (under review)
        Timmer, J., & Koenig, M. 1995, A&A, 300, 707
        """

        if P2 is None:
            P2 = P1
        elif P1.shape != P2.shape:
            raise ValueError("Both power spectra must have the same shape!")

        # Validate that cospec/quadspec and gamma/lag are not both specified
        use_cross_spectra = cospec is not None or quadspec is not None
        use_coh_lag = gamma is not None or lag is not None
        if use_cross_spectra and use_coh_lag:
            raise ValueError(
                "Cannot specify both cospec/quadspec and gamma/lag. " "Use one pair or the other."
            )

        # Convert cospec/quadspec to gamma and lag
        if use_cross_spectra:
            pds_size = P1.size
            if cospec is None:
                cospec = np.zeros(pds_size)
            elif isinstance(cospec, (float, int)):
                cospec = np.ones(pds_size) * cospec
            if quadspec is None:
                quadspec = np.zeros(pds_size)
            elif isinstance(quadspec, (float, int)):
                quadspec = np.ones(pds_size) * quadspec
            gamma, lag = self._cross_spectra_to_coh_lag(cospec, quadspec, P1, P2)

        pds_size = P1.size

        # Default: no phase lag
        if lag is None:
            lag = np.zeros(pds_size)
        elif isinstance(lag, (float, int)):
            lag = np.ones(pds_size) * lag

        # Default: perfect coherence
        if gamma is None:
            gamma = np.ones(pds_size)
        elif isinstance(gamma, (float, int)):
            gamma = np.ones(pds_size) * gamma

        randint = self.random_state.randint(low=0, high=10000)

        N = int(output_length / dt)
        long_N = int(output_length * red_noise / dt)

        # Generate random variables (one set per frequency)
        Ar = self.random_state.normal(size=pds_size)
        Br = self.random_state.normal(size=pds_size)
        Hr = self.random_state.normal(size=pds_size)
        Jr = self.random_state.normal(size=pds_size)

        # Equation 9: Reference Fourier transform
        # X = sqrt(P_X/2) * (A_r + i*B_r)
        X = np.sqrt(P1 / 2) * (Ar + 1j * Br)

        # Equation 15: Transfer function (includes phase lag)
        # T = sqrt(P_Y * γ² / P_X) * exp(i*φ)
        T = self._compute_transfer_function(gamma2=gamma, phi=lag, P_X=P1, P_Y=P2)

        # Equation 12: Normalization constant for incoherent component
        # K = sqrt((P_Y - P_X*|T|²) / 2)
        K = self._compute_normalization_constant(P_X=P1, P_Y=P2, T=T)

        # Equation 10: Dependent Fourier transform
        # Y = K*(H_r + i*J_r) + T*X
        Y = K * (Hr + 1j * Jr) + T * X

        # Handle separate RMS for each band if provided as tuple
        if isinstance(rms, tuple):
            rms1, rms2 = rms
        else:
            rms1 = rms2 = rms

        # Inverse FFT to get time series
        counts1 = self._invert_fft(X, mean=mean, nbins=long_N)
        counts1 = self._extract_and_scale(
            long_lc=counts1,
            N=N,
            red_noise=red_noise,
            rms=rms1,
            random_state=randint,
            mean=mean,
        )

        counts2 = self._invert_fft(Y, mean=mean, nbins=long_N)
        counts2 = self._extract_and_scale(
            long_lc=counts2,
            N=N,
            red_noise=red_noise,
            rms=rms2,
            random_state=randint,
            mean=mean,
        )

        if poisson:
            counts1 = self.random_state.poisson(counts1)
            counts2 = self.random_state.poisson(counts2)

        return (counts1, counts2)

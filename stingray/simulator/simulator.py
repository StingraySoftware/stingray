from __future__ import division, print_function, absolute_import
import numpy as np

from ..utils import simon
from ..lightcurve import Lightcurve
from ..powerspectrum import AveragedPowerspectrum
import stingray.simulator.models as models

class Simulator(object):

    def __init__(self, dt=1, N=1024, mean=0, rms=1, red_noise=1, seed=None):
        """
        Methods to simulate and visualize light curves.

        Parameters
        ----------
        dt: int, default 1
            time resolution of simulated light curve
        N: int, default 1024
            bins count of simulated light curve
        mean: float, default 0
            mean value of the simulated light curve
        rms: float, default 1
            fractional rms of the simulated light curve,
            actual rms is calculated by mean*rms
        red_noise: int, default 1
            multiple of real length of light curve, by
            which to simulate, to avoid red noise leakage
        seed: int, default None
            seed value for random processes
        """

        self.dt = dt
        self.N = N
        self.mean = mean
        self.rms = rms
        self.red_noise = red_noise
        self.time = dt*np.arange(N)
        self.lc = None

        if seed is not None:
            np.random.seed(seed)

        assert rms<=1, 'Fractional rms must be less than 1.'

    def simulate(self, *args):
        """
        Simulate light curve generation using power spectrum or
        impulse response.

        Examples
        --------
        - x = simulate(2)
            For generating a light curve using power law spectrum.

            Parameters
            ----------
            Beta: int
                Defines the shape of spectrum
            N: int
                Number of samples

            Returns
            -------
            lightCurve: `LightCurve` object

        - x = simulate(s)
            For generating a light curve from user-provided spectrum.

            Parameters
            ----------
            s: array-like
                power spectrum

            Returns
            -------
            lightCurve: `LightCurve` object

        - x = simulate('lorenzian', p)
            For generating a light curve from pre-defined model

            Parameters
            ----------
            model: str
                name of the pre-defined model

            p: list iterable
                model parameters. For details, see model definitions
                in model.py

            Returns
            -------
            lightCurve: `LightCurve` object

        - x = simulate(s,h)
            For generating a light curve using impulse response.

            Parameters
            ----------
            s: array-like
                Underlying variability signal
            h: array-like
                Impulse response

            Returns
            -------
            lightCurve: `LightCurve` object
        """

        if type(args[0]) == int:
            return  self._simulate_power_law(args[0])

        elif len(args) == 1:
            return self._simulate_power_spectrum(args[0])

        elif type(args[0]) == str:
            return self._simulate_model(args[0], args[1])

        elif len(args) == 2:
            return self._simulate_impulse_response(args[0], args[1])

        else:
            raise AssertionError("Length of arguments must be 1 or 2.")


    def _simulate_power_law(self, B):
        """
        Generate LightCurve from a power law spectrum.

        Parameters
        ----------
        B: int
            Defines the shape of power law spectrum.

        Returns
        -------
        lightCurve: array-like
        """

        # Define frequencies at which to compute PSD
        w = np.fft.rfftfreq(self.red_noise*self.N, d=self.dt)[1:]

        # Draw two set of 'N' guassian distributed numbers
        a1 = np.random.normal(size=len(w))
        a2 = np.random.normal(size=len(w))

        # Multiply by (1/w)^B to get real and imaginary parts
        real = a1 * np.power((1/w),B/2)
        imaginary = a2 * np.power((1/w),B/2)

        # Obtain time series
        long_lc = self._find_inverse(real, imaginary)
        self.lc = Lightcurve(self.time, self._extract_and_scale(long_lc))

        return self.lc

    def _simulate_power_spectrum(self, s):
        """
        Generate a light curve from user-provided spectrum.

        Parameters
        ----------
        s: array-like
            power spectrum

        Returns
        -------
        lightCurve: `LightCurve` object
        """

        # Cast spectrum as numpy array
        s = np.array(s)

        self.red_noise = 1

        # Draw two set of 'N' guassian distributed numbers
        a1 = np.random.normal(size=len(s))
        a2 = np.random.normal(size=len(s))

        lc = self._find_inverse(a1*s, a2*s)
        self.lc = Lightcurve(self.time, self._extract_and_scale(lc))

        return self.lc

    def _simulate_model(self, mod, p):
        """
        For generating a light curve from pre-defined model

        Parameters
        ----------
        mod: str
            name of the pre-defined model

        p: list iterable
            model parameters. For details, see model definitions
            in model.py

        Returns
        -------
        lightCurve: `LightCurve` object
        """

        # Define frequencies at which to compute PSD
        w = np.fft.rfftfreq(self.red_noise*self.N, d=self.dt)[1:]

        if mod in dir(models):
            mod = 'models.'+ mod
            s = eval(mod +'(w, p)')

            # Draw two set of 'N' guassian distributed numbers
            a1 = np.random.normal(size=len(s))
            a2 = np.random.normal(size=len(s))

            long_lc = self._find_inverse(a1*s, a2*s)
            self.lc = Lightcurve(self.time, self._extract_and_scale(long_lc))

            return self.lc

        else:
            simon('Model is not defined!')

    def _simulate_impulse_response(self, s, h):
        """
        Generate LightCurve from a power law spectrum.

        Parameters
        ----------
        s: array-like
                Underlying variability signal
        h: array-like
                Impulse response

        Returns
        -------
        lightCurve: array-like
        """
        pass

    def _find_inverse(self, real, imaginary):
        """
        Forms complex numbers corresponding to real and imaginary
        parts and finds inverse series.

        Parameters
        ----------
        real: numpy.ndarray
            Co-effients corresponding to real parts of complex numbers
        imaginary: numpy.ndarray
            Co-efficients correspondong to imaginary parts of complex
            numbers

        Returns
        -------
        ifft: numpy.ndarray
            Real inverse fourier transform of complex numbers
        """

        # Form complex numbers corresponding to each frequency
        f = [complex(r, i) for r,i in zip(real,imaginary)]

        f = np.hstack([self.mean, f])

        # Obtain real valued time series
        f_conj = np.conjugate(np.array(f))

        # Obtain time series
        return np.fft.irfft(f_conj, n=self.red_noise*self.N)

    def _extract_and_scale(self, long_lc):
        """
        i) Make a random cut and extract a light curve of required
        length.

        ii) Rescale light curve i) with zero mean and unit standard
        deviation, and ii) user provided mean and rms (fractional
        rms * mean)

        Parameters
        ----------
        long_lc: numpy.ndarray
            Simulated lightcurve of length 'N' times 'red_noise'

        Returns
        -------
        lc: numpy.ndarray
            Normalized and extracted lightcurve of length 'N'
        """
        if self.red_noise == 1:
            lc = long_lc
        else:
            # Make random cut and extract light curve of length 'N'
            extract = np.random.randint(self.N-1, self.red_noise*self.N - self.N+1)
            lc = np.take(long_lc, range(extract, extract + self.N))

        avg = np.mean(lc)
        std = np.std(lc)

        return (lc-avg)/std * self.mean * self.rms + self.mean

    def periodogram(self, lc, seg_size=None):
        """
        Make a periodogram of the simulated light curve.

        Parameters
        ----------
        lc: lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            The light curve data to be Fourier-transformed.

        Returns
        -------
        ps: numpy.ndarray
            The array of normalized squared absolute values of Fourier
            amplitudes

        """
        if seg_size is None:
            seg_size = self.lc.tseg

        return AveragedPowerspectrum(lc, seg_size).power

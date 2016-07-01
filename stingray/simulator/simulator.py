from __future__ import division, print_function, absolute_import
import numpy as np
from scipy import signal

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

        # Initialize a tuple of energy ranges with corresponding light curves
        self.channels = []

        if seed is not None:
            np.random.seed(seed)

        assert rms<=1, 'Fractional rms must be less than 1.'
        assert dt>0, 'Time resolution must be greater than 0'

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

        - x = simulate(s, h)
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

        - x = simulate(s, h, 'same')
            For generating a light curve of same length as input
            signal, using impulse response.

            Parameters
            ----------
            s: array-like
                Underlying variability signal
            h: array-like
                Impulse response
            mode: str
                mode can be either 'same' or 'full'.
                'same' indicates that the length of output light
                curve is same as that of input signal.
                'full' indicates that the length of output light
                curve is len(s) + len(h) -1

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

        elif len(args) == 3:
            return self._simulate_impulse_response(args[0], args[1], args[2])

        else:
            simon("Length of arguments must be 1, 2 or 3.")

    def simulate_channel(self, channel, *args):
        """
        Simulate a lightcurve and add it to corresponding energy 
        channel.

        Parameters
        ----------
        channel: str
            range of energy channel (e.g., 3.5-4.5)

        *args: 
            see description of simulate() for details

        Returns
        -------
            lightCurve: `LightCurve` object
        """

        # Add a warning here if channel name already exists

        self.channels.append((channel, self.simulate(args)))

    def get_channel(self, channel):
        """
        Get lightcurve belonging to the energy channel.
        """

        return [lc[1] for lc in self.channels if lc[0] == channel]

    def get_channels(self, channels):
        """
        Get multiple light curves belonging to the energy channels.
        """
        
        return [lc[1] for lc in self.channels if lc[0] in channels]

    def delete_channel(self, channel):
        """
        Delete an energy channel.
        """

        channel = [lc for lc in self.channels if lc[0] == channel]
        index = self.channels.index(channel[0])
        del self.channels[index]

    def delete_channels(self, channels):
        """
        Delete multiple energy channels.
        """

        channels = [lc for lc in self.channels if lc[0] in channels]
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
        start: int
            start time of impulse response
        width: int
            width of impulse response
        intensity: float
            scaling parameter to set the intensity of delayed emission
            corresponding to direct emission.

        Returns
        -------
        h: numpy.ndarray
            Constructed impulse response
        """

        # Fill in 0 entries until the start time
        h_zeros = np.zeros(start/self.dt)

        # Define constant impulse response
        h_ones = np.ones(width/self.dt) * intensity

        return np.append(h_zeros, h_ones)

    def relativistic_ir(self, t1=3, t2=4, t3=10, p1=1, p2=1.4, rise=0.6, decay=0.1):
        """
        Construct a realistic impulse response considering the relativistic
        effects.

        Parameters
        ----------
        t1: int
            primary peak time
        t2: int
            secondary peak time
        t3: int
            end time
        p1: float
            value of primary peak
        p2: float
            value of secondary peak
        rise: float
            slope of rising exponential from primary peak to secondary peak
        decay: float
            slope of decaying exponential from secondary peak to end time

        Returns
        -------
        h: numpy.ndarray
            Constructed impulse response
        """

        dt = self.dt

        assert t2>t1, 'Secondary peak must be after primary peak.'
        assert t3>t2, 'End time must be after secondary peak.'
        assert p2>p1, 'Secondary peak must be greater than primary peak.'

        # Append zeros before start time
        h_primary = np.append(np.zeros(int(t1/dt)), p1)

        # Create a rising exponential of user-provided slope
        x = np.linspace(t1/dt, t2/dt, (t2-t1)/dt)
        h_rise = np.exp(rise*x)
        
        # Evaluate a factor for scaling exponential
        factor = np.max(h_rise)/(p2-p1)
        h_secondary = (h_rise/factor) + p1

        # Create a decaying exponential until the end time
        x = np.linspace(t2/dt, t3/dt, (t3-t2)/dt)
        h_decay = (np.exp((-decay)*(x-4/dt))) 

        # Add the three responses
        h = np.append(h_primary, h_secondary)
        h = np.append(h, h_decay)

        return h
        
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
        lc = Lightcurve(self.time, self._extract_and_scale(long_lc))

        return lc

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
        lc = Lightcurve(self.time, self._extract_and_scale(lc))

        return lc

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
            lc = Lightcurve(self.time, self._extract_and_scale(long_lc))

            return lc
        
        else:
            simon('Model is not defined!')

    def _simulate_impulse_response(self, s, h, mode='full'):
        """
        Generate LightCurve from impulse response. To get
        accurate results, binning intervals (dt) of variability 
        signal 's' and impulse response 'h' must be equal.

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
        lc = signal.fftconvolve(s, h)
        if mode == 'same':
            lc = lc[:-(len(h) - 1)]

        time = self.dt * np.arange(len(lc))
        return Lightcurve(time, lc)
        
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
            seg_size = lc.tseg

        return AveragedPowerspectrum(lc, seg_size).ps

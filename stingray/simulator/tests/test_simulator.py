import numpy as np
import os
import warnings

from scipy.interpolate import interp1d
import pytest
import astropy.modeling.models
from stingray import Lightcurve, Crossspectrum, sampledata, Powerspectrum
from stingray.simulator import Simulator, CrossSpectrumSimulator
from stingray.simulator import models

_H5PY_INSTALLED = True

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False


class TestSimulator(object):
    @classmethod
    def setup_class(self):
        self.N = 1024
        self.mean = 0.5
        self.dt = 0.125
        self.rms = 1.0
        self.simulator = Simulator(N=self.N, mean=self.mean, dt=self.dt, rms=self.rms)
        self.simulator_odd = Simulator(N=self.N + 1, mean=self.mean, dt=self.dt, rms=self.rms)

    def calculate_lag(self, lc, h, delay):
        """
        Class method to calculate lag between two light curves.
        """
        s = lc.counts
        output = self.simulator.simulate(s, h, "same")[delay:]
        s = s[delay:]
        time = lc.time[delay:]
        output = output.counts

        lc1 = Lightcurve(time, s)
        lc2 = Lightcurve(time, output)
        cross = Crossspectrum(lc2, lc1)
        cross = cross.rebin(0.0075)

        return np.angle(cross.power) / (2 * np.pi * cross.freq)

    def test_simulate_with_seed(self):
        """
        Simulate with a random seed value.
        """
        self.simulator = Simulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, random_state=12
        )
        assert len(self.simulator.simulate(2).counts), self.N

    def test_simulate_with_tstart(self):
        """
        Simulate with a random seed value.
        """
        tstart = 10.0
        self.simulator = Simulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, tstart=tstart
        )
        assert self.simulator.time[0] == tstart

    def test_simulate_with_random_state(self):
        self.simulator = Simulator(
            N=self.N,
            mean=self.mean,
            dt=self.dt,
            rms=self.rms,
            random_state=np.random.RandomState(12),
        )

    def test_simulate_with_incorrect_arguments(self):
        with pytest.raises(ValueError):
            self.simulator.simulate(1, 2, 3, 4)

    def test_simulate_channel(self):
        """
        Simulate an energy channel.
        """
        self.simulator.simulate_channel("3.5-4.5", "generalized_lorentzian", [1, 2, 3, 4])
        self.simulator.delete_channel("3.5-4.5")

    def test_simulate_channel_odd(self):
        """
        Simulate an energy channel.
        """
        self.simulator_odd.simulate_channel("3.5-4.5", "generalized_lorentzian", [1, 2, 3, 4])
        self.simulator_odd.delete_channel("3.5-4.5")

    def test_incorrect_simulate_channel(self):
        """Test simulating a channel that already exists."""
        self.simulator.simulate_channel("3.5-4.5", 2)
        with pytest.raises(KeyError):
            self.simulator.simulate_channel("3.5-4.5", 2)
        self.simulator.delete_channel("3.5-4.5")

    def test_get_channel(self):
        """
        Retrieve an energy channel after it has been simulated.
        """
        self.simulator.simulate_channel("3.5-4.5", 2)
        lc = self.simulator.get_channel("3.5-4.5")
        self.simulator.delete_channel("3.5-4.5")

    def test_get_channels(self):
        """
        Retrieve multiple energy channel after it has been simulated.
        """
        self.simulator.simulate_channel("3.5-4.5", 2)
        self.simulator.simulate_channel("4.5-5.5", "smoothbknpo", [1, 2, 3, 4])
        lc = self.simulator.get_channels(["3.5-4.5", "4.5-5.5"])

        self.simulator.delete_channels(["3.5-4.5", "4.5-5.5"])

    def test_get_all_channels(self):
        """Retrieve all energy channels."""
        self.simulator.simulate_channel("3.5-4.5", 2)
        self.simulator.simulate_channel("4.5-5.5", 1)
        lc = self.simulator.get_all_channels()

        self.simulator.delete_channels(["3.5-4.5", "4.5-5.5"])

    def test_count_channels(self):
        """
        Count energy channels after they have been simulated.
        """
        self.simulator.simulate_channel("3.5-4.5", 2)
        self.simulator.simulate_channel("4.5-5.5", 1)

        assert self.simulator.count_channels() == 2
        self.simulator.delete_channels(["3.5-4.5", "4.5-5.5"])

    def test_delete_incorrect_channel(self):
        """
        Test if deleting incorrect channel raises a
        keyerror exception.
        """
        with pytest.raises(KeyError):
            self.simulator.delete_channel("3.5-4.5")

    def test_delete_incorrect_channels(self):
        """
        Test if deleting incorrect channels raises a
        keyerror exception.
        """
        with pytest.raises(KeyError):
            self.simulator.delete_channels(["3.5-4.5", "4.5-5.5"])

    def test_init_failure_with_noninteger_N(self):
        with pytest.raises(ValueError):
            simulator = Simulator(N=1024.5, mean=self.mean, rms=self.rms, dt=self.dt)

    def test_init_fails_if_arguments_missing(self):
        with pytest.raises(TypeError):
            simulator = Simulator()

    @pytest.mark.parametrize("model_kind", ["astropy", "array", "float"])
    def test_rms_and_mean(self, model_kind):
        np.random.seed(103442357)
        nbins = 8192
        dt = 1 / 128
        mean = 100
        rms = 0.2
        nsim = 128
        astropy_model = astropy.modeling.models.PowerLaw1D(alpha=2)
        if model_kind == "astropy":
            model = astropy_model
        elif model_kind == "array":
            freq_fine = np.fft.rfftfreq(nbins, d=dt)[1:]
            model = astropy_model(freq_fine)
        elif model_kind == "float":
            model = 2.0

        lc_all = [self.simulator.simulate(model) for i in range(nsim)]

        mean_all = np.mean([np.mean(lc.counts) for lc in lc_all])
        std_all = np.mean([np.std(lc.counts) for lc in lc_all])

        assert np.isclose(mean_all, self.mean, rtol=0.001)
        assert np.isclose(std_all / mean_all, self.rms, rtol=0.001)

        pds_all = [Powerspectrum(lc_all[i]) for i in range(nsim)]
        pds = pds_all[0]
        model_compare = (mc := astropy_model(pds.freq)) / (np.sum(mc) * pds.df) * rms**2

        ratios = [pds.power / model_compare for pds in pds_all]
        assert np.all([np.mean(rat) / (np.std(rat) * 3) < 1 for rat in ratios])

    def test_rms_zero_mean(self):
        nsim = 1000

        mean = 0.0
        with pytest.warns(UserWarning, match="Careful! A mean of zero is unphysical!"):
            sim = Simulator(dt=self.dt, N=self.N, rms=self.rms, mean=mean)
        lc_all = [sim.simulate(-2.0) for i in range(nsim)]

        mean_all = np.mean([np.mean(lc.counts) for lc in lc_all])
        std_all = np.mean([np.std(lc.counts) for lc in lc_all])

        assert np.isclose(mean_all, mean, rtol=0.1)
        assert np.isclose(std_all, self.rms, rtol=0.1)

    def test_simulate_powerlaw(self):
        """
        Simulate light curve from power law spectrum.
        """
        assert len(self.simulator.simulate(2).counts), 1024

    def test_simulate_powerlaw_odd(self):
        """
        Simulate light curve from power law spectrum.
        """
        assert len(self.simulator_odd.simulate(2).counts), 2039

    def test_compare_powerlaw(self):
        """
        Compare simulated power spectrum with actual one.
        """
        B, N, red_noise, dt = 2, 1024, 10, 1

        self.simulator = Simulator(N=N, dt=dt, mean=5, rms=1, red_noise=red_noise)
        lc = [self.simulator.simulate(B) for i in range(1, 30)]
        simulated = self.simulator.powerspectrum(lc, lc[0].tseg)

        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = np.power((1 / w), B / 2)[:-1]

        actual_prob = actual / float(sum(actual))
        simulated_prob = simulated / float(sum(simulated))

        assert np.all(np.abs(actual_prob - simulated_prob) < 3 * np.sqrt(actual_prob))

    def test_simulate_powerspectrum(self):
        """
        Simulate light curve from any power spectrum.
        """
        s = np.random.rand(1024)
        assert len(self.simulator.simulate(s)), self.N

    def test_simulate_model_pars_not_list_or_dict(self):
        """
        Simulate light curve using lorentzian model.
        """
        with pytest.raises(ValueError) as excinfo:
            self.simulator.simulate("generalized_lorentzian", 12345)
        assert "Params should be list or dictionary!" in str(excinfo.value)

    def test_simulate_lorentzian(self):
        """
        Simulate light curve using lorentzian model.
        """
        assert len(self.simulator.simulate("generalized_lorentzian", [1, 2, 3, 4])), 1024

    def test_simulate_lorentzian_odd(self):
        """
        Simulate light curve using lorentzian model.
        """
        assert len(self.simulator_odd.simulate("generalized_lorentzian", [1, 2, 3, 4])), 1024

    def test_compare_lorentzian(self):
        """
        Compare simulated lorentzian spectrum with original spectrum.
        """
        N, red_noise, dt = 1024, 10, 1

        self.simulator = Simulator(N=N, dt=dt, mean=0.1, rms=0.4, red_noise=red_noise)
        lc = [
            self.simulator.simulate("generalized_lorentzian", [0.3, 0.9, 0.6, 0.5])
            for i in range(1, 30)
        ]
        simulated = self.simulator.powerspectrum(lc, lc[0].tseg)

        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = models.generalized_lorentzian(w, [0.3, 0.9, 0.6, 0.5])[:-1]

        actual_prob = actual / float(sum(actual))
        simulated_prob = simulated / float(sum(simulated))

        assert np.all(np.abs(actual_prob - simulated_prob) < 3 * np.sqrt(actual_prob))

    def test_simulate_smoothbknpo(self):
        """
        Simulate light curve using smooth broken power law model.
        """
        assert len(self.simulator.simulate("smoothbknpo", [1, 2, 3, 4])), 1024

    def test_compare_smoothbknpo(self):
        """
        Compare simulated smooth broken power law spectrum with original
        spectrum.
        """
        N, red_noise, dt = 1024, 10, 1

        self.simulator = Simulator(N=N, dt=dt, mean=0.1, rms=0.7, red_noise=red_noise)
        lc = [self.simulator.simulate("smoothbknpo", [0.6, 0.2, 0.6, 0.5]) for i in range(1, 30)]

        simulated = self.simulator.powerspectrum(lc, lc[0].tseg)

        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = models.smoothbknpo(w, [0.6, 0.2, 0.6, 0.5])[:-1]

        actual_prob = actual / float(sum(actual))
        simulated_prob = simulated / float(sum(simulated))

        assert np.all(np.abs(actual_prob - simulated_prob) < 3 * np.sqrt(actual_prob))

    def test_simulate_GeneralizedLorentz1D_str(self):
        """
        Simulate a light curve using the GeneralizedLorentz1D model
        called as a string
        """
        assert len(
            self.simulator.simulate(
                "GeneralizedLorentz1D", {"x_0": 10, "fwhm": 1.0, "value": 10.0, "power_coeff": 2}
            )
        ), 1024

    def test_simulate_GeneralizedLorentz1D_odd_str(self):
        """
        Simulate a light curve using the GeneralizedLorentz1D model
        called as a string
        """
        assert len(
            self.simulator_odd.simulate(
                "GeneralizedLorentz1D", {"x_0": 10, "fwhm": 1.0, "value": 10.0, "power_coeff": 2}
            )
        ), 2039

    def test_simulate_GeneralizedLorentz1D(self):
        """
        Simulate a light curve using the GeneralizedLorentz1D model
        called as a astropy.modeling.Model class
        """
        mod = models.GeneralizedLorentz1D(x_0=10, fwhm=1.0, value=10.0, power_coeff=2)
        assert len(self.simulator.simulate(mod)), 1024

    def test_simulate_SmoothBrokenPowerLaw_str(self):
        """
        Simulate a light curve using SmoothBrokenPowerLaw model
        called as a string
        """
        assert len(
            self.simulator.simulate(
                "SmoothBrokenPowerLaw",
                {"norm": 1.0, "gamma_low": 1.0, "gamma_high": 2.0, "break_freq": 1.0},
            )
        ), 1024

    def test_simulate_SmoothBrokenPowerLaw(self):
        """
        Simulate a light curve using SmoothBrokenPowerLaw model
        called as a astropy.modeling.Model class
        """
        mod = models.SmoothBrokenPowerLaw(norm=1.0, gamma_low=1.0, gamma_high=2.0, break_freq=1.0)
        assert len(self.simulator.simulate(mod)), 1024

    def test_simulate_generic_model(self):
        """
        Simulate a light curve using a generic model
        called as a astropy.modeling.Model class
        """
        mod = astropy.modeling.models.Gaussian1D(amplitude=10.0, mean=1.0, stddev=2.0)
        assert len(self.simulator.simulate(mod)), 1024

    def test_simulate_generic_model_odd(self):
        """
        Simulate a light curve using a generic model
        called as a astropy.modeling.Model class
        """
        mod = astropy.modeling.models.Gaussian1D(amplitude=10.0, mean=1.0, stddev=2.0)
        assert len(self.simulator_odd.simulate(mod)), 2039

    @pytest.mark.parametrize("poisson", [True, False])
    def test_compare_composite(self, poisson):
        """
        Compare the PSD of a light curve simulated using a composite model
        (using SmoothBrokenPowerLaw plus GeneralizedLorentz1D)
        with the actual model
        """
        N = 50000
        dt = 0.01
        m = 30000.0

        self.simulator = Simulator(N=N, mean=m, dt=dt, rms=self.rms, poisson=poisson)
        smoothbknpo = models.SmoothBrokenPowerLaw(
            norm=1.0, gamma_low=1.0, gamma_high=2.0, break_freq=1.0
        )
        lorentzian = models.GeneralizedLorentz1D(x_0=10, fwhm=1.0, value=10.0, power_coeff=2.0)
        myModel = smoothbknpo + lorentzian

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc = [self.simulator.simulate(myModel) for i in range(1, 50)]

        simulated = self.simulator.powerspectrum(lc, lc[0].tseg)

        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = myModel(w)[:-1]

        actual_prob = actual / float(sum(actual))
        simulated_prob = simulated / float(sum(simulated))

        assert np.all(np.abs(actual_prob - simulated_prob) < 3 * np.sqrt(actual_prob))

    def test_simulate_wrong_model(self):
        """
        Simulate with a model that does not exist.
        """
        with pytest.raises(ValueError):
            self.simulator.simulate("unsupported", [0.6, 0.2, 0.6, 0.5])

    def test_construct_simple_ir(self):
        """
        Construct simple impulse response.
        """
        t0, w = 100, 500
        assert len(self.simulator.simple_ir(t0, w)) == (t0 + w) / self.simulator.dt

    def test_construct_simple_ir_odd(self):
        """
        Construct simple impulse response.
        """
        t0, w = 100, 500
        assert len(self.simulator_odd.simple_ir(t0, w)) == (t0 + w) / self.simulator.dt

    def test_construct_relativistic_ir(self):
        """
        Construct relativistic impulse response.
        """
        t1, t3 = 3, 10
        ir = self.simulator.relativistic_ir(t1=t1, t3=t3)
        assert np.allclose(ir[: int(t1 / self.simulator.dt)], 0)
        assert ir[int(t1 / self.simulator.dt)] == 1

    def test_construct_relativistic_ir_odd(self):
        """
        Construct relativistic impulse response.
        """
        t1, t3 = 3, 10
        ir = self.simulator_odd.relativistic_ir(t1=t1, t3=t3)
        assert np.allclose(ir[: int(t1 / self.simulator_odd.dt)], 0)
        assert ir[int(t1 / self.simulator_odd.dt)] == 1

    def test_simulate_simple_impulse(self):
        """
        Simulate light curve from simple impulse response.
        """
        lc = sampledata.sample_data()
        s = lc.counts
        h = self.simulator.simple_ir(10, 1, 1)
        _ = self.simulator.simulate(s, h)

    def test_simulate_simple_impulse_odd(self):
        """
        Simulate light curve from simple impulse response.
        """
        lc = sampledata.sample_data()
        s = lc.counts
        h = self.simulator_odd.simple_ir(10, 1, 1)
        _ = self.simulator_odd.simulate(s, h)

    def test_powerspectrum(self):
        """
        Create a power spectrum from light curve.
        """
        lc = self.simulator.simulate(2)
        self.simulator.powerspectrum(lc)

    def test_powerspectrum_odd(self):
        """
        Create a power spectrum from light curve.
        """
        lc = self.simulator_odd.simulate(2)
        self.simulator_odd.powerspectrum(lc)

    def test_simulate_relativistic_impulse(self):
        """
        Simulate light curve from relativistic impulse response.
        """
        lc = sampledata.sample_data()
        s = lc.counts

        h = self.simulator.relativistic_ir()
        output = self.simulator.simulate(s, h)

    def test_filtered_simulate(self):
        """
        Simulate light curve using 'filtered' mode.
        """
        lc = sampledata.sample_data()
        s = lc.counts

        h = self.simulator.simple_ir()
        output = self.simulator.simulate(s, h, "filtered")

    def test_filtered_simulate_odd(self):
        """
        Simulate light curve using 'filtered' mode.
        """
        lc = sampledata.sample_data()
        s = lc.counts

        h = self.simulator_odd.simple_ir()
        output = self.simulator_odd.simulate(s, h, "filtered")

    def test_simple_lag_spectrum(self):
        """
        Simulate light curve from simple impulse response and
        compute lag spectrum.
        """
        lc = sampledata.sample_data()
        h = self.simulator.simple_ir(start=14, width=1)
        delay = int(15 / lc.dt)

        lag = self.calculate_lag(lc, h, delay)
        bins = np.arange(lag.size)
        v_cutoff = 1.0 / (2 * 15.0)
        dist = (v_cutoff - 0.0075) / 0.0075
        spec_fun = interp1d(bins, lag)
        h_cutoff = spec_fun(dist)

        assert np.abs(15 - h_cutoff) < np.sqrt(15)

    def test_relativistic_lag_spectrum(self):
        """
        Simulate light curve from relativistic impulse response and
        compute lag spectrum.
        """
        lc = sampledata.sample_data()
        h = self.simulator.relativistic_ir(t1=3, t2=4, t3=10)
        delay = int(4 / lc.dt)

        lag = self.calculate_lag(lc, h, delay)
        v_cutoff = 1.0 / (2 * 4)
        h_cutoff = lag[int((v_cutoff - 0.0075) * 1 / 0.0075)]

        assert np.abs(4 - h_cutoff) < np.sqrt(4)

    def test_position_varying_channels(self):
        """
        Tests lags for multiple energy channels with each channel
        having same intensity and varying position.
        """
        lc = sampledata.sample_data()
        s = lc.counts
        h = []
        h.append(self.simulator.simple_ir(start=4, width=1))
        h.append(self.simulator.simple_ir(start=9, width=1))

        delays = [int(5 / lc.dt), int(10 / lc.dt)]

        outputs = []
        for i in h:
            lc2 = self.simulator.simulate(s, i)
            lc2 = lc2.shift(-lc2.time[0] + lc.time[0])
            outputs.append(lc2)

        with pytest.warns(UserWarning, match="Your lightcurves have different statistics"):
            cross = [Crossspectrum(lc2, lc).rebin(0.0075) for lc2 in outputs]
        lags = [np.angle(c.power) / (2 * np.pi * c.freq) for c in cross]

        v_cutoffs = [1.0 / (2.0 * 5), 1.0 / (2.0 * 10)]
        h_cutoffs = [lag[int((v - 0.0075) * 1 / 0.0075)] for lag, v in zip(lags, v_cutoffs)]

        assert np.abs(5 - h_cutoffs[0]) < np.sqrt(5)
        assert np.abs(10 - h_cutoffs[1]) < np.sqrt(10)

    def test_intensity_varying_channels(self):
        """
        Tests lags for multiple energy channels with each channel
        having same position and varying intensity.
        """
        lc = sampledata.sample_data()
        s = lc.counts
        h = []
        h.append(self.simulator.simple_ir(start=4, width=1, intensity=10))
        h.append(self.simulator.simple_ir(start=4, width=1, intensity=20))

        delay = int(5 / lc.dt)

        outputs = []
        for i in h:
            lc2 = self.simulator.simulate(s, i)
            lc2 = lc2.shift(-lc2.time[0] + lc.time[0])
            outputs.append(lc2)

        with pytest.warns(UserWarning, match="Your lightcurves have different statistics"):
            cross = [Crossspectrum(lc2, lc).rebin(0.0075) for lc2 in outputs]
        lags = [np.angle(c.power) / (2 * np.pi * c.freq) for c in cross]

        v_cutoff = 1.0 / (2.0 * 5)
        h_cutoffs = [lag[int((v_cutoff - 0.0075) * 1 / 0.0075)] for lag in lags]

        assert np.abs(5 - h_cutoffs[0]) < np.sqrt(5)
        assert np.abs(5 - h_cutoffs[1]) < np.sqrt(5)

    def test_io(self):
        sim = Simulator(N=self.N, dt=self.dt, rms=self.rms, mean=self.mean)
        sim.write("sim.pickle")
        sim = sim.read("sim.pickle")
        assert sim.N == self.N
        os.remove("sim.pickle")

    def test_io_with_unsupported_format(self):
        sim = Simulator(N=self.N, dt=self.dt, rms=self.rms, mean=self.mean)
        with pytest.raises(KeyError):
            sim.write("sim.hdf5", fmt="hdf5")
        sim.write("sim.pickle", fmt="pickle")
        with pytest.raises(KeyError):
            sim.read("sim.pickle", fmt="hdf5")
        os.remove("sim.pickle")


class TestCrossSpectrumSimulator(object):
    @classmethod
    def setup_class(cls):
        cls.N = 2048
        cls.mean = 100.0
        cls.dt = 0.1
        cls.rms = 0.1
        cls.sim = CrossSpectrumSimulator(
            N=cls.N, mean=cls.mean, dt=cls.dt, rms=cls.rms, random_state=42
        )

    # --- Construction and validation ---

    def test_simulate_channel_raises(self):
        """simulate_channel is disabled; users must use CS_simulate instead."""
        with pytest.raises(NotImplementedError):
            self.sim.simulate_channel("3-10keV", 2.0)

    def test_red_noise_gt1_raises(self):
        """CrossSpectrumSimulator does not support red_noise > 1."""
        with pytest.raises(NotImplementedError):
            CrossSpectrumSimulator(
                N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, red_noise=2
            )

    def test_rms_tuple_keyword(self):
        """rms tuple passed as keyword argument is stored correctly."""
        sim = CrossSpectrumSimulator(N=self.N, mean=self.mean, dt=self.dt, rms=(0.1, 0.2))
        assert sim.rms == (0.1, 0.2)

    def test_rms_tuple_positional(self):
        """rms tuple passed as positional argument is stored correctly."""
        sim = CrossSpectrumSimulator(self.dt, self.N, self.mean, (0.1, 0.2))
        assert sim.rms == (0.1, 0.2)

    def test_rms_tuple_invalid_raises(self):
        """rms tuple with any value > 1 is rejected."""
        with pytest.raises(AssertionError):
            CrossSpectrumSimulator(N=self.N, mean=self.mean, dt=self.dt, rms=(0.1, 1.5))

    # --- Return type and shape ---

    def test_cs_simulate_returns_tuple_of_lightcurves(self):
        """CS_simulate returns a 2-tuple of Lightcurve objects."""
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)
        assert isinstance(lc1, Lightcurve)
        assert isinstance(lc2, Lightcurve)

    def test_cs_simulate_output_length(self):
        """Both output light curves have length N."""
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)
        assert lc1.n == self.N
        assert lc2.n == self.N

    # --- Error handling ---

    def test_cs_simulate_lag_and_cospec_raises(self):
        """Combining lag with cospec raises ValueError."""
        with pytest.raises(ValueError):
            self.sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, cospec=0.5)

    def test_cs_simulate_coh_and_quadspec_raises(self):
        """Combining coh with quadspec raises ValueError."""
        with pytest.raises(ValueError):
            self.sim.CS_simulate(pds1=2.0, pds2=None, coh=0.5, quadspec=0.5)

    # --- Fallback path (no cross-spectral constraints) ---

    def test_cs_simulate_no_cross_terms_returns_two_lightcurves(self):
        """Without coh/lag/cospec/quadspec, two independent Lightcurves are returned."""
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None)
        assert isinstance(lc1, Lightcurve)
        assert isinstance(lc2, Lightcurve)
        assert lc1.n == self.N
        assert lc2.n == self.N

    # --- Input variety ---

    def test_cs_simulate_array_lag(self):
        """lag can be given as an array of per-frequency values."""
        lag_arr = np.ones_like(self.sim.get_refftfreq()) * 0.5
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, lag=lag_arr)
        assert lc1.n == self.N

    def test_cs_simulate_array_coh(self):
        """coh can be given as an array of per-frequency values."""
        coh_arr = np.ones_like(self.sim.get_refftfreq()) * 0.8
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, coh=coh_arr)
        assert lc1.n == self.N

    def test_cs_simulate_callable_pds(self):
        """pds1 can be a callable with signature f(frequency) -> pds."""
        pds_func = lambda f: np.power(1 / f, 2.0)
        lc1, lc2 = self.sim.CS_simulate(pds1=pds_func, pds2=None, lag=0.5, coh=1.0)
        assert lc1.n == self.N

    def test_cs_simulate_array_pds(self):
        """pds1 can be an array of PSD values at each rfftfreq."""
        pds_arr = np.power(1 / self.sim.get_refftfreq(), 2.0)
        lc1, lc2 = self.sim.CS_simulate(pds1=pds_arr, pds2=None, lag=0.5, coh=1.0)
        assert lc1.n == self.N

    def test_cs_simulate_different_pds(self):
        """pds1 and pds2 can be different power spectra."""
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=1.0, lag=0.5, coh=0.8)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_cospec_quadspec(self):
        """Cross spectrum can be specified directly via cospec and quadspec."""
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, cospec=0.5, quadspec=0.3)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_rms_tuple(self):
        """rms tuple applies different fractional RMS to each band."""
        sim = CrossSpectrumSimulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=(0.1, 0.2), random_state=42
        )
        lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)
        assert lc1.n == self.N
        assert lc2.n == self.N

    # --- Poisson ---

    def test_cs_simulate_poisson_nonnegative_integer_counts(self):
        """With poisson=True, both light curves have non-negative integer counts."""
        sim = CrossSpectrumSimulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, poisson=True, random_state=42
        )
        lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)
        assert np.all(lc1.counts >= 0)
        assert np.all(lc2.counts >= 0)
        assert np.all(lc1.counts == lc1.counts.astype(int))
        assert np.all(lc2.counts == lc2.counts.astype(int))

    # --- Reproducibility ---

    def test_cs_simulate_same_seed_reproducible(self):
        """Two simulators with the same random_state produce identical output."""
        sim1 = CrossSpectrumSimulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, random_state=42
        )
        lc1a, lc2a = sim1.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)

        sim2 = CrossSpectrumSimulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, random_state=42
        )
        lc1b, lc2b = sim2.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)

        np.testing.assert_array_equal(lc1a.counts, lc1b.counts)
        np.testing.assert_array_equal(lc2a.counts, lc2b.counts)

    def test_cs_simulate_different_seeds_differ(self):
        """Different random seeds produce different light curves."""
        sim1 = CrossSpectrumSimulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, random_state=42
        )
        lc1a, _ = sim1.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)

        sim2 = CrossSpectrumSimulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, random_state=99
        )
        lc1b, _ = sim2.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)

        assert not np.allclose(lc1a.counts, lc1b.counts)

    # --- crossspectrum static method ---

    def test_crossspectrum_static_method(self):
        """crossspectrum() can be called on the class without an instance."""
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=1.0)
        cs = CrossSpectrumSimulator.crossspectrum(lc1, lc2)
        assert isinstance(cs, np.ndarray)
        assert len(cs) == self.N // 2 - 1

    def test_crossspectrum_instance_method(self):
        """crossspectrum() can also be called on an instance."""
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=1.0)
        cs = self.sim.crossspectrum(lc1, lc2)
        assert isinstance(cs, np.ndarray)
        assert len(cs) == self.N // 2 - 1

    # --- Parent simulate still works via _extract_and_scale override ---

    def test_parent_simulate_still_works(self):
        """The overridden _extract_and_scale does not break the parent simulate()."""
        lc = self.sim.simulate(2.0)
        assert isinstance(lc, Lightcurve)
        assert lc.n == self.N

    # --- Statistical correctness ---

    def test_cs_simulate_mean_and_rms(self):
        """Simulated light curves have approximately the correct mean and fractional RMS."""
        nsim = 100
        sim = CrossSpectrumSimulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, random_state=42
        )
        means1, means2, rms1_list, rms2_list = [], [], [], []
        for _ in range(nsim):
            lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)
            means1.append(np.mean(lc1.counts))
            means2.append(np.mean(lc2.counts))
            rms1_list.append(np.std(lc1.counts) / np.mean(lc1.counts))
            rms2_list.append(np.std(lc2.counts) / np.mean(lc2.counts))

        assert np.isclose(np.mean(means1), self.mean, rtol=0.01)
        assert np.isclose(np.mean(rms1_list), self.rms, rtol=0.1)
        assert np.isclose(np.mean(means2), self.mean, rtol=0.01)
        assert np.isclose(np.mean(rms2_list), self.rms, rtol=0.1)

    def test_cs_simulate_phase_lag_recovery(self):
        r"""With perfect coherence (:math:`\gamma^2 = 1`), the cross-spectral phase
        equals the input lag exactly at every Fourier frequency for a single realisation.

        Derivation: with :math:`\gamma^2 = 1`, :math:`Y(f) = T(f) X(f)` where
        :math:`T(f) = \sqrt{P_2/P_1} \, e^{i\phi}`.
        The mean-subtraction and rescaling in ``_extract_and_scale`` divide each series
        by a real scalar, so the phase of :math:`\overline{X} Y = |X|^2 T` is
        preserved as :math:`\phi`.
        """
        target_lag = 0.5  # radians
        sim = CrossSpectrumSimulator(
            N=self.N, mean=500.0, dt=self.dt, rms=0.3, random_state=42
        )
        lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, lag=target_lag, coh=1.0)

        x = lc1.counts - lc1.counts.mean()
        y = lc2.counts - lc2.counts.mean()
        # Exclude DC (index 0) and Nyquist (index -1) bins
        X = np.fft.rfft(x)[1:-1]
        Y = np.fft.rfft(y)[1:-1]

        phases = np.angle(np.conj(X) * Y)
        np.testing.assert_allclose(phases, target_lag, atol=1e-6)

    def test_cs_simulate_coherence_recovery(self):
        r"""Average coherence over many realisations converges to the target value.

        The coherence estimator

        .. math::

            \hat{\gamma}^2(f) = \frac{|\langle C(f) \rangle|^2}
                                      {\langle P_1(f) \rangle \langle P_2(f) \rangle}

        (averages over independent realisations) converges to :math:`\gamma^2` as
        :math:`N_\mathrm{sim} \to \infty`.
        """
        target_coh = 0.6
        N = 1024
        nsim = 300
        sim = CrossSpectrumSimulator(N=N, mean=100.0, dt=self.dt, rms=self.rms, random_state=0)

        nfreq = N // 2  # length of rfft(x)[1:]
        sum_Cxy = np.zeros(nfreq, dtype=complex)
        sum_Pxx = np.zeros(nfreq)
        sum_Pyy = np.zeros(nfreq)

        for _ in range(nsim):
            lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, lag=0.0, coh=target_coh)
            x = lc1.counts - lc1.counts.mean()
            y = lc2.counts - lc2.counts.mean()
            X = np.fft.rfft(x)[1:]
            Y = np.fft.rfft(y)[1:]
            sum_Cxy += np.conj(X) * Y
            sum_Pxx += np.abs(X) ** 2
            sum_Pyy += np.abs(Y) ** 2

        coh_est = np.abs(sum_Cxy) ** 2 / (sum_Pxx * sum_Pyy)
        assert np.abs(np.mean(coh_est) - target_coh) < 0.05

    def test_cs_simulate_zero_coherence_uncorrelated(self):
        r"""With coh=0 the two light curves are statistically uncorrelated.

        The average estimated coherence (computed over many independent realisations)
        should converge to zero, because the incoherent component :math:`K(H + iJ)`
        carries all of the power and is drawn from a fully independent random draw.
        """
        nsim = 200
        N = 1024
        sim = CrossSpectrumSimulator(N=N, mean=100.0, dt=self.dt, rms=0.2, random_state=0)

        nfreq = N // 2
        sum_Cxy = np.zeros(nfreq, dtype=complex)
        sum_Pxx = np.zeros(nfreq)
        sum_Pyy = np.zeros(nfreq)

        for _ in range(nsim):
            lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, coh=0.0)
            x = lc1.counts - lc1.counts.mean()
            y = lc2.counts - lc2.counts.mean()
            X = np.fft.rfft(x)[1:]
            Y = np.fft.rfft(y)[1:]
            sum_Cxy += np.conj(X) * Y
            sum_Pxx += np.abs(X) ** 2
            sum_Pyy += np.abs(Y) ** 2

        coh_est = np.abs(sum_Cxy) ** 2 / (sum_Pxx * sum_Pyy)
        assert np.mean(coh_est) < 0.05

    def test_cs_simulate_perfect_coherence_same_pds_identical(self):
        r"""With coh=1, lag=0, and pds1=pds2 the two light curves are numerically identical.

        Derivation: :math:`T = \sqrt{P_2 \gamma^2 / P_1} \, e^{i\phi} = 1` and
        :math:`K = \sqrt{(P_2 - P_1 |T|^2) / 2} = 0`, so :math:`Y = X` exactly in
        Fourier space. Both series share the same extraction cut and rescaling
        parameters, so counts1 == counts2 to machine precision.
        """
        sim = CrossSpectrumSimulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=self.rms, random_state=42
        )
        lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, lag=0.0, coh=1.0)
        np.testing.assert_array_equal(lc1.counts, lc2.counts)

    def test_cs_simulate_zero_lag_zero_phase_different_pds(self):
        r"""With coh=1, lag=0, and pds1 != pds2 the cross-spectrum phase is identically zero.

        With :math:`\gamma^2 = 1` and :math:`\phi = 0`:
        :math:`T = \sqrt{P_2 / P_1}` (real positive at every bin).
        Then :math:`\overline{X} Y = T |X|^2` is real positive, so angle = 0 exactly.
        Using pds2 != pds1 ensures the light curves are not identical (non-trivial test).
        """
        sim = CrossSpectrumSimulator(
            N=self.N, mean=500.0, dt=self.dt, rms=0.3, random_state=42
        )
        lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=1.0, lag=0.0, coh=1.0)
        x = lc1.counts - lc1.counts.mean()
        y = lc2.counts - lc2.counts.mean()
        X = np.fft.rfft(x)[1:-1]
        Y = np.fft.rfft(y)[1:-1]
        phases = np.angle(np.conj(X) * Y)
        np.testing.assert_allclose(phases, 0.0, atol=1e-6)

    def test_cs_simulate_negative_lag_sign_convention(self):
        r"""Negative lag produces a negative cross-spectrum phase (sign convention check).

        With :math:`\gamma^2 = 1`: :math:`Y = e^{i\phi} X` so
        :math:`\angle(\overline{X} Y) = \phi` exactly.
        This test verifies that negative :math:`\phi` is preserved faithfully.
        """
        target_lag = -0.7  # radians
        sim = CrossSpectrumSimulator(
            N=self.N, mean=500.0, dt=self.dt, rms=0.3, random_state=42
        )
        lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, lag=target_lag, coh=1.0)
        x = lc1.counts - lc1.counts.mean()
        y = lc2.counts - lc2.counts.mean()
        X = np.fft.rfft(x)[1:-1]
        Y = np.fft.rfft(y)[1:-1]
        phases = np.angle(np.conj(X) * Y)
        np.testing.assert_allclose(phases, target_lag, atol=1e-6)

    def test_cs_simulate_frequency_dependent_lag_array_recovery(self):
        r"""Frequency-dependent lag array is recovered exactly at :math:`\gamma^2 = 1`.

        With a flat PSD (index=0, :math:`T = e^{i\phi(f)}`),
        :math:`\overline{X}(f) Y(f) = e^{i\phi(f)} |X(f)|^2` so the phase at each bin
        equals :math:`\phi(f)` exactly. The lag array spans ``get_refftfreq()`` which has
        N//2 entries; ``rfft[1:-1]`` gives N//2-1 entries (excluding Nyquist), matching
        ``lag_arr[:-1]``.
        """
        sim = CrossSpectrumSimulator(
            N=self.N, mean=500.0, dt=self.dt, rms=0.3, random_state=42
        )
        w = sim.get_refftfreq()  # N//2 elements
        median_f = np.median(w)
        # Step function: 0.3 rad below median frequency, 0.9 rad above
        lag_arr = np.where(w < median_f, 0.3, 0.9)

        # Flat PSD so all bins have equal power (avoids near-zero high-freq bins)
        lc1, lc2 = sim.CS_simulate(pds1=0.0, pds2=None, lag=lag_arr, coh=1.0)
        x = lc1.counts - lc1.counts.mean()
        y = lc2.counts - lc2.counts.mean()
        # rfft has N//2+1 coefficients; [1:-1] drops DC and Nyquist → N//2-1 entries
        X = np.fft.rfft(x)[1:-1]
        Y = np.fft.rfft(y)[1:-1]
        phases = np.angle(np.conj(X) * Y)
        np.testing.assert_allclose(phases, lag_arr[:-1], atol=1e-6)

    def test_cs_simulate_rms_tuple_independent_per_band(self):
        """rms tuple produces the correct, independent fractional RMS in each band.

        The mean fractional RMS of lc1 should converge to rms1 and lc2 to rms2,
        and the two values should be measurably different.
        """
        rms1_target, rms2_target = 0.1, 0.3
        nsim = 100
        sim = CrossSpectrumSimulator(
            N=self.N, mean=self.mean, dt=self.dt, rms=(rms1_target, rms2_target), random_state=42
        )
        rms1_list, rms2_list = [], []
        for _ in range(nsim):
            lc1, lc2 = sim.CS_simulate(pds1=2.0, pds2=None, lag=0.5, coh=0.8)
            rms1_list.append(np.std(lc1.counts) / np.mean(lc1.counts))
            rms2_list.append(np.std(lc2.counts) / np.mean(lc2.counts))

        assert np.isclose(np.mean(rms1_list), rms1_target, rtol=0.1)
        assert np.isclose(np.mean(rms2_list), rms2_target, rtol=0.1)
        # Verify the two values are actually different (not just equal to the same scalar)
        assert not np.isclose(np.mean(rms1_list), np.mean(rms2_list), rtol=0.05)

    def test_cross_spectra_to_coh_lag_conversion(self):
        r"""``_cross_spectra_to_coh_lag`` correctly implements :math:`\gamma^2` and
        :math:`\phi` formulae.

        For cospec = quadspec = C and :math:`P_X = P_Y = P`:

        .. math::

            \gamma^2 = \frac{C^2 + C^2}{P^2} = \frac{2 C^2}{P^2}, \qquad
            \phi = \arctan2(C,\, C) = \frac{\pi}{4}
        """
        C = np.sqrt(2.0)
        P = np.ones(10) * 4.0
        cospec = np.ones(10) * C
        quadspec = np.ones(10) * C
        gamma2, phi = self.sim._cross_spectra_to_coh_lag(cospec, quadspec, P, P)

        expected_gamma2 = (C**2 + C**2) / (4.0 * 4.0)  # 4 / 16 = 0.25
        expected_phi = np.pi / 4

        np.testing.assert_allclose(gamma2, expected_gamma2)
        np.testing.assert_allclose(phi, expected_phi)

    # --- pds1/pds2 model string and astropy model input types ---

    def test_cs_simulate_pds1_astropy_model(self):
        """pds1 accepts an astropy Model instance (callable)."""
        mod = models.GeneralizedLorentz1D(x_0=10, fwhm=1.0, value=10.0, power_coeff=2)
        lc1, lc2 = self.sim.CS_simulate(pds1=mod, pds2=None, lag=0.5, coh=0.8)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_pds1_string_list_params(self):
        """pds1 accepts a model-name string with list params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1="generalized_lorentzian",
            pds2=None,
            lag=0.5,
            coh=0.8,
            params=[0.3, 0.9, 0.6, 0.5],
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_pds1_string_dict_params(self):
        """pds1 accepts a model-name string with dict params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1="GeneralizedLorentz1D",
            pds2=None,
            lag=0.5,
            coh=0.8,
            params={"x_0": 10, "fwhm": 1.0, "value": 10.0, "power_coeff": 2},
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_pds1_invalid_string_raises(self):
        """pds1 as an unrecognised model string raises ValueError."""
        with pytest.raises(ValueError, match="Model string not defined"):
            self.sim.CS_simulate(pds1="nonexistent_model", pds2=None, lag=0.5, coh=0.8, params=[1])

    def test_cs_simulate_pds1_string_params_not_list_or_dict_raises(self):
        """pds1 string model with params neither list nor dict raises ValueError."""
        with pytest.raises(ValueError, match="Params should be list or dictionary"):
            self.sim.CS_simulate(
                pds1="generalized_lorentzian", pds2=None, lag=0.5, coh=0.8, params=12345
            )

    def test_cs_simulate_pds2_callable(self):
        """pds2 accepts a callable with signature f(frequency) -> pds."""
        pds_func = lambda f: np.power(1 / f, 2.0)
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=pds_func, lag=0.5, coh=0.8)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_pds2_array(self):
        """pds2 accepts an array of PSD values at each rfftfreq."""
        pds_arr = np.power(1 / self.sim.get_refftfreq(), 2.0)
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=pds_arr, lag=0.5, coh=0.8)
        assert lc1.n == self.N
        assert lc2.n == self.N

    # --- lag input types ---

    def test_cs_simulate_lag_callable(self):
        """lag accepts a callable with signature f(frequency) -> lag."""
        lag_func = lambda f: np.ones_like(f) * 0.5
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, lag=lag_func)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_lag_string_list_params(self):
        """lag accepts a model-name string with list params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1=2.0,
            pds2=None,
            lag="generalized_lorentzian",
            lag_params=[3.0, 2.0, 0.5, 2.0],
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_lag_string_dict_params(self):
        """lag accepts a model-name string with dict params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1=2.0,
            pds2=None,
            lag="GeneralizedLorentz1D",
            lag_params={"x_0": 3.0, "fwhm": 2.0, "value": 0.5, "power_coeff": 2},
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_lag_invalid_string_raises(self):
        """lag as an unrecognised model string raises ValueError."""
        with pytest.raises(ValueError, match="Model string not defined"):
            self.sim.CS_simulate(
                pds1=2.0, pds2=None, lag="nonexistent_model", lag_params=[1]
            )

    def test_cs_simulate_lag_string_bad_params_raises(self):
        """lag string model with params neither list nor dict raises ValueError."""
        with pytest.raises(ValueError, match="Params should be list or dictionary"):
            self.sim.CS_simulate(
                pds1=2.0, pds2=None, lag="generalized_lorentzian", lag_params=12345
            )

    # --- coh input types ---

    def test_cs_simulate_coh_callable(self):
        """coh accepts a callable with signature f(frequency) -> coherence."""
        coh_func = lambda f: np.ones_like(f) * 0.8
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, coh=coh_func)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_coh_string_list_params(self):
        """coh accepts a model-name string with list params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1=2.0,
            pds2=None,
            coh="generalized_lorentzian",
            coh_params=[3.0, 2.0, 0.7, 2.0],
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_coh_string_dict_params(self):
        """coh accepts a model-name string with dict params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1=2.0,
            pds2=None,
            coh="GeneralizedLorentz1D",
            coh_params={"x_0": 3.0, "fwhm": 2.0, "value": 0.7, "power_coeff": 2},
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_coh_invalid_string_raises(self):
        """coh as an unrecognised model string raises ValueError."""
        with pytest.raises(ValueError, match="Model string not defined"):
            self.sim.CS_simulate(
                pds1=2.0, pds2=None, coh="nonexistent_model", coh_params=[1]
            )

    def test_cs_simulate_coh_string_bad_params_raises(self):
        """coh string model with params neither list nor dict raises ValueError."""
        with pytest.raises(ValueError, match="Params should be list or dictionary"):
            self.sim.CS_simulate(
                pds1=2.0, pds2=None, coh="generalized_lorentzian", coh_params=12345
            )

    # --- cospec/quadspec input types ---

    def test_cs_simulate_cospec_only_float(self):
        """cospec as a float without quadspec produces two valid light curves."""
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, cospec=0.5)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_quadspec_only_float(self):
        """quadspec as a float without cospec produces two valid light curves."""
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, quadspec=0.3)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_cospec_callable(self):
        """cospec accepts a callable f(frequency) -> cospec."""
        cospec_func = lambda f: np.ones_like(f) * 0.5
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, cospec=cospec_func)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_cospec_array(self):
        """cospec accepts an array of per-frequency values."""
        cospec_arr = np.ones_like(self.sim.get_refftfreq()) * 0.5
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, cospec=cospec_arr)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_cospec_string_list_params(self):
        """cospec accepts a model-name string with list params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1=2.0,
            pds2=None,
            cospec="generalized_lorentzian",
            cospec_params=[3.0, 2.0, 0.5, 2.0],
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_cospec_string_dict_params(self):
        """cospec accepts a model-name string with dict params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1=2.0,
            pds2=None,
            cospec="GeneralizedLorentz1D",
            cospec_params={"x_0": 3.0, "fwhm": 2.0, "value": 0.5, "power_coeff": 2},
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_cospec_invalid_string_raises(self):
        """cospec as an unrecognised model string raises ValueError."""
        with pytest.raises(ValueError, match="Model string not defined"):
            self.sim.CS_simulate(
                pds1=2.0, pds2=None, cospec="nonexistent_model", cospec_params=[1]
            )

    def test_cs_simulate_cospec_string_bad_params_raises(self):
        """cospec string model with params neither list nor dict raises ValueError."""
        with pytest.raises(ValueError, match="Params should be list or dictionary"):
            self.sim.CS_simulate(
                pds1=2.0, pds2=None, cospec="generalized_lorentzian", cospec_params=12345
            )

    def test_cs_simulate_quadspec_callable(self):
        """quadspec accepts a callable f(frequency) -> quadspec."""
        quadspec_func = lambda f: np.ones_like(f) * 0.3
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, quadspec=quadspec_func)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_quadspec_array(self):
        """quadspec accepts an array of per-frequency values."""
        quadspec_arr = np.ones_like(self.sim.get_refftfreq()) * 0.3
        lc1, lc2 = self.sim.CS_simulate(pds1=2.0, pds2=None, quadspec=quadspec_arr)
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_quadspec_string_list_params(self):
        """quadspec accepts a model-name string with list params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1=2.0,
            pds2=None,
            quadspec="generalized_lorentzian",
            quadspec_params=[3.0, 2.0, 0.3, 2.0],
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_quadspec_string_dict_params(self):
        """quadspec accepts a model-name string with dict params."""
        lc1, lc2 = self.sim.CS_simulate(
            pds1=2.0,
            pds2=None,
            quadspec="GeneralizedLorentz1D",
            quadspec_params={"x_0": 3.0, "fwhm": 2.0, "value": 0.3, "power_coeff": 2},
        )
        assert lc1.n == self.N
        assert lc2.n == self.N

    def test_cs_simulate_quadspec_invalid_string_raises(self):
        """quadspec as an unrecognised model string raises ValueError."""
        with pytest.raises(ValueError, match="Model string not defined"):
            self.sim.CS_simulate(
                pds1=2.0, pds2=None, quadspec="nonexistent_model", quadspec_params=[1]
            )

    def test_cs_simulate_quadspec_string_bad_params_raises(self):
        """quadspec string model with params neither list nor dict raises ValueError."""
        with pytest.raises(ValueError, match="Params should be list or dictionary"):
            self.sim.CS_simulate(
                pds1=2.0, pds2=None, quadspec="generalized_lorentzian", quadspec_params=12345
            )

import numpy as np
import os
import warnings

from scipy.interpolate import interp1d
import pytest
import astropy.modeling.models
from stingray import Lightcurve, Crossspectrum, sampledata, Powerspectrum
from stingray.simulator import Simulator
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            self.simulator.simulate_channel("3.5-4.5", "generalized_lorentzian", [1, 2, 3, 4])
        self.simulator.delete_channel("3.5-4.5")

    def test_simulate_channel_odd(self):
        """
        Simulate an energy channel.
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            self.simulator_odd.simulate_channel("3.5-4.5", "generalized_lorentzian", [1, 2, 3, 4])
        self.simulator_odd.delete_channel("3.5-4.5")

    def test_incorrect_simulate_channel(self):
        """Test simulating a channel that already exists."""
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            self.simulator.simulate_channel("3.5-4.5", 2)
            with pytest.raises(KeyError):
                self.simulator.simulate_channel("3.5-4.5", 2)
        self.simulator.delete_channel("3.5-4.5")

    def test_get_channel(self):
        """
        Retrieve an energy channel after it has been simulated.
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            self.simulator.simulate_channel("3.5-4.5", 2)
        lc = self.simulator.get_channel("3.5-4.5")
        self.simulator.delete_channel("3.5-4.5")

    def test_get_channels(self):
        """
        Retrieve multiple energy channel after it has been simulated.
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            self.simulator.simulate_channel("3.5-4.5", 2)
            self.simulator.simulate_channel("4.5-5.5", "smoothbknpo", [1, 2, 3, 4])
        lc = self.simulator.get_channels(["3.5-4.5", "4.5-5.5"])

        self.simulator.delete_channels(["3.5-4.5", "4.5-5.5"])

    def test_get_all_channels(self):
        """Retrieve all energy channels."""
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            self.simulator.simulate_channel("3.5-4.5", 2)
            self.simulator.simulate_channel("4.5-5.5", 1)
        lc = self.simulator.get_all_channels()

        self.simulator.delete_channels(["3.5-4.5", "4.5-5.5"])

    def test_count_channels(self):
        """
        Count energy channels after they have been simulated.
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            lc_all = [sim.simulate(-2.0) for i in range(nsim)]

        mean_all = np.mean([np.mean(lc.counts) for lc in lc_all])
        std_all = np.mean([np.std(lc.counts) for lc in lc_all])

        assert np.isclose(mean_all, mean, rtol=0.1)
        assert np.isclose(std_all, self.rms, rtol=0.1)

    def test_simulate_powerlaw(self):
        """
        Simulate light curve from power law spectrum.
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            assert len(self.simulator.simulate(2).counts), 1024

    def test_simulate_powerlaw_odd(self):
        """
        Simulate light curve from power law spectrum.
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            assert len(self.simulator_odd.simulate(2).counts), 2039

    def test_compare_powerlaw(self):
        """
        Compare simulated power spectrum with actual one.
        """
        B, N, red_noise, dt = 2, 1024, 10, 1

        self.simulator = Simulator(N=N, dt=dt, mean=5, rms=1, red_noise=red_noise)
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            assert len(self.simulator.simulate("generalized_lorentzian", [1, 2, 3, 4])), 1024

    def test_simulate_lorentzian_odd(self):
        """
        Simulate light curve using lorentzian model.
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            assert len(self.simulator_odd.simulate("generalized_lorentzian", [1, 2, 3, 4])), 1024

    def test_compare_lorentzian(self):
        """
        Compare simulated lorentzian spectrum with original spectrum.
        """
        N, red_noise, dt = 1024, 10, 1

        self.simulator = Simulator(N=N, dt=dt, mean=0.1, rms=0.4, red_noise=red_noise)
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            assert len(self.simulator.simulate("smoothbknpo", [1, 2, 3, 4])), 1024

    def test_compare_smoothbknpo(self):
        """
        Compare simulated smooth broken power law spectrum with original
        spectrum.
        """
        N, red_noise, dt = 1024, 10, 1

        self.simulator = Simulator(N=N, dt=dt, mean=0.1, rms=0.7, red_noise=red_noise)
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            assert len(self.simulator.simulate(mod)), 1024

    def test_simulate_SmoothBrokenPowerLaw_str(self):
        """
        Simulate a light curve using SmoothBrokenPowerLaw model
        called as a string
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            assert len(self.simulator.simulate(mod)), 1024

    def test_simulate_generic_model(self):
        """
        Simulate a light curve using a generic model
        called as a astropy.modeling.Model class
        """
        mod = astropy.modeling.models.Gaussian1D(amplitude=10.0, mean=1.0, stddev=2.0)
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            assert len(self.simulator.simulate(mod)), 1024

    def test_simulate_generic_model_odd(self):
        """
        Simulate a light curve using a generic model
        called as a astropy.modeling.Model class
        """
        mod = astropy.modeling.models.Gaussian1D(amplitude=10.0, mean=1.0, stddev=2.0)
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            _ = self.simulator.simulate(s, h)

    def test_simulate_simple_impulse_odd(self):
        """
        Simulate light curve from simple impulse response.
        """
        lc = sampledata.sample_data()
        s = lc.counts
        h = self.simulator_odd.simple_ir(10, 1, 1)
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            _ = self.simulator_odd.simulate(s, h)

    def test_powerspectrum(self):
        """
        Create a power spectrum from light curve.
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            lc = self.simulator.simulate(2)
        self.simulator.powerspectrum(lc)

    def test_powerspectrum_odd(self):
        """
        Create a power spectrum from light curve.
        """
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            lc = self.simulator_odd.simulate(2)
        self.simulator_odd.powerspectrum(lc)

    def test_simulate_relativistic_impulse(self):
        """
        Simulate light curve from relativistic impulse response.
        """
        lc = sampledata.sample_data()
        s = lc.counts

        h = self.simulator.relativistic_ir()
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            output = self.simulator.simulate(s, h)

    def test_filtered_simulate(self):
        """
        Simulate light curve using 'filtered' mode.
        """
        lc = sampledata.sample_data()
        s = lc.counts

        h = self.simulator.simple_ir()
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
            output = self.simulator.simulate(s, h, "filtered")

    def test_filtered_simulate_odd(self):
        """
        Simulate light curve using 'filtered' mode.
        """
        lc = sampledata.sample_data()
        s = lc.counts

        h = self.simulator_odd.simple_ir()
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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
        with pytest.warns(UserWarning, match="Beware! Stingray only supports poisson err_dist at the moment in many methods, and 'gauss' in a few more."):
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

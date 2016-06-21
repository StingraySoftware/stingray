import numpy as np

from astropy.tests.helper import pytest
from stingray import sampledata
from stingray.simulator import simulator, models
from scipy.stats import chisquare

class TestSimulator(object):

    @classmethod
    def setup_class(self):
        self.simulator = simulator.Simulator(N=1024)

    def test_simulate_with_seed(self):
        """
        Simulate with a random seed value.
        """
        self.simulator = simulator.Simulator(N=1024, seed=12)
        assert len(self.simulator.simulate(2).counts), 1024

    def test_simulate_powerlaw(self):
        """
        Simulate light curve from power law spectrum.
        """
        assert len(self.simulator.simulate(2).counts), 1024
        
    def test_compare_powerlaw(self):
        """
        Compare simulated power spectrum with actual one.
        """        
        B, N, red_noise, dt = 2, 1024, 10, 1

        self.simulator = simulator.Simulator(N=N, dt=dt, mean=5, rms=1, red_noise=red_noise)
        lc = [self.simulator.simulate(B) for i in range(1,30)]
        simulated = self.simulator.periodogram(lc, lc[0].tseg)       

        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = np.power((1/w), B/2)[:-1]

        actual_prob = actual/float(sum(actual))
        simulated_prob = simulated/float(sum(simulated))

        assert np.all(np.abs(actual_prob - simulated_prob) < 3 * np.sqrt(actual_prob))

    def test_simulate_powerspectrum(self):
        """
        Simulate light curve from any power spectrum.
        """
        s = np.random.rand(1024)
        assert len(self.simulator.simulate(s)), 1024

    def test_simulate_lorenzian(self):
        """
        Simulate light curve using lorenzian model.
        """
        assert len(self.simulator.simulate('lorenzian',[1,2,3,4])), 1024

    def test_compare_lorenzian(self):
        """
        Compare simulated lorenzian spectrum with original spectrum.
        """
        N, red_noise, dt = 1024, 10, 1

        self.simulator = simulator.Simulator(N=N, dt=dt, mean=0.1, rms=0.4, red_noise=red_noise)
        lc = [self.simulator.simulate('lorenzian',[0.3, 0.9, 0.6, 0.5]) for i in range(1,30)] 
        simulated = self.simulator.periodogram(lc, lc[0].tseg)
        
        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = models.lorenzian(w,[0.3, 0.9, 0.6, 0.5])[:-1]

        actual_prob = actual/float(sum(actual))
        simulated_prob = simulated/float(sum(simulated))

        assert np.all(np.abs(actual_prob - simulated_prob) < 3 * np.sqrt(actual_prob))

    def test_simulate_smoothbknpo(self):
        """
        Simulate light curve using smooth broken power law model.
        """
        assert len(self.simulator.simulate('smoothbknpo',[1,2,3,4])), 1024

    def test_compare_smoothbknpo(self):
        """
        Compare simulated smooth broken power law spectrum with original
        spectrum.
        """
        N, red_noise, dt = 1024, 10, 1

        self.simulator = simulator.Simulator(N=N, dt=dt, mean=0.1, rms=0.7, red_noise=red_noise)
        lc = [self.simulator.simulate('smoothbknpo',[0.6, 0.2, 0.6, 0.5]) for i in range(1,30)] 
        simulated = self.simulator.periodogram(lc, lc[0].tseg)
        
        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = models.smoothbknpo(w,[0.6, 0.2, 0.6, 0.5])[:-1]

        actual_prob = actual/float(sum(actual))
        simulated_prob = simulated/float(sum(simulated))

        assert np.all(np.abs(actual_prob - simulated_prob) < 3 * np.sqrt(actual_prob))

    def test_construct_ir(self):
        """
        Construct impulse response.
        """
        t0, w = 100, 500
        assert len(self.simulator.construct_ir(t0, w)), (t0+w)/self.simulator.dt

    def test_simulate_impulse(self):
        """
        Simulate light curve from impulse response.
        """
        s = sampledata.sample_data().counts
        h = self.simulator.construct_ir(100,500)
        self.simulator.simulate(s,h)

    def test_periodogram_with_lc(self):
        """
        Create a periodogram from light curve.
        """
        self.simulator.simulate(2)
        self.simulator.periodogram(self.simulator.lc)

    def test_periodogram_without_lc(self):
        """
        Create a periodogram without light curve.
        """
        self.simulator.periodogram(self.simulator.lc) 
import numpy as np

from astropy.modeling import models

from stingray import Powerspectrum, Crossspectrum

from stingray.modeling.scripts import fit_lorentzians
from stingray.modeling.scripts import fit_powerspectrum, fit_crossspectrum


class TestFitLorentzians(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(150)
        cls.nlor = 3

        cls.x_0_0 = 0.5
        cls.x_0_1 = 2.0
        cls.x_0_2 = 7.5

        cls.amplitude_0 = 200.0
        cls.amplitude_1 = 100.0
        cls.amplitude_2 = 50.0

        cls.fwhm_0 = 0.1
        cls.fwhm_1 = 1.0
        cls.fwhm_2 = 0.5

        cls.whitenoise = 2.0

        cls.priors = {
            "x_0_0": cls.x_0_0,
            "x_0_1": cls.x_0_1,
            "x_0_2": cls.x_0_2,
            "amplitude_0": cls.amplitude_0,
            "amplitude_1": cls.amplitude_1,
            "amplitude_2": cls.amplitude_2,
            "fwhm_0": cls.fwhm_0,
            "fwhm_1": cls.fwhm_1,
            "fwhm_2": cls.fwhm_2,
            "whitenoise": cls.whitenoise,
        }

        cls.model = (
            models.Lorentz1D(cls.amplitude_0, cls.x_0_0, cls.fwhm_0)
            + models.Lorentz1D(cls.amplitude_1, cls.x_0_1, cls.fwhm_1)
            + models.Lorentz1D(cls.amplitude_2, cls.x_0_2, cls.fwhm_2)
            + models.Const1D(cls.whitenoise)
        )

        freq = np.linspace(0.01, 10.0, 1000)
        p = cls.model(freq)
        noise = np.random.exponential(size=len(freq))

        power = p * noise
        cls.ps = Powerspectrum()
        cls.ps.freq = freq
        cls.ps.power = power
        cls.ps.power_err = np.array([0.0] * len(power))
        cls.ps.df = cls.ps.freq[1] - cls.ps.freq[0]
        cls.ps.m = 1

        cls.cs = Crossspectrum()
        cls.cs.freq = freq
        cls.cs.power = power
        cls.cs.power_err = np.array([0.0] * len(power))
        cls.cs.df = cls.cs.freq[1] - cls.cs.freq[0]
        cls.cs.m = 1

        cls.t0 = np.asanyarray([200.0, 0.5, 0.1, 100.0, 2.0, 1.0, 50.0, 7.5, 0.5, 2.0])

        cls.parest, cls.res = fit_lorentzians(cls.ps, cls.nlor, cls.t0)

    def test_function_creates_right_number_of_lorentians(self):
        assert (len(self.parest.lpost.model.parameters) - 1) / 3 == self.nlor

    def test_correct_parameters_without_whitenoise(self):
        parest, res = fit_lorentzians(self.ps, self.nlor, self.t0[:-1], fit_whitenoise=False)

        assert len(parest.lpost.model.parameters) / 3 == self.nlor

    def test_parameters_in_right_ballpark(self):
        true_pars = [
            self.amplitude_0,
            self.x_0_0,
            self.fwhm_0,
            self.amplitude_1,
            self.x_0_1,
            self.fwhm_1,
            self.amplitude_2,
            self.x_0_2,
            self.fwhm_2,
            self.whitenoise,
        ]

        assert np.all(np.isclose(true_pars, self.res.p_opt, rtol=0.5))

    def test_fitting_with_tied_pars(self):
        double_f = lambda model: model.x_0_0 * 4
        triple_f = lambda model: model.x_0_0 * 15
        model = self.model.copy()
        model.x_0_1.tied = double_f
        model.x_0_2.tied = triple_f
        model.amplitude_0 = self.t0[0]
        # model.bounds = {}

        t0 = np.array(
            [
                self.amplitude_0,
                self.x_0_0,
                self.fwhm_0,
                self.amplitude_1,
                self.fwhm_1,
                self.amplitude_2,
                self.fwhm_2,
                self.whitenoise,
            ]
        )

        parest, res = fit_powerspectrum(self.ps, model, np.random.normal(t0, t0 / 10))

        true_pars = [
            self.amplitude_0,
            self.x_0_0,
            self.fwhm_0,
            self.amplitude_1,
            self.fwhm_1,
            self.amplitude_2,
            self.fwhm_2,
            self.whitenoise,
        ]

        assert np.all(np.isclose(true_pars, res.p_opt, rtol=0.5))

    def test_fitting_with_fixed_pars(self):
        model = self.model.copy()
        model.amplitude_0 = self.t0[0]
        model.amplitude_0.fixed = True
        # model.bounds = {}

        t0 = np.array(
            [
                self.x_0_0,
                self.fwhm_0,
                self.amplitude_1,
                self.x_0_1,
                self.fwhm_1,
                self.amplitude_2,
                self.x_0_2,
                self.fwhm_2,
                self.whitenoise,
            ]
        )

        parest, res = fit_powerspectrum(self.ps, model, np.random.normal(t0, t0 / 10))

        true_pars = [
            self.x_0_0,
            self.fwhm_0,
            self.amplitude_1,
            self.x_0_1,
            self.fwhm_1,
            self.amplitude_2,
            self.x_0_2,
            self.fwhm_2,
            self.whitenoise,
        ]

        assert np.all(np.isclose(true_pars, res.p_opt, rtol=0.5))

    def test_fit_crossspectrum(self):
        model = self.model.copy()

        t0 = model.parameters
        _, res1 = fit_crossspectrum(self.cs, model)
        _, res2 = fit_crossspectrum(self.cs, model, t0)
        _, res3 = fit_crossspectrum(self.cs, model, t0, priors=self.priors)

        assert np.all(np.isclose(t0, res1.p_opt, rtol=0.5, atol=0.5))
        assert np.all(np.isclose(t0, res2.p_opt, rtol=0.5, atol=0.5))
        assert np.all(np.isclose(t0, res3.p_opt, rtol=0.5, atol=0.5))

    def test_fit_powerspectrum(self):
        model = self.model.copy()

        t0 = model.parameters
        _, res1 = fit_powerspectrum(self.ps, model)
        _, res2 = fit_powerspectrum(self.ps, model, t0)
        _, res3 = fit_crossspectrum(self.ps, model, t0, priors=self.priors)

        assert np.all(np.isclose(t0, res1.p_opt, rtol=0.5, atol=0.5))
        assert np.all(np.isclose(t0, res2.p_opt, rtol=0.5, atol=0.5))
        assert np.all(np.isclose(t0, res3.p_opt, rtol=0.5, atol=0.5))

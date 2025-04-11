import numpy as np
import pytest
from numpy.testing import assert_allclose

from astropy.modeling.parameters import InputParameterError
from astropy.modeling import fitting

from stingray.simulator import models


class TestModel(object):
    @classmethod
    def setup_class(self):
        self.lorentz1D = models.GeneralizedLorentz1D(x_0=3, fwhm=32, value=2.5, power_coeff=2)
        self.smoothPowerlaw = models.SmoothBrokenPowerLaw(
            norm=1, gamma_low=-2, gamma_high=2, break_freq=10
        )

    def test_model_param(self):
        lorentz1D = self.lorentz1D
        smoothPowerlaw = self.smoothPowerlaw

        assert np.allclose(smoothPowerlaw.parameters, np.array([1, -2, 2, 10]))
        assert np.allclose(lorentz1D.parameters, np.array([3.0, 32.0, 2.5, 2.0]))

        assert np.array_equal(
            lorentz1D.param_names, np.array(["x_0", "fwhm", "value", "power_coeff"])
        )
        assert np.array_equal(
            smoothPowerlaw.param_names, np.array(["norm", "gamma_low", "gamma_high", "break_freq"])
        )

    def test_lorentz_power_coeff(self):
        with pytest.raises(
            InputParameterError, match="The power coefficient should be greater than zero."
        ):
            models.GeneralizedLorentz1D(x_0=2, fwhm=100, value=3, power_coeff=-1)

    def test_lorentz_bounding_box(self):
        lorentz1D = self.lorentz1D

        assert np.allclose(lorentz1D.bounding_box(), [-797, 803])

    @pytest.mark.parametrize(
        "model, yy_func, params, x_lim",
        [
            (models.SmoothBrokenPowerLaw, models.smoothbknpo, [1, -2, 2, 10], [0.01, 100]),
            (
                models.GeneralizedLorentz1D,
                models.generalized_lorentzian,
                [3, 32, 2.5, 2],
                [-10, 10],
            ),
        ],
    )
    def test_model_evaluate(self, model, yy_func, params, x_lim):
        model = model(*params)
        xx = np.logspace(x_lim[0], x_lim[1], 100)
        yy = model(xx)
        yy_ref = yy_func(xx, params)

        assert_allclose(yy, yy_ref, rtol=0, atol=1e-8)
        assert xx.shape == yy.shape == yy_ref.shape

    @pytest.mark.parametrize(
        "model, x_lim",
        [
            (models.SmoothBrokenPowerLaw(1, -2, 2, 10), [0.01, 70]),
            (models.GeneralizedLorentz1D(3, 32, 2.5, 2), [-10, 10]),
        ],
    )
    def test_model_fitting(self, model, x_lim):
        x = np.logspace(x_lim[0], x_lim[1], 100)

        model_with_deriv = model
        model_no_deriv = model

        # add 10% noise to the amplitude
        rng = np.random.default_rng(0)
        rsn_rand_0 = rng.random(x.shape)
        n = 0.1 * (rsn_rand_0 - 0.5)

        data = model_with_deriv(x) + n
        fitter_with_deriv = fitting.LevMarLSQFitter()
        new_model_with_deriv = fitter_with_deriv(model_with_deriv, x, data)
        fitter_no_deriv = fitting.LevMarLSQFitter()
        new_model_no_deriv = fitter_no_deriv(model_no_deriv, x, data, estimate_jacobian=True)
        assert_allclose(new_model_with_deriv.parameters, new_model_no_deriv.parameters, atol=0.5)

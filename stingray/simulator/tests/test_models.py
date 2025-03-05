import numpy as np
import pytest
from numpy.testing import assert_allclose

from astropy.modeling.parameters import InputParameterError
from astropy.modeling import fitting

from stingray.simulator import models


class TestModel(object):

    def test_power_coeff(self):
        with pytest.raises(
            InputParameterError, match="The power coefficient should be greater than zero."
        ):
            models.GeneralizedLorentz1D(x_0=2, fwhm=100, value=3, power_coeff=-1)

    def test_lorentz_model(self):
        model = models.GeneralizedLorentz1D(x_0=3, fwhm=32, value=2.5, power_coeff=2)
        xx = np.linspace(2, 4, 6)
        yy = model(xx)
        yy_ref = [
            2.4902723735,
            2.4964893119,
            2.4996094360,
            2.4996094360,
            2.4964893119,
            2.4902723735,
        ]
        assert_allclose(yy, yy_ref, rtol=0, atol=1e-8)

    def test_SmoothBrokenPowerLaw_fit_deriv(self):
        x_lim = [0.01, 100]
        x = np.logspace(x_lim[0], x_lim[1], 100)

        model_with_deriv = models.SmoothBrokenPowerLaw(1, 10, -2, 2)
        model_no_deriv = models.SmoothBrokenPowerLaw(1, 10, -2, 2)

        # add 10% noise to the amplitude
        # fmt: off
        rsn_rand_1234567890 = np.array(
            [
                0.61879477, 0.59162363, 0.88868359, 0.89165480, 0.45756748,
                0.77818808, 0.26706377, 0.99610621, 0.54009489, 0.53752161,
                0.40099938, 0.70540579, 0.40518559, 0.94999075, 0.03075388,
                0.13602495, 0.08297726, 0.42352224, 0.23449723, 0.74743526,
                0.65177865, 0.68998682, 0.16413419, 0.87642114, 0.44733314,
                0.57871104, 0.52377835, 0.62689056, 0.34869427, 0.26209748,
                0.07498055, 0.17940570, 0.82999425, 0.98759822, 0.11326099,
                0.63846415, 0.73056694, 0.88321124, 0.52721004, 0.66487673,
                0.74209309, 0.94083846, 0.70123128, 0.29534353, 0.76134369,
                0.77593881, 0.36985514, 0.89519067, 0.33082813, 0.86108824,
                0.76897859, 0.61343376, 0.43870907, 0.91913538, 0.76958966,
                0.51063556, 0.04443249, 0.57463611, 0.31382006, 0.41221713,
                0.21531811, 0.03237521, 0.04166386, 0.73109303, 0.74556052,
                0.64716325, 0.77575353, 0.64599254, 0.16885816, 0.48485480,
                0.53844248, 0.99690349, 0.23657074, 0.04119088, 0.46501519,
                0.35739006, 0.23002665, 0.53420791, 0.71639475, 0.81857486,
                0.73994342, 0.07948837, 0.75688276, 0.13240193, 0.48465576,
                0.20624753, 0.02298276, 0.54257873, 0.68123230, 0.35887468,
                0.36296147, 0.67368397, 0.29505730, 0.66558885, 0.93652252,
                0.36755130, 0.91787687, 0.75922703, 0.48668067, 0.45967890
            ]
        )
        # fmt: on
        # remove non-finite values from x, rsn_rand_1234567890
        # to fit the data because
        # these value results a non-finite output
        x = np.delete(x, 26)
        rsn_rand_1234567890 = np.delete(rsn_rand_1234567890, 26)

        n = 0.1 * (rsn_rand_1234567890 - 0.5)

        data = model_with_deriv(x) + n
        fitter_with_deriv = fitting.LevMarLSQFitter()
        new_model_with_deriv = fitter_with_deriv(model_with_deriv, x, data)
        fitter_no_deriv = fitting.LevMarLSQFitter()
        new_model_no_deriv = fitter_no_deriv(model_no_deriv, x, data, estimate_jacobian=True)
        assert_allclose(new_model_with_deriv.parameters, new_model_no_deriv.parameters, atol=0.5)

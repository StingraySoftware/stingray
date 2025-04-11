import numpy as np
from astropy.modeling import Fittable1DModel
from astropy.modeling.parameters import InputParameterError, Parameter


class GeneralizedLorentz1D(Fittable1DModel):
    """
    Generalized Lorentzian function,
    implemented using astropy.modeling.models Lorentz1D

    Parameters
    ----------
    x: numpy.ndarray
        non-zero frequencies

    x_0 : float
        peak central frequency

    fwhm : float
        FWHM of the peak (gamma)

    value : float
        peak value at x=x0

    power_coeff : float
        power coefficient [n]

    Notes
    -----
    Model formula (with :math:`V` for ``value``, :math:`x_0` for ``x_0``,
    :math:`w` for ``fwhm``, and :math:`p` for ``power_coeff``):

        .. math::

            f(x) = V \\cdot \\left( \\frac{w}{2} \\right)^p
                \\cdot \\frac{1}{\\left( |x - x_0|^p + \\left( \\frac{w}{2} \\right)^p \\right)}

    Returns
    -------
    model: astropy.modeling.Model
        generalized Lorentzian psd model
    """

    x_0 = Parameter(default=1.0, description="Peak central frequency")
    fwhm = Parameter(default=1.0, description="FWHM of the peak (gamma)")
    value = Parameter(default=1.0, description="Peak value at x=x0")
    power_coeff = Parameter(default=1.0, description="Power coefficient [n]")

    def _power_coeff_validator(self, power_coeff):
        if not np.any(power_coeff > 0):
            raise InputParameterError("The power coefficient should be greater than zero.")

    power_coeff._validator = _power_coeff_validator

    @staticmethod
    def evaluate(x, x_0, fwhm, value, power_coeff):
        """
        Generalized Lorentzian function
        """
        fwhm_pc = np.power(fwhm / 2, power_coeff)
        return value * fwhm_pc * 1.0 / (np.power(np.abs(x - x_0), power_coeff) + fwhm_pc)

    @staticmethod
    def fit_deriv(x, x_0, fwhm, value, power_coeff):
        """
        Gaussian1D model function derivatives with respect
        to parameters.
        """
        fwhm_pc = np.power(fwhm / 2, power_coeff)
        num = value * fwhm_pc
        mod_x_pc = np.power(np.abs(x - x_0), power_coeff)
        denom = mod_x_pc + fwhm_pc
        denom_sq = np.power(denom, 2)

        d_x_0 = 1.0 * num / denom_sq * (power_coeff * mod_x_pc / np.abs(x - x_0)) * np.sign(x - x_0)
        d_value = fwhm_pc / denom

        pre_compute = 1.0 / 2.0 * power_coeff * fwhm_pc / (fwhm / 2)
        d_fwhm = 1.0 / denom_sq * (denom * (value * pre_compute) - num * pre_compute)

        d_power_coeff = (
            1.0
            / denom_sq
            * (
                denom * (value * np.log(fwhm / 2) * fwhm_pc)
                - num * (np.log(abs(x - x_0)) * mod_x_pc + np.log(fwhm / 2) * fwhm_pc)
            )
        )
        return [d_x_0, d_value, d_fwhm, d_power_coeff]

    def bounding_box(self, factor=25):
        """Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.

        Parameters
        ----------
        factor : float
            The multiple of FWHM used to define the limits.
            Default is chosen to include most (99%) of the
            area under the curve, while still showing the
            central feature of interest.

        """
        x0 = self.x_0
        dx = factor * self.fwhm

        return (x0 - dx, x0 + dx)

    # NOTE:
    # In astropy 4.3 'Parameter' object has no attribute 'input_unit',
    # whereas newer versions of Astropy include this attribute.

    # TODO:
    # Add 'input_units' and '_parameter_units_for_data_units' methods
    # when we drop support for Astropy < 5.3.


class SmoothBrokenPowerLaw(Fittable1DModel):
    """
    Smooth broken power law function,
    implemented using astropy.modeling.models SmoothlyBrokenPowerLaw1D

    Parameters
    ----------
    x: numpy.ndarray
        non-zero frequencies

    norm: float
        normalization frequency

    gamma_low: float
        power law index for f --> zero

    gamma_high: float
        power law index for f --> infinity

    break_freq: float
        break frequency

    Notes
    -----
    Model formula (with :math:`N` for ``norm``, :math:`x_b` for
    ``break_freq``, :math:`\\gamma_1` for ``gamma_low``,
    and :math:`\\gamma_2` for ``gamma_high``):

        .. math::

            f(x) = N \\cdot x^{-\\gamma_1}
                   \\left( 1 + \\left( \\frac{x}{x_b} \\right)^2 \\right)^{\\frac{\\gamma_1 - \\gamma_2}{2}}

    Returns
    -------
    model: astropy.modeling.Model
        generalized smooth broken power law psd model
    """

    norm = Parameter(default=1.0, description="normalization frequency")
    gamma_low = Parameter(default=-1.0, description="Power law index for f --> zero")
    gamma_high = Parameter(default=1.0, description="Power law index for f --> infinity")
    break_freq = Parameter(default=1.0, description="Break frequency")

    def _norm_validator(self, value):
        if np.any(value <= 0):
            raise InputParameterError("norm parameter must be > 0")

    norm._validator = _norm_validator

    @staticmethod
    def evaluate(x, norm, gamma_low, gamma_high, break_freq):
        """One dimensional smoothly broken power law model function."""
        # Pre-calculate `x/x_b`
        xx = x / break_freq

        # Initialize the return value
        f = np.zeros_like(x, subok=False)

        # The quantity `t = (x / x_b)^(1 / 2)` can become quite
        # large.  To avoid overflow errors we will start by calculating
        # its natural logarithm:
        logt = np.log(xx) / 2

        # When `t >> 1` or `t << 1` we don't actually need to compute
        # the `t` value since the main formula (see docstring) can be
        # significantly simplified by neglecting `1` or `t`
        # respectively.  In the following we will check whether `t` is
        # much greater, much smaller, or comparable to 1 by comparing
        # the `logt` value with an appropriate threshold.
        threshold = 30  # corresponding to exp(30) ~ 1e13
        i = logt > threshold
        if i.max():
            f[i] = norm * np.power(x[i], -gamma_low) * np.power(xx[i], gamma_low - gamma_high)

        i = logt < -threshold
        if i.max():
            f[i] = norm * np.power(x[i], -gamma_low)

        i = np.abs(logt) <= threshold
        if i.max():
            # In this case the `t` value is "comparable" to 1, hence
            # we will evaluate the whole formula.
            f[i] = (
                norm
                * np.power(x[i], -gamma_low)
                * np.power(1.0 + np.power(xx[i], 2), (gamma_low - gamma_high) / 2)
            )
        return f

    @staticmethod
    def fit_deriv(x, norm, gamma_low, gamma_high, break_freq):
        """One dimensional smoothly broken power law derivative with respect
        to parameters.
        """
        # Pre-calculate `x_b` and `x/x_b` and `logt` (see comments in
        # SmoothBrokenPowerLaw.evaluate)
        xx = x / break_freq
        logt = np.log(xx) / 2

        # Initialize the return values
        f = np.zeros_like(x)
        d_norm = np.zeros_like(x)
        d_gamma_low = np.zeros_like(x)
        d_gamma_high = np.zeros_like(x)
        d_break_freq = np.zeros_like(x)

        threshold = 30  # (see comments in SmoothBrokenPowerLaw.evaluate)
        i = logt > threshold
        if i.max():
            f[i] = norm * np.power(x[i], -gamma_low) * np.power(xx[i], gamma_low - gamma_high)

            d_norm[i] = f[i] / norm
            d_gamma_low[i] = f[i] * (-np.log(x[i]) + np.log(xx[i]))
            d_gamma_high[i] = -f[i] * np.log(xx[i])
            d_break_freq[i] = f[i] * (gamma_high - gamma_low) / break_freq

        i = logt < -threshold
        if i.max():
            f[i] = norm * np.power(x[i], -gamma_low)

            d_norm[i] = f[i] / norm
            d_gamma_low[i] = -f[i] * np.log(x[i])
            d_gamma_high[i] = 0
            d_break_freq[i] = 0

        i = np.abs(logt) <= threshold
        if i.max():
            # In this case the `t` value is "comparable" to 1, hence we
            # we will evaluate the whole formula.
            f[i] = (
                norm
                * np.power(x[i], -gamma_low)
                * np.power(1.0 + np.power(xx[i], 2), (gamma_low - gamma_high) / 2)
            )
            d_norm[i] = f[i] / norm
            d_gamma_low[i] = f[i] * (-np.log(x[i]) + np.log(1.0 + np.power(xx[i], 2)) / 2)
            d_gamma_high[i] = -f[i] * np.log(1.0 + np.power(xx[i], 2)) / 2
            d_break_freq[i] = (
                f[i]
                * (np.power(x[i], 2) * (gamma_high - gamma_low))
                / (break_freq * (np.power(break_freq, 2) + np.power(x[i], 2)))
            )

        return [d_norm, d_gamma_low, d_gamma_high, d_break_freq]

    # NOTE:
    # In astropy 4.3 'Parameter' object has no attribute 'input_unit',
    # whereas newer versions of Astropy include this attribute.

    # TODO:
    # Add 'input_units' and '_parameter_units_for_data_units' methods
    # when we drop support for Astropy < 5.3.


def generalized_lorentzian(x, p):
    """
    Generalized Lorentzian function.

    Parameters
    ----------

    x: numpy.ndarray
        non-zero frequencies

    p: iterable
        p[0] = peak centeral frequency
        p[1] = FWHM of the peak (gamma)
        p[2] = peak value at x=x0
        p[3] = power coefficient [n]

    Returns
    -------
    model: numpy.ndarray
        generalized lorentzian psd model
    """

    assert p[3] > 0.0, "The power coefficient should be greater than zero."
    return p[2] * (p[1] / 2) ** p[3] * 1.0 / (abs(x - p[0]) ** p[3] + (p[1] / 2) ** p[3])


def smoothbknpo(x, p):
    """
    Smooth broken power law function.

    Parameters
    ----------

    x: numpy.ndarray
        non-zero frequencies

    p: iterable
        p[0] = normalization frequency
        p[1] = power law index for f --> zero
        p[2] = power law index for f --> infinity
        p[3] = break frequency

    Returns
    -------
    model: numpy.ndarray
        generalized smooth broken power law psd model
    """

    return p[0] * x ** (-p[1]) / (1.0 + (x / p[3]) ** 2) ** (-(p[1] - p[2]) / 2)

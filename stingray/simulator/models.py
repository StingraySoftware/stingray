import numpy as np
from astropy.modeling import Fittable1DModel
from astropy.modeling.parameters import InputParameterError, Parameter
from astropy.units import Quantity


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

    Returns
    -------
    model: astropy.modeling.Model
        generalized Lorentzian psd model
    """

    x_0 = Parameter(default=1.0, description="Peak central frequency")
    fwhm = Parameter(default=1.0, description="FWHM of the peak (gamma)")
    value = Parameter(default=1.0, description="Peak value at x=x0")
    power_coeff = Parameter(default=1.0, description="Power coefficient [n]")

    @staticmethod
    def evaluate(x, x_0, fwhm, value, power_coeff):
        """
        Generalized Lorentzian function
        """
        assert power_coeff > 0.0, "The power coefficient should be greater than zero."
        fwhm_pc = np.power(fwhm / 2, power_coeff)
        return value * fwhm_pc * 1.0 / (np.power(np.abs(x - x_0), power_coeff) + fwhm_pc)

    @staticmethod
    def fit_deriv(x, x_0, fwhm, value, power_coeff):
        """
        Gaussian1D model function derivatives.
        """
        assert power_coeff > 0.0, "The power coefficient should be greater than zero."
        fwhm_pc = np.power(fwhm / 2, power_coeff)
        num = value * fwhm_pc
        mod_x_pc = np.power(np.abs(x - x_0), power_coeff)
        denom = mod_x_pc + fwhm_pc
        denom_sq = np.power(denom, 2)

        del_func_x = (
            -1.0 * num / denom_sq * (power_coeff * mod_x_pc / np.abs(x - x_0)) * np.sign(x - x_0)
        )
        del_func_x_0 = -del_func_x
        del_func_value = fwhm_pc / denom

        pre_compute = 1.0 / 2.0 * power_coeff * fwhm_pc / (fwhm / 2)
        del_func_fwhm = 1.0 / denom_sq * (denom * (value * pre_compute) - num * pre_compute)

        del_func_p_coeff = (
            1.0
            / denom_sq
            * (
                denom * (value * np.log(fwhm / 2) * fwhm_pc)
                - num * (np.log(abs(x - x_0)) * mod_x_pc + np.log(fwhm / 2) * fwhm_pc)
            )
        )
        return [del_func_x, del_func_x_0, del_func_value, del_func_fwhm, del_func_p_coeff]

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

    @property
    def input_units(self):
        if self.x_0.input_unit is None:
            return None
        return {self.inputs[0]: self.x_0.input_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {
            "x_0": inputs_unit[self.inputs[0]],
            "fwhm": inputs_unit[self.inputs[0]],
            "value": outputs_unit[self.outputs[0]],
        }


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

    Returns
    -------
    model: astropy.modeling.Model
        generalized smooth broken power law psd model
    """

    norm = Parameter(default=1.0, description="normalization frequency")
    break_freq = Parameter(default=1.0, description="Break frequency")
    gamma_low = Parameter(default=-1.0, description="Power law index for f --> zero")
    gamma_high = Parameter(default=1.0, description="Power law index for f --> infinity")

    def _norm_validator(self, value):
        if np.any(value <= 0):
            raise InputParameterError("norm parameter must be > 0")

    norm._validator = _norm_validator

    @staticmethod
    def evaluate(x, norm, gamma_low, gamma_high, break_freq):
        norm_ = norm * x ** (-1 * gamma_low)
        if isinstance(norm_, Quantity):
            return_unit = norm_.unit
            norm = norm_.value
        else:
            return_unit = None

        exp_factor = (gamma_low - gamma_high) / 2
        break_freq_invsq = 1.0 / np.power(break_freq, 2)
        f = (
            norm
            * np.power(x, -gamma_low)
            * np.power(1.0 + np.power(x, 2) * break_freq_invsq, exp_factor)
        )
        return Quantity(f, unit=return_unit, copy=False, subok=True)

    @staticmethod
    def fit_deriv(x, norm, gamma_low, gamma_high, break_freq):
        exp_factor = (gamma_low - gamma_high) / 2
        x_g_low = np.power(x, -gamma_low)
        A = norm * x_g_low
        x_b_freq_sq = 1.0 + np.power(x / break_freq, 2)
        B = np.power(x_b_freq_sq, exp_factor)

        del_func_x = B * norm * (-gamma_low * x_g_low / x) + A * B / x_b_freq_sq * (
            2 * np.power(x / break_freq, 2)
        )

        del_func_norm = x_g_low * B

        del_func_g_low = (
            B * norm * -1.0 * np.log(x) * x_g_low + A * (1.0 / 2.0) * np.log(x_b_freq_sq) * B
        )

        del_func_g_high = A * -1.0 / 2.0 * np.log(x_b_freq_sq) * B
        del_func_b_freq = (
            A * (exp_factor) * B / x_b_freq_sq * (np.pow(x, 2) * -2 / np.power(break_freq, 3))
        )
        return [del_func_x, del_func_norm, del_func_g_low, del_func_g_high, del_func_b_freq]

    @property
    def input_units(self):
        if self.break_freq.input_unit is None:
            return None
        return {self.inputs[0]: self.break_freq.input_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {
            "break_freq": inputs_unit[self.inputs[0]],
            "norm_": outputs_unit[self.outputs[0]],
        }


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

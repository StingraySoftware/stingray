import numpy as np
from astropy.modeling.models import custom_model


@custom_model
def GeneralizedLorentz1D(x, x_0=1.0, fwhm=1.0, value=1.0, power_coeff=1.0):
    """
    Generalized Lorentzian function,
    implemented using astropy.modeling.models custom model

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
    assert power_coeff > 0.0, "The power coefficient should be greater than zero."
    num = value * (fwhm / 2) ** power_coeff
    denom = abs(x - x_0) ** power_coeff + (fwhm / 2) ** power_coeff
    func = num / denom

    del_func_x = (
        -1.0 * num / denom**2 * (power_coeff * abs(x - x_0) ** (power_coeff - 1)) * np.sign(x - x_0)
    )
    del_func_x_0 = (
        num / denom**2 * (power_coeff * abs(x - x_0) ** (power_coeff - 1)) * np.sign(x - x_0)
    )
    del_func_value = (fwhm / 2) ** power_coeff / denom
    del_func_fwhm = (
        1.0
        / denom**2
        * (
            denom * (value * 1.0 / 2 * power_coeff * (fwhm / 2) ** (power_coeff - 1))
            - num * (1.0 / 2.0 * power_coeff * (fwhm / 2) ** (power_coeff - 1))
        )
    )
    del_func_p_coeff = (
        1.0
        / denom**2
        * (
            denom * (value * np.log(fwhm / 2) * (fwhm / 2) ** power_coeff)
            - num
            * (
                np.log(abs(x - x_0)) * abs(x - x_0) ** power_coeff
                + np.log(fwhm / 2) * (fwhm / 2) ** power_coeff
            )
        )
    )

    jacob = [
        del_func_x,
        del_func_x_0,
        del_func_value,
        del_func_fwhm,
        del_func_p_coeff,
    ]
    return func, jacob


@custom_model
def SmoothBrokenPowerLaw(x, norm=1.0, gamma_low=1.0, gamma_high=1.0, break_freq=1.0):
    """
    Smooth broken power law function,
    implemented using astropy.modeling.models custom model

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
    A = norm * x ** (-gamma_low)
    B = (1.0 + (x / break_freq) ** 2) ** ((gamma_low - gamma_high) / 2)
    func = A * B

    del_func_x = B * norm * (-gamma_low) * x ** (-gamma_low - 1) + A * (
        1.0 + (x / break_freq) ** 2
    ) ** ((gamma_low - gamma_high) / 2 - 1) * (2 * x / break_freq**2)

    del_func_norm = x ** (-gamma_low) * B

    del_func_g_low = B * norm * -1.0 * np.log(x) * x ** (-gamma_low) + A * (1.0 / 2.0) * np.log(
        1.0 + (x / break_freq) ** 2
    ) * (1.0 + (x / break_freq) ** 2) ** ((gamma_low - gamma_high) / 2)

    del_func_g_high = (
        A
        * -1.0
        / 2.0
        * np.log(1.0 + (x / break_freq) ** 2)
        * (1.0 + (x / break_freq) ** 2) ** ((gamma_low - gamma_high) / 2)
    )
    del_func_b_freq = (
        A
        * ((gamma_low - gamma_high) / 2)
        * (1.0 + (x / break_freq) ** 2) ** ((gamma_low - gamma_high) / 2 - 1)
        * (x**2 * -2 / break_freq**3)
    )

    jacob = [
        del_func_x,
        del_func_norm,
        del_func_g_low,
        del_func_g_high,
        del_func_b_freq,
    ]
    return func, jacob


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

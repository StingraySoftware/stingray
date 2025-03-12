from astropy.modeling.models import custom_model
import numpy as np
import matplotlib.pyplot as plt

# TODO: Added Jacobian functions
def GeneralizedLorentz1DJacobian(x: np.ndarray, x_0: float, fwhm: float, value: float, power_coeff: float) -> np.ndarray:
    """
    Compute the Jacobian matrix for the Generalized Lorentzian function.

    Parameters
    ----------
    x : numpy.ndarray
        Non-zero frequencies.
    x_0 : float
        Peak central frequency.
    fwhm : float
        Full width at half maximum (FWHM) of the peak.
    value : float
        Peak value at x = x_0.
    power_coeff : float
        Power coefficient.

    Returns
    -------
    numpy.ndarray
        The computed Jacobian matrix of shape (len(x), 4).
    """
    dx = x - x_0
    gamma_pow = (fwhm / 2) ** power_coeff
    denom = dx**power_coeff + gamma_pow
    denom_sq = denom ** 2
    
    d_x0 = power_coeff * value * gamma_pow * dx**(power_coeff - 1) / denom_sq
    d_fwhm = power_coeff * value * (fwhm / 2) ** (power_coeff - 1) / 2 * (1 + power_coeff * gamma_pow / denom) / denom
    d_value = gamma_pow / denom
    d_power_coeff = value * gamma_pow * np.log(fwhm / 2) / denom - value * gamma_pow * np.log(abs(dx) + (fwhm / 2)) / denom
    
    return np.vstack([-d_x0, d_fwhm, d_value, d_power_coeff]).T

# TODO: Add Jacobian
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
    return (
        value
        * (fwhm / 2) ** power_coeff
        * 1.0
        / (abs(x - x_0) ** power_coeff + (fwhm / 2) ** power_coeff)
    )

# TODO: Added Jacobian functions
def SmoothBrokenPowerLawJacobian(x: np.ndarray, norm: float, gamma_low: float, gamma_high: float, break_freq: float) -> np.ndarray:
    """
    Compute the Jacobian matrix for the Smooth Broken Power Law function.

    Parameters
    ----------
    x : numpy.ndarray
        Non-zero frequencies.
    norm : float
        Normalization frequency.
    gamma_low : float
        Power law index for f → zero.
    gamma_high : float
        Power law index for f → infinity.
    break_freq : float
        Break frequency.

    Returns
    -------
    numpy.ndarray
        The computed Jacobian matrix of shape (len(x), 4).
    """
    x_bf2 = (x / break_freq) ** 2
    denom = (1.0 + x_bf2) ** (-(gamma_low - gamma_high) / 2)
    d_norm = x ** (-gamma_low) * denom
    d_gamma_low = -norm * np.log(x) * x ** (-gamma_low) * denom
    d_gamma_high = norm * x ** (-gamma_low) * denom * np.log(1 + x_bf2) / 2
    d_break_freq = norm * x ** (-gamma_low) * denom * (gamma_low - gamma_high) * x_bf2 / (break_freq * (1 + x_bf2))
    
    return np.vstack([d_norm, d_gamma_low, d_gamma_high, d_break_freq]).T

# TODO: Add Jacobian
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
    return (
        norm * x ** (-gamma_low) / (1.0 + (x / break_freq) ** 2) ** (-(gamma_low - gamma_high) / 2)
    )


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

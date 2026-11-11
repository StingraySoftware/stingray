import warnings
from stingray.utils import njit, prange

from stingray.loggingconfig import setup_logger

from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.interpolate import interp1d

logger = setup_logger()

MAX_FACTORIAL = 100
__INVERSE_FACTORIALS = 1.0 / factorial(np.arange(MAX_FACTORIAL))
PRECISION = np.finfo(float).precision
TWOPI = np.pi * 2

STERLING_PARAMETERS = np.array([1 / 12, 1 / 288, -139 / 51840, -571 / 2488320])

__all__ = [
    "r_det",
    "r_in",
    "pds_model_zhang",
    "non_paralyzable_dead_time_model",
    "deadtime_response",
    "vikhlinin_dead_time_model",
    "check_A",
    "check_B",
    "e_m_x_x_over_factorial",
    "A0",
    "A_single_k",
    "A",
    "limit_A",
    "B",
]


@njit()
def _stirling_factor(m):
    """First few terms of series expansion appearing in Stirling's approximation."""
    fact = 1.0
    power = 1.0
    for i in range(1, 5):
        power *= m
        fact += STERLING_PARAMETERS[i - 1] / power
    return fact


@njit()
def e_m_x_x_over_factorial(x: float, m: int):
    r"""Approximate the large number ratio in Eq. 34.

    The original formula for :math:`G_n` (eq. 34 in Zhang+95) has factors of the kind
    :math:`e^{-x} x^l / l!`. This is the product of very large numbers, that mostly
    balance each other. In this function, we approximate this product for large :math:`l`
    starting from Stirling's approximation for the factorial:

    .. math::

       l! \approx \sqrt{2\pi l} (\frac{l}{e})^l A(l)

    where :math:`A(l)` is a series expansion in 1/l of the form
    :math:`1 + 1 / 12l + 1 / 288l^2 - 139/51840/l^3 + ...`. This allows to transform the product
    above into

    .. math::

       \frac{e^{-x} x^l}{l!} \approx \frac{1}{A(l)\sqrt{2\pi l}}\left(\frac{x e}{l}\right)^l e^{-x}

    and then, bringing the exponential into the parenthesis

    .. math::

       \frac{e^{-x} x^l}{l!}\approx\frac{1}{A(l)\sqrt{2\pi l}} \left(\frac{x e^{1-x/l}}{l}\right)^l

    The function inside the brackets has a maximum around :math:`x \approx l` and is well-behaved,
    allowing to approximate the product for large :math:`l` without the need to calculate the
    factorial or the exponentials directly.
    """
    # Numerical errors occur when x is not a float
    x = float(x)

    if x == 0.0 and m == 0:
        return 1.0

    # For decently small numbers, use the direct formula
    if m < 50 or (x > 1 and m * max(np.log10(x), 1) < 50):
        return np.exp(-x) * x**m * __INVERSE_FACTORIALS[m]

    # Use Stirling's approximation
    return 1.0 / np.sqrt(TWOPI * m) * np.power(x * np.exp(1 - x / m) / m, m) / _stirling_factor(m)


def r_in(td, r_0):
    """Calculate incident countrate given dead time and detected countrate.

    Parameters
    ----------
    td : float
        Dead time
    r_0 : float
        Detected countrate
    """
    tau = 1 / r_0
    return 1.0 / (tau - td)


def r_det(td, r_i):
    """Calculate detected countrate given dead time and incident countrate.

    Parameters
    ----------
    td : float
        Dead time
    r_i : float
        Incident countrate
    """
    tau = 1 / r_i
    return 1.0 / (tau + td)


def _alias_sum(func, freqs, bin_time, n_alias):
    r"""Fold ``func`` onto the frequency grid of a light curve binned at ``bin_time``.

    Returns :math:`\sum_m F(f_m)\,{\rm sinc}^2(f_m t_b)` with :math:`f_m = f + m/t_b`,
    :math:`|m| \le` ``n_alias``: binning multiplies the Fourier transform by the bin
    window and sampling folds every frequency onto the grid below Nyquist.
    """
    out = 0.0
    for m in range(-n_alias, n_alias + 1):
        shifted = freqs + m / bin_time
        out = out + func(shifted) * np.sinc(shifted * bin_time) ** 2
    return out


def deadtime_response(freqs, dead_time, bin_time=None, n_alias=100):
    r"""Frequency response of a non-paralyzable detector's live time to the count rate.

    The detector is dead after each event for ``dead_time``, so the live time at a given
    moment responds to the count rate averaged over the preceding dead time, not to the
    instantaneous rate. The Fourier transform of that running mean is

    .. math::

        W(f) = e^{-i\pi f t_d}\,{\rm sinc}(f t_d),

    with numpy's convention :math:`{\rm sinc}(x) = \sin(\pi x)/(\pi x)` and the sign of
    numpy's forward FFT. :math:`W \to 1` well below :math:`1/(2\pi t_d)`, which is the
    limit in which the live time follows the rate instantaneously. This is the response
    of the Vikhlinin et al. 1994 noise model, see :func:`vikhlinin_dead_time_model`.

    Parameters
    ----------
    freqs : float or `np.ndarray`
        Frequencies, in Hz.
    dead_time : float
        Dead time, in s.

    Other Parameters
    ----------------
    bin_time : float, default None
        If given, the response seen on a light curve binned at ``bin_time``: the
        unbinned response folded back below the Nyquist frequency, weighted by the bin
        window, as it applies to a signal whose power is flat across the folds. This is
        the same aliasing that turns the unbinned noise model into the Zhang et al. 1995
        one. A red signal has little power near Nyquist and is better described by the
        unbinned response.
    n_alias : int, default 100
        Number of folds on each side used for the binned response. The terms fall as
        :math:`m^{-3}`, so the default is far more than enough.

    Returns
    -------
    response : complex or `np.ndarray`
        :math:`W(f)`, with the shape of ``freqs``.

    Examples
    --------
    >>> w = deadtime_response(np.array([0.0, 400.0]), 2.5e-3)
    >>> bool(np.isclose(w[0], 1)), bool(np.isclose(w[1], 0))
    (True, True)
    """
    freqs = np.asanyarray(freqs, dtype=float)

    def unbinned(f):
        return np.exp(-1j * np.pi * f * dead_time) * np.sinc(f * dead_time)

    if bin_time is None:
        return unbinned(freqs)
    return _alias_sum(unbinned, freqs, bin_time, n_alias)


def vikhlinin_dead_time_model(
    freqs, dead_time, rate, bin_time=None, background_rate=0.0, n_alias=100
):
    r"""Dead-time-modified Poisson noise level, after Vikhlinin et al. 1994.

    For unbinned events from a non-paralyzable detector, in Leahy normalisation,

    .. math::

        P(f) = \frac{2}{|1 + w\,W(f)|^2},
        \qquad w = r_{\rm in} t_d = \frac{1 - f_{\rm live}}{f_{\rm live}},

    where :math:`W` is :func:`deadtime_response` and
    :math:`f_{\rm live} = 1 - r_{\rm det} t_d`. At low frequency this is
    :math:`2 f_{\rm live}^2`. With ``bin_time`` it is folded onto the grid of the binned
    light curve,

    .. math::

        P_b(f) = 2 + \sum_m \left[P(f_m) - 2\right] {\rm sinc}^2(f_m t_b),
        \qquad f_m = f + m/t_b,

    which reproduces :func:`non_paralyzable_dead_time_model` (Zhang et al. 1995) to a
    few parts in a thousand, at a fraction of the cost and with no truncated sum to
    converge: the Zhang model with too small a ``limit_k`` is wrong when there are many
    bins per dead time, this one is not.

    Parameters
    ----------
    freqs : `np.ndarray`
        Frequencies, in Hz.
    dead_time : float
        Dead time, in s.
    rate : float
        Detected source count rate.

    Other Parameters
    ----------------
    bin_time : float, default None
        Bin time of the light curve. If None, the model for unbinned events.
    background_rate : float, default 0
        Detected background count rate, which causes dead time but is not part of the
        source; handled as in :func:`non_paralyzable_dead_time_model`.
    n_alias : int, default 100
        Number of folds on each side used for the binned model.

    Returns
    -------
    power : `np.ndarray`
        The Leahy-normalised noise level at ``freqs``.

    Examples
    --------
    >>> p = vikhlinin_dead_time_model(np.array([1.0]), 2.5e-3, 200.0)
    >>> bool(np.isclose(p[0], 2 * 0.5**2, rtol=1e-3))
    True
    """
    total = rate + background_rate
    if total > 1 / dead_time:
        raise ValueError(
            "The sum of the source and background count rates is larger than the inverse "
            "of the dead time. This is not a physical situation. Please check your input."
        )
    w = total * dead_time / (1 - total * dead_time)
    freqs = np.asanyarray(freqs, dtype=float)

    def unbinned(f):
        return 2 / np.abs(1 + w * deadtime_response(f, dead_time)) ** 2

    if bin_time is None:
        power = unbinned(freqs)
    else:
        # The bin windows sum to one over all folds, so folding P - 2 rather than P
        # leaves the flat part exact and makes the sum converge as m^-3.
        power = 2 + _alias_sum(lambda f: unbinned(f) - 2, freqs, bin_time, n_alias)

    if background_rate > 0:
        power = (power - 2) * rate / total + 2
    return power


@njit()
def _Gn(x, n):
    """Term in Eq. 34 in Zhang+95."""
    s = 0.0

    for m in range(0, n):
        new_val = e_m_x_x_over_factorial(x, m) * (n - m)

        s += new_val
        # The curve above has a maximum around x~l
        if x != 0 and s > 0 and m > 2 * x and -np.log10(np.abs(new_val / s)) > PRECISION:
            break

    return s


@njit()
def _h(k, n, td, tb, tau):
    """Term in Eq. 35 in Zhang+95."""
    # Typo in Zhang+95 corrected. k * tb, not k * td
    factor = k * tb - n * td

    if k * tb - n * td < 0:
        return 0.0

    val = k - n * (td + tau) / tb + tau / tb * _Gn(factor / tau, n)

    return val


INFINITE = 699


@njit()
def A0(r0, td, tb, tau):
    """Term in Eq. 38 in Zhang+95.

    Parameters
    ----------
    r0 : float
        Detected countrate
    td : float
        Dead time
    tb : float
        Bin time of the light curve
    tau : float
        Inverse of the incident countrate
    """
    s = 0.0
    for n in range(1, int(max(2, tb / td * 2 + 1))):
        s += _h(1, n, td, tb, tau)

    return r0 * tb * (1 + 2 * s)


@njit()
def A_single_k(k, r0, td, tb, tau):
    """Term in Eq. 39 in Zhang+95.

    Parameters
    ----------
    k : int
        Order of the term
    r0 : float
        Detected countrate
    td : float
        Dead time
    tb : float
        Bin time of the light curve
    tau : float
        Inverse of the incident countrate
    """
    if k == 0:
        return A0(r0, td, tb, tau)
    # Equation 39
    s = 0.0

    for n in range(1, int(max(3, (k + 1) * tb / td * 2))):
        new_val = _h(k + 1, n, td, tb, tau) - 2 * _h(k, n, td, tb, tau) + _h(k - 1, n, td, tb, tau)

        s += new_val

    return r0 * tb * s


def A(k, r0, td, tb, tau):
    """Term in Eq. 39 in Zhang+95.

    Parameters
    ----------
    k : int or array of ints
        Order of the term
    r0 : float
        Detected countrate
    td : float
        Dead time
    tb : float
        Bin time of the light curve
    tau : float
        Inverse of the incident countrate
    """
    if isinstance(k, Iterable):
        return np.array([A_single_k(ki, r0, td, tb, tau) for ki in k])

    return A_single_k(k, r0, td, tb, tau)


def limit_A(rate, td, tb):
    """Limit of A for k->infty, as per Eq. 43 in Zhang+95.

    Parameters
    ----------
    rate : float
        Incident count rate
    td : float
        Dead time
    tb : float
        Bin time of the light curve
    """
    r0 = r_det(td, rate)
    return r0**2 * tb**2


def check_A(rate, td, tb, max_k=100, save_to=None, linthresh=0.000001, rate_is_incident=True):
    r"""Test that A is well-behaved.

    This function produces a plot of :math:`A_k - r_0^2 t_b^2` vs :math:`k`, to visually check that
    :math:`A_k \rightarrow r_0^2 t_b^2` for :math:`k\rightarrow\infty`, as per Eq. 43 in Zhang+95.

    With this function is possible to determine how many inner loops ``k`` (``limit_k`` in function
    pds_model_zhang) are necessary for a correct approximation of the dead time model

    Parameters
    ----------
    rate : float
        Count rate, either incident or detected (use the ``rate_is_incident`` bool to specify)
    td : float
        Dead time
    tb : float
        Bin time of the light curve

    Other Parameters
    ----------------
    max_k : int
        Maximum k to plot
    save_to : str, default None
        If not None, save the plot to this file
    linthresh : float, default 0.000001
        Linear threshold for the "symlog" scale of the plot
    rate_is_incident : bool, default True
        If True, the input rate is the incident count rate. If False, it is the detected one.
    """
    if rate_is_incident:
        tau = 1 / rate
        r0 = r_det(td, rate)
    else:
        r0 = rate
        tau = 1 / r_in(td, rate)

    limit = limit_A(rate, td, tb)
    fig = plt.figure()

    k_values = np.arange(0, max_k + 1)
    A_values = A(k_values, r0, td, tb, tau)

    plt.plot(k_values, A_values - limit, color="k")
    plt.semilogx()
    plt.yscale("symlog", linthresh=linthresh)
    plt.axhline(0, ls="--", color="k")
    plt.xlabel("$k$")
    plt.ylabel("$A_k - r_0^2 t_b^2$")
    if save_to is not None:
        plt.savefig(save_to)
        plt.close(fig)
    return k_values, A_values


@njit()
def _B_raw(k, r0, td, tb, tau):
    """Term in Eq. 45 in Zhang+95."""
    if k == 0:
        return 2 * (A_single_k(0, r0, td, tb, tau) - r0**2 * tb**2) / (r0 * tb)

    new_val = A_single_k(k, r0, td, tb, tau) - r0**2 * tb**2

    return 4 * new_val / (r0 * tb)


@njit()
def _safe_B_single_k(k, r0, td, tb, tau, limit_k=60):
    """Term in Eq. 39 in Zhang+95, with a cut in the maximum k.

    This can be risky. Only use if B is really 0 for high k.
    """
    if k > limit_k:
        return 0.0
    return _B_raw(k, r0, td, tb, tau)


def _safe_B(k, r0, td, tb, tau, limit_k=60):
    """Term in Eq. 39 in Zhang+95, with a cut in the maximum k.

    This can be risky. Only use if B is really 0 for high k.
    """
    if isinstance(k, Iterable):
        return np.array([_safe_B_single_k(ki, r0, td, tb, tau, limit_k=limit_k) for ki in k])
    return _safe_B_single_k(int(k), r0, td, tb, tau, limit_k=limit_k)


def B(k, r0, td, tb, tau, limit_k=60):
    """Term in Eq. 39 in Zhang+95, with a cut in the maximum k.

    The cut can be risky. Only use if B is really 0 for high k. Use `check_B` to test.

    Parameters
    ----------
    k : int or array of ints
        Order of the term
    r0 : float
        Detected countrate
    td : float
        Dead time
    tb : float
        Bin time of the light curve
    tau : float
        Inverse of the incident countrate

    Other Parameters
    ----------------
    limit_k : int
        Limit to this value the number of terms in the inner loops of
        calculations. Check the plots returned by  the `check_B` and
        `check_A` functions to test that this number is adequate.
    """
    return _safe_B(k, r0, td, tb, tau, limit_k=limit_k)


def check_B(rate, td, tb, max_k=100, save_to=None, linthresh=0.000001, rate_is_incident=True):
    r"""Check that :math:`B\rightarrow 0` for :math:`k\rightarrow \infty`.

    This function produces a plot of :math:`B_k` vs :math:`k`, to visually check that
    :math:`B_k \rightarrow 0` for :math:`k\rightarrow\infty`, as per Eq. 43 in Zhang+95.

    With this function is possible to determine how many inner loops ``k`` (``limit_k`` in function
    pds_model_zhang) are necessary for a correct approximation of the dead time model

    Parameters
    ----------
    rate : float
        Count rate, either incident or detected (use the ``rate_is_incident`` bool to specify)
    td : float
        Dead time
    tb : float
        Bin time of the light curve

    Other Parameters
    ----------------
    max_k : int
        Maximum k to plot
    save_to : str, default None
        If not None, save the plot to this file
    linthresh : float, default 0.000001
        Linear threshold for the "symlog" scale of the plot
    rate_is_incident : bool, default True
        If True, the input rate is the incident count rate. If False, it is the detected one.
    """
    if rate_is_incident:
        tau = 1 / rate
        r0 = r_det(td, rate)
    else:
        r0 = rate
        tau = 1 / r_in(td, rate)

    fig = plt.figure()
    k_values = np.arange(1, max_k + 1)
    B_values = B(k_values, r0, td, tb, tau, limit_k=max_k)
    threshold = max(tb / td, 1) * 10.0 ** (3 * np.log10(k_values) - PRECISION)

    plt.plot(k_values, threshold, label="Est. propagation of float errors", color="red")
    plt.plot(k_values, -threshold, color="r")

    plt.plot(k_values, B_values, color="k", ds="steps-mid")
    plt.axhline(0, ls="--", color="k")
    plt.semilogx()
    plt.yscale("symlog", linthresh=linthresh)
    plt.xlabel("$k$")
    plt.ylabel("$B_k$")
    plt.xlim([0.9, max_k])
    plt.legend()
    if save_to is not None:
        plt.savefig(save_to)
        plt.close(fig)
    return k_values, B_values


@njit(parallel=True)
def _inner_loop_pds_zhang(N, tau, r0, td, tb, limit_k=60):
    """Calculate the power spectrum, as per Eq. 44 in Zhang+95."""
    # B_k does not depend on the frequency: compute each term once, not once per
    # frequency, which is 40-170x faster for an identical result.
    kmax = min(N, limit_k)
    b = np.zeros(max(kmax, 1))
    for k in prange(kmax):
        b[k] = _safe_B_single_k(k, r0, td, tb, tau, limit_k=limit_k)
    b[0] = _safe_B_single_k(0, r0, td, tb, tau)

    P = np.zeros(N // 2)
    for j in prange(N // 2):
        eq8_sum = 0.0

        for k in range(1, kmax):
            eq8_sum += (N - k) / N * b[k] * np.cos(2 * np.pi * j * k / N)

        P[j] = b[0] + eq8_sum

    return P


def _default_limit_k(f_live, td, tb):
    """Number of inner-sum terms the Zhang model needs, from the live time and binning.

    The dead-time correlations span a fixed *time*, of order ``td / f_live``, while
    ``limit_k`` counts *bins*, so the terms needed grow as ``td / tb``. Mapped over
    ``td / tb`` from 1 to 164 and ``f_live`` from 0.9 to 0.125, the smallest ``limit_k``
    within 0.1% of a converged reference at every frequency is ``c (td / tb) / f_live``
    with ``c`` from 1.5 at ``f_live = 0.9`` to 6.2 at 0.125, the same at every bin time
    from 5 bins per dead time upwards. ``c = 10`` clears that by 1.6 or better.

    Below ``f_live = 0.03`` the sums stop converging at all -- the model can return
    negative powers -- so no number of terms helps; warn and use the value for 0.03.
    """
    if f_live < 0.03:
        warnings.warn(
            f"The live time fraction implied by the detected rate is {f_live:.3g}. "
            "The Zhang model does not converge there: its result is unreliable "
            "whatever limit_k is used."
        )
        f_live = 0.03
    return max(int(np.ceil(10 * max(td / tb, 1) / f_live)), 30)


def pds_model_zhang(N, rate, td, tb, limit_k=None, rate_is_incident=True):
    """Calculate the dead-time-modified power spectrum.

    Parameters
    ----------
    N : int
        The number of spectral bins
    rate : float
        Incident count rate
    td : float
        Dead time
    tb : float
        Bin time of the light curve

    Other Parameters
    ----------------
    limit_k : int, default None
        Limit to this value the number of terms in the inner loops of
        calculations. If None, ``ceil(10 * max(td / tb, 1) / f_live)`` with a floor of
        30: the terms needed grow with the number of bins per dead time and as the live
        time fraction falls. It used to be a flat 60, which is 10% wrong at 41 bins per
        dead time and a factor 2.5 at 164. Check the plots returned by the `check_B` and
        `check_A` functions to test that this number is adequate.
    rate_is_incident : bool, default True
        If True, the input rate is the incident count rate. If False, it is the
        detected count rate.

    Returns
    -------
    freqs : array of floats
        Frequency array
    power : array of floats
        Power spectrum
    """
    if rate_is_incident:
        tau = 1 / rate
        r0 = r_det(td, rate)
    else:
        r0 = rate
        tau = 1 / r_in(td, rate)

    if limit_k is None:
        limit_k = _default_limit_k(1 - r0 * td, td, tb)

    # Nph = N / tau
    logger.info("Calculating PDS model (update)")
    P = _inner_loop_pds_zhang(N, tau, r0, td, tb, limit_k=limit_k)
    if tb > 10 * td:
        warnings.warn(
            f"The bin time is much larger than the dead time. "
            f" Calculations might be slow. tb={tb / td:.2f} * td"
        )

    maxf = 0.5 / tb
    df = maxf / len(P)
    freqs = np.arange(len(P)) * df

    return freqs, P


def non_paralyzable_dead_time_model(
    freqs,
    dead_time,
    rate,
    bin_time=None,
    limit_k=None,
    background_rate=0.0,
    n_approx=None,
):
    """Calculate the dead-time-modified power spectrum.

    Parameters
    ----------
    freqs : array of floats
        Frequency array
    dead_time : float
        Dead time
    rate : float
        Detected source count rate

    Other Parameters
    ----------------
    bin_time : float
        Bin time of the light curve
    limit_k : int, default None
        Limit to this value the number of terms in the inner loops of calculations. The
        dead-time correlations span a fixed time, of order ``dead_time / f_live``, so the
        number of terms needed grows with the number of bins per dead time and as the live
        time fraction falls. If None, ``ceil(10 * max(dead_time / bin_time, 1) / f_live)``
        is used, with a floor of 30, where ``f_live = 1 - (rate + background_rate) *
        dead_time`` is the live time fraction implied by the detected rate. That clears the
        measured requirement by a factor 1.6 or better from ``f_live = 0.9`` to 0.125 and
        from 1 to 164 bins per dead time. Check the plots returned by the `check_B` and
        `check_A` functions to test that the number is adequate.
    background_rate : float, default 0
        Detected background count rate. This is important to estimate when deadtime is given by the
        combination of the source counts and background counts (e.g. in an imaging X-ray detector).
    n_approx : int, default None
        Number of bins used to calculate the model power spectrum, which is then interpolated
        onto ``freqs``. The model grid always spans 0 to ``0.5 / bin_time``, so this sets its
        frequency resolution. If None, one bin per Hz is used, which is adequate because the
        model is smooth on the scale of ``1 / dead_time``. Relatively simple models (e.g. low
        count rates compared to dead time) can use a smaller number to speed up the
        calculation. Note that the value does *not* depend on ``freqs``: the model of a given
        detector is the same whether it is evaluated on a full Fourier grid or on a single
        narrow band.

    Returns
    -------
    power : array of floats
        Power spectrum
    """
    if rate + background_rate > 1 / dead_time:
        raise ValueError(
            "The sum of the source and background count rates is larger than the inverse "
            "of the dead time. This is not a physical situation. Please check your input."
        )

    if bin_time is None:
        bin_time = 1 / (2 * max(freqs))

    if limit_k is None:
        f_live = 1 - (rate + background_rate) * dead_time
        limit_k = _default_limit_k(f_live, dead_time, bin_time)

    # The Zhang grid always spans 0 to the Nyquist frequency of ``bin_time``, so
    # ``n_bins`` fixes the model's frequency resolution at ``0.5 / (bin_time * n_bins)``.
    # It therefore has to be set by ``bin_time`` alone. Deriving it from ``max(freqs)``
    # makes the resolution depend on *which* frequencies are asked for: on a band
    # restricted to 0.5-10 Hz of a 1024 Hz grid it left ten model bins to span the
    # whole 1024 Hz, a 25% error. One bin per Hz reproduces exactly what a full Fourier
    # grid got before, because there ``max(freqs)`` *is* the Nyquist frequency, and the
    # model is smooth on the scale of ``1 / dead_time``.
    if n_approx is not None:
        n_bins = n_approx
    else:
        n_bins = max(int(round(0.5 / bin_time)), 10)

    zh_f, zh_p = pds_model_zhang(
        int(n_bins) * 2,
        (rate + background_rate),
        dead_time,
        bin_time,
        limit_k=limit_k,
        rate_is_incident=False,
    )

    # Rescale by the source rate wrt background rate
    if background_rate > 0:
        zh_p = (zh_p - 2) * rate / (rate + background_rate) + 2

    pds_interp = interp1d(zh_f, zh_p, bounds_error=False, fill_value="extrapolate")
    return pds_interp(freqs)

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from .fourier import integrate_power_in_frequency_range
from .utils import force_array
from collections.abc import Iterable

DEFAULT_COLOR_CONFIGURATION = {
    "center": [4.51920, 0.453724],
    "ref_angle": 3 * np.pi / 4,
    "state_definitions": {
        "HSS": {"hue_limits": [300, 360], "color": "red"},
        "LHS": {"hue_limits": [-20, 140], "color": "blue"},
        "HIMS": {"hue_limits": [140, 220], "color": "green"},
        "SIMS": {"hue_limits": [220, 300], "color": "yellow"},
    },
    "rms_spans": {
        -20: [0.3, 0.7],
        0: [0.3, 0.7],
        10: [0.3, 0.6],
        40: [0.25, 0.4],
        100: [0.25, 0.35],
        150: [0.2, 0.3],
        170: [0.0, 0.3],
        200: [0, 0.15],
        370: [0, 0.15],
    },
}


def _get_rms_span_functions(configuration=DEFAULT_COLOR_CONFIGURATION):
    rms_spans = configuration["rms_spans"]

    x = list(rms_spans.keys())

    ymin = list([v[0] for v in rms_spans.values()])
    ymax = list([v[1] for v in rms_spans.values()])

    ymin_func = interp1d(x, ymin, kind="linear")
    ymax_func = interp1d(x, ymax, kind="linear")
    return ymin_func, ymax_func


def _create_rms_hue_plot(polar=False, plot_spans=False, configuration=DEFAULT_COLOR_CONFIGURATION):
    if polar:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.set_rmax(0.75)
        ax.set_rticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(True)
        ax.set_rlim([0, 0.75])
    else:
        plt.figure()
        plt.xlim(0, 360)
        plt.ylim(0, 0.7)
        plt.ylabel("Fractional rms")
        plt.xlabel("Hue")
        ax = plt.gca()

    if not plot_spans:
        return ax

    ymin_func, ymax_func = _get_rms_span_functions(configuration)

    for state in configuration["state_definitions"].keys():
        color = configuration["state_definitions"][state]["color"]
        xmin, xmax = configuration["state_definitions"][state]["hue_limits"]

        def x_func(x):
            return x

        if polar:

            def x_func(x):
                return np.radians(x)

        ax.fill_between(
            np.linspace(x_func(xmin), x_func(xmax), 20),
            ymin_func(np.linspace(xmin, xmax, 20)),
            ymax_func(np.linspace(xmin, xmax, 20)),
            color=color,
            alpha=0.1,
        )

        if xmin < 0 and not polar:
            ax.fill_between(
                np.linspace(x_func(xmin + 360), x_func(360), 20),
                ymin_func(np.linspace(xmin, 0, 20)),
                ymax_func(np.linspace(xmin, 0, 20)),
                color=color,
                alpha=0.1,
            )
        if xmax > 360 and not polar:
            ax.fill_between(
                np.linspace(0, x_func(xmax - 360), 20),
                ymin_func(np.linspace(360, xmax, 20)),
                ymax_func(np.linspace(360, xmax, 20)),
                color=color,
                alpha=0.1,
            )

    return ax


def _limit_angle_to_360(angle):
    while angle >= 360:
        angle = angle - 360
    while angle < 0:
        angle = angle + 360
    return angle


def _hue_line_data(center, angle, ref_angle=3 * np.pi / 4):
    plot_angle = (-angle + ref_angle) % (np.pi * 2)

    m = np.tan(plot_angle)
    if np.isinf(m):
        x = np.zeros_like(x) + center[0]
        y = np.linspace(-4, 4, 20)
    else:
        x = np.linspace(0, 4, 20) * np.sign(np.cos(plot_angle)) + center[0]
        y = center[1] + m * (x - center[0])
    return x, y


def _trace_states(ax, configuration=DEFAULT_COLOR_CONFIGURATION, **kwargs):
    center = [np.log10(c) for c in configuration["center"]]
    for state in configuration["state_definitions"].keys():
        color = configuration["state_definitions"][state]["color"]
        hue0, hue1 = configuration["state_definitions"][state]["hue_limits"]
        hue_mean = (hue0 + hue1) / 2
        hue_angle = (-np.radians(hue_mean) + 3 * np.pi / 4) % (np.pi * 2)

        radius = 1.4
        txt_x = radius * np.cos(hue_angle) + center[0]
        txt_y = radius * np.sin(hue_angle) + center[1]
        ax.text(txt_x, txt_y, state, color="k", ha="center", va="center")

        hue0, hue1 = configuration["state_definitions"][state]["hue_limits"]
        x0, y0 = _hue_line_data(center, np.radians(hue0), ref_angle=configuration["ref_angle"])

        next_angle = hue0 + 5.0
        x1, y1 = _hue_line_data(center, np.radians(hue0), ref_angle=configuration["ref_angle"])

        while next_angle <= hue1:
            x0, y0 = x1, y1
            x1, y1 = _hue_line_data(
                center, np.radians(next_angle), ref_angle=configuration["ref_angle"]
            )
            t1 = plt.Polygon(
                [[x0[0], y0[0]], [x0[-1], y0[-1]], [x1[-1], y1[-1]]],
                ls=None,
                lw=0,
                color=color,
                **kwargs,
            )
            ax.add_patch(t1)
            next_angle += 5.0


def _create_pc_plot(
    xrange=[-2, 2],
    yrange=[-2, 2],
    plot_spans=False,
    configuration=DEFAULT_COLOR_CONFIGURATION,
):
    """Creates an empty power color plot with labels in the right place."""
    fig = plt.figure()
    ax = plt.gca()

    ax.set_aspect("equal")

    ax.set_xlabel(r"log$_{10}$PC1")
    ax.set_ylabel(r"log$_{10}$PC2")
    ax.grid(False)

    if not plot_spans:
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        return ax

    center = np.log10(np.asanyarray(configuration["center"]))
    ax.set_xlim(center[0] + np.asanyarray(xrange))
    ax.set_ylim(center[1] + np.asanyarray(yrange))

    for angle in range(0, 360, 20):
        x, y = _hue_line_data(center, np.radians(angle), ref_angle=configuration["ref_angle"])

        ax.plot(x, y, lw=0.2, ls=":", color="k", alpha=0.3, zorder=10)

    ax.scatter(*center, marker="+", color="k")

    limit_angles = set(
        np.array(
            [
                configuration["state_definitions"][state]["hue_limits"]
                for state in configuration["state_definitions"].keys()
            ]
        ).flatten()
    )
    limit_angles = [_limit_angle_to_360(angle) for angle in limit_angles]

    for angle in limit_angles:
        x, y = _hue_line_data(center, np.radians(angle), ref_angle=configuration["ref_angle"])

        plt.plot(x, y, lw=1, ls=":", color="k", alpha=1, zorder=10)

    _trace_states(ax, configuration=configuration, alpha=0.1)

    return ax


def plot_power_colors(
    p1,
    p1e,
    p2,
    p2e,
    plot_spans=False,
    configuration=DEFAULT_COLOR_CONFIGURATION,
):
    """
    Plot power colors.

    Parameters
    ----------
    p1 : float
        The first power value.
    p1e : float
        The error in the first power value.
    p2 : float
        The second power value.
    p2e : float
        The error in the second power value.

    Other Parameters
    ----------------
    center : (float, float)
        The center coordinates of the plot. Default is (4.51920, 0.453724).
    plot_spans : bool
        Whether to plot the spans. Default is False.
    configuration: bool
        The color configuration to use. Default is DEFAULT_COLOR_CONFIGURATION.

    Returns
    -------
    ax : :class:`matplotlib.AxesSubplot`
        The matplotlib Axes object representing the power color plot.
    """
    p1e = 1 / p1 * p1e
    p2e = 1 / p2 * p2e
    p1 = np.log10(p1)
    p2 = np.log10(p2)
    # Create empty power color plot
    ax = _create_pc_plot(plot_spans=plot_spans, configuration=configuration)
    ax.errorbar(p1, p2, xerr=p1e, yerr=p2e, alpha=0.4, color="k")
    ax.scatter(p1, p2, zorder=10, color="k")
    return ax


def plot_hues(
    rms,
    rmse,
    pc1,
    pc2,
    polar=False,
    plot_spans=False,
    configuration=DEFAULT_COLOR_CONFIGURATION,
):
    hues = hue_from_power_color(force_array(pc1), force_array(pc2))

    ax = _create_rms_hue_plot(polar=polar, plot_spans=plot_spans, configuration=configuration)
    hues = hues % (np.pi * 2)
    if not polar:
        hues = np.degrees(hues)

    ax.errorbar(hues, rms, yerr=rmse, fmt="o", alpha=0.5)
    return ax


def power_color(
    frequency,
    power,
    power_err=None,
    freq_edges=[1 / 256, 1 / 32, 0.25, 2.0, 16.0],
    df=None,
    m=1,
    freqs_to_exclude=None,
    poisson_power=0,
    return_log=False,
):
    """
    Calculate two power colors from a power spectrum.

    Power colors are an alternative to spectral colors to understand the spectral state of an
    accreting source. They are defined as the ratio of the power in two frequency ranges,
    analogously to the colors calculated from electromagnetic spectra.

    This function calculates two power colors, using the four frequency ranges contained
    between the five frequency edges in ``freq_edges``. Given [f0, f1, f2, f3, f4], the
    two power colors are calculated as the following ratios of the integrated power
    (which are variances):

    + PC0 = Var([f0, f1]) / Var([f2, f3])

    + PC1 = Var([f1, f2]) / Var([f3, f4])

    Errors are calculated using simple error propagation from the integrated power errors.

    See Heil et al. 2015, MNRAS, 448, 3348

    Parameters
    ----------
    frequency : iterable
        The frequencies of the power spectrum
    power : iterable
        The power at each frequency

    Other Parameters
    ----------------
    power_err : iterable
        The power error bar at each frequency
    freq_edges : iterable, optional, default ``[0.0039, 0.031, 0.25, 2.0, 16.0]``
        The five edges defining the four frequency intervals to use to calculate the power color.
        If empty, the power color is calculated using the frequencies from Heil et al. 2015.
    df : float or float iterable, optional, default None
        The frequency resolution of the input data. If None, it is calculated
        from the median difference of input frequencies.
    m : int, optional, default 1
        The number of segments and/or contiguous frequency bins averaged to obtain power
    freqs_to_exclude : 1-d or 2-d iterable, optional, default None
        The ranges of frequencies to exclude from the calculation of the power color.
        For example, the frequencies containing strong QPOs.
        A 1-d iterable should contain two values for the edges of a single range. (E.g.
        ``[0.1, 0.2]``). ``[[0.1, 0.2], [3, 4]]`` will exclude the ranges 0.1-0.2 Hz and 3-4 Hz.
    poisson_power : float or iterable, optional, default 0
        The Poisson noise level of the power spectrum. If iterable, it should have the same
        length as ``frequency``. (This might apply to the case of a power spectrum with a
        strong dead time distortion
    return_log : bool, optional, default False
        Return the base-10 logarithm of the power color and the errors

    Returns
    -------
    PC0 : float
        The first power color
    PC0_err : float
        The error on the first power color
    PC1 : float
        The second power color
    PC1_err : float
        The error on the second power color
    """
    freq_edges = np.asanyarray(freq_edges)
    if len(freq_edges) != 5:
        raise ValueError("freq_edges must have 5 elements")

    frequency = np.asanyarray(frequency)
    power = np.asanyarray(power)

    if df is None:
        df = np.median(np.diff(frequency))
    input_frequency_low_edges = frequency - df / 2
    input_frequency_high_edges = frequency + df / 2

    if freq_edges.min() < input_frequency_low_edges[0]:
        raise ValueError("The minimum frequency is larger than the first frequency edge")
    if freq_edges.max() > input_frequency_high_edges[-1]:
        raise ValueError("The maximum frequency is lower than the last frequency edge")

    if power_err is None:
        power_err = power / np.sqrt(m)
    else:
        power_err = np.asanyarray(power_err)

    if freqs_to_exclude is not None:
        if len(np.shape(freqs_to_exclude)) == 1:
            freqs_to_exclude = [freqs_to_exclude]

        if (
            not isinstance(freqs_to_exclude, Iterable)
            or len(np.shape(freqs_to_exclude)) != 2
            or np.shape(freqs_to_exclude)[1] != 2
        ):
            raise ValueError("freqs_to_exclude must be of format [[f0, f1], [f2, f3], ...]")
        for f0, f1 in freqs_to_exclude:
            frequency_mask = (input_frequency_low_edges > f0) & (input_frequency_high_edges < f1)
            idx0, idx1 = np.searchsorted(frequency, [f0, f1])
            power[frequency_mask] = np.mean([power[idx0], power[idx1]])

    var00, var00_err = integrate_power_in_frequency_range(
        frequency,
        power,
        freq_edges[:2],
        power_err=power_err,
        df=df,
        m=m,
        poisson_power=poisson_power,
    )
    var01, var01_err = integrate_power_in_frequency_range(
        frequency,
        power,
        freq_edges[2:4],
        power_err=power_err,
        df=df,
        m=m,
        poisson_power=poisson_power,
    )
    var10, var10_err = integrate_power_in_frequency_range(
        frequency,
        power,
        freq_edges[1:3],
        power_err=power_err,
        df=df,
        m=m,
        poisson_power=poisson_power,
    )
    var11, var11_err = integrate_power_in_frequency_range(
        frequency,
        power,
        freq_edges[3:5],
        power_err=power_err,
        df=df,
        m=m,
        poisson_power=poisson_power,
    )
    pc0 = var00 / var01
    pc1 = var10 / var11
    pc0_err = pc0 * (var00_err / var00 + var01_err / var01)
    pc1_err = pc1 * (var10_err / var10 + var11_err / var11)
    if return_log:
        pc0_err = 1 / pc0 * pc0_err
        pc1_err = 1 / pc1 * pc1_err
        pc0 = np.log10(pc0)
        pc1 = np.log10(pc1)
    return pc0, pc0_err, pc1, pc1_err


def hue_from_power_color(pc0, pc1, center=[4.51920, 0.453724]):
    """Measure the angle of a point in the log-power color diagram wrt the center.

    Angles are measured in radians, **in the clockwise direction**, with respect to a line oriented
    at -45 degrees wrt the horizontal axis.

    See Heil et al. 2015, MNRAS, 448, 3348

    Parameters
    ----------
    pc0 : float
        The (linear, not log!) power color in the first frequency range
    pc1 : float
        The (linear, not log!) power color in the second frequency range

    Other Parameters
    ----------------
    center : iterable, optional, default [4.51920, 0.453724]
        The coordinates of the center of the power color diagram

    Returns
    -------
    hue : float
        The angle of the point wrt the center, in radians
    """
    pc0 = np.log10(pc0)
    pc1 = np.log10(pc1)

    center = np.log10(np.asanyarray(center))

    return hue_from_logpower_color(pc0, pc1, center=center)


def hue_from_logpower_color(log10pc0, log10pc1, center=(np.log10(4.51920), np.log10(0.453724))):
    """Measure the angle of a point in the log-power color diagram wrt the center.

    Angles are measured in radians, **in the clockwise direction**, with respect to a line oriented
    at -45 degrees wrt the horizontal axis.

    See Heil et al. 2015, MNRAS, 448, 3348

    Parameters
    ----------
    log10pc0 : float
        The log10 power color in the first frequency range
    log10pc1 : float
        The log10 power color in the second frequency range

    Other Parameters
    ----------------
    center : iterable, optional, default ``log10([4.51920, 0.453724])``
        The coordinates of the center of the power color diagram

    Returns
    -------
    hue : float
        The angle of the point wrt the center, in radians
    """
    hue = 3 / 4 * np.pi - np.arctan2(log10pc1 - center[1], log10pc0 - center[0])
    return hue

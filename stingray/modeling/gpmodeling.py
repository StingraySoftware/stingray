import numpy as np
import matplotlib.pyplot as plt
import functools
from stingray import Lightcurve

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, random

    jax.config.update("jax_enable_x64", True)
    jax_avail = True
except ImportError:
    jax_avail = False

try:
    from tinygp import GaussianProcess, kernels

    can_make_gp = True
except ImportError:
    can_make_gp = False

try:
    from jaxns import DefaultNestedSampler, TerminationCondition, Prior, Model
    from jaxns.utils import resample

    can_sample = True
except ImportError:
    can_sample = False
try:
    import tensorflow_probability.substrates.jax as tfp

    tfpd = tfp.distributions
    tfpb = tfp.bijectors
    tfp_available = True
except ImportError:
    tfp_available = False


__all__ = [
    "get_kernel",
    "get_mean",
    "get_prior",
    "get_log_likelihood",
    "GPResult",
    "get_gp_params",
    "run_prior_checks",
    "run_posterior_check",
]


def get_priors_samples(key, kernel_params, priors, loglike, n_samples=3000):
    """Sample from the prior distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    priors : list
        List of priors.
    loglike : callable
        Log likelihood function.
    n_samples : int
        Number of samples. Default is 3000.
    """

    num_params = len(kernel_params)

    # get the prior model
    prior_dict = dict(zip(kernel_params, priors))
    prior_model = get_prior(kernel_params, prior_dict)

    # define the model
    nsmodel = Model(prior_model=prior_model, log_likelihood=loglike)
    # nsmodel.sanity_check(key=jax.random.PRNGKey(0), S=1)

    # get the samples
    unit_samples = jax.random.uniform(key, (n_samples, num_params))
    prior_samples = jax.vmap(nsmodel.transform)(unit_samples)

    return prior_samples


def get_psd_and_approx(
    kernel_type,
    kernel_params,
    prior_samples,
    f0,
    fM,
    n_frequencies=1000,
    n_approx_components=20,
    approximate_with="SHO",
    with_normalisation=False,
):  # -> tuple[NDArray[Any], NDArray[Any]]:
    """Get the PSD and the approximate PSD for a given set of parameters and samples.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    kernel_params : list[str]
        List of kernel parameters.
    prior_samples : NDArray
        Prior samples.
    f0 : float
        Minimum frequency.
    fM : float
        Maximum frequency.
    n_frequencies : int
        Number of frequencies.

    """
    n_samples = prior_samples[kernel_params[0]].shape[0]
    f = np.geomspace(f0, fM, n_frequencies)
    psd_models = []
    psd_approx = []
    for k in range(n_samples):
        param_dict = {}
        for i, params in enumerate(kernel_params):
            if params[0:4] == "log_":
                param_dict[params[4:]] = jnp.exp(prior_samples[params][k])
            else:
                param_dict[params] = prior_samples[params][k]

        psd_model, psd_SHO = get_psd_approx_samples(
            f,
            kernel_type,
            param_dict,
            f0,
            fM,
            n_approx_components=n_approx_components,
            approximate_with=approximate_with,
        )
        if with_normalisation:
            f_c, a = _get_coefficients_approximation(
                kernel_type,
                param_dict,
                f0,
                fM,
                n_approx_components=n_approx_components,
                approximate_with=approximate_with,
            )
            norm = np.sum(a * f_c)
            psd_models.append(psd_model * param_dict["variance"] / norm)
            psd_approx.append(psd_SHO * param_dict["variance"] / norm)
        else:
            psd_models.append(psd_model)
            psd_approx.append(psd_SHO)
    psd_models = np.array(psd_models)
    psd_approx = np.array(psd_approx)
    return f, psd_models, psd_approx


def plot_psd_ppc(f, psd_quantiles, psd_approx_quantiles, psd_noise_levels, f_min, f_max, path):
    """Replot the PSD PPC plot.

    Parameters
    ----------
    f : array
        The frequency array.
    psd_quantiles : array
        The quantiles of the PSD model.
    psd_approx_quantiles : array
        The quantiles of the PSD approximation.
    psd_noise_levels : array
        The noise levels.
    f_min : float
        The minimum frequency.
    f_max : float
        The maximum frequency.
    path : str
        The path to save the figure.
    """
    approx_color = "C6"
    psd_color = "C3"
    noise_color = "C5"
    window_color = "k"

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.loglog(f, psd_quantiles[:, 2], label="Median", color=psd_color)
    ax.fill_between(f, psd_quantiles[:, 1], psd_quantiles[:, 3], color=psd_color, alpha=0.3)
    ax.fill_between(f, psd_quantiles[:, 0], psd_quantiles[:, 4], color=psd_color, alpha=0.15)
    ax.axhline(psd_noise_levels[0], color=noise_color, ls="-", label="Noise level")
    ax.loglog(f, psd_approx_quantiles[:, 2], color=approx_color)
    ax.fill_between(
        f,
        psd_approx_quantiles[:, 1],
        psd_approx_quantiles[:, 3],
        color=approx_color,
        alpha=0.3,
    )
    ax.fill_between(
        f,
        psd_approx_quantiles[:, 0],
        psd_approx_quantiles[:, 4],
        color=approx_color,
        alpha=0.15,
    )
    ax.axvline(f_min, color=window_color, ls=":")
    ax.axvline(f_max, color=window_color, ls=":")
    ax.update({"xlabel": r"Frequency $(d^{-1})$", "ylabel": "Power Spectral Density"})
    ax.set_xlim(np.min(f), np.max(f) / 10)
    ax.set_ylim(np.min(psd_noise_levels) / 10)

    legend_elements = [
        Line2D([0], [0], color=psd_color, lw=2, label="PSD model"),
        Line2D([0], [0], color=approx_color, lw=2, label="PSD approximation"),
        Line2D([0], [0], color=noise_color, lw=2, label="Noise level"),
        Patch(facecolor="k", edgecolor="k", alpha=0.1, label="95%"),
        Patch(facecolor="k", edgecolor="k", alpha=0.4, label="68%"),
        Line2D(
            [0], [0], color=window_color, lw=2, ls=":", label=r"$f_\mathrm{min}, f_\mathrm{max}$"
        ),
    ]

    ax.legend(
        handles=legend_elements,
        ncol=2,
        bbox_to_anchor=(0.5, -0.175),
        loc="lower center",
        bbox_transform=fig.transFigure,
    )
    fig.tight_layout()
    fig.savefig(f"{path}replot_psd_ppc.pdf", bbox_inches="tight")

    return fig


def run_posterior_check(
    kernel_type,
    kernel_params,
    posterior_dict,
    t,
    y,
    yerr,
    S_low=20,
    S_high=20,
    n_frequencies=1000,
    n_approx_components=20,
    approximate_with="SHO",
    path="./",
):

    f_min, f_max = 1 / (t[-1] - t[0]), 1 / (2 * np.min(np.diff(t)))
    f0, fM = f_min / S_low, f_max * S_high
    f, psd_models, psd_approx = get_psd_and_approx(
        kernel_type,
        kernel_params,
        posterior_dict,
        f0,
        fM,
        n_frequencies=n_frequencies,
        n_approx_components=n_approx_components,
        approximate_with=approximate_with,
        with_normalisation=True,
    )

    if "log_shift" in kernel_params:
        psd_noise_levels = [2 * np.median((yerr / y) ** 2) * np.median(np.diff(t))]
    else:
        psd_noise_levels = [2 * np.median((yerr) ** 2) * np.median(np.diff(t))]
    psd_quantiles = jnp.percentile(psd_models, jnp.array([2.5, 16, 50, 84, 97.5]), axis=0).T
    psd_approx_quantiles = jnp.percentile(psd_approx, jnp.array([2.5, 16, 50, 84, 97.5]), axis=0).T

    fig = plot_psd_ppc(
        f, psd_quantiles, psd_approx_quantiles, psd_noise_levels, f_min, f_max, path=path
    )


def run_prior_checks(
    kernel_type,
    kernel_params,
    priors,
    loglike,
    f_min,
    f_max,
    seed=42,
    path="./",
    n_samples=3000,
    n_frequencies=1000,
    S_low=20,
    S_high=20,
    n_approx_components=20,
    approximate_with="SHO",
):
    """Check the approximation of the power spectrum density.
    This function will plot various diagnostics on the residuals and the ratio
    of the PSD and the approximate PSD.

    Parameters
    ----------
    kernel_type : str
        The type of kernel to be used for the Gaussian Process
        Only designed for the following Power spectra ["PowL","DoubPowL]
    kernel_params : dict
        Dictionary containing the parameters for the kernel
        Should contain the parameters for the selected kernel
    priors : list
        List of priors.
    loglike : callable
        Log likelihood function.
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    path : str
        Path to save the plots. Default is "./".
    n_samples : int
        Number of samples. Default is 3000.
    n_frequencies : int
        Number of frequencies. Default is 1000.
    key : jax.random.PRNGKey
        Random key. Default is jax.random.PRNGKey(42).
    S_low : int
        Low frequency scaling factor. Default is 20.
    S_high : int
        High frequency scaling factor. Default is 20.
    """
    key = jax.random.PRNGKey(seed)
    if kernel_type not in ["PowL", "DoubPowL"]:
        raise ValueError("Only 'PowL' and 'DoubPowL' kernels need to be checked")

    f0, fM = f_min / S_low, f_max * S_high
    prior_samples = get_priors_samples(key, kernel_params, priors, loglike, n_samples)
    f, psd_models, psd_approx = get_psd_and_approx(
        kernel_type,
        kernel_params,
        prior_samples,
        f0,
        fM,
        n_frequencies=n_frequencies,
        n_approx_components=n_approx_components,
        approximate_with=approximate_with,
    )

    residuals = psd_approx - psd_models
    ratio = np.exp(np.log(psd_approx) - np.log(psd_models))

    fig1, _ = plot_psd_approx_quantiles(f, f_min, f_max, residuals, ratio)
    fig1.savefig(f"{path}/psd_check_approx_quantiles.png", dpi=300)

    fig2, _ = plot_boxplot_psd_approx(residuals, ratio)
    fig2.savefig(f"{path}/psd_check_approx_boxplot.png", dpi=300)
    return fig1, fig2


def plot_boxplot_psd_approx(residuals, ratios):
    """Plot the boxplot of the residuals and the ratios of the PSD and the approximate PSD."""
    meta_mean_res = np.mean(residuals, axis=1)
    meta_median_res = np.median(residuals, axis=1)
    meta_min_res = np.min(np.abs(residuals), axis=1)
    meta_max_res = np.max(np.abs(residuals), axis=1)

    meta_mean_rat = np.mean(ratios, axis=1)
    meta_median_rat = np.median(ratios, axis=1)
    meta_min_rat = np.min(ratios, axis=1)
    meta_max_rat = np.max(ratios, axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(7, 5.5))
    ax[0].boxplot(
        [meta_mean_res, meta_median_res, meta_min_res, meta_max_res],
        positions=[1, 2, 3, 4],
        flierprops=dict(marker=".", markersize=3),
    )
    ax[0].set_xticks([])
    ax[0].axhline(0, color="C2", lw=2, ls=":")

    ax[1].boxplot(
        [meta_mean_rat, meta_median_rat, meta_min_rat, meta_max_rat],
        positions=[1, 2, 3, 4],
        flierprops=dict(marker=".", markersize=3),
    )
    ax[1].set_xticks([1, 2, 3, 4])
    ax[1].axhline(1, color="C2", lw=2, ls=":")
    ax[1].set_xticklabels(["mean", "median", "minimum", "maximum"])
    ax[0].set_ylabel(r"$P_{\text{true}} - P_{\text{approx}} $")
    ax[1].set_ylabel(r"$P_{\text{approx}} / P_{\text{true}} $")
    fig.align_ylabels(ax)
    # fig.tight_layout()
    return fig, ax


def plot_psd_approx_quantiles(f, f_min, f_max, residuals, ratios):
    """Plot the quantiles of the residuals and the ratios of the PSD and
    the approximate PSD as a function of frequency."""
    res_quantiles = jnp.percentile(residuals, jnp.asarray([2.5, 16, 50, 84, 97.5]), axis=0)
    rat_quantiles = jnp.percentile(ratios, jnp.asarray([2.5, 16, 50, 84, 97.5]), axis=0)

    colors = "C0"
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6.5, 4.5), gridspec_kw={"hspace": 0.1})

    ax[0].fill_between(f, res_quantiles[0], res_quantiles[4], color=colors, alpha=0.25)
    ax[0].fill_between(f, res_quantiles[1], res_quantiles[3], color=colors, alpha=0.5)
    ax[0].plot(f, res_quantiles[2], color="black", lw=1)
    ax[0].update(
        {
            "xscale": "log",
            "yscale": "linear",
            "ylabel": r"$P_{\text{true}} - P_{\text{approx}} $",
        }
    )

    ax[0].axvline(f_min, color="black", linestyle="--")
    ax[0].axvline(f_max, color="black", linestyle="--")
    ax[1].axvline(f_min, color="black", linestyle="--")
    ax[1].axvline(f_max, color="black", linestyle="--")

    ax[1].fill_between(f, rat_quantiles[0], rat_quantiles[4], color=colors, alpha=0.25)
    ax[1].fill_between(f, rat_quantiles[1], rat_quantiles[3], color=colors, alpha=0.5)
    ax[1].plot(f, rat_quantiles[2], color="black", lw=1)
    ax[1].update(
        {
            "xscale": "log",
            "yscale": "linear",
            "xlabel": "Frequency",
            "ylabel": r"${P_{\text{approx}}}/{P_{\text{true}}}$",
        }
    )

    legend_elements = [
        Line2D([0], [0], color=colors, lw=2, label="Median"),
        Line2D([0], [0], color="k", lw=1, ls="--", label=r"$f_\mathrm{min}, f_\mathrm{max}$"),
        Patch(facecolor=colors, edgecolor=colors, alpha=0.25, label="95%"),
        Patch(facecolor=colors, edgecolor=colors, alpha=0.5, label="68%"),
    ]
    ax[1].legend(handles=legend_elements, ncol=3, bbox_to_anchor=(1.0, -0.4))
    fig.align_ylabels(ax)
    # fig.tight_layout()
    return fig, ax


def SHO_power_spectrum(f, A, f0):
    """Power spectrum of a stochastic harmonic oscillator.

    Parameters
    ----------
    f : jax.Array
        Frequency array.
    A : float
        Amplitude.
    f0 : float
        Position.
    """
    P = A / (1 + jnp.power((f / f0), 4))

    return P


def _get_coefficients_approximation(
    kernel_type, kernel_params, f_min, f_max, n_approx_components=20, approximate_with="SHO"
):
    """
    Get the coefficients of the approximation of the power law kernel
    with a sum of SHO kernels or a sum of DRW+SHO kernels.

    Parameters
    ----------
    kernel_type: string
        The type of kernel to be used for the Gaussian Process
        Only designed for the following Power spectra ["PowL","DoubPowL]
    kernel_params: dict
        Dictionary containing the parameters for the kernel
        Should contain the parameters for the selected kernel
    f_min: float
        The minimum frequency of the approximation grid.
        Should be the lowest frequency of the power spectrum
    f_max: float
        The maximum frequency of the approximation grid.
        Should be the highest frequency of the power spectrum
    n_approx_components: int
        The number of components to use to approximate the power law
    approximate_with: string
        The type of kernel to use to approximate the power law power spectra
    """
    # grid of frequencies for the approximation
    spectral_points = jnp.geomspace(f_min, f_max, n_approx_components)
    # build the matrix of the approximation
    if approximate_with == "SHO":
        spectral_matrix = 1 / (
            1 + jnp.power(jnp.atleast_2d(spectral_points).T / spectral_points, 4)
        )
    else:
        raise NotImplementedError(f"Approximation {approximate_with} not implemented")

    # get the psd values and normalize them to the first value
    psd_values = _psd_model(kernel_type, kernel_params)(spectral_points)
    psd_values /= psd_values[0]
    # compute the coefficients of the approximation
    spectral_coefficients = jnp.linalg.solve(spectral_matrix, psd_values)
    return spectral_points, spectral_coefficients


def get_psd_approx_samples(
    f, kernel_type, kernel_params, f_min, f_max, n_approx_components=20, approximate_with="SHO"
):
    """Get the true PSD model and the approximated PSD using SHO decomposition.

    Parameters
    ----------
    f : jax.Array
        Frequency array.
    kernel_type: string
        The type of kernel to be used for the Gaussian Process
        Only designed for the following Power spectra ["PowL","DoubPowL]
    kernel_params: dict
        Dictionary containing the parameters for the kernel
        Should contain the parameters for the selected kernel
    f_min: float
        The minimum frequency of the approximation grid.
    f_max: float
        The maximum frequency of the approximation grid.
    n_approx_components: int
        The number of components to use to approximate the power law
        must be greater than 2, default 20
    approximate_with: string
        The type of kernel to use to approximate the power law power spectra
        Default is "SHO"
    """
    f_c, a = _get_coefficients_approximation(
        kernel_type,
        kernel_params,
        f_min,
        f_max,
        n_approx_components=n_approx_components,
        approximate_with=approximate_with,
    )
    psd_SHO = SHO_power_spectrum(f, a[..., None], f_c[..., None]).sum(axis=0)

    psd_model = _psd_model(kernel_type, kernel_params)(f)
    psd_model /= psd_model[..., 0, None]
    return psd_model, psd_SHO


def _approximate_powerlaw(
    kernel_type, kernel_params, f_min, f_max, n_approx_components=20, approximate_with="SHO"
):
    """
    Approximate the power law kernel with a sum of SHO kernels or a sum of DRW+SHO kernels.

    Parameters
    ----------
    kernel_type: string
        The type of kernel to be used for the Gaussian Process
        Only designed for the following Power spectra ["PowL","DoubPowL]
    kernel_params: dict
        Dictionary containing the parameters for the kernel
        Should contain the parameters for the selected kernel
    f_min: float
        The minimum frequency of the approximation grid.
        Should be the lowest frequency of the power spectrum
    f_max: float
        The maximum frequency of the approximation grid.
        Should be the highest frequency of the power spectrum
    n_approx_components: int
        The number of components to use to approximate the power law
    approximate_with: string
        The type of kernel to use to approximate the power law power spectra
        Default is "SHO"
    """
    spectral_points, spectral_coefficients = _get_coefficients_approximation(
        kernel_type,
        kernel_params,
        f_min,
        f_max,
        n_approx_components=n_approx_components,
        approximate_with=approximate_with,
    )

    if approximate_with == "SHO":
        amplitudes = (
            spectral_coefficients
            * spectral_points
            * kernel_params["variance"]
            / jnp.sum(spectral_coefficients * spectral_points)
        )

        kernel = amplitudes[0] * kernels.quasisep.SHO(
            quality=1 / jnp.sqrt(2), omega=2 * jnp.pi * spectral_points[0]
        )
        for j in range(1, n_approx_components):
            kernel += amplitudes[j] * kernels.quasisep.SHO(
                quality=1 / jnp.sqrt(2), omega=2 * jnp.pi * spectral_points[j]
            )
        return kernel
    else:
        raise NotImplementedError(f"Approximation {approximate_with} not implemented")


def _psd_model(kernel_type, kernel_params):
    """Returns the power spectrum model for the given kernel type and parameters

    Parameters
    ----------
    kernel_type: string
        The type of kernel to be used for the Gaussian Process
        Only designed for the following Power spectra ["PowL","DoubPowL]
    kernel_params: dict
        Dictionary containing the parameters for the kernel
        Should contain the parameters for the selected kernel
    """
    if kernel_type == "PowL":
        return lambda f: jnp.power(f / kernel_params["f_bend"], -kernel_params["alpha_1"]) / (
            1
            + jnp.power(
                f / kernel_params["f_bend"], kernel_params["alpha_2"] - kernel_params["alpha_1"]
            )
        )
    elif kernel_type == "DoubPowL":
        return (
            lambda f: jnp.power(f / kernel_params["f_bend_1"], -kernel_params["alpha_1"])
            / (
                1
                + jnp.power(
                    f / kernel_params["f_bend_1"],
                    kernel_params["alpha_2"] - kernel_params["alpha_1"],
                )
            )
            / (
                1
                + jnp.power(
                    f / kernel_params["f_bend_2"],
                    kernel_params["alpha_3"] - kernel_params["alpha_2"],
                )
            )
        )
    else:
        raise ValueError("PSD type not implemented")


def get_kernel(
    kernel_type,
    kernel_params,
    f_min=0,
    f_max=0,
    n_approx_components=20,
    approximate_with="SHO",
    S_low=20,
    S_high=20,
):
    """
    Function for producing the kernel for the Gaussian Process.
    Returns the selected Tinygp kernel for the given parameters.

    Parameters
    ----------
    kernel_type: string
        The type of kernel to be used for the Gaussian Process
        To be selected from the kernels already implemented:
        ["RN", "QPO", "QPO_plus_RN","PowL","DoubPowL"]

    kernel_params: dict
        Dictionary containing the parameters for the kernel
        Should contain the parameters for the selected kernel
    f_min: float
        The minimum frequency of the time series.
    f_max: float
        The maximum frequency of the time series.

    n_approx_components: int
        The number of components to use to approximate the power law
        must be greater than 2, default 20
    approximate_with: string
        The type of kernel to use to approximate the power law power spectra
        Default is "SHO"

    """
    if not jax_avail:
        raise ImportError("Jax is required")

    if not can_make_gp:
        raise ImportError("Tinygp is required to make kernels")

    if kernel_type == "QPO_plus_RN":
        kernel = kernels.quasisep.Exp(
            scale=1 / kernel_params["crn"], sigma=(kernel_params["arn"]) ** 0.5
        ) + kernels.quasisep.Celerite(
            a=kernel_params["aqpo"],
            b=0.0,
            c=kernel_params["cqpo"],
            d=2 * jnp.pi * kernel_params["freq"],
        )
        return kernel
    elif kernel_type == "RN":
        kernel = kernels.quasisep.Exp(
            scale=1 / kernel_params["crn"], sigma=(kernel_params["arn"]) ** 0.5
        )
        return kernel
    elif kernel_type == "QPO":
        kernel = kernels.quasisep.Celerite(
            a=kernel_params["aqpo"],
            b=0.0,
            c=kernel_params["cqpo"],
            d=2 * jnp.pi * kernel_params["freq"],
        )
        return kernel
    elif kernel_type == "PowL" or kernel_type == "DoubPowL":
        if n_approx_components < 2:
            raise ValueError("Number of approximation components must be greater than 2")

        kernel = _approximate_powerlaw(
            kernel_type,
            kernel_params,
            f_min=f_min / S_low,
            f_max=f_max * S_high,
            n_approx_components=n_approx_components,
            approximate_with=approximate_with,
        )
        return kernel
    else:
        raise ValueError("Kernel type not implemented")


def get_mean(mean_type, mean_params):
    """
    Function for producing the mean function for the Gaussian Process.

    Parameters
    ----------
    mean_type: string
        The type of mean to be used for the Gaussian Process
        To be selected from the mean functions already implemented:
        ["gaussian", "exponential", "constant", "skew_gaussian",
         "skew_exponential", "fred"]

    mean_params: dict
        Dictionary containing the parameters for the mean
        Should contain the parameters for the selected mean

    Returns
    -------
    A function which takes in the time coordinates and returns the mean values.

    Examples
    --------
    Unimodal Gaussian Mean Function:
        mean_params = {"A": 3.0, "t0": 0.2,  "sig": 0.1}
        mean = get_mean("gaussian", mean_params)

    Multimodal Skew Gaussian Mean Function:
        mean_params = {"A": jnp.array([3.0, 4.0]), "t0": jnp.array([0.2, 1]),
                      "sig1": jnp.array([0.1, 0.4]), "sig2": jnp.array([0.4, 0.1])}
        mean = get_mean("skew_gaussian", mean_params)

    """
    if not jax_avail:
        raise ImportError("Jax is required")

    if mean_type == "gaussian":
        mean = functools.partial(_gaussian, params=mean_params)
    elif mean_type == "exponential":
        mean = functools.partial(_exponential, params=mean_params)
    elif mean_type == "constant":
        mean = functools.partial(_constant, params=mean_params)
    elif mean_type == "skew_gaussian":
        mean = functools.partial(_skew_gaussian, params=mean_params)
    elif mean_type == "skew_exponential":
        mean = functools.partial(_skew_exponential, params=mean_params)
    elif mean_type == "fred":
        mean = functools.partial(_fred, params=mean_params)
    else:
        raise ValueError("Mean type not implemented")
    return mean


def _gaussian(t, params):
    """A gaussian flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.

    params: dict
        The dictionary containing parameter values of the gaussian flare.

        The parameters for the gaussian flare are:
        A:  jnp.float / jnp.ndarray
            Amplitude of the flare.
        t0: jnp.float / jnp.ndarray
            The location of the maximum.
        sig1: jnp.float / jnp.ndarray
            The width parameter for the gaussian.

    Returns
    -------
    The y values for the gaussian flare.
    """
    A = jnp.atleast_1d(params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(params["t0"])[:, jnp.newaxis]
    sig = jnp.atleast_1d(params["sig"])[:, jnp.newaxis]

    return jnp.sum(A * jnp.exp(-((t - t0) ** 2) / (2 * (sig**2))), axis=0)


def _exponential(t, params):
    """An exponential flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.

    params: dict
        The dictionary containing parameter values of the exponential flare.

        The parameters for the exponential flare are:
        A:  jnp.float / jnp.ndarray
            Amplitude of the flare.
        t0: jnp.float / jnp.ndarray
            The location of the maximum.
        sig1: jnp.float / jnp.ndarray
            The width parameter for the exponential.

    Returns
    -------
    The y values for exponential flare.
    """
    A = jnp.atleast_1d(params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(params["t0"])[:, jnp.newaxis]
    sig = jnp.atleast_1d(params["sig"])[:, jnp.newaxis]

    return jnp.sum(A * jnp.exp(-jnp.abs(t - t0) / (2 * (sig**2))), axis=0)


def _constant(t, params):
    """A constant mean shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.

    params: dict
        The dictionary containing parameter values of the constant flare.

        The parameters for the constant flare are:
        A:  jnp.float
            Constant amplitude of the flare.

    Returns
    -------
    The constant value.
    """
    return params["A"] * jnp.ones_like(t)


def _skew_gaussian(t, params):
    """A skew gaussian flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.

    params: dict
        The dictionary containing parameter values of the skew gaussian flare.

        The parameters for the skew gaussian flare are:
        A:  jnp.float / jnp.ndarray
            Amplitude of the flare.
        t0: jnp.float / jnp.ndarray
            The location of the maximum.
        sig1: jnp.float / jnp.ndarray
            The width parameter for the rising edge.
        sig2: jnp.float / jnp.ndarray
            The width parameter for the falling edge.

    Returns
    -------
    The y values for skew gaussian flare.
    """
    A = jnp.atleast_1d(params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(params["t0"])[:, jnp.newaxis]
    sig1 = jnp.atleast_1d(params["sig1"])[:, jnp.newaxis]
    sig2 = jnp.atleast_1d(params["sig2"])[:, jnp.newaxis]

    y = jnp.sum(
        A
        * jnp.where(
            t > t0,
            jnp.exp(-((t - t0) ** 2) / (2 * (sig2**2))),
            jnp.exp(-((t - t0) ** 2) / (2 * (sig1**2))),
        ),
        axis=0,
    )
    return y


def _skew_exponential(t, params):
    """A skew exponential flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.

    params: dict
        The dictionary containing parameter values of the skew exponential flare.

        The parameters for the skew exponential flare are:
        A:  jnp.float / jnp.ndarray
            Amplitude of the flare.
        t0: jnp.float / jnp.ndarray
            The location of the maximum.
        sig1: jnp.float / jnp.ndarray
            The width parameter for the rising edge.
        sig2: jnp.float / jnp.ndarray
            The width parameter for the falling edge.

    Returns
    -------
    The y values for exponential flare.
    """
    A = jnp.atleast_1d(params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(params["t0"])[:, jnp.newaxis]
    sig1 = jnp.atleast_1d(params["sig1"])[:, jnp.newaxis]
    sig2 = jnp.atleast_1d(params["sig2"])[:, jnp.newaxis]

    y = jnp.sum(
        A
        * jnp.where(
            t > t0,
            jnp.exp(-(t - t0) / (2 * (sig2**2))),
            jnp.exp((t - t0) / (2 * (sig1**2))),
        ),
        axis=0,
    )
    return y


def _fred(t, params):
    """A fast rise exponential decay (FRED) flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.

    params: dict
        The dictionary containing parameter values of the FRED flare.

        The parameters for the FRED flare are:
        A:  jnp.float / jnp.ndarray
            Amplitude of the flare.
        t0: jnp.float / jnp.ndarray
            The location of the maximum.
        phi: jnp.float / jnp.ndarray
            Symmetry parameter of the flare.
        delta: jnp.float / jnp.ndarray
            Offset parameter of the flare.

    Returns
    -------
    The y values for exponential flare.
    """
    A = jnp.atleast_1d(params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(params["t0"])[:, jnp.newaxis]
    phi = jnp.atleast_1d(params["phi"])[:, jnp.newaxis]
    delta = jnp.atleast_1d(params["delta"])[:, jnp.newaxis]

    y = jnp.sum(
        A * jnp.exp(-phi * ((t + delta) / t0 + t0 / (t + delta))) * jnp.exp(2 * phi), axis=0
    )

    return y


def _get_kernel_params(kernel_type):
    """
    Generates a list of the parameters for the kernel for the GP model based on the kernel type.

    Parameters
    ----------
    kernel_type: string
        The type of kernel to be used for the Gaussian Process model
        The parameters in log scale have a prefix of "log_"

    Returns
    -------
        A list of the parameters for the kernel for the GP model
    """
    if kernel_type == "RN":
        return ["log_arn", "log_crn"]
    elif kernel_type == "QPO_plus_RN":
        return ["log_arn", "log_crn", "log_aqpo", "log_cqpo", "log_freq"]
    elif kernel_type == "QPO":
        return ["log_aqpo", "log_cqpo", "log_freq"]
    elif kernel_type == "PowL":
        return ["alpha_1", "log_f_bend", "alpha_2", "variance"]
    elif kernel_type == "DoubPowL":
        return ["alpha_1", "log_f_bend_1", "alpha_2", "log_f_bend_2", "alpha_3", "variance"]
    else:
        raise ValueError("Kernel type not implemented")


def _get_mean_params(mean_type):
    """
    Generates a list of the parameters for the mean for the GP model based on the mean type.

    Parameters
    ----------
    mean_type: string
        The type of mean to be used for the Gaussian Process model
        The parameters in log scale have a prefix of "log_"

    Returns
    -------
        A list of the parameters for the mean for the GP model
    """
    if (mean_type == "gaussian") or (mean_type == "exponential"):
        return ["log_A", "t0", "log_sig"]
    elif mean_type == "constant":
        return ["log_A"]
    elif (mean_type == "skew_gaussian") or (mean_type == "skew_exponential"):
        return ["log_A", "t0", "log_sig1", "log_sig2"]
    elif mean_type == "fred":
        return ["log_A", "t0", "delta", "phi"]
    else:
        raise ValueError("Mean type not implemented")


def get_gp_params(kernel_type, mean_type, scale_errors=False, log_transform=False):
    """
    Generates a list of the parameters for the GP model based on the kernel and mean type.
    To be used to set the order of the parameters for `get_prior` and `get_likelihood` functions.

    Parameters
    ----------
    kernel_type: string
        The type of kernel to be used for the Gaussian Process model:
        ["RN", "QPO", "QPO_plus_RN", "PowL", "DoubPowL"]

    mean_type: string
        The type of mean to be used for the Gaussian Process model:
        ["gaussian", "exponential", "constant", "skew_gaussian",
         "skew_exponential", "fred"]
    scale_errors: bool, default False
        Whether to include a scale parameter on the errors in the GP model
    log_transform: bool, default False
        Whether to take the log of the data to make the data normally distributed
        This will add a parameter to the model to model a shifted log normal distribution
        And will change the mean parameter "log_A" to "A" as the mean could be negative

    Returns
    -------
        A list of the parameters for the GP model

    Examples
    --------
    get_gp_params("QPO_plus_RN", "gaussian")
    ['log_arn', 'log_crn', 'log_aqpo', 'log_cqpo', 'log_freq', 'log_A', 't0', 'log_sig']
    """
    kernel_params = _get_kernel_params(kernel_type)
    mean_params = _get_mean_params(mean_type)
    kernel_params.extend(mean_params)
    if scale_errors:
        kernel_params.append("scale_err")
    if log_transform:
        kernel_params.append("log_shift")
        if "log_A" in kernel_params:
            kernel_params[kernel_params.index("log_A")] = "A"
    return kernel_params


def get_prior(params_list, prior_dict):
    """
    A prior generator function based on given values.
    Makes a jaxns specific prior function based on the given prior dictionary.
    Jaxns requires the parameters of the prior function and log_likelihood function to
    be in the same order. This order is made according to the params_list.

    Parameters
    ----------
    params_list:
        A list in order of the parameters to be used.

    prior_dict:
        A dictionary of the priors of parameters to be used.
        These parameters should be from tensorflow_probability distributions / Priors from jaxns
        or special priors from jaxns.
        **Note**: If jaxns priors are used, then the name given to them should be the same as
        the corresponding name in the params_list.
        Also, if a parameter is to be used in the log scale, it should have a prefix of "log_"

    Returns
    -------
    The Prior generator function.
    The arguments of the prior function are in the order of
        Kernel arguments (RN arguments, QPO arguments),
        Mean arguments
        Miscellaneous arguments

    Examples
    --------
    A prior function for a Red Noise kernel and a Gaussian mean function

    # Obtain the parameters list
    params_list = get_gp_params("RN", "gaussian")

    # Make a prior dictionary using tensorflow_probability distributions
    prior_dict = {
       "log_A": tfpd.Uniform(low = jnp.log(1e-1), high = jnp.log(2e+2)),
       "t0": tfpd.Uniform(low = 0.0 - 0.1, high = 1 + 0.1),
       "log_sig": tfpd.Uniform(low = jnp.log(0.5 * 1 / 20), high = jnp.log(2) ),
       "log_arn": tfpd.Uniform(low = jnp.log(0.1) , high = jnp.log(2) ),
       "log_crn": tfpd.Uniform(low = jnp.log(1 /5), high = jnp.log(20)),
    }

    prior_model = get_prior(params_list, prior_dict)

    """
    if not jax_avail:
        raise ImportError("Jax is required")

    if not can_sample:
        raise ImportError("Jaxns not installed. Cannot make jaxns specific prior.")

    if not tfp_available:
        raise ImportError("Tensorflow probability required to make priors.")

    def prior_model():
        prior_list = []
        for i in params_list:
            if isinstance(prior_dict[i], tfpd.Distribution):
                parameter = yield Prior(prior_dict[i], name=i)
            elif isinstance(prior_dict[i], Prior):
                parameter = yield prior_dict[i]
            else:
                raise ValueError("Prior should be a tfpd distribution or a jaxns prior.")
            prior_list.append(parameter)
        return tuple(prior_list)

    return prior_model


def get_log_likelihood(
    params_list,
    kernel_type,
    mean_type,
    times,
    counts,
    counts_err=None,
    S_low=20,
    S_high=20,
    n_approx_components=20,
    approximate_with="SHO",
):
    """
    A log likelihood generator function based on given values.
    Makes a jaxns specific log likelihood function which takes in the
    parameters in the order of the parameters list, and calculates the
    log likelihood of the data given the parameters, and the model
    (kernel, mean) of the GP model. **Note** Any parameters with a prefix
    of "log_" are taken to be in the log scale.

    Parameters
    ----------
    params_list:
        A list in order of the parameters to be used.

    prior_dict:
        A dictionary of the priors of parameters to be used.

    kernel_type:
        The type of kernel to be used in the model:
        ["RN", "QPO", "QPO_plus_RN", "PowL", "DoubPowL"]

    mean_type:
        The type of mean to be used in the model:
        ["gaussian", "exponential", "constant", "skew_gaussian",
         "skew_exponential", "fred"]

    times: np.array or jnp.array
        The time array of the lightcurve

    counts: np.array or jnp.array
        The photon counts array of the lightcurve

    counts_err: np.array or jnp.array
        The photon counts error array of the lightcurve

    n_approx_components: int
        The number of components to use to approximate the power law
        must be greater than 2, default 20

    approximate_with: string
        The type of kernel to use to approximate the power law power spectra
        Default is "SHO"
    log_transform: bool, default False
        Whether to take the log of the data to make the data normally distributed
        This will add a parameter to the model to model a shifted log normal distribution

    Returns
    -------
    The Jaxns specific log likelihood function.

    """
    if not jax_avail:
        raise ImportError("Jax is required")

    if not can_make_gp:
        raise ImportError("Tinygp is required to make the GP model.")
    f_min, f_max = 1 / (times[-1] - times[0]), 0.5 / jnp.min(jnp.diff(times))

    @jit
    def likelihood_model(*args):
        param_dict = {}
        for i, params in enumerate(params_list):
            if params[0:4] == "log_":
                param_dict[params[4:]] = jnp.exp(args[i])
            else:
                param_dict[params] = args[i]

        kernel = get_kernel(
            kernel_type=kernel_type,
            kernel_params=param_dict,
            f_min=f_min,
            f_max=f_max,
            n_approx_components=n_approx_components,
            approximate_with=approximate_with,
            S_low=S_low,
            S_high=S_high,
        )
        mean = get_mean(mean_type=mean_type, mean_params=param_dict)
        if "shift" in param_dict.keys():
            x = jnp.log(counts - param_dict["shift"])
            if counts_err is not None:
                xerr = jnp.divide(counts_err, counts - param_dict["shift"])
        else:
            x = counts
            xerr = counts_err

        if counts_err is None:
            gp = GaussianProcess(kernel, times, mean_value=mean(times))
        elif counts_err is not None and "scale_err" in param_dict.keys():
            gp = GaussianProcess(
                kernel,
                times,
                mean_value=mean(times),
                diag=param_dict["scale_err"] * jnp.square(xerr),
            )
        else:
            gp = GaussianProcess(kernel, times, mean_value=mean(times), diag=jnp.square(xerr))
        return gp.log_probability(x)

    return likelihood_model


class GPResult:
    """
    Makes a GPResult object which takes in a Stingray.Lightcurve and samples parameters of a model
    (Gaussian Process) based on the given prior and log_likelihood function.

    Parameters
    ----------
    lc: Stingray.Lightcurve object
        The lightcurve on which the bayesian inference is to be done

    """

    def __init__(self, lc: Lightcurve) -> None:
        self.lc = lc
        self.time = lc.time
        self.counts = lc.counts
        self.result = None

    def sample(self, prior_model=None, likelihood_model=None, max_samples=1e4, num_live_points=500):
        """
        Makes a Jaxns nested sampler over the Gaussian Process, given the
        prior and likelihood model

        Parameters
        ----------
        prior_model: jaxns.prior.PriorModelType object
            A prior generator object.
            Can be made using the get_prior function or can use your own jaxns
            compatible prior function.

        likelihood_model: jaxns.types.LikelihoodType object
            A likelihood function which takes in the arguments of the prior
            model and returns the loglikelihood of the model.
            Can be made using the get_log_likelihood function or can use your own
            log_likelihood function with same order of arguments as the prior_model.


        max_samples: int, default 1e4
            The maximum number of samples to be taken by the nested sampler

        num_live_points : int, default 500
            The number of live points to use in the nested sampling

        Returns
        ----------
        results: jaxns.results.NestedSamplerResults object
            The results of the nested sampling process

        """
        if not jax_avail:
            raise ImportError("Jax is required")

        if not can_sample:
            raise ImportError("Jaxns not installed! Can't sample!")

        self.prior_model = prior_model
        self.log_likelihood_model = likelihood_model

        nsmodel = Model(prior_model=self.prior_model, log_likelihood=self.log_likelihood_model)
        nsmodel.sanity_check(random.PRNGKey(10), S=100)

        # check the approximation of the model

        self.exact_ns = DefaultNestedSampler(
            nsmodel, num_live_points=num_live_points, max_samples=max_samples, verbose=True
        )

        termination_reason, state = self.exact_ns(
            random.PRNGKey(42), term_cond=TerminationCondition()  # live_evidence_frac=1e-4)
        )
        self.results = self.exact_ns.to_results(termination_reason, state)
        print("Simulation Complete")

    def get_evidence(self):
        """
        Returns the log evidence of the model
        """
        return self.results.log_Z_mean

    def print_summary(self):
        """
        Prints a summary table for the model parameters
        """
        self.exact_ns.summary(self.results)

    def plot_diagnostics(self):
        """
        Plots the diagnostic plots for the sampling process
        """
        self.exact_ns.plot_diagnostics(self.results)

    def plot_cornerplot(self):
        """
        Plots the corner plot for the sampled hyperparameters
        """
        self.exact_ns.plot_cornerplot(self.results)

    def get_parameters_names(self):
        """
        Returns the names of the parameters
        """
        return sorted(self.results.samples.keys())

    def get_max_posterior_parameters(self):
        """
        Returns the optimal parameters for the model based on the NUTS sampling
        """
        max_post_idx = jnp.argmax(self.results.log_posterior_density)
        map_points = jax.tree_map(lambda x: x[max_post_idx], self.results.samples)

        return map_points

    def get_max_likelihood_parameters(self):
        """
        Returns the maximum likelihood parameters
        """
        max_like_idx = jnp.argmax(self.results.log_L_samples)
        max_like_points = jax.tree_map(lambda x: x[max_like_idx], self.results.samples)

        return max_like_points

    def posterior_plot(self, name: str, n=0, axis=None, save=False, filename=None):
        """
        Plots the posterior histogram for the given parameter

        Parameters
        ----------
        name : str
            Name of the parameter.
            Should be from the names of the parameter list used or from the names of parameters
            used in the prior_function

        n : int, default 0
            The index of the parameter to be plotted (for multi component parameters).
            For multivariate parameters, the index of the specific parameter to be plotted.

        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for ``matplotlib.pyplot.axis()`` method.

        save : bool, optional, default ``False``
            If ``True``, save the figure with specified filename.

        filename : str
            File name and path of the image to save. Depends on the boolean ``save``.

        Returns
        -------
        plt : ``matplotlib.pyplot`` object
            Reference to plot, call ``show()`` to display it

        """
        nsamples = self.results.total_num_samples
        samples = self.results.samples[name].reshape((nsamples, -1))[:, n]
        plt.hist(
            samples, bins="auto", density=True, alpha=1.0, label=name, fc="None", edgecolor="black"
        )
        mean1 = jnp.mean(self.results.samples[name])
        std1 = jnp.std(self.results.samples[name])
        plt.axvline(mean1, color="red", linestyle="dashed", label="mean")
        plt.axvline(mean1 + std1, color="green", linestyle="dotted")
        plt.axvline(mean1 - std1, linestyle="dotted", color="green")
        plt.title("Posterior Histogram of " + str(name))
        plt.xlabel(name)
        plt.ylabel("Probability Density")
        plt.legend()

        if axis is not None:
            plt.axis(axis)

        if save:
            if filename is None:
                plt.savefig(str(name) + "_Posterior_plot.png")
            else:
                plt.savefig(filename)
        return plt

    def weighted_posterior_plot(
        self, name: str, n=0, rkey=None, axis=None, save=False, filename=None
    ):
        """
        Returns the weighted posterior histogram for the given parameter

        Parameters
        ----------
        name : str
            Name of the parameter.
            Should be from the names of the parameter list used or from the names of parameters
            used in the prior_function

        n : int, default 0
            The index of the parameter to be plotted (for multi component parameters).
            For multivariate parameters, the index of the specific parameter to be plotted.

        key: jax.random.PRNGKey, default ``random.PRNGKey(1234)``
            Random key for the weighted sampling

        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for ``matplotlib.pyplot.axis()`` method.

        save : bool, optionalm, default ``False``
            If ``True``, save the figure with specified filename.

        filename : str
            File name and path of the image to save. Depends on the boolean ``save``.

        Returns
        -------
        plt : ``matplotlib.pyplot`` object
            Reference to plot, call ``show()`` to display it
        """
        if rkey is None:
            rkey = random.PRNGKey(1234)

        nsamples = self.results.total_num_samples
        log_p = self.results.log_dp_mean
        samples = self.results.samples[name].reshape((nsamples, -1))[:, n]

        weights = jnp.where(jnp.isfinite(samples), jnp.exp(log_p), 0.0)
        log_weights = jnp.where(jnp.isfinite(samples), log_p, -jnp.inf)
        samples_resampled = resample(
            rkey, samples, log_weights, S=max(10, int(self.results.ESS)), replace=True
        )

        nbins = max(10, int(jnp.sqrt(self.results.ESS)) + 1)
        binsx = jnp.linspace(
            *jnp.percentile(samples_resampled, jnp.asanyarray([0, 100])), 2 * nbins
        )

        plt.hist(
            np.asanyarray(samples_resampled),
            bins=binsx,
            density=True,
            alpha=1.0,
            label=name,
            fc="None",
            edgecolor="black",
        )
        sample_mean = jnp.average(samples, weights=weights)
        sample_std = jnp.sqrt(jnp.average((samples - sample_mean) ** 2, weights=weights))
        plt.axvline(sample_mean, color="red", linestyle="dashed", label="mean")
        plt.axvline(sample_mean + sample_std, color="green", linestyle="dotted")
        plt.axvline(sample_mean - sample_std, linestyle="dotted", color="green")
        plt.title("Weighted Posterior Histogram of " + str(name))
        plt.xlabel(name)
        plt.ylabel("Probability Density")
        plt.legend()
        if axis is not None:
            plt.axis(axis)

        if save:
            if filename is None:
                plt.savefig(str(name) + "_Weighted_Posterior_plot.png")
            else:
                plt.savefig(filename)
        return plt

    def comparison_plot(
        self,
        param1: str,
        param2: str,
        n1=0,
        n2=0,
        rkey=None,
        axis=None,
        save=False,
        filename=None,
    ):
        """
        Plots the comparison plot between two given parameters

        Parameters
        ----------
        param1 : str
            Name of the first parameter.
            Should be from the names of the parameter list used or from the names of parameters
            used in the prior_function

        param2 : str
            Name of the second parameter.
            Should be from the names of the parameter list used or from the names of parameters
            used in the prior_function

        n1 : int, default 0
            The index of the first parameter to be plotted (for multi component parameters).
            For multivariate parameters, the index of the specific parameter to be plotted.

        n2 : int, default 0
            The index of the second parameter to be plotted (for multi component parameters).
            For multivariate parameters, the index of the specific parameter to be plotted.

        key: jax.random.PRNGKey, default ``random.PRNGKey(1234)``
            Random key for the shuffling the weights

        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for ``matplotlib.pyplot.axis()`` method.

        save : bool, optionalm, default ``False``
            If ``True``, save the figure with specified filename.

        filename : str
            File name and path of the image to save. Depends on the boolean ``save``.

        Returns
        -------
        plt : ``matplotlib.pyplot`` object
            Reference to plot, call ``show()`` to display it
        """
        if rkey is None:
            rkey = random.PRNGKey(1234)

        nsamples = self.results.total_num_samples
        log_p = self.results.log_dp_mean
        samples1 = self.results.samples[param1].reshape((nsamples, -1))[:, n1]
        samples2 = self.results.samples[param2].reshape((nsamples, -1))[:, n2]

        log_weights = jnp.where(jnp.isfinite(samples2), log_p, -jnp.inf)
        nbins = max(10, int(jnp.sqrt(self.results.ESS)) + 1)

        samples_resampled = resample(
            rkey,
            jnp.stack([samples1, samples2], axis=-1),
            log_weights,
            S=max(10, int(self.results.ESS)),
            replace=True,
        )
        plt.hist2d(
            samples_resampled[:, 1],
            samples_resampled[:, 0],
            bins=(nbins, nbins),
            density=True,
            cmap="GnBu",
        )
        plt.title("Comparison Plot of " + str(param1) + " and " + str(param2))
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.colorbar()
        if axis is not None:
            plt.axis(axis)

        if save:
            if filename is None:
                plt.savefig(str(param1) + "_" + str(param2) + "_Comparison_plot.png")
            else:
                plt.savefig(filename)

        return plt

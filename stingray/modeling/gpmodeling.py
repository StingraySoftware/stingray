import numpy as np
import matplotlib.pyplot as plt
import functools
from stingray import Lightcurve

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
    from jaxns import ExactNestedSampler, TerminationCondition, Prior, Model
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


__all__ = ["get_kernel", "get_mean", "get_prior", "get_log_likelihood", "GPResult", "get_gp_params"]


def get_kernel(kernel_type, kernel_params, p=1, q=0):
    """
    Function for producing the kernel for the Gaussian Process.
    Returns the selected Tinygp kernel for the given parameters.

    Parameters
    ----------
    kernel_type: string
        The type of kernel to be used for the Gaussian Process
        To be selected from the kernels already implemented:
        ["RN", "QPO", "QPO_plus_RN", "CARMA", "QPO_plus_CARMA"]

    kernel_params: dict
        Dictionary containing the parameters for the kernel
        Should contain the parameters for the selected kernel

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
    elif kernel_type == "CARMA":
        alpha = kernel_params["alpha"]
        beta = kernel_params["beta"]

        # jax.debug.print("alpha in get_kernel: {}", kernel_params["alpha"])
        # jax.debug.print("beta in get_kernel: {}", kernel_params["beta"])
        acarma = kernel_params["acarma"]
        kernel = kernels.quasisep.CARMA.init(alpha=alpha, beta=beta, sigma=acarma**0.5)
        return kernel
    elif kernel_type == "QPO_plus_CARMA":
        kernel = kernels.quasisep.CARMA.init(
            alpha=kernel_params["alpha"],
            beta=kernel_params["beta"],
            sigma=kernel_params["acarma"] ** 0.5,
        ) + kernels.quasisep.Celerite(
            a=kernel_params["aqpo"],
            b=0.0,
            c=kernel_params["cqpo"],
            d=2 * jnp.pi * kernel_params["freq"],
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
    elif kernel_type == "CARMA":
        return ["log_alpha", "log_beta", "log_acarma"]
    elif kernel_type == "QPO_plus_CARMA":
        return ["log_alpha", "log_beta", "log_acarma", "log_aqpo", "log_cqpo", "log_freq"]
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


def get_gp_params(kernel_type, mean_type):
    """
    Generates a list of the parameters for the GP model based on the kernel and mean type.
    To be used to set the order of the parameters for `get_prior` and `get_likelihood` functions.

    Parameters
    ----------
    kernel_type: string
        The type of kernel to be used for the Gaussian Process model:
        ["RN", "QPO", "QPO_plus_RN"]

    mean_type: string
        The type of mean to be used for the Gaussian Process model:
        ["gaussian", "exponential", "constant", "skew_gaussian",
         "skew_exponential", "fred"]

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
    return kernel_params


def get_prior(params_list, prior_dict, p=1, q=0):
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

    p, q: int, int, default 1
        The orders for the CARMA process, must be p >= q

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
        for param in params_list:
            if param == "log_alpha":
                for j in range(p):
                    if isinstance(prior_dict[param], tfpd.Distribution):
                        parameter = yield Prior(prior_dict[param], name=param + str(j))
                    elif isinstance(prior_dict[param], Prior):
                        parameter = yield prior_dict[param]
                    else:
                        raise ValueError("Prior should be a tfpd distribution or a jaxns prior.")
                    prior_list.append(parameter)
            elif param == "log_beta":
                for k in range(q):
                    if isinstance(prior_dict[param], tfpd.Distribution):
                        parameter = yield Prior(prior_dict[param], name=param + str(k))
                    elif isinstance(prior_dict[param], Prior):
                        parameter = yield prior_dict[param]
                    else:
                        raise ValueError("Prior should be a tfpd distribution or a jaxns prior.")
                    prior_list.append(parameter)
            else:
                if isinstance(prior_dict[param], tfpd.Distribution):
                    parameter = yield Prior(prior_dict[param], name=param + str(k))
                elif isinstance(prior_dict[param], Prior):
                    parameter = yield prior_dict[param]
                else:
                    raise ValueError("Prior should be a tfpd distribution or a jaxns prior.")
                prior_list.append(parameter)
        return tuple(prior_list)

    return prior_model


def get_log_likelihood(
    params_list, kernel_type, mean_type, times, counts, counts_err=None, p=1, q=0
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
        ["RN", "QPO", "QPO_plus_RN"]

    mean_type:
        The type of mean to be used in the model:
        ["gaussian", "exponential", "constant", "skew_gaussian",
         "skew_exponential", "fred"]

    times: np.array or jnp.array
        The time array of the lightcurve

    counts: np.array or jnp.array
        The photon counts array of the lightcurve

    counts_err : np.array or jnp.array
        The uncertainties on the counts or fluxes given
        in `counts`. Must be of same shape as `counts`

    Returns
    -------
    The Jaxns specific log likelihood function.

    """
    if not jax_avail:
        raise ImportError("Jax is required")

    if not can_make_gp:
        raise ImportError("Tinygp is required to make the GP model.")

    if counts_err is None:
        counts_err = jnp.zeros_like(counts)

    @jit
    def likelihood_model(*args):
        # print(f"p: {p}")
        # print(f"q: {q}")
        param_dict = {}
        i = 0

        for params in params_list:
            if params == "alpha":
                param_dict["alpha"] = args[i : i + p]
                i += p
            elif params == "log_alpha":
                param_dict["alpha"] = jnp.exp(jnp.array(args[i : i + p]))
                i += p
            elif params == "beta":
                param_dict["beta"] = args[i : i + q]
                i += q
            elif params == "log_beta":
                param_dict["beta"] = jnp.exp(jnp.array(args[i : i + q]))
                i += q
            else:
                if params[0:4] == "log_":
                    param_dict[params[4:]] = jnp.exp(args[i])
                else:
                    param_dict[params] = args[i]
                i += 1

        kernel = get_kernel(kernel_type=kernel_type, kernel_params=param_dict)
        mean = get_mean(mean_type=mean_type, mean_params=param_dict)
        gp = GaussianProcess(kernel, times, mean_value=mean(times), diag=counts_err)
        log_like = jnp.nan_to_num(gp.log_probability(counts), nan=-1e20, posinf=-1e20, neginf=-1e20)
        return log_like

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

        if lc.err_dist == "poisson":
            self.counts_err = jnp.sqrt(self.counts)

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

        self.exact_ns = ExactNestedSampler(
            nsmodel, num_live_points=num_live_points, max_samples=max_samples
        )

        termination_reason, state = self.exact_ns(
            random.PRNGKey(42), term_cond=TerminationCondition(live_evidence_frac=1e-4)
        )
        self.results = self.exact_ns.to_results(state, termination_reason)
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

    def plot_posterior_predictive(
        self,
        lc,
        kernel_type,
        mean_type,
        nmean=50,
        ngrid=1000,
        ax=None,
        rkey=None,
        p=1,
        q=0,
        counts_err=None,
    ):
        """
        Plot the posterior predictive distribution.
        Will plot the maximum posterior for the Gaussian Process, and
        `nsamples` random draws from the mean function.

        Parameters
        ----------
        lc : stingray.Lightcurve object
            The light curve with the time series data
            being modelled

        kernel_type : str
            The kernel type used in the modeling

        mean_type : str
            The type of the mean function used

        nmean: int, default 50
            The number of samples to use for drawing the
            posterior

        ngrid: int, default 1000
            The number of points in the linear grid to
            use for plotting the Gaussian Process

        ax : matplotlib.Axes object, default None
            A matplotlib.Axes object to plot into. If none is
            given, a new Figure object will be created

        rkey : jax.randomPRNGKey object
            A random key for setting the sampling. If None,
            set to random.PRNGKey(1234)

        p, q : int, int, default 1, 0
            If the kernel involves a CARMA model, then this
            sets the orders of the AR and MA processes involved.
            Note that p >= q is required

        counts_err : array, default None
            if `None`, the error bars for the data will be set
            to `sqrt(lc.counts)`, otherwise to whatever is set
            for `counts_err`

        Returns
        -------
        ax : matplotlib.Axes object
            The matplotlib.axes object that the plot is
            drawn in
        """
        if rkey is None:
            rkey = random.PRNGKey(1234)

        log_p = self.results.log_dp_mean  # log-prob
        nsamples = self.results.total_num_samples  # number of samples

        # array for resampled samples
        samples_resampled = {}

        # go through samples, resample with weights to get
        # a weighted posterior sample
        for name in self.results.samples.keys():
            samples = self.results.samples[name]

            weights = jnp.where(jnp.isfinite(samples), jnp.exp(log_p), 0.0)
            log_weights = jnp.where(jnp.isfinite(samples), log_p, -jnp.inf)
            sr = resample(
                rkey, samples, log_weights, S=max(10, int(self.results.ESS)), replace=True
            )
            samples_resampled[name] = sr

        print("Resampling done, calculating maximum posterior model ...")
        # split into samples belonging to the kernel, and samples
        # belonging to the mean function
        kernel_params = stingray.modeling.gpmodeling._get_kernel_params(kernel_type)
        mean_params = stingray.modeling.gpmodeling._get_mean_params(mean_type)

        maxpost_log = self.get_max_posterior_parameters()

        if not "CARMA" in kernel_type:
            mean_samples = dict((k, samples_resampled[k]) for k in mean_params)
            sk_maxpost_log = dict((k, maxpost_log[k]) for k in kernel_params)
            sm_maxpost_log = dict((k, maxpost_log[k]) for k in mean_params)
        else:
            mean_samples = dict((k, samples_resampled[k + "0"]) for k in mean_params)

            kernel_params_small = test_list = [
                i for i in kernel_params if not ("alpha" in i or "beta" in i)
            ]
            sk_maxpost_log = dict((k, maxpost_log[k + "0"]) for k in kernel_params_small)

            alpha, beta = [], []
            for j in range(p):
                print(j)
                if "log_alpha" in kernel_params:
                    alpha.append(maxpost_log["log_alpha" + str(j)])
                else:
                    alpha.append(maxpost_log["alpha" + str(j)])
            for k in range(q):
                print(k)
                if "log_beta" in kernel_params:
                    beta.append(maxpost_log["log_beta" + str(k)])
                else:
                    beta.append(maxpost_log["beta" + str(k)])

            sk_maxpost_log["log_alpha"] = alpha
            sk_maxpost_log["log_beta"] = beta

            sm_maxpost_log = dict((k, maxpost_log[k + "0"]) for k in mean_params)

        sk_maxpost, sm_maxpost = {}, {}
        for params in kernel_params:
            if params[0:4] == "log_":
                sk_maxpost[params[4:]] = np.exp(sk_maxpost_log[params])
            else:
                sk_maxpost[params] = sk_maxpost_log[params]

        for params in mean_params:
            if params[0:4] == "log_":
                sm_maxpost[params[4:]] = np.exp(sm_maxpost_log[params])
            else:
                sm_maxpost[params] = sm_maxpost_log[params]

        kernel = get_kernel(kernel_type=kernel_type, kernel_params=sk_maxpost)
        mean = get_mean(mean_type=mean_type, mean_params=sm_maxpost)

        gp = GaussianProcess(kernel, lc.time, mean_value=mean(lc.time))
        tgrid = np.linspace(lc.time[0], lc.time[-1], ngrid)
        _, cond = gp.condition(lc.counts, tgrid)

        mu = cond.loc + mean(tgrid)
        std = np.sqrt(cond.variance)

        print("GP calculated, plotting GP and data ...")

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        if counts_err is None:
            counts_err = np.sqrt(lc.counts)
        ax.errorbar(
            lc.time,
            lc.counts,
            yerr=counts_err,
            fmt="o",
            markersize=2,
            color="black",
            label="Observations",
        )
        ax.plot(tgrid, mu, color="C0", label="Gaussian Process Maximum Posterior")
        ax.fill_between(tgrid, mu + std, mu - std, color="C0", alpha=0.3)

        idx_all = np.random.choice(
            np.arange(0, int(gpresult.results.ESS), 1.0, dtype=int), size=nmean, replace=False
        )
        print("calculating and plotting mean functions ...")
        for i, idx in enumerate(idx_all):
            # sk_log = dict((k, kernel_samples[k][idx]) for k in kernel_params)
            sm_log = dict((k, mean_samples[k][idx]) for k in mean_params)
            sm = {}
            for params in mean_params:
                if params[0:4] == "log_":
                    sm[params[4:]] = jnp.exp(sm_log[params])
                else:
                    sm[params] = sm_log[params]

            mean = get_mean(mean_type=mean_type, mean_params=sm)
            mean_vals = mean(tgrid)

            # legend only for the first line being drawn
            if i == 0:
                ax.plot(
                    tgrid,
                    mean_vals,
                    color="orange",
                    alpha=0.1,
                    label="Mean function posterior draws",
                )
            else:
                ax.plot(tgrid, mean_vals, color="orange", alpha=0.1)

        # update legend opacity
        leg = ax.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)

        return ax

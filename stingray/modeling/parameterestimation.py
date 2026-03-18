__all__ = ["OptimizationResults", "ParameterEstimation", "PSDParEst", "SamplingResults"]


import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# check whether emcee is installed for sampling
try:
    import emcee

    can_sample = True
except ImportError:
    can_sample = False

try:
    import corner

    use_corner = True
except ImportError:
    use_corner = False

from multiprocessing import Pool

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import scipy.signal
import copy

try:
    from statsmodels.tools.numdiff import approx_hess

    comp_hessian = True
except ImportError:
    comp_hessian = False

from stingray.modeling.posterior import (
    Posterior,
    PSDPosterior,
    LogLikelihood,
    PSDLogLikelihood,
    logmin,
    fitter_to_model_params,
)
from stingray.loggingconfig import CustomFormatter, setup_logger

logger = setup_logger()


class OptimizationResults(object):
    """
    Helper class that will contain the results of the regression.
    Less fiddly than a dictionary.

    Parameters
    ----------
    lpost: instance of :class:`Posterior` or one of its subclasses
        The object containing the function that is being optimized
        in the regression

    res: instance of ``scipy.OptimizeResult``
        The object containing the results from a optimization run

    neg : bool, optional, default ``True``
        A flag that sets whether the log-likelihood or negative log-likelihood
        is being used

    log : a logging.getLogger() object, default None
        You can pass a pre-defined object for logging, else a new
        logger will be instantiated

    Attributes
    ----------
    result : float
        The result of the optimization, i.e. the function value at the
        minimum that the optimizer found

    p_opt : iterable
        The list of parameters at the minimum found by the optimizer

    model : ``astropy.models.Model`` instance
        The parametric model fit to the data

    cov : numpy.ndarray
        The covariance matrix for the parameters, has shape ``(len(p_opt), len(p_opt))``

    err : numpy.ndarray
        The standard deviation of the parameters, derived from the diagonal of ``cov``.
        Has the same shape as ``p_opt``

    mfit : numpy.ndarray
        The values of the model for all ``x``

    deviance : float
        The deviance, calculated as ``-2*log(likelihood)``

    aic : float
        The Akaike Information Criterion, derived from the log(likelihood) and often used
        in model comparison between non-nested models;
        For more details, see [#]_

    bic : float
        The Bayesian Information Criterion, derived from the log(likelihood) and often used
        in model comparison between non-nested models;
        For more details, see [#]_

    merit : float
        sum of squared differences between data and model, normalized by the
        model values

    dof : int
        The number of degrees of freedom in the problem, defined as the number of
        data points - the number of parameters

    sexp : int
        ``2*(number of parameters)*(number of data points)``

    ssd : float
        ``sqrt(2*(sexp))``, expected sum of data-model residuals

    sobs : float
        sum of data-model residuals

    References
    ----------
    .. [#] https://doi.org/10.1109/TAC.1974.1100705
    .. [#] https://projecteuclid.org/euclid.aos/1176344136

    """

    def __init__(self, lpost, res, neg=True, log=None):
        self.neg = neg
        self.result = res.fun
        self.p_opt = np.atleast_1d(res.x)
        self.model = lpost.model

        if log is None:
            self.log = logging.getLogger("Fitting summary")
            self.log.setLevel(logging.DEBUG)
            if not self.log.handlers:
                ch = logging.StreamHandler()
                formatter = CustomFormatter()
                ch.setFormatter(formatter)
                ch.setLevel(logging.DEBUG)
                self.log.addHandler(ch)

        self._compute_covariance(lpost, res)
        self._compute_model(lpost)
        self._compute_criteria(lpost)
        self._compute_statistics(lpost)

    def _compute_covariance(self, lpost, res):
        """
        Compute the covariance of the parameters using inverse of the Hessian, i.e.
        the second-order derivative of the log-likelihood. Also calculates an estimate
        of the standard deviation in the parameters, using the square root of the diagonal
        of the covariance matrix.

        The Hessian is either estimated directly by the chosen method of fitting, or
        approximated using the ``statsmodel`` ``approx_hess`` function.

        Parameters
        ----------
        lpost: instance of :class:`Posterior` or one of its subclasses
            The object containing the function that is being optimized
            in the regression

        res: instance of ``scipy``'s ``OptimizeResult`` class
            The object containing the results from a optimization run
        """

        if hasattr(res, "hess_inv"):
            if not isinstance(res.hess_inv, np.ndarray):
                self.cov = np.asanyarray(res.hess_inv.todense())
            else:
                self.cov = res.hess_inv

            self.err = np.sqrt(np.diag(self.cov))
        else:
            if comp_hessian:
                # calculate Hessian approximating with finite differences
                self.log.info("Approximating Hessian with finite differences ...")

                phess = approx_hess(np.atleast_1d(self.p_opt), lpost)

                self.cov = np.linalg.inv(phess)
                self.err = np.sqrt(np.diag(np.abs(self.cov)))

            else:
                self.cov = None
                self.err = None

    def _compute_model(self, lpost):
        """
        Compute the values of the best-fit model for all ``x``.

        Parameters
        ----------
        lpost: instance of :class:`Posterior` or one of its subclasses
            The object containing the function that is being optimized
            in the regression
        """
        fitter_to_model_params(lpost.model, self.p_opt)

        self.mfit = lpost.model(lpost.x)

    def _compute_criteria(self, lpost):
        """
        Compute various information criteria useful for model comparison in
        non-nested models.

        Currently implemented are the Akaike Information Criterion [#]_ and the
        Bayesian Information Criterion [#]_.

        Parameters
        ----------
        lpost: instance of :class:`Posterior` or one of its subclasses
            The object containing the function that is being optimized
            in the regression

        References
        ----------
        .. [#] https://doi.org/10.1109/TAC.1974.1100705
        .. [#] https://projecteuclid.org/euclid.aos/1176344136

        """
        if isinstance(lpost, Posterior):
            self.deviance = -2.0 * lpost.loglikelihood(self.p_opt, neg=False)
        elif isinstance(lpost, LogLikelihood):
            self.deviance = 2.0 * self.result

        # Akaike Information Criterion
        self.aic = self.result + 2.0 * self.p_opt.shape[0]

        # Bayesian Information Criterion
        self.bic = self.result + self.p_opt.shape[0] * np.log(lpost.x.shape[0])

        # Deviance Information Criterion
        # TODO: Add Deviance Information Criterion

    def _compute_statistics(self, lpost):
        """
        Compute some useful fit statistics, like the degrees of freedom and the
        figure of merit.

        Parameters
        ----------
        lpost: instance of :class:`Posterior` or one of its subclasses
            The object containing the function that is being optimized
            in the regression
        """
        try:
            self.mfit
        except AttributeError:
            self._compute_model(lpost)

        self.merit = np.sum(((lpost.y - self.mfit) / self.mfit) ** 2.0)
        self.dof = lpost.y.shape[0] - float(self.p_opt.shape[0])
        self.sexp = 2.0 * len(lpost.x) * len(self.p_opt)
        self.ssd = np.sqrt(2.0 * self.sexp)
        self.sobs = np.sum(lpost.y - self.mfit)

    def print_summary(self, lpost):
        """
        Print a useful summary of the fitting procedure to screen or
        a log file.

        Parameters
        ----------
        lpost : instance of :class:`Posterior` or one of its subclasses
            The object containing the function that is being optimized
            in the regression
        """

        self.log.info("The best-fit model parameters plus errors are:")

        fixed = [lpost.model.fixed[n] for n in lpost.model.param_names]
        tied = [lpost.model.tied[n] for n in lpost.model.param_names]
        bounds = [lpost.model.bounds[n] for n in lpost.model.param_names]

        parnames = [n for n, f in zip(lpost.model.param_names, np.logical_or(fixed, tied)) if not f]

        all_parnames = [n for n in lpost.model.param_names]
        for i, par in enumerate(all_parnames):
            self.log.info("{:3}) Parameter {:<20}: ".format(i, par))

            if par in parnames:
                idx = parnames.index(par)

                err_info = " (no error estimate)"
                if self.err is not None:
                    err_info = " +/- {:<20.5f}".format(self.err[idx])
                self.log.info("{:<20.5f}{} ".format(self.p_opt[idx], err_info))
                self.log.info("[{:>10} {:>10}]".format(str(bounds[i][0]), str(bounds[i][1])))
            elif fixed[i]:
                self.log.info("{:<20.5f} (Fixed) ".format(lpost.model.parameters[i]))
            elif tied[i]:
                self.log.info("{:<20.5f} (Tied) ".format(lpost.model.parameters[i]))

        self.log.info("\n")

        self.log.info("Fitting statistics: ")
        self.log.info(" -- number of data points: %i" % (len(lpost.x)))

        try:
            self.deviance
        except AttributeError:
            self._compute_criteria(lpost)

        self.log.info(" -- Deviance [-2 log L] D = %f.3" % self.deviance)
        self.log.info(
            " -- The Akaike Information Criterion of the model is: " + str(self.aic) + "."
        )

        self.log.info(
            " -- The Bayesian Information Criterion of the model is: " + str(self.bic) + "."
        )

        try:
            self.merit
        except AttributeError:
            self._compute_statistics(lpost)

        self.log.info(
            " -- The figure-of-merit function for this model "
            + " is: %f.5f" % self.merit
            + " and the fit for %i dof is %f.3f" % (self.dof, self.merit / self.dof)
        )

        self.log.info(" -- Summed Residuals S = %f.5f" % self.sobs)
        self.log.info(" -- Expected S ~ %f.5 +/- %f.5" % (self.sexp, self.ssd))

        return


class ParameterEstimation(object):
    """
    Parameter estimation of two-dimensional data, either via
    optimization or MCMC.
    Note: optimization with bounds is not supported. If something like
    this is required, define (uniform) priors in the ParametricModel
    instances to be used below.

    Parameters
    ----------
    fitmethod : string, optional, default ``L-BFGS-B``
        Any of the strings allowed in ``scipy.optimize.minimize`` in
        the method keyword. Sets the fit method to be used.

    max_post : bool, optional, default ``True``
        If ``True``, then compute the Maximum-A-Posteriori estimate. If ``False``,
        compute a Maximum Likelihood estimate.
    """

    def __init__(self, fitmethod="BFGS", max_post=True):
        self.fitmethod = fitmethod

        self.max_post = max_post

    def fit(self, lpost, t0, neg=True, scipy_optimize_options=None):
        """
        Do either a Maximum-A-Posteriori (MAP) or Maximum Likelihood (ML)
        fit to the data.

        MAP fits include priors, ML fits do not.

        Parameters
        ----------
        lpost : :class:`Posterior` (or subclass) instance
            and instance of class :class:`Posterior` or one of its subclasses
            that defines the function to be minimized (either in ``loglikelihood``
            or ``logposterior``)

        t0 : {``list`` | ``numpy.ndarray``}
            List/array with set of initial parameters

        neg : bool, optional, default ``True``
            Boolean to be passed to ``lpost``, setting whether to use the
            *negative* posterior or the *negative* log-likelihood. Useful for
            optimization routines, which are generally defined as *minimization* routines.

        scipy_optimize_options : dict, optional, default ``None``
            A dictionary with options for ``scipy.optimize.minimize``,
            directly passed on as keyword arguments.

        Returns
        -------
        res : :class:`OptimizationResults` object
            An object containing useful summaries of the fitting procedure.
            For details, see documentation of class:`OptimizationResults`.
        """

        if not isinstance(lpost, Posterior) and not isinstance(lpost, LogLikelihood):
            raise TypeError("lpost must be a subclass of " "Posterior or LogLikelihoood.")

        newmod = lpost.model.copy()

        p0 = t0

        # p0 will be shorter than t0, if there are any frozen/tied parameters
        # this has to match with the npar attribute.
        if not len(p0) == lpost.npar:
            raise ValueError("Parameter set t0 must be of right " "length for model in lpost.")

        args = (neg,)

        if not scipy_optimize_options:
            scipy_optimize_options = {}

        # different commands for different fitting methods,
        # at least until scipy 0.11 is out
        funcval = 100.0
        i = 0

        while funcval == 100 or funcval == 200 or funcval == 0.0 or not np.isfinite(funcval):
            if i > 20:
                raise RuntimeError("Fitting unsuccessful!")
            # perturb parameters slightly
            t0_p = np.random.multivariate_normal(p0, np.diag(np.abs(p0) / 100.0))

            params = [getattr(newmod, name) for name in newmod.param_names]
            bounds = np.array([p.bounds for p in params if not np.any([p.tied, p.fixed])])

            if any(elem is not None for elem in np.hstack(bounds)) and self.fitmethod not in [
                "L-BFGS-B",
                "TNC",
                "SLSQP",
            ]:
                logger.warning(
                    "Fitting method %s " % self.fitmethod + "cannot incorporate the bounds you set!"
                )

            if any(elem is not None for elem in np.hstack(bounds)) or self.fitmethod not in [
                "L-BFGS-B",
                "TNC",
                "SLSQP",
            ]:
                use_bounds = False
            else:
                use_bounds = True

            # if max_post is True, do the Maximum-A-Posteriori Fit
            if self.max_post:
                if use_bounds:
                    opt = scipy.optimize.minimize(
                        lpost,
                        t0_p,
                        method=self.fitmethod,
                        args=args,
                        tol=1.0e-10,
                        bounds=bounds,
                        **scipy_optimize_options,
                    )

                else:
                    opt = scipy.optimize.minimize(
                        lpost,
                        t0_p,
                        method=self.fitmethod,
                        args=args,
                        tol=1.0e-10,
                        **scipy_optimize_options,
                    )

            # if max_post is False, then do a Maximum Likelihood Fit
            else:
                if isinstance(lpost, Posterior):
                    if use_bounds:
                        # This could be a `Posterior` object
                        opt = scipy.optimize.minimize(
                            lpost.loglikelihood,
                            t0_p,
                            method=self.fitmethod,
                            args=args,
                            tol=1.0e-10,
                            bounds=bounds,
                            **scipy_optimize_options,
                        )
                    else:
                        opt = scipy.optimize.minimize(
                            lpost.loglikelihood,
                            t0_p,
                            method=self.fitmethod,
                            args=args,
                            tol=1.0e-10,
                            **scipy_optimize_options,
                        )

                elif isinstance(lpost, LogLikelihood):
                    if use_bounds:
                        # Except this could be a `LogLikelihood object
                        # In which case, use the evaluate function
                        # if it's not either, give up and break!
                        opt = scipy.optimize.minimize(
                            lpost.evaluate,
                            t0_p,
                            method=self.fitmethod,
                            args=args,
                            tol=1.0e-10,
                            # bounds=bounds,
                            **scipy_optimize_options,
                        )

                    else:
                        opt = scipy.optimize.minimize(
                            lpost.evaluate,
                            t0_p,
                            method=self.fitmethod,
                            args=args,
                            tol=1.0e-10,
                            **scipy_optimize_options,
                        )

            funcval = opt.fun

            if np.isclose(opt.fun, logmin) or np.isclose(opt.fun, 2 * logmin):
                funcval = 100

            i += 1

        res = OptimizationResults(lpost, opt, neg=neg)

        return res

    def compute_lrt(self, lpost1, t1, lpost2, t2, neg=True, max_post=False):
        """
        This function computes the Likelihood Ratio Test between two
        nested models.

        Parameters
        ----------
        lpost1 : object of a subclass of :class:`Posterior`
            The :class:`Posterior` object for model 1

        t1 : iterable
            The starting parameters for model 1

        lpost2 : object of a subclass of :class:`Posterior`
            The :class:`Posterior` object for model 2

        t2 : iterable
            The starting parameters for model 2

        neg : bool, optional, default ``True``
            Boolean flag to decide whether to use the negative log-likelihood
            or log-posterior

        max_post: bool, optional, default ``False``
            If ``True``, set the internal state to do the optimization with the
            log-likelihood rather than the log-posterior.

        Returns
        -------
        lrt : float
            The likelihood ratio for model 2 and model 1

        res1 : OptimizationResults object
            Contains the result of fitting ``lpost1``

        res2 : OptimizationResults object
            Contains the results of fitting ``lpost2``

        """

        self.max_post = max_post

        # fit data with both models
        res1 = self.fit(lpost1, t1, neg=neg)
        res2 = self.fit(lpost2, t2, neg=neg)

        # compute log likelihood ratio as difference between the deviances
        lrt = res1.deviance - res2.deviance

        return lrt, res1, res2

    def sample(
        self,
        lpost,
        t0,
        cov=None,
        nwalkers=500,
        niter=100,
        burnin=100,
        threads=1,
        print_results=True,
        plot=False,
        namestr="test",
        pool=False,
    ):
        """
        Sample the :class:`Posterior` distribution defined in ``lpost`` using MCMC.
        Here we use the ``emcee`` package, but other implementations could
        in principle be used.

        Parameters
        ----------
        lpost : instance of a :class:`Posterior` subclass
            and instance of class :class:`Posterior` or one of its subclasses
            that defines the function to be minimized (either in ``loglikelihood``
            or ``logposterior``)

        t0 : iterable
            list or array containing the starting parameters. Its length
            must match ``lpost.model.npar``.

        nwalkers : int, optional, default 500
            The number of walkers (chains) to use during the MCMC procedure.
            The more walkers are used, the slower the estimation will be, but
            the better the final distribution is likely to be.

        niter : int, optional, default 100
            The number of iterations to run the MCMC chains for. The larger this
            number, the longer the estimation will take, but the higher the
            chance that the walkers have actually converged on the true
            posterior distribution.

        burnin : int, optional, default 100
            The number of iterations to run the walkers before convergence is
            assumed to have occurred. This part of the chain will be discarded
            before sampling from what is then assumed to be the posterior
            distribution desired.

        threads : **DEPRECATED** int, optional, default 1
            The number of threads for parallelization.
            Default is ``1``, i.e. no parallelization
            With the change to the new emcee version 3, threads is
            deprecated. Use the `pool` keyword argument instead.
            This will no longer have any effect.

        print_results : bool, optional, default ``True``
            Boolean flag setting whether the results of the MCMC run should
            be printed to standard output. Default: True

        plot : bool, optional, default ``False``
            Boolean flag setting whether summary plots of the MCMC chains
            should be produced. Default: False

        namestr : str, optional, default ``test``
            Optional string for output file names for the plotting.

        pool : bool, default False
            If True, use pooling to parallelize the operation.

        Returns
        -------

        res : class:`SamplingResults` object
            An object of class :class:`SamplingResults` summarizing the
            results of the MCMC run.

        """

        if threads > 1:
            raise DeprecationWarning("Keyword 'threads' is deprecated. Please use 'pool' instead.")

        if not can_sample:
            raise ImportError("emcee not installed! Can't sample!")

        ndim = len(t0)

        if cov is None:
            # do a MAP fitting step to find good starting positions for
            # the sampler
            res = self.fit(lpost, t0, neg=True)
            cov = res.cov
        # sample random starting positions for each walker from
        # a multivariate Gaussian
        p0 = np.array([np.random.multivariate_normal(t0, cov) for i in range(nwalkers)])
        if pool:
            with Pool() as pooling:
                # initialize the sampler
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, args=[False], pool=pooling)

                # run the burn-in
                pos, prob, state = sampler.run_mcmc(p0, burnin)

                sampler.reset()

                state = emcee.State(pos, prob, random_state=state)
                # do the actual MCMC run

                _ = sampler.run_mcmc(initial_state=state, nsteps=niter)
        else:
            # initialize the sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, args=[False])

            # run the burn-in
            pos, prob, state = sampler.run_mcmc(p0, burnin)

            sampler.reset()
            state = emcee.State(pos, prob, random_state=state)

            # do the actual MCMC run
            _ = sampler.run_mcmc(initial_state=state, nsteps=niter)

        res = SamplingResults(sampler)

        if print_results:
            res.print_results()

        if plot:
            fig = res.plot_results(fig=None, save_plot=True, filename=namestr + "_corner.pdf")

        return res

    def _generate_model(self, lpost, pars):
        """
        Helper function that generates a fake PSD similar to the
        one in the data, but with different parameters.

        Parameters
        ----------
        lpost : instance of a :class:`Posterior` or :class:`LogLikelihood` subclass
            The object containing the relevant information about the
            data and the model

        pars : iterable
            A list of parameters to be passed to ``lpost.model`` in order
            to generate a model data set.

        Returns
        -------
        model_data : numpy.ndarray
            An array of model values for each bin in ``lpost.x``

        """

        assert isinstance(lpost, LogLikelihood) or isinstance(lpost, Posterior), (
            "lpost must be of type LogLikelihood or Posterior or one of its " "subclasses!"
        )

        # assert pars is of correct length
        assert len(pars) == lpost.npar, "pars must be a list " "of %i parameters" % lpost.npar
        # get the model
        m = lpost.model

        # reset the parameters
        fitter_to_model_params(m, pars)

        # make a model spectrum
        model_data = lpost.model(lpost.x)

        return model_data

    @staticmethod
    def _compute_pvalue(obs_val, sim):
        """
        Compute the p-value given an observed value of a test statistic
        and some simulations of that same test statistic.

        Parameters
        ----------
        obs_value : float
            The observed value of the test statistic in question

        sim: iterable
            A list or array of simulated values for the test statistic

        Returns
        -------
        pval : float in range [0, 1]
            The p-value for the test statistic given the simulations.

        """
        # cast the simulations as a numpy array
        sim = np.array(sim)

        # find all simulations that are larger than
        # the observed value
        ntail = sim[sim > obs_val].shape[0]

        # divide by the total number of simulations
        pval = float(ntail) / float(sim.shape[0])

        return pval

    def simulate_lrts(self, s_all, lpost1, t1, lpost2, t2, max_post=True, seed=None):
        """
        Simulate likelihood ratios.
        For details, see definitions in the subclasses that implement this
        task.
        """
        raise NotImplementedError(
            "The behaviour of `simulate_lrts` should be defined "
            "in the subclass appropriate for your problem, not in "
            "this super class!"
        )

    def calibrate_lrt(
        self,
        lpost1,
        t1,
        lpost2,
        t2,
        sample=None,
        neg=True,
        max_post=False,
        nsim=1000,
        niter=200,
        nwalkers=500,
        burnin=200,
        namestr="test",
        seed=None,
    ):
        """Calibrate the outcome of a Likelihood Ratio Test via MCMC.

        In order to compare models via likelihood ratio test, one generally
        aims to compute a p-value for the null hypothesis (generally the
        simpler model). There are two special cases where the theoretical
        distribution used to compute that p-value analytically given the
        observed likelihood ratio (a chi-square distribution) is not
        applicable:

        * the models are not nested (i.e. Model 1 is not a special, simpler
          case of Model 2),
        * the parameter values fixed in Model 2 to retrieve Model 1 are at the
          edges of parameter space (e.g. if one must set, say, an amplitude to
          zero in order to remove a component in the more complex model, and
          negative amplitudes are excluded a priori)

        In these cases, the observed likelihood ratio must be calibrated via
        simulations of the simpler model (Model 1), using MCMC to take into
        account the uncertainty in the parameters. This function does
        exactly that: it computes the likelihood ratio for the observed data,
        and produces simulations to calibrate the likelihood ratio and
        compute a p-value for observing the data under the assumption that
        Model 1 istrue.

        If ``max_post=True``, the code will use MCMC to sample the posterior
        of the parameters and simulate fake data from there.

        If ``max_post=False``, the code will use the covariance matrix derived
        from the fit to simulate data sets for comparison.

        Parameters
        ----------
        lpost1 : object of a subclass of :class:`Posterior`
            The :class:`Posterior` object for model 1

        t1 : iterable
            The starting parameters for model 1

        lpost2 : object of a subclass of :class:`Posterior`
            The :class:`Posterior` object for model 2

        t2 : iterable
            The starting parameters for model 2

        neg : bool, optional, default ``True``
            Boolean flag to decide whether to use the negative
            log-likelihood or log-posterior

        max_post: bool, optional, default ``False``
            If ``True``, set the internal state to do the optimization with the
            log-likelihood rather than the log-posterior.

        Returns
        -------
        pvalue : float [0,1]
            p-value 'n stuff
        """

        # compute the observed likelihood ratio
        lrt_obs, res1, res2 = self.compute_lrt(lpost1, t1, lpost2, t2, neg=neg, max_post=max_post)

        rng = np.random.RandomState(seed)

        if sample is None:
            # simulate parameter sets from the simpler model
            if not max_post:
                # using Maximum Likelihood, so I'm going to simulate parameters
                # from a multivariate Gaussian

                # set up the distribution
                mvn = scipy.stats.multivariate_normal(mean=res1.p_opt, cov=res1.cov, seed=seed)

                # sample parameters
                s_all = mvn.rvs(size=nsim)
                if lpost1.npar == 1:
                    s_all = np.atleast_2d(s_all).T

            else:
                # sample the :class:`Posterior` using MCMC
                s_mcmc = self.sample(
                    lpost1,
                    res1.p_opt,
                    cov=res1.cov,
                    nwalkers=nwalkers,
                    niter=niter,
                    burnin=burnin,
                    namestr=namestr,
                )

                # pick nsim samples out of the :class:`Posterior` sample
                s_all = s_mcmc.samples[rng.choice(s_mcmc.samples.shape[0], nsim, replace=False)]

                # if lpost1.npar == 1:
                #    s_all = np.atleast_2d(s_all).T

        else:
            s_all = sample[rng.choice(sample.shape[0], nsim, replace=False)]

        # simulate LRTs
        # this method is defined in the subclasses!
        lrt_sim = self.simulate_lrts(s_all, lpost1, t1, lpost2, t2, seed=seed)
        # now I can compute the p-value:
        pval = ParameterEstimation._compute_pvalue(lrt_obs, lrt_sim)

        return pval


class SamplingResults(object):
    """
    Helper class that will contain the results of the sampling
    in a handy format.

    Less fiddly than a dictionary.

    Parameters
    ----------
    sampler: ``emcee.EnsembleSampler`` object
        The object containing the sampler that's done all the work.

    ci_min: float out of [0,100]
        The lower bound percentile for printing credible intervals
        on the parameters

    ci_max: float out of [0,100]
        The upper bound percentile for printing credible intervals
        on the parameters
    log : a logging.getLogger() object, default None
        You can pass a pre-defined object for logging, else a new
        logger will be instantiated

    Attributes
    ----------
    samples : numpy.ndarray
        An array of samples from the MCMC run, including all chains
        flattened into one long (``nwalkers*niter``, ``ndim``) array

    nwalkers : int
        The number of chains used in the MCMC procedure

    niter : int
        The number of MCMC iterations in each chain

    ndim : int
        The dimensionality of the problem, i.e. the number of
        parameters in the model

    acceptance : float
        The mean acceptance ratio, calculated over all chains

    L : float
        The product of acceptance ratio and number of samples

    acor : float
        The autocorrelation length for the chains; should be shorter
        than the chains themselves for independent sampling

    rhat : float
        weighted average of between-sequence variance and within-sequence
        variance; Gelman-Rubin convergence statistic [#]_

    mean : numpy.ndarray
        An array of size ``ndim``, with the posterior means of the parameters
        derived from the MCMC chains

    std : numpy.ndarray
        An array of size ``ndim`` with the posterior standard deviations of
        the parameters derived from the MCMC chains

    ci : numpy.ndarray
        An array of shape ``(ndim, 2)`` containing the lower and upper bounds
        of the credible interval (the Bayesian equivalent of the confidence
        interval) for each parameter using the bounds set by ``ci_min`` and ``ci_max``

    References
    ----------
    .. [#] https://projecteuclid.org/euclid.ss/1177011136
    """

    def __init__(self, sampler, ci_min=5, ci_max=95, log=None):
        if log is None:
            self.log = logging.getLogger("MCMC summary")
            self.log.setLevel(logging.DEBUG)

            if not self.log.handlers:
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                self.log.addHandler(ch)

        # store all the samples
        self.samples = sampler.get_chain(flat=True)

        chain_dims = sampler.get_chain().shape
        self.nwalkers = float(chain_dims[0])
        self.niter = float(chain_dims[1])

        # store number of dimensions
        self.ndim = chain_dims[2]

        # compute and store acceptance fraction
        self.acceptance = np.nanmean(sampler.acceptance_fraction)
        self.L = self.acceptance * self.samples.shape[0]

        self._check_convergence(sampler)
        self._infer(ci_min, ci_max)

    def _check_convergence(self, sampler):
        """
        Compute common statistics for convergence of the MCMC
        chains. While you can never be completely sure that your chains
        converged, these present reasonable heuristics to give an
        indication whether convergence is very far off or reasonably close.

        Currently implemented are the autocorrelation time [#]_ and the
        Gelman-Rubin convergence criterion [#]_.

        Parameters
        ----------
        sampler : an ``emcee.EnsembleSampler`` object

        References
        ----------
        .. [#] https://arxiv.org/abs/1202.3665
        .. [#] https://projecteuclid.org/euclid.ss/1177011136
        """

        # compute and store autocorrelation time
        try:
            self.acor = sampler.get_autocorr_time()
        except emcee.autocorr.AutocorrError:
            self.log.info("Chains too short to compute autocorrelation lengths.")

        self.rhat = self._compute_rhat(sampler)

    def _compute_rhat(self, sampler):
        """
        Compute Gelman-Rubin convergence criterion [#]_.

        Parameters
        ----------
        sampler : an `emcee.EnsembleSampler` object

        References
        ----------
        .. [#] https://projecteuclid.org/euclid.ss/1177011136
        """
        chain = sampler.get_chain()
        # between-sequence variance
        mean_samples_iter = np.nanmean(chain, axis=1)

        # mean over the means over iterations: (self.ndim)
        mean_samples = np.nanmean(chain, axis=(0, 1))

        # now compute between-sequence variance
        bb = (self.niter / (self.nwalkers - 1)) * np.sum(
            (mean_samples_iter - mean_samples) ** 2.0, axis=0
        )

        # compute variance of each chain
        var_samples = np.nanvar(chain, axis=1)

        # compute mean of variance
        ww = np.nanmean(var_samples, axis=0)

        # compute weighted average of ww and bb:
        rhat = ((self.niter - 1) / self.niter) * ww + (1 / self.niter) * bb

        return rhat

    def _infer(self, ci_min=5, ci_max=95):
        """
        Infer the :class:`Posterior` means, standard deviations and credible intervals
        (i.e. the Bayesian equivalent to confidence intervals) from the :class:`Posterior` samples
        for each parameter.

        Parameters
        ----------
        ci_min : float
            Lower bound to the credible interval, given as percentage between
            0 and 100

        ci_max : float
            Upper bound to the credible interval, given as percentage between
            0 and 100
        """
        self.mean = np.mean(self.samples, axis=0)
        self.std = np.std(self.samples, axis=0)
        self.ci = np.percentile(self.samples, [ci_min, ci_max], axis=0)

    def print_results(self):
        """
        Print results of the MCMC run on screen or to a log-file.


        """

        self.log.info("-- The acceptance fraction is: %f.5" % self.acceptance)
        try:
            self.log.info("-- The autocorrelation time is: {}".format(self.acor))
        except AttributeError:
            pass

        self.log.info("R_hat for the parameters is: " + str(self.rhat))

        self.log.info("-- Posterior Summary of Parameters: \n")
        self.log.info("parameter \t mean \t\t sd \t\t 5% \t\t 95% \n")
        self.log.info("---------------------------------------------\n")
        for i in range(self.ndim):
            self.log.info(
                "theta["
                + str(i)
                + "] \t "
                + str(self.mean[i])
                + "\t"
                + str(self.std[i])
                + "\t"
                + str(self.ci[0, i])
                + "\t"
                + str(self.ci[1, i])
                + "\n"
            )

        return

    def plot_results(self, nsamples=1000, fig=None, save_plot=False, filename="test.pdf"):
        """
        Plot some results in a triangle plot.
        If installed, will use [corner]_
        for the plotting, if not,
        uses its own code to make a triangle plot.

        By default, this method returns a ``matplotlib.Figure`` object, but
        if ``save_plot=True`` the plot can be saved to file automatically,

        Parameters
        ----------

        nsamples : int, default 1000
            The maximum number of samples used for plotting.

        fig : matplotlib.Figure instance, default None
            If created externally, you can pass a Figure instance to this method.
            If none is passed, the method will create one internally.

        save_plot : bool, default ``False``
            If ``True`` save the plot to file with a file name specified by the
            keyword ``filename``. If ``False`` just return the ``Figure`` object

        filename : str
            Name of the output file with the figure

        References
        ----------
        .. [corner] https://github.com/dfm/corner.py
        """
        if use_corner:
            fig = corner.corner(
                self.samples,
                labels=None,
                fig=fig,
                bins=int(20),
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_args={"fontsize": 12},
            )

        else:
            if fig is None:
                fig = plt.figure(figsize=(15, 15))

            plt.subplots_adjust(
                top=0.925, bottom=0.025, left=0.025, right=0.975, wspace=0.2, hspace=0.2
            )

            ind_all = np.random.choice(np.arange(self.samples.shape[0]), size=nsamples)
            samples = self.samples[ind_all]
            for i in range(self.ndim):
                for j in range(self.ndim):
                    xmin, xmax = samples[:, j].min(), samples[:, j].max()
                    ymin, ymax = samples[:, i].min(), samples[:, i].max()
                    ax = fig.add_subplot(self.ndim, self.ndim, i * self.ndim + j + 1)

                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.ticklabel_format(style="sci", scilimits=(-2, 2))

                    if i == j:
                        ntemp, binstemp, patchestemp = ax.hist(
                            samples[:, i], 30, density=True, histtype="stepfilled"
                        )
                        ax.axis([ymin, ymax, 0, np.max(ntemp) * 1.2])

                    else:
                        ax.axis([xmin, xmax, ymin, ymax])

                        # make a scatter plot first
                        ax.scatter(samples[:, j], samples[:, i], s=7)
                        # then add contours
                        xmin, xmax = samples[:, j].min(), samples[:, j].max()
                        ymin, ymax = samples[:, i].min(), samples[:, i].max()

                        # Perform Kernel density estimate on data
                        try:
                            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                            positions = np.vstack([xx.ravel(), yy.ravel()])
                            values = np.vstack([samples[:, j], samples[:, i]])
                            kernel = scipy.stats.gaussian_kde(values)
                            zz = np.reshape(kernel(positions).T, xx.shape)

                            ax.contour(xx, yy, zz, 7)
                        except ValueError:
                            logger.info("Not making contours.")

        if save_plot:
            plt.savefig(filename, format="pdf")

        return fig


class PSDParEst(ParameterEstimation):
    """
    Parameter estimation for parametric modelling of power spectra.

    This class contains functionality that allows parameter estimation
    and related tasks that involve fitting a parametric model to an
    (averaged) power spectrum.

    Parameters
    ----------
    ps : class:`stingray.Powerspectrum` or class:`stingray.AveragedPowerspectrum` object
        The power spectrum to be modelled

    fitmethod : str, optional, default ``BFGS``
        A string allowed by ``scipy.optimize.minimize`` as a valid
        fitting method

    max_post : bool, optional, default ``True``
        If ``True``, do a Maximum-A-Posteriori (MAP) fit, i.e. fit with
        priors, otherwise do a Maximum Likelihood fit instead

    """

    def __init__(self, ps, fitmethod="BFGS", max_post=True):
        self.ps = ps
        ParameterEstimation.__init__(self, fitmethod=fitmethod, max_post=max_post)

    def fit(self, lpost, t0, neg=True, scipy_optimize_options=None):
        """
        Do either a Maximum-A-Posteriori (MAP) or Maximum Likelihood (ML)
        fit to the power spectrum.

        MAP fits include priors, ML fits do not.

        Parameters
        ----------
        lpost : :class:`stingray.modeling.PSDPosterior` object
            An instance of class :class:`stingray.modeling.PSDPosterior` that defines the
            function to be minimized (either in ``loglikelihood`` or ``logposterior``)

        t0 : {list | numpy.ndarray}
            List/array with set of initial parameters

        neg : bool, optional, default ``True``
            Boolean to be passed to ``lpost``, setting whether to use the
            *negative* posterior or the *negative* log-likelihood.

        scipy_optimize_options : dict, optional, default None
            A dictionary with options for ``scipy.optimize.minimize``,
            directly passed on as keyword arguments.

        Returns
        -------
        res : :class:`OptimizationResults` object
            An object containing useful summaries of the fitting procedure.
            For details, see documentation of :class:`OptimizationResults`.
        """

        self.lpost = lpost

        res = ParameterEstimation.fit(
            self, self.lpost, t0, neg=neg, scipy_optimize_options=scipy_optimize_options
        )

        res.maxpow, res.maxfreq, res.maxind = self._compute_highest_outlier(self.lpost, res)

        return res

    def sample(
        self,
        lpost,
        t0,
        cov=None,
        nwalkers=500,
        niter=100,
        burnin=100,
        threads=1,
        print_results=True,
        plot=False,
        namestr="test",
    ):
        """
        Sample the posterior distribution defined in ``lpost`` using MCMC.
        Here we use the ``emcee`` package, but other implementations could
        in principle be used.

        Parameters
        ----------
        lpost : instance of a :class:`Posterior` subclass
            and instance of class :class:`Posterior` or one of its subclasses
            that defines the function to be minimized (either in ``loglikelihood``
            or ``logposterior``)

        t0 : iterable
            list or array containing the starting parameters. Its length
            must match ``lpost.model.npar``.

        nwalkers : int, optional, default 500
            The number of walkers (chains) to use during the MCMC procedure.
            The more walkers are used, the slower the estimation will be, but
            the better the final distribution is likely to be.

        niter : int, optional, default 100
            The number of iterations to run the MCMC chains for. The larger this
            number, the longer the estimation will take, but the higher the
            chance that the walkers have actually converged on the true
            posterior distribution.

        burnin : int, optional, default 100
            The number of iterations to run the walkers before convergence is
            assumed to have occurred. This part of the chain will be discarded
            before sampling from what is then assumed to be the posterior
            distribution desired.

        threads : int, optional, default 1
            The number of threads for parallelization.
            Default is ``1``, i.e. no parallelization

        print_results : bool, optional, default True
            Boolean flag setting whether the results of the MCMC run should
            be printed to standard output

        plot : bool, optional, default False
            Boolean flag setting whether summary plots of the MCMC chains
            should be produced

        namestr : str, optional, default ``test``
            Optional string for output file names for the plotting.

        Returns
        -------

        res : :class:`SamplingResults` object
            An object containing useful summaries of the
            sampling procedure. For details see documentation of :class:`SamplingResults`.

        """
        self.lpost = lpost

        if cov is None:
            fit_res = ParameterEstimation.fit(self, self.lpost, t0, neg=True)
            cov = fit_res.cov
            t0 = fit_res.p_opt

        res = ParameterEstimation.sample(
            self,
            self.lpost,
            t0,
            cov=cov,
            nwalkers=nwalkers,
            niter=niter,
            burnin=burnin,
            threads=threads,
            print_results=print_results,
            plot=plot,
            namestr=namestr,
        )

        return res

    def _generate_data(self, lpost, pars, rng=None):
        """
        Generate a fake power spectrum from a model.

        Parameters
        ----------
        lpost : instance of a :class:`Posterior` or :class:`LogLikelihood` subclass
            The object containing the relevant information about the
            data and the model

        pars : iterable
            A list of parameters to be passed to ``lpost.model`` in order
            to generate a model data set.

        Returns
        -------
        sim_ps : :class:`stingray.Powerspectrum` object
            The simulated :class:`Powerspectrum` object

        """
        # create own random state object
        if rng is None:
            rng = np.random.RandomState(None)

        model_spectrum = self._generate_model(lpost, pars)

        # use chi-square distribution to get fake data
        model_powers = (
            model_spectrum
            * rng.chisquare(2 * self.ps.m, size=model_spectrum.shape[0])
            / (2.0 * self.ps.m)
        )

        sim_ps = copy.copy(self.ps)

        sim_ps.power = model_powers

        return sim_ps

    def simulate_lrts(self, s_all, lpost1, t1, lpost2, t2, seed=None):
        """
        Simulate likelihood ratios for two given models based on MCMC samples
        for the simpler model (i.e. the null hypothesis).

        Parameters
        ----------
        s_all : numpy.ndarray of shape ``(nsamples, lpost1.npar)``
            An array with MCMC samples derived from the null hypothesis model in
            ``lpost1``. Its second dimension must match the number of free
            parameters in ``lpost1.model``.

        lpost1 : :class:`LogLikelihood` or :class:`Posterior` subclass object
            Object containing the null hypothesis model

        t1 : iterable of length ``lpost1.npar``
            A starting guess for fitting the model in ``lpost1``

        lpost2 : :class:`LogLikelihood` or :class:`Posterior` subclass object
            Object containing the alternative hypothesis model

        t2 : iterable of length ``lpost2.npar``
            A starting guess for fitting the model in ``lpost2``

        max_post : bool, optional, default ``True``
            If ``True``, then ``lpost1`` and ``lpost2`` should be :class:`Posterior` subclass
            objects; if ``False``, then ``lpost1`` and ``lpost2`` should be
            :class:`LogLikelihood` subclass objects

        seed : int, optional default ``None``
            A seed to initialize the ``numpy.random.RandomState`` object to be
            passed on to ``_generate_data``. Useful for producing exactly
            reproducible results

        Returns
        -------
        lrt_sim : numpy.ndarray
            An array with the simulated likelihood ratios for the simulated
            data
        """

        assert lpost1.__class__ == lpost2.__class__, (
            "Both LogLikelihood or " "Posterior objects must be " "of the same class!"
        )

        nsim = s_all.shape[0]
        lrt_sim = np.zeros(nsim)

        rng = np.random.RandomState(seed)

        # now I can loop over all simulated parameter sets to generate a PSD
        for i, s in enumerate(s_all):
            # generate fake PSD
            sim_ps = self._generate_data(lpost1, s, rng)

            neg = True

            # make LogLikelihood objects for both:
            if isinstance(lpost1, LogLikelihood):
                sim_lpost1 = PSDLogLikelihood(
                    sim_ps.freq, sim_ps.power, model=lpost1.model, m=sim_ps.m
                )
                sim_lpost2 = PSDLogLikelihood(
                    sim_ps.freq, sim_ps.power, model=lpost2.model, m=sim_ps.m
                )
                max_post = False
            else:
                # make a :class:`Posterior` object
                sim_lpost1 = PSDPosterior(sim_ps.freq, sim_ps.power, lpost1.model, m=sim_ps.m)
                sim_lpost1.logprior = lpost1.logprior

                sim_lpost2 = PSDPosterior(sim_ps.freq, sim_ps.power, lpost2.model, m=sim_ps.m)

                sim_lpost2.logprior = lpost2.logprior
                max_post = True

            parest_sim = PSDParEst(sim_ps, max_post=max_post, fitmethod=self.fitmethod)

            try:
                lrt_sim[i], _, _ = parest_sim.compute_lrt(
                    sim_lpost1, t1, sim_lpost2, t2, neg=neg, max_post=max_post
                )
            except RuntimeError:
                logger.warning("Fitting was unsuccessful. " "Skipping this simulation!")
                continue

        return lrt_sim

    def calibrate_highest_outlier(
        self,
        lpost,
        t0,
        sample=None,
        max_post=False,
        nsim=1000,
        niter=200,
        nwalkers=500,
        burnin=200,
        namestr="test",
        seed=None,
    ):
        r"""
        Calibrate the highest outlier in a data set using MCMC-simulated
        power spectra.

        In short, the procedure does a MAP fit to the data, computes the
        statistic

        .. math::

            \max{(T_R = 2(\mathrm{data}/\mathrm{model}))}

        and then does an MCMC run using the data and the model, or generates parameter samples
        from the likelihood distribution using the derived covariance in a Maximum Likelihood
        fit.
        From the (posterior) samples, it generates fake power spectra. Each fake spectrum is fit
        in the same way as the data, and the highest data/model outlier extracted as for the data.
        The observed value of :math:`T_R` can then be directly compared to the simulated
        distribution of :math:`T_R` values in order to derive a p-value of the null
        hypothesis that the observed :math:`T_R` is compatible with being generated by
        noise.

        Parameters
        ----------
        lpost : :class:`stingray.modeling.PSDPosterior` object
            An instance of class :class:`stingray.modeling.PSDPosterior` that defines the
            function to be minimized (either in ``loglikelihood`` or ``logposterior``)

        t0 : {list | numpy.ndarray}
            List/array with set of initial parameters

        sample : :class:`SamplingResults` instance, optional, default ``None``
            If a sampler has already been run, the :class:`SamplingResults` instance can be
            fed into this method here, otherwise this method will run a sampler
            automatically

        max_post: bool, optional, default ``False``
            If ``True``, do MAP fits on the power spectrum to find the highest data/model outlier
            Otherwise, do a Maximum Likelihood fit. If ``True``, the simulated power spectra will
            be generated from an MCMC run, otherwise the method will employ the approximated
            covariance matrix for the parameters derived from the likelihood surface to generate
            samples from that likelihood function.

        nsim : int, optional, default ``1000``
            Number of fake power spectra to simulate from the posterior sample. Note that this
            number sets the resolution of the resulting p-value. For ``nsim=1000``, the highest
            resolution that can be achieved is :math:`10^{-3}`.

        niter : int, optional, default 200
            If ``sample`` is ``None``, this variable will be used to set the number of steps in the
            MCMC procedure *after* burn-in.

        nwalkers : int, optional, default 500
             If ``sample`` is ``None``, this variable will be used to set the number of MCMC chains
             run in parallel in the sampler.

        burnin : int, optional, default 200
             If ``sample`` is ``None``, this variable will be used to set the number of burn-in steps
             to be discarded in the initial phase of the MCMC run

        namestr : str, optional, default ``test``
            A string to be used for storing MCMC output and plots to disk

        seed : int, optional, default ``None``
            An optional number to seed the random number generator with, for reproducibility of
            the results obtained with this method.

        Returns
        -------
        pval : float
            The p-value that the highest data/model outlier is produced by random noise, calibrated
            using simulated power spectra from an MCMC run.

        References
        ----------
        For more details on the procedure employed here, see

            * Vaughan, 2010: https://arxiv.org/abs/0910.2706
            * Huppenkothen et al, 2013: https://arxiv.org/abs/1212.1011
        """
        # fit the model to the data
        res = self.fit(lpost, t0, neg=True)

        rng = np.random.RandomState(seed)

        # find the highest data/model outlier:
        out_high, _, _ = self._compute_highest_outlier(lpost, res)
        # simulate parameter sets from the simpler model
        if not max_post:
            # using Maximum Likelihood, so I'm going to simulate parameters
            # from a multivariate Gaussian

            # set up the distribution
            mvn = scipy.stats.multivariate_normal(mean=res.p_opt, cov=res.cov, seed=seed)

            if lpost.npar == 1:
                # sample parameters
                s_all = np.atleast_2d(mvn.rvs(size=nsim)).T

            else:
                s_all = mvn.rvs(size=nsim)

        else:
            if sample is None:
                # sample the :class:`Posterior` using MCMC
                sample = self.sample(
                    lpost,
                    res.p_opt,
                    cov=res.cov,
                    nwalkers=nwalkers,
                    niter=niter,
                    burnin=burnin,
                    namestr=namestr,
                )

            # pick nsim samples out of the :class:`Posterior` sample
            s_all = sample.samples[rng.choice(sample.samples.shape[0], nsim, replace=False)]

        # simulate LRTs
        # this method is defined in the subclasses!
        out_high_sim = self.simulate_highest_outlier(s_all, lpost, t0, max_post=max_post, seed=seed)
        # now I can compute the p-value:
        pval = ParameterEstimation._compute_pvalue(out_high, out_high_sim)

        return pval

    def simulate_highest_outlier(self, s_all, lpost, t0, max_post=True, seed=None):
        r"""
        Simulate :math:`n` power spectra from a model and then find the highest
        data/model outlier in each.

        The data/model outlier is defined as

        .. math::

             \max{(T_R = 2(\mathrm{data}/\mathrm{model}))} .

        Parameters
        ----------
        s_all : numpy.ndarray
            A list of parameter values derived either from an approximation of the
            likelihood surface, or from an MCMC run. Has dimensions ``(n, ndim)``, where
            ``n`` is the number of simulated power spectra to generate, and ``ndim`` the
            number of model parameters.

        lpost : instance of a :class:`Posterior` subclass
            an instance of class :class:`Posterior` or one of its subclasses
            that defines the function to be minimized (either in ``loglikelihood``
            or ``logposterior``)

        t0 : iterable
            list or array containing the starting parameters. Its length
            must match ``lpost.model.npar``.

        max_post: bool, optional, default ``False``
            If ``True``, do MAP fits on the power spectrum to find the highest data/model outlier
            Otherwise, do a Maximum Likelihood fit. If True, the simulated power spectra will
            be generated from an MCMC run, otherwise the method will employ the approximated
            covariance matrix for the parameters derived from the likelihood surface to generate
            samples from that likelihood function.

        seed : int, optional, default ``None``
            An optional number to seed the random number generator with, for reproducibility of
            the results obtained with this method.

        Returns
        -------
        max_y_all : numpy.ndarray
            An array of maximum outliers for each simulated power spectrum
        """
        # the number of simulations
        nsim = s_all.shape[0]

        # empty array for the simulation results
        max_y_all = np.zeros(nsim)

        rng = np.random.RandomState(seed)

        # now I can loop over all simulated parameter sets to generate a PSD
        for i, s in enumerate(s_all):
            # generate fake PSD
            sim_ps = self._generate_data(lpost, s, rng=rng)

            # make LogLikelihood objects for both:
            if not max_post:
                sim_lpost = PSDLogLikelihood(
                    sim_ps.freq, sim_ps.power, model=lpost.model, m=sim_ps.m
                )
            else:
                # make a :class:`Posterior` object
                sim_lpost = PSDPosterior(sim_ps.freq, sim_ps.power, lpost.model, m=sim_ps.m)
                sim_lpost.logprior = lpost.logprior

            parest_sim = PSDParEst(sim_ps, max_post=max_post)

            try:
                res = parest_sim.fit(sim_lpost, t0, neg=True)
                max_y, maxfreq, maxind = self._compute_highest_outlier(sim_lpost, res, nmax=1)
                max_y_all[i] = max_y[0]
            except RuntimeError:
                logger.warning("Fitting unsuccessful! " "Skipping this simulation!")
                continue

        return np.hstack(max_y_all)

    def _compute_highest_outlier(self, lpost, res, nmax=1):
        r"""
        Auxiliary method calculating the highest outlier statistic in
        a power spectrum.

        The maximum data/model outlier is defined as

        .. math::

             \max{(T_R = 2(\mathrm{data}/\mathrm{model}))}

        Parameters
        ----------
        lpost : instance of a :class:`Posterior` subclass
            and instance of class :class:`Posterior` or one of its subclasses
            that defines the function to be minimized (either in ``loglikelihood``
            or ``logposterior``)

        res : :class:`OptimizationResults` object
            An object containing useful summaries of the fitting procedure.
            For details, see documentation of :class:`OptimizationResults`.

        nmax : int, optional, default ``1``
            The number of maxima to extract from the power spectra. By default,
            only the highest data/model outlier is extracted. This number allows
            to extract the ``nmax`` highest outliers, useful when looking for
            multiple signals in a power spectrum.

        Returns
        -------
        max_y : {float | numpy.ndarray}
            The ``nmax`` highest data/model outliers

        max_x : {float | numpy.ndarray}
            The frequencies corresponding to the outliers in ``max_y``

        max_ind : {int | numpy.ndarray}
            The indices corresponding to the outliers in ``max_y``
        """
        residuals = 2.0 * lpost.y / res.mfit

        ratio_sort = copy.copy(residuals)
        ratio_sort.sort()
        max_y = ratio_sort[-nmax:]

        max_x = np.zeros(max_y.shape[0])
        max_ind = np.zeros(max_y.shape[0])

        for i, my in enumerate(max_y):
            max_x[i], max_ind[i] = self._find_outlier(lpost.x, residuals, my)

        return max_y, max_x, max_ind

    @staticmethod
    def _find_outlier(xdata, ratio, max_y):
        """
        Small auxiliary method that finds the index where an array has
        its maximum, and the corresponding value in ``xdata``.

        Parameters
        ----------
        xdata : numpy.ndarray
            A list of independent variables

        ratio : Numpy.ndarray
            A list of dependent variables corresponding to ``xdata``

        max_y : float
            The maximum value of ``ratio``

        Returns
        -------
        max_x : float
            The value in ``xdata`` corresponding to the entry in ``ratio`` where
            ``ratio == `max_y``

        max_ind : float
            The index of the entry in ``ratio`` where ``ratio == max_y``
        """
        max_ind = np.where(ratio == max_y)[0][0]
        max_x = xdata[max_ind]

        return max_x, max_ind

    def plotfits(self, res1, res2=None, save_plot=False, namestr="test", log=False):
        """
        Plotting method that allows to plot either one or two best-fit models
        with the data.

        Plots a power spectrum with the best-fit model, as well as the data/model
        residuals for each model.

        Parameters
        ----------
        res1 : :class:`OptimizationResults` object
            Output of a successful fitting procedure

        res2 : :class:`OptimizationResults` object, optional, default ``None``
            Optional output of a second successful fitting procedure, e.g. with a
            competing model

        save_plot : bool, optional, default ``False``
            If ``True``, the resulting figure will be saved to a file

        namestr : str, optional, default ``test``
            If ``save_plot`` is ``True``, this string defines the path and file name
            for the output plot

        log : bool, optional, default ``False``
            If ``True``, plot the axes logarithmically.
        """

        # make a figure
        f = plt.figure(figsize=(12, 10))
        # adjust subplots such that the space between the top and bottom
        # of each are zero
        plt.subplots_adjust(hspace=0.0, wspace=0.4)

        # first subplot of the grid, twice as high as the other two
        # This is the periodogram with the two fitted models overplotted
        s1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)

        if log:
            logx = np.log10(self.ps.freq)
            logy = np.log10(self.ps.power)
            logpar1 = np.log10(res1.mfit)

            (p1,) = s1.plot(logx, logy, color="black", drawstyle="steps-mid")
            (p2,) = s1.plot(logx, logpar1, color="blue", lw=2)
            s1.set_xlim([np.min(logx), np.max(logx)])
            s1.set_ylim([np.min(logy) - 1.0, np.max(logy) + 1])
            if self.ps.norm == "leahy":
                s1.set_ylabel("log(Leahy-Normalized Power)", fontsize=18)
            elif self.ps.norm == "rms":
                s1.set_ylabel("log(RMS-Normalized Power)", fontsize=18)
            else:
                s1.set_ylabel("log(Power)", fontsize=18)

        else:
            (p1,) = s1.plot(self.ps.freq, self.ps.power, color="black", drawstyle="steps-mid")
            (p2,) = s1.plot(self.ps.freq, res1.mfit, color="blue", lw=2)

            s1.set_xscale("log")
            s1.set_yscale("log")

            s1.set_xlim([np.min(self.ps.freq), np.max(self.ps.freq)])
            s1.set_ylim([np.min(self.ps.freq) / 10.0, np.max(self.ps.power) * 10.0])

            if self.ps.norm == "leahy":
                s1.set_ylabel("Leahy-Normalized Power", fontsize=18)
            elif self.ps.norm == "rms":
                s1.set_ylabel("RMS-Normalized Power", fontsize=18)
            else:
                s1.set_ylabel("Power", fontsize=18)

        if res2 is not None:
            if log:
                logpar2 = np.log10(res2.mfit)
                (p3,) = s1.plot(logx, logpar2, color="red", lw=2)
            else:
                (p3,) = s1.plot(self.ps.freq, res2.mfit, color="red", lw=2)
            s1.legend([p1, p2, p3], ["data", "model 1 fit", "model 2 fit"])
        else:
            s1.legend([p1, p2], ["data", "model fit"])

        s1.set_title("Periodogram and fits for data set " + namestr, fontsize=18)

        # second subplot: power/model for Power law and straight line
        s2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1)
        pldif = self.ps.power / res1.mfit

        s2.set_ylabel("Residuals, \n first model", fontsize=18)

        if log:
            s2.plot(logx, pldif, color="black", drawstyle="steps-mid")
            s2.plot(logx, np.ones(self.ps.freq.shape[0]), color="blue", lw=2)
            s2.set_xlim([np.min(logx), np.max(logx)])
            s2.set_ylim([np.min(pldif), np.max(pldif)])

        else:
            s2.plot(self.ps.freq, pldif, color="black", drawstyle="steps-mid")
            s2.plot(self.ps.freq, np.ones_like(self.ps.power), color="blue", lw=2)

            s2.set_xscale("log")
            s2.set_yscale("log")
            s2.set_xlim([np.min(self.ps.freq), np.max(self.ps.freq)])
            s2.set_ylim([np.min(pldif), np.max(pldif)])

        if res2 is not None:
            bpldif = self.ps.power / res2.mfit

            # third subplot: power/model for bent power law and straight line
            s3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)

            if log:
                s3.plot(logx, bpldif, color="black", drawstyle="steps-mid")
                s3.plot(logx, np.ones(len(self.ps.freq)), color="red", lw=2)
                s3.axis([np.min(logx), np.max(logx), np.min(bpldif), np.max(bpldif)])
                s3.set_xlabel("log(Frequency) [Hz]", fontsize=18)

            else:
                s3.plot(self.ps.freq, bpldif, color="black", drawstyle="steps-mid")
                s3.plot(self.ps.freq, np.ones(len(self.ps.freq)), color="red", lw=2)
                s3.set_xscale("log")
                s3.set_yscale("log")
                s3.set_xlim([np.min(self.ps.freq), np.max(self.ps.freq)])
                s3.set_ylim([np.min(bpldif), np.max(bpldif)])
                s3.set_xlabel("Frequency [Hz]", fontsize=18)

            s3.set_ylabel("Residuals, \n second model", fontsize=18)

        else:
            if log:
                s2.set_xlabel("log(Frequency) [Hz]", fontsize=18)
            else:
                s2.set_xlabel("Frequency [Hz]", fontsize=18)

        ax = plt.gca()

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(14)

        # make sure xticks are taken from first plots, but don't
        # appear there
        plt.setp(s1.get_xticklabels(), visible=False)

        if save_plot:
            # save figure in png file and close plot device
            plt.savefig(namestr + "_ps_fit.png", format="png")

        return

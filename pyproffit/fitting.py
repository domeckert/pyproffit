import numpy as np
import iminuit
import matplotlib.pyplot as plt

# Generic class to fit data with chi-square
class ChiSquared:
    """
    Class defining a chi-square likelihood based on a surface brightness profile and a model. Let :math:`S_i` be the measured surface brightness in annulus i and :math:`\\sigma_i` the corresponding Gaussian error. The likelihood function to be optimized is

    .. math::

        -2\\log \\mathcal{L} = \\sum_{i=1}^N \\frac{(S_i - f(r_i))^2}{\\sigma_i^2}

    This class is called by the Fitter object when using the method='chi2' option.

    :param model: Model definition. A :class:`pyproffit.models.Model` object defining the model to be used.
    :type model: class:`pyproffit.models.Model`
    :param x: x axis data
    :type x: class:`numpy.ndarray`
    :param dx: x bin size data. dx is defined as half of the total bin size.
    :type dx: class:`numpy.ndarray`
    :param y: y axis data
    :type y: class:`numpy.ndarray`
    :param dy: y error data
    :type dy: class:`numpy.ndarray`
    :param psfmat: PSF convolution matrix
    :type psfmat: class:`numpy.ndarray` , optional
    :param fitlow: Lower fitting boundary in arcmin. If fitlow=None the entire radial range is used, default to None
    :type fitlow: float , optional
    :param fithigh: Upper fitting boundary in arcmin. If fithigh=None the entire radial range is used, default to None
    :type fithigh: float , optional
    """

    errordef = iminuit.Minuit.LEAST_SQUARES

    def __init__(self, model, x , dx, y, dy, psfmat=None, fitlow=None, fithigh=None):
        """
        Constructor of class ChiSquared

        """
        self.model = model  # model predicts y for given x
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy
        fitl = 0.
        fith = 1e10
        if fitlow is not None:
            fitl = fitlow
        if fithigh is not None:
            fith = fithigh
        self.region = np.where(np.logical_and(x>=fitl,x<=fith))
        self.nonz = np.where(dy[self.region]>0.)
        if psfmat is not None:
            self.psfmat = psfmat.T
        else:
            self.psfmat = None
        self.func_code = iminuit.util.make_func_code(iminuit.util.describe(self.model)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        """
        Caller of class ChiSquared

        :param par: Parameter set to be passed to the model
        :return: chi-squared value
        :rtype: float
        """
        ym = self.model(self.x, *par)
        if self.psfmat is not None:
            ym = np.dot(self.psfmat, ym)

        reg = self.region
        nonz = self.nonz
        chi2 = np.sum((self.y[reg][nonz] - ym[reg][nonz])**2/self.dy[reg][nonz]**2)
        return chi2

# Generic class to fit data with C-stat
class Cstat:
    """
    Class defining a C-stat likelihood based on a surface brightness profile and a model. Let :math:`A_i` , :math:`T_i` be the area and the effective exposure time of annulus i. We set :math:`F_{i} = f(r_{i})A_{i}T_{i}` the predicted number of counts in the annulus. The Poisson likelihood is then given by

    .. math::

        -2\\log \\mathcal{L} = 2 \\sum_{i=1}^N F_i - C_i \\log F_i - C_i + C_i \\log C_i

    with :math:`C_i` the observed number of counts in annulus i.

    This class is called by the Fitter object when using the method='cstat' option.

    :param model: Model definition. A :class:`pyproffit.models.Model` object defining the model to be used.
    :type model: class:`pyproffit.models.Model`
    :param x: x axis data
    :type x: numpy.ndarray
    :param counts: counts per bin data
    :type counts: numpy.ndarray
    :param area: bin are in arcmin^2
    :type area: numpy.ndarray
    :param effexp: bin effective exposure in s
    :type effexp: numpy.ndarray
    :param bkgc: number of background counts per bin
    :type bkgc: numpy.ndarray
    :param psfmat: PSF convolution matrix
    :type psfmat: numpy.ndarray
    :param fitlow: Lower fitting boundary in arcmin. If fitlow=None (default) the entire radial range is used
    :type fitlow: float
    :param fithigh: Upper fitting boundary in arcmin. If fithigh=None (default) the entire radial range is used
    :type fithigh: float
    """

    errordef = iminuit.Minuit.LEAST_SQUARES

    def __init__(self, model, x, dx, counts, area, effexp, bkgc, psfmat=None, fitlow=None, fithigh=None):
        """
        Constructor of class Cstat

        """
        self.model = model  # model predicts y for given x
        self.x = x
        self.dx = dx
        self.c = counts
        self.area = area
        self.effexp = effexp
        self.bkgc = bkgc
        fitl = 0.
        fith = 1e10
        if fitlow is not None:
            fitl = fitlow
        if fithigh is not None:
            fith = fithigh
        self.region = np.where(np.logical_and(x>=fitl,x<=fith))
        self.nonz = np.where(counts[self.region]>0.)
        self.isz = np.where(counts[self.region]==0)
        if psfmat is not None:
            self.psfmat = psfmat
        else:
            self.psfmat = None
        self.func_code = iminuit.util.make_func_code(iminuit.util.describe(self.model)[1:])

    def __call__(self, *par):
        """
        Caller for class Cstat

        :param par: Parameter set to be passed to the model
        :return: C-stat value
        :rtype: float
        """
        ym = self.model(self.x, *par)
        if self.psfmat is not None:
            rminus = self.x - self.dx
            rplus = self.x + self.dx
            areatot = np.pi * (rplus ** 2 - rminus ** 2)
            ym = np.dot(self.psfmat, ym * areatot) / areatot

        modcounts = ym*self.area*self.effexp
        mm = modcounts + self.bkgc # model counts
        reg = self.region
        nc = self.c
        nonz = self.nonz
        cstat = 2.*np.sum(mm[reg][nonz]-nc[reg][nonz]*np.log(mm[reg][nonz])-nc[reg][nonz]+nc[reg][nonz]*np.log(nc[reg][nonz])) # normalized C-statistic
        isz = self.isz
        cstat = cstat + 2.*np.sum(mm[reg][isz])
        return cstat


# Class including fitting tool
class Fitter:
    """
    Class containing the tools to fit surface brightness profiles with a model. Sets up the likelihood and optimizes for the parameters.

    :param model: Object of type :class:`pyproffit.models.Model` defining the model to be used.
    :type model: class:`pyproffit.models.Model`
    :param profile: Object of type :class:`pyproffit.profextract.Profile` containing the surface brightness profile to be fitted
    :type profile: class:`pyproffit.profextract.Profile`
    :param method: Likelihood function to be optimized. Available likelihoods are 'chi2' (chi-squared) and 'cstat' (C statistic). Defaults to 'chi2'.
    :type method: str
    :param fitlow: Lower boundary of the active fitting radial range. If fitlow=None the entire range is used. Defaults to None
    :type fitlow: float
    :param fithigh: Upper boundary of the active fitting radial range. If fithigh=None the entire range is used. Defaults to None
    :type fithigh: float
    :param kwargs: List of arguments to be passed to the iminuit library. For instance, setting parameter boundaries, optimization options or fixing parameters.
        See the iminuit documentation: https://iminuit.readthedocs.io/en/stable/index.html
    """

    def __init__(self, model, profile, method='chi2', fitlow=None, fithigh=None, **kwargs):
        """
        Constructor of class Fitter
        """
        self.mod = model
        self.profile = profile
        if profile is None:
            print('Error: No valid profile exists in provided object')
            return

        if profile.psfmat is not None:
            psfmat = np.transpose(profile.psfmat)
        else:
            psfmat = None

        loglike = None
        if method == 'chi2':
            # Define the fitting algorithm
            loglike = ChiSquared(model=model.model,
                              x=profile.bins,
                              dx=profile.ebins,
                              y=profile.profile,
                              dy=profile.eprof,
                              psfmat=psfmat,
                              fitlow=fitlow,
                              fithigh=fithigh)

        elif method == 'cstat':
            if profile.counts is None:
                print('Error: No count profile exists')
                return
            # Define the fitting algorithm
            loglike = Cstat(model=model.model,
                          x=profile.bins,
                          dx=profile.ebins,
                          counts=profile.counts,
                          area=profile.area,
                          effexp=profile.effexp,
                          bkgc=profile.bkgcounts,
                          psfmat=psfmat,
                          fitlow=fitlow,
                          fithigh=fithigh)
        else:
            print('Unknown method ', method)
            return

        # Construct iminuit object
        minuit = iminuit.Minuit(loglike, **kwargs)

        self.minuit = minuit
        self.loglike = loglike
        self.mlike = None
        self.params = None
        self.errors = None
        self.out = None
        self.npar = model.npar
        self.fixed = np.zeros(self.npar, dtype=bool)
        self.method = method
        self.samples = None

    def Migrad(self, fixed=None):
        """
        Perform maximum-likelihood optimization of the model using the MIGRAD routine from the MINUIT library.

        :param fixed: A boolean array setting up whether parameters are fixed (True) or left free (False) while fitting. If None, all parameters are fitted. Defaults to None.
        :type fixed: class:`numpy.ndarray`
        """
        minuit = self.minuit

        if fixed is not None:
            self.fixed = fixed

        for i in range(self.mod.npar):

            if self.fixed[i]:
                minuit.fixed[i] = True

        out = minuit.migrad()
        print(out)
        reg = self.loglike.region
        freepars = self.mod.npar - len(np.where(minuit.fixed)[0])
        dof = len(self.profile.profile[reg]) - freepars
        self.mlike = out.fval

        if self.method == 'chi2':
            print('Best fit chi-squared: %g for %d bins and %d d.o.f' % (out.fval, self.profile.nbin, dof))
            print('Reduced chi-squared: %g' % (out.fval / dof))
        else:
            print('Best fit C-statistic: %g for %d bins and %d d.o.f' % (out.fval, self.profile.nbin, dof))
            print('Reduced C-statistic: %g' % (out.fval / dof))

        npar = len(minuit.values)
        outval = np.empty(npar)
        outerr = np.empty(npar)
        for i in range(npar):
            outval[i] = minuit.values[i]
            outerr[i] = minuit.errors[i]
        self.mod.SetParameters(outval)
        self.mod.SetErrors(outerr)
        self.mod.parnames = minuit.parameters
        self.params = minuit.values
        self.errors = minuit.errors
        self.minuit = minuit
        self.out = out

    def Emcee(self, nmcmc=5000, burnin=100, start=None, prior=None, walkers=32, thin=15):
        '''
        Run a Markov Chain Monte Carlo optimization using the affine-invariant ensemble sampler emcee. See https://emcee.readthedocs.io/en/stable/ for details.

        :param nmcmc: Number of MCMC samples. Defaults to 5000
        :type nmcmc: int
        :param burnin: Size of the burn-in phase that will eventually be ignored. Defaults to 100
        :type burnin: int
        :param start: Array of input parameter values. If None, the code will look for the results of a previous Migrad optimization and use the corresponding parameters as starting values. Defaults to None
        :type start: class:`numpy.ndarray`
        :param prior: Function defining the priors on the parameters. The function should take the parameter set as input and return the log prior probability. If None, the code will search for the results of a previous Migrad optimization and set up broad Gaussian priors on each parameter with sigma set to 5 times the Migrad errors. Defaults to None.
        :type prior: function
        :param walkers: Number of emcee walkers. Defaults to 32.
        :type walkers: int
        :param thin: Thinning number for the output samples. The total number of sample values will be nmcmc*walkers/thin. Defaults to 15.
        :type thin: int
        '''
        try:
            import emcee

        except ImportError as e:
            print('Error: package emcee not installed, please install it to run emcee')
            return

        import emcee

        npar = len(self.minuit.values)

        if start is None and self.params is None:

            print('Please provide initial values or run a maximum likelihood fit first')
            return

        elif start is None and self.params is not None:

            start = np.empty(npar)
            for i in range(npar):
                start[i] = self.params[i]

        is_fixed = np.where(self.fixed)

        if prior is None:

            if self.errors is None:

                print('No prior provided and no available errors, please provide a custom prior or run a maximum likelihood fit first')
                return

            def prior(pars):

                errors = np.empty(npar)

                for i in range(npar):

                    if not self.fixed[i]:
                        errors[i] = self.errors[i]

                    else:
                        errors[i] = 1.

                # Gaussian prior with width +/- 5 sigma

                tot_prior = - 0.5 * np.sum( (pars - start)**2 / (5.*errors)**2)

                return tot_prior

        def log_like(pars):

            log_prior = prior(pars)

            pars[is_fixed] = start[is_fixed]

            loglike = -0.5 * self.loglike(*pars)

            return loglike + log_prior

        pos = start + 1e-4 * np.random.randn(walkers, npar)

        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_like
        )
        sampler.run_mcmc(pos, nmcmc, progress=True)

        samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

        fig, axes = plt.subplots(npar, figsize=(10, 7), sharex=True)
        samp_plot = sampler.get_chain()
        labels = self.mod.parnames
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samp_plot[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samp_plot))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        self.samples = samples

    def Corner(self, labels=None, **kwargs):
        '''
        Produce a parameter corner plot from a loaded set of samples. Uses the corner library: https://corner.readthedocs.io/en/latest/

        :param labels: List of names to be used
        :type labels: list
        :param kwargs: Any additional parameter to be passed to the corner library. See https://corner.readthedocs.io/en/latest/api.html
        :return: Output matplotlib figure
        '''

        try:
            import corner

        except ImportError as e:
            print('Error: package corner not installed, please install it to extract corner plot')
            return

        import corner

        if labels is None:

            labels = self.mod.parnames

        fig = corner.corner(
            self.samples, labels=labels, **kwargs
        )

        return fig












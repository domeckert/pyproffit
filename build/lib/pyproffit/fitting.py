import numpy as np
import iminuit

# Generic class to fit data with chi-square
class ChiSquared:
    """
    Class defining a chi-square likelihood based on a surface brightness profile and a model. This class is called by the Fitter object when using the method='chi2' option.

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
            self.psfmat = psfmat
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
            rminus = self.x - self.dx

            rplus = self.x + self.dx

            area = np.pi * (rplus ** 2 - rminus ** 2)

            ym = np.dot(self.psfmat,ym * area) / area

        reg = self.region
        nonz = self.nonz
        chi2 = np.sum((self.y[reg][nonz] - ym[reg][nonz])**2/self.dy[reg][nonz]**2)
        return chi2

# Generic class to fit data with C-stat
class Cstat:
    """
    Class defining a C-stat likelihood based on a surface brightness profile and a model. This class is called by the Fitter object when using the method='cstat' option.

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
    def __init__(self, model, x, counts, area, effexp, bkgc, psfmat=None, fitlow=None, fithigh=None):
        """
        Constructor of class Cstat

        """
        self.model = model  # model predicts y for given x
        self.x = x
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
        modcounts = ym*self.area*self.effexp
        if self.psfmat is not None:
            modcounts = np.dot(self.psfmat,modcounts)
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
    """
    def __init__(self, model, profile):
        """
        Constructor of class Fitter
        """
        self.mod=model
        self.profile=profile
        self.mlike=None
        self.params=None
        self.errors=None
        self.minuit=None
        self.out=None

    def Migrad(self, method='chi2', fitlow=None, fithigh=None, **kwargs):
        """
        Perform maximum-likelihood optimization of the model using the MIGRAD routine from the MINUIT library.

        :param method: Likelihood function to be optimized. Available likelihoods are 'chi2' (chi-squared) and 'cstat' (C statistic). Defaults to 'chi2'.
        :type method: str
        :param fitlow: Lower boundary of the active fitting radial range. If fitlow=None the entire range is used. Defaults to None
        :type fitlow: float
        :param fithigh: Upper boundary of the active fitting radial range. If fithigh=None the entire range is used. Defaults to None
        :type fithigh: float
        :param kwargs: List of arguments to be passed to the iminuit library. For instance, setting parameter boundaries, optimization options or fixing parameters.
            See the iminuit documentation: https://iminuit.readthedocs.io/en/stable/index.html
        """
        prof=self.profile
        if prof.profile is None:
            print('Error: No valid profile exists in provided object')
            return
        model=self.mod.model
        if prof.psfmat is not None:
            psfmat = np.transpose(prof.psfmat)
        else:
            psfmat = None
        if method=='chi2':
             # Define the fitting algorithm
            chi2=ChiSquared(model,prof.bins,prof.ebins,prof.profile,prof.eprof,psfmat=psfmat,fitlow=fitlow,fithigh=fithigh)
            # Construct iminuit object
            minuit=iminuit.Minuit(chi2,**kwargs)
        elif method=='cstat':
            if prof.counts is None:
                print('Error: No count profile exists')
                return
            # Define the fitting algorithm
            cstat=Cstat(model,prof.bins,prof.counts,prof.area,prof.effexp,prof.bkgcounts,psfmat=psfmat,fitlow=fitlow,fithigh=fithigh)
            # Construct iminuit object
            minuit=iminuit.Minuit(cstat,**kwargs)
        else:
            print('Unknown method ',method)
            return
        fmin, param=minuit.migrad()
        print(fmin)
        print(param)
        npar = len(minuit.values)
        outval = np.empty(npar)
        outerr = np.empty(npar)
        for i in range(npar):
            outval[i] = minuit.values[i]
            outerr[i] = minuit.errors[i]
        self.mod.SetParameters(outval)
        self.mod.SetErrors(outerr)
        self.mod.parnames = minuit.parameters
        self.params=minuit.values
        self.errors=minuit.errors
        self.mlike=fmin
        self.minuit=minuit
        self.out=param


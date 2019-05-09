import numpy as np
import iminuit

# Generic class to fit data with chi-square
class ChiSquared:
    def __init__(self, model, x , y, dy, psfmat=None, fitlow=None, fithigh=None):
        self.model = model  # model predicts y for given x
        self.x = x
        self.y = y
        self.dy = dy
        fitl = 0.
        fith = 1e10
        if fitlow is not None:
            fitl = fitlow
        if fithigh is not None:
            fith = fithigh
        self.region = np.where(np.logical_and(x>=fitl,x<=fith))
        if psfmat is not None:
            self.psfmat = psfmat
        else:
            self.psfmat = None
        self.func_code = iminuit.util.make_func_code(iminuit.util.describe(self.model)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        ym = self.model(self.x, *par)
        if self.psfmat is not None:
            ym = np.dot(self.psfmat,ym)
        reg = self.region
        chi2 = np.sum((self.y[reg] - ym[reg])**2/self.dy[reg]**2)
        return chi2

# Generic class to fit data with C-stat
class Cstat:
    def __init__(self, model, x, counts, area, effexp, bkgc, psfmat=None, fitlow=None, fithigh=None):
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
        if psfmat is not None:
            self.psfmat = psfmat
        else:
            self.psfmat = None
        self.func_code = iminuit.util.make_func_code(iminuit.util.describe(self.model)[1:])

    def __call__(self, *par):
        ym = self.model(self.x, *par)
        modcounts = ym*self.area*self.effexp
        if self.psfmat is not None:
            modcounts = np.dot(self.psfmat,modcounts)
        mm = modcounts + self.bkgc # model counts
        reg = self.region
        nc = self.c
        cstat = 2.*np.sum(mm[reg]-nc[reg]*np.log(mm[reg])-nc[reg]+nc[reg]*np.log(nc[reg])) # normalized C-statistic
        return cstat

# Class including fitting tool
class Fitter:
    def __init__(self, model, profile):
        self.mod=model
        self.profile=profile
        self.mlike=None
        self.params=None
        self.errors=None
        self.minuit=None
        self.out=None

    def Migrad(self, method='chi2', fitlow=None, fithigh=None, **kwargs):
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
            chi2=ChiSquared(model,prof.bins,prof.profile,prof.eprof,psfmat=psfmat,fitlow=fitlow,fithigh=fithigh)
            # Construct iminuit object
            minuit=iminuit.Minuit(chi2,**kwargs)
        elif method=='cstat':
            # Define the fitting algorithm
            cstat=Cstat(model,prof.bins,prof.counts,prof.area,prof.effexp,prof.bkgcounts,psfmat=psfmat,fitlow=fitlow,fithigh=fithigh)
            # Construct iminuit object
            minuit=iminuit.Minuit(cstat,**kwargs)
        else:
            print('Unknown method ',method)
            return
        fmin, param=minuit.migrad()
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


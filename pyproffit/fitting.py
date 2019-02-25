import numpy as np
import iminuit

# Generic class to fit data with chi-square
class ChiSquared:
    def __init__(self, model, x , y, dy):
        self.model = model  # model predicts y for given x
        self.x = x
        self.y = y
        self.dy = dy
        self.func_code = iminuit.util.make_func_code(iminuit.util.describe(self.model)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        ym = self.model(self.x, *par)
        chi2 = np.sum((self.y - ym)**2/self.dy**2)
        return chi2

# Generic class to fit data with C-stat
class Cstat:
    def __init__(self, model, x, counts, area, effexp, bkgc):
        self.model = model  # model predicts y for given x
        self.x = x
        self.c = counts
        self.area = area
        self.effexp = effexp
        self.bkgc = bkgc
        self.func_code = iminuit.util.make_func_code(iminuit.util.describe(self.model)[1:])

    def __call__(self, *par):
        ym = self.model(self.x, *par)
        mm = ym*self.area*self.effexp + self.bkgc # model counts
        nc = self.c
        cstat = 2.*np.sum(mm-nc*np.log(mm)-nc+nc*np.log(nc)) # normalized C-statistic
        return cstat

# Class including fitting tool
class Fitter:
    def __init__(self, model, profile):
        self.model=model
        self.profile=profile
        self.mlike=None
        self.params=None
        self.errors=None

    def Migrad(self, method='chi2', fitlow=None, fithigh=None, **kwargs):
        prof=self.profile
        if prof.profile is None:
            print('Error: No valid profile exists in provided object')
            return
        if method=='chi2':
            x=prof.bins
            y=prof.profile
            dy=prof.eprof
            # Define boundaries
            if fitlow is not None:
                reg=np.where(x>=fitlow)
                x=x[reg]
                y=y[reg]
                dy=dy[reg]
            if fithigh is not None:
                reg=np.where(x<=fithigh)
                x=x[reg]
                y=y[reg]
                dy=dy[reg]
            # Define the fitting algorithm
            chi2=ChiSquared(self.model,x,y,dy)
            # Run Migrad
            minuit=iminuit.Minuit(chi2,**kwargs)
            fmin, param=minuit.migrad()
            self.params=minuit.values
            self.errors=minuit.errors
            self.mlike=fmin
            self.minuit=minuit
            self.out=param
        elif method=='cstat':
            x=prof.bins
            counts=prof.counts
            area=prof.area
            effexp=prof.effexp
            bkgc=prof.bkgcounts
            # Define boundaries
            if fitlow is not None:
                reg=np.where(x>=fitlow)
                x=x[reg]
                area=area[reg]
                counts=counts[reg]
                effexp=effexp[reg]
                bkgc=bkgc[reg]
            if fithigh is not None:
                reg=np.where(x<=fithigh)
                x=x[reg]
                area=area[reg]
                counts=counts[reg]
                effexp=effexp[reg]
                bkgc=bkgc[reg]
            # Define the fitting algorithm
            cstat=Cstat(self.model,x,counts,area,effexp,bkgc)
            # Run Migrad
            minuit=iminuit.Minuit(cstat,**kwargs)
            fmin, param=minuit.migrad()
            self.params=minuit.values
            self.errors=minuit.errors
            self.mlike=fmin
            self.minuit=minuit
            self.out=param


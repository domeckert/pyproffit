import numpy as np

class Models:
    # Class containing PROFFIT models
    def BetaModel(x,beta,rc,norm,bkg):
        n2=np.power(10.,norm)
        c2=np.power(10.,bkg)
        out = n2*np.power(1.+(x/rc)**2,-3.*beta+0.5)+c2
        return out

    def DoubleBeta(x,beta,rc1,rc2,ratio,norm,bkg):
        comp1=np.power(1.+(x/rc1)**2,-3.*beta+0.5)
        comp2=np.power(1.+(x/rc2)**2,-3.*beta+0.5)
        n2=np.power(10.,norm)
        c2=np.power(10.,bkg)
        out=n2*(comp1+ratio*comp2)+c2
        return out

    def PowerLaw(x,alpha,norm,pivot,bkg):
        n2=np.power(10.,norm)
        c2=np.power(10.,bkg)
        out=n2*np.power(x/pivot,-alpha)+c2
        return  out

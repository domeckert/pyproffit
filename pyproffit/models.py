import numpy as np

def BetaModel(x, beta, rc, norm, bkg):
    n2 = np.power(10., norm)
    c2 = np.power(10., bkg)
    out = n2 * np.power(1. + (x / rc) ** 2, -3. * beta + 0.5) + c2
    return out


def DoubleBeta(x, beta, rc1, rc2, ratio, norm, bkg):
    comp1 = np.power(1. + (x / rc1) ** 2, -3. * beta + 0.5)
    comp2 = np.power(1. + (x / rc2) ** 2, -3. * beta + 0.5)
    n2 = np.power(10., norm)
    c2 = np.power(10., bkg)
    out = n2 * (comp1 + ratio * comp2) + c2
    return out


def PowerLaw(x, alpha, norm, pivot, bkg):
    n2 = np.power(10., norm)
    c2 = np.power(10., bkg)
    out = n2 * np.power(x / pivot, -alpha) + c2
    return out

class Model:
    # Class containing PROFFIT models
    def __init__(self,model,vals=None):
        self.model=model
        if vals is not None:
            self.params=vals
        else:
            self.params=None

    def SetParameters(self,vals):
        self.params=vals

    def SetErrors(self,vals):
        self.errors=vals

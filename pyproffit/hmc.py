import pymc as pm
import numpy as np
import time
from .models import IntFunc

def BetaModelPM(x, beta, rc, norm, bkg):
    """
    """
    n2 = 10. ** norm
    c2 = 10. ** bkg
    out = n2 * (1. + (x / rc) ** 2) ** (-3. * beta + 0.5) + c2
    return out


def DoubleBetaPM(x, beta, rc1, rc2, ratio, norm, bkg):
    """
    """
    comp1 = (1. + (x / rc1) ** 2) ** (-3. * beta + 0.5)
    comp2 = (1. + (x / rc2) ** 2) ** (-3. * beta + 0.5)
    n2 = 10. ** norm
    c2 = 10. ** bkg
    out = n2 * (comp1 + ratio * comp2) + c2
    return out


def PowerLawPM(x, alpha, norm, pivot, bkg):
    """
    """
    n2 = 10. ** norm
    c2 = 10. ** bkg
    out = n2 * (x / pivot) ** (-alpha) + c2
    return out

def ConstPM(x, bkg):
    """
    """
    out = 10. ** bkg
    return out


def VikhlininPM(x,beta,rc,alpha,rs,epsilon,gamma,norm,bkg):
    """
    """
    term1 = (x/rc) ** (-alpha) * (1. + (x/rc) ** 2) ** (-3 * beta + alpha/2)
    term2 = (1. + (x / rs) ** gamma) ** (-epsilon / gamma)
    n2 = 10. ** norm
    c2 = 10. ** bkg
    return n2 * term1 * term2 + c2

def IntFuncPM(omega,rf,alpha,xmin,xmax):
    """
    """
    nb = 100
    logmin = pm.math.log(xmin) / pm.math.log(10.)
    logmax = pm.math.log(xmax) / pm.math.log(10.)
    intot = 0.
    for i in range(nb):
        basex_low = (logmin + i / nb * (logmax - logmin))
        basex_high = (logmin + (i+1) / nb * (logmax - logmin))
        z = (10 ** basex_low + 10 ** basex_high) / 2.
        width = 10 ** basex_high - 10 ** basex_low
        term1 = (omega**2 + z**2) / rf**2
        term2 = term1 ** (-alpha)
        intot = intot + term2 * width
    return intot

def BknPowPM(x, alpha1, alpha2, norm, jump, bkg, rf=3.0):
    """
    Broken power law 3D model projected along the line of sight for discontinuity modeling

    .. math::

        I(r) = I_0 \\int F(\\omega)^2 d\\ell + B

    with :math:`\\omega^2 = r^2 + \ell^2` and

    .. math::

        F(\\omega) = \left\{ \\begin{array}{ll} \omega^{-\\alpha_1}, & \omega<r_f \\\\ \\frac{1}{C}\omega ^{-\\alpha_2}, & \omega\\geq r_f
        \end{array} \\right.

    :param x: Radius in arcmin
    :type x: numpy.ndarray
    :param alpha1: :math:`\\alpha_1` parameter
    :type alpha1: class:`theano.tensor`
    :param alpha2: :math:`\\alpha_2` parameter
    :type alpha2: class:`theano.tensor`
    :param rf: rf parameter
    :type rf: float
    :param norm: log of I0 parameter
    :type norm: class:`theano.tensor`
    :param jump: C parameter
    :type jump: class:`theano.tensor`
    :param bkg: log of B parameter
    :type bkg: class:`theano.tensor`
    :return: Calculated model
    :rtype: :class:`numpy.ndarray`
    """
    A1 = 10. ** norm
    A2 = A1 / (jump ** 2)
    inreg = np.where(x < rf)
    term1 = IntFunc(x[inreg], rf, alpha1, 0.01 * np.ones(len(x[inreg])), np.sqrt(rf ** 2 - x[inreg] ** 2))
    term2 = IntFunc(x[inreg], rf, alpha2, np.sqrt(rf ** 2 - x[inreg] ** 2), 1e3 * np.ones(len(x[inreg])))
    inside = A1 * term1 + A2 * term2
    outreg = np.where(x >= rf)
    term = IntFunc(x[outreg], rf, alpha2, 0.01 * np.ones(len(x[outreg])), 1e3 * np.ones(len(x[outreg])))
    outside = A2 * term
    out = pm.math.stack([inside, outside])
    c2 = 10. ** bkg
    return out + c2


def fit_profile_pymc3(hmcmod, prof, nmcmc=1000, tune=500, find_map=True, fitlow=0., fithigh=1e10):

    if hmcmod.start is None or hmcmod.sd is None:

        print('Missing prior definition, cannot continue; please provide both "start" and "sd" parameters')

        return

    npar = hmcmod.npar

    reg = np.where(np.logical_and(prof.bins>=fitlow, prof.bins<=fithigh))

    sb = prof.profile[reg]
    esb = prof.eprof[reg]
    rad = prof.bins[reg]
    erad = prof.ebins[reg]
    counts = prof.counts[reg]
    area = prof.area[reg]
    exposure = prof.effexp[reg]
    bkgcounts = prof.bkgcounts[reg]

    if prof.psfmat is not None:
        psfmat = np.transpose(prof.psfmat)
    else:
        psfmat = np.eye(prof.nbin)

    hmc_model = pm.Model()

    with hmc_model:

        # Set up model parameters
        allpmod = []

        numrf = None

        for i in range(npar):

            name = hmcmod.parnames[i]

            if not hmcmod.fix[i]:

                print('Setting Gaussian prior for parameter %s with center %g and standard deviation %g' % (name, hmcmod.start[i], hmcmod.sd[i]))

                if hmcmod.limits is not None:

                    lim = hmcmod.limits[i]

                    modpar = pm.TruncatedNormal(name, mu=hmcmod.start[i], sd=hmcmod.sd[i], lower=lim[0], upper=lim[1])

                else:

                    modpar = pm.Normal(name, mu=hmcmod.start[i], sd=hmcmod.sd[i])

            else:

                print('Parameter %s is fixed' % (name))

                modpar = pm.ConstantDist(name, hmcmod.start[i])

            if name == 'rf':

                numrf = modpar.random()

            else:

                allpmod.append(modpar)

        pmod = pm.math.stack(allpmod, axis=0)

        if numrf is not None:

            modcounts = hmcmod.model(rad, *pmod, rf=numrf) * area * exposure

        else:

            modcounts = hmcmod.model(rad, *pmod) * area * exposure

        pred = pm.math.dot(psfmat, modcounts) + bkgcounts

        count_obs = pm.Poisson('counts', mu=pred, observed=counts)  # counts likelihood

    tinit = time.time()

    print('Running MCMC...')

    with hmc_model:

        if find_map:

            start = pm.find_MAP()

            trace = pm.sample(nmcmc, start=start, tune=tune)

        else:

            trace = pm.sample(nmcmc, tune=tune)

    print('Done.')

    tend = time.time()

    print(' Total computing time is: ', (tend - tinit) / 60., ' minutes')

    hmcmod.trace = trace




class HMCModel(object):
    """
    Class containing pyproffit model structure for HMC optimization

    :param model: Function to be used as surface brightness model
    :type model: function
    :param vals: Array containing initial values for the parameters (optional)
    :type vals: :class:`numpy.ndarray`
    """
    def __init__(self,model, start=None, sd=None, limits=None, fix=None):
        """
        Constructor of class HMCModel
        """
        self.model = model

        npar = model.__code__.co_argcount - 1

        self.npar = npar

        self.parnames = model.__code__.co_varnames[1:npar + 1]

        if start is not None:

            if len(start) != self.npar:

                print(
                    'Wrong number of parameters in input parameter vector, the provided function requires %d but the vector contains %d. Ignoring.' % (
                    self.npar, len(start)))

                self.start = None

            else:

                self.start = start
        else:

            self.start = None

        if sd is not None:

            if len(sd) != self.npar:

                print(
                    'Wrong number of parameters in prior standard deviation vector, the provided function requires %d but the vector contains %d. Ignoring.' % (
                    self.npar, len(sd)))

                self.sd = None

            else:

                self.sd = sd
        else:

            self.sd = None

        if limits is not None:

            try:
                assert (limits.shape == (self.npar, 2))

            except AssertionError:

                print('Wrong number of parameters in parameter limits, the vector should have shape (%d , 2). Ignoring.' % (
                    self.npar))

                return

            self.limits = limits

        else:

            self.limits = None

        if fix is not None:

            if len(fix) != self.npar:

                print(
                    'Wrong number of parameters in input vector, the provided function requires %d but the vector contains %d. Ignoring.' % (
                    self.npar, len(fix)))

                self.fix = np.zeros(self.npar, dtype=bool)

            else:

                self.fix = fix
        else:

            self.fix = np.zeros(self.npar, dtype=bool)



    def SetPriors(self, start=None, sd=None, limits=None, fix=None):
        """
        Set prior definition for the function parameters

        :param start:
        :param sd:
        :param limits:
        :param fix:
        """
        if start is not None:

            if len(start) != self.npar:

                print(
                    'Wrong number of parameters in input parameter vector, the provided function requires %d but the vector contains %d. Ignoring.' % (
                    self.npar, len(start)))

                self.start = None

            else:

                self.start = start
        else:

            self.start = None

        if sd is not None:

            if len(sd) != self.npar:

                print(
                    'Wrong number of parameters in prior standard deviation vector, the provided function requires %d but the vector contains %d. Ignoring.' % (
                    self.npar, len(sd)))

                self.sd = None

            else:

                self.sd = sd
        else:

            self.sd = None

        if limits is not None:

            try:
                assert (limits.shape == (self.npar, 2))

            except AssertionError:

                print('Wrong number of parameters in parameter limits, the vector should have shape (%d , 2). Ignoring.' % (
                    self.npar))

                return

            self.limits = limits

        else:

            self.limits = None

        if fix is not None:

            if len(fix) != self.npar:

                print(
                    'Wrong number of parameters in input vector, the provided function requires %d but the vector contains %d. Ignoring.' % (
                    self.npar, len(fix)))

                self.fix = np.zeros(self.npar, dtype=bool)

            else:

                self.fix = fix
        else:

            self.fix = np.zeros(self.npar, dtype=bool)


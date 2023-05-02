import numpy as np
import pymc as pm

def BetaModel(x, beta, rc, norm, bkg):
    """
    Single Beta model

    .. math::

        I(r) = I_0 \\left(1 + (x/r_c)^2\\right) ^ {-3 \\beta + 0.5} + B

    :param x: Radius in arcmin
    :type x: numpy.ndarray
    :param beta: :math:`\\beta` parameter
    :type beta: float
    :param rc: rc parameter
    :type rc: float
    :param norm: log of I0 parameter
    :type norm: float
    :param bkg: log of B parameter
    :type bkg: float
    :return: Calculated model
    :rtype: :class:`numpy.ndarray`
    """
    n2 = np.power(10., norm)
    c2 = np.power(10., bkg)
    out = n2 * np.power(1. + (x / rc) ** 2, -3. * beta + 0.5) + c2
    return out


def DoubleBeta(x, beta, rc1, rc2, ratio, norm, bkg):
    """
    Double beta model

    .. math::

        I(r) = I_0 \\left[ (1 + (x/r_{c,1}) ^ 2)^{-3 \\beta + 0.5} + R ( 1 + (x/r_{c,2}) ^ 2 )^{-3 \\beta+0.5} \\right] + B

    :param x: Radius in arcmin
    :type x: numpy.ndarray
    :param beta: :math:`\\beta` parameter
    :type beta: float
    :param rc1: rc1 parameter
    :type rc1: float
    :param rc2: rc2 parameters
    :type rc2: float
    :param ratio: R parameter
    :type ratio: float
    :param norm: log of I0 parameter
    :type norm: float
    :param bkg: log of B parameter
    :type bkg: float
    :return: Calculated model
    :rtype: :class:`numpy.ndarray`
    """
    comp1 = np.power(1. + (x / rc1) ** 2, -3. * beta + 0.5)
    comp2 = np.power(1. + (x / rc2) ** 2, -3. * beta + 0.5)
    n2 = np.power(10., norm)
    c2 = np.power(10., bkg)
    out = n2 * (comp1 + ratio * comp2) + c2
    return out


def PowerLaw(x, alpha, norm, pivot, bkg):
    """
    Single power law model in projected space

    .. math::

        I(r)=I_0\\left( \\frac{x}{x_p} \\right)^{-\\alpha} + B

    :param x: Radius in arcmin
    :type x: numpy.ndarray
    :param alpha: :math:`\\alpha` parameter
    :type alpha: float
    :param norm: log of I0 parameter
    :type norm: float
    :param pivot: :math:`x_p` parameter
    :type pivot: float
    :param bkg: log of B parameter
    :type bkg: float
    :return: Calculated model
    :rtype: :class:`numpy.ndarray`
    """
    n2 = np.power(10., norm)
    c2 = np.power(10., bkg)
    out = n2 * np.power(x / pivot, -alpha) + c2
    return out

def Const(x, bkg):
    """
    Constal model for background fitting

    :param x: Radius in arcmin
    :type x: numpy.ndarray
    :param bkg: log of B parameter
    :type bkg: float
    :return: Calculated model
    :rtype: :class:`numpy.ndarray`
    """
    out = np.power(10., bkg) * np.ones(len(x))
    return out


def Vikhlinin(x,beta,rc,alpha,rs,epsilon,gamma,norm,bkg):
    """
    Simplified Vikhlinin+06 surface brightness model used in Ghirardini+19

    .. math::

        I(r) = I_0\\left( \\frac{x}{r_c}\\right)^{-\\alpha} \\left(1+(x/r_c)^2 \\right)^{-3\\beta + \\alpha/2} \\left( 1 + (x/r_s)^{\\gamma} \\right)^{-\\epsilon/\\gamma}

    :param x: Radius in arcmin
    :type x: numpy.ndarray
    :param beta: :math:`\\beta` parameter
    :type beta: float
    :param rc: rc parameter
    :type rc: float
    :param norm: log of I0 parameter
    :type norm: float
    :param alpha: :math:`\\alpha` parameter
    :type alpha: float
    :param rs: rs parameter
    :type rs: float
    :param epsilon: :math:`\\epsilon` parameter
    :type epsilon: float
    :param gamma: :math:`\\gamma` parameter
    :type gamma: float
    :param bkg: log of B parameter
    :type bkg: float
    :return: Calculated model
    :rtype: :class:`numpy.ndarray`
    """
    term1 = np.power(x/rc, -alpha)*np.power(1. + (x/rc) ** 2, -3 * beta + alpha/2)
    term2 = np.power(1. + (x / rs) ** gamma, -epsilon / gamma)
    n2 = np.power(10., norm)
    b2 = np.power(10., bkg)
    return n2 * term1 * term2 + b2

def IntFunc(omega,rf,alpha,xmin,xmax):
    """
    Numerical integration of a power law along the line of sight

    .. math::

        \\int_{x_{min}}^{x_{max}} \\left(\\frac{\\omega^2 + \\ell^2}{r_f^2}\\right)^{-\\alpha} d\\ell

    :param omega: Projected radius
    :type omega: float
    :param rf: rf parameter
    :type rf: float
    :param alpha: :math:`\\alpha` parameter
    :type alpha: float
    :param xmin: xmin parameter
    :type xmin: float
    :param xmax: xmax parameter
    :type xmax: float
    :return: Line-of-sight integral
    :rtype: float
    """
    nb = 100
    logmin = np.log10(xmin)
    logmax = np.log10(xmax)
    x = np.logspace(logmin,logmax,nb+1)
    z = (x[:nb] + np.roll(x, -1, axis=0)[:nb]) / 2.
    width = (np.roll(x, -1, axis=0)[:nb] - x[:nb])
    term1 = (omega**2 + z**2) / rf**2
    term2 = np.power(term1,-alpha)
    intot = np.sum(term2*width,axis=0)
    return intot

def BknPow(x,alpha1,alpha2,rf,norm,jump,bkg):
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
    :type alpha1: float
    :param alpha2: :math:`\\alpha_2` parameter
    :type alpha2: float
    :param rf: rf parameter
    :type rf: float
    :param norm: log of I0 parameter
    :type norm: float
    :param jump: C parameter
    :type jump: float
    :param bkg: log of B parameter
    :type bkg: float
    :return: Calculated model
    :rtype: :class:`numpy.ndarray`
    """
    A1 = np.power(10.,norm)
    A2 = A1 / jump**2
    out = np.empty(len(x))
    inreg = np.where(x < rf)
    term1 = IntFunc(x[inreg],rf,alpha1,0.01*np.ones(len(x[inreg])),np.sqrt(rf**2-x[inreg]**2))
    term2 = IntFunc(x[inreg],rf,alpha2,np.sqrt(rf**2-x[inreg]**2),1e3*np.ones(len(x[inreg])))
    out[inreg] = A1 * term1 + A2 * term2
    outreg = np.where(x >= rf)
    term = IntFunc(x[outreg],rf,alpha2,0.01*np.ones(len(x[outreg])),1e3*np.ones(len(x[outreg])))
    out[outreg] = A2 * term
    c2 = np.power(10., bkg)
    return out + c2


class Model(object):
    """
    Class containing pyproffit models

    :param model: Function to be used as surface brightness model
    :type model: function
    :param vals: Array containing initial values for the parameters (optional)
    :type vals: :class:`numpy.ndarray`
    """
    def __init__(self,model,vals=None):
        """
        Constructor of class Model
        """
        self.model=model

        npar = model.__code__.co_argcount

        self.npar = npar - 1

        self.parnames = model.__code__.co_varnames[1:npar]

        if vals is not None:

            if len(vals) != self.npar:

                print('Wrong number of parameters in input parameter vector, the provided function requires %d but the vector contains %d. Ignoring.' % (npar, len(vals)))

                self.params = None

            else:

                self.params = vals
        else:

            self.params=None

    def __call__(self, x, *pars):

        return self.model(x, *pars)


    def SetParameters(self,vals):
        """
        Set input values for the model parameters

        :param vals: Array containing initial values for the parameters
        :type vals: :class:`numpy.ndarray`
        """
        if len(vals) != self.npar:
            print(
                'Wrong number of parameters in input parameter vector, the provided function requires %d but the vector contains %d. Ignoring.' % (
                self.npar, len(vals)))

            self.params = None

        else:

            self.params = vals

    def SetErrors(self,vals):
        """
        Set input values for the errors on the parameters

        :param vals: Array containing initial values for the errors
        :type vals: :class:`numpy.ndarray`
        """
        if len(vals) != self.npar:
            print(
                'Wrong number of parameters in input error vector, the provided function requires %d but the vector contains %d. Ignoring.' % (
                self.npar, len(vals)))

            self.errors = None

        else:

            self.errors = vals

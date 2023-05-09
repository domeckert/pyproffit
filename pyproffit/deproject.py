import numpy as np
import pymc as pm
import time
from scipy.special import gamma
import matplotlib.pyplot as plt
#plt.switch_backend('Agg')
from scipy.interpolate import interp1d
import os
from astropy.io import fits
from astropy import units as u

Mpc = 3.0856776e+24 #cm
kpc = 3.0856776e+21 #cm
msun = 1.9891e33 #g
mh = 1.66053904e-24 #proton mass in g


def plot_multi_methods(profs, deps, labels=None, outfile=None, xunit='kpc', figsize=(13, 10), fontsize=40, xscale='log', yscale='log', fmt='.', markersize=7):
    """
    Plot multiple gas density profiles (e.g. obtained through several methods, centers or sectors) to compare them

    :param profs: List of Profile objects to be plotted
    :type profs: tuple
    :param deps: List of Deproject objects to be plotted
    :type deps: tuple
    :param labels: List of labels for the legend (default=None)
    :type labels: tuple
    :param outfile: If outfile is not None, path to file name to output the plot
    :type outfile: str
    :param figsize: Size of figure. Defaults to (13, 10)
    :type figsize: tuple , optional
    :param fontsize: Font size of the axis labels. Defaults to 40
    :type fontsize: int , optional
    :param xscale: Scale of the X axis. Defaults to 'log'
    :type xscale: str , optional
    :param yscale: Scale of the Y axis. Defaults to 'log'
    :type yscale: str , optional
    :param fmt: Marker type following matplotlib convention. Defaults to 'd'
    :type fmt: str , optional
    :param markersize: Marker size. Defaults to 7
    :type markersize: int , optional
    """
    if len(profs) != len(deps):
        print("ERROR: different numbers of profiles and deprojection elements")
        return

    if xunit != 'kpc' and xunit != 'arcmin':
        print('Unknown X unit %s , reverting to kpc' % xunit)

        xunit = 'kpc'

    print("Showing %d density profiles" % len(deps))
    if labels is None:
        labels = [None] * len(deps)

    fig = plt.figure(figsize=figsize)
    ax_size = [0.14, 0.14,
               0.83, 0.83]
    ax = fig.add_axes(ax_size)
    ax.minorticks_on()
    ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
    ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)

    if xunit == 'kpc':
        plt.xlabel('Radius [kpc]', fontsize=fontsize)
    else:
        plt.xlabel('Radius [arcmin]', fontsize=fontsize)

    plt.ylabel('$n_{H}$ [cm$^{-3}$]', fontsize=fontsize)
    plt.xscale(xscale)
    plt.yscale(yscale)
    for i in range(len(deps)):
        dep = deps[i]
        prof = profs[i]


        kpcp = prof.cosmo.kpc_proper_per_arcmin(dep.z).value

        sourcereg = np.where(prof.bins < dep.bkglim)

        if xunit == 'kpc':
            rkpc = prof.bins[sourcereg] * kpcp
            erkpc = prof.ebins[sourcereg] * kpcp
        else:
            rkpc = prof.bins[sourcereg]
            erkpc = prof.bins[sourcereg]

        plt.errorbar(rkpc, dep.dens, xerr=erkpc, yerr=[dep.dens - dep.dens_lo, dep.dens_hi - dep.dens], fmt=fmt,
                     color='C%d' % i, elinewidth=2,
                     markersize=markersize, capsize=3, label=labels[i])
        plt.fill_between(rkpc, dep.dens_lo, dep.dens_hi, color='C%d' % i, alpha=0.3)
    plt.legend(loc=0,fontsize=22)
    if outfile is not None:
        plt.savefig(outfile)


def fbul19(R,z,cosmo,Runit='kpc'):
    """
    Compute Mgas from input R500 using Bulbul+19 M-Mgas scaling relation

    :param R: Input R500 value
    :type R: float
    :param z: Input redshift
    :type z: float
    :param Runit: Unit of input radis, kpc or arcmin (default='kpc')
    :type Runit: str
    :return: Mgas
    :rtype: float
    """
    if Runit == 'arcmin':
        amin2kpc = cosmo.kpc_proper_per_arcmin(z).value
        R=R*amin2kpc

    rho_cz = cosmo.critical_density(z).to(u.Msun / u.kpc ** 3).value
    efunc = np.asarray(cosmo.efunc(z))
    M = 4. / 3. * np.pi * 500 * rho_cz * R ** 3

    zpiv = 0.45
    Mpiv = 6.35e14
    efuncpiv = np.asarray(cosmo.efunc(zpiv))

    A = 7.09
    B = 1.26
    C = 0
    sigma = 0.10
    gamma = 0.16
    delta = 0.16
    Bprime = B + delta * np.log((1 + z) / (1 + zpiv))

    Mgas = 1e13 * A * (M / Mpiv) ** Bprime * (efunc / efuncpiv) ** (2 / 3) * ((1 + z) / (1 + zpiv)) ** gamma

    return Mgas


def calc_linear_operator(rad,sourcereg,pars,area,expo,psf):
    """
    Function to calculate a linear operator transforming parameter vector into predicted model counts

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param pars: List of beta model parameters obtained through list_params
    :type pars: numpy.ndarray
    :param area: Bin area in arcmin^2
    :type area: numpy.ndarray
    :param expo: Bin effective exposure in s
    :type expo: numpy.ndarray
    :param psf: PSF mixing matrix
    :type psf: numpy.ndarray
    :return: Linear projection and PSF mixing operator
    :rtype: numpy.ndarray
    """
    # Select values in the source region
    rfit=rad[sourcereg]
    npt=len(rfit)
    npars=len(pars[:,0])
    areamul=np.tile(area[0:npt],npars).reshape(npars,npt)
    expomul=np.tile(expo[0:npt],npars).reshape(npars,npt)
    spsf=psf[0:npt,0:npt]
    
    # Compute linear combination of basis functions in the source region
    beta=np.repeat(pars[:,0],npt).reshape(npars,npt)
    rc=np.repeat(pars[:,1],npt).reshape(npars,npt)
    base=1.+np.power(rfit/rc,2)
    expon=-3.*beta+0.5
    func_base=np.power(base,expon)
    
    # Predict number of counts per annulus and convolve with PSF
    Ktrue=func_base*areamul*expomul
    Kconv=np.dot(spsf,Ktrue.T)
    
    # Recast into full matrix and add column for background
    nptot=len(rad)
    Ktot=np.zeros((nptot,npars+1))
    Ktot[0:npt,0:npars]=Kconv
    Ktot[:,npars]=area*expo
    return Ktot

# Function to create the list of parameters for the basis functions
nsh=4. # number of basis functions to set

def list_params(rad,sourcereg,nrc=None,nbetas=6,min_beta=0.6):
    """
    Define a list of parameters to define the dictionary of basis functions

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param nrc: Number of core radii. If nrc=None (default), the number of core radiis will be defined on-the-fly
    :type nrc: int
    :param nbetas: Number of beta values. Default=6
    :type nbetas: int
    :param min_beta: Minimum value of beta. Default=0.6
    :type min_beta: float
    :return: Array containing sets of values to set up the function dictionary
    :rtype: numpy.ndarray
    """
    rfit=rad[sourcereg]
    npfit=len(rfit)
    if nrc is None:
        nrc = np.max([int(npfit/nsh),1])
    allrc=np.logspace(np.log10(rfit[2]),np.log10(rfit[npfit-1]/2.),nrc)
    #allbetas=np.linspace(0.4,3.,6)
    allbetas = np.linspace(min_beta, 3., nbetas)
    nrc=len(allrc)
    nbetas=len(allbetas)
    rc=allrc.repeat(nbetas)
    betas=np.tile(allbetas,nrc)
    ptot=np.empty((nrc*nbetas,2))
    ptot[:,0]=betas
    ptot[:,1]=rc
    return ptot

# Function to create a linear operator transforming parameters into surface brightness

def calc_sb_operator(rad,sourcereg,pars):
    """
    Function to calculate a linear operator transforming parameter vector into surface brightness

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param pars: List of beta model parameters obtained through list_params
    :type pars: numpy.ndarray
    :return: Linear projection operator
    :rtype: numpy.ndarray
    """

    # Select values in the source region
    rfit=rad[sourcereg]
    npt=len(rfit)
    npars=len(pars[:,0])
    
    # Compute linear combination of basis functions in the source region
    beta=np.repeat(pars[:,0],npt).reshape(npars,npt)
    rc=np.repeat(pars[:,1],npt).reshape(npars,npt)
    base=1.+np.power(rfit/rc,2)
    expon=-3.*beta+0.5
    func_base=np.power(base,expon)
    
    # Recast into full matrix and add column for background
    nptot=len(rad)
    Ktot=np.zeros((nptot,npars+1))
    Ktot[0:npt,0:npars]=func_base.T
    Ktot[:,npars]=0.0
    return Ktot


def calc_sb_operator_psf(rad, sourcereg, pars, area, expo, psf):
    """
    Same as calc_sb_operator but convolving the model surface brightness with the PSF model

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param pars: List of beta model parameters obtained through list_params
    :type pars: numpy.ndarray
    :param area: Bin area in arcmin^2
    :type area: numpy.ndarray
    :param expo: Bin effective exposure in s
    :type expo: numpy.ndarray
    :param psf: PSF mixing matrix
    :type psf: numpy.ndarray
    :return: Linear projection and PSF mixing operator
    :rtype: numpy.ndarray
    """
    # Select values in the source region
    rfit = rad[sourcereg]
    npt = len(rfit)
    npars = len(pars[:, 0])

    areamul = np.tile(area[0:npt], npars).reshape(npars, npt)
    expomul = np.tile(expo[0:npt], npars).reshape(npars, npt)
    spsf = psf[0:npt, 0:npt]

    # Compute linear combination of basis functions in the source region
    beta = np.repeat(pars[:, 0], npt).reshape(npars, npt)
    rc = np.repeat(pars[:, 1], npt).reshape(npars, npt)
    base = 1. + np.power(rfit / rc, 2)
    expon = -3. * beta + 0.5
    func_base = np.power(base, expon)

    Ktrue = func_base * areamul * expomul
    Kconv = np.dot(spsf, Ktrue.T)

    # Recast into full matrix and add column for background
    nptot = len(rad)
    Ktot = np.zeros((nptot, npars + 1))
    Ktot[0:npt, 0:npars] = Kconv
    Ktot[:, npars] = area * expo
    return Ktot


def calc_int_operator(a, b, pars):
    """
    Compute a linear operator to integrate analytically the basis functions within some radial range and return count rate and luminosities

    :param a: Lower integration boundary
    :type a: float
    :param b: Upper integration boundary
    :type b: float
    :param pars: List of beta model parameters obtained through list_params
    :type pars: numpy.ndarray
    :return: Linear integration operator
    :rtype: numpy.ndarray
    """
    # Select values in the source region
    npars = len(pars[:, 0])
    rads = np.array([a, b])
    npt = 2

    # Compute linear combination of basis functions in the source region
    beta = np.repeat(pars[:, 0], npt).reshape(npars, npt)
    rc = np.repeat(pars[:, 1], npt).reshape(npars, npt)
    base = 1. + np.power(rads / rc, 2)
    expon = -3. * beta + 1.5
    func_base = 2. * np.pi * np.power(base, expon) / (3 - 6 * beta) * rc**2

    # Recast into full matrix and add column for background
    Kint = np.zeros((npt, npars + 1))
    Kint[0:npt, 0:npars] = func_base.T
    Kint[:, npars] = 0.0
    return Kint


def list_params_density(rad,sourcereg,z,cosmo,nrc=None,nbetas=6,min_beta=0.6):
    """
    Define a list of parameters to transform the basis functions into gas density profiles

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param z: Source redshift
    :type z: float
    :param nrc: Number of core radii. If nrc=None (default), the number of core radiis will be defined on-the-fly
    :type nrc: int
    :param nbetas: Number of beta values. Default=6
    :type nbetas: int
    :param min_beta: Minimum value of beta. Default=0.6
    :type min_beta: float
    :return: Array containing sets of values to set up the function dictionary
    :rtype: numpy.ndarray
    """
    rfit=rad[sourcereg]
    npfit=len(rfit)
    kpcp=cosmo.kpc_proper_per_arcmin(z).value
    if nrc is None:
        nrc = np.max([int(npfit/nsh),1])
    allrc=np.logspace(np.log10(rfit[2]),np.log10(rfit[npfit-1]/2.),nrc)*kpcp
    #allbetas=np.linspace(0.5,3.,6)
    allbetas = np.linspace(min_beta, 3., nbetas)
    nrc=len(allrc)
    nbetas=len(allbetas)
    rc=allrc.repeat(nbetas)
    betas=np.tile(allbetas,nrc)
    ptot=np.empty((nrc*nbetas,2))
    ptot[:,0]=betas
    ptot[:,1]=rc
    return ptot

# Linear operator to transform parameters into density

def calc_density_operator(rad,pars,z,cosmo):
    """
    Compute linear operator to transform parameters into gas density profiles

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param pars: List of beta model parameters obtained through list_params
    :type pars: numpy.ndarray
    :param z: Source redshift
    :type z: float
    :return: Linear operator for gas density
    :rtype: numpy.ndarray
    """
    # Select values in the source region
    kpcp=cosmo.kpc_proper_per_arcmin(z).value
    rfit=rad*kpcp
    npt=len(rfit)
    npars=len(pars[:,0])
    
    # Compute linear combination of basis functions in the source region
    beta=np.repeat(pars[:,0],npt).reshape(npars,npt)
    rc=np.repeat(pars[:,1],npt).reshape(npars,npt)
    base=1.+np.power(rfit/rc,2)
    expon=-3.*beta
    func_base=np.power(base,expon)
    cfact=gamma(3*beta)/gamma(3*beta-0.5)/np.sqrt(np.pi)/rc
    fng=func_base*cfact
    
    # Recast into full matrix and add column for background
    nptot=len(rfit)
    Ktot=np.zeros((nptot,npars+1))
    Ktot[0:npt,0:npars]=fng.T
    Ktot[:,npars]=0.0
    return Ktot

def Deproject_Multiscale_Stan(deproj,bkglim=None,nmcmc=1000,back=None,samplefile=None,nrc=None,nbetas=6,depth=10,min_beta=0.6):
    """
    Run the multiscale deprojection optimization using the Stan backend

    :param deproj: Object of type :class:`pyproffit.deproject.Deproject` containing the data and parameters
    :type deproj: class:`pyproffit.deproject.Deproject`
    :param bkglim: Limit beyond which it is assumed that the background dominates, i.e. the source is set to 0. If bkglim=None (default), the entire radial range is used
    :type bkglim: float
    :param nmcmc: Number of HMC points in the output sample
    :type nmcmc: int
    :param back: Input value for the background, around which a gaussian prior is set. If back=None (default), the input background value will be computed as the average of the source-free region
    :type back: float
    :param samplefile: Path to output file to write the output samples. If samplefile=None (default), the data are not written to file and only loaded into memory
    :type samplefile: str
    :param nrc: Number of core radii. If nrc=None (default), the number of core radiis will be defined on-the-fly
    :type nrc: int
    :param nbetas: Number of beta values. Default=6
    :type nbetas: int
    :param depth: Set the max_treedepth parameter for Stan (default=10)
    :type depth: int
    :param min_beta: Minimum value of beta. Default=0.6
    :type min_beta: float
    """
    prof = deproj.profile
    sb = prof.profile
    rad = prof.bins
    erad = prof.ebins
    counts = prof.counts
    area = prof.area
    exposure = prof.effexp
    bkgcounts = prof.bkgcounts

    # Define maximum radius for source deprojection, assuming we have only background for r>bkglim
    if bkglim is None:
        bkglim=np.max(rad+erad)
        deproj.bkglim = bkglim
        if back is None:
            back = sb[len(sb) - 1]
    else:
        deproj.bkglim = bkglim
        backreg = np.where(rad>bkglim)
        if back is None:
            back = np.mean(sb[backreg])

    # Set source region
    sourcereg = np.where(rad < bkglim)
    nptfit = len(sb[sourcereg])

    # Set vector with list of parameters
    pars = list_params(rad, sourcereg, nrc, nbetas, min_beta)

    npt = len(pars)

    if prof.psfmat is not None:
        psfmat = np.transpose(prof.psfmat)
    else:
        psfmat = np.eye(prof.nbin)

    # Compute linear combination kernel
    K = calc_linear_operator(rad, sourcereg, pars, area, exposure, psfmat)
    if np.isnan(sb[0]) or sb[0] <= 0:
        testval = -10.
    else:
        testval = np.log(sb[0] / npt)
    if np.isnan(back) or back == 0:
        testbkg = -10.
    else:
        testbkg = np.log(back)

    norm0=np.append(np.repeat(testval,npt),testbkg)

    import stan
    import stan_utility as su

    stan_dir = os.path.expanduser('~/.stan_cache')
    if not os.path.exists(stan_dir):
        os.makedirs(stan_dir)

    code = """
    data {
    int<lower=0> N;
    int<lower=0> M;
    int cts_tot[N];
    vector[N] cts_back;
    matrix[N,M] K;
    vector[M] norm0;
    }
    parameters {
    vector[M] log_norm;
    }
    transformed parameters{
    vector[M] norm = exp(log_norm);
    }
    model {
    log_norm ~ normal(norm0,10);
    cts_tot ~ poisson(K * norm + cts_back);
    }"""


    f = open('mybeta_GP.stan', 'w')
    print(code, file=f)
    f.close()
    sm = su.compile_model('mybeta_GP.stan', model_name='model_GP')

    datas = dict(K=K, cts_tot=counts.astype(int), cts_back=bkgcounts, N=K.shape[0], M=K.shape[1],
                 norm0=norm0)
    tinit = time.time()
    print('Running MCMC...')
    fit = sm.sampling(data=datas, chains=1, iter=nmcmc, thin=1, n_jobs=1, control={'max_treedepth': depth})
    print('Done.')
    tend = time.time()
    print(' Total computing time is: ', (tend - tinit) / 60., ' minutes')
    chain = fit.extract()
    samples = chain['log_norm']

    # Get chains and save them to file

    if samplefile is not  None:
        np.savetxt(samplefile, samples)
        np.savetxt(samplefile+'.par',np.array([pars.shape[0]/nbetas,nbetas,min_beta,nmcmc]),header='stan')

    # Compute output deconvolved brightness profile
    Ksb = calc_sb_operator(rad, sourcereg, pars)
    allsb = np.dot(Ksb, np.exp(samples.T))
    bfit = np.median(np.exp(samples[:, npt]))
    pmc = np.median(allsb, axis=1)
    pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)

    deproj.samples = samples
    deproj.sb = pmc
    deproj.sb_lo = pmcl
    deproj.sb_hi = pmch
    deproj.bkg = bfit
    
    
    
def Deproject_Multiscale_PyMC3(deproj,bkglim=None,nmcmc=1000,tune=500,back=None,samplefile=None,nrc=None,nbetas=6,min_beta=0.6):
    """
    Run the multiscale deprojection optimization using the PyMC backend

    :param deproj: Object of type :class:`pyproffit.deproject.Deproject` containing the data and parameters
    :type deproj: class:`pyproffit.deproject.Deproject`
    :param bkglim: Limit beyond which it is assumed that the background dominates, i.e. the source is set to 0. If bkglim=None (default), the entire radial range is used
    :type bkglim: float
    :param nmcmc: Number of HMC points in the output sample
    :type nmcmc: int
    :param back: Input value for the background, around which a gaussian prior is set. If back=None (default), the input background value will be computed as the average of the source-free region
    :type back: float
    :param samplefile: Path to output file to write the output samples. If samplefile=None (default), the data are not written to file and only loaded into memory
    :type samplefile: str
    :param nrc: Number of core radii. If nrc=None (default), the number of core radiis will be defined on-the-fly
    :type nrc: int
    :param nbetas: Number of beta values. Default=6
    :type nbetas: int
    :param min_beta: Minimum value of beta. Default=0.6
    :type min_beta: float
    """

    prof = deproj.profile
    sb = prof.profile
    rad = prof.bins
    erad = prof.ebins
    counts = prof.counts
    area = prof.area
    exposure = prof.effexp
    bkgcounts = prof.bkgcounts

    # Define maximum radius for source deprojection, assuming we have only background for r>bkglim
    if bkglim is None:
        bkglim=np.max(rad+erad)
        deproj.bkglim = bkglim
        if back is None:
            back = sb[len(sb) - 1]
    else:
        deproj.bkglim = bkglim
        backreg = np.where(rad>bkglim)
        if back is None:
            back = np.mean(sb[backreg])

    # Set source region
    sourcereg = np.where(rad < bkglim)
    nptfit = len(sb[sourcereg])

    # Set vector with list of parameters
    pars = list_params(rad, sourcereg, nrc, nbetas, min_beta)

    npt = len(pars)

    if prof.psfmat is not None:
        psfmat = np.transpose(prof.psfmat)
    else:
        psfmat = np.eye(prof.nbin)

    # Compute linear combination kernel
    K = calc_linear_operator(rad, sourcereg, pars, area, exposure, psfmat)
    basic_model = pm.Model()
    if np.isnan(sb[0]) or sb[0] <= 0:
        testval = -10.
    else:
        testval = np.log(sb[0] / npt)
    if np.isnan(back) or back == 0:
        testbkg = -10.
    else:
        testbkg = np.log(back)

    with basic_model:
        # Priors for unknown model parameters
        coefs = pm.Normal('coefs', mu=testval, sigma=20, shape=npt)
        bkgd = pm.Normal('bkg', mu=testbkg, sigma=0.05, shape=1)
        ctot = pm.math.concatenate((coefs, bkgd), axis=0)

        # Expected value of outcome
        al = pm.math.exp(ctot)
        pred = pm.math.dot(K, al) + bkgcounts

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Poisson('counts', mu=pred, observed=counts)

    tinit = time.time()

    isjax = False

    try:
        import pymc.sampling.jax as pmjax

    except ImportError:
        print('JAX not found, using default sampler')

    else:
        isjax = True
        import pymc.sampling.jax as pmjax

    print('Running MCMC...')
    with basic_model:
        tm = pm.find_MAP()

        if not isjax:

            trace = pm.sample(nmcmc, initvals=tm, tune=tune)

        else:

            trace = pmjax.sample_numpyro_nuts(nmcmc, initvals=tm, tune=tune)

    print('Done.')
    tend = time.time()
    print(' Total computing time is: ', (tend - tinit) / 60., ' minutes')

    # Get chains and save them to file
    chain_coefs = trace.posterior['coefs'].to_numpy()
    sc_coefs = chain_coefs.shape
    sampc = chain_coefs.reshape(sc_coefs[0] * sc_coefs[1], sc_coefs[2])

    sampb = trace.posterior['bkg'].to_numpy().flatten()

    spb = np.array([sampb]).T

    samples = np.append(sampc, spb, axis=1)
    if samplefile is not None:
        np.savetxt(samplefile, samples)
        np.savetxt(samplefile+'.par',np.array([pars.shape[0]/nbetas,nbetas,min_beta,nmcmc]),header='pymc3')

    # Compute output deconvolved brightness profile
    Ksb = calc_sb_operator(rad, sourcereg, pars)
    allsb = np.dot(Ksb, np.exp(samples.T))
    bfit = np.median(np.exp(samples[:, npt]))
    pmc = np.median(allsb, axis=1)
    pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)

    deproj.samples = samples
    deproj.sb = pmc
    deproj.sb_lo = pmcl
    deproj.sb_hi = pmch
    deproj.bkg = bfit



class MyDeprojVol:
    """
    Class to compute the projection volumes

    :param radin: Array of inner radii of the bins
    :type radin: class:`numpy.ndarray`
    :param radout: Array of outer radii of the bins
    :type radout: class:`numpy.ndarray`
    """
    def __init__(self, radin, radout):
        """
        Constructor for class MyDeprojVol

        """
        self.radin=radin
        self.radout=radout
        self.help=''

    def deproj_vol(self):
        """
        Compute the projection volumes

        :return: Volume matrix
        :rtype: numpy.ndarray
        """
        ###############volume=deproj_vol(radin,radot)
        ri=np.copy(self.radin)
        ro=np.copy(self.radout)

        diftot=0
        for i in range(1,len(ri)):
            dif=abs(ri[i]-ro[i-1])/ro[i-1]*100.
            diftot=diftot+dif
            ro[i-1]=ri[i]

        if abs(diftot) > 0.1:
            print(' DEPROJ_VOL: WARNING - abs(ri(i)-ro(i-1)) differs by',diftot,' percent')
            print(' DEPROJ_VOL: Fixing up radii ... ')
            for i in range(1,len(ri)-1):
                dif=abs(ri[i]-ro[i-1])/ro[i-1]*100.
                diftot=diftot+dif
        nbin=len(ro)
        volconst=4./3.*np.pi
        volmat=np.zeros((nbin, nbin))

        for iring in list(reversed(range(0,nbin))):
            volmat[iring,iring]=volconst * ro[iring]**3 * (1.-(ri[iring]/ro[iring])**2.)**1.5
            for ishell in list(reversed(range(iring+1,nbin))):
                f1=(1.-(ri[iring]/ro[ishell])**2.)**1.5 - (1.-(ro[iring]/ro[ishell])**2.)**1.5
                f2=(1.-(ri[iring]/ri[ishell])**2.)**1.5 - (1.-(ro[iring]/ri[ishell])**2.)**1.5
                volmat[ishell,iring]=volconst * (f1*ro[ishell]**3 - f2*ri[ishell]**3)

                if volmat[ishell,iring] < 0.0:
                    exit()

        volume2=np.copy(volmat)
        return volume2

def medsmooth(prof):
    """
    Smooth a given profile by taking the median value of surrounding points instead of the initial value

    :param prof: Input profile to be smoothed
    :type prof: numpy.ndarray
    :return: Smoothd profile
    :rtype: numpy.ndarray
    """
    width=5
    nbin=len(prof)
    xx=np.empty((nbin,width))
    xx[:,0]=np.roll(prof,2)
    xx[:,1]=np.roll(prof,1)
    xx[:,2]=prof
    xx[:,3]=np.roll(prof,-1)
    xx[:,4]=np.roll(prof,-2)
    smoothed=np.median(xx,axis=1)
    smoothed[1]=np.median(xx[1,1:width])
    smoothed[nbin-2]=np.median(xx[nbin-2,0:width-1])
    Y0=3.*prof[0]-2.*prof[1]
    xx=np.array([Y0,prof[0],prof[1]])
    smoothed[0]=np.median(xx)
    Y0=3.*prof[nbin-1]-2.*prof[nbin-2]
    xx=np.array([Y0,prof[nbin-2],prof[nbin-1]])
    smoothed[nbin-1]=np.median(xx)
    return  smoothed

def EdgeCorr(nbin,rin_cm,rout_cm,em0):
    # edge correction
    rin_kpc = rin_cm / kpc
    rout_kpc = rout_cm / kpc

    mrad = [rin_cm[nbin - 1], rout_cm[nbin - 1]]
    edge0 = (mrad[0] + mrad[1]) * mrad[0] * mrad[1] / rout_cm ** 3
    edge1 = 2. * rout_cm / mrad[1] + np.arccos(rout_cm / mrad[1])
    edge2 = rout_cm / mrad[1] * np.sqrt(1. - rout_cm ** 2 / mrad[1] ** 2)
    edget = edge0 * (-1. + 2. / np.pi * (edge1 - edge2))
    j = np.where(rin_cm != 0)
    edge0[j] = (mrad[0] + mrad[1]) * mrad[0] * mrad[1] / (rin_cm[j] + rout_cm[j]) / rin_cm[j] / rout_cm[j]
    edge1[j] = rout_cm[j] / rin_cm[j] * np.arccos(rin_cm[j] / mrad[1]) - np.arccos(rout_cm[j] / mrad[1])
    edge2[j] = rout_cm[j] / mrad[1] * (
                np.sqrt(1. - rin_cm[j] ** 2 / mrad[1] ** 2) - np.sqrt(1. - rout_cm[j] ** 2 / mrad[1] ** 2))
    edget[j] = edge0[j] * (1. - 2. / np.pi * (edge1[j] - edge2[j]) / (rout_cm[j] / rin_cm[j] - 1.))
    surf = (rout_cm ** 2 - rin_cm ** 2) / (rout_cm[nbin - 1] ** 2 - rin_cm[nbin - 1] ** 2)
    corr = edget * surf * em0[nbin-1] / em0.clip(min=1e-10)
    return corr

def OP(deproj,nmc=1000):
    # Standard onion peeling
    prof=deproj.profile
    nbin=prof.nbin
    rinam=prof.bins - prof.ebins
    routam=prof.bins + prof.ebins
    area=np.pi*(routam**2-rinam**2) # full area in arcmin^2
    cosmo = prof.cosmo

    # Projection volumes
    if deproj.z is not None and deproj.cf is not None:
        amin2kpc = cosmo.kpc_proper_per_arcmin(deproj.z).value
        rin_cm = rinam * amin2kpc * kpc
        rout_cm = routam * amin2kpc * kpc
        x=MyDeprojVol(rin_cm,rout_cm)
        vol=np.transpose(x.deproj_vol())
        dlum=cosmo.luminosity_distance(deproj.z).value*Mpc
        K2em=4.*np.pi*1e14*dlum**2/(1+deproj.z)**2/deproj.nhc/deproj.cf

        # Projected emission measure profiles
        em0 = prof.profile * K2em * area
        e_em0 = prof.eprof * K2em * area
        corr = EdgeCorr(nbin, rinam, routam, em0)
    else:
        x=MyDeprojVol(rinam,routam)
        vol=np.transpose(x.deproj_vol()).T
        em0 = prof.profile * area
        e_em0 = prof.profile * area
        corr = EdgeCorr(nbin,rinam,routam)

    # Deproject and propagate error using Monte Carlo
    emres = np.repeat(e_em0,nmc).reshape(nbin,nmc) * np.random.randn(nbin,nmc) + np.repeat(em0,nmc).reshape(nbin,nmc)
    ct = np.repeat(corr,nmc).reshape(nbin,nmc)
    #print(ct)
    allres = np.linalg.solve(vol, emres * (1. - ct))
    ev0 = np.std(allres,axis=1)
    v0 = np.median(allres,axis=1)
    bsm = medsmooth(v0)

    deproj.sb = bsm
    deproj.sb_lo = bsm - ev0
    deproj.sb_hi = bsm + ev0
    deproj.rout = routam

    deproj.dens = medsmooth(np.sign(bsm)*np.sqrt(np.abs(bsm)))
    edens = 0.5/np.sqrt(np.abs(bsm))*ev0
    deproj.dens_lo = deproj.dens - edens
    deproj.dens_hi = deproj.dens + edens

class Deproject(object):
    """
    Class to perform all calculations of deprojection, density profile, gas mass, count rate, and luminosity

    :param z: Source redshift. If z=None, only the surface brightness reconstruction can be done.
    :type z: float
    :param profile: Object of type :class:`pyproffit.profextract.Profile` containing the surface brightness profile data
    :type profile: class:`pyproffit.profextract.Profile`
    :param cf: Conversion factor from count rate to emissivity. If cf=None, only the surface brightness reconstruction can be done.
    :type cf: float
    :param f_abund: Solar abundance table to compute the electron-to-proton ratio and mean molecular weight. Available tables are 'aspl' (Aspling+09, default), 'angr' (Anders & Grevesse 89), and 'grsa' (Grevesse & Sauval 98)
    :type f_abund: str
     """
    def __init__(self,z=None,profile=None,cf=None,f_abund='aspl'):
        """
        Constructor for class pyproffit.Deproject
       """
        self.profile = profile
        self.z = z
        self.samples = None
        self.cf = cf
        if cf is None and profile.ccf is not None:
            self.cf = profile.ccf
        self.lumfact = None
        if profile.lumfact is not None:
            self.lumfact = profile.lumfact
        self.dens = None
        self.dens_lo = None
        self.dens_hi = None
        self.sb = None
        self.sb_lo = None
        self.sb_hi = None
        self.covmat = None
        self.bkg = None
        self.samples = None
        self.bkglim = None
        self.rout = None
        self.pmc = None
        self.pmcl = None
        self.pmch = None
        self.mg = None
        self.mgl = None
        self.mgh = None
        self.rec_counts=None
        self.rec_counts_lo=None
        self.rec_counts_hi=None

        # mu_e: mean molecular weight per electron in pristine fully ionized gas with given abundance table
        # mup: mean molecular weight per particle  in pristine fully ionized gas with given abundance table
        # nhc: conversion factor from H n-density to e- n-density

        if f_abund == 'angr':
            nhc = 1 / 0.8337
            mup = 0.6125
            mu_e = 1.1738
        elif f_abund == 'aspl':
            nhc = 1 / 0.8527
            mup = 0.5994
            mu_e = 1.1548
        elif f_abund == 'grsa':
            nhc = 1 / 0.8520
            mup = 0.6000
            mu_e = 1.1555
        else:  # aspl default
            nhc = 1 / 0.8527
            mup = 0.5994
            mu_e = 1.1548
        self.nhc=nhc
        self.mup=mup
        self.mu_e=mu_e


    def Multiscale(self,backend='pymc3',nmcmc=1000,tune=500,bkglim=None,back=None,samplefile=None,nrc=None,nbetas=6,depth=10,min_beta=0.6):
        """
        Run Multiscale deprojection using the method described in Eckert+20

        :param backend: Backend to run the optimization problem. Available backends are 'pymc3' (default) and 'stan'
        :type backend: str
        :param nmcmc: Number of HMC points in the output sample
        :type nmcmc: int
        :param tune: Number of HMC tuning steps
        :type tune: int
        :param bkglim: Limit beyond which it is assumed that the background dominates, i.e. the source is set to 0. If bkglim=None (default), the entire radial range is used
        :type bkglim: float
        :param back: Input value for the background, around which a gaussian prior is set. If back=None (default), the input background value will be computed as the average of the source-free region
        :type back: float
        :param samplefile: Path to output file to write the output samples. If samplefile=None (default), the data are not written to file and only loaded into memory
        :type samplefile: str
        :param nrc: Number of core radii. If nrc=None (default), the number of core radiis will be defined on-the-fly
        :type nrc: int
        :param nbetas: Number of beta values. Default=6
        :type nbetas: int
        :param depth: Set the max_treedepth parameter for Stan (default=10)
        :type depth: int
        :param min_beta: Minimum value of beta. Default=0.6
        :type min_beta: float
        """
        self.backend=backend
        self.nmcmc=nmcmc
        self.bkglim=bkglim
        self.back=back
        self.samplefile=samplefile
        self.nrc=nrc
        self.nbetas=nbetas
        self.min_beta=min_beta
        self.depth=depth
        if backend=='pymc3':
            Deproject_Multiscale_PyMC3(self,bkglim=bkglim,back=back,nmcmc=nmcmc,tune=tune,samplefile=samplefile,nrc=nrc,nbetas=nbetas,min_beta=min_beta)
        elif backend=='stan':
            Deproject_Multiscale_Stan(self,bkglim=bkglim,back=back,nmcmc=nmcmc,samplefile=samplefile,nrc=nrc,nbetas=nbetas,depth=depth,min_beta=min_beta)
        else:
            print('Unknown method '+backend)


    def OnionPeeling(self,nmc=1000):
        """
        Run standard Onion Peeling deprojection using the Kriss+83 method and the McLaughlin+99 edge correction

        :param nmc: Number of Monte Carlo realizations of the profile to compute the uncertainties (default=1000)
        :type nmc: int
        """
        OP(self,nmc)

    def PlotDensity(self,outfile=None,xunit='kpc', figsize=(13, 10), fontsize=40, color='C0', lw=2, **kwargs):
        """
        Plot the loaded density profile

        :param outfile: Output file name. If outfile=None (default) plot only to stdout
        :type outfile: str
        :param xunit: Choose whether the x axis should be in unit of 'kpc' (default), 'arcmin', or 'both', in which case two axes are drawn at the top and the bottom of the plot
        :type xunit: str
        :param figsize: Size of figure. Defaults to (13, 10)
        :type figsize: tuple , optional
        :param fontsize: Font size of the axis labels. Defaults to 40
        :type fontsize: int , optional
        :param color: Line color following matplotlib conventions. Defaults to 'blue'
        :type color: str , optional
        :param lw: Line width. Defaults to 2
        :type lw: int , optional
        :param kwargs: Additional arguments to be passed to :class:`matplotlib.pyplot.plot`
        """
        # Plot extracted profile
        if self.profile is None:
            print('Error: No profile extracted')
            return
        if self.dens is None:
            print('Error: No density profile extracted')
            return
        if xunit not in ['arcmin','kpc','both']:
            xunit='kpc'

        cosmo = self.profile.cosmo

        sourcereg_out = np.where(self.rout <= self.bkglim)

        kpcp = cosmo.kpc_proper_per_arcmin(self.z).value

        rkpc = self.rout[sourcereg_out] * kpcp

        rout = self.rout[sourcereg_out]
        #erkpc = self.profile.ebins * kpcp

        plt.clf()
        fig = plt.figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(111)
        ax.minorticks_on()
        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        ax.set_ylabel('$n_{H}$ [cm$^{-3}$]', fontsize=fontsize)
        ax.set_xscale('log')
        ax.set_yscale('log')

        if xunit == 'kpc' or xunit == 'both':
            ax.plot(rkpc, self.dens, color=color, lw=lw, **kwargs)
            ax.fill_between(rkpc, self.dens_lo, self.dens_hi, color=color, alpha=0.5)
            ax.set_xlabel('Radius [kpc]', fontsize=fontsize)
        else:
            ax.plot(rout, self.dens, color=color, lw=lw, **kwargs)
            ax.fill_between(rout, self.dens_lo, self.dens_hi, color=color, alpha=0.5)
            ax.set_xlabel('Radius [arcmin]', fontsize=fontsize)

        if xunit == 'both':
            limx=ax.get_xlim()
            ax2 = ax.twiny()
            ax2.set_xlim([limx[0]/ kpcp,limx[1]/ kpcp])
            ax2.set_xscale('log')
            ax2.tick_params(length=20, width=1, which='major', direction='in', right='on', top='on')
            ax2.tick_params(length=10, width=1, which='minor', direction='in', right='on', top='on')
            ax2.set_xlabel('Radius [arcmin]', fontsize=fontsize, labelpad=20)
            for item in (ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(18)

        #ax.errorbar(rkpc, self.dens, xerr=erkpc, yerr=[self.dens-self.dens_lo,self.dens_hi-self.dens], fmt='.', color='C0', elinewidth=2, markersize=7, capsize=3)

        if outfile is not None:
            plt.savefig(outfile)
            plt.close()
        else:
            plt.show(block=False)

    def Density(self,rout=None):
        """
        Compute a density profile from a multiscale reconstruction

        :param rout: Radial binning of the density profile. If rout=None, the original binning of the surface brightness profile is used
        :type rout: numpy.ndarray
        """
        z = self.z
        cf = self.cf
        samples = self.samples
        prof = self.profile
        rad = prof.bins
        sourcereg = np.where(rad < self.bkglim)

        if z is not None and cf is not None:
            transf = 4. * (1. + z) ** 2 * (180. * 60.) ** 2 / np.pi / 1e-14 / self.nhc / Mpc * 1e3
            pardens = list_params_density(rad, sourcereg, z, prof.cosmo,
                                          nrc=self.nrc, nbetas=self.nbetas, min_beta=self.min_beta)
            if rout is None:
                sourcereg_out=sourcereg
                rout=rad
            else:
                sourcereg_out=np.where(rout < self.bkglim)

            rd = rout[sourcereg_out]
            Kdens = calc_density_operator(rd, pardens, z, prof.cosmo)
            alldens = np.sqrt(np.dot(Kdens, np.exp(samples.T)) / cf * transf)  # [0:nptfit, :]
            covmat = np.cov(alldens)
            self.covmat = covmat
            pmcd = np.median(alldens, axis=1)
            pmcdl = np.percentile(alldens, 50. - 68.3 / 2., axis=1)
            pmcdh = np.percentile(alldens, 50. + 68.3 / 2., axis=1)
            self.dens = pmcd
            self.dens_lo = pmcdl
            self.dens_hi = pmcdh
            self.rout=rout

        else:
            print('No redshift and/or conversion factor, nothing to do')

    def PlotSB(self,outfile=None, figsize=(13, 10), fontsize=40, xscale='log', yscale='log', lw=2,
               fmt='d', markersize=7, data_color='red', bkg_color='green', model_color='C0', skybkg_color='k'):
        """
        Plot the surface brightness profile reconstructed after applying the multiscale deprojection and PSF deconvolution technique, and compare it with the input brightness profile

        :param outfile: File name of saving the output figure. If outfile=None (default), plot only to stdout
        :type outfile: str
        :param figsize: Size of figure. Defaults to (13, 10)
        :type figsize: tuple , optional
        :param fontsize: Font size of the axis labels. Defaults to 40
        :type fontsize: int , optional
        :param xscale: Scale of the X axis. Defaults to 'log'
        :type xscale: str , optional
        :param yscale: Scale of the Y axis. Defaults to 'log'
        :type yscale: str , optional
        :param lw: Line width. Defaults to 2
        :type lw: int , optional
        :param fmt: Marker type following matplotlib convention. Defaults to 'd'
        :type fmt: str , optional
        :param markersize: Marker size. Defaults to 7
        :type markersize: int , optional
        :param data_color: Color of the data points following matplotlib convention. Defaults to 'red'
        :type data_color: str , optional
        :param bkg_color: Color of the particle background following matplotlib convention. Defaults to 'green'
        :type bkg_color: str , optional
        :param model_color: Color of the surface brightness model following matplotlib convention. Defaults to 'C0'
        :type model_color: str , optional
        :param skybkg_color: Color of the fitted sky background model following matplotlib convention. Defaults to 'k'
        :type skybkg_color: str , optional
        """
        if self.profile is None:
            print('Error: No profile extracted')
            return
        if self.sb is None:
            print('Error: No reconstruction available')
            return
        prof=self.profile
        plt.clf()
        fig = plt.figure(figsize=figsize)

        ax=fig.add_axes([0.12,0.2,0.8,0.7])
        ax_res=fig.add_axes([0.12,0.1,0.8,0.1])

        ax_res.set_xlabel('Radius [arcmin]', fontsize=fontsize)
        ax.set_ylabel('SB [counts s$^{-1}$ arcmin$^{-2}$]', fontsize=fontsize)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        #ax.errorbar(prof.bins, prof.profile, xerr=prof.ebins, yerr=prof.eprof, fmt='o', color='black', elinewidth=2,
        #            markersize=7, capsize=0, mec='black', label='Bkg - subtracted Data')

        ax.errorbar(prof.bins, prof.counts / prof.area / prof.effexp, xerr=prof.ebins, yerr=prof.eprof, fmt=fmt,
                    color=data_color, elinewidth=2,
                    markersize=markersize, capsize=0, label='Data')
        ax.plot(prof.bins, prof.bkgprof, color=bkg_color, label='Particle background')

        # plt.errorbar(self.profile.bins, self.sb, xerr=self.profile.ebins, yerr=[self.sb-self.sb_lo,self.sb_hi-self.sb], fmt='o', color='blue', elinewidth=2,  markersize=7, capsize=0,mec='blue',label='Reconstruction')
        ax.plot(prof.bins, self.sb, color=model_color, lw=lw, label='Source model')
        ax.fill_between(prof.bins, self.sb_lo, self.sb_hi, color=model_color, alpha=0.5)

        ax.axhline(self.bkg,color=skybkg_color,label='Sky background')

        #compute SB profile without bkg subtraction to get residuals on fit
        # Set vector with list of parameters
        sourcereg = np.where(prof.bins < self.bkglim)
        pars = list_params(prof.bins, sourcereg, self.nrc, self.nbetas, self.min_beta)
        npt = len(pars)
        # Compute output deconvolved brightness profile
        if prof.psfmat is not None:
            psfmat = np.transpose(prof.psfmat)
        else:
            psfmat = np.eye(prof.nbin)
        samples=self.samples
        Ksb = calc_sb_operator_psf(prof.bins, sourcereg, pars, prof.area, prof.effexp, psfmat)
        allsb = np.dot(Ksb, np.exp(samples.T))
        bfit = np.median(np.exp(samples[:, npt]))
        pmc = np.median(allsb, axis=1) / prof.area / prof.effexp + prof.bkgprof
        pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1) / prof.area / prof.effexp + prof.bkgprof
        pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1) / prof.area / prof.effexp + prof.bkgprof

        ax.plot(prof.bins, pmc, color='C1', lw=lw, label='Total model')
        ax.fill_between(prof.bins, pmcl, pmch, color='C1', alpha=0.5)

        self.pmc=pmc
        self.pmcl=pmcl
        self.pmch=pmch

        ax.legend(loc=0,fontsize=22)

        res = (pmc * prof.area * prof.effexp - prof.counts) / (pmc * prof.area * prof.effexp)
        vmin=-0.5
        veff=np.max(np.abs(res))
        if veff > vmin:
            vmin=veff*1.2
        ax_res.scatter(prof.bins, res, color=data_color, lw=lw)
        ax_res.axhline(0, color='k')

        ax.set_xticklabels([])
        ax_res.set_xscale(xscale)
        ax.legend(loc=0)

        ax.minorticks_on()
        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
        ax_res.minorticks_on()
        ax_res.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
        ax_res.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        ax_res.set_xlim(ax.get_xlim())
        ii=np.where(self.sb > 0)
        ax.set_ylim([0.8 * np.min(self.sb[ii]), 1.2 * np.max(self.sb)])
        ax_res.set_ylim([-vmin,vmin])
        if outfile is not None:
            plt.savefig(outfile)
            plt.close()
        else:
            plt.show(block=False)


    def CountRate(self,a,b,plot=True,outfile=None, figsize=(13, 10), nbins=30, fontsize=40, yscale='linear', **kwargs):
        """
        Compute the model count rate integrated between radii a and b. Optionally, the count rate distribution can be plotted and saved.

        :param a: Inner integration boundary in arcmin
        :type a: float
        :param b: Outer integration boundary in arcmin
        :type b: float
        :param plot: Plot the posterior count rate distribution (default=True)
        :type plot: bool
        :param outfile: Output file name to save the figure. If outfile=None, plot only to stdout
        :type outfile: str
        :param figsize: Size of figure. Defaults to (13, 10)
        :type figsize: tuple , optional
        :param nbins: Number of bins on the X axis to construct the posterior distribution. Defaults to 30
        :type nbins: int , optional
        :param fontsize: Font size of the axis labels. Defaults to 40
        :type fontsize: int , optional
        :param yscale: Scale on the Y axis. Defaults to 'linear'
        :type yscale: str , optional
        :param kwargs: Options to be passed to :class:`matplotplib.pyplot.hist`
        :return: Median count rate, 16th and 84th percentiles
        :rtype: float
        """
        if self.samples is None:
            print('Error: no MCMC samples found')
            return
        # Set source region
        prof = self.profile
        rad = prof.bins
        sourcereg = np.where(rad < self.bkglim)

        # Avoid diverging profiles in the center by cutting to the innermost points, if necessary
        if a<prof.bins[0]/2.:
            a = prof.bins[0]/2.

        # Set vector with list of parameters
        pars = list_params(rad, sourcereg, self.nrc, self.nbetas, self.min_beta)
        Kint = calc_int_operator(a, b, pars)
        allint = np.dot(Kint, np.exp(self.samples.T))
        medint = np.median(allint[1, :] - allint[0, :])
        intlo = np.percentile(allint[1, :] - allint[0, :], 50. - 68.3 / 2.)
        inthi = np.percentile(allint[1, :] - allint[0, :], 50. + 68.3 / 2.)
        print('Reconstructed count rate: %g (%g , %g)' % (medint, intlo, inthi))
        if plot:
            plt.clf()
            fig = plt.figure(figsize=figsize)
            ax_size = [0.14, 0.12,
                       0.85, 0.85]
            ax = fig.add_axes(ax_size)
            ax.minorticks_on()
            ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
            ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(22)
            plt.yscale(yscale)
            plt.hist(allint[1,:]-allint[0,:], bins=nbins, **kwargs)
            plt.xlabel('Count Rate [cts/s]', fontsize=fontsize)
            plt.ylabel('Frequency', fontsize=fontsize)
            if outfile is not None:
                plt.savefig(outfile)
                plt.close()
            else:
                plt.show(block=False)

        return  medint,intlo,inthi

    def Luminosity(self,a,b,plot=True,outfile=None, figsize=(13, 10), nbins=30, fontsize=40, yscale='linear', **kwargs):
        """
        Compute the luminosity integrated between radii a and b. Optionally, the luminosity distribution can be plotted and saved. Requires the luminosity factor to be computed using the :meth:`pyproffit.profextract.Profile.Emissivity` method.

        :param a: Inner integration boundary in arcmin
        :type a: float
        :param b: Outer integration boundary in arcmin
        :type b: float
        :param plot: Plot the posterior count rate distribution (default=True)
        :type plot: bool
        :param outfile: Output file name to save the figure. If outfile=None, plot only to stdout
        :type outfile: str
        :param figsize: Size of figure. Defaults to (13, 10)
        :type figsize: tuple , optional
        :param nbins: Number of bins on the X axis to construct the posterior distribution. Defaults to 30
        :type nbins: int , optional
        :param fontsize: Font size of the axis labels. Defaults to 40
        :type fontsize: int , optional
        :param yscale: Scale on the Y axis. Defaults to 'linear'
        :type yscale: str , optional
        :param kwargs: Options to be passed to :class:`matplotplib.pyplot.hist`
        :return: Median luminosity, 16th and 84th percentiles
        :rtype: float
        """
        if self.samples is None:
            print('Error: no MCMC samples found')
            return

        if self.lumfact is None:
            print('Error: no luminosity conversion factor found')
            return

        # Set source region
        prof = self.profile
        rad = prof.bins
        sourcereg = np.where(rad < self.bkglim)

        # Avoid diverging profiles in the center by cutting to the innermost points, if necessary
        if a<prof.bins[0]/2.:
            a = prof.bins[0]/2.

        # Set vector with list of parameters
        pars = list_params(rad, sourcereg, self.nrc, self.nbetas, self.min_beta)
        Kint = calc_int_operator(a, b, pars)
        allint = np.dot(Kint, np.exp(self.samples.T)) * self.lumfact
        medint = np.median(allint[1, :] - allint[0, :])
        intlo = np.percentile(allint[1, :] - allint[0, :], 50. - 68.3 / 2.)
        inthi = np.percentile(allint[1, :] - allint[0, :], 50. + 68.3 / 2.)
        print('Reconstructed luminosity: %g (%g , %g)' % (medint, intlo, inthi))
        if plot:
            plt.clf()
            fig = plt.figure(figsize=figsize)
            ax_size = [0.14, 0.12,
                       0.85, 0.85]
            ax = fig.add_axes(ax_size)
            ax.minorticks_on()
            ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
            ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(22)
            plt.yscale(yscale)
            plt.hist(allint[1,:]-allint[0,:], bins=nbins, **kwargs)
            plt.xlabel('$L_{X}$ [erg/s]', fontsize=fontsize)
            plt.ylabel('Frequency', fontsize=fontsize)
            if outfile is not None:
                plt.savefig(outfile)
                plt.close()
            else:
                plt.show(block=False)

        return  medint,intlo,inthi

    def Ncounts(self,plot=True,outfile=None, figsize=(13, 10), nbins=30, fontsize=40, yscale='linear', **kwargs):
        """
        Compute the total model number of counts. Optionally, the posterior distribution can be plotted and saved.

        :param plot: Plot the posterior distribution of number of counts (default=True)
        :type plot: bool
        :param outfile: Output file name to save the figure. If outfile=None, plot only to stdout
        :type outfile: str
        :param figsize: Size of figure. Defaults to (13, 10)
        :type figsize: tuple , optional
        :param nbins: Number of bins on the X axis to construct the posterior distribution. Defaults to 30
        :type nbins: int , optional
        :param fontsize: Font size of the axis labels. Defaults to 40
        :type fontsize: int , optional
        :param yscale: Scale on the Y axis. Defaults to 'linear'
        :type yscale: str , optional
        :param kwargs: Options to be passed to :class:`matplotplib.pyplot.hist`
        :return: Median number of counts, 16th and 84th percentiles
        :rtype: float
        """
        if self.samples is None:
            print('Error: no MCMC samples found')
            return
        # Set source region
        prof = self.profile
        rad = prof.bins
        sourcereg = np.where(rad < self.bkglim)
        area = prof.area
        exposure = prof.effexp

        if prof.psfmat is not None:
            psfmat = np.transpose(prof.psfmat)
        else:
            psfmat = np.eye(prof.nbin)

        # Set vector with list of parameters
        pars = list_params(rad, sourcereg, self.nrc, self.nbetas, self.min_beta)
        K = calc_linear_operator(rad, sourcereg, pars, area, exposure, psfmat)
        npars = len(pars[:, 0])
        K[:,npars] = 0.
        allnc = np.dot(K, np.exp(self.samples.T))
        self.rec_counts,self.rec_counts_lo,self.rec_counts_hi=np.percentile(allnc,[50.,50.-68.3/2.,50.+68.3/2.],axis=1)
        ncv = np.sum(allnc, axis=0)
        pnc = np.median(ncv)
        pncl = np.percentile(ncv, 50. - 68.3 / 2.)
        pnch = np.percentile(ncv, 50. + 68.3 / 2.)
        print('Reconstructed counts: %g (%g , %g)' % (pnc, pncl, pnch))
        if plot:
            plt.clf()
            fig = plt.figure(figsize=figsize)
            ax_size = [0.14, 0.12,
                       0.85, 0.85]
            ax = fig.add_axes(ax_size)
            ax.minorticks_on()
            ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
            ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(22)
            plt.yscale(yscale)
            plt.hist(ncv, bins=nbins, **kwargs)
            plt.xlabel('$N_{count}$', fontsize=fontsize)
            plt.ylabel('Frequency', fontsize=fontsize)
            if outfile is not None:
                plt.savefig(outfile)
                plt.close()
            else:
                plt.show(block=False)

        return  pnc,pncl,pnch


    # Compute Mgas within radius in kpc
    def Mgas(self, radius, radius_err=None, plot=True, outfile=None, rad_scale='normal', figsize=(13, 10), nbins=30, fontsize=40, yscale='linear', **kwargs):
        """
        Compute the posterior cumulative gas mass within a given radius. Optionally, the posterior distribution can be plotted and saved.

        :param radius: Gas mass integration radius in kpc
        :type radius: float
        :param radius_err: (Gaussian) error on the input radius to be propagated to the gas mass measurement. To be used in case one wants to evaluate :math:`M_{gas}` at an overdensity radius with a known uncertainty
        :type radius_err: float , optional
        :param plot: Plot the posterior Mgas distribution (default=True)
        :type plot: bool
        :param outfile: Output file name to save the figure. If outfile=None, plot only to stdout
        :type outfile: str , optional
        :param rad_scale: If radius_err is not None, specify whether the distribution of radii is drawn from a normal distribution (rad_scale='normal') or a log-normal distribution (rad_scale='lognormal'). Defaults to 'normal'.
        :type rad_scale: str
        :param figsize: Size of figure. Defaults to (13, 10)
        :type figsize: tuple , optional
        :param nbins: Number of bins on the X axis to construct the posterior distribution. Defaults to 30
        :type nbins: int , optional
        :param fontsize: Font size of the axis labels. Defaults to 40
        :type fontsize: int , optional
        :param yscale: Scale on the Y axis. Defaults to 'linear'
        :type yscale: str , optional
        :param kwargs: Options to be passed to :class:`matplotplib.pyplot.hist`
        :return: Median :math:`M_{gas}`, 16th and 84th percentiles
        :rtype: float
        """
        if self.samples is None or self.z is None or self.cf is None:
            print('Error: no gas density profile found')
            return

        prof = self.profile
        cosmo = prof.cosmo

        kpcp = cosmo.kpc_proper_per_arcmin(self.z).value
        rkpc = prof.bins * kpcp
        erkpc = prof.ebins * kpcp
        nhconv =  mh * self.mu_e * self.nhc * kpc ** 3 / msun  # Msun/kpc^3

        rad = prof.bins
        sourcereg = np.where(rad < self.bkglim)

        transf = 4. * (1. + self.z) ** 2 * (180. * 60.) ** 2 / np.pi / 1e-14 / self.nhc / Mpc * 1e3
        pardens = list_params_density(rad, sourcereg, self.z, cosmo, self.nrc, self.nbetas, self.min_beta)
        Kdens = calc_density_operator(rad, pardens, self.z, cosmo)

        # All gas density profiles
        alldens = np.sqrt(np.dot(Kdens, np.exp(self.samples.T)) / self.cf * transf)  # [0:nptfit, :]

        # Matrix containing integration volumes
        volmat = np.repeat(4. * np.pi * rkpc ** 2 * 2. * erkpc, alldens.shape[1]).reshape(len(prof.bins),alldens.shape[1])

        # Compute Mgas profile as cumulative sum over the volume
        mgas = np.cumsum(alldens * nhconv * volmat, axis=0)

        # Interpolate at the radius of interest

        # Set randomization of the radius if radius_err is not None
        rho = None

        if radius_err is not None:

            nsim = len(self.samples)

            if rad_scale == 'normal':

                radii = radius_err * np.random.randn(nsim) + radius

            elif rad_scale == 'lognormal':

                rad_log = np.log(radius)

                err_rad_log = radius_err / radius

                radii = np.exp(err_rad_log * np.random.randn(nsim) + rad_log)

            else:

                print('Unknown value rad_scale=%s , reverting to normal' % (rad_scale))

                radii = radius_err * np.random.randn(nsim) + radius

            if np.any(radii < 0.0):

                isneg = np.where(radii < 0.0)

                radii[isneg] = 0.0

            mgasdist = np.empty(len(self.samples))

            for i in range(len(self.samples)):

                mgasdist[i] = np.interp(radii[i], rkpc, mgas[:, i])

            rho = np.corrcoef(radii, mgasdist)[0,1]

        else:

            f = interp1d(rkpc, mgas, axis=0)

            mgasdist = f(radius)

        mg, mgl, mgh = np.percentile(mgasdist,[50.,50.-68.3/2.,50.+68.3/2.])
        if plot:
            plt.clf()
            fig = plt.figure(figsize=figsize)
            ax_size = [0.14, 0.12,
                       0.85, 0.85]
            ax = fig.add_axes(ax_size)
            ax.minorticks_on()
            ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
            ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(22)
            plt.yscale(yscale)
            plt.hist(mgasdist, bins=nbins, **kwargs)
            plt.xlabel('$M_{gas} [M_\odot]$', fontsize=fontsize)
            plt.ylabel('Frequency', fontsize=fontsize)
            if outfile is not None:
                plt.savefig(outfile)
                plt.close()
            else:
                plt.show(block=False)

        return mg,mgl,mgh,rho

    def Cov_Mgas_Lx(self, radius, radius_err=None, rad_scale='normal'):

        if self.samples is None or self.z is None or self.cf is None:
            print('Error: no gas density profile found')
            return
        prof = self.profile
        cosmo = prof.cosmo

        kpcp = cosmo.kpc_proper_per_arcmin(self.z).value
        rkpc = prof.bins * kpcp
        erkpc = prof.ebins * kpcp
        nhconv =  mh * self.mu_e * self.nhc * kpc ** 3 / msun  # Msun/kpc^3

        rad = prof.bins
        sourcereg = np.where(rad < self.bkglim)

        transf = 4. * (1. + self.z) ** 2 * (180. * 60.) ** 2 / np.pi / 1e-14 / self.nhc / Mpc * 1e3
        pardens = list_params_density(rad, sourcereg, self.z, cosmo, self.nrc, self.nbetas, self.min_beta)
        Kdens = calc_density_operator(rad, pardens, self.z, cosmo)

        # All gas density profiles
        alldens = np.sqrt(np.dot(Kdens, np.exp(self.samples.T)) / self.cf * transf)  # [0:nptfit, :]

        # Matrix containing integration volumes
        volmat = np.repeat(4. * np.pi * rkpc ** 2 * 2. * erkpc, alldens.shape[1]).reshape(len(prof.bins),alldens.shape[1])

        # Compute Mgas profile as cumulative sum over the volume
        mgas = np.cumsum(alldens * nhconv * volmat, axis=0)

        # Interpolate at the radius of interest
        # Avoid diverging profiles in the center by cutting to the innermost points, if necessary
        a = prof.bins[0]/2.

        # Set vector with list of parameters
        pars = list_params(rad, sourcereg, self.nrc, self.nbetas, self.min_beta)

        # Set randomization of the radius if radius_err is not None
        rho = None

        if radius_err is not None:

            nsim = len(self.samples)

            if rad_scale == 'normal':

                radii = radius_err * np.random.randn(nsim) + radius

            elif rad_scale == 'lognormal':

                rad_log = np.log(radius)

                err_rad_log = radius_err / radius

                radii = np.exp(err_rad_log * np.random.randn(nsim) + rad_log)

            else:

                print('Unknown value rad_scale=%s , reverting to normal' % (rad_scale))

                radii = radius_err * np.random.randn(nsim) + radius

            if np.any(radii < 0.0):

                isneg = np.where(radii < 0.0)

                radii[isneg] = 0.0

            mgasdist = np.empty(len(self.samples))
            allint = np.empty(len(self.samples))

            b = radii / kpcp

            for i in range(len(self.samples)):

                mgasdist[i] = np.interp(radii[i], rkpc, mgas[:, i])

                # Compute linear combination of basis functions in the source region
                Kint = calc_int_operator(a, b[i], pars)
                tal = np.dot(Kint, np.exp(self.samples.T)) * self.lumfact
                allint[i] = tal[1, i] - tal[0, i]

        else:

            f = interp1d(rkpc, mgas, axis=0)

            Kint = calc_int_operator(a, radius/kpcp, pars)

            tal = np.dot(Kint, np.exp(self.samples.T)) * self.lumfact
            allint = tal[1, :] - tal[0, :]

            mgasdist = f(radius)

        rho_lxmg = np.corrcoef(allint, mgasdist)[0,1]

        mg, mgl, mgh = np.percentile(mgasdist,[50.,50.-68.3/2.,50.+68.3/2.])
        medint, intlo, inthi = np.percentile(allint, [50.,50.-68.3/2.,50.+68.3/2.])

        outdict = {'MGAS': mg,
                   'MGAS_LO': mgl,
                   'MGAS_HI': mgh,
                   'LX': medint,
                   'LX_LO': intlo,
                   'LX_HI': inthi,
                   'RHO_LXMG': rho_lxmg
                   }
        if radius_err is not None:
            rho_rlx =  np.corrcoef(radii, allint)[0,1]
            rho_rmg =  np.corrcoef(radii, mgasdist)[0,1]

            outdict['RHO_RLX'] = rho_rlx
            outdict['RHO_RMG'] = rho_rmg

        return outdict


    def PlotMgas(self,rout=None,outfile=None,xunit="kpc", figsize=(13, 10), color='C0', lw=2, fontsize=40, xscale='log', yscale='log'):
        """
        Plot the cumulative gas mass profile from the output of a reconstruction

        :param rout: Radial binning of the gas mass profile. If rout=None, the original binning of the surface brightness profile is used
        :type rout: numpy.ndarray
        :param outfile: Output file name to save the figure. If outfile=None, plot only to stdout
        :type outfile: str
        :param xunit: Choose whether the x axis should be in unit of 'kpc' (default), 'arcmin', or 'both', in which case two axes are drawn at the top and the bottom of the plot
        :type xunit: str
        :param figsize: Size of figure. Defaults to (13, 10)
        :type figsize: tuple , optional
        :param color: Line color following matplotlib conventions. Defaults to 'C0'
        :type color: str , optional
        :param fontsize: Font size of the axis labels. Defaults to 40
        :type fontsize: int , optional
        :param xscale: Scale of the X axis. Defaults to 'log'
        :type xscale: str , optional
        :param yscale: Scale of the Y axis. Defaults to 'log'
        :type yscale: str , optional
        :param lw: Line width. Defaults to 2
        :type lw: int , optional
        """
        if self.samples is None or self.z is None or self.cf is None:
            print('Error: no gas density profile found')
            return

        if xunit not in ['arcmin','kpc','both']:
            xunit='kpc'


        prof = self.profile
        cosmo = prof.cosmo
        kpcp = cosmo.kpc_proper_per_arcmin(self.z).value
        if rout is None:
            rkpc = prof.bins * kpcp
            erkpc = prof.ebins * kpcp
        else:
            rkpc = rout * kpcp
            erkpc = (rout-np.append(0,rout[:-1]))/2 * kpcp
        nhconv =  mh * self.mu_e * self.nhc * kpc ** 3 / msun  # Msun/kpc^3

        rad = prof.bins
        sourcereg = np.where(rad < self.bkglim)

        transf = 4. * (1. + self.z) ** 2 * (180. * 60.) ** 2 / np.pi / 1e-14 / self.nhc / Mpc * 1e3
        pardens = list_params_density(rad, sourcereg, self.z, cosmo, self.nrc, self.nbetas, self.min_beta)
        if rout is None:
            sourcereg_out = sourcereg
            rout = rad
        else:
            sourcereg_out = np.where(rout < self.bkglim)

        Kdens = calc_density_operator(rout, pardens, self.z, cosmo)

        # All gas density profiles
        alldens = np.sqrt(np.dot(Kdens, np.exp(self.samples.T)) / self.cf * transf)  # [0:nptfit, :]


        # Matrix containing integration volumes
        volmat = np.repeat(4. * np.pi * rkpc ** 2 * 2. * erkpc, alldens.shape[1]).reshape(len(rout), alldens.shape[1])

        # Compute Mgas profile as cumulative sum over the volume
        mgasdist = np.cumsum(alldens * nhconv * volmat, axis=0)


        mg, mgl, mgh = np.percentile(mgasdist,[50.,50.-68.3/2.,50.+68.3/2.],axis=1)

        self.mg=mg
        self.mgl=mgl
        self.mgh=mgh


        #now compute mtot from mgas-mtot scaling relation
        rho_cz = cosmo.critical_density(self.z).to(u.Msun / u.kpc ** 3).value

        Mgas = fbul19(rkpc,self.z,cosmo,Runit='kpc')

        Mgasdist = np.repeat(Mgas, alldens.shape[1]).reshape(len(rout), alldens.shape[1])

        self.r500, self.r500_l, self.r500_h = np.percentile(rkpc[np.argmin(np.abs(Mgasdist / mgasdist - 1), axis=0)], [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

        self.m500, self.m500_l, self.m500_h = 4. / 3. * np.pi * 500 * rho_cz * self.r500 ** 3, 4. / 3. * np.pi * 500 * rho_cz * self.r500_l ** 3, 4. / 3. * np.pi * 500 * rho_cz * self.r500_h ** 3

        self.t500, self.t500_l, self.t500_h = self.r500/kpcp, self.r500_l/kpcp, self.r500_h/kpcp

        fig = plt.figure(figsize=figsize,tight_layout=True)
        ax=fig.add_subplot(111)

        if xunit == 'kpc' or xunit == 'both':
            ax.plot(rkpc, mg, color=color, lw=lw, label='Gas mass')
            ax.fill_between(rkpc, mgl, mgh, color=color, alpha=0.5)
            ax.set_xlabel('Radius [kpc]', fontsize=fontsize)
        else:
            ax.plot(rout, mg, color=color, lw=lw, label='Gas mass')
            ax.fill_between(rout, mgl, mgh, color=color, alpha=0.5)
            ax.set_xlabel('Radius [arcmin]', fontsize=fontsize)


        ax.set_xscale('log')
        ax.set_yscale(yscale)
        ax.set_ylabel('$M_{gas} [M_\odot]$', fontsize=fontsize)

        ax.minorticks_on()
        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)

        if xunit == 'both':
            limx=ax.get_xlim()
            ax2 = ax.twiny()
            ax2.set_xlim([limx[0]/ kpcp,limx[1]/ kpcp])
            ax2.set_xscale(xscale)
            ax2.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
            ax2.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
            ax2.set_xlabel('Radius [arcmin]', fontsize=fontsize, labelpad=20)
            for item in (ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(18)


        if outfile is not None:
            plt.savefig(outfile)
            plt.close()
        else:
            plt.show(block=False)


    def Reload(self,samplefile,bkglim=None):
        """
        Reload the samples stored from a previous reconstruction run

        :param samplefile: Path to file containing the saved HMC samples
        :type samplefile: str
        :param bkglim: Limit beyond which it is assumed that the background dominates, i.e. the source is set to 0. This parameter needs to be the same as the value used to run the reconstruction. If bkglim=None (default), the entire radial range is used
        :type bkglim: float
        """
        # Reload the output of a previous PyMC3 run
        samples = np.loadtxt(samplefile)
        pars=np.loadtxt(samplefile+'.par')
        self.nrc=int(pars[0])
        self.nbetas=int(pars[1])
        self.min_beta=pars[2]
        self.nmcmc=int(pars[3])
        self.samplefile=samplefile
        self.samples = samples
        f = open(samplefile+'.par', 'r')
        header = f.readline()
        self.backend=header[2:].split('\n')[0]
        if self.profile is None:
            print('Error: no profile provided')
            return

        prof = self.profile
        sb = prof.profile
        rad = prof.bins
        erad = prof.ebins

        if bkglim is not None:
            self.bkglim = bkglim
        else:
            bkglim = np.max(rad + erad)
            self.bkglim = bkglim

        # Set source region
        sourcereg = np.where(rad < bkglim)

        # Set vector with list of parameters
        pars = list_params(rad, sourcereg, self.nrc, self.nbetas, self.min_beta)
        npt = len(pars)
        # Compute output deconvolved brightness profile
        Ksb = calc_sb_operator(rad, sourcereg, pars)
        allsb = np.dot(Ksb, np.exp(samples.T))
        bfit = np.median(np.exp(samples[:, npt]))
        pmc = np.median(allsb, axis=1)
        pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
        pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)

        self.sb = pmc
        self.sb_lo = pmcl
        self.sb_hi = pmch
        self.bkg = bfit

    def CSB(self,rin=40.,rout=400.,plot=True,outfile=None, figsize=(13, 10), nbins=30, fontsize=40, yscale='linear', **kwargs):
        """
        Compute the surface brightness concentration from a loaded brightness profile reconstruction. The surface brightness concentration is defined as the ratio of fluxes computed within two apertures.

        :param rin: Lower aperture value in kpc (default=40)
        :type rin: float
        :param rout: Higher aperture value in kpc (default=400)
        :type rout: float
        :param plot: Plot the posterior CSB distribution (default=True)
        :type plot: bool
        :param outfile: Output file name to save the figure. If outfile=None, plot only to stdout
        :type outfile: str
        :param figsize: Size of figure. Defaults to (13, 10)
        :type figsize: tuple , optional
        :param nbins: Number of bins on the X axis to construct the posterior distribution. Defaults to 30
        :type nbins: int , optional
        :param fontsize: Font size of the axis labels. Defaults to 40
        :type fontsize: int , optional
        :param yscale: Scale on the Y axis. Defaults to 'linear'
        :type yscale: str , optional
        :param kwargs: Options to be passed to :class:`matplotplib.pyplot.hist`
        :return: Median count rate, 16th and 84th percentiles
        :rtype: float
        """
        if self.samples is None or self.z is None:
            print('Error: no profile reconstruction found')
            return
        prof = self.profile
        cosmo = prof.cosmo
        kpcp = cosmo.kpc_proper_per_arcmin(self.z).value

        sourcereg = np.where(prof.bins < self.bkglim)

        # Set vector with list of parameters
        pars = list_params(prof.bins, sourcereg, self.nrc, self.nbetas, self.min_beta)
        Kin = calc_int_operator(prof.bins[0]/2., rin/kpcp, pars)
        allvin = np.dot(Kin, np.exp(self.samples.T))
        Kout = calc_int_operator(prof.bins[0]/2., rout/kpcp, pars)
        allvout = np.dot(Kout, np.exp(self.samples.T))
        medcsb = np.median((allvin[1, :] - allvin[0, :]) / (allvout[1, :] - allvout[0, :]))
        csblo = np.percentile((allvin[1, :] - allvin[0, :]) / (allvout[1, :] - allvout[0, :]), 50. - 68.3 / 2.)
        csbhi = np.percentile((allvin[1, :] - allvin[0, :]) / (allvout[1, :] - allvout[0, :]), 50. + 68.3 / 2.)
        print('Surface brightness concentration: %g (%g , %g)' % (medcsb, csblo, csbhi))

        if plot:
            plt.clf()
            fig = plt.figure(figsize=figsize)
            ax_size = [0.14, 0.12,
                       0.85, 0.85]
            ax = fig.add_axes(ax_size)
            ax.minorticks_on()
            ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
            ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(22)
            plt.yscale(yscale)
            plt.hist((allvin[1, :] - allvin[0, :]) / (allvout[1, :] - allvout[0, :]), bins=nbins, **kwargs)
            plt.xlabel('$C_{SB}$', fontsize=fontsize)
            plt.ylabel('Frequency', fontsize=fontsize)
            if outfile is not None:
                plt.savefig(outfile)
                plt.close()
            else:
                plt.show(block=False)

        return  medcsb,csblo,csbhi


    def SaveAll(self, outfile=None):
        """
        Save the results of a profile reconstruction into an output FITS file
        First extension is data
        Second extension is density
        Third extension is Mgas
        Fourth extension is PSF

        :param outfile: Output file name
        :type outfile: str
        """
        if outfile is None:
            print('No output file name given')
            return
        else:
            hdul = fits.HDUList([fits.PrimaryHDU()])
            if self.profile is not None:
                cols = []
                cols.append(fits.Column(name='RADIUS', format='E', unit='arcmin', array=self.profile.bins))
                cols.append(fits.Column(name='WIDTH', format='E', unit='arcmin', array=self.profile.ebins))
                cols.append(fits.Column(name='SB', format='E', unit='cts s-1 arcmin-2', array=self.profile.profile))
                cols.append(fits.Column(name='ERR_SB', format='E', unit='cts s-1 arcmin-2', array=self.profile.eprof))
                if self.profile.counts is not None:
                    cols.append(fits.Column(name='COUNTS', format='I', unit='', array=self.profile.counts))
                    cols.append(fits.Column(name='AREA', format='E', unit='arcmin2', array=self.profile.area))
                    cols.append(fits.Column(name='EFFEXP', format='E', unit='s', array=self.profile.effexp))
                    cols.append(fits.Column(name='BKG', format='E', unit='cts s-1 arcmin-2', array=self.profile.bkgprof))
                    cols.append(fits.Column(name='BKGCOUNTS', format='E', unit='', array=self.profile.bkgcounts))
                cols = fits.ColDefs(cols)
                tbhdu = fits.BinTableHDU.from_columns(cols, name='DATA')
                hdr = tbhdu.header
                hdr['X_C'] = self.profile.cx + 1
                hdr['Y_C'] = self.profile.cy + 1
                hdr.comments['X_C'] = 'X coordinate of center value'
                hdr.comments['Y_C'] = 'Y coordinate of center value'
                hdr['RA_C'] = self.profile.cra
                hdr['DEC_C'] = self.profile.cdec
                hdr.comments['RA_C'] = 'Right ascension of center value'
                hdr.comments['DEC_C'] = 'Declination of center value'
                hdr['COMMENT'] = 'Written by pyproffit (Eckert et al. 2011)'
                hdul.append(tbhdu)
            if self.pmc is not None:
                cols = []
                cols.append(fits.Column(name='RADIUS', format='E', array=self.profile.bins))
                cols.append(fits.Column(name='SB_MODEL_TOT', format='E', array=self.pmc))
                cols.append(fits.Column(name='SB_MODEL_TOT_L', format='E', array=self.pmcl))
                cols.append(fits.Column(name='SB_MODEL_TOT_H', format='E', array=self.pmch))
                cols.append(fits.Column(name='SB_MODEL', format='E', array=self.sb))
                cols.append(fits.Column(name='SB_MODEL_L', format='E', array=self.sb_lo))
                cols.append(fits.Column(name='SB_MODEL_H', format='E', array=self.sb_hi))
                if self.rec_counts is not None:
                    cols.append(fits.Column(name='COUNTS_MODEL', format='E', array=self.rec_counts))
                    cols.append(fits.Column(name='COUNTS_MODEL_L', format='E', array=self.rec_counts_lo))
                    cols.append(fits.Column(name='COUNTS_MODEL_H', format='E', array=self.rec_counts_hi))
                cols = fits.ColDefs(cols)
                tbhdu = fits.BinTableHDU.from_columns(cols, name='SB_MODEL')
                hdr = tbhdu.header
                hdr['BACKEND'] = self.backend
                hdr['N_MCMC'] = self.nmcmc
                hdr['BKGLIM'] = self.bkglim
                hdr['SAMPLEFILE'] = self.samplefile
                hdr['N_RC'] = self.nrc
                hdr['N_BETAS'] = self.nbetas
                hdul.append(tbhdu)
            if self.dens is not None:
                cols = []
                cols.append(fits.Column(name='RADIUS', format='E', array=self.rout))
                cols.append(fits.Column(name='DENSITY', format='E', array=self.dens))
                cols.append(fits.Column(name='DENSITY_L', format='E', array=self.dens_lo))
                cols.append(fits.Column(name='DENSITY_H', format='E', array=self.dens_hi))
                if self.mg is not None:
                    cols.append(fits.Column(name='MGAS', format='E', array=self.mg))
                    cols.append(fits.Column(name='MGAS_L', format='E', array=self.mgl))
                    cols.append(fits.Column(name='MGAS_H', format='E', array=self.mgh))
                cols = fits.ColDefs(cols)
                tbhdu = fits.BinTableHDU.from_columns(cols, name='DENSITY')
                hdul.append(tbhdu)
            if self.profile.psfmat is not None:
                psfhdu = fits.ImageHDU(self.profile.psfmat, name='PSF')
                hdul.append(psfhdu)
            hdul.writeto(outfile, overwrite=True)




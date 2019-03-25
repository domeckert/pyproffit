from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import pymc3 as pm
import time
from scipy.special import gamma

# Function to calculate a linear operator transforming parameter vector into predicted model counts

def calc_linear_operator(rad,sourcereg,pars,area,expo,psf):
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

def list_params(rad,sourcereg):
    rfit=rad[sourcereg]
    npfit=len(rfit)
    allrc=np.logspace(np.log10(rfit[0]/nsh),np.log10(rfit[npfit-1]/2.),int(npfit/nsh))
    allbetas=np.linspace(0.4,3.,int(npfit/nsh))
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
    Ktot[:,npars]=1.0
    return Ktot

def list_params_density(rad,sourcereg,z):
    rfit=rad[sourcereg]
    npfit=len(rfit)
    kpcp=cosmo.kpc_proper_per_arcmin(z).value
    allrc=np.logspace(np.log10(rfit[0]/nsh),np.log10(rfit[npfit-1]/2.),int(npfit/nsh))*kpcp
    allbetas=np.linspace(0.5,3.,int(npfit/nsh))
    nrc=len(allrc)
    nbetas=len(allbetas)
    rc=allrc.repeat(nbetas)
    betas=np.tile(allbetas,nrc)
    ptot=np.empty((nrc*nbetas,2))
    ptot[:,0]=betas
    ptot[:,1]=rc
    return ptot

# Linear operator to transform parameters into density

def calc_density_operator(rad,sourcereg,pars,z):
    # Select values in the source region
    kpcp=cosmo.kpc_proper_per_arcmin(z).value
    rfit=rad[sourcereg]*kpcp
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
    nptot=len(rad)
    Ktot=np.zeros((nptot,npars+1))
    Ktot[0:npt,0:npars]=fng.T
    Ktot[:,npars]=0.0
    return Ktot


def Deproject_Sparsity(prof,bkglim=None,nmcmc=1000,samplefile=None):
    sb = prof.profile
    esb = prof.eprof
    rad = prof.bins
    erad = prof.ebins
    counts = prof.counts
    area = prof.area
    exposure = prof.effexp

    # Define maximum radius for source deprojection, assuming we have only background for r>bkglim
    if bkglim is None:
        bkglim=np.max(rad+erad)
        back = sb[len(sb)-1]
    else:
        backreg=np.where(rad>bkglim)
        back=np.mean(sb[backreg])

    # Set source region
    sourcereg = np.where(rad < bkglim)

    # Set vector with list of parameters
    pars = list_params(rad, sourcereg)
    npt = len(pars)

    if prof.psfmat is not None:
        psfmat = np.transpose(prof.psfmat)
    else:
        psfmat = np.eye(prof.nbin)

    # Compute linear combination kernel
    K = calc_linear_operator(rad, sourcereg, pars, area, exposure, psfmat)
    basic_model = pm.Model()
    with basic_model:
        # Priors for unknown model parameters
        coefs = pm.Normal('coefs', mu=np.log(sb[0] / npt), sd=20, shape=npt)
        bkgd = pm.Normal('bkg', mu=np.log(back), sd=0.05, shape=1)
        ctot = pm.math.concatenate((coefs, bkgd), axis=0)

        # Expected value of outcome
        al = pm.math.exp(ctot)
        pred = pm.math.dot(K, al)

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Poisson('counts', mu=pred, observed=counts)

    tinit = time.time()
    print('Running MCMC...')
    with basic_model:
        tm = pm.find_MAP()
        trace = pm.sample(nmcmc, start=tm)
    print('Done.')
    tend = time.time()
    print(' Total computing time is: ', (tend - tinit) / 60., ' minutes')

    # Get chains and save them to file
    sampc = trace.get_values('coefs')
    sampb = trace.get_values('bkg')
    samples = np.append(sampc, sampb, axis=1)
    if samplefile is not  None:
        np.savetxt(samplefile, samples)

    # Compute output deconvolved brightness profile
    Ksb = calc_sb_operator(rad, sourcereg, pars)
    allsb = np.dot(Ksb, np.exp(samples.T))
    bfit = np.median(np.exp(samples[:, npt]))
    pmc = np.median(allsb, axis=1)
    pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)
    return pmc,pmcl,pmch

import numpy as np
from scipy.spatial.distance import  cdist
from scipy.stats import poisson
import  copy

def logbinning(binsize,maxrad):
    """
    Set up a logarithmic binning scheme with a minimum bin size

    :param binsize: Minimum bin size in arcsec
    :type binsize: float
    :param maxrad: Maximum extraction radius in arcmin
    :type maxrad: float
    :return: Bins and bin width
    :rtype: class:`numpy.ndarray`
    """
    nbin=int(maxrad/binsize*60.+0.5)
    bins=np.arange(binsize/60./2.,(nbin+1.5)*binsize/60.,binsize/60.)
    ebins=np.ones(nbin)*binsize/60./2.
    binedge=binsize/60./2
    db=0.0
    i=0
    while db<binsize/60.:
        bb=nbin/np.log10(maxrad/binsize*60.)*(np.log10(binedge)-np.log10(binsize/60.))
        base1=np.log10(binsize/60.)+np.log10(maxrad/binsize*60.)*(bb-1)/nbin
        base2=np.log10(binsize/60.)+np.log10(maxrad/binsize*60.)*bb/nbin
        ebe=np.power(10.,base2)-np.power(10.,base1)
        db=ebe
        binedge=bins[i]
        i=i+1
        if i>nbin:
            break
    binedge=binedge+binsize/60./2.
    thisbin=binedge
    b2=1
    while thisbin<maxrad:
        base1=np.log10(binedge)+np.log10(maxrad/binsize*60.)*(b2-1)/nbin
        base2=np.log10(binedge)+np.log10(maxrad/binsize*60.)*b2/nbin
        thisbin=1./2.*(np.power(10.,base1)+np.power(10.,base2))
        ebe=1./2.*(np.power(10.,base2)-np.power(10.,base1))
        bins[i]=thisbin
        ebins[i]=ebe
        b2=b2+1
        i=i+1
        if i>nbin:
            break
    newnbin=i-1
    bn,ebn=np.empty(newnbin),np.empty(newnbin)
    bn=bins[:newnbin]
    ebn=ebins[:newnbin]
    return bn,ebn


def median_all_cov(dat, bins, ebins, rads, nsim=1000, fitter=None, thin=10):
    """
    Generate Monte Carlo simulations of a Voronoi image and compute the median profile for each of them. The function returns an array of size (nbin, nsim) with nbin the number of bins in the profile and nsim the number of Monte Carlo simulations.

    :param dat:  A :class:`pyproffit.data.Data` object containing the input Voronoi image and error map
    :type dat: class:`pyproffit.data.Data`
    :param bins: Central value of radial binning
    :type bins: class:`numpy.ndarray`
    :param ebins: Half-width of radial binning
    :type ebins: class:`numpy.ndarray`
    :param rads: Array containing the distance to the center, in arcmin, for each pixel
    :type rads: class:`numpy.ndarray`
    :param nsim: Number of Monte Carlo simulations to generate
    :type nsim: int
    :param fitter: A :class:`pyproffit.fitter.Fitter` object containing the result of a fit to the background region, for subtraction of the background to the resulting profile
    :type fitter: class:`pyproffit.fitter.Fitter`
    :param thin: Number of blocks into which the calculation of the bootstrap will be divided. Increasing thin reduces memory usage drastically, at the cost of a modest increase in computation time.
    :type thin: int
    :return:
        - Samples of median profiles
        - Area of each bin
    :rtype: class:`numpy.ndarray`
    """
    if not dat.voronoi:
        print('This routine is meant to work with Voronoi images, aborting')
        return

    img, errmap = dat.img, dat.errmap
    expo = dat.exposure

    rad = bins
    erad = ebins
    nbin = len(rad)

    tol = 0.5e-5
    sort_list = []
    for n in range(nbin):
        if n == 0:
            sort_list.append(np.where(
                np.logical_and(np.logical_and(
                    np.logical_and(rads >= 0, rads < np.round(rad[n] + erad[n], 5) + tol), errmap > 0.0), expo > 0.0)))
        else:
            sort_list.append(np.where(np.logical_and(np.logical_and(np.logical_and(
                rads >= np.round(rad[n] - erad[n], 5) + tol,
                rads < np.round(rad[n] + erad[n], 5) + tol), errmap > 0.0), expo > 0.0)))

    nsimthin = int(nsim / thin)

    shape = (dat.axes[0], dat.axes[1], nsimthin)

    imgmul = np.repeat(img, nsimthin).reshape(shape)
    errmul = np.repeat(errmap, nsimthin).reshape(shape)

    if fitter is not None:
        bkg = np.power(10., fitter.minuit.values['bkg'])
    else:
        bkg = 0.

    all_prof = np.empty((nbin, nsim))

    area = np.empty(nbin)

    for th in range(thin):

        gen_img = imgmul + errmul * np.random.randn(shape[0], shape[1], shape[2])

        nth1 = th * nsimthin

        nth2 = (th + 1) * nsimthin

        for i in range(nbin):
            tid = sort_list[i]

            gen_bin = gen_img[tid]

            all_prof[i, nth1:nth2] = np.median(gen_bin, axis=0) - bkg

            if th == 0:

                area[i] = len(img[tid]) * dat.pixsize ** 2

    return all_prof, area


def medianval(vals,errs,nsim):
    """
    Compute the median value of a sample of values and compute its uncertainty using Monte Carlo simulations

    :param vals: Array containing the set of values in the sample
    :type vals: class:`numpy.ndarray`
    :param errs: Array contatining the error on each value
    :type errs: class:`numpy.ndarray`
    :param nsim: Number of Monte Carlo simulations to be performed
    :type nsim: int
    :return:
            - med (float): Median of sample
            - err (float): Error on median
    """
    allmeds=np.empty(nsim)
    npt=len(vals)
#    erep=np.repeat(errs,nsim).reshape(npt,nsim)
#    vrep=np.repeat(vals,nsim).reshape(npt,nsim)
#    randsamp=erep*np.random.randn(npt,nsim)+vrep
#    allmeds=np.median(randsamp,axis=1)
    for i in range(nsim):
        vrand=errs*np.random.randn(npt)+vals
        allmeds[i]=np.median(vrand)
    return np.mean(allmeds),np.std(allmeds)

def dist_eval(coords2d,index=None,x_c=None,y_c=None,metric='euclidean',selected=None):
    """
    Computed distance of a set of points to a centroid

    :param coords2d: 2D array with shape (N,2), N is the number of points
    :type coords2d: class:`numpy.ndarray`
    :param index: index defining input value for the centroid
    :type index: class:`numpy.ndarray`
    :param x_c: input centroid X axis coordinate
    :type x_c: float
    :param y_c: input centroid Y axis coordinate
    :type y_c: float
    :param metric: metric system (defaults to 'euclidean')
    :type metric: str , optional
    :param selected: index defining a selected subset of points to be used
    :type selected: class:`numpy.ndarray` , optional
    :return: distance
    :rtype: class:`numpy.ndarray`
    """

    if index is None and x_c is  None and  y_c is  None:
        raise RuntimeError("you must provide either index or x_c,y_c")
    
    if index is None and (x_c is None or y_c is None):
        raise RuntimeError("you must provide either index or x_c,y_c")
    
    
    if index is not None:
        x_ref=[coords2d[index]]
    else:
        x_ref=[[x_c,y_c]]
    
    
    if selected is None:
        d= cdist(x_ref,coords2d,metric)[0]
    else:
        d= cdist(x_ref,coords2d[selected],metric)[0]
    
    return d

def get_bary(x,y,x_c=None,y_c=None,weight=None, wdist=False):
    """
    Compute centroid position and ellipse parameters from a set of points using principle component analysis

    :param x: Array of positions on the X axis
    :type x: class:`numpy.ndarray`
    :param y: Array of positions on the Y axis
    :type y: class:`numpy.ndarray`
    :param x_c: Initial guess for the centroid X axis value
    :type x_c: float , optional
    :param y_c: Initial guess for the centroid X axis value
    :type y_c: float , optional
    :param weight: Weights to be applied to the points when computing the average
    :type weight: class:`numpy.ndarray` , optional
    :param wdist: Switch to apply the weights. Defaults to False
    :type wdist: bool
    :return:
            - x_c_w (float): Centroid X coordinate
            - y_c_w (float): Centroid Y coordinate
            - sig_x (float): X axis standard deviation
            - sig_y (float): Y axis standard deviation
            - r_cluster (float): characteristic size of cluster
            - semi_major_angle (float): rotation angle of ellipse
            - pos_err (float): positional error on the centroid
    """
    xy=np.column_stack((x,y))
    _w_tot=None
    if weight is None:
        if y_c is None:
            y_c_ave= y.mean()
        
        if x_c is None:
            x_c_ave= x.mean()
    else:
        if y_c is None:
            y_c_ave=np.average( y,weights=weight)
        
        if x_c is None:
            x_c_ave=np.average( x,weights=weight)

    cdist= dist_eval(xy,x_c=x_c_ave,y_c=y_c_ave)
    cdist[cdist==0]=1
    if wdist == True:
        _w_pos= 1.0/(cdist)
    else:
        _w_pos=np.ones(x.shape,dtype=x.dtype)
    
    if weight is not None:
        _w_tot=_w_pos*weight
    else:
        _w_tot=_w_pos
    y_c_w=np.average( y,weights=_w_tot)
    x_c_w=np.average( x,weights=_w_tot)
    pos_err=1.0/np.sqrt(_w_tot.sum())
    #########################################
    #CHECK wHAT HAPPES if you subtract or not
    #x_c_ave,y_c_ave
    #x_c_w,y_c_w
    ########################################
    cova= np.cov(xy,rowvar=0,aweights=weight)
    eigvec, eigval, V = np.linalg.svd(cova, full_matrices=True)
    ind=np.argsort(eigval)[::-1]
    eigval=eigval[ind]
    eigvec=eigvec[:,ind]

    sd=[np.sqrt(eigval[0]),np.sqrt(eigval[1])]
    
    semi_major_angle=np.rad2deg(np.arctan2(eigvec[0][1],eigvec[0][0]))
    
    sig_x= sd[0]
    sig_y= sd[1]
    
    r_cluster= np.sqrt(sig_x*sig_x+sig_y*sig_y)
    return x_c_w,y_c_w,sig_x,sig_y,r_cluster,semi_major_angle,pos_err

def heaviside(x):
    """
    Heavyside theta function

    :param x: 2D array
    :type x: class:`numpy.ndarray`
    :return: filter
    :rtype: class:`numpy.ndarray`
    """
    nsmooth = 10
    a = x.reshape((nsmooth,nsmooth))
    weights = np.ones([nsmooth,nsmooth])
    a = np.multiply(a,weights)
    a = np.sum(a)
    return a

from scipy.ndimage.filters import gaussian_filter

def bkg_smooth(data,smoothing_scale=25):
    """
    Smooth an input background image using a broad Gaussian

    :param data: 2D array
    :type data: class:`numpy.ndarray`
    :param smoothing_scale: Size of Gaussian smoothing kernel in pixel. Defaults to 25
    :type smoothing_scale: float
    :return: smoothed image
    :rtype: class:`numpy.ndarray`
    """
    bkg = data.bkg
    expo = data.exposure
    gsb = gaussian_filter(bkg,smoothing_scale)
    gsexp = gaussian_filter(expo,smoothing_scale)
    bkgsmoothed = np.nan_to_num(np.divide(gsb,gsexp))*expo
    return  bkgsmoothed

from scipy.ndimage import generic_filter

def clean_bkg(img,bkg):
    """
    Subtract statistically the background from a Poisson image

    :param img: Input image
    :type img: class:`numpy.ndarray`
    :param bkg: Background map
    :type bkg: class:`numpy.ndarray`
    :return: Background subtracted Poisson image
    :rtype: class:`numpy.ndarray`
    """
    id=np.where(img>0.0)
    yp, xp = np.indices(img.shape)
    y=yp[id]
    x=xp[id]
    npt=len(img[id])
    nsm=10
    ons=np.ones((nsm,nsm))
    timg=generic_filter(img,heaviside,footprint=ons,mode='constant')
    tbkg=generic_filter(np.ones(img.shape)*bkg,heaviside,footprint=ons,mode='constant',cval=0,origin=0)
    prob=1.-poisson.cdf(timg[id],tbkg[id])
    vals=np.random.rand(npt)
    remove=np.where(vals<prob)
    img[y[remove],x[remove]]=0
    return img


def model_from_samples(x, model, samples, psfmat=None):
    '''
    Compute the median model and 1-sigma model envelope from a loaded chain, either from HMC or Emcee

    :param x: Vector containing the X axis definition
    :type x: class:`numpy.ndarray`
    :param model: Fitted model
    :type model:  class:`pyproffit.models.Model`
    :param samples: 2-dimensional array containing the parameter samples
    :type samples: class:`numpy.ndarray`
    :return:
            - median (class:`numpy.ndarray`): Median model array
            - model_lo (class:`numpy.ndarray`): Lower 1-sigma envelope array
            - model_hi (class:`numpy.ndarray`): Upper 1-sigma envelope array
    '''
    nsamp = len(samples)

    npt = len(x)

    all_mod = np.empty((nsamp, npt))

    for i in range(nsamp):
        tmod = model(x, *samples[i])

        if psfmat is not None:
            all_mod[i] = np.dot(psfmat, tmod)

    mod_med, mod_lo, mod_hi = np.percentile(all_mod, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=0)

    return mod_med, mod_lo, mod_hi


import copy


def Rebin(prof, minc=None, snr=None):
    '''
    Rebin an existing surface brightness profile to reach a given target number of counts per bin (minc) or a minimum S/N (snr).

    :param prof: A :class:`pyproffit.profextract.Profile` object including the current profile to be rebinned
    :type prof: :class:`pyproffit.profextract.Profile`
    :param minc: Minimum number of counts per bin for the output profile. If None, a minimum S/N is used. Defaults to None.
    :type minc: int
    :param snr: Minimum signal-to-noise ratio of the output profile. If None, a minimum number of counts is used. Defaults to None.
    :type snr: float
    :return: A new :class:`pyproffit.profextract.Profile` object with the rebinned surface brightness profile.
    :rtype: :class:`pyproffit.profextract.Profile`
    '''

    is_minc = True
    if minc is None and snr is None:
        print('No target number of counts or S/N provided, aborting')
        return

    if minc is not None and snr is not None:
        print('Both a target number of counts and a target S/N provided, just pick one')
        return

    if minc is not None:
        print('We will rebin the profile to reach a minimum of %d counts per bin' % (minc))

    if snr is not None:
        print('We will rebin the profile to reach a minimum S/N of %g' % (snr))
        is_minc = False

    prof_new, eprof_new, counts_new, bins_new, ebins_new = np.array([]), np.array([]), np.array([]), np.array(
        []), np.array([])

    back_new, area_new, exp_new, bkgcounts_new = np.array([]), np.array([]), np.array([]), np.array([])

    nbin = prof.nbin

    skybkg, eskybkg = 0., 0.

    if prof.bkgval is not None:
        skybkg = prof.bkgval
        eskybkg = prof.bkgerr

    i = 0
    while i < nbin:
        if is_minc:
            if prof.counts[i] >= minc:
                prof_new = np.append(prof_new, prof.profile[i])
                bins_new = np.append(bins_new, prof.bins[i])
                ebins_new = np.append(ebins_new, prof.ebins[i])
                eprof_new = np.append(eprof_new, prof.eprof[i])
                back_new = np.append(back_new, prof.bkgprof[i])
                area_new = np.append(area_new, prof.area[i])
                exp_new = np.append(exp_new, prof.effexp[i])
                bkgcounts_new = np.append(bkgcounts_new, prof.bkgcounts[i])
                counts_new = np.append(counts_new, prof.counts[i])
                i = i + 1

            else:
                if i < nbin - 1:
                    l = 1
                    tcounts = prof.counts[i]
                    tarea = prof.area[i]
                    texp = prof.effexp[i]
                    tbkgc = prof.bkgcounts[i]
                    bin_low = prof.bins[i] - prof.ebins[i]
                    bin_high = prof.bins[i] + prof.ebins[i]

                    while tcounts < minc and i + l < nbin:
                        tcounts = tcounts + prof.counts[i + l]
                        tarea = tarea + prof.area[i + l]
                        texp = texp + prof.effexp[i + l]
                        tbkgc = tbkgc + prof.bkgcounts[i + l]
                        bin_high = prof.bins[i + l] + prof.ebins[i + l]
                        l = l + 1

                    # if i + l < nbin:
                    bins_new = np.append(bins_new, (bin_low + bin_high) / 2.)
                    ten = texp / (l + 1)
                    exp_new = np.append(exp_new, ten)
                    counts_new = np.append(counts_new, tcounts)
                    area_new = np.append(area_new, tarea)
                    bkgcounts_new = np.append(bkgcounts_new, tbkgc)
                    ebins_new = np.append(ebins_new, (bin_high - bin_low) / 2.)
                    sb_new = (tcounts - tbkgc) / tarea / ten - skybkg
                    prof_new = np.append(prof_new, sb_new)
                    bnew = tbkgc / tarea / ten
                    back_new = np.append(back_new, bnew)
                    epnew = np.sqrt(tcounts / (ten * tarea) ** 2 + eskybkg ** 2)
                    eprof_new = np.append(eprof_new, epnew)

                    i = i + l
                if i == nbin - 1:
                    prof_new = np.append(prof_new, prof.profile[i])
                    bins_new = np.append(bins_new, prof.bins[i])
                    ebins_new = np.append(ebins_new, prof.ebins[i])
                    eprof_new = np.append(eprof_new, prof.eprof[i])
                    back_new = np.append(back_new, prof.bkgprof[i])
                    area_new = np.append(area_new, prof.area[i])
                    exp_new = np.append(exp_new, prof.effexp[i])
                    bkgcounts_new = np.append(bkgcounts_new, prof.bkgcounts[i])
                    counts_new = np.append(counts_new, prof.counts[i])
                    i = i + 1



        else:
            if prof.profile[i] / prof.eprof[i] >= snr:
                prof_new = np.append(prof_new, prof.profile[i])
                bins_new = np.append(bins_new, prof.bins[i])
                ebins_new = np.append(ebins_new, prof.ebins[i])
                eprof_new = np.append(eprof_new, prof.eprof[i])
                back_new = np.append(back_new, prof.bkgprof[i])
                area_new = np.append(area_new, prof.area[i])
                exp_new = np.append(exp_new, prof.effexp[i])
                bkgcounts_new = np.append(bkgcounts_new, prof.bkgcounts[i])
                counts_new = np.append(counts_new, prof.counts[i])
                i = i + 1

            else:
                if i < nbin - 1:
                    l = 1
                    tcounts = prof.counts[i]
                    tarea = prof.area[i]
                    texp = prof.effexp[i]
                    tbkgc = prof.bkgcounts[i]
                    bin_low = prof.bins[i] - prof.ebins[i]
                    bin_high = prof.bins[i] + prof.ebins[i]
                    tsn = prof.profile[i] / prof.eprof[i]

                    while tsn < snr and i + l < nbin:
                        tcounts = tcounts + prof.counts[i + l]
                        tarea = tarea + prof.area[i + l]
                        texp = texp + prof.effexp[i + l]
                        tbkgc = tbkgc + prof.bkgcounts[i + l]
                        bin_high = prof.bins[i + l] + prof.ebins[i + l]
                        l = l + 1
                        ten = texp / l
                        tprof = (tcounts - tbkgc) / tarea / ten - skybkg
                        terr = np.sqrt(tcounts / (tarea * ten) ** 2 + eskybkg ** 2)
                        tsn = tprof / terr

                    # if i + l < nbin:
                    bins_new = np.append(bins_new, (bin_low + bin_high) / 2.)
                    exp_new = np.append(exp_new, ten)
                    counts_new = np.append(counts_new, tcounts)
                    area_new = np.append(area_new, tarea)
                    bkgcounts_new = np.append(bkgcounts_new, tbkgc)
                    ebins_new = np.append(ebins_new, (bin_high - bin_low) / 2.)
                    sb_new = (tcounts - tbkgc) / tarea / ten - skybkg
                    prof_new = np.append(prof_new, sb_new)
                    bnew = tbkgc / tarea / ten
                    back_new = np.append(back_new, bnew)
                    epnew = np.sqrt(tcounts / (ten * tarea) ** 2 + eskybkg ** 2)
                    eprof_new = np.append(eprof_new, epnew)

                    i = i + l

                if i == nbin - 1:
                    prof_new = np.append(prof_new, prof.profile[i])
                    bins_new = np.append(bins_new, prof.bins[i])
                    ebins_new = np.append(ebins_new, prof.ebins[i])
                    eprof_new = np.append(eprof_new, prof.eprof[i])
                    back_new = np.append(back_new, prof.bkgprof[i])
                    area_new = np.append(area_new, prof.area[i])
                    exp_new = np.append(exp_new, prof.effexp[i])
                    bkgcounts_new = np.append(bkgcounts_new, prof.bkgcounts[i])
                    counts_new = np.append(counts_new, prof.counts[i])
                    i = i + 1

    prof_out = copy.copy(prof)
    prof_out.nbin = len(prof_new)
    prof_out.profile = prof_new
    prof_out.bins = bins_new
    prof_out.ebins = ebins_new
    prof_out.eprof = eprof_new
    prof_out.area = area_new
    prof_out.effexp = exp_new
    prof_out.counts = counts_new
    prof_out.bkgcounts = bkgcounts_new
    prof_out.bkgprof = back_new

    return prof_out


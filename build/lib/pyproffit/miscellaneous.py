import numpy as np
from scipy.spatial.distance import  cdist
from scipy.stats import poisson

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
    nsm: int=10
    ons=np.ones((nsm,nsm))
    timg=generic_filter(img,heaviside,footprint=ons,mode='constant')
    tbkg=generic_filter(np.ones(img.shape)*bkg,heaviside,footprint=ons,mode='constant',cval=0,origin=0)
    prob=1.-poisson.cdf(timg[id],tbkg[id])
    vals=np.random.rand(npt)
    remove=np.where(vals<prob)
    img[y[remove],x[remove]]=0
    return img



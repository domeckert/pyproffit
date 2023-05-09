from astropy.io import fits
from scipy.signal import convolve
from .miscellaneous import *
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from .emissivity import *
from astropy.cosmology import FlatLambdaCDM

def plot_multi_profiles(profs, labels=None, outfile=None, axes=None, figsize=(13, 10), fontsize=40, xscale='log', yscale='log', fmt='o', markersize=7):
    """
    Plot multiple surface brightness profiles on a single plot. This feature can be useful e.g. to compare profiles across multiple sectors

    :param profs: List of Profile objects to be plotted
    :type profs: tuple
    :param labels: List of labels for the legend (default=None)
    :type labels: tuple
    :param outfile: If outfile is not None, path to file name to output the plot
    :type outfile: str
    :param axes: List of 4 numbers defining the X and Y axis ranges for the plot. Gives axes=[x1, x2, y1, y2], the X axis will be set between x1 and x2, and the Y axis will be set between y1 and y2.
    :type axes: list , optional
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
    """

    print("Showing %d brightness profiles" % len(profs))
    if labels is None:
        labels = [None] * len(profs)
    else:
        if len(labels) != len(profs):
            print('Error: the number of provided labels does not match the number of input profiles, we will not plot labels')
            labels = [None] * len(profs)


    fig = plt.figure(figsize=figsize)
    ax_size = [0.14, 0.14,
               0.83, 0.83]
    ax = fig.add_axes(ax_size)
    ax.minorticks_on()
    ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
    ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)

    plt.xlabel('Radius [arcmin]', fontsize=fontsize)

    plt.ylabel('SB [cts/s/arcmin$^2$]', fontsize=fontsize)
    plt.xscale(xscale)
    plt.yscale(yscale)
    for i in range(len(profs)):
        prof = profs[i]
        plt.errorbar(prof.bins, prof.profile, xerr=prof.ebins, yerr=prof.eprof, fmt=fmt, color='C%d' % i, elinewidth=2,
                     markersize=markersize, capsize=3, label=labels[i])

    plt.legend(loc=0,fontsize=22)
    if axes is not None:
        plt.axis(axes)

    if outfile is not None:
        plt.savefig(outfile)

    else:
        plt.show(block=False)

class Profile(object):
    """
    pyproffit.Profile class. The class allows the user to extract surface brightness profiles and use them to fit models, extract density profiles, etc.

    :param data: Object of type :class:`pyproffit.data.Data` containing the data to be used
    :type data: class:`pyproffit.Data`
    :param center_choice: Choice of the center of the surface brightness profile. Available options are "centroid", "peak", "custom_ima" and "custom_fk5".
        Args:
            - 'centroid': Compute image centroid and ellipticity. This is done by performing principle component analysis on the count image. If a dmfilth image exists, it will be used instead of the original count image.
            - 'peak': Compute the surface brightness peak, The peak is computed as the maximum of the count image after a light smoothing. If a dmfilth image exists, it will be used instead of the original count image.
            - 'custom_fk5': Use any custom center in FK5 coordinates, provided by the "center_ra" and "center_dec" arguments
            - 'custom_ima': Similar to 'custom_fk5' but with input coordinates in image pixels
    :type center_choice: str
    :param maxrad: The maximum radius (in arcmin) out to which the surface brightness profile will be computed
    :type maxrad: float
    :param binsize: Minumum bin size (in arcsec).
    :type binsize: float
    :param center_ra: User defined center R.A. If center_choice='custom_fk5' this is the right ascension in degrees. If center_choice='custom_ima' this is the image pixel on the X axis. If center_choice='peak' or 'centroid' this is not used.
    :type center_ra: float
    :param center_dec: User defined center declination. If center_choice='custom_fk5' this is the declination in degrees. If center_choice='custom_ima' this is the image pixel on the Y axis. If center_choice='peak' or 'centroid' this is not used.
    :type center_dec: float
    :param binning: Binning type. Available types are 'linear', 'log' or 'custom'. Defaults to 'linear'.
        Args:
            - 'linear': Use a linear radial binning with bin size equal to 'binsize'
            - 'log': Use logarithmic binning, i.e. bin size increasing logarithmically with radius with a minimum bin size given by 'binsize'
            - 'custom': Any user-defined binning in the form of an input numpy array provided through the 'bins' option
    :type binning: str
    :param centroid_region: If center_choice='centroid', this option defines the radius of the region (in arcmin), centered on the center of the image, within which the centroid will be calculated. If centroid_region=None the entire image is used. Defaults to None.
    :type centroid_region: float
    :param bins: in case binning is set to 'custom', a numpy array containing the binning definition. For an input array of length N, the binning will contain N-1 bins with boundaries set as the values of the input array.
    :type bins: class:`numpy.ndarray`
    :param cosmo: An :class:`astropy.cosmology` object containing the definition of the cosmological model. If cosmo=None, Planck 2015 cosmology is used.
    :type cosmo: class:`astropy.cosmology`
    """
    def __init__(self, data=None, center_choice=None, maxrad=None, binsize=None, center_ra=None, center_dec=None,
                 binning='linear', centroid_region=None, bins=None, cosmo=None):
        """
        Constructor of class Profile
        """
        if data is None:
            print('No data given')
            return
        if binning!='custom':
            if maxrad is None:
                print('No maximum radius given, using maximum distance of image from center')
            if binsize is None:
                print('No bin size given')
                return
        else:
            if bins is None:
                print('The custom binning option is selected but no bin definition is provided, use the \'bins=\' option')
                return
        self.data = data

        method = center_choice
        if method == 'custom_ima':
            if center_ra is None or center_dec is None:
                print('Error: please provide X and Y coordinates')
                return
            self.cx = center_ra - 1.
            self.cy = center_dec - 1.
            self.ellangle = None
            self.ellratio = None
            pixcrd = np.array([[self.cx, self.cy]], np.float_)
            world = data.wcs_inp.all_pix2world(pixcrd, 0)
            self.cra = world[0][0]
            self.cdec = world[0][1]
            print('Corresponding FK5 coordinates: ',self.cra,self.cdec)
        elif method == 'custom_fk5':
            if center_ra is None or center_dec is None:
                print('Error: please provide X and Y coordinates')
                return
            self.cra = center_ra
            self.cdec = center_dec
            wc = np.array([[center_ra, center_dec]])
            x = data.wcs_inp.wcs_world2pix(wc, 1)
            self.cx = x[0][0] - 1.
            self.cy = x[0][1] - 1.
            self.ellangle = None
            self.ellratio = None
            print('Corresponding pixels coordinates: ', self.cx + 1, self.cy + 1)
        elif method == 'centroid':
            print('Computing centroid and ellipse parameters using principal component analysis')
            # In case a filled image exists, use it; otherwise use the raw image
            if data.filth is not None:
                img = np.copy(data.filth).astype(int)
            else:
                img = np.copy(data.img).astype(int)
            yp, xp = np.indices(img.shape)
            if centroid_region is not None:
                regrad = centroid_region / data.pixsize
            else:
                regrad = np.max(np.array([data.axes[0], data.axes[1]])/ 2.)
                centroid_region = regrad * data.pixsize
            if center_ra is None or center_dec is None:
                print('No approximate center provided, will search for the centroid within a radius of %g arcmin from the center of the image' % (centroid_region))
                xc_temp, yc_temp = data.axes[1] / 2., data.axes[0] / 2.  # Assume by default the cluster is at the center
            else:
                print('Will search for the centroid within a region of %g arcmin centered on RA=%g, DEC=%g' % (centroid_region,center_ra,center_dec))
                wc = np.array([[center_ra, center_dec]])
                x = data.wcs_inp.wcs_world2pix(wc, 1)
                xc_temp = x[0][0] - 1.
                yc_temp = x[0][1] - 1.
            if data.exposure is None or data.filth is not None:
                region = np.where(np.logical_and(np.hypot(xc_temp - xp, yc_temp - yp) < regrad, img > 0))
                #print('No exposure map given, proceeding with no weights')
                print('Denoising image...')
                if data.exposure is None:
                    bkg = np.mean(img)
                else:
                    nonzero = np.where(data.exposure > 0.0)
                    bkg = np.mean(img[nonzero])
                imgc = clean_bkg(img, bkg)
                x = np.repeat(xp[region], imgc[region])
                y = np.repeat(yp[region], imgc[region])
                print('Running PCA...')
                x_c, y_c, sig_x, sig_y, r_cluster, ellangle, pos_err = get_bary(x, y)
            else:
                region = np.where(np.logical_and(np.logical_and(np.hypot(xc_temp - xp, yc_temp - yp) < regrad, img > 0),data.exposure>0.))
                nonzero = np.where(data.exposure > 0.0)
                print('Denoising image...')
                bkg = np.mean(img[nonzero])
                imgc = clean_bkg(img, bkg)
                x = np.repeat(xp[region], imgc[region])
                y = np.repeat(yp[region], imgc[region])
                weights = np.repeat(1. / data.exposure[region], img[region])
                print('Running PCA...')
                x_c, y_c, sig_x, sig_y, r_cluster, ellangle, pos_err = get_bary(x, y, weight=weights, wdist=True)
            print('Centroid position:', x_c + 1, y_c + 1)
            self.cx = x_c
            self.cy = y_c
            pixcrd = np.array([[self.cx, self.cy]], np.float_)
            world = data.wcs_inp.all_pix2world(pixcrd, 0)
            self.cra = world[0][0]
            self.cdec = world[0][1]
            print('Corresponding FK5 coordinates: ',self.cra,self.cdec)
            print('Ellipse axis ratio and position angle:', sig_x / sig_y, ellangle)
            self.ellangle = ellangle
            self.ellratio = sig_x / sig_y
        elif method == 'peak':

            print('Determining X-ray peak')

            smc = 5  # 5-pixel smoothing to get the peak

            if data.filth is not None:

                gsb = gaussian_filter(data.filth, smc)

            else:

                timg = np.copy(data.img)

                zeroexp = np.where(data.exposure<=0.0)

                timg[zeroexp] = 0.0

                gsb = gaussian_filter(timg, smc)

            maxval = np.max(gsb)

            ismax = np.where(gsb == maxval)

            yp, xp = np.indices(data.img.shape)

            y_peak = yp[ismax]

            x_peak = xp[ismax]

            self.cy = np.mean(y_peak)

            self.cx = np.mean(x_peak)

            self.ellangle = None

            self.ellratio = None

            print('Coordinates of surface-brightness peak:', self.cx + 1, self.cy + 1)

            pixcrd = np.array([[self.cx, self.cy]], np.float_)

            world = data.wcs_inp.all_pix2world(pixcrd, 0)

            self.cra = world[0][0]

            self.cdec = world[0][1]

            print('Corresponding FK5 coordinates: ',self.cra,self.cdec)

        else:
            print('Unknown method', method)
            print('Available methods: "centroid", "peak", "custom_fk5", "custom_ima" ')
            return

        pixsize = data.header['CDELT2'] * 60  # 1 pix = pixsize arcmin
        yima, xima = np.indices(data.img.shape)
        rads = np.hypot(xima - self.cx, yima - self.cy)
        ii = np.where(data.exposure > 0)
        mrad = np.max(rads[ii])*pixsize
        if maxrad is None and binning!='custom':
            maxrad=mrad
            print("Maximum radius is %.4f arcmin"%maxrad)
        elif binning=='custom':
            maxrad=bins[len(bins)-1]
        else:
            if maxrad > mrad:
                maxrad=mrad

        self.maxrad = maxrad
        self.binsize = binsize
        self.psfmat = None
        self.nbin = None
        self.bins = None
        self.ebins = None
        self.profile = None
        self.eprof = None
        self.counts = None
        self.area = None
        self.effexp = None
        self.bkgprof = None
        self.bkgcounts = None
        self.custom = False
        self.ccf = None
        self.lumfact = None
        self.box = False
        self.anglow = 0.
        self.anghigh = 360.
        self.binning = binning
        if data.voronoi:
            self.voronoi = True
        else:
            self.voronoi = False

        if binning=='log':
            self.islogbin = True
        elif binning=='linear':
            self.islogbin = False
        elif binning=='custom':
            self.nbin = len(bins) - 1
            self.bins = (bins + np.roll(bins, -1))[:self.nbin]/2.
            self.ebins = (np.roll(bins, -1) - bins)[:self.nbin]/2.
            self.custom = True
        else:
            print('Unknown binning option '+binning+', reverting to linear')
            self.islogbin = False

        if cosmo is None:
            from astropy.cosmology import Planck15 as cosmo
        self.cosmo = cosmo
        self.scatter = None
        self.escat = None

    def SBprofile(self, ellipse_ratio=1.0, rotation_angle=0.0, angle_low=0., angle_high=360., minexp=0.05, box=False, width=None):
        """
        Extract a surface brightness profile and store the results in the input Profile object

        :param ellipse_ratio: Ratio a/b of major to minor axis in the case of an elliptical annulus definition. Defaults to 1.0, i.e. circular annuli.
        :type ellipse_ratio: float
        :param rotation_angle: Rotation angle of the ellipse or box respective to the R.A. axis. Defaults 0.
        :type rotation_angle: float
        :param angle_low: In case the surface brightness profile should be extracted across a sector instead of the whole azimuth, lower position angle of the sector respective to the R.A. axis. Defaults to 0
        :type angle_low: float
        :param angle_high: In case the surface brightness profile should be extracted across a sector instead of the whole azimuth, upper position angle of the sector respective to the R.A. axis. Defaults to 360
        :type angle_high: float
        :param voronoi: Set whether the input data is a Voronoi binned image (True) or a standard raw count image (False). Defaults to False.
        :type voronoi: bool
        :param box: Define whether the profile should be extract along an annulus or a box. The parameter definition of the box matches the DS9 definition. Defaults to False.
        :type box: bool
        :param width: In case box=True, set the full width of the box (in arcmin)
        :type width: float
        """
        data = self.data
        img = data.img
        voronoi = self.voronoi

        rmsmap = False
        if data.rmsmap is not None:
            rmsmap = True

        if not voronoi:
            exposure = np.copy(data.exposure)

            maxexp = np.max(exposure)

            lowexp = np.where(exposure <= minexp * maxexp)

            exposure[lowexp] = 0.0

        else:
            exposure = data.errmap
        bkg = data.bkg
        pixsize = data.pixsize
        if not self.custom:
            if (self.islogbin):
                self.bins, self.ebins = logbinning(self.binsize, self.maxrad)
                nbin = len(self.bins)
                self.nbin = nbin
            else:
                nbin = int(self.maxrad / self.binsize * 60. + 0.5)
                self.bins = np.arange(self.binsize / 60. / 2., (nbin + 0.5) * self.binsize / 60., self.binsize / 60.)
                self.bins = self.bins[self.bins<self.maxrad]
                self.ebins = np.ones(nbin) * self.binsize / 60. / 2.
                self.nbin = nbin
        else:
            nbin = self.nbin
        profile, eprof, counts, area, effexp, bkgprof, bkgcounts = np.empty(self.nbin), np.empty(self.nbin), np.empty(
                self.nbin), np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin)
        y, x = np.indices(data.axes)
        if rotation_angle is not None:
            self.ellangle = rotation_angle
        else:
            self.ellangle = 0.0

        if ellipse_ratio is not None:
            self.ellratio = ellipse_ratio
        else:
            self.ellratio = 1.0
        tta = rotation_angle - 90.
        if tta < -90. or tta > 270.:
            print('Error: input angle must be between 0 and 360 degrees')
            return
        ellang = tta * np.pi / 180.
        xtil = np.cos(ellang) * (x - self.cx) * pixsize + np.sin(ellang) * (y - self.cy) * pixsize
        ytil = -np.sin(ellang) * (x - self.cx) * pixsize + np.cos(ellang) * (y - self.cy) * pixsize
        rads = ellipse_ratio * np.hypot(xtil, ytil / ellipse_ratio)
        self.anglow = angle_low
        self.anghigh = angle_high
        # Convert degree to radian and rescale to 0-2pi
        if angle_low != 0.0 and angle_high != 360.:
            if angle_low < 0.0:
                anglow = np.deg2rad(np.fmod(angle_low, 360.) + 360.)
            else:
                anglow = np.deg2rad(np.fmod(angle_low, 360.))
            if angle_high < 0.0:
                anghigh = np.deg2rad(np.fmod(angle_high, 360.) + 360.)
            else:
                anghigh = np.deg2rad(np.fmod(angle_high, 360.))
        else:
            anglow = 0.
            anghigh = 2. * np.pi
        # Compute angles and set them between 0 and 2pi
        angles = np.arctan2(y - self.cy , x - self.cx)
        aneg = np.where( angles < 0.)
        angles[aneg] = angles[aneg] + 2. * np.pi
        # Now set angles relative to anglow
        if anghigh<anglow: #We cross the zero
            anghigh = anghigh + 2.*np.pi - anglow
        else:
            anghigh = anghigh - anglow
        if angle_high < angle_low:
            zcross = np.where(angles < np.deg2rad(angle_low))
            angles[zcross] = angles[zcross] + 2.*np.pi - anglow
            zgr = np.where(angles >= np.deg2rad(angle_low))
            angles[zgr] = angles[zgr] - anglow
        else:
            angles = angles - anglow
        tol = 0.5e-5
        for i in range(nbin):
            if not box:
                if i == 0:
                    id = np.where(
                        np.logical_and(np.logical_and(np.logical_and(np.logical_and(rads >= 0, rads < np.round(self.bins[i] + self.ebins[i], 5) + tol),
                                       exposure > 0.0), angles >= 0.), angles <= anghigh))
                else:
                    id = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(rads >= np.round(self.bins[i] - self.ebins[i], 5) + tol,
                                                rads < np.round(self.bins[i] + self.ebins[i], 5) + tol), exposure > 0.0), angles >= 0.), angles <= anghigh))

            else:
                if width is None:
                    print('Error: box width not provided')
                    return
                else:
                    self.box = True
                    if i == 0:
                        id = np.where(
                            np.logical_and(np.logical_and(np.logical_and(ytil + self.maxrad/2. >= 0, ytil + self.maxrad/2. < np.round(self.bins[i] + self.ebins[i], 5) + tol),
                                           exposure > 0.0), np.fabs(xtil) <= width/2.))
                    else:
                        id = np.where(np.logical_and(np.logical_and(np.logical_and(ytil + self.maxrad/2. >= np.round(self.bins[i] - self.ebins[i], 5) + tol,
                                                    ytil + self.maxrad/2. < np.round(self.bins[i] + self.ebins[i], 5) + tol), exposure > 0.0), np.fabs(xtil) <= width/2.))

            #            id=np.where(np.logical_and(np.logical_and(rads>=self.bins[i]-self.ebins[i],rads<self.bins[i]+self.ebins[i]),exposure>0.0)) #left-inclusive
            nv = len(img[id])
            if voronoi or rmsmap:
                if voronoi:
                    errmap = data.errmap
                else:
                    errmap = data.rmsmap
                profile[i] = np.sum(img[id]) / nv
                eprof[i] = np.sqrt(np.sum(errmap[id] ** 2)) / nv
                area[i] = nv * pixsize ** 2
                effexp[i] = 1. # Dummy, but to be consistent with PSF calculation
            else:
                if nv > 0:
                    bkgprof[i] = np.sum(bkg[id] / exposure[id]) / nv / pixsize ** 2
                    profile[i] = np.sum(img[id] / exposure[id]) / nv / pixsize ** 2 - bkgprof[i]
                    eprof[i] = np.sqrt(np.sum(img[id] / exposure[id] ** 2)) / nv / pixsize ** 2
                    counts[i] = np.sum(img[id])
                    bkgcounts[i] = np.sum(bkg[id])
                    area[i] = nv * pixsize ** 2
                    effexp[i] = np.sum(exposure[id]) / nv
                else:
                    bkgprof[i] = 0.
                    profile[i] = 0.
                    eprof[i] = 0.
                    counts[i] = 0.
                    bkgcounts[i] = 0.
                    area[i] = 0.
                    effexp[i] = 0.
        self.profile = profile
        self.eprof = eprof
        self.area = area
        self.effexp = effexp
        self.bkgval = None
        self.bkgerr = None

        if not voronoi:
            self.counts = counts
            self.bkgprof = bkgprof
            self.bkgcounts = bkgcounts


    def MedianSB(self, ellipse_ratio=1.0, rotation_angle=0.0, nsim=1000, outsamples=None, fitter=None, thin=10):
        """
        Extract the median surface brightness profile in circular annuli from a provided Voronoi binned image, following the method outlined in Eckert et al. 2015

        :param ellipse_ratio: Ratio a/b of major to minor axis in the case of an elliptical annulus definition. Defaults to 1.0, i.e. circular annuli.
        :type ellipse_ratio: float
        :param rotation_angle: Rotation angle of the ellipse or box respective to the R.A. axis. Defaults 0.
        :type rotation_angle: float
        :param nsim: Number of Monte Carlo realizations of the Voronoi image to be performed
        :type nsim: int
        :param outsamples: Name of output FITS file to store the bootstrap realizations of the median profile. Defaults to None
        :type outsamples: str
        :param fitter: A :class:`pyproffit.fitter.Fitter` object containing the result of a fit to the background region, for subtraction of the background to the resulting profile
        :type fitter: class:`pyproffit.fitter.Fitter`
        :param thin: Number of blocks into which the calculation of the bootstrap will be divided. Increasing thin reduces memory usage drastically, at the cost of a modest increase in computation time.
        :type thin: int
        """
        data = self.data
        img = data.img
        errmap = data.errmap

        if data.rmsmap is not None:
            errmap = data.rmsmap

        expo = data.exposure
        if errmap is None:
            print('Error: No Voronoi or RMS map has been loaded')
            return
        pixsize = data.pixsize
        if not self.custom:
            if (self.islogbin):
                self.bins, self.ebins = logbinning(self.binsize, self.maxrad)
                nbin = len(self.bins)
                self.nbin = nbin
            else:
                nbin = int(self.maxrad / self.binsize * 60. + 0.5)
                self.bins = np.arange(self.binsize / 60. / 2., (nbin + 0.5) * self.binsize / 60., self.binsize / 60.)
                self.ebins = np.ones(nbin) * self.binsize / 60. / 2.
                self.nbin = nbin
        #profile, eprof, area, effexp = np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin)
        y, x = np.indices(data.axes)
        if rotation_angle is not None:
            self.ellangle = rotation_angle
        else:
            self.ellangle = 0.0

        if ellipse_ratio is not None:
            self.ellratio = ellipse_ratio
        else:
            self.ellratio = 1.0
        tta = rotation_angle - 90.
        if tta < -90. or tta > 270.:
            print('Error: input angle must be between 0 and 360 degrees')
            return
        ellang = tta * np.pi / 180.
        xtil = np.cos(ellang) * (x - self.cx) * pixsize + np.sin(ellang) * (y - self.cy) * pixsize
        ytil = -np.sin(ellang) * (x - self.cx) * pixsize + np.cos(ellang) * (y - self.cy) * pixsize
        rads = ellipse_ratio * np.hypot(xtil, ytil / ellipse_ratio)
        #for i in range(self.nbin):
        #    id = np.where(np.logical_and(
        #        np.logical_and(np.logical_and(rads >= self.bins[i] - self.ebins[i], rads < self.bins[i] + self.ebins[i]),
        #        errmap > 0.0),expo>0.0))  # left-inclusive
        #    profile[i], eprof[i] = medianval(img[id], errmap[id], 1000)
        #    area[i] = len(img[id]) * pixsize ** 2
        #    effexp[i] = 1. # Dummy, but to be consistent with PSF calculation

        all_prof, area = median_all_cov(data, self.bins, self.ebins, rads, nsim=nsim, fitter=fitter, thin=thin)
        profile, eprof = np.median(all_prof, axis=1), np.std(all_prof, axis=1)
        effexp = np.ones(self.nbin) # Dummy, but to be consistent with PSF calculation
        cov = np.cov(all_prof)
        self.profile = profile
        self.eprof = eprof
        self.area = area
        self.effexp = effexp
        self.cov = cov

        if outsamples is not None:
            hdu = fits.PrimaryHDU(all_prof)
            hdu.writeto(outsamples, overwrite=True)

    def AzimuthalScatter(self, nsect=12, model=None):
        '''
        Compute the azimuthal scatter profile around the loaded profile. The azimuthal scatter is defined as the standard deviation of the surface brightness in equispaced sectors with respect to the azimuthal mean,

        .. math::

            \\Sigma_X(r) = \\frac{1}{N} \\sum_{i=1}^N \\frac{(S_i(r) - \\langle S(r) \\rangle)^2}{\\langle S(r) \\rangle^2}

        with N the number of sectors and :math:`\\langle S(r) \\rangle` the loaded mean surface brightness profile.

        :param nsect: Number of sectors from which the azimuthal scatter will be computed. Defaults to nsect=12
        :type nsect: int
        :param model: A :class:`pyproffit.models.Model` object containing the background to be subtracted, in case the scatter is to be computed on background-subtracted profiles. Defaults to None (i.e. no background subtraction).
        :type model: class:`pyproffit.models.Model`
        '''

        if self.profile is None:
            print('Error: please extract a SB profile first')
            return
        dat = self.data
        exposure = dat.exposure
        img = dat.img

        y, x = np.indices(dat.axes)
        # Compute angles and set them between 0 and 2pi
        angles = np.arctan2(y - self.cy, x - self.cx)
        aneg = np.where(angles < 0.)
        angles[aneg] = angles[aneg] + 2. * np.pi

        all_sb = np.empty((self.nbin, nsect))
        all_err = np.empty((self.nbin, nsect))

        ellang = (self.ellangle - 90.) * np.pi / 180.
        xtil = np.cos(ellang) * (x - self.cx) * dat.pixsize + np.sin(ellang) * (y - self.cy) * dat.pixsize
        ytil = -np.sin(ellang) * (x - self.cx) * dat.pixsize + np.cos(ellang) * (y - self.cy) * dat.pixsize
        rads = self.ellratio * np.hypot(xtil, ytil / self.ellratio)

        skybkg = 0.
        skybkg_err = 0.
        if model is not None:
            tp = 0
            for p in range(model.npar):
                if model.parnames[p] == 'bkg':
                    tp = p
            skybkg = np.power(10., model.params[tp])
            skybkg_err = skybkg * np.log(10.) * model.errors[tp]

        tol = 1e-5
        for i in range(self.nbin):
            anglow = 0.
            anghigh = 2. * np.pi / nsect
            for ns in range(nsect):
                if i == 0:
                    id = np.where(
                        np.logical_and(np.logical_and(np.logical_and(np.logical_and(rads >= 0, rads < np.round(self.bins[i] + self.ebins[i], 5) + tol),
                                       exposure > 0.0), angles >= anglow), angles <= anghigh))
                else:
                    id = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(rads >= np.round(self.bins[i] - self.ebins[i], 5) + tol,
                                                rads < np.round(self.bins[i] + self.ebins[i], 5) + tol), exposure > 0.0), angles >= anglow), angles <= anghigh))
            #            id=np.where(np.logical_and(np.logical_and(rads>=self.bins[i]-self.ebins[i],rads<self.bins[i]+self.ebins[i]),exposure>0.0)) #left-inclusive
                nv = len(img[id])
                if dat.voronoi:
                    errmap = dat.errmap
                    all_sb[i, ns] = np.sum(img[id]) / nv
                    all_err[i, ns] = np.sqrt(np.sum(errmap[id] ** 2)) / nv
                else:
                    if nv > 0:
                        bkgprof = np.sum(dat.bkg[id] / exposure[id]) / nv / dat.pixsize ** 2
                        all_sb[i, ns] = np.sum(img[id] / exposure[id]) / nv / dat.pixsize ** 2 - bkgprof - skybkg
                        all_err[i, ns] = np.sqrt(np.sum(img[id] / exposure[id] ** 2)) / nv / dat.pixsize ** 2
                        all_err[i, ns] = np.sqrt(all_err[i, ns]**2 + skybkg_err**2)
                    else:
                        all_sb[i, ns] = 0.
                        all_err[i, ns] = 0.

                anglow = anglow + 2. * np.pi / nsect
                anghigh = anghigh + 2. * np.pi / nsect

        prof_mul = np.repeat(self.profile, nsect).reshape(self.nbin, nsect)
        statscat = 1./nsect * np.sum(all_err**2 / prof_mul**2, axis=1)

        nsim = 100
        emul = np.repeat(all_err, nsim).reshape(self.nbin, nsect, nsim)
        all_sb_mul = np.repeat(all_sb, nsim).reshape(self.nbin, nsect, nsim)

        realiz = all_sb_mul + emul * np.random.randn(self.nbin, nsect, nsim)

        profsect = np.repeat(self.profile, nsect).reshape(self.nbin, nsect)

        profmul = np.repeat(profsect, nsim).reshape(self.nbin, nsect, nsim)

        totscat_mul = 1. / nsect * np.sum((realiz - profmul) ** 2 / profmul ** 2, axis=1)

        statscat_mul = np.repeat(statscat, nsim).reshape(self.nbin, nsim)

        allvars = totscat_mul - statscat_mul
        negscat = np.where(allvars < 0.)
        allvars[negscat] = 0.

        self.scatter = np.median(np.sqrt(allvars), axis=1)

        self.err_scat = np.std(np.sqrt(allvars), axis=1)


    def Save(self, outfile=None, model=None):
        """
        Save the data loaded in the Profile class into an output FITS file.

        :param outfile: Output file name
        :type outfile: str
        :param model: If model is not None, Object of type :class:`pyproffit.models.Model` including the fitted model. Defaults to None
        :type model: class:`pyproffit.models.Model`  , optional
        """
        if outfile is None:
            print('No output file name given')
            return
        else:
            hdul = fits.HDUList([fits.PrimaryHDU()])
            if self.profile is not None:
                cols = []
                cols.append(fits.Column(name='RADIUS', format='E', unit='arcmin', array=self.bins))
                cols.append(fits.Column(name='WIDTH', format='E', unit='arcmin', array=self.ebins))
                cols.append(fits.Column(name='SB', format='E', unit='cts s-1 arcmin-2', array=self.profile))
                cols.append(fits.Column(name='ERR_SB', format='E', unit='cts s-1 arcmin-2', array=self.eprof))
                cols.append(fits.Column(name='AREA', format='E', unit='arcmin2', array=self.area))
                cols.append(fits.Column(name='EFFEXP', format='E', unit='s', array=self.effexp))
                if self.counts is not None:
                    cols.append(fits.Column(name='BKG', format='E', unit='cts s-1 arcmin-2', array=self.bkgprof))
                    cols.append(fits.Column(name='COUNTS', format='I', unit='', array=self.counts))
                    cols.append(fits.Column(name='BKGCOUNTS', format='E', unit='', array=self.bkgcounts))
                if self.scatter is not None:
                    cols.append(fits.Column(name='SCATTER', format='E', array=self.scatter))
                    cols.append(fits.Column(name='ERR_SCATTER', format='E', array=self.err_scat))
                cols = fits.ColDefs(cols)
                tbhdu = fits.BinTableHDU.from_columns(cols, name='DATA')
                hdr = tbhdu.header
                hdr['X_C'] = self.cx + 1
                hdr['Y_C'] = self.cy + 1
                hdr.comments['X_C'] = 'X coordinate of center value'
                hdr.comments['Y_C'] = 'Y coordinate of center value'
                hdr['RA_C'] = self.cra
                hdr['DEC_C'] = self.cdec
                hdr.comments['RA_C'] = 'Right ascension of center value'
                hdr.comments['DEC_C'] = 'Declination of center value'
                hdr['ROT_ANGLE'] = self.ellangle
                hdr['ELL_RATIO'] = self.ellratio
                hdr.comments['ROT_ANGLE'] = 'Ellipse rotation angle'
                hdr.comments['ELL_RATIO'] = 'Ellipse major-to-minor-axis ratio'
                hdr['ANGLOW'] = self.anglow
                hdr['ANGHIGH'] = self.anghigh
                hdr.comments['ANGLOW'] = 'Lower position angle for sector definition'
                hdr.comments['ANGHIGH'] = 'Upper position angle for sector definition'
                hdr['BINSIZE'] = self.binsize
                hdr['MAXRAD'] = self.maxrad
                hdr.comments['BINSIZE'] = 'Profile bin size in arcsec'
                hdr.comments['MAXRAD'] = 'Profile maximum radius in arcmin'
                hdr['BINNING'] = self.binning
                hdr.comments['BINNING'] = 'Binning scheme (linear, log or custom)'
                hdr['IMAGE'] = self.data.imglink
                hdr.comments['IMAGE'] = 'Path to input image file'
                hdr['EXPMAP'] = self.data.explink
                hdr.comments['EXPMAP'] = 'Path to exposure map file'
                hdr['BKGMAP'] = self.data.bkglink
                hdr.comments['BKGMAP'] = 'Path to background map file'
                hdr['RMSMAP'] = self.data.rmsmap
                hdr.comments['RMSMAP'] = 'Path to RMS file'
                hdr['VORONOI'] = self.data.voronoi
                hdr.comments['VORONOI'] = 'Voronoi on/off switch'
                hdr['COMMENT'] = 'Written by pyproffit (Eckert et al. 2020)'
                hdul.append(tbhdu)
            if model is not None:
                cols = []
                npar = len(model.params)
                plist = np.arange(1, npar + 1, 1)
                cols.append(fits.Column(name='PAR', format='1I', array=plist))
                cols.append(fits.Column(name='NAME', format='16A', array=model.parnames))
                cols.append(fits.Column(name='VALUE', format='E', array=model.params))
                cols.append(fits.Column(name='ERROR', format='E', array=model.errors))
                cols = fits.ColDefs(cols)
                tbhdu = fits.BinTableHDU.from_columns(cols, name='MODEL')
                hdul.append(tbhdu)
            if self.psfmat is not None:
                psfhdu = fits.ImageHDU(self.psfmat, name='PSF')
                hdul.append(psfhdu)
            hdul.writeto(outfile, overwrite=True)

    def PSF(self, psffunc=None, psffile=None, psfimage=None, psfpixsize=None, sourcemodel=None, psfmin = 0):
        """
        Function to calculate a PSF convolution matrix given an input PSF image or function.
        To compute the PSF mixing matrix, images of each annuli are convolved with the PSF image using FFT and determine the fraction of photons leaking into neighbouring annuli. FFT-convolved images are then used to determine a mixing matrix. See Eckert et al. 2020 for more details.

        :param psffunc: Function describing the radial shape of the PSF, with the radius in arcmin
        :type psffunc: function
        :param psffile: Path to file containing an image of the PSF. The pixel size must be equal to the pixel size of the image.
        :type psffile: str
        :param psfimage: Array containing an image of the PSF. The pixel size must be equal to the pixel size of the image.
        :type psfimage: class:`numpy.ndarray`
        :param psfpixsize: (currently inactive) Pixel size of the PSF image in arcsec. Currently not implemented.
        :type psfpixsize: float
        :param sourcemodel: Object of type :class:`pyproffit.models.Model` including a surface brightness model to account for surface brightness gradients across the bins. If sourcemodel=None a flat distribution is assumed across each bin. Defaults to None
        :type sourcemodel: class:`pyproffit.models.Model`
        """
        if psffile is None and psfimage is None and psffunc is None:
            print('No PSF image given')
            return
        else:
            data = self.data
            if psffile is not None:
                fpsf = fits.open(psffile)
                psfimage = fpsf[0].data.astype(float)
                if psfpixsize is not None:
                    psfpixsize = float(fpsf[0].header['CDELT2'])
                    if psfpixsize == 0.0:
                        print('Error: no pixel size is provided for the PSF image and the CDELT2 keyword is not set')
                        return
                fpsf.close()
            if psfimage is not None:
                if psfpixsize is None or psfpixsize <= 0.0:
                    print('Error: no pixel size is provided for the PSF image')
                    return
            if self.bins is None:
                print('No radial profile extracted')
                return
            rad = self.bins
            erad = self.ebins
            nbin = self.nbin
            psfout = np.zeros((nbin, nbin))
            exposure = data.exposure
            y, x = np.indices(exposure.shape)
            rads = np.hypot(x - self.cx, y - self.cy) * self.data.pixsize  # arcmin
            kernel = None
            if psffunc is not None:
                 # truncation radius, i.e. we exclude the regions where the PSF signal is less than this value
                kernel = psffunc(rads)
                norm = np.sum(kernel)
                # psfmin = 0.
                frmax = lambda x: psffunc(x) * 2. * np.pi * x / norm - psfmin
                if frmax(exposure.shape[0] / 2) < 0.:
                    rmax = brentq(frmax, 0., exposure.shape[0]) / self.data.pixsize  # pixsize
                    npix = int(rmax)
                else:
                    npix = int(exposure.shape[0] / 2)
                yp, xp = np.indices((2 * npix + 1, 2 * npix + 1))
                rpix = np.sqrt((xp - npix) ** 2 + (yp - npix) ** 2) * self.data.pixsize
                kernel = psffunc(rpix)
                norm = np.sum(kernel)
                kernel = kernel / norm
            if psfimage is not None:
                norm = np.sum(psfimage)
                kernel = psfimage / norm
            if kernel is None:
                print('No kernel provided, bye bye')
                return

            # Sort pixels into radial bins
            tol = 0.5e-5
            sort_list = []
            for n in range(nbin):
                if n == 0:
                    sort_list.append(np.where(
                        np.logical_and(rads >= 0, rads < np.round(rad[n] + erad[n], 5) + tol)))
                else:
                    sort_list.append(np.where(np.logical_and(rads >= np.round(rad[n] - erad[n], 5) + tol,
                                                                    rads < np.round(rad[n] + erad[n], 5) + tol)))
            # Calculate average of PSF image pixel-by-pixel and sort it by radial bins
            for n in range(nbin):
                # print('Working with bin',n+1)
                region = sort_list[n]
                npt = len(x[region])
                imgt = np.zeros(exposure.shape)
                if sourcemodel is None or sourcemodel.params is None:
                    imgt[region] = 1. / npt
                else:
                    imgt[region] = sourcemodel.model(rads[region],*sourcemodel.params)
                    norm = np.sum(imgt[region])
                    imgt[region] = imgt[region]/norm
                # FFT-convolve image with kernel
                blurred = convolve(imgt, kernel, mode='same')
                numnoise = np.where(blurred<1e-15)
                blurred[numnoise]=0.0
                for p in range(nbin):
                    sn = sort_list[p]
                    psfout[n, p] = np.sum(blurred[sn])
            self.psfmat = psfout

    def SaveModelImage(self, outfile, model=None, vignetting=True):
        """
        Compute a model image and output it to a FITS file

        :param model: Object of type :class:`pyproffit.models.Model` including the surface brightness model. If model=None (default), the azimuthally averaged radial profile is interpolated at each point.
        :type model: class:`pyproffit.models.Model`
        :param outfile: Name of output file
        :type outfile: str
        :param vignetting: Choose whether the model will be convolved with the vignetting model (i.e. multiplied by the exposure map) or if the actual surface brightness will be extracted (False). Defaults to True
        :type vignetting: bool
        """
        head = self.data.header
        pixsize = self.data.pixsize
        y, x = np.indices(self.data.axes)
        ellipse_angle = self.ellangle
        ellipse_ratio = self.ellratio
        tta = ellipse_angle - 90.
        if tta < -90. or tta > 270.:
            print('Error: input angle must be between 0 and 360 degrees')
            return
        ellang = tta * np.pi / 180.
        xtil = np.cos(ellang) * (x - self.cx) * pixsize + np.sin(ellang) * (y - self.cy) * pixsize
        ytil = -np.sin(ellang) * (x - self.cx) * pixsize + np.cos(ellang) * (y - self.cy) * pixsize
        rads = ellipse_ratio * np.hypot(xtil, ytil / ellipse_ratio)
        if model is not None:
            outmod = model(rads, *model.params)
        else:
            outmod = np.interp(rads, self.bins, self.profile)
        if vignetting:
            modimg = outmod * pixsize ** 2 * self.data.exposure
        else:
            modimg = outmod * pixsize ** 2
        smoothing_scale=25
        gsb = gaussian_filter(self.data.bkg, smoothing_scale)
        gsexp = gaussian_filter(self.data.exposure, smoothing_scale)
        bkgsmoothed = np.nan_to_num(np.divide(gsb, gsexp)) * self.data.exposure
 #       bkgsmoothed = bkg_smooth(self.data,smoothing_scale)
        hdu = fits.PrimaryHDU(modimg+bkgsmoothed)
        hdu.header = head
        hdu.writeto(outfile, overwrite=True)

    def Plot(self, model=None, samples=None, outfile=None, axes=None, scatter=False, offset=None, figsize=(13, 10),
             fontsize=40., xscale='log', yscale='log', fmt='o', markersize=7, lw=2,
             data_color='black', bkg_color='green', model_color='blue', **kwargs):
        """
        Plot the loaded surface brightness profile

        :param model: If model is not None, plot the provided model of type :class:`pyproffit.models.Model` together with the data. Defaults to None
        :type model: class:`pyproffit.models.Model` , optional
        :param samples: Use parameter samples outputted either by Emcee or HMC optimization to compute the median optimized models and upper and lower envelopes. Defaults to None
        :type samples: class:`numpy.ndarray`
        :param outfile: If outfile is not None, name of output file to save the plot. Defaults to None
        :type outfile: str , optional
        :param axes: List of 4 numbers defining the X and Y axis ranges for the plot. Gives axes=[x1, x2, y1, y2], the X axis will be set between x1 and x2, and the Y axis will be set between y1 and y2.
        :type axes: list , optional
        :param scatter: Set whether the azimuthal scatter profile will be displayed instead of the surface brightness profile. Defaults to False.
        :type scatter: bool
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
        :param model_color: Color of the model curve following matplotlib convention. Defaults to 'blue'
        :type model_color: str , optional
        :param kwargs: Arguments to be passed to :class:`matplotlib.pyplot.errorbar`
        """
        # Plot extracted profile
        if self.profile is None:
            print('Error: No profile extracted')
            return
        plt.clf()
        fig = plt.figure(figsize=figsize)
        gs0 = gridspec.GridSpec(1, 1)
        if model is not None:
            gs0.update(left=0.12, right=0.95, wspace=0.0, top=0.95, bottom=0.35)
        else:
            gs0.update(left=0.12, right=0.95, wspace=0.0, top=0.95, bottom=0.12)
        ax = plt.subplot(gs0[0])
        ax.minorticks_on()
        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
        for item in (ax.get_yticklabels()):
            item.set_fontsize(18)
        for item in (ax.get_xticklabels()):
            if model is not None:
                item.set_fontsize(0)
            else:
                item.set_fontsize(18)

        if not self.box:
            rads = self.bins
        else:
            if offset is None:
                rads = self.bins - self.maxrad/2.

            else: rads = self.bins - offset

        mod_med, mod_lo, mod_hi = None, None, None
        if samples is not None and model is not None:

            mod_med, mod_lo, mod_hi = model_from_samples(rads, model, samples, self.psfmat)

        if model is None:
            plt.xlabel('Radius [arcmin]', fontsize=fontsize)
            if not scatter:
                plt.ylabel('SB [cts/s/arcmin$^2$]', fontsize=fontsize)
            else:
                plt.ylabel('$\Sigma_{X}$', fontsize=fontsize)
        else:
            plt.ylabel('SB [cts/s/arcmin$^2$]', fontsize=fontsize)
        plt.yscale(yscale)
        plt.xscale(xscale)

        if not scatter:
            plt.errorbar(rads, self.profile, xerr=self.ebins, yerr=self.eprof, fmt=fmt, color=data_color, elinewidth=2,
                         markersize=markersize, capsize=0, mec=data_color, label='Brightness', **kwargs)
            if self.data.bkglink is not None:
                plt.plot(rads, self.bkgprof, color=bkg_color, lw=lw, label='Background')
            if model is not None and samples is None:
                tmod = model(rads, *model.params)
                if self.psfmat is not None:
                    tmod = np.dot(self.psfmat, tmod)

                plt.plot(rads, tmod, color=model_color, lw=lw, label='Model')

            elif mod_med is not None:

                plt.plot(rads, mod_med, color=model_color, lw=lw, label='Model')
                plt.fill_between(rads, mod_lo, mod_hi, color=model_color, alpha=0.4)

        else:
            plt.errorbar(rads, self.scatter, xerr=self.ebins, yerr=self.escat, fmt=fmt, color=data_color, elinewidth=2,
                         markersize=markersize, capsize=0, mec=data_color, label='Scatter', **kwargs)
        xmin = rads[0] * 0.9
        xmax = rads[len(self.bins) - 1] * 1.1
        ylim = ax.get_ylim()
        if axes is None:
            ax.axis([xmin,xmax,ylim[0],ylim[1]])
        else:
            ax.axis(axes)
        plt.legend(fontsize=22)
        if model is not None:
            gs1 = gridspec.GridSpec(1, 1)
            gs1.update(left=0.12, right=0.95, wspace=0.0, top=0.35, bottom=0.12)
            ax2 = plt.subplot(gs1[0])

            if mod_med is None:
                chi = (self.profile-tmod)/self.eprof

            else:
                chi = (self.profile - mod_med) / self.eprof

            plt.errorbar(rads, chi, yerr=np.ones(len(rads)), fmt=fmt, color=data_color, elinewidth=2,
                     markersize=markersize, capsize=0, mec=data_color)

            plt.xlabel('Radius [arcmin]', fontsize=fontsize)
            plt.ylabel('$\chi$', fontsize=fontsize)
            plt.xscale(xscale)
            if not self.box:
                xl = np.logspace(np.log10(rads[0] / 2.), np.log10(rads[len(rads) - 1] * 2.), 100)
            #xmin=np.min(self.bins-self.ebins)
            #if xmin<=0:
            #    xmin=1e-2
            else:
                xl = np.linspace(rads[0] , rads[len(rads) - 1] , 100)
            plt.plot(xl, np.zeros(len(xl)), color='blue', linestyle='--')
            ax2.yaxis.set_label_coords(-0.07, 0.5)
            ax2.minorticks_on()
            ax2.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
            ax2.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
            for item in (ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(18)
            ylim = ax2.get_ylim()
            if axes is None:
                ax2.axis([xmin, xmax, ylim[0], ylim[1]])
            else:
                xmin = axes[0]
                xmax = axes[1]
                reg = np.where(np.logical_and(rads>=xmin, rads<=xmax))
                ymin = np.min(chi[reg]) - 1.
                ymax = np.max(chi[reg]) + 1.
                ax2.axis([xmin, xmax, ymin, ymax])
        if outfile is not None:
            plt.savefig(outfile)


    def Backsub(self,fitter):
        """
        Subtract a fitted background value from the loaded surface brightness profile. Each pyproffit model contains a 'bkg' parameter, which will be fitted and loaded in a Fitter object. The routine reads the value of 'bkg', subtracts it from the data, and adds its error in quadrature to the error profile.

        :param fitter: Object of type :class:`pyproffit.fitting.Fitter` containing a model and optimization results
        :type fitter: class:`pyproffit.fitting.Fitter`
        """
        if fitter.minuit is None:
            print('Error: no adequate fit found')
            return
        if self.profile is None:
            print('Error: no surface brightness profile found')
            return
        val=np.power(10.,fitter.minuit.values['bkg'])
        eval=val*np.log(10.)*fitter.minuit.errors['bkg']
        self.profile = self.profile - val
        self.eprof = np.sqrt(self.eprof**2 + eval**2)
        self.bkgval = val
        self.bkgerr = eval


    def Emissivity(self, z=None, nh=None, kt=6.0, rmf=None, Z=0.3, elow=0.5, ehigh=2.0, arf=None, type='cr',
                   lum_elow=0.5, lum_ehigh=2.0, abund='angr'):
        """
        Use XSPEC to compute the conversion from count rate to emissivity using the pyproffit.calc_emissivity routine (see its description)

        :param z: Source redshift
        :type z: float
        :param nh: Source NH in units of 1e22 cm**(-2)
        :type nh: float
        :param kt: Source temperature in keV. Default to 6.0
        :type kt: float
        :param rmf: Path to response file (RMF/RSP)
        :type rmf: str
        :param Z: Metallicity with respect to solar. Defaults to 0.3
        :type Z: float
        :param elow: Low-energy bound of the input image in keV. Defaults to 0.5
        :type elow: float
        :param ehigh: High-energy bound of the input image in keV. Defaults to 2.0
        :type ehigh: float
        :param arf: Path to on-axis ARF in case response file type is RMF)
        :type arf: str , optional
        :param type: Specify whether the exposure map is in units of sec (type='cr') or photon flux (type='photon'). Defaults to 'cr'
        :type type: str
        :param lum_elow: Low energy bound (rest frame) for luminosity calculation. Defaults to 0.5
        :type lum_elow: float
        :param lum_ehigh: High energy bound (rest frame) for luminosity calculation. Defaults to 2.0
        :type lum_ehigh: float
        :return: Conversion factor
        :rtype: float
        """
        self.ccf, self.lumfact = calc_emissivity(cosmo=self.cosmo,
                                        z=z,
                                        nh=nh,
                                        kt=kt,
                                        rmf=rmf,
                                        Z=Z,
                                        elow=elow,
                                        ehigh=ehigh,
                                        arf=arf,
                                        type=type,
                                        lum_elow=lum_elow,
                                        lum_ehigh=lum_ehigh,
                                        abund=abund)

        return self.ccf

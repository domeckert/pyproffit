from astropy.io import fits
from scipy.signal import convolve
from .miscellaneous import *
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from .emissivity import *
from astropy.cosmology import Planck15 as cosmo

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
    """
    def __init__(self, data=None, center_choice=None, maxrad=None, binsize=None, center_ra=None, center_dec=None,
                 binning='linear', centroid_region=None, bins=None):
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

                gsb = gaussian_filter(data.img, smc)

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

    def SBprofile(self, ellipse_ratio=1.0, ellipse_angle=0.0, angle_low=0., angle_high=360., voronoi=False):
        """
        Extract a surface brightness profile and store the results in the input Profile object

        :param ellipse_ratio: Ratio a/b of major to minor axis in the case of an elliptical annulus definition. Defaults to 1.0, i.e. circular annuli.
        :type ellipse_ratio: float
        :param ellipse_angle: Position angle of the elliptical annulus respective to the R.A. axis. Defaults 0.
        :type ellipse_angle: float
        :param angle_low: In case the surface brightness profile should be extracted across a sector instead of the whole azimuth, lower position angle of the sector respective to the R.A. axis. Defaults to 0
        :type angle_low: float
        :param angle_high: In case the surface brightness profile should be extracted across a sector instead of the whole azimuth, upper position angle of the sector respective to the R.A. axis. Defaults to 360
        :type angle_high: float
        :param voronoi: Set whether the input data is a Voronoi binned image (True) or a standard raw count image (False). Defaults to False.
        :type voronoi: bool
        """
        data = self.data
        img = data.img
        exposure = data.exposure
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
                self.ebins = np.ones(nbin) * self.binsize / 60. / 2.
                self.nbin = nbin
        else:
            nbin = self.nbin
        profile, eprof, counts, area, effexp, bkgprof, bkgcounts = np.empty(self.nbin), np.empty(self.nbin), np.empty(
                self.nbin), np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin)
        y, x = np.indices(data.axes)
        if ellipse_angle is not None:
            self.ellangle = ellipse_angle
        else:
            self.ellangle = 0.0

        if ellipse_ratio is not None:
            self.ellratio = ellipse_ratio
        else:
            self.ellratio = 1.0
        tta = ellipse_angle - 90.
        if tta < -90. or tta > 270.:
            print('Error: input angle must be between 0 and 360 degrees')
            return
        ellang = tta * np.pi / 180.
        xtil = np.cos(ellang) * (x - self.cx) * pixsize + np.sin(ellang) * (y - self.cy) * pixsize
        ytil = -np.sin(ellang) * (x - self.cx) * pixsize + np.cos(ellang) * (y - self.cy) * pixsize
        rads = ellipse_ratio * np.hypot(xtil, ytil / ellipse_ratio)
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
            if i == 0:
                id = np.where(
                    np.logical_and(np.logical_and(np.logical_and(np.logical_and(rads >= 0, rads < np.round(self.bins[i] + self.ebins[i], 5) + tol),
                                   exposure > 0.0), angles >= 0.), angles <= anghigh))
            else:
                id = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(rads >= np.round(self.bins[i] - self.ebins[i], 5) + tol,
                                            rads < np.round(self.bins[i] + self.ebins[i], 5) + tol), exposure > 0.0), angles >= 0.), angles <= anghigh))

            #            id=np.where(np.logical_and(np.logical_and(rads>=self.bins[i]-self.ebins[i],rads<self.bins[i]+self.ebins[i]),exposure>0.0)) #left-inclusive
            nv = len(img[id])
            if voronoi:
                errmap = data.errmap
                profile[i] = np.average(img[id], weights=1. / errmap[id] ** 2)
                eprof[i] = np.sqrt(1. / np.sum(1. / errmap[id] ** 2))
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

        if not voronoi:
            self.counts = counts
            self.bkgprof = bkgprof
            self.bkgcounts = bkgcounts

    def MedianSB(self):
        """
        Extract the median surface brightness profile in circular annuli from a provided Voronoi binned image, following the method outlined in Eckert et al. 2015

        """
        data = self.data
        img = data.img
        errmap = data.errmap
        if errmap is None:
            print('Error: No Voronoi map has been loaded')
            return
        pixsize = data.pixsize
        if (self.islogbin):
            self.bins, self.ebins = logbinning(self.binsize, self.maxrad)
            nbin = len(self.bins)
            self.nbin = nbin
        else:
            nbin = int(self.maxrad / self.binsize * 60. + 0.5)
            self.bins = np.arange(self.binsize / 60. / 2., (nbin + 0.5) * self.binsize / 60., self.binsize / 60.)
            self.ebins = np.ones(nbin) * self.binsize / 60. / 2.
            self.nbin = nbin
        profile, eprof, area, effexp = np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin)
        y, x = np.indices(data.axes)
        rads = np.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2) * pixsize
        for i in range(self.nbin):
            id = np.where(np.logical_and(
                np.logical_and(rads >= self.bins[i] - self.ebins[i], rads < self.bins[i] + self.ebins[i]),
                errmap > 0.0))  # left-inclusive
            profile[i], eprof[i] = medianval(img[id], errmap[id], 1000)
            area[i] = len(img[id]) * pixsize ** 2
            effexp[i] = 1. # Dummy, but to be consistent with PSF calculation
        self.profile = profile
        self.eprof = eprof
        self.area = area
        self.effexp = effexp

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
                if self.counts is not None:
                    cols.append(fits.Column(name='COUNTS', format='I', unit='', array=self.counts))
                    cols.append(fits.Column(name='AREA', format='E', unit='arcmin2', array=self.area))
                    cols.append(fits.Column(name='EFFEXP', format='E', unit='s', array=self.effexp))
                    cols.append(fits.Column(name='BKG', format='E', unit='cts s-1 arcmin-2', array=self.bkgprof))
                    cols.append(fits.Column(name='BKGCOUNTS', format='E', unit='', array=self.bkgcounts))
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
                hdr['COMMENT'] = 'Written by pyproffit (Eckert et al. 2011)'
                hdul.append(tbhdu)
            if model is not None:
                cols = []
                npar = len(model.params)
                plist = np.arange(1,npar+1,1)
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

    def PSF(self, psffunc=None, psffile=None, psfimage=None, psfpixsize=None, sourcemodel=None):
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
                psfmin = 1e-7 # truncation radius, i.e. we exclude the regions where the PSF signal is less than this value
                kernel = psffunc(rads)
                norm = np.sum(kernel)
                # psfmin = 0.
                frmax = lambda x: psffunc(x) * 2. * np.pi * x / norm - psfmin
                if frmax(exposure.shape[0] / 2) < 0.:
                    rmax = brentq(frmax, 1., exposure.shape[0]) / self.data.pixsize  # pixsize
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

    def SaveModelImage(self, model, outfile, vignetting=True):
        """
        Compute a model image and output it to a FITS file

        :param model: Object of type :class:`pyproffit.models.Model` including the surface brightness model
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
        outmod = lambda x: model.model(x, *model.params)
        if vignetting:
            modimg = outmod(rads) * pixsize ** 2 * self.data.exposure
        else:
            modimg = outmod(rads) * pixsize ** 2
        smoothing_scale=25
        gsb = gaussian_filter(self.data.bkg, smoothing_scale)
        gsexp = gaussian_filter(self.data.exposure, smoothing_scale)
        bkgsmoothed = np.nan_to_num(np.divide(gsb, gsexp)) * self.data.exposure
 #       bkgsmoothed = bkg_smooth(self.data,smoothing_scale)
        hdu = fits.PrimaryHDU(modimg+bkgsmoothed)
        hdu.header = head
        hdu.writeto(outfile, overwrite=True)

    def Plot(self,model=None,outfile=None,axes=None):
        """
        Plot the loaded surface brightness profile

        :param model: If model is not None, plot the provided model of type :class:`pyproffit.models.Model` together with the data. Defaults to None
        :type model: class:`pyproffit.models.Model` , optional
        :param outfile: If outfile is not None, name of output file to save the plot. Defaults to None
        :type outfile: str , optional
        :param axes: List of 4 numbers defining the X and Y axis ranges for the plot. Gives axes=[x1, x2, y1, y2], the X axis will be set between x1 and x2, and the Y axis will be set between y1 and y2.
        :type axes: list , optional
        """
        # Plot extracted profile
        if self.profile is None:
            print('Error: No profile extracted')
            return
        plt.clf()
        fig = plt.figure(figsize=(13, 10))
        gs0 = gridspec.GridSpec(1, 1)
        if model is not None:
            gs0.update(left=0.12, right=0.95, wspace=0.0, top=0.95, bottom=0.35)
        else:
            gs0.update(left=0.12, right=0.95, wspace=0.0, top=0.95, bottom=0.12)
        ax = plt.subplot(gs0[0])
        ax.minorticks_on()
        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        if model is None:
            plt.xlabel('Radius [arcmin]', fontsize=40)
            plt.ylabel('SB [cts/s/arcmin$^2$]', fontsize=40)
        else:
            plt.ylabel('SB [cts/s/arcmin$^2$]', fontsize=28)
        plt.xscale('log')
        plt.yscale('log')
        plt.errorbar(self.bins, self.profile, xerr=self.ebins, yerr=self.eprof, fmt='o', color='black', elinewidth=2,
                     markersize=7, capsize=0,mec='black', label='Brightness')
        if self.bkgprof is not None:
            plt.plot(self.bins, self.bkgprof, color='green', label='Background')
        if model is not None:
            tmod = model.model(self.bins, *model.params)
            if self.psfmat is not None:
                tmod = np.dot(np.transpose(self.psfmat),tmod*self.area*self.effexp)/self.area/self.effexp
            plt.plot(self.bins, tmod, color='blue', label='Model')
        xmin = self.bins[0] * 0.9
        xmax = self.bins[len(self.bins) - 1] * 1.1
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
            chi = (self.profile-tmod)/self.eprof
            plt.errorbar(self.bins, chi, yerr=np.ones(len(self.bins)), fmt='o', color='black', elinewidth=2,
                     markersize=7, capsize=0,mec='black')
            plt.xlabel('Radius [arcmin]', fontsize=28)
            plt.ylabel('$\chi$', fontsize=28)
            plt.xscale('log')
            #xmin=np.min(self.bins-self.ebins)
            #if xmin<=0:
            #    xmin=1e-2
            xl = np.logspace(np.log10(self.bins[0] / 2.), np.log10(self.bins[len(self.bins) - 1] * 2.), 100)
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
                reg = np.where(np.logical_and(self.bins>=xmin, self.bins<=xmax))
                ymin = np.min(chi[reg]) - 1.
                ymax = np.max(chi[reg]) + 1.
                ax2.axis([xmin, xmax, ymin, ymax])
        if outfile is not None:
            plt.savefig(outfile)
        else:
            plt.show()

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


    def Emissivity(self, z=None, nh=None, kt=6.0, rmf=None, Z=0.3, elow=0.5, ehigh=2.0, arf=None, type='cr'):
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
        :return: Conversion factor
        :rtype: float
        """
        self.ccf = calc_emissivity(cosmo=cosmo,
                                        z=z,
                                        nh=nh,
                                        kt=kt,
                                        rmf=rmf,
                                        Z=Z,
                                        elow=elow,
                                        ehigh=ehigh,
                                        arf=arf,
                                        type=type)

        return self.ccf
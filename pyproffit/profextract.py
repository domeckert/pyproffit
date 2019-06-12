from astropy.io import fits
from scipy.signal import convolve
from .miscellaneous import *
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq


class Profile:
    ################################
    # Class defining brightness profiles
    # method=1: compute centroid and ellipse parameters
    # method=2: find X-ray peak
    # method=3/4: user-given center in image (3) of FK5 (4) coordinates
    ################################
    def __init__(self, data=None, method=None, maxrad=None, binsize=None, centroid_ra=None, centroid_dec=None,
                 islogbin=None):
        if data is None:
            print('No data given')
            return
        if maxrad is None:
            print('No maximum radius given')
            return
        if binsize is None:
            print('No bin size given')
            return
        self.data = data

        if method == 3:
            if centroid_ra is None or centroid_dec is None:
                print('Error: please provide X and Y coordinates')
                return
            self.cra = centroid_ra - 1.
            self.cdec = centroid_dec - 1.
            self.ellangle = None
            self.ellratio = None
        elif method == 4:
            if centroid_ra is None or centroid_dec is None:
                print('Error: please provide X and Y coordinates')
                return
            wc = np.array([[centroid_ra, centroid_dec]])
            x = data.wcs_inp.wcs_world2pix(wc, 1)
            self.cra = x[0][0] - 1.
            self.cdec = x[0][1] - 1.
            self.ellangle = None
            self.ellratio = None
            print('Corresponding pixels coordinates: ', self.cra + 1, self.cdec + 1)
        elif method == 1:
            print('Computing centroid and ellipse parameters using principal component analysis')
            img = data.img.astype(int)
            yp, xp = np.indices(img.shape)
            if data.exposure is None:
                print('No exposure map given, proceeding with no weights')
                print('Denoising image...')
                bkg = np.mean(img)
                imgc = clean_bkg(img, bkg)
                x = np.repeat(xp, imgc.flat)
                y = np.repeat(yp, imgc.flat)
                print('Running PCA...')
                x_c, y_c, sig_x, sig_y, r_cluster, ellangle, pos_err = get_bary(x, y)
            else:
                expo = data.exposure
                nonzero = np.where(expo > 0.0)
                print('Denoising image...')
                bkg = np.mean(img[nonzero])
                imgc = clean_bkg(img, bkg)
                x = np.repeat(xp[nonzero], imgc[nonzero])
                y = np.repeat(yp[nonzero], imgc[nonzero])
                weights = np.repeat(1. / expo[nonzero], img[nonzero])
                print('Running PCA...')
                x_c, y_c, sig_x, sig_y, r_cluster, ellangle, pos_err = get_bary(x, y, weight=weights, wdist=True)
            print('Centroid position:', x_c + 1, y_c + 1)
            print('Ellipse axis ratio and position angle:', sig_x / sig_y, ellangle)
            self.cra = x_c
            self.cdec = y_c
            self.ellangle = ellangle
            self.ellratio = sig_x / sig_y
        elif method == 2:
            print('Determining X-ray peak')
            if data.exposure is None:
                maxval = np.max(data.img)
                ismax = np.where(data.img == maxval)
            else:
                expo = data.exposure
                nonzero = np.where(expo > 0.1 * np.max(expo))
                maxval = np.max(data.img[nonzero] / expo[nonzero])
                imgcorr = data.img * 1.0
                imgcorr[nonzero] = data.img[nonzero] / expo[nonzero]
                ismax = np.where(imgcorr == maxval)
            yp, xp = np.indices(data.img.shape)
            cdec = yp[ismax]
            cra = xp[ismax]
            self.cra = np.mean(cra)
            self.cdec = np.mean(cdec)
            self.ellangle = None
            self.ellratio = None
            print('Coordinates of surface-brightness peak:', self.cra + 1, self.cdec + 1)
        else:
            print('Unknown method', method)
            return
        self.maxrad = maxrad
        self.binsize = binsize
        if islogbin is None:
            self.islogbin = False
        else:
            self.islogbin = islogbin
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

    def SBprofile(self, ellipse_ratio=1.0, ellipse_angle=0.0, angle_low=0., angle_high=360., voronoi=False):
        #######################################
        # Function to extract surface-brightness profiles
        # Currently ellipse is not yet implemented
        ######################################
        data = self.data
        img = data.img
        exposure = data.exposure
        bkg = data.bkg
        pixsize = data.pixsize
        if (self.islogbin):
            self.bins, self.ebins = logbinning(self.binsize, self.maxrad)
            nbin = len(self.bins)
            self.nbin = nbin
        else:
            nbin = int(self.maxrad / self.binsize * 60.)
            self.bins = np.arange(self.binsize / 60. / 2., (nbin + 0.5) * self.binsize / 60., self.binsize / 60.)
            self.ebins = np.ones(nbin) * self.binsize / 60. / 2.
            self.nbin = nbin
        profile, eprof, counts, area, effexp, bkgprof, bkgcounts = np.empty(self.nbin), np.empty(self.nbin), np.empty(
            self.nbin), np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin), np.empty(self.nbin)
        y, x = np.indices(data.axes)
        self.ellangle = ellipse_angle
        self.ellratio = ellipse_ratio
        tta = ellipse_angle - 90.
        if tta < -90. or tta > 270.:
            print('Error: input angle must be between 0 and 360 degrees')
            return
        ellang = tta * np.pi / 180.
        xtil = np.cos(ellang) * (x - self.cra) * pixsize + np.sin(ellang) * (y - self.cdec) * pixsize
        ytil = -np.sin(ellang) * (x - self.cra) * pixsize + np.cos(ellang) * (y - self.cdec) * pixsize
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
        angles = np.arctan2(y - self.cdec , x - self.cra)
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
        if not voronoi:
            self.counts = counts
            self.area = area
            self.effexp = effexp
            self.bkgprof = bkgprof
            self.bkgcounts = bkgcounts

    def MedianSB(self):
        # Function to compute median profile from Voronoi map
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
            nb = int(self.maxrad / self.binsize * 60.)
            self.bins = np.arange(self.binsize / 60. / 2., (nb + 0.5) * self.binsize / 60., self.binsize / 60.)
            self.ebins = np.ones(nb) * self.binsize / 60. / 2.
            self.nbin = nb
        profile, eprof = np.empty(self.nbin), np.empty(self.nbin)
        y, x = np.indices(data.axes)
        rads = np.sqrt((x - self.cra) ** 2 + (y - self.cdec) ** 2) * pixsize
        for i in range(nbin):
            id = np.where(np.logical_and(
                np.logical_and(rads >= self.bins[i] - self.ebins[i], rads < self.bins[i] + self.ebins[i]),
                errmap > 0.0))  # left-inclusive
            profile[i], eprof[i] = medianval(img[id], errmap[id], int(1e3))
        self.profile = profile
        self.eprof = eprof

    def Save(self, outfile=None, model=None):
        #####################################################
        # Function to save profile into FITS file
        # First extension is data
        # Second extension is model
        # Third extension is PSF
        #####################################################
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
                hdr['X_C'] = self.cra + 1
                hdr['Y_C'] = self.cdec + 1
                hdr.comments['X_C'] = 'X coordinate of center value'
                hdr.comments['Y_C'] = 'Y coordinate of center value'
                pixcrd = np.array([[self.cdec, self.cra]], np.float_)
                world = self.data.wcs_inp.all_pix2world(pixcrd, 0)
                hdr['RA_C'] = world[0][0]
                hdr['DEC_C'] = world[0][1]
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

    def PSF(self, psffunc=None, psffile=None, psfimage=None, psfpixsize=None):
        #####################################################
        # Function to calculate a PSF convolution matrix given an input PSF image or function
        # Images of each annuli are convolved with the PSF image using FFT
        # FFT-convolved images are then used to determine PSF mixing
        #####################################################
        if psffile is None and psfimage is None and psffunc is None:
            print('No PSF image given')
            return
        else:
            data = self.data
            if psffile is not None:
                fpsf = fits.open(psffile)
                psfimage = fpsf[0].data.astype(float)
                if psfpixsize is not None:
                    psfpixsize = float(psfimage[0].header['CDELT2'])
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
            rads = np.hypot(x - self.cra, y - self.cdec) * self.data.pixsize  # arcmin
            kernel = None
            if psffunc is not None:
                kernel = psffunc(rads)
                norm = np.sum(kernel)
                psfmin = 1e-5 # truncation radius, i.e. we exclude the regions where the PSF signal is less than this value
                frmax = lambda x: psffunc(x) * 2. * np.pi * x / norm - psfmin
                rmax = brentq(frmax, 1., exposure.shape[0]) / self.data.pixsize # pixsize
                npix = int(rmax)
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
                imgt[region] = 1. / npt
                # FFT-convolve image with kernel
                blurred = convolve(imgt, kernel, mode='same')
                numnoise = np.where(blurred<1e-15)
                blurred[numnoise]=0.0
                for p in range(nbin):
                    sn = sort_list[p]
                    psfout[n, p] = np.sum(blurred[sn])
            self.psfmat = psfout

    def SaveModelImage(self, model, outfile, vignetting=True):
        # Save model image to output FITS file
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
        xtil = np.cos(ellang) * (x - self.cra) * pixsize + np.sin(ellang) * (y - self.cdec) * pixsize
        ytil = -np.sin(ellang) * (x - self.cra) * pixsize + np.cos(ellang) * (y - self.cdec) * pixsize
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
        ax.tick_params(length=20, width=1, which='major', direction='in', right='on', top='on')
        ax.tick_params(length=10, width=1, which='minor', direction='in', right='on', top='on')
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
            ax2.tick_params(length=20, width=1, which='major', direction='in', right='on', top='on')
            ax2.tick_params(length=10, width=1, which='minor', direction='in', right='on', top='on')
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
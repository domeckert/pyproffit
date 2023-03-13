import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata

def get_extnum(fitsfile):
    """
    Find the extension number of the first IMAGE extension in an input FITS file

    :param fitsfile: Input FITS file to be read
    :type fitsfile: str
    :return: extension number
    :rtype: int
    """
    next = 0
    if fitsfile[0].header['NAXIS'] == 2:
        return 0
    else:
        print('Primary HDU is not an image, moving on')
        nhdu = len(fitsfile)
        if nhdu == 1:
            print('Error: No IMAGE extension found in input file')
            return -1
        cont = 1
        next = 1
        while (cont and next < nhdu):
            extension = fitsfile[next].header['XTENSION']
            if extension == 'IMAGE':
                print('IMAGE HDU found in extension ', next)
                cont = 0
            else:
                next = next + 1
        if cont == 1:
            print('Error: No IMAGE extension found in input file')
            return -1
        return next


class Data(object):
    '''Class containing the data to be loaded and used by other pyproffit routines

    :param imglink: Path to input image
    :type imglink: str
    :param explink: Path to exposure map. If none, assume a flat exposure of 1s or an input error map provided through rmsmap
    :type explink: str , optional
    :param bkglink: Path to background map. If none, assume zero background
    :type bkglink: str , optional
    :param voronoi: Define whether the input image is a Voronoi image or not (default=False)
    :type voronoi: bool , optional
    :param rmsmap: Path to error map if the data is not Poisson distributed
    :type rmsmap: str , optional
    '''
    def __init__(self, imglink, explink=None, bkglink=None, voronoi=False, rmsmap=None):
        '''
        Constructor of class Data

        '''
        if imglink is None:
            print('Error: Image file not provided')
            return
        fimg = fits.open(imglink)
        next = get_extnum(fimg)
        self.img = fimg[next].data.astype(float)
        self.imglink = imglink
        self.explink = explink
        self.bkglink = bkglink
        self.voronoi = voronoi
        self.rmsmap = rmsmap
        head = fimg[next].header
        self.header = head
        self.wcs_inp = wcs.WCS(head, relax=False)
        if 'CDELT2' in head:
            self.pixsize = head['CDELT2'] * 60.  # arcmin
        elif 'CD2_2' in head:
            self.pixsize = head['CD2_2'] * 60.  # arcmin
        else:
            print('No pixel size could be found in header, will assume a pixel size of 2.5 arcsec')
            self.pixsize = 2.5 / 60.
        self.axes = self.img.shape
        if voronoi:
            self.errmap = fimg[1].data.astype(float)
        fimg.close()
        if explink is None:
            self.exposure = np.ones(self.axes)
        else:
            fexp = fits.open(explink)
            next = get_extnum(fexp)
            expo = fexp[next].data.astype(float)
            if expo.shape != self.axes:
                print('Error: Image and exposure map sizes do not match')
                return
            self.exposure = expo
            self.defaultexpo = np.copy(expo)
            fexp.close()

        if bkglink is None:
            self.bkg = np.zeros(self.axes)
        else:
            fbkg = fits.open(bkglink)
            next = get_extnum(fbkg)
            bkg = fbkg[next].data.astype(float)
            if bkg.shape != self.axes:
                print('Error: Image and background map sizes do not match')
                return
            self.bkg = bkg
            fbkg.close()
        if rmsmap is not None:
            frms = fits.open(rmsmap)
            next = get_extnum(frms)
            rms = frms[next].data.astype(float)
            if rms.shape != self.axes:
                print('Error: Image and RMS map sizes do not match')
                return
            self.rmsmap = rms
            frms.close()
        else:
            self.rmsmap = None
        self.filth = None

    def region(self, regfile):
        '''
        Filter out regions provided in an input DS9 region file

        :param regfile: Path to region file. Accepted region file formats are fk5 and image.
        :type regfile: str
        '''
        freg = open(regfile)
        lreg = freg.readlines()
        freg.close()
        nsrc = 0
        nreg = len(lreg)
        if self.exposure is None:
            print('No exposure given')
            return
        expo = np.copy(self.exposure)
        y, x = np.indices(self.axes)
        regtype = None
        for i in range(nreg):
            if 'fk5' in lreg[i]:
                regtype = 'fk5'
            elif 'image' in lreg[i]:
                regtype = 'image'
        if regtype is None:
            print('Error: invalid format')
            return
        for i in range(nreg):
            if 'circle' in lreg[i]:
                vals = lreg[i].split('(')[1].split(')')[0]
                if regtype == 'fk5':
                    xsrc = float(vals.split(',')[0])
                    ysrc = float(vals.split(',')[1])
                    rad = vals.split(',')[2]
                    if '"' in rad:
                        rad = float(rad.split('"')[0]) / self.pixsize / 60.
                    elif '\'' in rad:
                        rad = float(rad.split('\'')[0]) / self.pixsize
                    else:
                        rad = float(rad) / self.pixsize * 60.
                    wc = np.array([[xsrc, ysrc]])
                    pixcrd = self.wcs_inp.wcs_world2pix(wc, 1)
                    xsrc = pixcrd[0][0] - 1.
                    ysrc = pixcrd[0][1] - 1.
                else:
                    xsrc = float(vals.split(',')[0])
                    ysrc = float(vals.split(',')[1])
                    rad = float(vals.split(',')[2])

                # Define box around source to spped up calculation
                boxsize = np.round(rad + 0.5).astype(int)
                intcx = np.round(xsrc).astype(int)
                intcy = np.round(ysrc).astype(int)
                xmin = np.max([intcx-boxsize, 0])
                xmax = np.min([intcx+boxsize + 1, self.axes[1]])
                ymin = np.max([intcy-boxsize, 0])
                ymax = np.min([intcy+boxsize + 1, self.axes[0]])
                rbox = np.hypot(x[ymin:ymax,xmin:xmax] - xsrc,y[ymin:ymax,xmin:xmax] - ysrc)
                # Mask source
                src = np.where(rbox < rad)
                expo[ymin:ymax,xmin:xmax][src] = 0.0
                nsrc = nsrc + 1
            elif 'ellipse' in lreg[i]:
                vals = lreg[i].split('(')[1].split(')')[0]
                if regtype == 'fk5':
                    xsrc = float(vals.split(',')[0])
                    ysrc = float(vals.split(',')[1])
                    rad1 = vals.split(',')[2]
                    rad2 = vals.split(',')[3]
                    angle = float(vals.split(',')[4])
                    if '"' in rad1:
                        rad1 = float(rad1.split('"')[0]) / self.pixsize / 60.
                        rad2 = float(rad2.split('"')[0]) / self.pixsize / 60.
                    elif '\'' in rad1:
                        rad1 = float(rad1.split('\'')[0]) / self.pixsize
                        rad2 = float(rad2.split('\'')[0]) / self.pixsize
                    else:
                        rad1 = float(rad1) / self.pixsize * 60.
                        rad2 = float(rad2) / self.pixsize * 60.
                    wc = np.array([[xsrc, ysrc]])
                    pixcrd = self.wcs_inp.wcs_world2pix(wc, 1)
                    xsrc = pixcrd[0][0] - 1.
                    ysrc = pixcrd[0][1] - 1.
                else:
                    xsrc = float(vals.split(',')[0])
                    ysrc = float(vals.split(',')[1])
                    rad1 = float(vals.split(',')[2])
                    rad2 = float(vals.split(',')[3])
                    angle = float(vals.split(',')[2])
                ellang = angle * np.pi / 180. + np.pi / 2.
                aoverb = rad1/rad2
                # Define box around source to spped up calculation
                boxsize = np.round(np.max([rad1, rad2]) + 0.5).astype(int)
                intcx = np.round(xsrc).astype(int)
                intcy = np.round(ysrc).astype(int)
                xmin = np.max([intcx-boxsize, 0])
                xmax = np.min([intcx+boxsize + 1, self.axes[1]])
                ymin = np.max([intcy-boxsize, 0])
                ymax = np.min([intcy+boxsize + 1, self.axes[0]])
                xtil = np.cos(ellang)*(x[ymin:ymax,xmin:xmax]-xsrc) + np.sin(ellang)*(y[ymin:ymax,xmin:xmax]-ysrc)
                ytil = -np.sin(ellang)*(x[ymin:ymax,xmin:xmax]-xsrc) + np.cos(ellang)*(y[ymin:ymax,xmin:xmax]-ysrc)
                rbox = aoverb * np.hypot(xtil, ytil / aoverb)
                # Mask source
                src = np.where(rbox < rad1)
                expo[ymin:ymax,xmin:xmax][src] = 0.0
                nsrc = nsrc + 1

        print('Excluded %d sources' % (nsrc))
        self.exposure = expo

    def reset_exposure(self):
        """
        Revert to the original exposure map and ignore the current region file

        """
        self.exposure = self.defaultexpo

    def dmfilth(self, outfile=None, smoothing_scale=8):
        '''
        Mask the regions provided in a region file and fill in the holes by interpolating the smoothed image into the gaps and generating a Poisson realization

        :param outfile: If outfile is not None, file name to output the dmfilth image into a FITS file
        :type outfile: str , optional
        :param smoothing_scale: Size of smoothing scale (in pixel) to estimate the surface brightness distribution outside of the masked areas
        :type smoothing_scale: int
        '''
        if self.img is None:
            print('No data given')
            return
        # Apply source mask on image
        chimg = np.where(self.exposure == 0.0)
        imgc = np.copy(self.img)
        imgc[chimg] = 0.0

        # High-pass filter
        print('Applying high-pass filter')
        gsb = gaussian_filter(imgc, smoothing_scale)
        gsexp = gaussian_filter(self.exposure, smoothing_scale)
        #img_smoothed = np.nan_to_num(np.divide(gsb, gsexp)) * self.exposure
        img_smoothed = np.nan_to_num(np.divide(gsb, gsexp))
        img_smoothed[chimg] = 0.

        # Interpolate
        print('Interpolating in the masked regions')
        y, x = np.indices(self.axes)
        nonz = np.where(img_smoothed > 0.)
        p_ok = np.array([x[nonz], y[nonz]]).T
        vals = img_smoothed[nonz]
        int_vals = np.nan_to_num(griddata(p_ok, vals, (x, y), method='cubic'))

        # Fill holes
        print('Filling holes')
        area_to_fill = np.where(np.logical_and(int_vals > 0., self.exposure == 0))
        dmfilth = np.copy(self.img)
        dmfilth[area_to_fill] = np.random.poisson(int_vals[area_to_fill] * self.defaultexpo[area_to_fill])

        self.filth = dmfilth

        if outfile is not None:
            hdu = fits.PrimaryHDU(dmfilth)
            hdu.header = self.header
            hdu.writeto(outfile, overwrite=True)
            print('Dmfilth image written to file '+outfile)

import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata

def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis=f[0].header['NAXIS']
    if naxis<2:
        raise RadioError('Can\'t make map from this')
    if naxis==2:
        return fits.PrimaryHDU(header=f[0].header,data=f[0].data)

    w = wcs.WCS(f[0].header)
    wn=wcs.WCS(naxis=2)

    wn.wcs.crpix[0]=w.wcs.crpix[0]
    wn.wcs.crpix[1]=w.wcs.crpix[1]
    wn.wcs.cdelt=w.wcs.cdelt[0:2]
    wn.wcs.crval=w.wcs.crval[0:2]
    wn.wcs.ctype[0]=w.wcs.ctype[0]
    wn.wcs.ctype[1]=w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"]=2
    copy=('EQUINOX','EPOCH','BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r=f[0].header.get(k)
        if r is not None:
            header[k]=r

    slice=[]
    for i in range(naxis,0,-1):
        if i<=2:
            slice.append(np.s_[:],)
        else:
            slice.append(0)

    hdu = fits.PrimaryHDU(header=header,data=f[0].data[tuple(slice)])
    return hdu

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
    :param rmsval: Value of rms to make an rmsmap with constant value
    :type rmsval: float , optional
    :param radio: Define whether the input image is a radio image or not (default=False)
    :type radio: bool , optional
    '''
    def __init__(self, imglink, explink=None, bkglink=None, voronoi=False, rmsmap=None, rmsval=None, radio=False):
        '''
        Constructor of class Data

        '''
        if imglink is None:
            print('Error: Image file not provided')
            return
        if radio:
            fimg = flatten(fits.open(imglink))
            self.img = fimg.data.astype(float)
            head = fimg.header
        else:
            fimg = fits.open(imglink)
            next = get_extnum(fimg)
            self.img = fimg[next].data.astype(float)
            head = fimg[next].header
            fimg.close()
        self.imglink = imglink
        self.explink = explink
        self.bkglink = bkglink
        self.voronoi = voronoi
        self.radio  = radio
        self.rmsmap = rmsmap
        self.header = head
        self.wcs_inp = wcs.WCS(head, relax=False)
        self.axes = self.img.shape

        if 'CDELT2' in head:
            self.pixsize = head['CDELT2'] * 60.  # arcmin
        elif 'CD2_2' in head:
            self.pixsize = head['CD2_2'] * 60.  # arcmin
        else:
            print('No pixel size could be found in header, will assume a pixel size of 2.5 arcsec')
            self.pixsize = 2.5 / 60.

        if radio:
            self.bmaj = head['BMAJ'] * 60.  # arcmin
            self.bmin = head['BMIN'] * 60.  # arcmin
            self.bpa = head['BPA']  # deg
            print('The beam is: {:.2f} arcsec x {:.2f} arcsec, PA {:.2f} deg'.format(self.bmaj*60., self.bmin*60., self.bpa))

            # calc beam area
            beammaj = self.bmaj/(2.0*(2*np.log(2))**0.5) # Convert to sigma
            beammin = self.bmin/(2.0*(2*np.log(2))**0.5) # Convert to sigma
            beamarea_arcmin = 2*np.pi*1.0*beammaj*beammin
            self.beamarea_pix = beamarea_arcmin/(self.pixsize*self.pixsize)
            beamarea_arcsec = self.beamarea_pix*self.pixsize*self.pixsize*3600. #in arcsec^2
            print('The beam area is: {:.2f} arcsec^2'.format(beamarea_arcsec))
            if rmsval:
                self.rms_jy_beam = rmsval
                self.rms_jy_arcmin = rmsval/(self.beamarea_pix*self.pixsize*self.pixsize)
                print('The noise is: {:.2e} Jy/beam = {:.2e} Jy/arcmin^2'.format(self.rms_jy_beam, self.rms_jy_arcmin))
                self.rmsmap = np.full(self.axes, rmsval)
            elif rmsmap:
                frms = flatten(fits.open(rmsmap))
                rms = frms.data.astype(float)
                if rms.shape != self.axes:
                    print('Error: Image and RMS map sizes do not match')
                    return
                self.rmsmap = rms
            else:
                print('No rmsmap nor rmsval provided, will assume an rms of zero (so no error bars will be displayed)')
                self.rmsmap = np.zeros(self.axes)
                self.rms_jy_beam = None
                self.rms_jy_arcmin = None
        else:
            if rmsval:
                self.rmsmap = np.full(self.axes, rmsval)
            elif rmsmap:
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
                self.rms_jy_beam = None
                self.rms_jy_arcmin = None

        if voronoi:
            self.errmap = fimg[1].data.astype(float)

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

        if self.radio:
            print('!! Working with a radio image: forcing smoothing_scale=0. Use dmfilth at your own risk !!')
            smoothing_scale = 0

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
        if self.radio:
            dmfilth[area_to_fill] = int_vals[area_to_fill]
        else:
            dmfilth[area_to_fill] = np.random.poisson(int_vals[area_to_fill] * self.defaultexpo[area_to_fill])

        self.filth = dmfilth

        if outfile is not None:
            hdu = fits.PrimaryHDU(dmfilth)
            hdu.header = self.header
            hdu.writeto(outfile, overwrite=True)
            print('Dmfilth image written to file '+outfile)

import numpy as np
from astropy.io import fits
from astropy import wcs


def get_extnum(fitsfile):
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


class Data:
    def __init__(self, imglink, explink=None, bkglink=None, voronoi=False):
        if imglink is None:
            print('Error: Image file not provided')
            return
        fimg = fits.open(imglink)
        next = get_extnum(fimg)
        self.img = fimg[next].data.astype(float)
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

    def region(self, regfile):
        freg = open(regfile)
        lreg = freg.readlines()
        freg.close()
        nsrc = 0
        nreg = len(lreg)
        if self.exposure is None:
            print('No exposure given')
            return
        expo = self.exposure
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
                src = np.where(np.hypot(x - xsrc, y - ysrc) < rad)
                expo[src] = 0.0
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
                xtil = np.cos(ellang)*(x-xsrc) + np.sin(ellang)*(y-ysrc)
                ytil = -np.sin(ellang)*(x-xsrc) + np.cos(ellang)*(y-ysrc)
                src = np.where(aoverb*np.hypot(xtil, ytil/aoverb) < rad1)
                expo[src] = 0.0
                nsrc = nsrc + 1

        print('Excluded %d sources' % (nsrc))
        self.exposure = expo

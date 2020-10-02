from .data import Data
from .profextract import Profile
from astropy.io import fits
import os
import numpy as np

def Reload(infile, model=None):
    """
    Reload the results of a previous session into new Profile and Data objects. If model=None and a Model structure is present, it will be read and its values will be stored into the input Model structure.

    :param infile: Path to a FITS file saved through the :class:`pyproffit.profextract.Profile.Save` method
    :type infile: str
    :param model: A :class:`pyproffit.models.Model` object including the model to be saved
    :type model: class:`pyproffit.models.Model` , str
    :return:
        - dat: a :class:`pyproffit.data.Data` object including the reloaded data files
        - prof: a :class:`pyproffit.profextract.Profile` object including the reloaded profile
    """
    if not os.path.exists(infile):
        print('Input file not found')
        return

    fin = fits.open(infile)
    nhdu = len(fin)
    prof = None
    dat = None
    for i in range(1, nhdu):
        hduname = fin[i].name
        if hduname == 'DATA':
            print('DATA structure found')
            head = fin[i].header

            dat = Data(imglink=head['IMAGE'], explink=head['EXPMAP'], bkglink=head['BKGMAP'], voronoi=head['VORONOI'], rmsmap=head['RMSMAP'])

            din = fin[i].data
            prof = Profile(dat, binsize=head['BINSIZE'], maxrad=head['MAXRAD'], center_choice='custom_fk5',
                           center_dec=head['DEC_C'], center_ra=head['RA_C'])

            prof.bins = din['RADIUS']
            prof.nbin = len(prof.bins)
            prof.ebins = din['WIDTH']
            prof.profile = din['SB']
            prof.eprof = din['ERR_SB']
            prof.area = din['AREA']
            prof.effexp = din['EFFEXP']
            if not head['VORONOI']:
                prof.counts = din['COUNTS']
                prof.bkgprof = din['BKG']
                prof.bkgcounts = din['BKGCOUNTS']
            prof.binning = head['BINNING']
            prof.anglow = head['ANGLOW']
            prof.anghigh = head['ANGHIGH']
            if prof.binning == 'log':
                prof.islogbin = True
            if prof.binning == 'custom':
                prof.custom = True
            prof.ellangle = head['ROT_ANGLE']
            prof.ellratio = head['ELL_RATIO']

        if hduname == 'PSF':
            print('PSF structure found')
            prof.psfmat = fin[i].data

        if hduname == 'MODEL':
            if model is None:
                print('MODEL structure found but no input model given, skipping')
                continue

            else:
                print('MODEL structure found')
                dmod = fin[i].data
                names = dmod['NAME']
                if len(names) != model.npar:
                    print('Wrong number of parameters in input model, skipping')
                    continue

                if np.any(names != model.parnames):
                    print('The parameters of the input model do not match with the MODEL structure, skipping')
                    continue

                model.params = dmod['VALUE']
                model.errors = dmod['ERROR']

    fin.close()

    return dat, prof

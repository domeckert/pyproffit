import pyproffit
import sys
import numpy as np
import os
from astropy.io import fits

def run():
    # Data handling tests
    if not os.path.exists('b_37.fits.gz') or not os.path.exists('expose_mask_37.fits.gz') or not os.path.exists('back_37.fits.gz'):
        print('Test data not found, please move to the proper directory')
        sys.exit()

    dat=pyproffit.Data('b_37.fits.gz',explink='expose_mask_37.fits.gz',bkglink='back_37.fits.gz')
    if dat.img.shape != (512,512) or dat.exposure.shape != (512,512) or dat.bkg.shape != (512,512):
        print('Data reading step failed.')
        sys.exit()
    else:
        print('File reading test passed')

    expo_orig=np.copy(dat.exposure)
    dat.region('test_region.reg')
    nactive_new=np.sum(dat.exposure)
    nactive_old=np.sum(expo_orig)
    if nactive_old - nactive_new == 5570958.2799835205:
        print('Region file test passed')
    else:
        print('Region file test failed.')
        sys.exit()

    dat.dmfilth()
    if dat.filth is not None:
        print('Dmfilth test passed')
    else:
        print('Dmfilth test failed.')
        sys.exit()


    # Profile extraction tests

    cra=55.68516349389606
    cdec=-53.62244440011402
    prof_cen=pyproffit.Profile(dat,center_choice='centroid',maxrad=50.,binsize=30.,center_ra=55.5399273,center_dec=-53.5573404,centroid_region=20.)
    if np.abs(prof_cen.cra - cra) < 2e-3 and np.abs(prof_cen.cdec - cdec) < 2e-3 :
        print('Centroid calculation test passed')
    else:
        print('Centroid calculation test failed, difference RA=%g , difference DEC=%g'%(np.abs(prof_cen.cra - cra),np.abs(prof_cen.cdec - cdec)))

    peak_ra=55.72147733144434
    peak_dec=-53.628226297404545
    prof_peak=pyproffit.Profile(dat,center_choice='peak',maxrad=50.,binsize=30.)
    if np.abs(prof_peak.cra - peak_ra) < 1e-3 and np.abs(prof_peak.cdec - peak_dec) < 1e-3 :
        print('SB peak test passed')
    else:
        print('SB peak test failed, difference RA=%g , difference DEC=%g'%(np.abs(prof_peak.cra - peak_ra),np.abs(prof_peak.cdec - peak_dec)))

    prof=pyproffit.Profile(dat,center_choice='custom_fk5',maxrad=50.,binsize=30.,center_ra=cra,center_dec=cdec)
    ellrat=1.205958181442464
    ellang=-173.04667140946398
    prof.SBprofile(ellipse_ratio=ellrat,rotation_angle=ellang+180.)

    fsb=fits.open('reference_depr.fits')
    datsb=fsb[1].data
    dsb=np.transpose([datsb['RADIUS'],datsb['WIDTH'],datsb['SB'],datsb['ERR_SB'],datsb['AREA'],datsb['EFFEXP']])
    test_sb=np.transpose([prof.bins,prof.ebins,prof.profile,prof.eprof,prof.area,prof.effexp])
    if np.any(np.abs(np.log(dsb/test_sb))>1e-3):
        print('Surface brightness profile test failed')
        sys.exit()
    else:
        print('Surface brightness profile test passed')

    betaparams=np.array([ 0.74974619,  4.23294907, -1.47141617, -3.68761514])
    mod=pyproffit.Model(pyproffit.BetaModel)
    fitobj=pyproffit.Fitter(mod,prof,beta=0.7,rc=2.,norm=-2,bkg=-4)
    fitobj.Migrad()

    if os.path.exists('test_plot_fit.pdf'):
        os.remove('test_plot_fit.pdf')
    prof.Plot(model=mod,outfile='test_plot_fit.pdf')
    if np.any(np.abs(mod.params - betaparams)>1e-3) or not os.path.exists('test_plot_fit.pdf'):
        print('Fitting test failed')
        sys.exit()
    else:
        print('Fitting test passed, output written to file test_plot_fit.pdf')

    if os.path.exists('test_outmod.fits'):
        os.remove('test_outmod.fits')
    prof.SaveModelImage(outfile='test_outmod.fits', model=mod)
    if os.path.exists('test_outmod.fits'):
        print('Saving model image test passed, test image written to test_outmod.fits')
    else:
        print('Saving model image test failed')
        sys.exit()

    # PSF test
    def fking(x):
        r0=25./60. # arcmin
        alpha=1.5 # 20 arcmin off-axis PANTER data
        return np.power(1.+(x/r0)**2,-alpha)

    prof.PSF(psffunc=fking)

    refpsf=np.loadtxt('reference_psf.dat')
    if np.any(np.logical_or(prof.psfmat/refpsf>1.001,prof.psfmat/refpsf<0.999)):
        print('PSF calculation test failed')
        sys.exit()
    else:
        print('PSF calculation test passed')

    # Saving test
    if os.path.exists('test_sb.fits'):
        os.remove('test_sb.fits')
    prof.Save(outfile='test_sb.fits',model=mod)
    if os.path.exists('test_sb.fits'):
        print('Output profile saving test passed, file written to test_sb.fits')
    else:
        print('Output profile saving test failed')
        sys.exit()

    # Backsub test
    prof2=pyproffit.Profile(dat,center_choice='peak',maxrad=30.,binsize=30.,binning='log')
    prof2.SBprofile()
    prof2.Backsub(fitter=fitobj)
    dop=np.loadtxt('reference_OP.dat')
    test_sb=np.transpose([prof2.bins,prof2.ebins,prof2.profile,prof2.eprof,prof2.area,prof2.effexp])
    #if np.any(np.abs(np.log(dop[:,:6]/test_sb))>1e-3):
    #    print('Background subtraction test failed')
    #    sys.exit()
    #else:
    #    print('Background subtraction test passed')

    # Deprojection tests
    cf_a3158 = 44.489999
    z_a3158 = 0.0597
    depr = pyproffit.Deproject(z=z_a3158,cf=cf_a3158,profile=prof)


    # PyMC3 test

    print('Running PyMC3 reconstruction')
    depr.Multiscale(backend='pymc3',nmcmc=100,bkglim=25.)

    dsb = fsb[2].data
    ref_rec = dsb['SB_MODEL']
    ref_pymc3=np.loadtxt('reference_pymc3.dat')
    isz = np.where(ref_rec == 0)

    check_pymc3_sb=np.nan_to_num(ref_pymc3[:,0]/depr.sb)
    check_pymc3_sb[isz]=1.0
    if np.any(np.abs(np.log(check_pymc3_sb))>1e-2):
        print('Possible issue with PyMC3 reconstruction, check plots')
    else:
        print('PyMC3 reconstruction terminated successfully')

    depr.Density()

    # Density and Mgas test

    if os.path.exists('test_density.pdf'):
        os.remove('test_density.pdf')
    if os.path.exists('test_mgas.pdf'):
        os.remove('test_mgas.pdf')
    depr.PlotDensity(outfile='test_density.pdf')
    depr.PlotMgas(outfile='test_mgas.pdf')

    sourcereg = np.where(prof.bins < depr.bkglim)

    dnh=fsb[3].data
    check_dens=dnh['DENSITY'][sourcereg]/depr.dens
    check_mgas=dnh['MGAS']/depr.mg
    nonz=np.where(depr.sb>0.)
    if np.any(np.log(check_dens[nonz])>1e-2):
        print('Possible issue with gas density calculation, check plots')
    else:
        print('Gas density test passed')

    if np.any(np.log(check_mgas[nonz])>1e-2):
        print('Possible issue with Mgas calculation, check plots')
    else:
        print('Mgas test passed')

    if os.path.exists('test_density.pdf'):
        print('Density profile outputted to test_density.pdf')
    else:
        print('Error outputting density profile')
        sys.exit()

    if os.path.exists('test_mgas.pdf'):
        print('Mgas profile outputted to test_mgas.pdf')
    else:
        print('Error outputting Mgas profile')
        sys.exit()

    if os.path.exists('test_save_all.fits'):
        os.remove('test_save_all.fits')

    depr.SaveAll(outfile='test_save_all.fits')
    if not os.path.exists('test_save_all.fits'):
        print('Error: could not save reconstruction results')
    else:
        print('Reconstruction test passed, results written to file test_save_all.fits')

    # Onion peeling test
    print('Running Onion Peeling test')
    deprop=pyproffit.Deproject(profile=prof2,cf=cf_a3158,z=z_a3158)
    deprop.OnionPeeling()

    if os.path.exists('comp_rec.pdf'):
        os.remove('comp_rec.pdf')

    # Stan test
    try:
        import stan

    except ImportError as e:
        print('pystan package not found, skipping')  # module doesn't exist, deal with it.

        pyproffit.plot_multi_methods(deps=(depr, deprop), profs=(prof, prof2),
                                     labels=('PyMC3', 'OP'), outfile='comp_rec.pdf')
        if os.path.exists('comp_rec.pdf'):
            print('Onion peeling test passed. Comparison of reconstruction methods written in file comp_rec.pdf')
        else:
            print('Onion Peeling test failed')
            sys.exit()

        pass

    else:
        print('Running Stan reconstruction')
        depr_stan = pyproffit.Deproject(z=z_a3158, cf=cf_a3158, profile=prof)

        depr_stan.Multiscale(backend='stan', nmcmc=1000, bkglim=25.)
        depr_stan.Density()

        check_sb = np.nan_to_num(ref_rec / depr_stan.sb)
        check_sb[isz] = 1.
        if np.any(np.abs(np.log(check_sb)) > 1e-2):
            print('Possible issue with Stan reconstruction, check plots')
        else:
            print('Stan reconstruction terminated successfully')

        if os.path.exists('test_rec_stan.pdf'):
            os.remove('test_rec_stan.pdf')
        depr_stan.PlotSB(outfile='test_rec_stan.pdf')
        if os.path.exists('test_rec_stan.pdf'):
            print('Stan reconstruction plotted in test_rec_stan.pdf')
        else:
            print('Error with plot of Stan reconstruction')
            sys.exit()

        pyproffit.plot_multi_methods(deps=(depr, depr_stan, deprop), profs=(prof, prof, prof2),
                                     labels=('PyMC3', 'Stan', 'OP'), outfile='comp_rec.pdf')
        if os.path.exists('comp_rec.pdf'):
            print('Onion peeling test passed. Comparison of reconstruction methods written in file comp_rec.pdf')
        else:
            print('Onion Peeling test failed')
            sys.exit()



if __name__ == '__main__':
    run()

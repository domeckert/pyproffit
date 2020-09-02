import numpy as np
from scipy.ndimage.filters import gaussian_filter
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from scipy.signal import convolve
from scipy.special import gamma
import matplotlib.pyplot as plt
import time

epsilon = 1e-3
Yofn = np.pi
alpha = 11. / 3.  # Kolmogorov slope
cf = np.power(2., alpha / 2.) * gamma(3. - alpha / 2.) / gamma(
    3.)  # correction factor for power spectrum, Arevalo et al. Eq. B1
a3d = 0.1  # Fractional perturbations


# Function to Mexican Hat filter images at a given scale sc
def calc_mexicanhat(sc, img, mask, simmod):
    """
    Filter an input image with a Mexican-hat filter

    :param sc: Mexican Hat scale in pixel
    :type sc: float
    :param img: Image to be smoothed
    :type img: class:`numpy.ndarray`
    :param mask: Mask image
    :type mask: class:`numpy.ndarray`
    :param simmod: Model surface brightness image
    :type simmod: class:`numpy.ndarray`
    :return: Mexican Hat convolved image and SB model
    :rtype: class:`numpy.ndarray`
    """
    # Define Gaussian convolution kernel
    gf = np.zeros(img.shape)
    cx = int(img.shape[0] / 2 + 0.5)
    cy = int(img.shape[1] / 2 + 0.5)
    gf[cx, cy] = 1.
    gfm = gaussian_filter(gf, sc / np.sqrt(1. + epsilon))
    gfp = gaussian_filter(gf, sc * np.sqrt(1. + epsilon))
    # FFT-convolve image with the two scales
    gsigma1 = convolve(img, gfm, mode='same')
    gsigma2 = convolve(img, gfp, mode='same')
    # FFT-convolve mask with the two scales
    gmask1 = convolve(mask, gfm, mode='same')
    gmask2 = convolve(mask, gfp, mode='same')
    # FFT-convolve model with the two scales
    gbeta1 = convolve(simmod, gfm, mode='same')
    gbeta2 = convolve(simmod, gfp, mode='same')
    # Eq. 6 of Arevalo et al. 2012
    fout1 = np.nan_to_num(np.divide(gsigma1, gmask1))
    fout2 = np.nan_to_num(np.divide(gsigma2, gmask2))
    fout = (fout1 - fout2) * mask
    # Same for simulated model
    bout1 = np.nan_to_num(np.divide(gbeta1, gmask1))
    bout2 = np.nan_to_num(np.divide(gbeta2, gmask2))
    bout = (bout1 - bout2) * mask
    return fout, bout


# Bootstrap function to compute the covariance matrix
def do_bootstrap(vals, nsample):
    """
    Compute the covariance matrix of power spectra by bootstrapping the image

    :param vals: Set of values
    :type vals: class:`numpy.ndarray`
    :param nsample: Number of bootstrap samples
    :type nsample: int
    :return: 2D covariance matrix
    :rtype: class:`numpy.ndarray`
    """
    nval = len(vals[0])
    nsc = len(vals)
    vout = np.zeros([nsc, nsample])
    for ns in range(nsample):
        idb = [int(np.floor(np.random.rand() * nval)) for i in range(nval)]
        for k in range(nsc):
            vout[k, ns] = np.mean(vals[k][idb])
    cov = np.cov(vout)
    return cov


def calc_ps(region, img, mod, kr, nreg):
    """
    Function to compute the power at a given scale kr from the Mexican Hat filtered images

    :param region: Index defining the region from which the power spectrum will be extracted
    :type region: class:`numpy.ndarray`
    :param img: Mexican Hat filtered image
    :type img: class:`numpy.ndarray`
    :param mod: Mexican Hat filtered SB model
    :type mod: class:`numpy.ndarray`
    :param kr: Extraction scale
    :type kr: float
    :param nreg: Number of subregions into which the image should be splitted to perform the bootstrap
    :type nreg: int
    :return:
            - ps (float): Power at scale kr
            - psnoise (float): Noise at scale kr
            - vals (class:`numpy.ndarray`): set of values for bootstrap error calculation
    """
    var = np.var(img[region])  # Eq. 17 of Churazov et al. 2012
    vmod = np.var(mod[region])
    ps = (var - vmod) / epsilon ** 2 / Yofn / kr ** 2
    psnoise = vmod / epsilon ** 2 / Yofn / kr ** 2
    # amp=np.sqrt(ps*2.*np.pi*kr**2)
    # Compute power in subregions
    nptot = len(img[region])
    vals = np.empty(nreg)
    for l in range(nreg):
        step = np.double(nptot / nreg)
        imin = int(l * step)
        imax = int((l + 1) * step - 1)
        vals[l] = (np.var(img[region][imin:imax]) - np.var(mod[region][imin:imax])) / (epsilon ** 2 * Yofn * kr ** 2)
    return ps, psnoise, vals


# 3D density for beta model
def betamodel(x, par):
    beta = par[0]
    rc = par[1]
    norm = np.power(10., par[2])
    y = norm * np.power(1. + (x / rc) ** 2, -3. * beta / 2.)
    return y


# 3D density for double beta model
def doublebeta(x, pars):
    beta = pars[0]
    rc1 = pars[1]
    rc2 = pars[2]
    ratio = pars[3]
    norm = np.power(10., pars[4])
    base1 = 1 + x ** 2 / rc1 ** 2
    base2 = 1 + x ** 2 / rc2 ** 2
    xx = norm * (np.power(base1, -3 * beta) + ratio * np.power(base2, -3 * beta))
    return np.sqrt(xx)


# Function to compute numerically the 2D to 3D deprojection factor
def calc_projection_factor(nn, mask, betaparams, scale):
    """
    Compute numerically the 2D to 3D deprojection factor. The routine is simulating 3D fluctuations using the surface brightness model and the region mask, projecting the 3D data along one axis, and computing the ratio of 2D to 3D power as a function of scale.

    Caution: this is a memory intensive computation which will run into memory overflow if the image size is too large or the available memory is insufficient.

    :param nn: Image size in pixel
    :type nn: int
    :param mask: Array defining the mask of active pixels, of size (nn , nn)
    :type mask: class:`numpy.ndarray`
    :param betaparams: Parameters of the beta model or double beta model
    :type betaparams: class:`numpy.ndarray`
    :param scale: Array of scales at which the projection factor should be computed
    :type scale: class:`numpy.ndarray`
    :return: Array containing wave number, 2D power, 2D amplitude, and 3D power
    :rtype: class:`numpy.ndarray`
    """
    tinit = time.time()

    # params defining cutoff scales in ADIM UNITS, i.e. divide by (2PI/L)
    wave_cutoff_largek = nn / 2.  # nn/4. ;small scales cutoff || nn/2 is max possible (Nyquist)
    wave_cutoff_smallk = 2.  # large scales cutoff; AT LEAST k > 0.! if k^-alpha

    # note: these k are 3D, BUT same for 1D because L_3d/dr = L_x/dx

    # ******* ARRAY DECLARATION **********
    tinit = time.time()
    print('Initializing 3D k-space ... ')

    kxq, kyq, kzq = np.indices((nn, nn, nn))
    lfr = np.where(kxq > nn / 2.)
    kxq[lfr] = kxq[lfr] - nn
    mfr = np.where(kyq > nn / 2.)
    kyq[mfr] = kyq[mfr] - nn
    nfr = np.where(kzq > nn / 2.)
    kzq[nfr] = kzq[nfr] - nn
    k3D = np.sqrt(kxq ** 2 + kyq ** 2 + kzq ** 2)

    # ******* PERTURBATIONS in k-space **********

    print('Initializing fluctuations in the k-space ... ')

    cube = nn * nn * nn

    amp = np.sqrt(np.power(k3D, -alpha))
    cut = np.where(np.logical_or(k3D <= wave_cutoff_smallk, k3D > wave_cutoff_largek))
    amp[cut] = 0.0
    gauss1 = np.random.randn(nn ** 3).reshape(nn, nn, nn)
    gauss2 = np.random.randn(nn ** 3).reshape(nn, nn, nn)
    akx = amp * (gauss1 + 1j * gauss2)

    # ******* INVERSE FOURIER TRANSFORM **********
    # DISCRETE (IDFT): x_n = 1/N * SUM (k=0 --> N-1) X_k * exp(2PI*i*(k/N)*n)
    # with n = 0, ..., N-1 ; X_k --> complex numb == represents amplitude
    # and phase of different sinusoidal components of input 'signal' x_n:
    # AMPLITUDE A_k   = |X_k|    = sqrt(Re(X_k)^2 + Im(X_k)^2)
    # PHASE     phi_k = arg(X_k) = atan2(Im(X_k),Re(X_k))
    # FREQUENCY of each sinusoidal == k/N (cycles per sample)

    print('Computing inverse 3D FFT  ... ')

    bx = np.fft.ifftn(akx)  # note: 3D FFT --> k along (nn,nn,nn) ;PARALLEL
   # print('MINIMUM amp delta_k is: ', np.min(np.absolute(akx)))
   # print('MAXIMUM amp delta_k is: ', np.max(np.absolute(akx)))

    # find normalization (rms, mean 0.)
    # note: bx is COMPLEX => take real part
    avbx = np.sqrt(np.sum(np.real(bx) ** 2) / cube)
    # normalize fluctuations such that sigma=1
    fluct = np.real(bx) / avbx

    print('RMS (= 1 sigma) delta_x is: ', avbx)
    print('Minimum (real) delta_x/rms is: ', np.min(fluct))
    print('Maximum (real) delta_x/rms is: ', np.max(fluct))

    hdu = fits.PrimaryHDU(fluct)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto('fluctuations.fits', overwrite=True)
    print('3D fluctuations field saved to fluctuations.fits')

    # Read data
    print('Now computing 2D and 3D power spectra...')

    # ******* PROJECT AND CONVOLVE WITH BETA MODEL **********

    def project(cone, ax):
        image = np.sum(cone, axis=ax)
        return image

    c = [nn / 2., nn / 2., nn / 2.]  # Set center at the middle of the cube

    x, y, z = np.indices((nn, nn, nn))
    rads = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)

    npar = len(betaparams)
    if npar == 4:
        rho_unpert = betamodel(rads, betaparams)
    else:
        rho_unpert = doublebeta(rads, betaparams)

    em = np.power(rho_unpert * (1. + a3d * fluct), 2.)
    em3d = em / np.power(rho_unpert, 2.)
    mod = project(np.power(rho_unpert, 2.), 0)
    imgo = project(em, 0)

    img = np.nan_to_num(np.divide(imgo, mod))

    nsc = len(scale)
    imgs = []
    flucts = []
    # Convolve the image with the given scales
    print('Convolving images with Mexican Hat filters')
    for i in range(nsc):
        sc = scale[i]
        print('Convolving with scale ', sc)
        gf = np.zeros((nn,nn))
        center = int(nn / 2)
        gf[center, center] = 1.
        gfm = gaussian_filter(gf, sc / np.sqrt(1. + epsilon))
        gfp = gaussian_filter(gf, sc * np.sqrt(1. + epsilon))
        # Convolve image with the two scales
        gsigma1 = convolve(img, gfm, mode='same')
        gsigma2 = convolve(img, gfp, mode='same')

        # Convolve 3D fluctuation field
        gf = np.zeros((nn,nn,nn))
        gf[center, center, center] = 1.
        gfm = gaussian_filter(gf, sc / np.sqrt(1. + epsilon))
        gfp = gaussian_filter(gf, sc * np.sqrt(1. + epsilon))
        gfluct1 = convolve(em3d, gfm, mode='same')
        gfluct2 = convolve(em3d, gfp, mode='same')

        # Eq. 6 of Arevalo et al. 2012
        fout = gsigma1 - gsigma2
        imgs.append(fout)

        ffluct = gfluct1 - gfluct2
        flucts.append(ffluct)

    # Compute power spectrum in nreg independent subregions
    print('Computing 2D and 3D power spectra...')
    # size=np.double(nlin)/2./np.pi
    kr = 1. / np.sqrt(2. * np.pi ** 2) * np.divide(1., scale)
    Yofn = np.pi
    Yofn3d = 15. * np.power(np.pi, 3. / 2.) / 8. / np.sqrt(2.)
    # Applying mask to 2D images to correct for missing area due to gaps and point sources
    nonzero = np.where(mask>0.0)
    ps, ps3d, amp = np.empty(nsc), np.empty(nsc), np.empty(nsc)
    for i in range(nsc):
        tkr = kr[i]
        timg = imgs[i][nonzero]
        var = np.var(timg)  # Eq. 17 of Churazov et al. 2012
        tp = var / epsilon ** 2 / Yofn / tkr ** 2  # no noise
        ps[i] = tp
        amp[i] = np.sqrt(tp * 2. * np.pi * tkr ** 2)
        # 3D
        t3d = flucts[i]
        v3d = np.var(t3d) / epsilon ** 2 / Yofn3d / tkr ** 3
        ps3d[i] = v3d

    # Save data into file
    pout = np.transpose([kr, ps, amp, ps3d])[::-1]
    np.savetxt('conv2d3d.txt', pout)
    print('Results written in file conv2d3d.txt')
    tend = time.time()
    print(' Total computing time is: ', (tend - tinit) / 60., ' minutes')
    return pout


class PowerSpectrum(object):
    """
    Class to perform fluctuation power spectrum analysis from Poisson count images. This is the code used in Eckert et al. 2017.

    :param data: Object of type :class:`pyproffit.data.Data` including the image, exposure map, background map, and region definition
    :type data: class:`pyproffit.data.Data`
    :param profile: Object of type :class:`pyproffit.profextract.Profile` including the extracted surface brightness profile
    :type profile: class:`pyproffit.profextract.Profile`
    """
    def __init__(self, data, profile):
        """
        Constructor for class PowerSpectrum

        """
        self.data = data
        self.profile = profile

    def MexicanHat(self, modimg_file, z, region_size=1., factshift=1.5):
        """
        Convolve the input image and model image with a set of Mexican Hat filters at various scales. The convolved images are automatically stored into FITS images called conv_scale_xx.fits and conv_beta_xx.fits, with xx the scale in kpc.

        :param modimg_file: Path to a FITS file including the model image, typically produced with :meth:`pyproffit.profextract.Profile.SaveModelImage`
        :type modimg_file: str
        :param z: Source redshift
        :type z: float
        :param region_size: Size of the region of interest in Mpc. Defaults to 1.0
        :type region_size: float
        :param factshift: Size of the border around the region, i.e. a region of size factshift * region_size is used for the computation. Defaults to 1.5
        :type factshift: float
        """
        imgo = self.data.img
        expo = self.data.exposure
        bkg = self.data.bkg
        pixsize = self.data.pixsize
        # Read model image
        fmod = fits.open(modimg_file)
        modimg = fmod[0].data.astype(float)
        # Define the mask
        nonz = np.where(expo > 0.0)
        masko = np.copy(expo)
        masko[nonz] = 1.0
        imgt = np.copy(imgo)
        noexp = np.where(expo == 0.0)
        imgt[noexp] = 0.0
        # Set the region of interest
        x_c = self.profile.cx  # Center coordinates
        y_c = self.profile.cy
        kpcp = cosmo.kpc_proper_per_arcmin(z).value
        Mpcpix = 1000. / kpcp / pixsize  # 1 Mpc in pixel
        regsizepix = region_size * Mpcpix
        self.regsize = regsizepix
        minx = int(np.round(x_c - factshift * regsizepix))
        maxx = int(np.round(x_c + factshift * regsizepix + 1))
        miny = int(np.round(y_c - factshift * regsizepix))
        maxy = int(np.round(y_c + factshift * regsizepix + 1))
        if minx < 0: minx = 0
        if miny < 0: miny = 0
        if maxx > self.data.axes[1]: maxx = self.data.axes[1]
        if maxy > self.data.axes[0]: maxy = self.data.axes[0]
        img = np.nan_to_num(np.divide(imgt[miny:maxy, minx:maxx], modimg[miny:maxy, minx:maxx]))
        mask = masko[miny:maxy, minx:maxx]
        self.size = img.shape
        self.mask = mask
        fmod[0].data = mask
        fmod.writeto('mask.fits', overwrite=True)
        # Simulate perfect model with Poisson noise
        randmod = np.random.poisson(modimg[miny:maxy, minx:maxx])
        simmod = np.nan_to_num(np.divide(randmod, modimg[miny:maxy, minx:maxx]))
        # Set the scales
        minscale = 2  # minimum scale of 2 pixels
        maxscale = regsizepix / 2. # at least 4 resolution elements on a side
        scale = np.logspace(np.log10(minscale), np.log10(maxscale), 10)  # 10 scale logarithmically spaced
        sckpc = scale * pixsize * kpcp
        # Convolve images
        for i in range(len(scale)):
            sc = scale[i]
            print('Convolving with scale', sc)
            convimg, convmod = calc_mexicanhat(sc, img, mask, simmod)
            # Save image
            fmod[0].data = convimg
            fmod.writeto('conv_scale_%d_kpc.fits' % (int(np.round(sckpc[i]))), overwrite=True)
            fmod[0].data = convmod
            fmod.writeto('conv_model_%d_kpc.fits' % (int(np.round(sckpc[i]))), overwrite=True)
        fmod.close()

    #
    def PS(self, z, region_size=1., radius_in=0., radius_out=1.):
        """
        Function to compute the power spectrum from existing Mexican Hat images in a given circle or annulus

        :param z: Source redshift
        :type z: float
        :param region_size: Size of the region of interest in Mpc. Defaults to 1.0. This value must be equal to the region_size parameter used in :meth:`pyproffit.power_spectrum.PowerSpectrum.MexicanHat`.
        :type region_size: float
        :param radius_in: Inner boundary in Mpc of the annulus to be used. Defaults to 0.0
        :type radius_in: float
        :param radius_out: Outer boundary in Mpc of the annulus to be used. Defaults to 1.0
        :type radius_out: float
        """
        kpcp = cosmo.kpc_proper_per_arcmin(z).value
        Mpcpix = 1000. / kpcp / self.data.pixsize  # 1 Mpc in pixel
        regsizepix = region_size * Mpcpix
        ######################
        # Set the scales
        ######################
        minscale = 2  # minimum scale of 2 pixels
        maxscale = regsizepix / 2.
        scale = np.logspace(np.log10(minscale), np.log10(maxscale), 10)  # 10 scale logarithmically spaced
        sckpc = scale * self.data.pixsize * kpcp
        kr = 1. / np.sqrt(2. * np.pi ** 2) * np.divide(1., scale)  # Eq. A5 of Arevalo et al. 2012
        ######################
        # Define the region where the power spectrum will be extracted
        ######################
        fmask = fits.open('mask.fits')
        mask = fmask[0].data
        data_size = mask.shape
        fmask.close()
        y, x = np.indices(data_size)
        rads = np.hypot(y - data_size[0] / 2., x - data_size[1] / 2.)
        region = np.where(
            np.logical_and(np.logical_and(rads > radius_in * Mpcpix, rads <= radius_out * Mpcpix), mask > 0.0))
        ######################
        # Extract the PS from the various images
        ######################
        nsc = len(scale)
        ps, psnoise, amp, eamp = np.empty(nsc), np.empty(nsc), np.empty(nsc), np.empty(nsc)
        vals = []
        nreg = 20  # Number of subregions for bootstrap calculation
        for i in range(nsc):
            # Read images
            fco = fits.open('conv_scale_%d_kpc.fits' % (int(np.round(sckpc[i]))))
            convimg = fco[0].data.astype(float)
            fco.close()
            fmod = fits.open('conv_model_%d_kpc.fits' % (int(np.round(sckpc[i]))))
            convmod = fmod[0].data.astype(float)
            fmod.close()
            print('Computing the power at scale', sckpc[i], 'kpc')
            ps[i], psnoise[i], vps = calc_ps(region, convimg, convmod, kr[i], nreg)
            vals.append(vps)
        # Bootstrap the data and compute covariance matrix
        print('Computing the covariance matrix...')
        nboot = int(1e4)  # number of bootstrap resamplings
        cov = do_bootstrap(vals, nboot)
        # compute eigenvalues of covariance matrix to verify that the matrix is positive definite
        la, v = np.linalg.eig(cov)
        print('Eigenvalues: ', la)
        eps = np.empty(nsc)
        for i in range(nsc):
            eps[i] = np.sqrt(cov[i, i])
        amp = np.sqrt(np.abs(ps) * 2. * np.pi * kr ** 2 / cf)
        eamp = 1. / 2. * np.power(np.abs(ps) * 2. * np.pi * kr ** 2 / cf, -0.5) * 2. * np.pi * kr ** 2 / cf * eps
        self.kpix = kr
        self.k = 1. / np.sqrt(2. * np.pi ** 2) * np.divide(1., sckpc)
        self.ps = ps
        self.eps = eps
        self.psnoise = psnoise
        self.amp = amp
        self.eamp = eamp
        self.cov = cov

    def Plot(self, save_plots=True, outps='power_spectrum.pdf', outamp='a2d.pdf', plot_3d=False, cfact=None):
        """
        Plot the loaded power spectrum

        :param save_plots: Indicate whether the plot should be saved to disk or not. Defaults to True.
        :type save_plots: bool
        :param outps: Name of output file. Defaults to 'power_spectrum.pdf'
        :type outps: str
        :param outamp: Name of output file to save the 2D amplitude plot. Defaults to 'a2d.pdf'
        :type outamp: str
        :param plot_3d: Add or not the 3D power spectrum to the plot. Defaults to False
        :type plot_3d: bool
        :param cfact: 2D to 3D projection factor
        :type cfact: class:`numpy.ndarray` , optional
        """
        if self.ps is None:
            print('Error: No power spectrum exists in structure')
            return
        plt.clf()
        fig = plt.figure(figsize=(13, 10))
        ax_size = [0.1, 0.1,
                   0.87, 0.87]
        ax = fig.add_axes(ax_size)
        ax.minorticks_on()
        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        plt.xlabel('k [kpc$^{-1}$]', fontsize=40)
        plt.ylabel('2D Power', fontsize=40)
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(self.k, self.ps, color='red', linewidth=2, label='P$_{2D}$')
        plt.plot(self.k, self.psnoise, color='blue', label='Poisson noise')
        plt.fill_between(self.k, self.ps - self.eps, self.ps + self.eps, color='red', alpha=0.4)
        if plot_3d:
            kcf = cfact[:,0]
            cf = cfact[:,3]/cfact[:,1]
            interp_cf = np.interp(self.kpix,kcf,cf)
            ps3d = self.ps * interp_cf
            eps3d = self.eps * interp_cf
            plt.plot(self.k, ps3d, color='green', linewidth=2, label='P$_{3D}$')
            plt.fill_between(self.k, ps3d - eps3d, ps3d + eps3d, color='green', alpha=0.4)
        plt.legend(fontsize=22)
        if save_plots:
            plt.savefig(outps)
        else:
            plt.show()

        plt.clf()
        fig = plt.figure(figsize=(13, 10))
        ax_size = [0.1, 0.1,
                   0.87, 0.87]
        ax = fig.add_axes(ax_size)
        ax.minorticks_on()
        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)
        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        plt.xlabel('k [kpc$^{-1}$]', fontsize=40)
        plt.ylabel('$A_{2D}$', fontsize=40)
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(self.k, self.amp, color='red', linewidth=2,label='A$_{2D}$')
        plt.fill_between(self.k, self.amp - self.eamp, self.amp + self.eamp, color='red', alpha=0.4)
        if plot_3d:
            a3d=np.sqrt(ps3d*4.*np.pi*self.kpix**3)/2.
            ea3d=1./2.*np.power(ps3d*4.*np.pi*self.kpix**3,-0.5)*4.*np.pi*self.kpix**3*eps3d/2.
            plt.plot(self.k, a3d, color='green', linewidth=2,label='A$_{3D}$')
            plt.fill_between(self.k, a3d - ea3d, a3d + ea3d, color='green', alpha=0.4)
        plt.legend(fontsize=22)
        if save_plots:
            plt.savefig(outamp)
        else:
            plt.show()

    def Save(self, outfile, outcov='covariance.txt'):
        """
        Save the loaded power spectra to an output ASCII file

        :param outfile: Name of output ASCII file
        :type outfile: str
        :param outcov: Output covariance matrix. Defaults to 'covariance.txt'
        :type outcov: str
        """
        if self.ps is None:
            print('Error: Nothing to save')
            return
        np.savetxt(outfile, np.transpose([self.k, self.ps, self.eps, self.psnoise, self.amp, self.eamp])[::-1],
                   header='k/kpc-1  PS2D  dPS2D  Noise  A2D  dA2D')
        np.savetxt(outcov, self.cov)

    def ProjectionFactor(self, z, betaparams, region_size=1.):
        """
        Compute numerically the 2D to 3D deprojection factor. The routine is simulating 3D fluctuations using the surface brightness model and the region mask, projecting the 3D data along one axis, and computing the ratio of 2D to 3D power as a function of scale.

        Caution: this is a memory intensive computation which will run into memory overflow if the image size is too large or the available memory is insufficient.

        :param z: Source redshift
        :type z: float
        :param betaparams: Parameters of the beta model or double beta model
        :type betaparams: class:`numpy.ndarray`
        :param region_size: Size of the region of interest in Mpc. Defaults to 1.0. This value must be equal to the region_size parameter used in :meth:`pyproffit.power_spectrum.PowerSpectrum.MexicanHat`.
        :type region_size: float
        :return: Array of projection factors
        :rtype: class:`numpy.ndarray`
        """
        pixsize = self.data.pixsize
        npar = len(betaparams)
        if npar == 4:
            print('We will use a single beta profile')
            betaparams[1] = betaparams[1] / pixsize
            betaparams[2] = 0.
        elif npar == 6:
            print('We will use a double beta profile')
            betaparams[1] = betaparams[1] / pixsize
            betaparams[2] = betaparams[2] / pixsize
            betaparams[4] = 0.
        else:
            print('Invalid number of SB parameters')
            return
        fmask = fits.open('mask.fits')
        mask = fmask[0].data
        data_size = mask.shape
        fmask.close()
        kpcp = cosmo.kpc_proper_per_arcmin(z).value
        Mpcpix = 1000. / kpcp / self.data.pixsize  # 1 Mpc in pixel
        regsizepix = region_size * Mpcpix
        if regsizepix>data_size[0]/2:
            print('Error: region size larger than image size')
            return
        minx = int(np.round(data_size[1]/2 - regsizepix))
        maxx = int(np.round(data_size[1]/2 + regsizepix))
        miny = int(np.round(data_size[0]/2 - regsizepix))
        maxy = int(np.round(data_size[0]/2 + regsizepix))
        msk = mask[miny:maxy, minx:maxx]
        npix = len(msk)
        minscale = 2  # minimum scale of 2 pixels
        maxscale = regsizepix / 2. # at least 4 resolution elements on a side
        scale = np.logspace(np.log10(minscale), np.log10(maxscale), 10)  # 10 scale logarithmically spaced
        self.cfact = calc_projection_factor(npix, msk, betaparams, scale)
        return  self.cfact
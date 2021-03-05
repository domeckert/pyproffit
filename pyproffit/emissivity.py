import os

def is_tool(name):
    """Check whether `name` is on PATH."""

    from distutils.spawn import find_executable

    return find_executable(name) is not None


def calc_emissivity(cosmo, z, nh, kt, rmf, Z=0.3, elow=0.5, ehigh=2.0, arf=None, type='cr', lum_elow=0.5, lum_ehigh=2.0, abund='aspl'):
    """

    Function calc_emissivity. The function computes the scaling factor between count rate and APEC/MEKAL norm using XSPEC, which is needed to extract density profiles.
    Requires XSPEC to be in PATH

    :param cosmo: Astropy cosmology object
    :type cosmo: class: astropy.cosmology
    :param z: Source redshift
    :type z: float
    :param nh: Source NH in units of 1e22 cm**(-2)
    :type nh: float
    :param kt: Source temperature in keV
    :type kt: float
    :param rmf: Path to response file (RMF/RSP)
    :type rmf: str
    :param Z: Metallicity with respect to solar (default = 0.3)
    :type Z: float
    :param elow: Low-energy bound of the input image in keV (default = 0.5)
    :type elow: float
    :param ehigh: High-energy bound of the input image in keV (default = 2.0)
    :type ehigh: float
    :param arf: Path to on-axis ARF (optional, in case response file is RMF)
    :type arf: str
    :param type: Specify whether the exposure map is in units of sec (type='cr') or photon flux (type='photon'). By default type='cr'.
    :type type: str
    :param lum_elow: Low energy bound (rest frame) for luminosity calculation. Defaults to 0.5
    :type lum_elow: float
    :param lum_ehigh: High energy bound (rest frame) for luminosity calculation. Defaults to 2.0
    :type lum_ehigh: float
    :return: Conversion factor
    :rtype: float
    """

    check_xspec = is_tool('xspec')

    if not check_xspec:

        print('Error: XSPEC cannot be found in path')

        return

    if type != 'cr' and type != 'photon':

        print('Unknown type '+type+', reverting to CR by default')

        type = 'cr'

    H0 = cosmo.H0.value

    Ode = cosmo.Ode0

    fsim=open('commands.xcm','w')

    # fsim.write('query y\n')

    fsim.write('cosmo %g 0 %g\n' % (H0, Ode))

    fsim.write('abund %s\n' %(abund))

    fsim.write('model phabs(apec)\n')

    fsim.write('%g\n'%(nh))

    fsim.write('%g\n'%(kt))

    fsim.write('%g\n'%(Z))

    fsim.write('%g\n'%(z))

    fsim.write('1.0\n')

    fsim.write('fakeit none\n')

    fsim.write('%s\n' % (rmf))

    if arf is not None:

        fsim.write('%s\n' % (arf))

    else:

        fsim.write('\n')

    fsim.write('\n')

    fsim.write('\n')

    fsim.write('\n')

    fsim.write('10000, 1\n')

    fsim.write('ign **-%1.2lf\n' % (elow))

    fsim.write('ign %1.2lf-**\n' % (ehigh))

    fsim.write('log sim.txt\n')

    if type == 'cr':

        fsim.write('show rates\n')

    elif type == 'photon':

        fsim.write('flux %1.2lf %1.2lf\n' % (elow, ehigh))

    fsim.write('log none\n')

    fsim.write('delcomp 1\n')

    fsim.write('log lumin.txt\n')

    fsim.write('lumin %1.2lf %1.2lf %g\n' % (lum_elow, lum_ehigh, z))

    fsim.write('log none\n')

    fsim.write('quit\n')

    fsim.close()

    nrmf_tot = rmf.split('.')[0]

    if os.path.exists('%s.fak' % (nrmf_tot)):

        os.system('rm %s.fak' % (nrmf_tot))

    srmf = nrmf_tot.split('/')

    nrmf = srmf[len(srmf) - 1]

    if os.path.exists('%s.fak' % (nrmf)):

        os.system('rm %s.fak' % (nrmf))

    os.system('xspec < commands.xcm')

    if type == 'cr':

        ssim = os.popen('grep cts/s sim.txt', 'r')

        lsim = ssim.readline()

        cr = float(lsim.split()[6])

    elif type == 'photon':

        ssim = os.popen('grep photons sim.txt', 'r')

        lsim = ssim.readline()

        cr = float(lsim.split()[3])

    slum = os.popen('grep Luminosity lumin.txt', 'r')

    llum = slum.readline()

    lumtot = float(llum.split()[2])

    lumfact = lumtot / cr

    return cr, lumfact





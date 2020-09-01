Introduction
============

``pyproffit`` is a high-level Python package which aims to provide an easy and intuitive way of performing photometric analysis with X-ray images of galaxy clusters. It is essentially a Python replacement for the PROFFIT C++ interactive package (Eckert et al. 2011). It includes all features of the original PROFFIT package, and more. Available features include:

- Extraction of surface brightness profiles in circular and elliptical annuli, over the entire azimuth or any given sector
- Fitting of profiles with a number of built-in model or any given user-defined model, using chi-squared or C statistic
- Deprojection and extraction of gas density profiles and gas masses
- PSF deconvolution, count rate and luminosity reconstruction in any user defined radial range, surface brightness concentration
- Two-dimensional model images and surface brightness deviations
- Surface brightness fluctuation power spectra and conversion into 3D density power spectra

The current implementation has been developed in Python 3 and tested on Python 3.6+ under Linux and Mac OS.

Motivation
**********

While the original PROFFIT package has attracted a substantial number of users, its structure was extremely rigid and outdated, making it difficult to maintain and very difficult for the user to add any custom features. ``pyproffit`` aims at providing all the popular features of PROFFIT in the form of an easy-to-use Python package. The modular structure of pyproffit allows the user to easily interface with other Python packages and develop additional features and models. The ultimate goal is to allow the user to perform any type of analysis directly within a Jupyter notebook.

Limitations
***********

- The current fitting process uses a maximum-likelihood optimization through the iminuit library. While this is fine for simple cases, it is difficult to sample the entire parameter space and provide accurate uncertainties through this method. I am currently working on replacing the iminuit optimizer with the powerful PyMC3 optimization package.

- The computation of the PSF mixing matrix currently only works with PSF images that have the same pixel size as the provided image.

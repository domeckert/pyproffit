Installation
============

``pyproffit`` is available on `Github <https://github.com/domeckert/pyproffit>`_ and `PyPI <https://pypi.org/project/pyproffit/>`_. 

The easiest way of installing ``pyproffit`` is obviously to use pip::

    pip3 install pyproffit
    
The PyPI repository should contain the latest stable release (as judged by the developer), it may not be the latest version thus some features may be missing. To install the latest version from Github::

    git clone https://github.com/domeckert/pyproffit.git
    cd pyproffit
    pip3 install .
    
``pyproffit`` depends on numpy, scipy, astropy, matplotlib, iminuit, pymc3, and pystan.

from setuptools import setup


setup(
      name='pyproffit',    # This is the name of your PyPI-package.
      version='0.8',
      description='Python package for PROFFIT',
      author='Dominique Eckert',
      author_email='Dominique.Eckert@unige.ch',
      url="https://github.com/domeckert/pyproffit",
      packages=['pyproffit'],
      install_requires=[
            'numpy','scipy','astropy','matplotlib','iminuit','pymc'
      ],
)


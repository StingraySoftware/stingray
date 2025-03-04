*****************************************
Stingray: Next-Generation Spectral Timing
*****************************************

.. image:: images/stingray_logo.png
   :width: 700
   :scale: 40%
   :alt: Stingray logo, outline of a stingray on top of a graph of the power spectrum of an X-ray binary
   :align: center

.. csv-table::
   :header: "Usage", "Release", "Development", "Community"
   :widths: auto
   :align: center

   |Python version|, |GitHub release|, |Build Status Master|, |Slack|
   |Docs|, |joss|, |Repo status|,  |pyOpenSci Peer-Reviewed|
   |License|, |doi|, |Coverage Status Master|, " "


Stingray is a Python library designed to perform times series analysis and related tasks on astronomical light curves.
It supports a range of commonly-used Fourier analysis techniques, as well as extensions for analyzing pulsar data, simulating data sets, and statistical modelling.
Stingray is designed to be easy to extend, and easy to incorporate into data analysis workflows and pipelines.

.. important::

   If you use Stingray for work presented in a publication or talk, please help the project by providing a proper :doc:`citation <citing>`.

Features
========
Current Capabilities
--------------------

1. Data handling and simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* loading event lists from fits files (and generally good handling of OGIP-compliant missions, like RXTE/PCA, NuSTAR/FPM, XMM-Newton/EPIC, NICER/XTI)
* constructing light curves and time series from event data
* various operations on time series (e.g. addition, subtraction, joining, and truncation)
* simulating a light curve with a given power spectrum
* simulating a light curve from another light curve and a 1-d (time) or 2-d (time-energy) impulse response
* simulating an event list from a given light curve _and_ with a given energy spectrum
* Good Time Interval operations
* Filling gaps in light curves with statistically sound fake data

2. Fourier methods
~~~~~~~~~~~~~~~~~~
* power spectra and cross spectra in Leahy, rms normalization, absolute rms and no normalization
* averaged power spectra and cross spectra
* dynamical power spectra and cross spectra
* maximum likelihood fitting of periodograms/parametric models
* (averaged) cross spectra
* coherence, time lags
* Variability-Energy spectra, like covariance spectra and lags *needs testing*
* covariance spectra; *needs testing*
* bispectra; *needs testing*
* (Bayesian) quasi-periodic oscillation searches
* Lomb-Scargle periodograms and cross spectra
* Power Colors

3. Other time series methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* pulsar searches with Epoch Folding, :math:`Z^2_n` test
* Gaussian Processes for QPO studies
* cross correlation functions

Future Plans
------------

We welcome feature requests: if you need a particular tool that's currently not available or have a new method you think might be usefully implemented in Stingray, please :doc:`get in touch <contributing>`!

Other future additions we are currently implementing are:

* bicoherence
* phase-resolved spectroscopy of quasi-periodic oscillations
* Fourier-frequency-resolved spectroscopy
* full HEASARC-compatible mission support
* pulsar searches with :math:`H`-test
* binary pulsar searches

Platform-specific issues
------------------------

Windows uses an internal 32-bit representation for ``int``. This might create numerical errors when using large integer numbers (e.g. when calculating the sum of a light curve, if the ``lc.counts`` array is an integer).
On Windows, we automatically convert the ``counts`` array to float. The small numerical errors should be a relatively small issue compare to the above.

Installation instructions
=========================

There are currently three ways to install Stingray:

* via ``conda``
* via ``pip``
* from source

Below, you can find instructions for each of these methods.

Dependencies
------------
A **minimal installation** of Stingray requires the following dependencies:

+ astropy>=4.0
+ numpy>=1.17.0
+ scipy>=1.1.0
+ matplotlib>=3.0,!=3.4.0

In **typical** uses, requiring input/output, caching of results, and faster processing, we **recommend the following dependencies**:

+ numba (**highly** recommended)
+ tqdm (for progress bars, always useful)
+ pyfftw (for the fastest FFT in the West)
+ h5py (for input/output)
+ pyyaml (for input/output)
+ emcee (for MCMC analysis, e.g. for PSD fitting)
+ corner (for the plotting of MCMC results)
+ statsmodels (for some statistical analysis)

For **pulsar searches and timing**, we recommend installing

+ pint-pulsar

Some of the dependencies are available in ``conda``, the others via ``pip``.
To install all required and recommended dependencies in a recent installation, you should be good running the following command::

    $ pip install astropy scipy matplotlib numpy h5py tqdm numba pint-pulsar emcee corner statsmodels pyfftw tbb

For the Gaussian Process modeling in `stingray.modeling.gpmodeling`, you'll need the following extra packages

+ jax
+ jaxns
+ tensorflow
+ tensorflow-probability
+ tinygp
+ etils
+ typing_extensions

For the Bexvar calculations in `stingray.bexvar` and `stingray.lightcurve`, you'll need `UltraNest <https://johannesbuchner.github.io/UltraNest/index.html>`_.

Most of these are installed via ``pip``, but if you have an Nvidia GPU available, you'll want to take special care
following the installation instructions for jax and tensorflow(-probability) in order to enable GPU support and
take advantage of those speed-ups.

For development work, you will need the following extra libraries:

+ pytest
+ pytest-astropy
+ tox
+ jinja2==3.1.5
+ docutils
+ sphinx-astropy
+ nbsphinx>=0.8.3,!=0.8.8
+ pandoc
+ ipython
+ jupyter
+ notebook
+ towncrier
+ black

Which can be installed with the following command:

    $ pip install pytest pytest-astropy jinja2 docutils sphinx-astropy nbsphinx pandoc ipython jupyter notebook towncrier tox black

Installation
------------
Installing via ``conda``
~~~~~~~~~~~~~~~~~~~~~~~~

If you manage your Python installation and packages
via Anaconda or miniconda, you can install ``stingray``
via the ``conda-forge`` build::

    $ conda install -c conda-forge stingray

That should be all you need to do! Just remember to :ref:`run the tests <testsuite>` before
you use it!

Installing via ``pip``
~~~~~~~~~~~~~~~~~~~~~~

``pip``-installing Stingray is easy! Just do::

    $ pip install stingray

And you should be done! Just remember to :ref:`run the tests <testsuite>` before you use it!

Installing from source (bleeding edge version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For those of you wanting to install the bleeding-edge development version from
source (it *will* have bugs; you've been warned!), first clone
`our repository <https://github.com/StingraySoftware/stingray>`_ on GitHub::

    $ git clone --recursive https://github.com/StingraySoftware/stingray.git

Now ``cd`` into the newly created ``stingray`` directory.
Finally, install ``stingray`` itself::

    $ pip install -e "."

Installing development environment (for new contributors)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For those of you wanting to contribute to the project, install the bleeding-edge development version from
source. First fork
`our repository <https://github.com/StingraySoftware/stingray>`_ on GitHub and clone the forked repository using::

    $ git clone --recursive https://github.com/<your github username>/stingray.git

Now, navigate to this folder and run
the following command to add an upstream remote that's linked to Stingray's main repository.
(This will be necessary when submitting PRs later.)::

    $ cd stingray
    $ git remote add upstream https://github.com/StingraySoftware/stingray.git

Now, install the necessary dependencies::

    $ pip install astropy scipy matplotlib numpy pytest pytest-astropy h5py tqdm

Finally, install ``stingray`` itself::

    $ pip install -e "."

.. _testsuite:

Test Suite
----------

Please be sure to run the test suite before you use the package, and please report anything
you think might be bugs on our GitHub `Issues page <https://github.com/StingraySoftware/stingray/issues>`_.

Stingray uses `py.test <https://pytest.org>`_ and `tox
<https://tox.readthedocs.io>`_ for testing.

If you downloaded the source code, e.g. via `git clone`, run the tests with::

   $ tox -e test

You may need to install tox first::

   $ pip install tox

To run a specific test file (e.g., test_io.py), try::

    $ cd stingray
    $ py.test tests/test_io.py

If you have installed Stingray via pip or conda, the source directory might
not be easily accessible. Once installed, you can also run the tests using::

   $ python -c 'import stingray; stingray.test()'

or from within a python interpreter:

.. doctest-skip::

   >>> import stingray
   >>> stingray.test()

Building the Documentation
--------------------------

The documentation including tutorials is hosted `here <https://docs.stingray.science/>`_.
The documentation uses `sphinx <https://www.sphinx-doc.org/en/stable/>`_ to build and requires the extensions `sphinx-astropy <https://pypi.org/project/sphinx-astropy/>`_ and `nbsphinx <https://pypi.org/project/nbsphinx/>`_.

To build the documentation, you will need to clone the development version of Stingray from GitHub.
Once you have the source code, you can go to the root directory and make the build in one of the following ways.
The quickest is probably using our tox environment::

    $ tox -e build_docs

You can build the API reference yourself by going into the ``docs`` folder within the ``stingray`` root
directory and running the ``Makefile``::

    $ cd stingray/docs
    $ make html

If that doesn't work on your system, you can invoke ``sphinx-build`` itself from the stingray source directory::

    $ cd stingray
    $ sphinx-build docs docs/_build

The documentation should be located in ``stingray/docs/_build``. Try opening ``./docs/_build/index.rst`` from
the stingray source directory.

Using Stingray
===============

The documentation below is built on top of Jupyter notebooks that can be run locally.
The easiest way to retrieve the notebooks is by `cloning the notebooks repository <https://github.com/StingraySoftware/notebooks>`_ and browsing the directories, which are conveniently divided by topic.

A Spectral timing exploration
-----------------------------

In this Tutorial, we will show an example spectral timing exploration of a
black hole binary using NICER data. The tutorial includes a hardness-intensity
diagram, the modeling of the power density spectrum, power colors, lag-frequency,
lag-energy, and rms/covariance spectra.

.. toctree::
   :maxdepth: 1

   notebooks/Spectral Timing/Spectral Timing Exploration.ipynb


Stingray fundamentals
---------------------
.. toctree::
   :maxdepth: 2

   core
   dataexplo
   pulsar
   modeling
   simulator
   deadtime

Advanced
--------

.. toctree::
   :maxdepth: 2

   timeseries
   largedata
   api

Additional information
======================

.. toctree::
   :maxdepth: 2

   history
   contributing
   citing
   acknowledgements

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |Python version| image:: https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/stingraysoftware/stingray/main/pyproject.toml
   :alt: Python Version from PEP 621 TOML
.. |Build Status Master| image:: https://github.com/StingraySoftware/stingray/workflows/CI%20Tests/badge.svg
   :target: https://github.com/StingraySoftware/stingray/actions/
.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://docs.stingray.science/
.. |Slack| image:: https://img.shields.io/badge/Join%20Our%20Community-Slack-blue
   :target: https://join.slack.com/t/stingraysoftware/shared_invite/zt-49kv4kba-mD1Y~s~rlrOOmvqM7mZugQ
.. |Coverage Status Master| image:: https://codecov.io/gh/StingraySoftware/stingray/branch/master/graph/badge.svg?token=FjWeFfhU9F
   :target: https://codecov.io/gh/StingraySoftware/stingray
.. |GitHub release| image:: https://img.shields.io/github/v/release/StingraySoftware/stingray
   :target: https://github.com/StingraySoftware/stingray/releases/latest
.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.07389/status.svg
   :target: https://doi.org/10.21105/joss.07389
.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1490116.svg
   :target: https://doi.org/10.5281/zenodo.1490116
.. |Repo status| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active
.. |License| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. |pyOpenSci Peer-Reviewed| image:: https://pyopensci.org/badges/peer-reviewed.svg
   :target: https://github.com/pyOpenSci/software-submission/issues/201

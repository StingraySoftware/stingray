========
Stingray
========

.. csv-table::
   :header: "Usage", "Release", "Development", "Community"
   :widths: 50, 50, 50, 50

   |Python version|, |GitHub release|, |Build Status Master|, |Slack|
   |Docs|, |joss|, |Repo status|,  |pyOpenSci Peer-Reviewed|
   |License|, |doi|, |Coverage Status Master|, " "



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
X-Ray Spectral Timing Made Easy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stingray is a spectral-timing software package for astrophysical X-ray (and other) data.
Stingray merges existing efforts for a (spectral-)timing package in Python, and is structured with the best guidelines for modern open-source programming, following the example of `Astropy`_.

It provides:

- a library of time series methods, including power spectra, cross spectra, covariance spectra, lags, and so on;
- a set of scripts to load FITS data files from different missions;
- a light curve and event list simulator, with the ability to simulate different kinds of variability and more complicated phenomena based on the impulse response of given physical events (e.g. reverberation).

There are a number of official software packages for X-ray spectral fitting (Xspec, ISIS, Sherpa, ...).
However, an equivalent widely-used package does not exist for X-ray timing: to date, that has generally been done with custom software.
Stingray aims not only to fill that gap, but also to provide implementations of the most advanced spectral timing techniques available in the literature.
The ultimate goal of this project is to provide the community with a package that eases the learning curve for the advanced spectral timing techniques with a correct statistical framework.

More details of current and planned capabilities are available in the `Stingray documentation <https://docs.stingray.science/en/stable/#features>`_.

Installation and Testing
------------------------

Stingray can be installed via `conda`, `pip`, or directly from the source repository itself.
Our documentation provides `comprehensive installation instructions <https://docs.stingray.science/en/stable/#installation>`_.

After installation, it's a good idea to run the test suite.
We use `py.test <https://pytest.org>`_ and `tox <https://tox.readthedocs.io>`_ for testing, and, again, our documentation provides `step-by-step instructions <https://docs.stingray.science/en/stable/#test-suite>`_.

Documentation
-------------

Stingray's documentation can be found at https://docs.stingray.science/.

Getting In Touch, and Getting Involved
--------------------------------------

We welcome contributions and feedback, and we need your help!
The best way to get in touch is via the `issues_` page.
We're especially interested in hearing from you:

- If something breaks;
- If you spot missing functionality, find the API unintuitive, or have suggestions for future development;
- If you have your own code implementing any of the methods provided Stingray and it produces different answers.

Even better: if you have code you'd be willing to contribute, please send a `pull request`_ or open an `issue`_.

Related Packages
----------------

- `HENDRICS <https://hendrics.stingray.science/>`_ provides a set of command-line scripts which use Stingray to perform quick-look spectral timing analysis of X-ray data.
- `DAVE <https://github.com/StingraySoftware/dave>`_ is a graphical user interface built on top of Stingray.

Citing Stingray
---------------

If you find this package useful in your research, please provide appropriate acknowledgement and citation.
`Our documentation <https://docs.stingray.science/en/stable/citing.html>`_ gives further guidance, including links to appropriate papers and convenient BibTeX entries.

Copyright & Licensing
---------------------

All content © 2024 The Authors.
The code is distributed under the MIT license; see `LICENSE.rst <LICENSE.rst>`_ for details.

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

.. _Astropy: https://www.github.com/astropy/astropy
.. _Issues: https://www.github.com/stingraysoftware/stingray/issues
.. _Issue: https://www.github.com/stingraysoftware/stingray/issues
.. _pull request: https://github.com/StingraySoftware/stingray/pulls

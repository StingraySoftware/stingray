*******
History
*******

For a brief overview of the history and state-of-the-art in spectral timing, and for more information about the design and capabilities of Stingray, please refer to `Huppenkothen et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019ApJ...881...39H/abstract>`_.

Stingray originated during the 2016 workshop `The X-ray Spectral-Timing Revolution <http://www.lorentzcenter.nl/lc/web/2016/720/info.php3?wsid=720&venue=Oort/>`_: a group of X-ray astronomers and developers decided to agree on a common platform to develop a new software package.
At that time, there were a number of official software packages for X-ray spectral fitting (XSPEC, ISIS, Sherpa, ...), but
such a widely used and standard software package did not exist for X-ray timing, that was mostly the domain of custom, proprietary software.
Our goals were to merge existing efforts towards a timing package in Python, following the best guidelines for modern open-source programming, thereby providing the basis for developing spectral-timing analysis tools.
We needed to provide an easily accessible scripting interface, a GUI, and an API for experienced coders.
Stingray's ultimate goal is to provide the community with a package that eases the learning curve for advanced spectral-timing techniques, with a correct statistical framework.

Further spectral-timing functionality, in particularly command line scripts based on the API defined within Stingray, is available in the package `HENDRICS <https://github.com/StingraySoftware/HENDRICS>`_.
A graphical user interface is under development as part of the project `DAVE <https://github.com/StingraySoftware/dave>`_.

Previous projects merged to Stingray
====================================

* Daniela Huppenkothen's original Stingray
* Matteo Bachetti's `MaLTPyNT <https://github.com/matteobachetti/MaLTPyNT>`_
* Abigail Stevens' RXTE power spectra code and phase-resolved spectroscopy code
* Simone Migliari's and Paul Balm's X-ray data exploration GUI commissioned by ESA


Changelog
=========

.. include:: ../CHANGELOG.rst


Presentations
=============

Members of the Stingray team have given a number of presentations which introduce Stingray.
These include:

- `2nd Severo Ochoa School on Statistics, Data Mining, and Machine Learning (2021) <https://github.com/abigailStev/timeseries-tutorial>`_
- `9th Microquasar Workshop (2021) <https://speakerdeck.com/abigailstev/time-series-exploration-with-stingray>`_
- `European Week of Astronomy and Space Science (2018) <http://ascl.net/wordpress/2018/05/24/software-in-astronomy-symposium-presentations-part-3/>`_
- `ADASS (Astronomical Data Analysis Software and Systems; meeting 2017, proceedings 2020) <https://ui.adsabs.harvard.edu/abs/2020ASPC..522..521M/abstract>`_
- `AAS 16th High-Energy Astrophysics Division meeting (2017) <https://speakerdeck.com/abigailstev/stingray-open-source-spectral-timing-software>`_
- `European Week of Astronomy and Space Science 2017 <http://ascl.net/wordpress/2017/07/23/special-session-on-and-about-software-at-ewass-2017/>`_
- `Python in Astronomy (2016) <https://speakerdeck.com/abigailstev/stingray-pyastro16>`_

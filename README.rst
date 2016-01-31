X-Ray Timing Made Easy
=======================

We are writing a general-purpose timing package for X-ray time series.

Authors
--------
* Abigail Stevens (UvA)
* Daniela Huppenkothen (NYU CDS)

Contributers
------------
* Evandro Martinez Ribeiro (RuG)

Contents
--------

The contents will be:

- make a light curve from event data
- make periodograms in Leahy and rms normalization
- average periodograms
- cross spectra and lags (time vs energy, time vs frequency)
- maximum likelihood fitting of periodograms/parametric models
- bispectra (?)
- cross correlation functions, coherence
- spectral-timing functionality
- Bayesian QPO searches
- power colours
- rms spectra

Documentation
-------------

Is generated using `Sphinx`_. Try::

   $ sphinx-build doc doc/_build

Then open ``./doc/_build/index.html`` in the browser of your choice.

.. _Sphinx: http://sphinx-doc.org

Test suite
----------

Try::

   $ nosetests

Copyright
---------

All content © 2015 the authors. The code is distributed under the MIT license.

Pull requests are welcome! If you are interested in the further development of
this project, please `get in touch via the issues
<https://github.com/dhuppenkothen/stingray/issues>`_!

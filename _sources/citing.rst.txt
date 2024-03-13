***************
Citing Stingray
***************

Citations are still the main currency of the academic world, and *the* best way to ensure that Stingray continues to be supported and we can continue to work on it.
If you use Stingray in data analysis leading to a publication, we ask that you cite *both* a `DOI <https://www.doi.org>`_, which points to the software itself, *and* our papers describing the Stingray project.

DOI
===

If possible, we ask that you cite a DOI corresponding to the specific version of Stingray that you used to carry out your analysis.

.. include:: _zenodo.rst

If this isn't possible — for example, because you worked with an unreleased version of the code — you can cite Stingray's `concept DOI <https://help.zenodo.org/faq/#versioning>`__, `10.5281/zenodo.1490116 <https://zenodo.org/record/1490116>`__ (`BibTeX <https://zenodo.org/record/1490116/export/hx>`__), which will always resolve to the latest release.

Papers
======

Please cite both of the following papers:

.. raw:: html

   <script type="text/javascript">
       function copyApjBib() {
           var bibtex = `@ARTICLE{2019ApJ...881...39H,
             author = {{Huppenkothen}, Daniela and {Bachetti}, Matteo and
                       {Stevens}, Abigail L. and {Migliari}, Simone and {Balm}, Paul and
                       {Hammad}, Omar and {Khan}, Usman Mahmood and {Mishra}, Himanshu and
                       {Rashid}, Haroon and {Sharma}, Swapnil and {Martinez Ribeiro}, Evandro and
                       {Valles Blanco}, Ricardo},
             title = "{Stingray: A Modern Python Library for Spectral Timing}",
             journal = {\apj},
             keywords = {methods: data analysis, methods: statistical, X-rays: binaries, X-rays: general, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - High Energy Astrophysical Phenomena},
             year = 2019,
             month = aug,
             volume = {881},
             number = {1},
             eid = {39},
             pages = {39},
             doi = {10.3847/1538-4357/ab258d},
             archivePrefix = {arXiv},
             eprint = {1901.07681},
             primaryClass = {astro-ph.IM},
             adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJ...881...39H},
             adsnote = {Provided by the SAO/NASA Astrophysics Data System}
           }`;
           const el = document.createElement('textarea');
           el.value = bibtex;
           document.body.appendChild(el);
           el.select();
           document.execCommand('copy');
           document.body.removeChild(el);
       }

       function copyJossBib() {
           var bibtex = `@ARTICLE{Huppenkothen2019,
             doi = {10.21105/joss.01393},
             url = {https://doi.org/10.21105/joss.01393},
             year = {2019},
             publisher = {The Open Journal},
             volume = {4},
             number = {38},
             pages = {1393},
             author = {Daniela Huppenkothen and Matteo Bachetti and Abigail Stevens and Simone Migliari and Paul Balm and Omar Hammad and Usman Mahmood Khan and Himanshu Mishra and Haroon Rashid and Swapnil Sharma and Evandro Martinez Ribeiro and Ricardo Valles Blanco},
             title = {stingray: A modern Python library for spectral timing},
             journal = {Journal of Open Source Software}
           }`;
           const el = document.createElement('textarea');
           el.value = bibtex;
           document.body.appendChild(el);
           el.select();
           document.execCommand('copy');
           document.body.removeChild(el);
       }

   </script>

   <ul>
     <li>Huppenkothen et al., 2019. Astrophysical Journal, 881, 39.
         [<a href="https://doi.org/10.3847/1538-4357/ab258d">DOI</a>]
         [<a href="https://ui.adsabs.harvard.edu/abs/2019ApJ...881...39H">ADS</a>]
         [<a onclick="copyApjBib()">Copy BibTeX to clipboard</a>]</li>

     <li>Huppenkothen et al., 2019. Journal of Open Source Software, 4(38), 1393.
         [<a href="https://doi.org/10.21105/joss.01393">DOI</a>]
         [<a href="https://joss.theoj.org/papers/10.21105/joss.01393#">JOSS</a>]
         [<a onclick="copyJossBib()">Copy BibTeX to clipboard</a>]</li>
   </ul>

Other Useful References
=======================

.. raw:: html

   Stingray is listed in the <a href="https://ascl.net/1608.001">Astrophysics Source Code Library</a>.
   <a onclick="copyAsclBib()">Copy the corresponding BibTeX to clipboard</a>.

***************
Citing Stingray
***************

Citations are still the main currency of the academic world, and *the* best way to ensure that Stingray continues to be supported and we can continue to work on it.
If you use Stingray in data analysis leading to a publication, we ask that you cite *both* a `DOI <https://www.doi.org>`_, which points to the software itself, *and* our papers describing the Stingray project.

DOI
===

If possible, we ask that you cite a DOI corresponding to the specific version of Stingray that you used to carry out your analysis.

.. include:: _zenodo.rst

If this isn't possible — for example, because you worked with an unreleased version of the code — you can cite Stingray's `concept DOI <https://zenodo.org/help/versioning>`__, `10.5281/zenodo.1490116 <https://zenodo.org/records/1490116>`__ (`BibTeX <https://zenodo.org/records/1490116/export/hx>`__), which will always resolve to the latest release.

Papers
======

If you are using Stingray 2.0 or newer, please cite both of the following papers:

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
           var bibtex = `@article{bachettiStingrayFastModern2024,
            title = {Stingray 2: {{A}} Fast and Modern {{Python}} Library for Spectral Timing},
            shorttitle = {Stingray 2},
            author = {Bachetti, Matteo and Huppenkothen, Daniela and Stevens, Abigail and Swinbank, John and Mastroserio, Guglielmo and Lucchini, Matteo and Lai, Eleonora Veronica and Buchner, Johannes and Desai, Amogh and Joshi, Gaurav and Pisanu, Francesco and Pisupati, Sri Guru Datta and Sharma, Swapnil and Tripathi, Mihir and Vats, Dhruv},
            year = {2024},
            month = oct,
            journal = {Journal of Open Source Software},
            volume = {9},
            number = {102},
            pages = {7389},
            issn = {2475-9066},
            doi = {10.21105/joss.07389},
            urldate = {2024-10-25},
            abstract = {Bachetti et al., (2024). Stingray 2: A fast and modern Python library for spectral timing. Journal of Open Source Software, 9(102), 7389, https://doi.org/10.21105/joss.07389},
            langid = {english}
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

     <li>Bachetti et al., 2024. Journal of Open Source Software, 9(102), 7389.
         [<a href="https://doi.org/10.21105/joss.07389">DOI</a>]
         [<a href="https://joss.theoj.org/papers/10.21105/joss.07389#">JOSS</a>]
         [<a onclick="copyJossBib()">Copy BibTeX to clipboard</a>]</li>
   </ul>

Other Useful References
=======================

.. raw:: html

   <script type="text/javascript">
       function copyOldJossBib() {
           var bibtex = `@article{bachettiStingrayFastModern2024,
            title = {Stingray 2: {{A}} Fast and Modern {{Python}} Library for Spectral Timing},
            shorttitle = {Stingray 2},
            author = {Bachetti, Matteo and Huppenkothen, Daniela and Stevens, Abigail and Swinbank, John and Mastroserio, Guglielmo and Lucchini, Matteo and Lai, Eleonora Veronica and Buchner, Johannes and Desai, Amogh and Joshi, Gaurav and Pisanu, Francesco and Pisupati, Sri Guru Datta and Sharma, Swapnil and Tripathi, Mihir and Vats, Dhruv},
            year = {2024},
            month = oct,
            journal = {Journal of Open Source Software},
            volume = {9},
            number = {102},
            pages = {7389},
            issn = {2475-9066},
            doi = {10.21105/joss.07389},
            urldate = {2024-10-25},
            abstract = {Bachetti et al., (2024). Stingray 2: A fast and modern Python library for spectral timing. Journal of Open Source Software, 9(102), 7389, https://doi.org/10.21105/joss.07389},
            langid = {english}
           }`;
           const el = document.createElement('textarea');
           el.value = bibtex;
           document.body.appendChild(el);
           el.select();
           document.execCommand('copy');
           document.body.removeChild(el);
         }
       function copyAsclBib() {
            var bibtex = `@software{2016ascl.soft08001H,
              author = {{Huppenkothen}, Daniela and {Bachetti}, Matteo and {Stevens}, Abigail L. and {Migliari}, Simone and {Balm}, Paul},
              title = "{Stingray: Spectral-timing software}",
              howpublished = {Astrophysics Source Code Library, record ascl:1608.001},
              year = 2016,
              month = aug,
              eid = {ascl:1608.001},
              adsurl = {https://ui.adsabs.harvard.edu/abs/2016ascl.soft08001H},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }
           }`;
           const el = document.createElement('textarea');
           el.value = bibtex;
           document.body.appendChild(el);
           el.select();
           document.execCommand('copy');
           document.body.removeChild(el);
       }
      </script>
   Stingray is listed in the <a href="https://ascl.net/1608.001">Astrophysics Source Code Library</a>.
   <a onclick="copyAsclBib()">Copy the corresponding BibTeX to clipboard</a>.

   Our first JOSS paper, describing the development until 2019, is Huppenkothen et al. 2019b, "Stingray: a modern python library for spectral timing", <a href="https://joss.theoj.org/papers/10.21105/joss.01393">JOSS</a>; <a href="https://doi.org/10.21105/joss.01393">DOI: 10.21105/joss.01393'</a>.
   <a onclick="copyOldJossBib()">Copy the corresponding BibTeX to clipboard</a>.

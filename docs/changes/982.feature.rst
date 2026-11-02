The ``bispectrum`` module has been rewritten around the Fourier-decomposition
estimator (Maccarone 2013) and brought in line with the modern spectral-timing
classes such as ``Powerspectrum``. New user-facing capabilities include:

- ``CrossBispectrum`` and ``AveragedCrossBispectrum``, measuring quadratic phase
  coupling *between* channels (sampled on the full signed frequency plane), with
  ``Bispectrum`` recovered as the single-channel special case.
- ``DynamicalBispectrum`` and ``DynamicalCrossBispectrum`` for time-resolved
  (cross-)bispectra, with ``plot_diagonal``, ``plot_slice``, ``plot_frame``,
  ``plot_montage`` and ``plot_trace`` plotting options and a ``shift_and_add``
  method for tracking a drifting coupling.
- Three bicoherence normalizations -- Kim & Powers (the default, squared form),
  Sigl & Chamoun, and Hagihira -- switchable after the fact with
  ``recompute_bicoherence`` without recomputing the FFTs.
- Poisson-noise bias subtraction (Wirnitzer 1985) via ``poisson_subtract``,
  overlap-aware for the cross case through ``channels_overlap``.
- The biphase and its circular-statistics (Fisher 1993) uncertainty.
- Jellyfish plots (``plot_jellyfish``), plus ``plot_mag``, ``plot_phase`` and
  ``plot_bicoherence``, all returning their ``matplotlib`` axes.
- Construction from light curves, event lists, arrays of event times,
  ``StingrayTimeseries`` objects and iterables of light curves.
- ``save_all`` and ``save_diagonal`` options to control the memory used by the
  per-segment bispectrum cube.

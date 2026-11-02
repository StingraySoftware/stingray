The default bispectrum estimator is now the direct Fourier-decomposition method
(Maccarone 2013) rather than the indirect third-order-cumulant method. The
cumulant estimator is retained and available via ``Bispectrum(..., method="cumulant")``
and ``AveragedBispectrum(..., method="cumulant")`` (with the ``maxlag``,
``window`` and ``scale`` options), but it is no longer the default and its
results are no longer returned by default. Code relying on the old default
behaviour or on the previous standalone ``Bispectrum`` attributes must be updated
to select ``method="cumulant"`` and/or the new ``Bispectrum`` /
``AveragedBispectrum`` interface.

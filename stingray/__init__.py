# Licensed under MIT license - see LICENSE.rst

try:
    from ._version import version as __version__
except ImportError:
    # Written by setuptools_scm at build time, so absent in a source tree that
    # has never been installed.
    __version__ = ""

# Only the main classes are available directly as ``stingray.<name>``. Functions
# are imported from their modules, e.g. ``from stingray.fourier import ...``.
from stingray.base import (
    StingrayObject,
    StingrayTimeseries,
)
from stingray.events import (
    EventList,
)
from stingray.lightcurve import (
    Lightcurve,
)
from stingray.powerspectrum import (
    Powerspectrum,
    AveragedPowerspectrum,
    DynamicalPowerspectrum,
    GtiCorrPowerspectrum,
)
from stingray.crossspectrum import (
    Crossspectrum,
    AveragedCrossspectrum,
    DynamicalCrossspectrum,
)
from stingray.lombscargle import (
    LombScarglePowerspectrum,
    LombScargleCrossspectrum,
)
from stingray.multitaper import (
    Multitaper,
)
from stingray.covariancespectrum import (
    Covariancespectrum,
    AveragedCovariancespectrum,
)
from stingray.crosscorrelation import (
    CrossCorrelation,
    AutoCorrelation,
)
from stingray.bispectrum import (
    CrossBispectrum,
    AveragedCrossBispectrum,
    Bispectrum,
    AveragedBispectrum,
    DynamicalCrossBispectrum,
    DynamicalBispectrum,
)
from stingray.varenergyspectrum import (
    VarEnergySpectrum,
    RmsEnergySpectrum,
    RmsSpectrum,
    LagEnergySpectrum,
    LagSpectrum,
    ExcessVarianceSpectrum,
    CovarianceSpectrum,
    ComplexCovarianceSpectrum,
    CountSpectrum,
)
from stingray.exceptions import (
    StingrayError,
    StingrayWarning,
)

__all__ = [
    "StingrayObject",
    "StingrayTimeseries",
    "EventList",
    "Lightcurve",
    "Powerspectrum",
    "AveragedPowerspectrum",
    "DynamicalPowerspectrum",
    "GtiCorrPowerspectrum",
    "Crossspectrum",
    "AveragedCrossspectrum",
    "DynamicalCrossspectrum",
    "LombScarglePowerspectrum",
    "LombScargleCrossspectrum",
    "Multitaper",
    "Covariancespectrum",
    "AveragedCovariancespectrum",
    "CrossCorrelation",
    "AutoCorrelation",
    "CrossBispectrum",
    "AveragedCrossBispectrum",
    "Bispectrum",
    "AveragedBispectrum",
    "DynamicalCrossBispectrum",
    "DynamicalBispectrum",
    "VarEnergySpectrum",
    "RmsEnergySpectrum",
    "RmsSpectrum",
    "LagEnergySpectrum",
    "LagSpectrum",
    "ExcessVarianceSpectrum",
    "CovarianceSpectrum",
    "ComplexCovarianceSpectrum",
    "CountSpectrum",
    "StingrayError",
    "StingrayWarning",
]

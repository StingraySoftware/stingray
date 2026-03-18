# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ["__version__"]

# this indicates whether or not we are in the package's setup.py
try:
    _ASTROPY_SETUP_
except NameError:
    import builtins

    builtins._ASTROPY_SETUP_ = False

try:
    from .version import version as __version__
except ImportError:
    __version__ = ""

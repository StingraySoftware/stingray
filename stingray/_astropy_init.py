# Licensed under MIT license - see LICENSE.rst

__all__ = ["__version__"]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = ""

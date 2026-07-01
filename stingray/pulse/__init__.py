# Licensed under MIT license - see LICENSE.rst

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from .._astropy_init import *

# ----------------------------------------------------------------------------

try:
    import pint.toa as toa
    import pint
    from pint.models import get_model

    HAS_PINT = True
except ImportError:
    HAS_PINT = False
    get_model = toa = None


# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from stingray.pulse.pulsar import *
    from stingray.pulse.search import *
    from stingray.pulse.modeling import *
    from .periodogram import *

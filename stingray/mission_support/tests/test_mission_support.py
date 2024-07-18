import pytest
import os
from astropy.io import fits
from stingray.mission_support import (
    rxte_pca_event_file_interpretation,
    get_rough_conversion_function,
)

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "..", "..", "tests", "data")


class TestAll(object):
    def test_wrong_mission_raises(self):
        with pytest.raises(ValueError, match="Mission blah not recognized"):
            get_rough_conversion_function("blah")


class TestXTE(object):
    @classmethod
    def setup_class(cls):
        cls.wrongfile = os.path.join(datadir, "monol_testA.evt")

    def test_wrong_file_raises(self):
        with pytest.raises(ValueError, match="No XTE_SE extension found."):
            rxte_pca_event_file_interpretation(fits.open(self.wrongfile))

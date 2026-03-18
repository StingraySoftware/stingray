import pytest
import os
import numpy as np
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

    def test_rxte_pca_event_file_interpretation(self):
        filename = os.path.join(datadir, "xte_test.evt.gz")
        unchanged_hdulist = fits.open(filename)
        assert np.min(unchanged_hdulist["XTE_SE"].data["PHA"]) == 0
        assert np.max(unchanged_hdulist["XTE_SE"].data["PHA"]) == 60

        first_new_hdulist = rxte_pca_event_file_interpretation(filename)

        second_new_hdulist = rxte_pca_event_file_interpretation(fits.open(filename))

        new_hdu = rxte_pca_event_file_interpretation(fits.open(filename)["XTE_SE"])
        new_data = rxte_pca_event_file_interpretation(
            fits.open(filename)["XTE_SE"].data, header=fits.open(filename)["XTE_SE"].header
        )
        new_data_hdr_str = rxte_pca_event_file_interpretation(
            fits.open(filename)["XTE_SE"].data,
            header=fits.open(filename)["XTE_SE"].header.tostring(),
        )

        for data in (
            new_data,
            new_data_hdr_str,
            new_hdu.data,
            first_new_hdulist["XTE_SE"].data,
            second_new_hdulist["XTE_SE"].data,
        ):
            assert np.min(data["PHA"]) == 2
            assert np.max(data["PHA"]) == 221

import pytest
import os
import numpy as np
import scipy.stats
from stingray import bexvar
from astropy.table import Table
from astropy.io import fits
import signal

pytestmark = pytest.mark.slow


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


_HAS_ULTRANEST = True

try:
    import ultranest
except ImportError:
    _HAS_ULTRANEST = False

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


class TestBexvarResult(object):
    @classmethod
    def setup_class(cls):
        fname_data = os.path.join(datadir, "LightCurve_bexvar.fits")
        lightcurve = Table.read(fname_data, hdu="RATE", format="fits")
        band = 0

        cls.time = lightcurve["TIME"] - lightcurve["TIME"][0]
        cls.time_delta = lightcurve["TIMEDEL"]
        cls.bg_counts = lightcurve["BACK_COUNTS"][:, band]
        cls.src_counts = lightcurve["COUNTS"][:, band]
        cls.bg_ratio = lightcurve["BACKRATIO"]
        cls.frac_exp = lightcurve["FRACEXP"][:, band]

        cls.fname_result = os.path.join(datadir, "bexvar_results_band_0.npy")
        cls.quantile = scipy.stats.norm().cdf([-1])

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_bexvar(self):
        log_cr_sigma_from_function = bexvar.bexvar(
            self.time,
            self.time_delta,
            self.src_counts,
            self.bg_counts,
            self.bg_ratio,
            self.frac_exp,
        )
        log_cr_sigma_result = np.load(self.fname_result, allow_pickle=True)[1]

        scatt_lo_function = scipy.stats.mstats.mquantiles(log_cr_sigma_from_function, self.quantile)
        scatt_lo_result = scipy.stats.mstats.mquantiles(log_cr_sigma_result, self.quantile)

        # Compares lower 1 sigma quantile of the estimated scatter of the log(count rate) in dex
        assert np.isclose(scatt_lo_function, scatt_lo_result, rtol=0.1)

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_if_bg_counts_none(self):
        log_cr_sigma = bexvar.bexvar(
            time=self.time,
            time_del=self.time_delta,
            src_counts=self.src_counts,
            bg_counts=None,
            bg_ratio=self.bg_ratio,
            frac_exp=self.frac_exp,
        )

        scatt_lo = scipy.stats.mstats.mquantiles(log_cr_sigma, self.quantile)
        assert np.isclose(scatt_lo, 0.0143, rtol=0.1)

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_if_bg_ratio_none(self):
        log_cr_sigma = bexvar.bexvar(
            time=self.time,
            time_del=self.time_delta,
            src_counts=self.src_counts,
            bg_counts=self.bg_counts,
            bg_ratio=None,
            frac_exp=self.frac_exp,
        )

        scatt_lo = scipy.stats.mstats.mquantiles(log_cr_sigma, self.quantile)
        assert np.isclose(scatt_lo, 0.0106, rtol=0.1)

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_if_frac_exp_none(self):
        log_cr_sigma = bexvar.bexvar(
            time=self.time,
            time_del=self.time_delta,
            src_counts=self.src_counts,
            bg_counts=self.bg_counts,
            bg_ratio=self.bg_ratio,
            frac_exp=None,
        )

        scatt_lo = scipy.stats.mstats.mquantiles(log_cr_sigma, self.quantile)
        assert np.isclose(scatt_lo, 0.0100, rtol=0.1)

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_if_all_optional_param_none(self):
        log_cr_sigma = bexvar.bexvar(
            time=self.time,
            time_del=self.time_delta,
            src_counts=self.src_counts,
            bg_counts=None,
            bg_ratio=None,
            frac_exp=None,
        )

        scatt_lo = scipy.stats.mstats.mquantiles(log_cr_sigma, self.quantile)
        assert np.isclose(scatt_lo, 0.0100, rtol=0.1)


class TestInternalFunctions(object):
    @classmethod
    def setup_class(cls):
        fname_data = os.path.join(datadir, "LightCurve_bexvar.fits")
        lightcurve = Table.read(fname_data, hdu="RATE", format="fits")
        band = 0

        lightcurve = lightcurve[lightcurve["FRACEXP"][:, band] > 0.1]
        cls.time_delta = lightcurve["TIMEDEL"]
        cls.bg_counts = lightcurve["BACK_COUNTS"][:, band]
        cls.src_counts = lightcurve["COUNTS"][:, band]
        cls.bg_ratio = lightcurve["BACKRATIO"]
        cls.frac_exp = lightcurve["FRACEXP"][:, band]
        cls.bg_area = 1.0 / cls.bg_ratio
        cls.rate_conversion = cls.frac_exp * cls.time_delta

        fname_result = os.path.join(datadir, "bexvar_results_band_0.npy")
        cls.function_result = np.load(fname_result, allow_pickle=True)

    def test_lscg_gen(self):
        log_src_crs_grid_from_function = bexvar._lscg_gen(
            self.src_counts, self.bg_counts, self.bg_area, self.rate_conversion, 100
        )
        log_src_crs_grid_result = self.function_result[0]

        assert np.allclose(log_src_crs_grid_from_function, log_src_crs_grid_result)

    def test_estimate_source_cr_marginalised(self):
        log_src_crs_grid = self.function_result[0]
        weights_from_function = bexvar._estimate_source_cr_marginalised(
            log_src_crs_grid,
            self.src_counts[0],
            self.bg_counts[0],
            self.bg_area[0],
            self.rate_conversion[0],
        )
        weights_from_results = self.function_result[2][0]

        assert np.allclose(weights_from_function, weights_from_results)

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_calculate_bexvar(self):
        log_src_crs_grid = self.function_result[0]
        pdfs = self.function_result[2]

        posterior_log_sigma_src_cr_from_function = bexvar._calculate_bexvar(log_src_crs_grid, pdfs)
        posterior_log_sigma_src_cr_results = self.function_result[1]

        quantile = scipy.stats.norm().cdf([-1])
        scatt_lo_function = scipy.stats.mstats.mquantiles(
            posterior_log_sigma_src_cr_from_function, quantile
        )
        scatt_lo_result = scipy.stats.mstats.mquantiles(
            posterior_log_sigma_src_cr_results, quantile
        )

        # Compares lower 1 sigma quantile of the estimated scatter of the log(count rate) in dex

        assert np.isclose(scatt_lo_function, scatt_lo_result, rtol=0.1)

    @pytest.mark.skipif("_HAS_ULTRANEST")
    def test_ultranest_not_installed(self):
        with pytest.raises(ImportError) as excinfo:
            log_src_crs_grid = self.function_result[0]
            pdfs = self.function_result[2]

            _ = bexvar._calculate_bexvar(log_src_crs_grid, pdfs)
        assert "ultranest not installed! Can't sample!" in str(excinfo.value)


class TestBadValues(object):
    @classmethod
    def setup_class(cls):
        fname_data = os.path.join(datadir, "lcurveA.fits")
        hdul = fits.open(fname_data)[1]
        lightcurve = Table.read(hdul, format="fits")

        cls.time = lightcurve["TIME"]
        time_delta = np.diff(cls.time)
        cls.time_delta = np.append(time_delta, float(1.0))
        cls.src_count = np.array(lightcurve["RATE1"] * cls.time_delta)
        cls.frac_exp = lightcurve["FRACEXP"]

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_non_integer_src_counts_warning(self):
        with pytest.warns(UserWarning) as record:
            signal.signal(signal.SIGALRM, timeout_handler)

            signal.alarm(5)
            try:
                _ = bexvar.bexvar(
                    time=self.time,
                    time_del=self.time_delta,
                    src_counts=self.src_count,
                    frac_exp=self.frac_exp,
                )
            except TimeoutException:
                print("function terminated")

            assert any(
                ["src_counts are not all positive integers" in r.message.args[0] for r in record]
            )

    def test_weights_sum_warning(self):
        with pytest.warns(UserWarning) as record:
            _ = bexvar._estimate_source_cr_marginalised(
                log_src_crs_grid=[2.0, 2.1],
                src_counts=3.0,
                bkg_counts=1.0,
                bkg_area=np.inf,
                rate_conversion=0,
            )
        assert any(["Weight problem! sum is <= 0" in r.message.args[0] for r in record])

import numpy as np
from stingray.pulse import fold_events, get_TOA, phase_exposure
from stingray.pulse import ef_profile_stat, z_n, pulse_phase
from stingray.pulse import pdm_profile_stat
from stingray.pulse import z_n_events, z_n_binned_events, z_n_gauss, htest
from stingray.pulse import z_n_events_all, z_n_binned_events_all, z_n_gauss_all
from stingray.pulse import get_orbital_correction_from_ephemeris_file, p_to_f
from stingray.pulse import HAS_PINT
import pytest
import os
import warnings
import matplotlib.pyplot as plt


def _template_fun(phase, ph0, amplitude, baseline=0):
    return baseline + amplitude * np.cos((phase - ph0) * 2 * np.pi)


def test_p_to_f_integer_input():
    # Integer periods must not truncate the frequencies
    assert np.allclose(p_to_f(2), [0.5])
    assert np.allclose(p_to_f(2, 1), [0.5, -0.25])
    # long double inputs keep their precision
    assert p_to_f(np.longdouble(2), 1).dtype == np.longdouble


def test_p_to_f_warns():
    with pytest.warns(UserWarning, match="Derivatives above third are not supported"):
        assert np.allclose(p_to_f(1, 2, 3, 4, 32, 22), [1, -2, 5, -16, 0, 0])


class TestAll(object):
    """Unit tests for the stingray.pulsar module."""

    @classmethod
    def setup_class(cls):
        cls.curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(cls.curdir, "data")

    @pytest.mark.slow
    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_pint_installed_correctly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)

            import pint.toa as toa
            from pint.residuals import Residuals
            import pint.models.model_builder as mb
            import astropy.units as u

            parfile = os.path.join(self.datadir, "example_pint.par")
            timfile = os.path.join(self.datadir, "example_pint.tim")

            toas = toa.get_TOAs(timfile, ephem="DE405", planets=False, include_bipm=False)
            model = mb.get_model(parfile)

            pint_resids_us = Residuals(toas, model).time_resids.to(u.s)

        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(np.abs(pint_resids_us.value) < 3e-6)

    @pytest.mark.slow
    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_orbit_from_parfile(self):
        import pint.toa as toa

        parfile = os.path.join(self.datadir, "example_pint.par")
        timfile = os.path.join(self.datadir, "example_pint.tim")

        toas = toa.get_TOAs(timfile, ephem="DE405", planets=False, include_bipm=False)
        mjds = np.array([m.value for m in toas.get_mjds(high_precision=True)])

        mjdstart, mjdstop = mjds[0] - 1, mjds[-1] + 1

        with pytest.warns(UserWarning, match="Assuming events are already referred to "):
            correction_sec, correction_mjd, model = get_orbital_correction_from_ephemeris_file(
                mjdstart, mjdstop, parfile, ntimes=1000, return_pint_model=True
            )

        mjdref = 50000
        toa_sec = (mjds - mjdref) * 86400
        corr = correction_mjd(mjds)
        corr_s = correction_sec(toa_sec, mjdref)
        assert np.allclose(corr, corr_s / 86400 + mjdref)

    @pytest.mark.skipif("HAS_PINT")
    def test_orbit_from_parfile_raises(self):
        print("Doesn't have pint")
        with pytest.raises(ImportError):
            get_orbital_correction_from_ephemeris_file(0, 0, parfile="ciaociao")

    def test_stat(self):
        """Test pulse phase calculation, frequency only."""
        prof = np.array([2, 2, 2, 2])
        np.testing.assert_array_almost_equal(ef_profile_stat(prof), 0)

    def test_pdm_stat(self):
        """Test pulse phase calculation, frequency only."""
        prof = np.array([1, 1, 1, 1, 1])
        sample_var = 2
        nsample = 10
        sum_dev2 = sample_var * (nsample - 1)
        np.testing.assert_array_almost_equal(pdm_profile_stat(prof, sum_dev2, nsample), 0.5)

    def test_weighted_pdm_stat(self):
        """Test the PDM calculation with weighted events (variances)."""
        period = 1.0
        nbin = 5

        # Constant signal: flux=10 everywhere. The pdm should be 0.
        times = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        fluxes = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        std_var = (
            0.1  # instead of 1.0, to check that avoiding division by zero isn't changing results
        )
        variances_base = np.ones_like(fluxes) * std_var

        _, profile_base, _ = fold_events(
            times, 1 / period, nbin=nbin, weights=fluxes, mode="pdm", ref_time=0
        )
        assert np.allclose(profile_base, 0.0)  # equal with the mean at every bin

        # An outlier point at 0.15 (Bin 0) with flux 1000
        times_out = np.append(times, 0.15)
        fluxes_out = np.append(fluxes, 1000.0)

        # High variance for the outlier, standard for the rest
        outlier_var = 1.0e6
        variances = np.append(variances_base, outlier_var)

        # Unweighted PDM on dirty data (should be skewed)
        _, profile_unweighted, _ = fold_events(
            times_out, 1 / period, nbin=nbin, weights=fluxes_out, mode="pdm", ref_time=0
        )
        assert profile_unweighted[0] > 1000.0  # the profile bin should deviate from 0

        # Weighted PDM on dirty data (should effectively ignore outlier)
        _, profile_weighted, _ = fold_events(
            times_out,
            1 / period,
            nbin=nbin,
            weights=fluxes_out,
            variances=variances,
            mode="pdm",
            ref_time=0,
        )
        # With high variance, the outlier's weight is tiny.
        # The bin statistics should be dominated by the point with flux 10.
        # The result should be very close to the baseline (0) and smaller than the unweighted result.
        assert np.all(profile_weighted <= profile_unweighted)
        assert np.allclose(profile_weighted, 0.0, atol=2.0)

    def test_zn(self):
        """Test pulse phase calculation, frequency only."""
        ph = np.array([0, 1])
        np.testing.assert_array_almost_equal(z_n(ph, 2), 8)
        ph = np.array([])
        np.testing.assert_array_almost_equal(z_n(ph, 2), 0)
        ph = np.array([0.2, 0.7])
        ph2 = np.array([0, 0.5])
        np.testing.assert_array_almost_equal(z_n(ph, 2), z_n(ph2, 2))

    def test_pulse_phase1(self):
        """Test pulse phase calculation, frequency only."""
        times = np.arange(0, 4, 0.5)
        ph = pulse_phase(times, 1, ph0=0, to_1=False)
        np.testing.assert_array_almost_equal(ph, times)

    def test_pulse_phase2(self):
        """Test pulse phase calculation, fdot only."""
        times = np.arange(0, 4, 0.5)
        ph = pulse_phase(times, 0, 1, ph0=0, to_1=False)
        np.testing.assert_array_almost_equal(ph, 0.5 * times**2)

    def test_pulse_phase3(self):
        """Test pulse phase calculation, fddot only."""
        times = np.arange(0, 4, 0.5)
        ph = pulse_phase(times, 0, 0, 1, ph0=0, to_1=False)
        np.testing.assert_array_almost_equal(ph, 1 / 6 * times**3)

    def test_phase_exposure1(self):
        start_time = 0
        stop_time = 1
        period = 1
        nbin = 16
        expo = phase_exposure(start_time, stop_time, period, nbin)
        np.testing.assert_array_almost_equal(expo, np.ones(nbin))

    def test_phase_exposure2(self):
        start_time = 0
        stop_time = 0.5
        period = 1
        nbin = 16
        expo = phase_exposure(start_time, stop_time, period, nbin)
        expected = np.ones(nbin)
        expected[nbin // 2 :] = 0
        np.testing.assert_array_almost_equal(expo, expected)

    def test_phase_exposure3(self):
        start_time = 0
        stop_time = 1
        gti = np.array([[0, 0.5]])
        period = 1
        nbin = 16
        expo = phase_exposure(start_time, stop_time, period, nbin, gti=gti)
        expected = np.ones(nbin)
        expected[nbin // 2 :] = 0
        np.testing.assert_array_almost_equal(expo, expected)

    def test_phase_exposure4(self):
        start_time = 0
        stop_time = 1
        gti = np.array([[-0.2, 1.2]])
        period = 1
        nbin = 16
        expo = phase_exposure(start_time, stop_time, period, nbin, gti=gti)
        expected = np.ones(nbin)
        np.testing.assert_array_almost_equal(expo, expected)

    def test_pulse_profile1(self):
        nbin = 16
        times = np.arange(0, 1, 1 / nbin)

        period = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ph, p, pe = fold_events(times, 1, nbin=nbin)

        np.testing.assert_array_almost_equal(p, np.ones(nbin))
        np.testing.assert_array_almost_equal(ph, np.arange(nbin) / nbin + 0.5 / nbin)
        np.testing.assert_array_almost_equal(pe, np.ones(nbin))

    def test_pulse_profile2(self):
        nbin = 16
        dt = 1 / nbin
        times = np.arange(0, 2, dt)
        gti = np.array([[-0.5 * dt, 2 + 0.5 * dt]])

        period = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ph, p, pe = fold_events(times, 1, nbin=nbin, expocorr=True, gti=gti)

        np.testing.assert_array_almost_equal(ph, np.arange(nbin) / nbin + 0.5 / nbin)
        np.testing.assert_array_almost_equal(p, 2 * np.ones(nbin))
        np.testing.assert_array_almost_equal(pe, 2**0.5 * np.ones(nbin))

    def test_pulse_profile3(self):
        nbin = 16
        dt = 1 / nbin
        times = np.arange(0, 2 - dt, dt)
        gti = np.array([[-0.5 * dt, 2 - dt]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ph, p, pe = fold_events(times, 1, nbin=nbin, expocorr=True, gti=gti)

        np.testing.assert_array_almost_equal(ph, np.arange(nbin) / nbin + 0.5 / nbin)
        np.testing.assert_array_almost_equal(p, 2 * np.ones(nbin))
        expected_err = 2**0.5 * np.ones(nbin)
        expected_err[-1] = 2  # Because of the change of exposure
        np.testing.assert_array_almost_equal(pe, expected_err)

    def test_pulse_profile_err_default_is_poisson(self):
        """Without variances, fold_events treats weights as Poisson counts."""
        rng = np.random.default_rng(0)
        nbin = 8
        times = np.arange(0, 4, 1 / nbin) + 0.5 / nbin
        counts = rng.poisson(50, times.size)
        _, prof, err = fold_events(times, 1, nbin=nbin, weights=counts)
        _, prof_v, err_v = fold_events(times, 1, nbin=nbin, weights=counts, variances=counts)
        np.testing.assert_allclose(err, np.sqrt(prof))
        np.testing.assert_allclose(prof_v, prof)
        np.testing.assert_allclose(err_v, err)

    def test_pulse_profile_err_photon_weights(self):
        """With photon weights and variances=w**2, bin errors equal sqrt(sum(w**2))."""
        rng = np.random.default_rng(1)
        nbin = 16
        times = np.sort(rng.uniform(0, 100, 5000))
        weights = rng.uniform(0, 1, times.size)
        phases = times - np.floor(times)
        bin_idx = np.floor(phases * nbin).astype(int)

        _, prof, err = fold_events(times, 1, nbin=nbin, weights=weights, variances=weights**2)

        np.testing.assert_allclose(prof, np.bincount(bin_idx, weights, minlength=nbin))
        np.testing.assert_allclose(err, np.sqrt(np.bincount(bin_idx, weights**2, minlength=nbin)))
        # The old estimate, sqrt(sum(w)), overestimates the error for w < 1
        assert np.all(err < np.sqrt(prof))

    def test_pulse_profile_err_scalar_variance(self):
        """A scalar variance is broadcast to every event before summing per bin."""
        nbin = 4
        times = np.arange(0, 2, 1 / nbin) + 0.5 / nbin
        flux = np.ones(times.size) * 3.0
        _, prof, err = fold_events(times, 1, nbin=nbin, weights=flux, variances=0.25)
        # Two points per bin, each with variance 0.25
        np.testing.assert_allclose(prof, 6)
        np.testing.assert_allclose(err, np.sqrt(0.5))

    def test_pulse_profile_err_expocorr_with_variances(self):
        """Exposure correction scales weighted profile errors along with the profile."""
        nbin = 16
        dt = 1 / nbin
        times = np.arange(0, 2, dt)
        gti = np.array([[-0.5 * dt, 2 + 0.5 * dt]])
        weights = np.full(times.size, 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            _, p, pe = fold_events(
                times, 1, nbin=nbin, expocorr=True, gti=gti, weights=weights, variances=weights**2
            )
        np.testing.assert_array_almost_equal(p, np.ones(nbin))
        np.testing.assert_array_almost_equal(pe, np.sqrt(0.5) * np.ones(nbin))

    @pytest.mark.parametrize(
        "variances,match",
        [
            ([1.0, 2.0], "one element per event"),
            (np.nan, "must be finite"),
            (-0.5, "must be non-negative"),
        ],
    )
    def test_pulse_profile_bad_variances(self, variances, match):
        """Invalid variances arrays raise ValueError in fold_events."""
        times = np.arange(0, 2, 0.125)
        with pytest.raises(ValueError, match=match):
            fold_events(times, 1, nbin=4, variances=variances)

    def test_pulse_profile_pdm(self):
        period = 0.237
        nbin = 10
        phases = np.array([0.05, 1.05])
        times = phases * period
        counts = [3, 5]
        _, profile, prof_err = fold_events(
            times, 1 / period, nbin=nbin, weights=counts, mode="pdm", ref_time=0
        )
        assert np.all(prof_err == 0)
        _, profile_ef, _ = fold_events(
            times, 1 / period, nbin=nbin, weights=counts, mode="ef", ref_time=0
        )
        for pdm, ef in zip(profile, profile_ef):
            if ef == 0:
                assert np.isnan(pdm)
            else:
                assert pdm == 2
                assert ef == 8

    def test_mode_incorrect(self):
        nbin = 16
        dt = 1 / nbin
        times = np.arange(0, 2 - dt, dt)
        counts = np.random.normal(3, 0.5, size=len(times))
        gti = np.array([[-0.5 * dt, 2 - dt]])

        wrong_mode = "blarg"
        with pytest.raises(ValueError) as excinfo:
            fold_events(times, 1, nbin=nbin, weights=counts, mode=wrong_mode)

    def test_pdm_fails_without_weights(self):
        nbin = 16
        dt = 1 / nbin
        times = np.arange(0, 2 - dt, dt)
        with pytest.raises(ValueError) as excinfo:
            fold_events(times, 1, nbin=nbin, mode="pdm")

    def test_zn_2(self):
        np.testing.assert_almost_equal(z_n(np.arange(1), n=1), 2)
        np.testing.assert_almost_equal(z_n(np.arange(1), n=2), 4)
        np.testing.assert_almost_equal(z_n(np.arange(2), n=2), 8)
        np.testing.assert_almost_equal(z_n(np.arange(2) + 0.5, n=2), 8)

    def test_get_TOA1(self):
        np.random.seed(1234)
        period = 1.2
        tstart = 122
        start_phase = 0.2123
        phases = np.arange(0, 1, 1 / 32)
        template = _template_fun(phases, start_phase, 10, 20)
        prof = np.random.poisson(template)

        toa, toaerr = get_TOA(prof, period, tstart, template=_template_fun(phases, 0, 1, 0))

        real_toa = tstart + start_phase * period
        assert (real_toa >= toa - toaerr * 3) & (real_toa <= toa + toaerr * 3)

    def test_get_TOA2(self):
        np.random.seed(1234)
        period = 1.2
        tstart = 122
        start_phase = 0.2123
        phases = np.arange(0, 1, 1 / 32)
        template = _template_fun(phases, start_phase, 10, 20)
        prof = np.random.poisson(template)

        toa, toaerr = get_TOA(
            prof, period, tstart, template=_template_fun(phases, 0, 1, 0), nstep=200
        )

        real_toa = tstart + start_phase * period
        assert (real_toa >= toa - toaerr * 3) & (real_toa <= toa + toaerr * 3)

    def test_get_TOA3(self):
        np.random.seed(1234)
        period = 1.2
        tstart = 122
        start_phase = 0.2123
        phases = np.arange(0, 1, 1 / 32)
        template = _template_fun(phases, start_phase, 10, 20)
        prof = np.random.poisson(template)

        toa, toaerr = get_TOA(
            prof,
            period,
            tstart,
            quick=True,
            template=_template_fun(phases, 0, 1, 0),
            nstep=200,
            use_bootstrap=True,
        )

        real_toa = tstart + start_phase * period
        assert (real_toa >= toa - toaerr * 3) & (real_toa <= toa + toaerr * 3)

    def test_get_TOA4(self):
        np.random.seed(1234)
        period = 1.2
        tstart = 122
        start_phase = 0.2123
        phases = np.arange(0, 1, 1 / 32)
        template = _template_fun(phases, start_phase, 10, 20)
        prof = np.random.poisson(template)

        toa, toaerr = get_TOA(
            prof, period, tstart, template=_template_fun(phases, 0, 1, 0), nstep=200, debug=True
        )

        real_toa = tstart + start_phase * period
        assert (real_toa >= toa - toaerr * 3) & (real_toa <= toa + toaerr * 3)

    def test_get_TOA_notemplate(self):
        np.random.seed(1234)
        period = 1.2
        tstart = 122
        start_phase = 0.2123
        phases = np.arange(0, 1, 1 / 32)
        template = _template_fun(phases, start_phase, 10, 20)
        prof = np.random.poisson(template)

        toa, toaerr = get_TOA(prof, period, tstart, nstep=200, debug=True)

        real_toa = tstart + start_phase * period
        assert (real_toa >= toa - toaerr * 3) & (real_toa <= toa + toaerr * 3)


def create_pulsed_events(nevents, freq, t0=0, t1=1000, nback=0):
    from numpy.random import Generator, PCG64

    rg = Generator(PCG64())
    events = rg.normal(0.5, 0.1, nevents - nback)
    events = events - np.floor(events)

    if nback > 0:
        events = np.concatenate((events, rg.uniform(0, 1, nback)))
    pulse_no = rg.integers(0, np.rint((t1 - t0) * freq), nevents)
    events = np.sort(events + pulse_no)
    return t1 + events / freq


def poissonize_gaussian_profile(prof, err):
    lam = 10000
    factor = np.sqrt(lam) / err
    return (prof - prof.mean()) * factor + lam


class TestZandH(object):
    """Unit tests for the stingray.pulsar module."""

    @classmethod
    def setup_class(cls):
        nevents = 10000
        f = 1.2123
        cls.events = create_pulsed_events(nevents, f)
        phases = cls.events * f
        cls.phases = phases - np.floor(phases)

        cls.prof512, cls.bins = np.histogram(cls.phases, range=[0, 1], bins=512)

    def test_zn_events(self):
        phases = self.phases
        ks, ze = z_n_events_all(phases, nmax=10)
        m, h = htest(phases, nmax=10, datatype="events")
        assert np.isclose(h + 4 * m - 4, z_n(phases, n=m, datatype="events"))
        assert np.isclose(h + 4 * m - 4, ze[m - 1])

    def test_zn_poisson(self):
        phases = self.phases
        prof512, bins = self.prof512, self.bins
        ks, ze = z_n_events_all(phases, nmax=10)
        ksp, zp = z_n_binned_events_all(prof512, nmax=10)

        assert np.allclose(ze, zp, rtol=0.01)
        m, h = htest(prof512, datatype="binned", nmax=10)

        assert np.isclose(h + 4 * m - 4, z_n(prof512, n=m, datatype="binned"))
        assert np.isclose(h + 4 * m - 4, zp[m - 1])

    def test_zn_poisson_zeros(self):
        prof512 = np.zeros(512)
        ksp, zp = z_n_binned_events_all(prof512, nmax=10)
        assert np.allclose(zp, 0)

    def test_zn_gauss(self):
        nbin = 512
        dph = 1 / nbin
        err = 0.1
        ph = np.arange(-0.5 + dph / 2, 0.5, dph)
        prof = np.random.normal(np.exp(-(ph**2) / 2 / 0.1**2), err)

        prof_poiss = poissonize_gaussian_profile(prof, err)

        ksp, zp = z_n_binned_events_all(prof_poiss, nmax=10)
        ksg, zg = z_n_gauss_all(prof, err, nmax=10)
        assert np.allclose(zg, zp, rtol=0.01)

        mg, hg = htest(prof, err=err, datatype="gauss", nmax=10)
        mp, hp = htest(prof_poiss, datatype="binned", nmax=10)

        assert np.isclose(hg, hp)
        assert np.isclose(mg, mp)
        assert np.isclose(hg + 4 * mg - 4, z_n(prof, n=mg, err=err, datatype="gauss"))

    def test_wrong_args_H_datatype(self):
        with pytest.raises(ValueError) as excinfo:
            htest([1], 2, datatype="gibberish")
        assert "Unknown datatype requested for htest (gibberish)" in str(excinfo.value)

    def test_wrong_args_Z_datatype(self):
        with pytest.raises(ValueError) as excinfo:
            z_n([1], 2, datatype="gibberish")
        assert "Unknown datatype requested for Z_n (gibberish)" in str(excinfo.value)

    def test_wrong_args_H_gauss_noerr(self):
        with pytest.raises(ValueError) as excinfo:
            htest([1], 2, datatype="gauss")
        assert "If datatype='gauss', you need to " in str(excinfo.value)

    def test_wrong_args_Z_gauss_noerr(self):
        with pytest.raises(ValueError) as excinfo:
            z_n([1], 2, datatype="gauss")
        assert "If datatype='gauss', you need to " in str(excinfo.value)


def _direct_weighted_z2(phase, weights, nmax):
    """Brute-force implementation of the weighted Z^2_n (Kerr 2011)."""
    k = np.arange(1, nmax + 1)[:, np.newaxis]
    arg = 2 * np.pi * k * phase
    powers = np.sum(weights * np.cos(arg), axis=1) ** 2 + np.sum(weights * np.sin(arg), axis=1) ** 2
    return 2 / np.sum(weights**2) * np.cumsum(powers)


class TestWeightedZandH:
    """Unit tests for weighted Z^2_n and H-test statistics (Kerr 2011)."""

    @classmethod
    def setup_class(cls):
        rng = np.random.default_rng(20110304)
        cls.nsrc, cls.nback = 300, 3000
        source = rng.normal(0.3, 0.04, cls.nsrc) % 1
        background = rng.uniform(0, 1, cls.nback)
        cls.phases = np.concatenate([source, background])
        cls.weights = np.concatenate(
            [rng.uniform(0.6, 1.0, cls.nsrc), rng.uniform(0.0, 0.2, cls.nback)]
        )

    @pytest.mark.parametrize("nmax", [1, 2, 20, 50])
    def test_matches_direct_sum(self, nmax):
        """Harmonic recurrence matches the direct trigonometric sum up to nmax."""
        ks, z = z_n_events_all(self.phases, nmax=nmax, weights=self.weights)
        np.testing.assert_array_equal(ks, np.arange(1, nmax + 1))
        np.testing.assert_allclose(z, _direct_weighted_z2(self.phases, self.weights, nmax))

    def test_unweighted_matches_direct_sum(self):
        """Unweighted z_n_events_all matches the unit-weight direct sum."""
        _, z = z_n_events_all(self.phases, nmax=20)
        unit = np.ones(self.phases.size)
        np.testing.assert_allclose(z, _direct_weighted_z2(self.phases, unit, 20))

    def test_unit_weights_give_standard_statistic(self):
        """Unit weights reproduce the standard unweighted Z^2_n statistic."""
        _, z = z_n_events_all(self.phases, nmax=10)
        _, z_w = z_n_events_all(self.phases, nmax=10, weights=1)
        np.testing.assert_allclose(z_w, z)
        assert np.isclose(z_n(self.phases, 10, weights=np.ones(self.phases.size)), z[-1])

    def test_scale_invariance(self):
        """Rescaling all weights by a positive constant leaves Z^2_n unchanged."""
        _, z = z_n_events_all(self.phases, nmax=5, weights=self.weights)
        _, z_scaled = z_n_events_all(self.phases, nmax=5, weights=7.5 * self.weights)
        np.testing.assert_allclose(z_scaled, z)

    def test_zero_weights_drop_events(self):
        """Events with zero weight do not contribute to weighted Z^2_n."""
        _, z = z_n_events_all(self.phases[: self.nsrc], nmax=4)
        weights = np.zeros(self.phases.size)
        weights[: self.nsrc] = 1
        _, z_w = z_n_events_all(self.phases, nmax=4, weights=weights)
        np.testing.assert_allclose(z_w, z)

    def test_all_zero_weights(self):
        """All-zero weights yield zero Z^2_n without division by zero."""
        _, z = z_n_events_all(self.phases, nmax=4, weights=0)
        np.testing.assert_array_equal(z, 0)

    def test_weights_improve_detection(self):
        """Down-weighting background photons boosts both Z^2_n and H-test."""
        z_unw = z_n(self.phases, 2)
        z_w = z_n(self.phases, 2, weights=self.weights)
        assert z_w > 2 * z_unw

        m_unw, h_unw = htest(self.phases, datatype="events")
        m_w, h_w = htest(self.phases, datatype="events", weights=self.weights)
        assert h_w > 2 * h_unw

    def test_weighted_htest_consistent_with_zn(self):
        """Weighted htest maximizes Z^2_m - 4m + 4 over harmonics."""
        m, h = htest(self.phases, nmax=20, datatype="events", weights=self.weights)
        _, z = z_n_events_all(self.phases, nmax=20, weights=self.weights)
        assert np.isclose(h, np.max(z - 4 * np.arange(1, 21) + 4))
        assert np.isclose(h + 4 * m - 4, z_n(self.phases, m, weights=self.weights))

    def test_null_distribution_is_chi2(self):
        """Null weighted Z^2_1 follows chi^2_2 with mean 2 and variance 4."""
        rng = np.random.default_rng(7)
        zs = np.array(
            [z_n(rng.uniform(0, 1, 200), 1, weights=rng.uniform(0, 1, 200)) for _ in range(2000)]
        )
        assert np.isclose(np.mean(zs), 2, rtol=0.1)
        assert np.isclose(np.var(zs), 4, rtol=0.2)

    def test_binned_total_variance(self):
        """Binned Z^2_n and H-test with total_variance=sum(w**2) match unbinned weighted values."""
        nbin = 256
        total_var = np.sum(self.weights**2)
        profile, _ = np.histogram(self.phases, bins=nbin, range=[0, 1], weights=self.weights)
        _, z_b = z_n_binned_events_all(profile, 4, total_variance=total_var)
        _, z_e = z_n_events_all(self.phases, 4, weights=self.weights)
        np.testing.assert_allclose(z_b, z_e, rtol=0.01)

        assert np.isclose(z_n(profile, 4, datatype="binned", total_variance=total_var), z_b[-1])
        m_b, h_b = htest(profile, nmax=10, datatype="binned", total_variance=total_var)
        m_e, h_e = htest(self.phases, nmax=10, datatype="events", weights=self.weights)
        assert m_b == m_e
        assert np.isclose(h_b, h_e, rtol=0.01)

    def test_binned_default_total_variance(self):
        """Passing total_variance=sum(profile) matches the unweighted binned default."""
        profile, _ = np.histogram(self.phases, bins=64, range=[0, 1])
        _, z = z_n_binned_events_all(profile, 4)
        _, z_tv = z_n_binned_events_all(profile, 4, total_variance=profile.sum())
        np.testing.assert_allclose(z_tv, z)

    def test_binned_zero_total_variance(self):
        """A zero total_variance returns zeros without dividing by zero."""
        _, z = z_n_binned_events_all(np.zeros(16), 4, total_variance=0.0)
        np.testing.assert_array_equal(z, 0)

    @pytest.mark.parametrize(
        "total_variance,match",
        [
            (np.nan, "must be finite"),
            (np.inf, "must be finite"),
            (-1.0, "must be non-negative"),
            ([1.0, 2.0], "must be a scalar"),
        ],
    )
    def test_bad_total_variance(self, total_variance, match):
        """Non-finite, negative, or non-scalar total_variance raises ValueError."""
        with pytest.raises(ValueError, match=match):
            z_n_binned_events_all(np.ones(16), 4, total_variance=total_variance)

    @pytest.mark.parametrize("func", [z_n, htest])
    @pytest.mark.parametrize("datatype", ["binned", "gauss"])
    def test_weights_rejected_for_profiles(self, func, datatype):
        """Passing per-event weights with binned or gauss datatype raises ValueError."""
        with pytest.raises(ValueError, match="weights can only be used with datatype='events'"):
            func(np.ones(16), 2, datatype=datatype, err=1, weights=np.ones(16))

    @pytest.mark.parametrize("func", [z_n, htest])
    @pytest.mark.parametrize("datatype", ["events", "gauss"])
    def test_total_variance_rejected_for_non_binned(self, func, datatype):
        """Passing total_variance with events or gauss datatype raises ValueError."""
        with pytest.raises(
            ValueError, match="total_variance can only be used with datatype='binned'"
        ):
            func(np.ones(16), 2, datatype=datatype, err=1, total_variance=16.0)

    def test_bad_weights(self):
        """Mismatched length or non-finite weights raise ValueError."""
        with pytest.raises(ValueError, match="one element per event"):
            z_n_events_all(self.phases, 2, weights=[1, 2, 3])
        with pytest.raises(ValueError, match="must be finite"):
            z_n_events_all(self.phases, 2, weights=np.full(self.phases.size, np.inf))

import numpy as np
from stingray.pulse import fold_events, get_TOA, phase_exposure
from stingray.pulse import ef_profile_stat, z_n, pulse_phase
from stingray.pulse import pdm_profile_stat
from stingray.pulse import z_n, z_n_events, z_n_binned_events, z_n_gauss, htest
from stingray.pulse import z_n_events_all, z_n_binned_events_all, z_n_gauss_all
from stingray.pulse import get_orbital_correction_from_ephemeris_file, p_to_f
from stingray.pulse import HAS_PINT
import pytest
import os
import warnings
import matplotlib.pyplot as plt


def _template_fun(phase, ph0, amplitude, baseline=0):
    return baseline + amplitude * np.cos((phase - ph0) * 2 * np.pi)


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
                # PDM implementation avoids division by 0, instead returns 0
                assert pdm == 0
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
        with pytest.warns(DeprecationWarning) as record:
            np.testing.assert_almost_equal(z_n(np.arange(1), n=1, norm=1), 2)
            np.testing.assert_almost_equal(z_n(np.arange(1), n=2, norm=1), 4)
            np.testing.assert_almost_equal(z_n(np.arange(2), n=2, norm=1), 8)
            np.testing.assert_almost_equal(z_n(np.arange(2) + 0.5, n=2, norm=1), 8)

        assert np.any(
            ["The use of ``z_n(phase, norm=profile)``" in r.message.args[0] for r in record]
        )

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

    def test_deprecated_norm_use(self):
        prof512, bins = self.prof512, self.bins
        with pytest.warns(DeprecationWarning) as record:
            z = z_n(prof512, n=3, datatype="binned")
            z_dep = z_n(np.zeros(prof512.size), norm=prof512, n=3, datatype="binned")
            np.testing.assert_almost_equal(z, z_dep)

        assert np.any(
            ["The use of ``z_n(phase, norm=profile)``" in r.message.args[0] for r in record]
        )

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

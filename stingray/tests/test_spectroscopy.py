import numpy as np
import pytest
import stingray.spectroscopy as spec
from stingray.filters import Window1D
from stingray import (
    Lightcurve,
    AveragedPowerspectrum,
    AveragedCrossspectrum,
    Powerspectrum,
    Crossspectrum,
)
from stingray.utils import fft
from stingray.modeling import fit_lorentzians, fit_crossspectrum, fit_powerspectrum
from stingray.filters import Optimal1D, Window1D
from astropy.modeling import models
from astropy.table import Table, Column

from os import remove

np.random.seed(123412525)


def random_walk(n, step):
    """
    Parameters
    ----------
    n : int
        number of points
    step : float
        maximum step
    """

    return np.cumsum(np.random.uniform(-step, step, n))


def normalize_angle(angle):
    """Refer angle to 0-pi."""
    while angle < 0:
        angle += np.pi
    while angle > np.pi:
        angle -= np.pi
    return angle


def waveform_simple(phases, dph=np.pi / 5, dampl=2.0, initial_phase=np.pi / 3):
    """Phases are in radians!"""
    fine_phases = np.arange(0, 2 * np.pi, 0.002)
    fine_shape = np.cos(fine_phases) + 1 / dampl * np.cos(2 * (fine_phases + dph))
    mean = np.mean(fine_shape)
    ampl = (np.max(fine_shape) - np.min(fine_shape)) / 2

    shape = np.cos(phases + initial_phase) + 1 / dampl * np.cos(2 * (initial_phase + phases + dph))
    shape -= mean
    shape /= ampl

    return shape


def fake_qpo(
    t, f0=1.0, waveform=None, timescale=10, astep=0, phstep=0, waveform_opts=None, rms=0.1
):
    """
    Parameters
    ----------
    t : array
        Times at which the lightcurve is calculated
    f0: float
        mean frequency of the QPO
    waveform : function
        Function that accepts phases from 0 to pi as input and returns an
        waveform with mean 0 and total amplitude 2 (typically from -1 to 1).
        Default is ``np.sin``
    timescale : float
        In this timescale, expressed in seconds, the waveform changes its phase
        or amplitude or both by an entire step
    astep : float
        Steps of variation of amplitude
    phstep : float
        Steps of variation of phase
    """

    if waveform is None:
        waveform = np.sin

    n = len(t)
    phase = 2 * np.pi * f0 * t + random_walk(n, phstep / timescale)
    amp = 1 + random_walk(n, astep / timescale)
    qpo_lc = amp * (1 + rms * waveform(phase, **waveform_opts))

    return phase, qpo_lc


@pytest.mark.slow
@pytest.mark.xfail
class TestCCF(object):
    @classmethod
    def setup_class(cls):
        total_length = 16000
        f_qpo = 1.5
        cls.dt = 1 / f_qpo / 40
        approx_Q = 10
        q_len = approx_Q / f_qpo
        sigma = 0.1
        astep = 0.01
        phstep = 1
        real_dphi = 0.4 * np.pi
        cls.n_seconds = 500
        cls.n_seg = int(total_length / cls.n_seconds)
        cls.n_bins = cls.n_seconds / cls.dt

        times = np.arange(0, total_length, cls.dt)
        _, cls.ref_counts = fake_qpo(
            times,
            f0=f_qpo,
            astep=astep,
            rms=sigma,
            waveform=waveform_simple,
            phstep=phstep,
            timescale=q_len,
            waveform_opts={"dph": real_dphi},
        )
        _, ci_counts = fake_qpo(
            times,
            f0=f_qpo,
            astep=astep,
            rms=sigma,
            waveform=waveform_simple,
            phstep=phstep,
            timescale=q_len,
            waveform_opts={"dph": real_dphi},
        )
        cls.ci_counts = np.array([ci_counts])

        cls.ref_times = np.arange(0, cls.n_seconds * cls.n_seg, cls.dt)
        cls.ref_lc = Lightcurve(cls.ref_times, cls.ref_counts, dt=cls.dt)
        ref_aps = AveragedPowerspectrum(cls.ref_lc, segment_size=cls.n_seconds, norm="abs")
        df = ref_aps.freq[1] - ref_aps.freq[0]
        amplitude_0 = np.max(ref_aps.power)
        x_0_0 = ref_aps.freq[np.argmax(ref_aps.power)]
        amplitude_1 = amplitude_0 / 2
        x_0_1 = x_0_0 * 2
        fwhm = df

        cls.model = models.Lorentz1D(
            amplitude=amplitude_0, x_0=x_0_0, fwhm=fwhm
        ) + models.Lorentz1D(amplitude=amplitude_1, x_0=x_0_1, fwhm=fwhm)
        cls.ref_aps = ref_aps

    def test_ccf(self):
        # to make testing faster, fitting is not done.
        ref_ps = Powerspectrum(self.ref_lc, norm="abs")

        ci_counts_0 = self.ci_counts[0]
        ci_times = np.arange(0, self.n_seconds * self.n_seg, self.dt)
        ci_lc = Lightcurve(ci_times, ci_counts_0, dt=self.dt)

        # rebinning factor used in `rebin_log`
        rebin_log_factor = 0.4

        acs = AveragedCrossspectrum(
            data1=ci_lc,
            data2=self.ref_lc,
            segment_size=self.n_seconds,
            norm="leahy",
            power_type="absolute",
        )
        acs = acs.rebin_log(rebin_log_factor)

        # parest, res = fit_crossspectrum(acs, self.model, fitmethod="CG")
        acs_result_model = self.model

        # using optimal filter
        optimal_filter = Optimal1D(acs_result_model)
        optimal_filter_freq = optimal_filter(acs.freq)
        filtered_acs_power = optimal_filter_freq * np.abs(acs.power)

        # rebinning power spectrum
        new_df = spec.get_new_df(ref_ps, self.n_bins)
        ref_ps_rebinned = ref_ps.rebin(df=new_df)

        # parest, res = fit_powerspectrum(ref_ps_rebinned, self.model)
        ref_ps_rebinned_result_model = self.model

        # calculating rms from power spectrum
        ref_ps_rebinned_rms = spec.compute_rms(
            ref_ps_rebinned, ref_ps_rebinned_result_model, criteria="optimal"
        )
        print(ref_ps_rebinned_rms)

        # calculating normalized ccf
        ccf_norm = spec.ccf(filtered_acs_power, ref_ps_rebinned_rms, self.n_bins)
        print(ccf_norm)
        # calculating ccf error
        meta = {
            "N_SEG": self.n_seg,
            "NSECONDS": self.n_seconds,
            "DT": self.dt,
            "N_BINS": self.n_bins,
        }

        error_ccf, avg_seg_ccf = spec.ccf_error(
            self.ref_counts,
            ci_counts_0,
            acs_result_model,
            rebin_log_factor,
            meta,
            ref_ps_rebinned_rms,
            filter_type="optimal",
        )
        print(avg_seg_ccf)

        assert np.all(np.isclose(ccf_norm, avg_seg_ccf, atol=0.01))
        assert np.all(np.isclose(error_ccf, np.zeros(shape=error_ccf.shape), atol=0.01))

        # using window function
        tophat_filter = Window1D(acs_result_model)
        tophat_filter_freq = tophat_filter(acs.freq)
        filtered_acs_power = tophat_filter_freq * np.abs(acs.power)

        ref_ps_rebinned_rms = spec.compute_rms(
            ref_ps_rebinned, ref_ps_rebinned_result_model, criteria="window"
        )

        ccf_norm = spec.ccf(filtered_acs_power, ref_ps_rebinned_rms, self.n_bins)

        error_ccf, avg_seg_ccf = spec.ccf_error(
            self.ref_counts,
            ci_counts_0,
            acs_result_model,
            rebin_log_factor,
            meta,
            ref_ps_rebinned_rms,
            filter_type="window",
        )

        assert np.all(np.isclose(ccf_norm, avg_seg_ccf, atol=0.01))
        assert np.all(np.isclose(error_ccf, np.zeros(shape=error_ccf.shape), atol=0.01))


def test_load_lc_fits():
    dt = 0.1
    n_seg = 5
    n_seconds = 100
    output_file = "out.fits"

    ref_lc = np.arange(0, n_seconds, dt)
    ci_lc = np.array(np.array_split(np.arange(0, n_seconds * n_seg, dt), n_seg))

    n_bins = len(ref_lc)

    lightcurves = Table()
    lightcurves.add_column(Column(name="REF", data=ref_lc.T / dt))
    lightcurves.add_column(Column(name="CI", data=ci_lc.T / dt))
    lightcurves.meta["N_BINS"] = n_bins
    lightcurves.meta["DT"] = dt
    lightcurves.meta["N_SEG"] = n_seg
    lightcurves.meta["NSECONDS"] = n_seconds
    lightcurves.write(output_file, format="fits", overwrite=True)

    ref, ci, meta = spec.load_lc_fits(output_file, counts_type=False)
    remove(output_file)

    assert np.allclose(ref_lc, ref)
    assert np.allclose(ci_lc, ci)
    assert meta["N_BINS"] == n_bins
    assert meta["DT"] == dt
    assert meta["N_SEG"] == n_seg
    assert meta["NSECONDS"] == n_seconds


def test_psi_distance():
    a = np.arange(-10, 11)
    dist = spec.psi_distance(0, a)
    assert np.allclose(np.abs(a), dist)


def test_get_new_df():
    np.random.seed(150)

    amplitude_0 = 200.0
    amplitude_1 = 100.0
    amplitude_2 = 50.0

    x_0_0 = 0.5
    x_0_1 = 2.0
    x_0_2 = 7.5

    fwhm_0 = 0.1
    fwhm_1 = 1.0
    fwhm_2 = 0.5

    whitenoise = 100.0

    model = (
        models.Lorentz1D(amplitude_0, x_0_0, fwhm_0)
        + models.Lorentz1D(amplitude_1, x_0_1, fwhm_1)
        + models.Lorentz1D(amplitude_2, x_0_2, fwhm_2)
        + models.Const1D(whitenoise)
    )

    freq = np.linspace(0.01, 10.0, 1000)
    p = model(freq)
    noise = np.random.exponential(size=len(freq))

    power = p * noise
    cs = Crossspectrum()
    cs.freq = freq
    cs.power = power
    cs.df = cs.freq[1] - cs.freq[0]
    cs.n = len(freq)
    cs.m = 1

    assert np.isclose(cs.df, spec.get_new_df(cs, cs.n), rtol=0.001)


def test_waveform():
    mu = 1
    avg_sigma_1 = 1
    avg_sigma_2 = 2
    cap_phi_1 = np.pi / 2
    cap_phi_2 = np.pi / 2
    x = np.linspace(0, 2 * 2 * np.pi, 100)

    y1 = spec.waveform(x, mu, avg_sigma_1, avg_sigma_2, cap_phi_1, cap_phi_2)
    y2 = mu * (
        1
        + np.sqrt(2)
        * (avg_sigma_1 * np.cos(x - cap_phi_1) + avg_sigma_2 * np.cos(2 * x - cap_phi_2))
    )

    assert np.allclose(y1, y2)


def test_x_2_function():
    a = [-2, -1, 0, 1, 2]
    y1 = spec.x_2_function(0, a)
    y2 = np.sum(np.array(a) ** 2)
    assert np.allclose(y1, y2)


def test_compute_rms():
    np.random.seed(150)

    amplitude_0 = 200.0
    amplitude_1 = 100.0
    amplitude_2 = 50.0

    x_0_0 = 0.5
    x_0_1 = 2.0
    x_0_2 = 7.5

    fwhm_0 = 0.1
    fwhm_1 = 1.0
    fwhm_2 = 0.5

    whitenoise = 100.0

    model = (
        models.Lorentz1D(amplitude_0, x_0_0, fwhm_0)
        + models.Lorentz1D(amplitude_1, x_0_1, fwhm_1)
        + models.Lorentz1D(amplitude_2, x_0_2, fwhm_2)
        + models.Const1D(whitenoise)
    )

    freq = np.linspace(-10.0, 10.0, 1000)
    p = model(freq)
    noise = np.random.exponential(size=len(freq))

    power = p * noise
    cs = Crossspectrum()
    cs.freq = freq
    cs.power = power
    cs.df = cs.freq[1] - cs.freq[0]
    cs.n = len(freq)
    cs.m = 1

    rms = np.sqrt(np.sum(model(cs.freq) * cs.df)).mean()

    assert rms == spec.compute_rms(cs, model, criteria="all")

    rms_pos = np.sqrt(np.sum(model(cs.freq[cs.freq > 0]) * cs.df)).mean()

    assert rms_pos == spec.compute_rms(cs, model, criteria="posfreq")

    optimal_filter = Window1D(model)
    optimal_filter_freq = optimal_filter(cs.freq)
    filtered_cs_power = optimal_filter_freq * np.abs(model(cs.freq))

    rms = np.sqrt(np.sum(filtered_cs_power * cs.df)).mean()
    assert rms == spec.compute_rms(cs, model, criteria="window")

    with pytest.raises(ValueError):
        spec.compute_rms(cs, model, criteria="filter")


def test_get_mean_phase_difference():
    dt = 0.01
    time = np.arange(0, 100, dt)
    qpo_freq = 0.1
    harmonic_freq = 2 * qpo_freq
    counts_qpo = 100 * np.sin(2 * np.pi * qpo_freq * time - np.pi / 4)
    counts_harmonic = 50 * np.sin(2 * np.pi * harmonic_freq * time - np.pi / 4)
    counts = counts_qpo + counts_harmonic

    lc = Lightcurve(time, counts, dt=dt)
    cs = Crossspectrum(lc, lc)

    model = models.Lorentz1D(x_0=qpo_freq, amplitude=100) + models.Lorentz1D(
        x_0=harmonic_freq, amplitude=50
    )
    avg_psi, stddev = spec.get_mean_phase_difference(cs, model)

    assert np.isclose(avg_psi, np.pi / 2, 1e-5)
    assert np.isclose(stddev, np.pi / 2, 1e-5)


def test_get_phase_lag():
    dt = 0.01
    time = np.arange(0, 100, dt)
    qpo_freq = 0.1
    harmonic_freq = 2 * qpo_freq
    counts_qpo = 100 * np.sin(2 * np.pi * qpo_freq * time + np.pi / 4)
    counts_harmonic = 50 * np.sin(2 * np.pi * harmonic_freq * time + np.pi / 4)
    counts = counts_qpo + counts_harmonic

    lc = Lightcurve(time, counts, dt=dt)
    cs = Crossspectrum(lc, lc)

    qpo_phase = np.pi / 4
    harmonic_phase = np.pi / 4
    model = models.Lorentz1D(x_0=qpo_freq) + models.Lorentz1D(x_0=harmonic_freq)

    cap_phi_1, cap_phi_2, small_psi = spec.get_phase_lag(cs, model)

    assert np.isclose(cap_phi_1, np.pi / 2, atol=1e-2)
    assert np.isclose(cap_phi_2, 2 * np.pi, atol=1e-2)
    assert np.isclose(small_psi, np.pi / 2, atol=1e-2)

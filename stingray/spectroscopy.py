import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import brent


from astropy.table import Table
from astropy.modeling.models import Lorentz1D
from stingray import Lightcurve, Crossspectrum
from stingray.utils import standard_error, find_nearest, fft, ifft
from stingray.filters import Optimal1D, Window1D


def load_lc_fits(file, counts_type=True):
    """
    Function to load FITS file having reference band and channel of interest
    bands.

    Parameters
    ----------
    file : string
        Path of the FITS file.

    counts_type : bool, optional, default True
        Set this parameter to False if the unit is ct/s.

    Returns
    -------
    ref : 1-d float array
        Reference band counts array

    ci : 1-d float array
        Channel of Interest band counts array

    meta : dict
        metadata

    """
    lc_fits = Table.read(file)
    meta = lc_fits.meta
    dt = meta["DT"]

    ref = np.asanyarray(lc_fits["REF"].T, dtype=np.float64)
    ci = np.asanyarray(lc_fits["CI"].T, dtype=np.float64)

    if not counts_type:
        print(dt)
        ref *= dt
        ci *= dt

    return ref, ci, meta


def get_new_df(spectrum, n_bins):
    """
    Return the new df used to re-bin the spectrum.

    Parameters
    ----------
    spectrum : Powerspectrum or Crossspectrum class instance

    n_bins : int
        New bin size.

    Returns
    -------
    new_df : float

    """
    _, f_bin_edges, _ = binned_statistic(
        spectrum.freq, spectrum.power, statistic="mean", bins=n_bins
    )
    new_df = np.median(np.diff(f_bin_edges))
    return new_df


def ccf(cs_power, ps_rms, n_bins):
    """
    Return the normalised cross-correlation function of the cross spectrum. It
    is normalised using the average RMS of power spectrum.

    Parameters
    ----------
    cs_power : 1-d float array
        Power of cross spectrum.

    ps_rms : 1-d float array
        RMS of power spectrum.

    n_bins : int
        Number of buns.

    Returns
    -------
    ccf_real_norm : 1-d float array
        Real part of normalised ccf.
    """
    ccf = ifft(cs_power)  # inverse fast fourier transformation
    ccf_real = ccf.real  # real part of ccf
    ccf_real_norm = ccf_real * (2 / n_bins / ps_rms)  # normalisation
    return ccf_real_norm


def ccf_error(
    ref_counts, ci_counts_0, cs_res_model, rebin_log_factor, meta, ps_rms, filter_type="optimal"
):
    n_seg = meta["N_SEG"]
    n_seconds = meta["NSECONDS"]
    dt = meta["DT"]
    n_bins = meta["N_BINS"]

    seg_ref_counts = np.array(np.split(ref_counts, n_seg))
    seg_ci_counts = np.array(np.split(ci_counts_0, n_seg))
    seg_css = np.array([])
    seg_ccfs = np.array([])
    seg_times = np.arange(0, n_seconds, dt)  # light curve time bins

    for i in range(n_seg):  # for each segment
        # Creating cross spectrum
        seg_ci_lc = Lightcurve(seg_times, seg_ci_counts[i], dt=dt)  # CoI light curve
        seg_ref_lc = Lightcurve(seg_times, seg_ref_counts[i], dt=dt)  # reference band light curve
        seg_cs = Crossspectrum(
            data2=seg_ci_lc, data1=seg_ref_lc, norm="leahy", power_type="absolute"
        )  # cross spectrum
        seg_cs = seg_cs.rebin_log(rebin_log_factor)  # cross spectrum rebinning

        # applying filter
        if filter_type == "optimal":
            cs_filter = Optimal1D(cs_res_model)
            filter_freq = cs_filter(seg_cs.freq)
            filtered_seg_cs_power = filter_freq * np.abs(seg_cs.power)
        else:
            cs_filter = Window1D(cs_res_model)
            filter_freq = cs_filter(seg_cs.freq)
            filtered_seg_cs_power = filter_freq * np.abs(seg_cs.power)

        # calculating normalized ccf
        seg_ccf = ifft(filtered_seg_cs_power)  # inverse FFT
        seg_ccf_real = seg_ccf.real  # real part of ccf
        seg_ccf_real_norm = seg_ccf_real * (2 / n_bins / ps_rms)  # normalisation

        if i == 0:
            seg_css = np.hstack((seg_css, np.array(filtered_seg_cs_power)))
        else:
            seg_css = np.vstack((seg_css, np.array(filtered_seg_cs_power)))
        if i == 0:
            seg_ccfs = np.hstack((seg_ccfs, np.array(seg_ccf_real_norm)))
        else:
            seg_ccfs = np.vstack((seg_ccfs, np.array(seg_ccf_real_norm)))

    # ccf after taking avg
    avg_seg_css = np.average(seg_css, axis=0)  # average of cross spectrum
    avg_seg_ccf = ccf(avg_seg_css, ps_rms, n_bins)

    error = standard_error(seg_ccfs, avg_seg_ccf)
    return error, avg_seg_ccf


def get_parameters(counts, dt, model):
    """
    Function to calculate mean count rate, phase offset and phase difference
    between the harmonics.

    Parameters
    ----------
    counts : np.array of ints or floats
        1-D array of numbers to search through. Should already be sorted
        from low values to high values.

    dt : float
        Time resolution of the light curve.

    model: astropy.modeling.models class instance
        The parametric model supposed to represent the data. For details
        see the astropy.modeling documentation. It assumes the first model
        is Lorentzian model fit at QPO component (fundamental) and the
        next Lorentzian model fit at the second harmonic.

    Returns
    -------
    mu : float
        Mean count rate of light curve.

    cap_phi_1 : float
        Phase offset of the first harmonic.

    cap_phi_2 : float
        Phase offset of the second harmonic.

    small_psi : float
        Phase difference between the first and second harmonics.

    """
    x = counts / dt
    X = fft(x)  # fourier transform of count rate

    n = X.size
    timestep = dt
    ffreq = np.fft.fftfreq(n, d=timestep)

    x_0_0 = model[0].x_0.value
    x_0_1 = model[1].x_0.value

    _, idx_0 = find_nearest(ffreq, x_0_0)
    _, idx_1 = find_nearest(ffreq, x_0_1)

    dft_X_1 = X[idx_0]  # 1st harmonic
    dft_X_2 = X[idx_1]  # 2nd harmonic

    small_psi_1 = np.angle(dft_X_1)
    small_psi_2 = np.angle(dft_X_2)

    cap_phi_1 = small_psi_1
    cap_phi_2 = small_psi_2

    mu = np.mean(x)
    small_psi = (cap_phi_2 / 2 - cap_phi_1) % np.pi

    return mu, cap_phi_1, cap_phi_2, small_psi


def waveform(x, mu, avg_sigma_1, avg_sigma_2, cap_phi_1, cap_phi_2):
    """
    Return the QPO waveform (periodic function of QPO phase).

    Parameters
    ----------
    x : np.array of ints or floats
        QPO phase

    mu : float
        Mean count rate of light curve.

    avg_sigma_1 : float
        Average RMS in the first harmonic.

    avg_sigma_2 : float
        Average RMS in the second harmonic.

    cap_phi_1 : float
        Phase offset of the first harmonic.

    cap_phi_2 : float
        Phase offset of the second harmonic.

    Returns
    -------
    y : np.array
        QPO waveform.

    """
    y = mu * (
        1
        + np.sqrt(2)
        * (avg_sigma_1 * np.cos(x - cap_phi_1) + avg_sigma_2 * np.cos(2 * x - cap_phi_2))
    )
    return y


def psi_distance(avg_psi, psi):
    """
    Return the distance between array of phase differences of the segments
    and the mean phase difference.

    Parameters
    ----------
    avg_psi : float
        Mean phase difference between the first and second harmonics.

    psi: np.array of list of floats
        Phase difference of the segments.

    Returns
    -------
    dm : np.array

    """
    delta = np.abs(psi - avg_psi)
    dm = np.array([delta_i if avg_psi < np.pi / 2 else np.pi - delta for delta_i in delta])
    return dm


def x_2_function(x, *args):
    """
    Function to minimise to find the average phase difference of the segments.
    """
    psi_m = np.array(args)
    X_2 = np.sum(psi_distance(x, psi_m) ** 2)
    return X_2


def get_mean_phase_difference(cs, model):
    """
    Return the mean phase difference between the first and second harmonics.

    Parameters
    ----------
    cs : Crossspectrum
        A Crossspectrum instance.

    model : astropy.modeling.models class instance
        The parametric model supposed to represent the data. For details
        see the astropy.modeling documentation. It assumes the first model
        is Lorentzian model fit at QPO component (fundamental) and the
        next Lorentzian model fit at the second harmonic.

    Returns
    -------
    avg_psi : float
        Mean phase difference.

    stddev:
        Standard deviation on the mean.
    """
    counts = cs.lc1.counts  # counts in CoI lightcurve
    dt = cs.lc2.dt  # dt
    n_seg = cs.m  # number of segments

    counts_seg = np.array_split(counts, n_seg)

    small_psis = np.array([])

    for i in range(n_seg):
        counts_seg_i = counts_seg[i]  # for each count segment
        _, _, _, small_psi_m = get_parameters(counts_seg_i, dt, model)
        small_psis = np.append(small_psis, small_psi_m)

    avg_psi, _, _, _ = brent(x_2_function, args=tuple(small_psis), full_output=True)

    stddev = avg_psi / n_seg

    return avg_psi, stddev


def get_phase_lag(cs, model):
    """
    Return the phase offset of the waveform.

    Parameters
    ----------
    cs : Crossspectrum
        A Crossspectrum instance.

    model: astropy.modeling.models class instance
        The parametric model supposed to represent the data. For details
        see the astropy.modeling documentation. It assumes the first model
        is Lorentzian model fit at QPO component (fundamental) and the
        next Lorentzian model fit at the second harmonic.

    Returns
    -------
    cap_phi_1 : float
        Phase offset of the first harmonic.

    cap_phi_2 : float
        Phase offset of the second harmonic.

    small_psi : float
        Phase difference between the first and second harmonics.

    """
    x_0_0 = model[0].x_0.value
    x_0_1 = model[1].x_0.value

    _, idx_0 = find_nearest(cs.freq, x_0_0)
    _, idx_1 = find_nearest(cs.freq, x_0_1)

    C_E_1 = cs.power[idx_0]  # 1st harmonic
    C_E_2 = cs.power[idx_1]  # 2nd harmonic

    delta_E_1 = np.angle(C_E_1)
    delta_E_2 = np.angle(C_E_2)

    avg_psi, _ = get_mean_phase_difference(cs, model)

    cap_phi_1 = np.pi / 2 + delta_E_1
    cap_phi_2 = 2 * (cap_phi_1 + avg_psi) + delta_E_2

    return cap_phi_1, cap_phi_2, avg_psi


def compute_rms(spectrum, model, criteria="all"):
    """
    Return the average RMS based of the fitting model used and frequency
    selection criteria.

    Parameters
    ----------
    spectrum : Powerspectrum or Crossspectrum class instance

    model: astropy.modeling.models class instance
        The parametric model supposed to represent the data. For details
        see the astropy.modeling documentation

    criteria : string, optional, default "all The parameter to decide
    which part of the output to be used to calculate rms. Allowed values
    are `all`, `posfreq`, `window` and `optimal`.

    Returns
    -------
    rms : float
        Average RMS.

    """

    if criteria == "all":
        model_output = model(spectrum.freq)
    elif criteria == "posfreq" or criteria == "optimal":
        model_output = model(spectrum.freq[spectrum.freq > 0])
    elif criteria == "window":
        model_output = model(spectrum.freq)
        assert isinstance(model[0], Lorentz1D)
        x_0 = model[0].x_0.value
        fwhm = model[0].fwhm.value
        for i in range(len(spectrum.freq)):
            if np.abs(spectrum.freq[i] - x_0) >= (fwhm / 2):
                model_output[i] = 0
    else:
        raise ValueError("Incorrect frequency selection criteria.")

    rms = np.sqrt(np.sum(model_output * spectrum.df)).mean()

    return rms

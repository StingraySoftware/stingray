import numpy as np
from scipy.optimize import brentq


def _find_delay_with_ccf(amp, pha):
    """Measure a phase shift between pulsed profile and template.

    Let P = F(p), S = F(t), where p and t are the pulse profile and the
    template and F indicates the Fourier transform
    The shift between p and t is then given by the maximum of the
    cross-correlation function.
    Here we use a quick-and-dirty algorithm, that works for pulse profiles but
    is not very general (does not substitute ``CrossCorrelation``).
    In particular, we only use part of the frequencies in the input product of
    Fourier Transforms.

    Input
    -----
    amp: array of floats
        Absolute value of the product between the Fourier transform of the
        pulse profile and of the template (|P S|)

    pha: array of floats
        Difference between the angles of P and S

    Output
    ------
    shift: float
        Phase shift between pulse profile and template (interval -0.5 -- 0.5)
    """
    nh = 32
    nprof = nh * 2
    ccf_inv = np.zeros(64, dtype=complex)
    ccf_inv[:nh] = amp[:nh] * np.cos(pha[:nh]) + 1.0j * amp[:nh] * np.sin(pha[:nh])
    ccf_inv[nprof : nprof - nh : -1] = np.conj(ccf_inv[nprof : nprof - nh : -1])
    ccf_inv[nh // 2 : nh] = 0
    ccf_inv[nprof - nh // 2 : nprof - nh : -1] = 0
    ccf = np.fft.ifft(ccf_inv)

    imax = np.argmax(ccf.real)
    shift = normalize_phase_0d5(imax / nprof)

    return shift


def best_phase_func(tau, amp, pha, ngood=20):
    """Function to minimize for FFTFIT (Taylor 1992), eqn. A7.

    Input
    -----
    tau: float
        Trial phase shift
    amp: array of floats
        Absolute value of the product between the Fourier transform of the
        pulse profile and of the template (|P S|)
    pha: array of floats
        Difference between the angles of P and S

    Results
    -------
    res: float
        Result of the function
    """
    good = slice(1, ngood + 1)
    idx = np.arange(1, ngood + 1, dtype=int)
    res = np.sum(idx * amp[good] * np.sin(-pha[good] + TWOPI * idx * tau))
    return res


TWOPI = 2 * np.pi


def fftfit(prof, template):
    """Align a template to a pulse profile.

    Parameters
    ----------
    prof : array
        The pulse profile
    template : array, default None
        The template of the pulse used to perform the TOA calculation. If None,
        a simple sinusoid is used

    Returns
    -------
    mean_amp, std_amp : floats
        Mean and standard deviation of the amplitude
    mean_phase, std_phase : floats
        Mean and standard deviation of the phase
    """
    # Subtract mean
    prof = prof - np.mean(prof)
    nbin = len(prof)

    template = template - np.mean(template)

    # Calculate the Fourier transforms of template and profile
    temp_ft = np.fft.fft(template)
    prof_ft = np.fft.fft(prof)
    freq = np.fft.fftfreq(prof.size)

    # Make sure all frequencies are real
    good = freq == freq

    # Calculate modulus and angle of the Fourier transforms above
    P = np.abs(prof_ft[good])
    theta = np.angle(prof_ft[good])
    S = np.abs(temp_ft[good])
    phi = np.angle(temp_ft[good])

    # Check that the absolute values make sense
    assert np.allclose(temp_ft[good], S * np.exp(1.0j * phi))
    assert np.allclose(prof_ft[good], P * np.exp(1.0j * theta))

    # Calculate phase shift with the cross-correlation function
    amp = P * S
    pha = theta - phi
    dph_ccf = _find_delay_with_ccf(amp, pha)

    mean = np.mean(amp)
    ngood = np.count_nonzero(amp >= mean)

    idx = np.arange(0, len(P), dtype=int)
    sigma = np.std(prof_ft[good])

    # Now minimize the chi squared by minimizing the following function.
    # We start from the phase calculated above.
    def func_to_minimize(tau):
        return best_phase_func(-tau, amp, pha, ngood=ngood)

    start_val = dph_ccf
    start_sign = np.sign(func_to_minimize(start_val))

    # We find the interval of phases for which func_to_minimize is monotonic
    # around the approximate solution
    count_down = 0
    count_up = 0
    trial_val_up = start_val
    trial_val_down = start_val
    while True:
        if np.sign(func_to_minimize(trial_val_up)) != start_sign:
            best_dph = trial_val_up
            break
        if np.sign(func_to_minimize(trial_val_down)) != start_sign:
            best_dph = trial_val_down
            break
        trial_val_down -= 1 / nbin
        count_down += 1
        trial_val_up += 1 / nbin
        count_up += 1

    a, b = best_dph - 2 / nbin, best_dph + 2 / nbin

    # Finally, we use the BRENTQ method to find the best estimate of tau in the
    # interval around the approximate solution
    shift, res = brentq(func_to_minimize, a, b, full_output=True)
    # print(shift, normalize_phase_0d5(shift))
    shift = normalize_phase_0d5(shift)
    nmax = ngood
    good = slice(1, nmax)

    # We end with the error calculation, from Taylor 1992, eqns. A10--11
    big_sum = np.sum(
        idx[good] ** 2 * amp[good] * np.cos(-pha[good] + 2 * np.pi * idx[good] * -shift)
    )

    b = np.sum(amp[good] * np.cos(-pha[good] + 2 * np.pi * idx[good] * -shift)) / np.sum(
        S[good] ** 2
    )

    eshift = sigma**2 / (2 * b * big_sum)

    eb = sigma**2 / (2 * np.sum(S[good] ** 2))

    return b, np.sqrt(eb), shift, np.sqrt(eshift)


def normalize_phase_0d5(phase):
    """Normalize phase between -0.5 and 0.5

    Examples
    --------
    >>> normalize_phase_0d5(0.5)
    0.5
    >>> normalize_phase_0d5(-0.5)
    0.5
    >>> normalize_phase_0d5(4.25)
    0.25
    >>> normalize_phase_0d5(-3.25)
    -0.25
    """
    while phase > 0.5:
        phase -= 1
    while phase <= -0.5:
        phase += 1
    return phase

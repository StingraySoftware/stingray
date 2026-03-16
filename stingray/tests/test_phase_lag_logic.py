import numpy as np
import pytest
from unittest.mock import patch
from astropy.modeling import models
from stingray.spectroscopy import get_phase_lag

# A simple Mock object to simulate the Crossspectrum class
class MockCrossspectrum:
    def __init__(self, freqs, power):
        self.freq = freqs
        self.power = power

def test_get_phase_lag_logic():
    # 1. Setup Frequency Domain (0 to 5 Hz)
    freqs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 2. Setup Power (Complex numbers to simulate phase)
    # At freq=1.0 (Index 1): Phase = 0 (Real number)
    # At freq=3.0 (Index 3): Phase = pi/4 (Real == Imaginary)
    power = np.zeros_like(freqs, dtype=complex)
    power[1] = 10.0 + 0j          # Angle = 0.0
    power[3] = 10.0 + 10.0j       # Angle = pi/4 (approx 0.785)
    
    cs = MockCrossspectrum(freqs, power)

    # 3. Setup Model (Astropy Compound Model)
    # Fundamental at 1.0 Hz, Harmonic at 3.0 Hz
    # FIX: Changed Lorentzian1D to Lorentz1D
    m1 = models.Lorentz1D(amplitude=1, x_0=1.0, fwhm=0.5)
    m2 = models.Lorentz1D(amplitude=1, x_0=3.0, fwhm=0.5)
    model = m1 + m2

    # 4. Mock the internal dependency 'get_mean_phase_difference'
    # We force it to return a fixed value (0.1) so we can verify the math formula strictly.
    with patch('stingray.spectroscopy.get_mean_phase_difference', return_value=(0.1, 0.0)):
        
        # 5. Call the function
        phi1, phi2, avg_psi = get_phase_lag(cs, model)

        # 6. Verify the Math manually
        # Formula from source code: cap_phi_1 = pi/2 + delta_E_1
        expected_phi1 = np.pi/2 + 0.0
        
        # Formula from source code: cap_phi_2 = 2 * (cap_phi_1 + avg_psi) + delta_E_2
        # avg_psi is mocked as 0.1
        expected_phi2 = 2 * (expected_phi1 + 0.1) + (np.pi/4)

        # Assertions
        assert np.isclose(phi1, expected_phi1), f"Phi1 mismatch: Got {phi1}, Expected {expected_phi1}"
        assert np.isclose(phi2, expected_phi2), f"Phi2 mismatch: Got {phi2}, Expected {expected_phi2}"
        assert avg_psi == 0.1
import numpy as np

import pytest

from stingray import Lightcurve
from stingray.bispectrum import Bispectrum

import matplotlib.pyplot as plt


def allclose_with_wrap(array1, array2):
    """Calculates if numbers are all close, considering -pi = pi."""
    for a1, a2 in zip(array1.flatten(), array2.flatten()):
        condition = np.isclose(a1, a2)
        if np.isclose(a1, np.pi) or np.isclose(a1, -np.pi):
            condition = condition or np.isclose(a1, -a2)

        if not condition:
            return False
    return True


class TestBispectrum(object):
    @classmethod
    def setup_class(cls):
        cls.lc = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        cls.lc1 = Lightcurve([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [2, 1, 3, 1, 4, 2])

    def test_create_bispectrum(self):
        """
        Demonstrate that we can create a Bispectrum object.
        """
        Bispectrum(self.lc)

    def test_wrong_lc(self):
        lc = [1, 2, 3, 4]
        with pytest.raises(TypeError):
            Bispectrum(lc)

    def test_wrong_maxlag(self):
        with pytest.raises(ValueError):
            Bispectrum(self.lc, maxlag="123")

    def test_maxlag_none(self):
        bs = Bispectrum(self.lc)
        assert bs.maxlag == int(self.lc.n / 2)

    def test_neg_maxlag(self):
        bs = Bispectrum(self.lc, maxlag=-2)
        assert bs.maxlag == 2

    def test_bispectrum_with_none_maxlag(self):
        bs = Bispectrum(self.lc)
        lags = np.array([-2, -1, 0, 1, 2])
        freq = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
        cum3 = np.array(
            [
                [0.0576, 0.1216, 0.1376, -0.2176, -0.0448],
                [0.1216, -0.6752, 0.4128, 0.1216, -0.0096],
                [0.1376, 0.4128, 0.288, -0.6752, 0.0576],
                [-0.2176, 0.1216, -0.6752, 0.4128, 0.1216],
                [-0.0448, -0.0096, 0.0576, 0.1216, 0.1376],
            ]
        )

        bispec = np.array(
            [
                [
                    1.26572936 - 1.96410531e00j,
                    1.01680863 - 2.33088331e00j,
                    0.53357150 - 3.20079088e-01j,
                    -0.39280863 + 1.79793854e00j,
                    0.37973002 + 0.00000000e00j,
                ],
                [
                    1.01680863 - 2.33088331e00j,
                    -0.29772936 - 1.20447928e00j,
                    -0.11757150 - 7.55604229e-02j,
                    0.03626998 + 1.11022302e-16j,
                    -0.39280863 - 1.79793854e00j,
                ],
                [
                    0.53357150 - 3.20079088e-01j,
                    -0.11757150 - 7.55604229e-02j,
                    0.27200000 + 0.00000000e00j,
                    -0.11757150 + 7.55604229e-02j,
                    0.53357150 + 3.20079088e-01j,
                ],
                [
                    -0.39280863 + 1.79793854e00j,
                    0.03626998 - 1.11022302e-16j,
                    -0.11757150 + 7.55604229e-02j,
                    -0.29772936 + 1.20447928e00j,
                    1.01680863 + 2.33088331e00j,
                ],
                [
                    0.37973002 + 0.00000000e00j,
                    -0.39280863 - 1.79793854e00j,
                    0.53357150 + 3.20079088e-01j,
                    1.01680863 + 2.33088331e00j,
                    1.26572936 + 1.96410531e00j,
                ],
            ]
        )

        bispec_mag = np.array(
            [
                [2.33661732, 2.54301333, 0.62221312, 1.84034823, 0.37973002],
                [2.54301333, 1.24073087, 0.13975849, 0.03626998, 1.84034823],
                [0.62221312, 0.13975849, 0.272, 0.13975849, 0.62221312],
                [1.84034823, 0.03626998, 0.13975849, 1.24073087, 2.54301333],
                [0.37973002, 1.84034823, 0.62221312, 2.54301333, 2.33661732],
            ]
        )

        bispec_phase = np.array(
            [
                [-9.98346367e-01, -1.15944968e00, -5.40331561e-01, 1.78589369e00, 0.00000000e00],
                [-1.15944968e00, -1.81312395e00, -2.57038310e00, 3.06099713e-15, -1.78589369e00],
                [-5.40331561e-01, -2.57038310e00, 0.00000000e00, 2.57038310e00, 5.40331561e-01],
                [1.78589369e00, -3.06099713e-15, 2.57038310e00, 1.81312395e00, 1.15944968e00],
                [0.00000000e00, -1.78589369e00, 5.40331561e-01, 1.15944968e00, 9.98346367e-01],
            ]
        )
        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert bs.scale == "biased"
        assert np.allclose(bs.lags, lags)
        assert np.allclose(bs.freq, freq)
        assert np.allclose(bs.cum3, cum3)
        assert np.allclose(bs.bispec, bispec)
        assert np.allclose(bs.bispec_mag, bispec_mag)
        assert allclose_with_wrap(bs.bispec_phase, bispec_phase)

    def test_wrong_scale_type(self):
        with pytest.raises(TypeError):
            Bispectrum(self.lc, scale=1)

    def test_wrong_scale_value(self):
        with pytest.raises(ValueError):
            Bispectrum(self.lc, scale="non-biased")

    def test_bispectrum_unbiased_scale(self):
        bs = Bispectrum(self.lc, scale="unbiased")

        lags = np.array([-2, -1, 0, 1, 2])
        freq = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])

        cum3 = np.array(
            [
                [0.096, 0.20266667, 0.22933333, -0.544, -0.224],
                [0.20266667, -0.844, 0.516, 0.20266667, -0.024],
                [0.22933333, 0.516, 0.288, -0.844, 0.096],
                [-0.544, 0.20266667, -0.844, 0.516, 0.20266667],
                [-0.224, -0.024, 0.096, 0.20266667, 0.22933333],
            ]
        )

        bispec = np.array(
            [
                [
                    1.78211085 - 2.10567238e00j,
                    1.56502881 - 3.02261738e00j,
                    0.63208770 - 8.00197720e-01j,
                    -1.20769548 + 2.56558544e00j,
                    0.49792363 + 2.22044605e-16j,
                ],
                [
                    1.56502881 - 3.02261738e00j,
                    -1.12477752 - 1.08193727e00j,
                    0.12524563 - 1.88901057e-01j,
                    0.25940971 + 2.22044605e-16j,
                    -1.20769548 - 2.56558544e00j,
                ],
                [
                    0.63208770 - 8.00197720e-01j,
                    0.12524563 - 1.88901057e-01j,
                    -0.08800000 + 0.00000000e00j,
                    0.12524563 + 1.88901057e-01j,
                    0.63208770 + 8.00197720e-01j,
                ],
                [
                    -1.20769548 + 2.56558544e00j,
                    0.25940971 - 2.22044605e-16j,
                    0.12524563 + 1.88901057e-01j,
                    -1.12477752 + 1.08193727e00j,
                    1.56502881 + 3.02261738e00j,
                ],
                [
                    0.49792363 - 2.22044605e-16j,
                    -1.20769548 - 2.56558544e00j,
                    0.63208770 + 8.00197720e-01j,
                    1.56502881 + 3.02261738e00j,
                    1.78211085 + 2.10567238e00j,
                ],
            ]
        )

        bispec_mag = np.array(
            [
                [2.75858211, 3.40375249, 1.01973097, 2.83562286, 0.49792363],
                [3.40375249, 1.56067701, 0.22664968, 0.25940971, 2.83562286],
                [1.01973097, 0.22664968, 0.088, 0.22664968, 1.01973097],
                [2.83562286, 0.25940971, 0.22664968, 1.56067701, 3.40375249],
                [0.49792363, 2.83562286, 1.01973097, 3.40375249, 2.75858211],
            ]
        )

        bispec_phase = np.array(
            [
                [-8.68432005e-01, -1.09303185e00, -9.02235465e-01, 2.01075415e00, 4.45941091e-16],
                [-1.09303185e00, -2.37560564e00, -9.85320934e-01, 8.55961046e-16, -2.01075415e00],
                [-9.02235465e-01, -9.85320934e-01, 3.14159265e00, 9.85320934e-01, 9.02235465e-01],
                [2.01075415e00, -8.55961046e-16, 9.85320934e-01, 2.37560564e00, 1.09303185e00],
                [-4.45941091e-16, -2.01075415e00, 9.02235465e-01, 1.09303185e00, 8.68432005e-01],
            ]
        )

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert bs.scale == "unbiased"
        assert np.allclose(bs.lags, lags)
        assert np.allclose(bs.freq, freq)
        assert np.allclose(bs.cum3, cum3)
        assert np.allclose(bs.bispec, bispec)
        assert np.allclose(bs.bispec_mag, bispec_mag)
        assert allclose_with_wrap(bs.bispec_phase, bispec_phase)

    def test_lc1_with_diff_lag(self):
        bs = Bispectrum(self.lc1, maxlag=1)
        lags = np.array([-0.5, 0, 0.5])
        freq = np.array([-1, 0.0, 1])
        cum3 = np.array(
            [
                [0.37114198, -0.62885802, -0.02160494],
                [-0.62885802, 0.59259259, 0.37114198],
                [-0.02160494, 0.37114198, -0.62885802],
            ]
        )

        bispec = np.array(
            [
                [0.93595679 + 2.59807621e00j, 0.61419753 + 0j, 0.61419753 + 2.22044605e-16j],
                [0.61419753 + 1.11022302e-16j, -0.22376543 + 0j, 0.61419753 - 1.11022302e-16j],
                [0.61419753 - 2.22044605e-16j, 0.61419753 + 0j, 0.93595679 - 2.59807621e00j],
            ]
        )

        bispec_mag = np.array(
            [
                [2.76152406, 0.61419753, 0.61419753],
                [0.61419753, 0.22376543, 0.61419753],
                [0.61419753, 0.61419753, 2.76152406],
            ]
        )

        bispec_phase = np.array(
            [
                [1.22501950e00, 0, 3.61519859e-16],
                [1.80759930e-16, 3.14159265e00, -1.80759930e-16],
                [-3.61519859e-16, 0, -1.22501950e00],
            ]
        )

        assert bs.lc == self.lc1
        assert np.isclose(bs.fs, 2)
        assert bs.maxlag == 1
        assert bs.n == 6
        assert bs.scale == "biased"
        assert np.allclose(bs.lags, lags)
        assert np.allclose(bs.freq, freq)
        assert np.allclose(bs.cum3, cum3)
        assert np.allclose(bs.bispec, bispec)
        assert np.allclose(bs.bispec_mag, bispec_mag)
        assert allclose_with_wrap(bs.bispec_phase, bispec_phase)

    def test_lc1_unbiased_scale(self):
        bs = Bispectrum(self.lc1, maxlag=1, scale="unbiased")
        lags = np.array([-0.5, 0, 0.5])
        freq = np.array([-1, 0.0, 1])

        cum3 = np.array(
            [
                [0.44537037, -0.75462963, -0.03240741],
                [-0.75462963, 0.59259259, 0.44537037],
                [-0.03240741, 0.44537037, -0.75462963],
            ]
        )

        bispec = np.array(
            [
                [0.99166667 + 3.11769145e00j, 0.62500000 + 0j, 0.62500000 + 0j],
                [0.62500000 + 2.22044605e-16j, -0.40000000 + 0j, 0.62500000 - 2.22044605e-16j],
                [0.62500000 + 0j, 0.62500000 + 0j, 0.99166667 - 3.11769145e00j],
            ]
        )

        bispec_mag = np.array(
            [[3.27160554, 0.625, 0.625], [0.625, 0.4, 0.625], [0.625, 0.625, 3.27160554]]
        )

        bispec_phase = np.array([[1.26283852, 0, 0], [0, 3.14159265, 0], [0, 0, -1.26283852]])

        assert bs.lc == self.lc1
        assert np.isclose(bs.fs, 2)
        assert bs.maxlag == 1
        assert bs.n == 6
        assert bs.scale == "unbiased"
        assert np.allclose(bs.lags, lags)
        assert np.allclose(bs.freq, freq)
        assert np.allclose(bs.cum3, cum3)
        assert np.allclose(bs.bispec, bispec)
        assert np.allclose(bs.bispec_mag, bispec_mag)
        assert allclose_with_wrap(bs.bispec_phase, bispec_phase)

    def test_bispectrum_window_none(self):
        bs = Bispectrum(self.lc, scale="unbiased")
        assert bs.window is None
        assert bs.window_name == "No Window"

    def test_bispectrum_window_uniform(self):
        bs = Bispectrum(self.lc, maxlag=2, window="uniform")
        lags = np.array([-2, -1, 0, 1, 2])
        freq = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
        bispec = np.array(
            [
                [
                    1.23378944 - 1.03450204e00j,
                    0.52344952 - 1.79651918e00j,
                    0.12704520 - 9.59589442e-01j,
                    -0.10264952 + 5.93058891e-01j,
                    0.71979565 + 1.11022302e-16j,
                ],
                [
                    0.52344952 - 1.79651918e00j,
                    0.13901056 - 1.67385947e00j,
                    0.20575480 - 1.11030991e00j,
                    0.26900435 + 0.00000000e00j,
                    -0.10264952 - 5.93058891e-01j,
                ],
                [
                    0.12704520 - 9.59589442e-01j,
                    0.20575480 - 1.11030991e00j,
                    -0.53760000 + 0.00000000e00j,
                    0.20575480 + 1.11030991e00j,
                    0.12704520 + 9.59589442e-01j,
                ],
                [
                    -0.10264952 + 5.93058891e-01j,
                    0.26900435 + 0.00000000e00j,
                    0.20575480 + 1.11030991e00j,
                    0.13901056 + 1.67385947e00j,
                    0.52344952 + 1.79651918e00j,
                ],
                [
                    0.71979565 - 1.11022302e-16j,
                    -0.10264952 - 5.93058891e-01j,
                    0.12704520 + 9.59589442e-01j,
                    0.52344952 + 1.79651918e00j,
                    1.23378944 + 1.03450204e00j,
                ],
            ]
        )

        window = np.array(
            [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
        )

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert np.allclose(bs.lags, lags)
        assert np.allclose(bs.freq, freq)
        assert np.allclose(bs.bispec, bispec)
        assert bs.window_name == "uniform"
        assert np.allclose(bs.window, window)

    def test_bispectrum_window_parzen(self):
        bs = Bispectrum(self.lc, maxlag=2, window="parzen")
        bispec = np.array(
            [
                [
                    0.32973576 - 7.99387943e-02j,
                    0.30089706 - 1.04641240e-01j,
                    0.27257082 - 3.99693972e-02j,
                    0.28390294 + 2.47024460e-02j,
                    0.31923282 + 6.93889390e-18j,
                ],
                [
                    0.30089706 - 1.04641240e-01j,
                    0.29306424 - 1.29343686e-01j,
                    0.27122918 - 6.46718431e-02j,
                    0.26556718 + 0.00000000e00j,
                    0.28390294 - 2.47024460e-02j,
                ],
                [
                    0.27257082 - 3.99693972e-02j,
                    0.27122918 - 6.46718431e-02j,
                    0.27040000 + 0.00000000e00j,
                    0.27122918 + 6.46718431e-02j,
                    0.27257082 + 3.99693972e-02j,
                ],
                [
                    0.28390294 + 2.47024460e-02j,
                    0.26556718 + 0.00000000e00j,
                    0.27122918 + 6.46718431e-02j,
                    0.29306424 + 1.29343686e-01j,
                    0.30089706 + 1.04641240e-01j,
                ],
                [
                    0.31923282 - 6.93889390e-18j,
                    0.28390294 - 2.47024460e-02j,
                    0.27257082 + 3.99693972e-02j,
                    0.30089706 + 1.04641240e-01j,
                    0.32973576 + 7.99387943e-02j,
                ],
            ]
        )

        window = np.array(
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [0.0, 0.0, 0.0625, 0.0625, 0.0],
                [0.0, 0.0625, 1.0, 0.0625, 0.0],
                [0.0, 0.0625, 0.0625, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert np.allclose(bs.bispec, bispec)
        assert bs.window_name == "parzen"
        assert np.allclose(bs.window, window)

    def test_bispectrum_window_hamming(self):
        bs = Bispectrum(self.lc, maxlag=2, window="hamming")
        bispec = np.array(
            [
                [
                    0.49072469 - 3.67258307e-01j,
                    0.34762420 - 4.91066236e-01j,
                    0.21848648 - 1.93948024e-01j,
                    0.26176961 + 1.19866471e-01j,
                    0.43090809 + 5.55111512e-17j,
                ],
                [
                    0.34762420 - 4.91066236e-01j,
                    0.30777863 - 5.94236424e-01j,
                    0.21286804 - 3.03495625e-01j,
                    0.19173603 + 5.55111512e-17j,
                    0.26176961 - 1.19866471e-01j,
                ],
                [
                    0.21848648 - 1.93948024e-01j,
                    0.21286804 - 3.03495625e-01j,
                    0.19471176 + 0.00000000e00j,
                    0.21286804 + 3.03495625e-01j,
                    0.21848648 + 1.93948024e-01j,
                ],
                [
                    0.26176961 + 1.19866471e-01j,
                    0.19173603 - 5.55111512e-17j,
                    0.21286804 + 3.03495625e-01j,
                    0.30777863 + 5.94236424e-01j,
                    0.34762420 + 4.91066236e-01j,
                ],
                [
                    0.43090809 - 5.55111512e-17j,
                    0.26176961 - 1.19866471e-01j,
                    0.21848648 + 1.93948024e-01j,
                    0.34762420 + 4.91066236e-01j,
                    0.49072469 + 3.67258307e-01j,
                ],
            ]
        )

        window = np.array(
            [
                [0, 0, 0, 0.023328, 0.0064],
                [0.0, 0.0, 0.2916, 0.2916, 0.023328],
                [0.0, 0.2916, 1.0, 0.2916, 0.0],
                [0.023328, 0.2916, 0.2916, 0.0, 0.0],
                [0.0064, 0.023328, 0.0, 0.0, 0.0],
            ]
        )

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert np.allclose(bs.bispec, bispec)
        assert bs.window_name == "hamming"
        assert np.allclose(bs.window, window)

    def test_bispectrum_window_hanning(self):
        bs = Bispectrum(self.lc, maxlag=2, window="hanning")
        bispec = np.array(
            [
                [
                    0.45494303 - 3.19755177e-01j,
                    0.33958823 - 4.18564961e-01j,
                    0.22628328 - 1.59877589e-01j,
                    0.27161177 + 9.88097838e-02j,
                    0.41293126 + 5.55111512e-17j,
                ],
                [
                    0.33958823 - 4.18564961e-01j,
                    0.30825697 - 5.17374745e-01j,
                    0.22091672 - 2.58687372e-01j,
                    0.19826874 + 5.55111512e-17j,
                    0.27161177 - 9.88097838e-02j,
                ],
                [
                    0.22628328 - 1.59877589e-01j,
                    0.22091672 - 2.58687372e-01j,
                    0.21760000 + 0.00000000e00j,
                    0.22091672 + 2.58687372e-01j,
                    0.22628328 + 1.59877589e-01j,
                ],
                [
                    0.27161177 + 9.88097838e-02j,
                    0.19826874 - 5.55111512e-17j,
                    0.22091672 + 2.58687372e-01j,
                    0.30825697 + 5.17374745e-01j,
                    0.33958823 + 4.18564961e-01j,
                ],
                [
                    0.41293126 - 5.55111512e-17j,
                    0.27161177 - 9.88097838e-02j,
                    0.22628328 + 1.59877589e-01j,
                    0.33958823 + 4.18564961e-01j,
                    0.45494303 + 3.19755177e-01j,
                ],
            ]
        )

        window = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.25, 0.0],
                [0.0, 0.25, 1.0, 0.25, 0.0],
                [0.0, 0.25, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert np.allclose(bs.bispec, bispec)
        assert bs.window_name == "hanning"
        assert np.allclose(bs.window, window)

    def test_bispectrum_window_triangular(self):
        bs = Bispectrum(self.lc, maxlag=2, window="triangular")
        bispec = np.array(
            [
                [
                    0.82428321 - 7.24678086e-01j,
                    0.42949926 - 1.11847388e00j,
                    0.16365995 - 5.32196997e-01j,
                    0.12187354 + 3.28915833e-01j,
                    0.57999943 + 5.55111512e-17j,
                ],
                [
                    0.42949926 - 1.11847388e00j,
                    0.25368159 - 1.17255377e00j,
                    0.18598485 - 6.91254876e-01j,
                    0.18948537 + 1.11022302e-16j,
                    0.12187354 - 3.28915833e-01j,
                ],
                [
                    0.16365995 - 5.32196997e-01j,
                    0.18598485 - 6.91254876e-01j,
                    -0.09896960 + 0.00000000e00j,
                    0.18598485 + 6.91254876e-01j,
                    0.16365995 + 5.32196997e-01j,
                ],
                [
                    0.12187354 + 3.28915833e-01j,
                    0.18948537 - 1.11022302e-16j,
                    0.18598485 + 6.91254876e-01j,
                    0.25368159 + 1.17255377e00j,
                    0.42949926 + 1.11847388e00j,
                ],
                [
                    0.57999943 - 5.55111512e-17j,
                    0.12187354 - 3.28915833e-01j,
                    0.16365995 + 5.32196997e-01j,
                    0.42949926 + 1.11847388e00j,
                    0.82428321 + 7.24678086e-01j,
                ],
            ]
        )

        window = np.array(
            [
                [0.0, 0.0, 0.0, 0.384, 0.36],
                [0.0, 0.0, 0.64, 0.64, 0.384],
                [0.0, 0.64, 1.0, 0.64, 0.0],
                [0.384, 0.64, 0.64, 0.0, 0.0],
                [0.36, 0.384, 0.0, 0.0, 0.0],
            ]
        )

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert np.allclose(bs.bispec, bispec)
        assert bs.window_name == "triangular"
        assert np.allclose(bs.window, window)

    def test_bispectrum_window_welch(self):
        bs = Bispectrum(self.lc, maxlag=2, window="welch")
        bispec = np.array(
            [
                [
                    0.66362182 - 7.19449149e-01j,
                    0.40407352 - 9.41771162e-01j,
                    0.14913738 - 3.59724574e-01j,
                    0.25112648 + 2.22322014e-01j,
                    0.56909534 + 5.55111512e-17j,
                ],
                [
                    0.40407352 - 9.41771162e-01j,
                    0.33357818 - 1.16409318e00j,
                    0.13706262 - 5.82046588e-01j,
                    0.08610466 + 0.00000000e00j,
                    0.25112648 - 2.22322014e-01j,
                ],
                [
                    0.14913738 - 3.59724574e-01j,
                    0.13706262 - 5.82046588e-01j,
                    0.12960000 + 0.00000000e00j,
                    0.13706262 + 5.82046588e-01j,
                    0.14913738 + 3.59724574e-01j,
                ],
                [
                    0.25112648 + 2.22322014e-01j,
                    0.08610466 + 0.00000000e00j,
                    0.13706262 + 5.82046588e-01j,
                    0.33357818 + 1.16409318e00j,
                    0.40407352 + 9.41771162e-01j,
                ],
                [
                    0.56909534 - 5.55111512e-17j,
                    0.25112648 - 2.22322014e-01j,
                    0.14913738 + 3.59724574e-01j,
                    0.40407352 + 9.41771162e-01j,
                    0.66362182 + 7.19449149e-01j,
                ],
            ]
        )

        window = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5625, 0.5625, 0.0],
                [0.0, 0.5625, 1.0, 0.5625, 0.0],
                [0.0, 0.5625, 0.5625, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert np.allclose(bs.bispec, bispec)
        assert bs.window_name == "welch"
        assert np.allclose(bs.window, window)

    def test_bispectrum_window_blackmann(self):
        bs = Bispectrum(self.lc, maxlag=2, window="blackmann")
        bispec = np.array(
            [
                [
                    0.36998520 - 1.56242334e-01j,
                    0.31320687 - 2.04896068e-01j,
                    0.25789699 - 7.84933643e-02j,
                    0.27972923 + 4.85115670e-02j,
                    0.34901011 + 2.77555756e-17j,
                ],
                [
                    0.31320687 - 2.04896068e-01j,
                    0.29778797 - 2.52805407e-01j,
                    0.25527601 - 1.26632734e-01j,
                    0.24440392 + 2.77555756e-17j,
                    0.27972923 - 4.85115670e-02j,
                ],
                [
                    0.25789699 - 7.84933643e-02j,
                    0.25527601 - 1.26632734e-01j,
                    0.25316762 + 0.00000000e00j,
                    0.25527601 + 1.26632734e-01j,
                    0.25789699 + 7.84933643e-02j,
                ],
                [
                    0.27972923 + 4.85115670e-02j,
                    0.24440392 - 2.77555756e-17j,
                    0.25527601 + 1.26632734e-01j,
                    0.29778797 + 2.52805407e-01j,
                    0.31320687 + 2.04896068e-01j,
                ],
                [
                    0.34901011 - 2.77555756e-17j,
                    0.27972923 - 4.85115670e-02j,
                    0.25789699 + 7.84933643e-02j,
                    0.31320687 + 2.04896068e-01j,
                    0.36998520 + 1.56242334e-01j,
                ],
            ]
        )

        window = np.array(
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 8.41430799e-04, 4.73205937e-05],
                [0.00000000e00, 0.00000000e00, 1.22318645e-01, 1.22318645e-01, 8.41430799e-04],
                [0.00000000e00, 1.22318645e-01, 9.99997000e-01, 1.22318645e-01, 0.00000000e00],
                [8.41430799e-04, 1.22318645e-01, 1.22318645e-01, 0.00000000e00, 0.00000000e00],
                [4.73205937e-05, 8.41430799e-04, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            ]
        )

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert np.allclose(bs.bispec, bispec)
        assert bs.window_name == "blackmann"
        assert np.allclose(bs.window, window)

    def test_bispectrum_window_flattop(self):
        bs = Bispectrum(self.lc, maxlag=2, window="flat-top")
        bispec = np.array(
            [
                [
                    28.90860041 - 0.40702734j,
                    28.76176133 - 0.53280571j,
                    28.61753157 - 0.20351367j,
                    28.67523175 + 0.12577837j,
                    28.85512219 + 0.0j,
                ],
                [
                    28.76176133 - 0.53280571j,
                    28.72187869 - 0.65858408j,
                    28.61070029 - 0.32929204j,
                    28.58187089 + 0.0j,
                    28.67523175 - 0.12577837j,
                ],
                [
                    28.61753157 - 0.20351367j,
                    28.61070029 - 0.32929204j,
                    28.60647832 + 0.0j,
                    28.61070029 + 0.32929204j,
                    28.61753157 + 0.20351367j,
                ],
                [
                    28.67523175 + 0.12577837j,
                    28.58187089 + 0.0j,
                    28.61070029 + 0.32929204j,
                    28.72187869 + 0.65858408j,
                    28.76176133 + 0.53280571j,
                ],
                [
                    28.85512219 + 0.0j,
                    28.67523175 - 0.12577837j,
                    28.61753157 + 0.20351367j,
                    28.76176133 + 0.53280571j,
                    28.90860041 + 0.40702734j,
                ],
            ]
        )

        window = np.array(
            [
                [0.00000000e00, -0.00000000e00, 0.00000000e00, 5.95391791e-18, 3.48773876e-32],
                [-0.00000000e00, 0.00000000e00, 3.18233584e-01, 3.18233584e-01, 5.95391791e-18],
                [0.00000000e00, 3.18233584e-01, 9.96392115e01, 3.18233584e-01, 0.00000000e00],
                [5.95391791e-18, 3.18233584e-01, 3.18233584e-01, 0.00000000e00, -0.00000000e00],
                [3.48773876e-32, 5.95391791e-18, 0.00000000e00, -0.00000000e00, 0.00000000e00],
            ]
        )

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert np.allclose(bs.bispec, bispec)
        assert bs.window_name == "flat-top"
        assert np.allclose(bs.window, window)

    def test_bad_window(self):
        window_bad = 123
        with pytest.raises(TypeError):
            bs = Bispectrum(self.lc, maxlag=2, window=window_bad)

    def test_not_available_window(self):
        window_not = "kaiser"
        with pytest.raises(ValueError):
            bs = Bispectrum(self.lc, maxlag=2, window=window_not)

    def test_plot_cum3(self):
        bs = Bispectrum(self.lc)
        bs.plot_cum3()
        assert plt.fignum_exists(1)

    def test_plot_mag(self):
        bs = Bispectrum(self.lc)
        bs.plot_mag()
        assert plt.fignum_exists(1)

    def test_plot_phase(self):
        bs = Bispectrum(self.lc)
        bs.plot_phase()
        assert plt.fignum_exists(1)

    def test_plot_cum3_axis(self):
        bs = Bispectrum(self.lc)
        bs.plot_cum3(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    def test_plot_mag_axis(self):
        bs = Bispectrum(self.lc)
        bs.plot_mag(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    def test_plot_phase_axis(self):
        bs = Bispectrum(self.lc)
        bs.plot_phase(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

import pytest
import numpy as np
import stingray.utils as utils
from stingray.utils import HAS_NUMBA
from scipy.stats import sem

np.random.seed(20150907)


class TestRebinData(object):
    @classmethod
    def setup_class(cls):
        cls.dx = 1.0
        cls.n = 10
        cls.counts = 2.0
        cls.x = np.arange(cls.dx / 2, cls.dx / 2 + cls.n * cls.dx, cls.dx)
        cls.y = np.zeros_like(cls.x) + cls.counts
        cls.yerr = np.sqrt(cls.y)

    def test_new_stepsize(self):
        dx_new = 2.0
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new, self.yerr)
        assert np.allclose(step_size, dx_new / self.dx)

    def test_arrays(self):
        dx_new = 2.0
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new, self.yerr)
        assert isinstance(xbin, np.ndarray)
        assert isinstance(ybin, np.ndarray)

    def test_length_matches(self):
        dx_new = 2.0
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new, self.yerr)
        assert xbin.shape[0] == ybin.shape[0]

    def test_binned_counts(self):
        dx_new = 2.0

        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new, self.yerr)

        ybin_test = np.zeros_like(xbin) + self.counts * dx_new / self.dx
        assert np.allclose(ybin, ybin_test)

    def test_uneven_bins(self):
        dx_new = 1.5
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new, self.yerr)
        assert np.isclose(xbin[1] - xbin[0], dx_new)

    def test_uneven_binned_counts(self):
        dx_new = 1.5
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new, self.yerr)
        ybin_test = np.zeros_like(xbin) + self.counts * dx_new / self.dx
        assert np.allclose(ybin_test, ybin)

    def test_rebin_data_should_raise_error_when_method_is_not_allowed(self):
        dx_new = 2.0
        with pytest.raises(ValueError):
            utils.rebin_data(self.x, self.y, dx_new, self.yerr, method="not_allowed_method")

    def test_rebin_variable_input_sampling(self):
        x1 = np.linspace(0, 10, 11)
        x2 = np.linspace(10.33, 20.0, 30)
        x3 = np.linspace(21, 30, 10)
        x = np.hstack([x1, x2, x3])

        counts = 2.0
        y = np.zeros_like(x) + counts

        yerr = np.sqrt(y)
        dx_new = 1.5
        xbin, ybin, yerr_bin, step_size = utils.rebin_data(x, y, dx_new, yerr)

        assert np.allclose(ybin, counts * step_size)
        assert len(xbin) == 20

    def test_rebin_variable_input_mean(self):
        x1 = np.linspace(0, 10, 11)
        x2 = np.linspace(10.33, 20.0, 30)
        x3 = np.linspace(21, 30, 10)
        x = np.hstack([x1, x2, x3])

        counts = 2.0
        y = np.zeros_like(x) + counts

        yerr = np.sqrt(y)
        dx_new = 1.5
        xbin, ybin, yerr_bin, step_size = utils.rebin_data(x, y, dx_new, yerr, method="average")
        assert len(xbin) == 20
        assert np.allclose(ybin, counts)

    def test_rebin_error_propagation(self):
        x = np.arange(0, 4, 1.0)  # 4 points
        y = np.ones(4) * 10
        yerr = np.ones(4) * 2  # Each point has σ=2

        xbin, ybin, ybinerr, _ = utils.rebin_data(x, y, 4.0, yerr=yerr, method="sum")
        # Correct: sqrt(4*2**2) = sqrt(16) = 4
        assert np.allclose(ybinerr, 4.0)

        xbin, ybin, ybinerr, _ = utils.rebin_data(x, y, 4.0, yerr=yerr, method="mean")
        # Correct: sqrt(4*2**2) / 4 = 4/4 = 1
        assert np.allclose(ybinerr, 1.0)


class TestRebinDataLog(object):
    @classmethod
    def setup_class(cls):
        cls.dx = 1
        cls.xmax = 21
        cls.xmin = 1
        cls.x = np.arange(cls.xmin, cls.xmax, cls.dx)
        cls.y = np.arange(cls.xmin, cls.xmax, cls.dx)
        cls.y_err = np.ones_like(cls.y)

        cls.true_bins = np.array([1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.0])

        cls.true_bin_edges = np.array(
            [1.0, 2.0, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 13.0, 15.5, 18.0, 20.0]
        )

        cls.true_values = np.array(
            [1.0, 2.0, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 13.0, 15.5, 18.0, 20.0]
        )
        cls.true_nsamples = np.array([1, 1, 1, 2, 1, 2, 1, 2, 3, 2, 3, 1])

        cls.f = 0.1

    def test_rebin_data_log_runs(self):
        _, _, _, _ = utils.rebin_data_log(self.x, self.y, self.f, y_err=self.y_err, dx=self.dx)

    def test_method_fails_if_x_and_y_of_unequal_length(self):
        with pytest.raises(ValueError):
            _, _, _, _ = utils.rebin_data_log(
                self.x[1:], self.y, self.f, y_err=self.y_err, dx=self.dx
            )

    def test_method_fails_if_y_and_yerr_of_unequal_length(self):
        with pytest.raises(ValueError):
            _, _, _, _ = utils.rebin_data_log(
                self.x, self.y, self.f, y_err=self.y_err[1:], dx=self.dx
            )

    def test_all_outputs_have_the_same_dimension_except_binx(self):
        binx, biny, binyerr, nsamples = utils.rebin_data_log(
            self.x, self.y, self.f, y_err=self.y_err, dx=self.dx
        )

        # binx describes the bin _edges_ rather than midpoints, so has one
        # more entry than biny and the rest
        assert binx.shape[0] == biny.shape[0]
        assert biny.shape[0] == binyerr.shape[0]
        assert binyerr.shape[0] == nsamples.shape[0]

    def test_binning_works_correctly(self):
        binx, biny, binyerr, nsamples = utils.rebin_data_log(self.x, self.y, self.f, y_err=self.y_err, dx=self.dx)

        assert np.allclose(np.diff(binx), self.true_bins)
        assert np.allclose(binx, self.true_bin_edges)
        assert np.allclose(biny, self.true_values)
        assert np.allclose(nsamples, self.true_nsamples)
        assert np.allclose(binyerr, 2/np.sqrt(self.true_nsamples))

    def test_method_works_on_complex_numbers(self):
        re = np.arange(self.xmin, self.xmax, self.dx)
        im = np.arange(self.xmin, self.xmax, self.dx)

        y = np.zeros(re.shape[0], dtype=np.complex64)
        yerr = np.zeros(re.shape[0], dtype=complex)

        for k, (r, i) in enumerate(zip(re, im)):
            y[k] = r + i * 1j
            yerr[k] = r + i * 1j

        real_binned = np.zeros(self.true_values.shape[0], dtype=complex)

        for i in range(self.true_values.shape[0]):
            real_binned[i] = self.true_values[i] + self.true_values[i] * 1j

        _, _, binyerr_real, _ = utils.rebin_data_log(
            self.x, y.real, self.f, y_err=yerr.real, dx=self.dx
        )
        _, biny, binyerr, _ = utils.rebin_data_log(self.x, y, self.f, y_err=yerr, dx=self.dx)

        assert np.iscomplexobj(biny)
        assert np.iscomplexobj(binyerr)
        assert np.allclose(biny, real_binned)
        assert np.allclose(binyerr, binyerr_real + 1.0j * binyerr_real)

    def test_return_float_with_floats(self):
        _, biny, binyerr, _ = utils.rebin_data_log(
            self.x, self.y, self.f, y_err=self.y_err, dx=self.dx
        )
        assert not np.iscomplexobj(biny)
        assert not np.iscomplexobj(binyerr)


class TestUtils(object):
    def test_optimal_bin_time(self):
        assert utils.optimal_bin_time(512, 2.1) == 2
        assert utils.optimal_bin_time(512, 3.9) == 2
        assert utils.optimal_bin_time(512, 4.1) == 4

    def test_order_list_of_arrays(self):
        alist = [np.array([1, 0]), np.array([2, 3])]

        order = np.argsort(alist[0])
        assert np.allclose(
            np.array([np.array([0, 1]), np.array([3, 2])]),
            np.array(utils.order_list_of_arrays(alist, order)),
        )

        alist = {"a": np.array([1, 0]), "b": np.array([2, 3])}
        alist_new = utils.order_list_of_arrays(alist, order)
        assert np.allclose(np.array([0, 1]), alist_new["a"])
        assert np.allclose(np.array([3, 2]), alist_new["b"])

        alist = 0
        assert utils.order_list_of_arrays(alist, order) is None

    def test_look_for_array(self):
        assert utils.look_for_array_in_array(np.arange(2), np.arange(1, 3))
        assert not utils.look_for_array_in_array(np.arange(2), np.arange(2, 4))

    def test_assign_value_if_none(self):
        assert utils.assign_value_if_none(None, 2) == 2
        assert utils.assign_value_if_none(1, 2) == 1

    def test_apply_function_if_none(self):
        assert utils.apply_function_if_none(None, [1, 2, 3], np.median) == 2
        assert utils.apply_function_if_none(1, [1, 2, 3], np.median) == 1

    def test_contiguous(self):
        """A more complicated example of intersection of GTIs."""
        array = np.array([0, 1, 1, 0, 1, 1, 1], dtype=bool)
        cont = utils.contiguous_regions(array)
        assert np.allclose(cont, np.array([[1, 3], [4, 7]])), "Contiguous region wrong"

    def test_get_random_state(self):
        # Life, Universe and Everything
        lue = 42
        random_state = np.random.RandomState(lue)

        assert utils.get_random_state(None) is np.random.mtrand._rand
        assert np.all(
            utils.get_random_state(lue).randn(lue) == np.random.RandomState(lue).randn(lue)
        )
        assert np.all(
            utils.get_random_state(np.random.RandomState(lue)).randn(lue)
            == np.random.RandomState(lue).randn(lue)
        )

        with pytest.raises(ValueError):
            utils.get_random_state("foobar")

    def test_check_allclose_and_print(self):
        with pytest.raises(AssertionError, match="Different values in the arrays check"):
            utils.check_allclose_and_print(
                [1.1],
                [4.5],
                rtol=1e-05,
                atol=1e-08,
            )


class TestCreateWindow(object):
    @classmethod
    def setup_class(cls):
        cls.N = 5
        cls.uniform_window = "uniform"
        cls.parzen_window = "parzen"
        cls.hamming_window = "hamming"
        cls.hanning_window = "hanning"
        cls.triangular_window = "triangular"
        cls.welch_window = "welch"
        cls.blackmann_window = "blackmann"
        cls.flattop_window = "flat-top"

    def test_bad_N(self):
        N_bad = "abc"
        with pytest.raises(TypeError):
            window = utils.create_window(N_bad, self.uniform_window)

    def test_bad_window_type(self):
        window_bad = 123
        with pytest.raises(TypeError):
            window = utils.create_window(self.N, window_bad)

    def test_not_available_window(self):
        window_not = "kaiser"
        with pytest.raises(ValueError):
            window = utils.create_window(self.N, window_not)

    def test_N_equals_zero(self):
        N = 0
        window = utils.create_window(N)
        assert len(window) == 0

    def test_uniform_window(self):
        result = np.ones(self.N)
        window = utils.create_window(self.N)
        assert np.allclose(window, result)

    def test_parzen_window(self):
        result = np.array([0, 0.25, 1, 0.25, 0])
        window = utils.create_window(self.N, self.parzen_window)
        assert np.allclose(window, result)

    def test_hamming_window(self):
        result = np.array([0.08, 0.54, 1, 0.54, 0.08])
        window = utils.create_window(self.N, self.hamming_window)
        assert np.allclose(window, result)

    def test_hanning_window(self):
        result = np.array([0, 0.5, 1, 0.5, 0])
        window = utils.create_window(self.N, self.hanning_window)
        assert np.allclose(window, result)

    def test_triangular_window(self):
        result = np.array([0.6, 0.8, 1, 0.8, 0.6])
        window = utils.create_window(self.N, self.triangular_window)
        assert np.allclose(window, result)

    def test_welch_window(self):
        result = np.array([0, 0.75, 1, 0.75, 0])
        window = utils.create_window(self.N, self.welch_window)
        assert np.allclose(window, result)

    def test_blackmann_window(self):
        result = np.array([0.006879, 0.349741, 0.999999, 0.349741, 0.006879])
        window = utils.create_window(self.N, self.blackmann_window)
        assert np.allclose(window, result)

    def test_flat_top_window(self):
        result = np.array(
            [8.67361738e-17, -2.62000000e-01, 4.63600000e00, -2.62000000e-01, 8.67361738e-17]
        )
        window = utils.create_window(self.N, self.flattop_window)
        assert np.allclose(window, result)


def test_standard_error():
    x = np.arange(1, 100)
    x = np.array(np.split(x, 3))
    error = utils.standard_error(x, x.mean(axis=0))
    assert np.allclose(error, sem(x))


def test_nearest_power_of_two():
    assert utils.nearest_power_of_two(4) == 4
    assert utils.nearest_power_of_two(5) == 4
    assert utils.nearest_power_of_two(6) == 8
    assert utils.nearest_power_of_two(7) == 8


def test_find_nearest():
    x = np.arange(1, 10)
    assert utils.find_nearest(x, 2) == (2, 1)
    assert utils.find_nearest(x, 4.5) == (5, 4)
    assert utils.find_nearest(x, 7.4) == (7, 6)


def test_equal_count_energy_ranges():
    energies = np.random.uniform(0.3, 12, 1000000)
    edges = utils.equal_count_energy_ranges(energies, 100, emin=0.3, emax=12)
    for e0, e1 in zip(edges[:-1], edges[1:]):
        good = energies[(energies >= e0) & (energies < e1)]
        assert np.count_nonzero(good) == 10000


def test_heaviside():
    assert utils.heaviside(2) == 1
    assert utils.heaviside(0) == 1
    assert utils.heaviside(-1) == 0


def test_histogram_equiv_numpy():
    x = np.random.uniform(0.0, 1.0, 100)
    (H, _) = np.histogram(x, bins=5, range=(0.0, 1.0))

    Hn = utils.histogram(x, bins=5, range=np.array([0.0, 1.0]))
    assert np.all(H == Hn)


def test_histogram2d_equiv_numpy():
    x = np.random.uniform(0.0, 1.0, 100)
    y = np.random.uniform(2.0, 3.0, 100)
    H, _, _ = np.histogram2d(x, y, bins=np.array((5, 5)), range=[(0.0, 1.0), (2.0, 3.0)])

    Hn = utils.histogram2d(x, y, bins=np.array([5, 5]), range=np.array([[0.0, 1.0], [2.0, 3.0]]))
    assert np.all(H == Hn)


@pytest.mark.skipif("not HAS_NUMBA")
def test_histogramnd_numba_fail_safely():
    x = np.random.uniform(0.0, 1.0, 100)
    y = np.random.uniform(2.0, 3.0, 100)
    z = np.random.uniform(4.0, 5.0, 100)
    w = np.random.uniform(6.0, 8.0, 100)
    H, _ = np.histogramdd(
        (x, y, z, w),
        bins=np.array((5, 6, 7, 8)),
        range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5), (6.0, 8)],
        density=True,
    )

    with pytest.warns(UserWarning, match="Cannot calculate the histogram with the numba"):
        Hn = utils.histogramnd(
            (x, y, z, w),
            bins=np.array((5, 6, 7, 8)),
            range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5), (6.0, 8)],
            density=True,
        )
    assert np.all(H == Hn)


@pytest.mark.skipif("not HAS_NUMBA")
def test_histogram3d_numba_fail_safely():
    x = np.random.uniform(0.0, 1.0, 100)
    y = np.random.uniform(2.0, 3.0, 100)
    z = np.random.uniform(4.0, 5.0, 100)
    H, _ = np.histogramdd(
        (x, y, z), bins=np.array((5, 6, 7)), range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5)], density=True
    )

    with pytest.warns(UserWarning, match="Cannot calculate the histogram with the numba"):
        Hn = utils.histogram3d(
            (x, y, z),
            bins=np.array((5, 6, 7)),
            range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5)],
            density=True,
        )
    assert np.all(H == Hn)


@pytest.mark.skipif("not HAS_NUMBA")
def test_histogram2d_numba_fail_safely():
    x = np.random.uniform(0.0, 1.0, 100)
    y = np.random.uniform(2.0, 3.0, 100)
    H, _, _ = np.histogram2d(
        x, y, bins=np.array((5, 5)), range=[(0.0, 1.0), (2.0, 3.0)], density=True
    )

    with pytest.warns(UserWarning, match="Cannot calculate the histogram with the numba"):
        Hn = utils.histogram2d(
            x,
            y,
            bins=np.array([5, 5]),
            range=np.array([[0.0, 1.0], [2.0, 3.0]]),
            density=True,
        )
    assert np.all(H == Hn)


@pytest.mark.skipif("not HAS_NUMBA")
def test_histogram_numba_fail_safely():
    x = np.random.uniform(0.0, 1.0, 100)
    (H, _) = np.histogram(x, bins=5, range=(0.0, 1.0), density=True)
    with pytest.warns(UserWarning, match="Cannot calculate the histogram with the numba"):
        Hn = utils.histogram(
            x,
            bins=5,
            range=np.array([0.0, 1.0]),
            density=True,
        )
    assert np.all(H == Hn)


@pytest.mark.skipif("not HAS_NUMBA")
def test_histogramnd_accept_complex_warns_with_numba():
    x = np.random.uniform(0.0, 1.0, 100)
    y = np.random.uniform(2.0, 3.0, 100)
    z = np.random.uniform(4.0, 5.0, 100)
    w = np.random.uniform(6.0, 8.0, 100)
    H, _ = np.histogramdd(
        (x, y, z, w),
        bins=np.array((5, 6, 7, 8)),
        range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5), (6.0, 8)],
    )

    # This implementation does not support weights
    with pytest.warns(
        UserWarning, match="Cannot calculate the histogram with the numba implementation"
    ):
        Hn = utils.histogramnd(
            (x, y, z, w),
            bins=np.array((5, 6, 7, 8)),
            range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5), (6.0, 8)],
            weights=np.ones_like(x) + 1.0j,
        )
    assert np.all(H == Hn.real)
    assert np.all(H == Hn.imag)


@pytest.mark.skipif("HAS_NUMBA")
def test_histogramnd_accept_complex():
    x = np.random.uniform(0.0, 1.0, 100)
    y = np.random.uniform(2.0, 3.0, 100)
    z = np.random.uniform(4.0, 5.0, 100)
    w = np.random.uniform(6.0, 8.0, 100)
    H, _ = np.histogramdd(
        (x, y, z, w),
        bins=np.array((5, 6, 7, 8)),
        range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5), (6.0, 8)],
    )

    Hn = utils.histogramnd(
        (x, y, z, w),
        bins=np.array((5, 6, 7, 8)),
        range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5), (6.0, 8)],
        weights=np.ones_like(x) + 1.0j,
    )
    assert np.all(H == Hn.real)
    assert np.all(H == Hn.imag)


def test_histogram3d_accept_complex():
    x = np.random.uniform(0.0, 1.0, 100)
    y = np.random.uniform(2.0, 3.0, 100)
    z = np.random.uniform(4.0, 5.0, 100)
    H, _ = np.histogramdd(
        (x, y, z), bins=np.array((5, 6, 7)), range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5)]
    )

    Hn = utils.histogram3d(
        (x, y, z),
        bins=np.array((5, 6, 7)),
        range=[(0.0, 1.0), (2.0, 3.0), (4.0, 5)],
        weights=np.ones_like(x) + 1.0j,
    )
    assert np.all(H == Hn.real)
    assert np.all(H == Hn.imag)


def test_histogram2d_accept_complex():
    x = np.random.uniform(0.0, 1.0, 100)
    y = np.random.uniform(2.0, 3.0, 100)
    H, _, _ = np.histogram2d(x, y, bins=np.array((5, 5)), range=[(0.0, 1.0), (2.0, 3.0)])

    Hn = utils.histogram2d(
        x,
        y,
        bins=np.array([5, 5]),
        range=np.array([[0.0, 1.0], [2.0, 3.0]]),
        weights=np.ones_like(x) + 1.0j,
    )
    assert np.all(H == Hn.real)
    assert np.all(H == Hn.imag)


def test_histogram_accept_complex():
    x = np.random.uniform(0.0, 1.0, 100)
    (H, _) = np.histogram(x, bins=5, range=(0.0, 1.0))
    Hn = utils.histogram(x, bins=5, range=np.array([0.0, 1.0]), weights=np.ones_like(x) + 1.0j)
    assert np.all(H == Hn.real)
    assert np.all(H == Hn.imag)


def test_compute_bin():
    bin_edges = np.array([0, 5, 10])
    assert utils.compute_bin(1, bin_edges) == 0
    assert utils.compute_bin(5, bin_edges) == 1
    assert utils.compute_bin(10, bin_edges) == 1


@pytest.mark.parametrize("kind", [list, np.float32, np.float64, np.longdouble])
def test_is_sorted(kind):
    input_array = kind([1, 2, 3, 4, 5])
    input_array_unsrt = kind([1, 2, 3, 5, 4])

    assert utils.is_sorted(input_array)
    assert not utils.is_sorted(input_array_unsrt)


def test_is_sorted_no_array():
    input_array = 1.0

    assert utils.is_sorted(input_array)


@utils.njit()
def bla_1():
    return


@utils.njit
def bla_2():
    return


@utils.vectorize(["float64(float64)"])
def bla_3(j):
    return j


def test_numba_compiled_or_mocked():
    bla_1()
    bla_2()
    bla_3(1.0)

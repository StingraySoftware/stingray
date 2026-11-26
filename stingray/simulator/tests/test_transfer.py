import os
import numpy as np
import pytest

from stingray.simulator import transfer


class TestSimulator(object):
    @classmethod
    def setup_class(self):
        arr = [[1 for j in range(5)] for i in range(10)]
        self.transfer = transfer.TransferFunction(arr)

    def test_incorrect_rows(self):
        """Test if exception is raised in case there is 1
        or no row.
        """
        arr = [[0 for j in range(5)] for i in range(1)]
        with pytest.raises(ValueError):
            transfer.TransferFunction(arr)

    def test_incorrect_columns(self):
        """Test if exception is raised in case there is 1
        or no column.
        """
        arr = [[0 for j in range(1)] for i in range(10)]
        with pytest.raises(ValueError):
            transfer.TransferFunction(arr)

    def test_time_response(self):
        """Test obtaining a time-resolved response."""
        self.transfer.time_response()

    def test_time_response_with_energy_ranges(self):
        self.transfer.time_response(e0=3.3, e1=4.7)

    def test_time_response_with_incorrect_ranges(self):
        """Test that incorrect energy ranges raises a
        Value Error.
        """
        with pytest.raises(ValueError):
            self.transfer.time_response(e0=-1, e1=2)

        with pytest.raises(ValueError):
            self.transfer.time_response(e0=3, e1=12)

        with pytest.raises(ValueError):
            self.transfer.time_response(e0=3.1, e1=3.2)

    def test_energy_response(self):
        """Test obtaining an energy-resolved response."""
        self.transfer.energy_response()

    def test_plot_with_incorrect_type(self):
        with pytest.raises(ValueError):
            self.transfer.plot("unsupported")

    def test_plot_time(self):
        self.transfer.plot(response="time")

    def test_plot_energy(self):
        self.transfer.plot(response="energy")

    def test_plot_2d(self):
        self.transfer.plot(response="2d")

    def test_plot_with_save(self):
        self.transfer.plot(save=True)
        os.remove("out.png")

    def test_plot_with_filename(self):
        self.transfer.plot(save=True, filename="response.png")
        os.remove("response.png")

    def test_io_with_pickle(self):
        self.transfer.write("transfer.pickle", fmt="pickle")
        tr = self.transfer.read("transfer.pickle", fmt="pickle")
        assert (tr.data == self.transfer.data).all()
        os.remove("transfer.pickle")

    def test_io_with_unsupported_type(self):
        with pytest.raises(KeyError):
            self.transfer.write("transfer", fmt="unsupported")
        self.transfer.write("transfer", fmt="pickle")
        with pytest.raises(KeyError):
            self.transfer.read("transfer", fmt="unsupported")
        os.remove("transfer")

    def test_simple_ir(self):
        """Test constructing a simple impulse response."""
        t0, w = 100, 500
        h = transfer.simple_ir(1, t0, w, intensity=2)
        assert len(h) == t0 + w
        assert np.allclose(h[:t0], 0)
        assert np.isclose(h.sum(), 2)

    def test_relativistic_ir(self):
        """
        Test constructing a relativistic impulse response."""
        t1, t3 = 3, 10
        h = transfer.relativistic_ir(1, t1=t1, t3=t3, intensity=0.5)
        assert len(h) == t3
        assert np.allclose(h[:t1], 0)
        assert np.isclose(h.sum(), 0.5)

import numpy as np
import pytest
from stingray.pulse.accelsearch import accelsearch, _create_responses
from stingray.utils import HAS_NUMBA

pytestmark = pytest.mark.slow


np.random.seed(235425899)


def phase(time, freq, fdot=0.0):
    return time * freq + 0.5 * time**2 * fdot


def pulsar(phase, amplitude=1, pf=1):
    mean = amplitude * (1 - pf / 2)
    TWOPI = np.pi * 2
    return mean + 0.5 * amplitude * pf * np.sin(TWOPI * phase)


class TestAccelsearch(object):
    """Unit tests for the stingray.pulse.search module."""

    @classmethod
    def setup_class(cls):
        cls.freq = 3.3456
        cls.fdot = 5e-8
        cls.tstart = 0
        cls.tstop = 10000
        cls.dt = 1 / cls.freq / 5.11
        cls.df = 1 / cls.tstop
        cls.dfdot = 1 / cls.tstop**2

        cls.times = np.arange(cls.tstart, cls.tstop, cls.dt)
        cls.phases = phase(cls.times, cls.freq, cls.fdot)
        cls.signal = pulsar(cls.phases)

        cls.phases_neg = phase(cls.times, cls.freq, -cls.fdot)
        cls.signal_neg = pulsar(cls.phases_neg)

        cls.noisy = np.random.uniform(cls.signal, 0.1)
        cls.noisy_neg = np.random.uniform(cls.signal_neg, 0.1)
        cls.rescale_fdot = 10 ** (-int(np.log10(cls.fdot)) + 1)

    def test_prepare(self):
        pass

    @pytest.mark.parametrize("kind", [float, np.longdouble])
    def test_create_response_types(self, kind):
        z = np.array([-1, 0, 1], dtype=kind)
        zint = np.array([-1, 0, 1], dtype=int)

        rs = _create_responses(z)
        rints = _create_responses(zint)
        for r, rint in zip(rs, rints):
            assert np.allclose(r, rint)

    def test_signal(self):
        candidate_table = accelsearch(
            self.times,
            self.signal,
            zmax=10,
            candidate_file="bubu.csv",
            delta_z=0.5,
            gti=[[self.tstart, self.tstop]],
            debug=True,
            interbin=True,
            nproc=1,
        )
        best = np.argmax(candidate_table["power"])
        assert np.isclose(candidate_table["frequency"][best], self.freq, atol=5 * self.df)

        print(candidate_table["fdot"][best] * self.rescale_fdot, self.fdot * self.rescale_fdot)

        assert np.isclose(
            candidate_table["fdot"][best] * self.rescale_fdot,
            self.fdot * self.rescale_fdot,
            atol=2 * self.dfdot * self.rescale_fdot,
        )

    def test_signal_neg_fdot(self):
        candidate_table = accelsearch(
            self.times, self.signal_neg, zmax=10, candidate_file="bubu.csv", delta_z=0.5, nproc=1
        )
        best = np.argmax(candidate_table["power"])
        assert np.isclose(candidate_table["frequency"][best], self.freq, atol=5 * self.df)

        assert np.isclose(
            candidate_table["fdot"][best] * self.rescale_fdot,
            -self.fdot * self.rescale_fdot,
            atol=2 * self.dfdot * self.rescale_fdot,
        )

    def test_noisy(self):
        candidate_table = accelsearch(
            self.times, self.noisy, zmax=10, candidate_file="bubu.csv", delta_z=0.5, nproc=1
        )
        best = np.argmax(candidate_table["power"])
        assert np.isclose(candidate_table["frequency"][best], self.freq, atol=5 * self.df)

        assert np.isclose(
            candidate_table["fdot"][best] * self.rescale_fdot,
            self.fdot * self.rescale_fdot,
            atol=2 * self.dfdot * self.rescale_fdot,
        )

    def test_noisy_neg_fdot(self):
        candidate_table = accelsearch(
            self.times, self.noisy_neg, zmax=10, candidate_file="bubu.csv", delta_z=0.5, nproc=1
        )
        best = np.argmax(candidate_table["power"])
        assert np.isclose(candidate_table["frequency"][best], self.freq, atol=5 * self.df)

        assert np.isclose(
            candidate_table["fdot"][best] * self.rescale_fdot,
            -self.fdot * self.rescale_fdot,
            atol=2 * self.dfdot * self.rescale_fdot,
        )

    def test_signal_with_gaps(self):
        with pytest.warns(UserWarning, match="Data contain multiple GTIs."):
            candidate_table = accelsearch(
                self.times,
                self.signal,
                zmax=10,
                candidate_file="bubu.csv",
                delta_z=0.5,
                gti=[[self.tstart, self.tstart + 10], [self.tstart + 20, self.tstop]],
                debug=True,
                interbin=True,
                nproc=1,
            )
        best = np.argmax(candidate_table["power"])
        assert np.isclose(candidate_table["frequency"][best], self.freq, atol=5 * self.df)

        print(candidate_table["fdot"][best] * self.rescale_fdot, self.fdot * self.rescale_fdot)

        assert np.isclose(
            candidate_table["fdot"][best] * self.rescale_fdot,
            self.fdot * self.rescale_fdot,
            atol=2 * self.dfdot * self.rescale_fdot,
        )

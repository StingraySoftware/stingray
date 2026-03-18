import os
import numpy as np
import pytest

# import warnings
import warnings

from astropy.table import Table
from stingray import Lightcurve, EventList
from stingray.deadtime.fad import calculate_FAD_correction, FAD
from stingray.deadtime.fad import get_periodograms_from_FAD_results
from stingray.filters import filter_for_deadtime
from stingray.crossspectrum import AveragedCrossspectrum
from stingray.powerspectrum import AveragedPowerspectrum

try:
    import h5py

    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
pytestmark = pytest.mark.slow
# np.random.seed(2134791)


def generate_events(length, ncounts):
    ev = np.random.uniform(0, length, ncounts)
    ev.sort()
    return ev


def generate_deadtime_lc(ev, dt, tstart=0, tseg=None, deadtime=2.5e-3):
    ev = filter_for_deadtime(ev, deadtime)
    return Lightcurve.make_lightcurve(
        ev, dt=dt, tstart=tstart, tseg=tseg, gti=np.array([[tstart, tseg]])
    )


def test_fad_only_one_good_interval():
    ev1 = EventList([1.1, 3], gti=[[0, 11.0]])
    ev2 = EventList([2.4, 3.6], gti=[[0, 11.0]])
    results_out_ev = FAD(ev1, ev2, 5.0, dt=0.1)

    assert results_out_ev.meta["M"] == 1


@pytest.mark.parametrize("ctrate", [0.5, 5, 50, 200])
def test_fad_power_spectrum_compliant(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    segment_size = 256.0
    ncounts = int(ctrate * length)
    ev1 = generate_events(length, ncounts)
    ev2 = generate_events(length, ncounts)

    lc1 = generate_deadtime_lc(ev1, dt, tstart=0, tseg=length, deadtime=deadtime)
    lc2 = generate_deadtime_lc(ev2, dt, tstart=0, tseg=length, deadtime=deadtime)
    ncounts1 = np.sum(lc1.counts)
    ncounts2 = np.sum(lc2.counts)

    results = calculate_FAD_correction(
        lc1,
        lc2,
        segment_size,
        plot=True,
        smoothing_alg="gauss",
        strict=True,
        verbose=True,
        tolerance=0.05,
        norm="none",
    )

    pds1_f = results["pds1"]
    pds2_f = results["pds2"]
    cs_f = results["cs"]
    ptot_f = results["ptot"]

    n = length / segment_size
    ncounts_per_intv1 = ncounts1 * segment_size / length
    ncounts_per_intv2 = ncounts2 * segment_size / length
    ncounts_per_intvtot = (ncounts1 + ncounts2) * segment_size / length
    ncounts_per_intv_geomav = np.sqrt(ncounts1 * ncounts2) * segment_size / length

    pds_std_theor = 2 / np.sqrt(n)
    cs_std_theor = np.sqrt(2 / n)

    assert np.isclose(pds1_f.std() * 2 / ncounts_per_intv1, pds_std_theor, rtol=0.1)
    assert np.isclose(pds2_f.std() * 2 / ncounts_per_intv2, pds_std_theor, rtol=0.1)
    assert np.isclose(cs_f.real.std() * 2 / ncounts_per_intv_geomav, cs_std_theor, rtol=0.1)
    assert np.isclose(ptot_f.std() * 2 / ncounts_per_intvtot, pds_std_theor, rtol=0.1)


@pytest.mark.parametrize("ctrate", [0.5, 5, 50, 200])
def test_fad_power_spectrum_compliant_objects(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    segment_size = 256.0
    ncounts = int(ctrate * length)
    ev1 = generate_events(length, ncounts)
    ev2 = generate_events(length, ncounts)

    lc1 = generate_deadtime_lc(ev1, dt, tstart=0, tseg=length, deadtime=deadtime)
    lc2 = generate_deadtime_lc(ev2, dt, tstart=0, tseg=length, deadtime=deadtime)
    ncounts1 = np.sum(lc1.counts)
    ncounts2 = np.sum(lc2.counts)

    results = calculate_FAD_correction(
        lc1,
        lc2,
        segment_size,
        plot=True,
        smoothing_alg="gauss",
        norm="none",
        strict=True,
        verbose=True,
        tolerance=0.05,
        return_objects=True,
    )

    pds1_f = results["pds1"].power
    pds2_f = results["pds2"].power
    cs_f = results["cs"].power
    ptot_f = results["ptot"].power

    n = length / segment_size
    ncounts_per_intv1 = ncounts1 * segment_size / length
    ncounts_per_intv2 = ncounts2 * segment_size / length
    ncounts_per_intvtot = (ncounts1 + ncounts2) * segment_size / length
    ncounts_per_intv_geomav = np.sqrt(ncounts1 * ncounts2) * segment_size / length

    pds_std_theor = 2 / np.sqrt(n)
    cs_std_theor = np.sqrt(2 / n)

    assert np.isclose(pds1_f.std() * 2 / ncounts_per_intv1, pds_std_theor, rtol=0.1)
    assert np.isclose(pds2_f.std() * 2 / ncounts_per_intv2, pds_std_theor, rtol=0.1)
    assert np.isclose(cs_f.real.std() * 2 / ncounts_per_intv_geomav, cs_std_theor, rtol=0.1)
    assert np.isclose(ptot_f.std() * 2 / ncounts_per_intvtot, pds_std_theor, rtol=0.1)


@pytest.mark.parametrize("ctrate", [200, 400])
def test_fad_power_spectrum_equal_ev_lc(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    gti = [[0, length]]
    segment_size = 256.0
    ncounts = int(ctrate * length)
    ev1 = filter_for_deadtime(generate_events(length, ncounts), deadtime)
    ev2 = filter_for_deadtime(generate_events(length, ncounts), deadtime)

    lc1 = Lightcurve.make_lightcurve(ev1, dt=dt, gti=gti, tstart=0, tseg=length)
    lc2 = Lightcurve.make_lightcurve(ev2, dt=dt, gti=gti, tstart=0, tseg=length)

    ev1 = EventList(ev1, gti=gti)
    ev2 = EventList(ev2, gti=gti)

    results_out_ev = FAD(
        ev1,
        ev2,
        segment_size,
        dt=dt,
        plot=True,
        strict=True,
        verbose=True,
        tolerance=0.05,
        norm="leahy",
    )
    results_out_lc = calculate_FAD_correction(
        lc1, lc2, segment_size, plot=True, strict=True, verbose=True, tolerance=0.05, norm="leahy"
    )

    for attr in ["pds1", "pds2", "cs", "ptot"]:
        assert np.allclose(results_out_ev[attr], results_out_lc[attr])


@pytest.mark.skipif("not HAS_HDF5")
@pytest.mark.parametrize("ctrate", [200])
def test_fad_power_spectrum_compliant_leahy(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    segment_size = 256.0
    ncounts = int(ctrate * length)
    ev1 = generate_events(length, ncounts)
    ev2 = generate_events(length, ncounts)

    lc1 = generate_deadtime_lc(ev1, dt, tstart=0, tseg=length, deadtime=deadtime)
    lc2 = generate_deadtime_lc(ev2, dt, tstart=0, tseg=length, deadtime=deadtime)

    results_out = calculate_FAD_correction(
        lc1,
        lc2,
        segment_size,
        plot=True,
        strict=True,
        verbose=True,
        tolerance=0.05,
        norm="leahy",
        output_file="table.hdf5",
    )

    results = Table.read("table.hdf5")
    os.unlink("table.hdf5")

    for attr in ["pds1", "pds2", "cs", "ptot"]:
        assert np.allclose(results_out[attr], results[attr])

    pds1_f = results["pds1"]
    pds2_f = results["pds2"]
    cs_f = results["cs"]
    ptot_f = results["ptot"]

    n = length / segment_size

    pds_std_theor = 2 / np.sqrt(n)
    cs_std_theor = np.sqrt(2 / n)

    assert np.isclose(pds1_f.std(), pds_std_theor, rtol=0.1)
    assert np.isclose(pds2_f.std(), pds_std_theor, rtol=0.1)
    assert np.isclose(cs_f.real.std(), cs_std_theor, rtol=0.1)
    assert np.isclose(ptot_f.std(), pds_std_theor, rtol=0.1)

    results_cs = get_periodograms_from_FAD_results(results_out, kind="cs")
    assert isinstance(results_cs, AveragedCrossspectrum)
    assert np.allclose(results_cs.power, cs_f)
    results_ps = get_periodograms_from_FAD_results(results_out, kind="pds1")
    assert isinstance(results_ps, AveragedPowerspectrum)
    assert np.allclose(results_ps.power, pds1_f)
    results_ps = get_periodograms_from_FAD_results(results_out, kind="pds2")
    assert isinstance(results_ps, AveragedPowerspectrum)
    assert np.allclose(results_ps.power, pds2_f)
    with pytest.raises(ValueError) as excinfo:
        _ = get_periodograms_from_FAD_results(results_out, kind="a")
    assert "Unknown periodogram type" in str(excinfo.value)


@pytest.mark.skipif("not HAS_HDF5")
@pytest.mark.parametrize("ctrate", [200])
def test_fad_power_spectrum_compliant_leahy_objects(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    segment_size = 256.0
    ncounts = int(ctrate * length)
    ev1 = generate_events(length, ncounts)
    ev2 = generate_events(length, ncounts)

    lc1 = generate_deadtime_lc(ev1, dt, tstart=0, tseg=length, deadtime=deadtime)
    lc2 = generate_deadtime_lc(ev2, dt, tstart=0, tseg=length, deadtime=deadtime)

    results_out = calculate_FAD_correction(
        lc1,
        lc2,
        segment_size,
        plot=True,
        strict=True,
        verbose=True,
        tolerance=0.05,
        norm="leahy",
        output_file="table.hdf5",
        return_objects=True,
    )

    results = Table.read("table.hdf5")
    os.unlink("table.hdf5")

    for attr in ["pds1", "pds2", "cs", "ptot"]:
        assert np.allclose(results_out[attr].power, results[attr])

    pds1_f = results_out["pds1"].power
    pds2_f = results_out["pds2"].power
    cs_f = results_out["cs"].power
    ptot_f = results_out["ptot"].power

    n = length / segment_size

    pds_std_theor = 2 / np.sqrt(n)
    cs_std_theor = np.sqrt(2 / n)

    assert np.isclose(pds1_f.std(), pds_std_theor, rtol=0.1)
    assert np.isclose(pds2_f.std(), pds_std_theor, rtol=0.1)
    assert np.isclose(cs_f.real.std(), cs_std_theor, rtol=0.1)
    assert np.isclose(ptot_f.std(), pds_std_theor, rtol=0.1)


@pytest.mark.parametrize("ctrate", [0.5])
def test_fad_power_spectrum_unknown_alg(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    segment_size = 256.0
    ncounts = int(ctrate * length)
    ev1 = generate_events(length, ncounts)
    ev2 = generate_events(length, ncounts)

    lc1 = generate_deadtime_lc(ev1, dt, tstart=0, tseg=length, deadtime=deadtime)
    lc2 = generate_deadtime_lc(ev2, dt, tstart=0, tseg=length, deadtime=deadtime)

    with pytest.raises(ValueError) as excinfo:
        calculate_FAD_correction(
            lc1,
            lc2,
            segment_size,
            plot=True,
            smoothing_alg="babasone",
            strict=False,
            verbose=False,
            tolerance=0.0001,
        )
    assert "Unknown smoothing algorithm" in str(excinfo.value)


@pytest.mark.parametrize("ctrate", [100])
def test_fad_power_spectrum_non_compliant(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    segment_size = 256.0
    ncounts = int(ctrate * length)
    ev1 = generate_events(length, ncounts)
    ev2 = generate_events(length, int(ncounts * 0.6))

    lc1 = generate_deadtime_lc(ev1, dt, tstart=0, tseg=length, deadtime=deadtime)
    lc2 = generate_deadtime_lc(ev2, dt, tstart=0, tseg=length, deadtime=deadtime)

    with pytest.warns(UserWarning, match="results ARE NOT complying"):
        results = calculate_FAD_correction(
            lc1,
            lc2,
            segment_size,
            plot=True,
            smoothing_alg="gauss",
            strict=False,
            verbose=False,
            tolerance=0.0001,
        )

        is_compliant = results.meta["is_compliant"]

        assert not is_compliant


@pytest.mark.parametrize("ctrate", [50])
def test_fad_power_spectrum_non_compliant_raise(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    segment_size = 256.0
    ncounts = int(ctrate * length)
    ev1 = generate_events(length, ncounts)
    ev2 = generate_events(length, ncounts)

    lc1 = generate_deadtime_lc(ev1, dt, tstart=0, tseg=length, deadtime=deadtime)
    lc2 = generate_deadtime_lc(ev2, dt, tstart=0, tseg=length, deadtime=deadtime)

    with pytest.raises(RuntimeError, match="Results are not compliant, and"):
        with pytest.warns(UserWarning, match="results ARE NOT complying"):
            _ = calculate_FAD_correction(
                lc1,
                lc2,
                segment_size,
                plot=True,
                smoothing_alg="gauss",
                strict=True,
                verbose=False,
                tolerance=0.0001,
            )

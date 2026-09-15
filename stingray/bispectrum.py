import warnings
from collections.abc import Generator, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.linalg import toeplitz

from stingray.base import StingrayObject

from .events import EventList
from .lightcurve import Lightcurve
from .gti import cross_two_gtis, time_intervals_from_gtis
from .utils import create_window, fft2, fftshift, ifftshift
from .fourier import (
    avg_bispectrum_from_iterable,
    avg_bispectrum_from_timeseries,
    avg_cross_bispectrum_from_iterables,
    avg_cross_bispectrum_from_timeseries,
    bicoherence_from_sums,
    get_flux_iterable_from_segments,
)

#: Bispectrum estimation methods. ``"fourier"`` is the direct Fourier-
#: decomposition estimator (Maccarone 2013); ``"cumulant"`` is the legacy
#: indirect estimator (FFT of the 3rd-order cumulant). Auto-bispectrum only.
BISPECTRUM_METHODS = ("fourier", "cumulant")

#: Lag windows accepted by ``method="cumulant"`` (passed to
#: ``stingray.utils.create_window``).
CUMULANT_WINDOWS = (
    "uniform",
    "parzen",
    "hamming",
    "hanning",
    "triangular",
    "welch",
    "blackmann",
    "flat-top",
)

__all__ = [
    "CrossBispectrum",
    "AveragedCrossBispectrum",
    "Bispectrum",
    "AveragedBispectrum",
    "DynamicalCrossBispectrum",
    "DynamicalBispectrum",
]

#: Set once the interface-change notice has been emitted, so it is shown only
#: on the first bispectrum object created in a session (see
#: ``_warn_interface_change``).
_INTERFACE_CHANGE_WARNED = False


def _warn_interface_change():
    """Emit a one-time notice that the bispectrum interface has changed.

    The direct Fourier-decomposition estimator (Maccarone 2013) is now the
    default, replacing the legacy third-order-cumulant method, and the
    ``Bispectrum`` / ``AveragedBispectrum`` and cross-bispectrum classes have
    been reworked. The cumulant estimator is still available via
    ``method="cumulant"``. The notice is shown once per session, the first time
    any bispectrum object is constructed, and points to the documentation for
    the changes.
    """
    global _INTERFACE_CHANGE_WARNED
    if _INTERFACE_CHANGE_WARNED:
        return
    _INTERFACE_CHANGE_WARNED = True
    warnings.warn(
        "The stingray bispectrum interface has changed. The default estimator is "
        "now the direct Fourier-decomposition method (Maccarone 2013) instead of "
        "the third-order-cumulant method, and the Bispectrum/AveragedBispectrum "
        "and cross-bispectrum classes have been reworked. The cumulant estimator "
        "is still available via method='cumulant'. See the bispectrum "
        "documentation for the changes: "
        "https://docs.stingray.science/en/stable/notebooks/Bispectrum/"
        "bispectrum_tutorial.html",
        UserWarning,
        stacklevel=3,
    )


class CrossBispectrum(StingrayObject):
    main_array_attr = "freq"
    type = "crossbispectrum"

    r"""Make a :class:`CrossBispectrum` from up to three (binned) light curves.

    The cross-bispectrum is the higher-order analogue of the cross spectrum. It
    measures quadratic phase coupling *between channels*: for three (real)
    simultaneous time series with per-segment Fourier transforms :math:`X_i`,
    :math:`Y_i`, :math:`Z_i`,

    .. math::

        B(f_1, f_2) = \frac{1}{m} \sum_{i=0}^{m-1}
            X_i(f_1)\, Y_i(f_2)\, Z_i^{*}(f_1 + f_2).

    The :class:`Bispectrum` (auto-bispectrum) is the special case where the
    three inputs are the same light curve, in the same way that
    :class:`stingray.Powerspectrum` is the auto case of
    :class:`stingray.Crossspectrum`. In X-ray timing the usual use is two energy
    bands: e.g. testing whether the variability at ``f1 + f2`` in one band is
    quadratically coupled to ``f1``, ``f2`` in another.

    Unlike the auto-bispectrum, the cross-bispectrum is **not** symmetric under
    ``f1 <-> f2`` (because ``X(f1) Y(f2)`` is not), so it is sampled on the full
    signed frequency plane (``f1``, ``f2`` positive and negative, with
    ``|f1 + f2| <= f_Nyq``) rather than the Nyquist triangle.

    You can also make an empty :class:`CrossBispectrum` object to populate with
    your own data. A single object (no ``segment_size``) uses the whole light
    curve as one segment; for a statistically meaningful cross-bicoherence and
    cross-biphase you normally want :class:`AveragedCrossBispectrum`.

    Parameters
    ----------
    data1, data2, data3 : :class:`stingray.Lightcurve` or :class:`stingray.EventList`, optional
        The three channels, mapped to the factors ``X(f1)``, ``Y(f2)`` and
        ``Z(f1+f2)`` respectively. ``data2`` and ``data3`` default to ``data1``
        (recovering the auto-bispectrum). All three must be simultaneous (same
        ``dt``, same time bins, compatible GTIs). If :class:`EventList`, ``dt``
        must be specified.

    Other Parameters
    ----------------
    dt : float
        The time resolution of the light curves. Only needed for
        :class:`EventList` inputs.

    gti : ``[[gti0_0, gti0_1], ...]``
        Good time intervals. Defaults to the intersection of the inputs' GTIs.

    bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
        Which normalization to use for the ``bicoherence`` attribute. See
        :class:`Bispectrum` for the definitions; the cross denominators are
        :math:`\sum_i |X_i(f_1) Y_i(f_2)|^2` and :math:`\sum_i |Z_i(f_1+f_2)|^2`.

    poisson_subtract : bool, default False
        Subtract the Poisson-noise bias. Only has an effect when
        ``channels_overlap`` is True; for independent channels the Poisson noise
        is uncorrelated between factors and the cross-bispectrum is unbiased.

    bias_subtract : bool, default False
        Subtract the ``~1/M`` statistical bias of the squared bicoherence, where
        ``M`` is the number of averaged segments (Fackrell 1996; Elgar & Guza
        1988). Only defined for the ``"kim_powers"`` and ``"sigl_chamoun"``
        normalizations. Can also be toggled after the fact with
        :meth:`recompute_bicoherence`.

    channels_overlap : bool, default False
        Whether the three channels are the same photon stream. Independent
        energy bands (the usual cross case): leave ``False``.

    skip_checks : bool, default False
        Skip initial checks, for speed or other reasons.

    Attributes
    ----------
    freq : numpy.ndarray
        The (signed) Fourier frequencies the transform samples.

    bispec : numpy.ndarray
        The complex cross-bispectrum, an ``nf x nf`` matrix. The unresolved
        region (``|f1 + f2| > f_Nyq``) is set to ``NaN``.

    bicoherence : numpy.ndarray
        The cross-bicoherence in ``[0, 1]``, in the ``bicoherence_norm``
        convention. Use :meth:`recompute_bicoherence` for a different one.

    bicoherence_norm : str
        The normalization used for ``bicoherence``.

    poisson_subtracted : bool
        Whether the Poisson-noise bias was subtracted.

    bias_subtract : bool
        Whether the ``~1/M`` statistical bias of the squared bicoherence was
        subtracted.

    channels_overlap : bool
        Whether the input channels share the same photons.

    biphase : numpy.ndarray
        The phase of the cross-bispectrum, defined over the full
        :math:`2\pi` interval. Also available as ``bispec_phase``.

    bispec_mag : numpy.ndarray
        Magnitude of the cross-bispectrum.

    bispec_err, biphase_err : numpy.ndarray
        Approximate 1-sigma uncertainties (as in :class:`Bispectrum`).

    df : float
        The frequency resolution.

    m : int
        The number of averaged cross-bispectra.

    n : int
        The number of data points in each segment.

    nphots1, nphots2, nphots3 : float
        The mean photon count per segment in each channel.

    nphots : float
        The geometric mean of ``nphots1``, ``nphots2`` and ``nphots3``.

    References
    ----------
    1) T. J. Maccarone, MNRAS 435, 3547 (2013).

    2) Y. C. Kim and E. J. Powers, IEEE Trans. Plasma Sci. PS-7, 120 (1979).
    """

    def __init__(
        self,
        data1=None,
        data2=None,
        data3=None,
        dt=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        channels_overlap=False,
        skip_checks=False,
    ):
        _warn_interface_change()
        self._type = None
        # Missing channels default to data1 (the auto-bispectrum).
        if data1 is not None:
            if data2 is None:
                data2 = data1
            if data3 is None:
                data3 = data1

        good_input = data1 is not None
        if good_input and not skip_checks:
            good_input = self.initial_checks(data1=data1, data2=data2, data3=data3, dt=dt)

        self.dt = dt
        self.gti = gti
        self.bicoherence_norm = bicoherence_norm
        self.channels_overlap = channels_overlap
        self.poisson_subtracted = poisson_subtract and channels_overlap
        self.bias_subtract = bias_subtract

        if not good_input:
            return self._initialize_empty()

        return self._initialize_from_any_input(
            data1,
            data2,
            data3,
            dt=dt,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            channels_overlap=channels_overlap,
        )

    def initial_checks(self, data1=None, data2=None, data3=None, dt=None, segment_size=None):
        """Run basic checks on the inputs.

        Returns ``True`` if the inputs can be used to build a cross-bispectrum,
        raises otherwise. An empty (``None``) ``data1`` returns ``False`` so that
        an empty object is created.
        """
        if data1 is None:
            return False

        inputs = (data1, data2, data3)
        for data in inputs:
            if isinstance(data, EventList):
                if dt is None:
                    raise ValueError(
                        "If the input is an event list, the time resolution dt "
                        "must be specified."
                    )
            elif isinstance(data, Lightcurve):
                pass
            elif isinstance(data, (tuple, list, Generator)):
                pass
            else:
                raise TypeError(f"Bad input to CrossBispectrum: {type(data)}")

        if not (isinstance(data1, type(data2)) and isinstance(data1, type(data3))):
            raise ValueError("The input channels must all be of the same kind.")

        if segment_size is not None and dt is not None and segment_size < 2 * dt:
            raise ValueError("segment_size must be at least 2 * dt.")

        return True

    def _initialize_from_any_input(
        self,
        data1,
        data2,
        data3,
        dt=None,
        segment_size=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        channels_overlap=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
    ):
        """Initialize the object, dispatching on the type of ``data1``."""
        kwargs = dict(
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            channels_overlap=channels_overlap,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )
        if isinstance(data1, EventList):
            spec = crossbispectrum_from_events(data1, data2, data3, dt, **kwargs)
        elif isinstance(data1, Lightcurve):
            spec = crossbispectrum_from_lightcurve(data1, data2, data3, **kwargs)
        elif isinstance(data1, (tuple, list, Generator)):
            data1, data2, data3 = list(data1), list(data2), list(data3)
            if len(data1) == 0 or not isinstance(data1[0], Lightcurve):  # pragma: no cover
                raise TypeError("Bad inputs to CrossBispectrum")
            dt = data1[0].dt
            spec = crossbispectrum_from_lc_iterable(data1, data2, data3, dt, **kwargs)
        else:  # pragma: no cover
            raise TypeError(f"Bad inputs to CrossBispectrum: {type(data1)}")

        for key, val in spec.__dict__.items():
            setattr(self, key, val)
        return

    def _initialize_empty(self):
        """Set all attributes to ``None`` (or sensible defaults)."""
        self.freq = None
        self.bispec = None
        self.bicoherence = None
        self.bicoherence_norm = getattr(self, "bicoherence_norm", "kim_powers")
        self.poisson_subtracted = getattr(self, "poisson_subtracted", False)
        self.bias_subtract = getattr(self, "bias_subtract", False)
        self.channels_overlap = getattr(self, "channels_overlap", False)
        self.biphase = None
        self.bispec_mag = None
        self.bispec_phase = None
        self.bispec_err = None
        self.biphase_err = None
        self.valid = None
        self.bispec_all = None
        self.bispec_diagonal = None
        self._bicoh_abs_bispec_sum = None
        self._bicoh_denom1 = None
        self._bicoh_denom2 = None
        self._bicoh_sum_abs = None
        self.df = None
        self.dt = None
        self.m = 1
        self.n = None
        self.nphots = None
        self.nphots1 = None
        self.nphots2 = None
        self.nphots3 = None
        self.segment_size = None
        self.gti = None
        # method / cumulant-estimator attributes (auto-bispectrum only)
        self.method = getattr(self, "method", "fourier")
        self.cum3 = None
        self.lags = None
        self.maxlag = None
        self.window = None
        self.scale = None
        return

    def recompute_bicoherence(self, norm=None, bias_subtract=False, inplace=False):
        """Recompute the bicoherence under a different normalization or debiasing.

        Uses the accumulated bispectrum sums stored on the object, so no FFTs
        are recomputed. See :class:`Bispectrum` for the definitions.

        Parameters
        ----------
        norm : {"kim_powers", "sigl_chamoun", "hagihira"}, optional
            The bicoherence normalization. Defaults to the object's current
            ``bicoherence_norm`` (useful when only toggling ``bias_subtract``).

        Other Parameters
        ----------------
        bias_subtract : bool, default False
            Subtract the ``~1/M`` statistical bias of the squared bicoherence
            (``M`` = number of averaged segments; Fackrell 1996, Elgar & Guza
            1988). See :func:`stingray.fourier.bicoherence_from_sums`. Only
            defined for ``"kim_powers"`` and ``"sigl_chamoun"``.
        inplace : bool, default False
            If ``True``, also overwrite ``self.bicoherence``,
            ``self.bicoherence_norm`` and ``self.bias_subtract``.

        Returns
        -------
        bicoherence : numpy.ndarray
            The recomputed bicoherence.
        """
        if getattr(self, "_bicoh_denom1", None) is None:
            raise ValueError("This bispectrum has no data to compute a bicoherence from.")
        if norm is None:
            norm = self.bicoherence_norm
        bicoh = bicoherence_from_sums(
            norm,
            self._bicoh_abs_bispec_sum,
            self._bicoh_denom1,
            self._bicoh_denom2,
            self._bicoh_sum_abs,
            valid=self.valid,
            n_seg=self.m,
            bias_subtract=bias_subtract,
        )
        if inplace:
            self.bicoherence = bicoh
            self.bicoherence_norm = norm.lower()
            self.bias_subtract = bias_subtract
        return bicoh

    @staticmethod
    def from_lightcurve(
        lc1,
        lc2=None,
        lc3=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        channels_overlap=False,
        silent=False,
    ):
        """Calculate a :class:`CrossBispectrum` from light curves."""
        return crossbispectrum_from_lightcurve(
            lc1,
            lc2,
            lc3,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            channels_overlap=channels_overlap,
            silent=silent,
        )

    @staticmethod
    def from_events(
        events1,
        events2=None,
        events3=None,
        dt=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        channels_overlap=False,
        silent=False,
    ):
        """Calculate a :class:`CrossBispectrum` from event lists."""
        return crossbispectrum_from_events(
            events1,
            events2,
            events3,
            dt,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            channels_overlap=channels_overlap,
            silent=silent,
        )

    @staticmethod
    def from_time_array(
        times1,
        times2,
        times3,
        dt,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        channels_overlap=False,
        silent=False,
    ):
        """Calculate a :class:`CrossBispectrum` from arrays of event times."""
        return crossbispectrum_from_time_array(
            times1,
            times2,
            times3,
            dt,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            channels_overlap=channels_overlap,
            silent=silent,
        )

    def plot_mag(self, ax=None, save=False, filename=None):
        """Plot the magnitude of the bispectrum as a function of frequency."""
        return self._plot_matrix(
            self.bispec_mag, "Bispectrum Magnitude", ax, save, filename, "bispec_mag.png"
        )

    def plot_phase(self, ax=None, save=False, filename=None):
        """Plot the biphase as a function of frequency."""
        return self._plot_matrix(self.biphase, "Biphase", ax, save, filename, "bispec_phase.png")

    def plot_bicoherence(self, ax=None, save=False, filename=None, log=False):
        """Plot the bicoherence as a function of frequency.

        Set ``log=True`` to plot :math:`\\log_{10}` of the bicoherence, which
        stretches the low end and makes weak couplings easier to see. Since the
        bicoherence lies in ``[0, 1]``, the log is non-positive; bins that are
        exactly zero (and the unresolved region) are left blank.
        """
        return self._plot_matrix(
            self.bicoherence, "Bicoherence", ax, save, filename, "bicoherence.png", log=log
        )

    def _plot_matrix(self, matrix, title, ax, save, filename, default_filename, log=False):
        """Shared helper for the 2D bispectrum plots."""
        if matrix is None:
            raise ValueError("This bispectrum has no data to plot.")

        if ax is None:
            _, ax = plt.subplots()

        if log:
            with np.errstate(divide="ignore", invalid="ignore"):
                matrix = np.log10(matrix)
            # log10 of zero/negative gives -inf/NaN; blank those bins out.
            matrix = np.where(np.isfinite(matrix), matrix, np.nan)
            title = f"log10({title})"

        # ``matrix[i, j]`` is indexed (f1, f2); transpose so that f1 is on the
        # x-axis (matters only for a non-symmetric, i.e. cross, bispectrum).
        cont = ax.contourf(self.freq, self.freq, matrix.T, 100, cmap=plt.cm.Spectral_r)
        ax.figure.colorbar(cont, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Frequency 1 (Hz)")
        ax.set_ylabel("Frequency 2 (Hz)")

        if save:
            ax.figure.savefig(filename if filename is not None else default_filename)
        return ax

    def plot_jellyfish(
        self,
        f0=None,
        freqs=None,
        bicoherence_levels=(0.01, 0.05),
        fundamental_color="tab:blue",
        subharmonic_color="tab:orange",
        other_color="0.6",
        ax=None,
        save=False,
        filename=None,
    ):
        r"""Draw a "jellyfish plot" of the (auto-)bispectrum (Nathan et al. 2022).

        For each frequency :math:`\nu` on the diagonal (:math:`f_1 = f_2 = \nu`,
        which couples :math:`\nu` and its harmonic :math:`2\nu`), the per-segment
        triple products are accumulated segment by segment and the running
        (cumulative) sum is traced as a path in the complex plane, normalized so
        the amplitude of its end point is the bicoherence (Sigl & Chamoun
        convention) and its angle is the biphase.

        Phase-coupled frequencies walk outward into a "tentacle"; uncoupled ones
        random-walk near the origin, forming the "body". Reference circles of
        constant bicoherence give the scale.

        Requires an averaged (cross-)bispectrum built with ``save_diagonal=True``
        (which keeps only the diagonal the plot needs) or ``save_all=True``.

        Parameters
        ----------
        f0 : float, optional
            A reference (e.g. QPO fundamental) frequency to highlight. The
            diagonal path closest to ``f0`` is drawn in ``fundamental_color``,
            and the path closest to ``f0 / 2`` in ``subharmonic_color``. If
            ``None``, every path is drawn in ``other_color``.

        Other Parameters
        ----------------
        freqs : iterable of float, optional
            The diagonal frequencies to draw. Defaults to every resolved
            diagonal frequency.
        bicoherence_levels : iterable of float, default ``(0.01, 0.05)``
            Radii, in bicoherence units, of the reference circles.
        fundamental_color, subharmonic_color, other_color : color
            Colors for the fundamental, subharmonic and remaining paths.
        ax : ``matplotlib.axes.Axes``, optional
            Axes to draw onto. A new one is created if ``None``.
        save : bool, default ``False``
            If ``True``, save the figure to ``filename``.
        filename : str, optional
            File name to save to. Defaults to ``bispec_jellyfish.png``.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            The axes with the plot.
        """
        # The jellyfish only needs the per-segment diagonal. Prefer the light
        # ``bispec_diagonal`` store (save_diagonal=True); fall back to the full
        # ``bispec_all`` cube (save_all=True).
        diag_store = getattr(self, "bispec_diagonal", None)
        full_store = getattr(self, "bispec_all", None)
        if diag_store is None and full_store is None:
            raise ValueError(
                "plot_jellyfish needs the per-segment bispectra. Build the "
                "averaged bispectrum with save_diagonal=True (light) or save_all=True."
            )

        nf = self.freq.size
        diag = np.arange(nf)
        valid_diag = diag[np.diag(self.valid)]

        if freqs is not None:
            wanted = [int(np.argmin(np.abs(self.freq - f))) for f in np.atleast_1d(freqs)]
            valid_diag = np.array([j for j in wanted if self.valid[j, j]])

        if valid_diag.size == 0:
            raise ValueError("No resolved diagonal frequencies to plot.")

        norm = np.sqrt(
            self._bicoh_denom1[valid_diag, valid_diag] * self._bicoh_denom2[valid_diag, valid_diag]
        )

        # Per-segment triples along the diagonal, shape (m, n_freq).
        if diag_store is not None:
            diag_triples = np.asarray(diag_store)[:, valid_diag]
        else:
            diag_triples = np.asarray(full_store)[:, valid_diag, valid_diag]

        # Cumulative sum along the diagonal, starting from the origin.
        cumsum = np.cumsum(diag_triples, axis=0)
        paths = np.vstack([np.zeros(valid_diag.size), cumsum]) / norm[np.newaxis, :]

        j_fund = j_sub = None
        if f0 is not None:
            j_fund = valid_diag[np.argmin(np.abs(self.freq[valid_diag] - f0))]
            j_sub = valid_diag[np.argmin(np.abs(self.freq[valid_diag] - f0 / 2.0))]

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        # Reference circles of constant bicoherence
        theta = np.linspace(0, 2 * np.pi, 200)
        for level in bicoherence_levels:
            ax.plot(
                level * np.cos(theta),
                level * np.sin(theta),
                ls="--",
                color="0.7",
                lw=1,
                zorder=1,
                label=f"bicoherence {level:g}",
            )

        # Draw the "other" paths first, highlighted ones on top
        fund_label_done = sub_label_done = False
        for col, j in enumerate(valid_diag):
            path = paths[:, col]
            if j == j_fund:
                color, lw, z = fundamental_color, 1.8, 4
                label = None if fund_label_done else "QPO fundamental"
                fund_label_done = True
            elif j == j_sub:
                color, lw, z = subharmonic_color, 1.8, 4
                label = None if sub_label_done else "subharmonic"
                sub_label_done = True
            else:
                color, lw, z = other_color, 0.8, 2
                label = None
            ax.plot(path.real, path.imag, color=color, lw=lw, alpha=0.9, zorder=z, label=label)
            ax.plot(path.real[-1], path.imag[-1], ".", color=color, ms=6, zorder=z + 1)

        lim = 1.15 * max(np.max(np.abs(paths)), max(bicoherence_levels))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.9", lw=0.8, zorder=0)
        ax.axvline(0, color="0.9", lw=0.8, zorder=0)
        ax.set_xlabel("Re(B)")
        ax.set_ylabel("Im(B)")
        ax.set_title("Bispectrum jellyfish plot")
        ax.legend(loc="upper right", fontsize="small")

        if save:
            ax.figure.savefig(filename if filename is not None else "bispec_jellyfish.png")
        return ax


class AveragedCrossBispectrum(CrossBispectrum):
    type = "crossbispectrum"

    r"""Make an averaged cross-bispectrum from three simultaneous light curves.

    Each channel is split into segments of length ``segment_size`` on a common
    GTI, a cross-bispectrum is computed per segment, and the results are
    averaged. See :class:`CrossBispectrum` for the estimator definition, and
    :class:`AveragedBispectrum` for the (auto) memory-saving ``save_all`` /
    ``save_diagonal`` options, which apply here unchanged.

    Parameters
    ----------
    data1, data2, data3 : :class:`stingray.Lightcurve`, iterable, or :class:`stingray.EventList`
        The three channels. ``data2``/``data3`` default to ``data1``.

    segment_size : float
        The size, in seconds, of each averaged segment.

    Other Parameters
    ----------------
    See :class:`CrossBispectrum`, plus ``dt``, ``silent``, ``save_all`` and
    ``save_diagonal`` as in :class:`AveragedBispectrum`.
    """

    def __init__(
        self,
        data1=None,
        data2=None,
        data3=None,
        segment_size=None,
        gti=None,
        dt=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        channels_overlap=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
        skip_checks=False,
    ):
        _warn_interface_change()
        self._type = None
        if data1 is not None:
            if data2 is None:
                data2 = data1
            if data3 is None:
                data3 = data1

        good_input = data1 is not None
        if good_input and not skip_checks:
            good_input = self.initial_checks(
                data1=data1, data2=data2, data3=data3, dt=dt, segment_size=segment_size
            )

        self.dt = dt
        self.gti = gti
        self.bicoherence_norm = bicoherence_norm
        self.channels_overlap = channels_overlap
        self.poisson_subtracted = poisson_subtract and channels_overlap
        self.bias_subtract = bias_subtract
        self.segment_size = segment_size
        self.save_all = save_all

        if not good_input:
            return self._initialize_empty()

        return self._initialize_from_any_input(
            data1,
            data2,
            data3,
            dt=dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            channels_overlap=channels_overlap,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )

    def initial_checks(self, data1=None, data2=None, data3=None, dt=None, segment_size=None):
        if data1 is not None and segment_size is None:
            raise ValueError("segment_size must be specified for an averaged bispectrum.")
        return super().initial_checks(
            data1=data1, data2=data2, data3=data3, dt=dt, segment_size=segment_size
        )

    @staticmethod
    def from_lightcurve(
        lc1,
        lc2,
        lc3,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        channels_overlap=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
    ):
        """Calculate an :class:`AveragedCrossBispectrum` from light curves."""
        return crossbispectrum_from_lightcurve(
            lc1,
            lc2,
            lc3,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            channels_overlap=channels_overlap,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )

    @staticmethod
    def from_events(
        events1,
        events2,
        events3,
        dt,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        channels_overlap=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
    ):
        """Calculate an :class:`AveragedCrossBispectrum` from event lists."""
        return crossbispectrum_from_events(
            events1,
            events2,
            events3,
            dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            channels_overlap=channels_overlap,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )


class Bispectrum(CrossBispectrum):
    type = "bispectrum"

    r"""Make a :class:`Bispectrum` (auto-bispectrum) from a (binned) light curve.

    The auto-bispectrum is the special case of the :class:`CrossBispectrum` in
    which the three inputs are the same light curve, exactly as
    :class:`stingray.Powerspectrum` is the auto case of
    :class:`stingray.Crossspectrum`. It measures quadratic phase coupling within
    a single time series. For ``m`` averaged segments,

    .. math::

        B(f_1, f_2) = \frac{1}{m} \sum_{i=0}^{m-1}
            X_i(f_1)\, X_i(f_2)\, X_i^{*}(f_1 + f_2)

    Because ``X(f1) X(f2)`` is symmetric under ``f1 <-> f2``, the auto case is
    sampled only on the positive Nyquist triangle (unlike the full signed grid
    of the cross case).

    A single :class:`Bispectrum` uses the whole light curve as one segment; use
    :class:`AveragedBispectrum` for a statistically meaningful bicoherence and
    biphase.

    Parameters
    ----------
    data : :class:`stingray.Lightcurve` or :class:`stingray.events.EventList`, optional
        The light curve or event list to be Fourier-transformed. If an
        :class:`EventList` is given, ``dt`` must be specified.

    Other Parameters
    ----------------
    dt : float
        The time resolution of the light curve. Only needed for an
        :class:`EventList`.

    bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
        Which normalization to use for the ``bicoherence`` attribute. All lie in
        ``[0, 1]``. With :math:`T_i = X_i(f_1) X_i(f_2) X_i^{*}(f_1+f_2)`:

        ``"kim_powers"``
            The **squared** bicoherence of Kim & Powers (1979):
            :math:`b^2 = |\sum_i T_i|^2 / (\sum_i |X_i(f_1)X_i(f_2)|^2 \sum_i |X_i(f_1+f_2)|^2)`.
        ``"sigl_chamoun"``
            Sigl & Chamoun (1994); the unsquared square root of the above.
        ``"hagihira"``
            Hagihira (2001) / Hayashi (2007);
            :math:`b = |\sum_i T_i| / \sum_i |T_i|`.

    poisson_subtract : bool, default False
        Subtract the Poisson-noise bias per segment (Wirnitzer 1985):
        :math:`T_i - |X_i(f_1)|^2 - |X_i(f_2)|^2 - |X_i(f_1+f_2)|^2 + 2N_i`.
        Only appropriate for photon-counting light curves given in counts.

    bias_subtract : bool, default False
        Subtract the ``~1/M`` statistical bias of the squared bicoherence, where
        ``M`` is the number of averaged segments (Fackrell 1996; Elgar & Guza
        1988). Only defined for the ``"kim_powers"`` and ``"sigl_chamoun"``
        normalizations. Can also be toggled after the fact with
        :meth:`recompute_bicoherence`.

    method : {"fourier", "cumulant"}, default "fourier"
        Which estimator to use. ``"fourier"`` is the direct Fourier-decomposition
        estimator (Maccarone 2013) described above. ``"cumulant"`` is the legacy
        *indirect* estimator: the bispectrum as the 2-D Fourier transform of the
        3rd-order cumulant of the light curve (Rao & Gabr 1984), retained for
        backward comparison. It uses ``maxlag``, ``window`` and ``scale`` below,
        ignores ``bicoherence_norm`` / ``poisson_subtract`` / ``bias_subtract``,
        and produces no bicoherence; the auto-bispectrum only.

    maxlag : int, optional
        (``method="cumulant"`` only.) Maximum lag of the 3rd-order cumulant.
        Defaults to half the light-curve length.

    window : str, optional
        (``method="cumulant"`` only.) Lag window to taper the cumulant
        (``"parzen"``, ``"hamming"``, ``"hanning"``, ...); ``None`` applies no
        window.

    scale : {"biased", "unbiased"}, default "biased"
        (``method="cumulant"`` only.) Normalization of the 3rd-order cumulant.

    skip_checks : bool, default False
        Skip initial checks.

    lc : :class:`stingray.Lightcurve`, optional
        For backwards compatibility only. Deprecated; use ``data``.

    Attributes
    ----------
    See :class:`CrossBispectrum`. ``nphots1 = nphots2 = nphots3 = nphots``. With
    ``method="cumulant"`` the extra attributes ``cum3`` (the 3rd-order cumulant),
    ``lags``, ``maxlag``, ``window`` and ``scale`` are set, and ``bicoherence``
    is ``None``.

    References
    ----------
    Rao, T. S. & Gabr, M. M., *An Introduction to Bispectral Analysis and
    Bilinear Time Series Models*, Lecture Notes in Statistics 24 (1984).
    """

    def __init__(
        self,
        data=None,
        dt=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        method="fourier",
        maxlag=None,
        window=None,
        scale="biased",
        skip_checks=False,
        lc=None,
    ):
        _warn_interface_change()
        self._type = None
        if lc is not None:
            warnings.warn("The lc keyword is now deprecated. Use data instead", DeprecationWarning)
        if data is None:
            data = lc

        if method not in BISPECTRUM_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose one of {BISPECTRUM_METHODS}.")
        self.method = method

        good_input = data is not None
        if good_input and not skip_checks:
            good_input = self.initial_checks(data=data, dt=dt)

        self.dt = dt
        self.gti = gti
        self.bicoherence_norm = bicoherence_norm
        self.poisson_subtracted = poisson_subtract
        self.bias_subtract = bias_subtract
        self.channels_overlap = True  # the auto case is fully overlapping

        if not good_input:
            return self._initialize_empty()

        if method == "cumulant":
            return self._make_cumulant_bispectrum(
                data, dt=dt, maxlag=maxlag, window=window, scale=scale
            )

        return self._initialize_from_any_input(
            data,
            dt=dt,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
        )

    def initial_checks(self, data=None, dt=None, segment_size=None):
        """Basic checks on a single (auto) input."""
        if data is None:
            return False
        if isinstance(data, EventList):
            if dt is None:
                raise ValueError(
                    "If the input is an event list, the time resolution dt must be specified."
                )
        elif isinstance(data, (Lightcurve, tuple, list, Generator)):
            pass
        else:
            raise TypeError(f"Bad input to Bispectrum: {type(data)}")
        if segment_size is not None and dt is not None and segment_size < 2 * dt:
            raise ValueError("segment_size must be at least 2 * dt.")
        return True

    def _initialize_from_any_input(
        self,
        data,
        dt=None,
        segment_size=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
    ):
        """Initialize from a single input, using the auto-bispectrum path."""
        kwargs = dict(
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )
        if isinstance(data, EventList):
            spec = bispectrum_from_events(data, dt, **kwargs)
        elif isinstance(data, Lightcurve):
            spec = bispectrum_from_lightcurve(data, **kwargs)
        elif isinstance(data, (tuple, list, Generator)):
            data = list(data)
            if len(data) == 0 or not isinstance(data[0], Lightcurve):  # pragma: no cover
                raise TypeError("Bad inputs to Bispectrum")
            dt = data[0].dt
            spec = bispectrum_from_lc_iterable(data, dt, **kwargs)
        else:  # pragma: no cover
            raise TypeError(f"Bad inputs to Bispectrum: {type(data)}")

        for key, val in spec.__dict__.items():
            setattr(self, key, val)
        return

    @staticmethod
    def from_lightcurve(
        lc,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
    ):
        """Calculate a :class:`Bispectrum` from a light curve."""
        return bispectrum_from_lightcurve(
            lc,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
        )

    @staticmethod
    def from_events(
        events,
        dt,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
    ):
        """Calculate a :class:`Bispectrum` from an event list."""
        return bispectrum_from_events(
            events,
            dt,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
        )

    @staticmethod
    def from_time_array(
        times,
        dt,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
    ):
        """Calculate a :class:`Bispectrum` from an array of event times."""
        return bispectrum_from_time_array(
            times,
            dt,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
        )

    @staticmethod
    def from_stingray_timeseries(
        ts,
        flux_attr,
        error_flux_attr=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
    ):
        """Calculate a :class:`Bispectrum` from a time series."""
        return bispectrum_from_stingray_timeseries(
            ts,
            flux_attr,
            error_flux_attr=error_flux_attr,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
        )

    # -- legacy indirect (3rd-order cumulant) estimator --------------------
    #
    # The following methods implement ``method="cumulant"``: the bispectrum as
    # the 2-D Fourier transform of the 3rd-order cumulant of the light curve.
    # This is the original stingray algorithm, retained for comparison with the
    # Fourier-decomposition default. Auto-bispectrum only.

    @staticmethod
    def _cumulant_lightcurve(data, dt):
        """Return a :class:`Lightcurve` for the cumulant estimator."""
        if isinstance(data, Lightcurve):
            return data
        if isinstance(data, EventList):
            if dt is None:
                raise ValueError("dt must be specified for an EventList with method='cumulant'.")
            return data.to_lc(dt)
        raise TypeError("method='cumulant' accepts a Lightcurve or EventList.")

    @staticmethod
    def _validate_cumulant_params(n, maxlag, window, scale):
        """
        Validate/normalize the cumulant parameters.

        Parameters
        ----------
        n : int
            The length of the signal.
        maxlag : int or None
            The maximum lag to compute the cumulant over. If ``None``,
            defaults to ``n // 2``.
        window : str or None
            The window function to apply. Must be one of ``CUMULANT_WINDOWS``
            or ``None``.
        scale : str
            The scaling method, either ``'biased'`` or ``'unbiased'``.

        Returns
        -------
        maxlag : int
            The validated maximum lag.
        window : str or None
            The validated window function.
        scale : str
            The validated scaling method.
        """
        if not isinstance(scale, str) or scale.lower() not in ("biased", "unbiased"):
            raise ValueError("scale must be 'biased' or 'unbiased'.")
        scale = scale.lower()
        if window is not None:
            if not isinstance(window, str):
                raise TypeError("window must be a string.")
            if window.lower() not in CUMULANT_WINDOWS:
                raise ValueError(f"Unknown window '{window}'. Choose one of {CUMULANT_WINDOWS}.")
            window = window.lower()
        if maxlag is None:
            maxlag = int(n / 2)
        else:
            if not np.issubdtype(type(maxlag), np.integer):
                raise ValueError("maxlag must be an integer.")
            maxlag = abs(int(maxlag))
        if maxlag < 1 or maxlag >= n:
            raise ValueError("maxlag must be between 1 and the segment length.")
        return maxlag, window, scale

    @staticmethod
    def _cumulant3(signal, n, maxlag):
        """
        Raw 3rd-order cumulant (a ``2*maxlag+1`` square) of a ``1 x n`` signal.

        Parameters
        ----------
        signal : array_like
            The input signal, shape (1, n).
        n : int
            The length of the signal.
        maxlag : int
            The maximum lag to compute the cumulant over.

        Returns
        -------
        cum3 : ndarray
            The raw 3rd-order cumulant, shape (2*maxlag+1, 2*maxlag+1).
        """
        # Initializes cumulant matrix
        cum3 = np.zeros((2 * maxlag + 1, 2 * maxlag + 1))
        # Define indices for the Toeplitz matrix
        ind = np.arange((n - maxlag) - 1, n)
        ind_t = np.arange(maxlag, n)
        zero_maxlag = np.zeros((1, maxlag))
        zero_maxlag_t = zero_maxlag.transpose()

        sig = signal.transpose()

        rev_signal = np.array([signal[0][::-1]])
        col = np.concatenate((sig[ind], zero_maxlag_t), axis=0)
        row = np.concatenate((rev_signal[0][ind_t], zero_maxlag[0]), axis=0)

        # Converts to Toeplitz matrix and calculates the cumulant
        toep = toeplitz(np.ravel(col), np.ravel(row))
        rev_signal = np.repeat(rev_signal, [2 * maxlag + 1], axis=0)
        return cum3 + np.matmul(np.multiply(toep, rev_signal), toep.transpose())

    @staticmethod
    def _normalize_cumulant3(cum3, n, maxlag, scale):
        """
        Biased or unbiased normalization of the 3rd-order cumulant.

        Parameters
        ----------
        cum3 : ndarray
            The raw 3rd-order cumulant, shape (2*maxlag+1, 2*maxlag+1).
        n : int
            The length of the signal.
        maxlag : int
            The maximum lag to compute the cumulant over.
        scale : str
            The normalization scale, either "biased" or "unbiased".

        Returns
        -------
        cum3 : ndarray
            The normalized 3rd-order cumulant, shape (2*maxlag+1, 2*maxlag+1).
        """

        # Biased normalization is just the raw cumulant divided by n.
        if scale == "biased":
            return cum3 / n
        else:  # Ubiased normalization of cumulant
            maxlag1 = maxlag + 1

            scal_matrix = np.zeros((maxlag1, maxlag1), dtype="int64")

            for k in range(maxlag1):
                maxlag1k = maxlag1 - (k + 1)
                scal_matrix[k, k:maxlag1] = np.tile(n - maxlag1k, (1, maxlag1k + 1))
            scal_matrix += np.triu(scal_matrix, k=1).transpose()

            maxlag1ind = np.arange(maxlag - 1, -1, -1)
            lagdiff = n - maxlag1

            col = np.arange(lagdiff, n - 1)
            col = np.reshape(col, (1, len(col))).transpose()
            row = np.arange(lagdiff, (n - 2 * maxlag) - 1, -1)
            row = np.reshape(row, (1, len(row)))

            toep_matrix = toeplitz(np.ravel(col), np.ravel(row))

            conc_mat = np.array([scal_matrix[maxlag, maxlag1ind]])

            join_matrix = np.concatenate((toep_matrix, conc_mat), axis=0)
            scal_matrix = np.concatenate((scal_matrix, join_matrix), axis=1)
            co_mat = scal_matrix[maxlag1ind, :]
            co_mat = co_mat[:, np.arange(2 * maxlag, -1, -1)]

            scal_matrix = np.concatenate((scal_matrix, co_mat), axis=0)
            scal_matrix[scal_matrix < 1] = 1
            return np.divide(cum3, scal_matrix)

    @staticmethod
    def _cumulant_window(maxlag, window_name):
        """
        2-D lag window for the cumulant, or ``None``.

        Parameters
        ----------
        maxlag : int
            The maximum lag to compute the cumulant over.
        window_name : str or None
            The name of the window to apply. If ``None``, no window is applied.

        Returns
        -------
        window : ndarray or None
            The 2-D lag window, shape (2*maxlag+1, 2*maxlag+1), or ``None`` if
            no window is applied.
        """
        if window_name is None:
            return None
        N = 2 * maxlag + 1
        window_even = create_window(N, window_name)
        window2d = np.array([window_even] * N)
        window = np.zeros(N)
        window[: maxlag + 1] = window_even[maxlag:]
        window[maxlag:] = 0
        row = np.concatenate(([window[0]], np.zeros(2 * maxlag)))
        toep_matrix = toeplitz(np.ravel(window), np.ravel(row))
        toep_matrix += np.tril(toep_matrix, -1).transpose()
        return toep_matrix[..., ::-1] * window2d * window2d.transpose()

    def _finalize_cumulant(self, cum3, n, maxlag, dt, window_name, scale):
        """
        Populate the object from a (normalized) cumulant matrix.

        Parameters
        ----------
        cum3 : ndarray
            The normalized 3rd-order cumulant, shape (2*maxlag+1, 2*maxlag+1).
        n : int
            The length of the signal.
        maxlag : int
            The maximum lag to compute the cumulant over.
        dt : float
            The time resolution of the light curve.
        window_name : str or None
            The name of the window applied to the cumulant. If ``None``, no
            window is applied.
        scale : str
            The normalization scale, either "biased" or "unbiased".
        """
        lagindex = np.arange(-maxlag, maxlag + 1)
        self.lags = lagindex * dt
        self.freq = 0.5 * (1.0 / dt) * lagindex / maxlag
        window = self._cumulant_window(maxlag, window_name)
        if window is None:
            bispec = fftshift(fft2(ifftshift(cum3)))
        else:
            bispec = fftshift(fft2(ifftshift(cum3 * window)))
        self.cum3 = cum3
        self.maxlag = maxlag
        self.window = window_name
        self.scale = scale
        self.bispec = bispec
        self.bispec_mag = np.abs(bispec)
        self.biphase = np.angle(bispec)
        self.bispec_phase = self.biphase
        self.n = n
        self.df = float(self.freq[1] - self.freq[0]) if self.freq.size > 1 else None
        self.bicoherence = None

    def _make_cumulant_bispectrum(self, data, dt=None, maxlag=None, window=None, scale="biased"):
        """
        Single (whole light curve) cumulant bispectrum.

        Parameters
        ----------
        data : :class:`stingray.Lightcurve` or :class:`stingray.events.EventList`
            The light curve or event list to be Fourier-transformed.
        dt : float, optional
            The time resolution of the light curve. Only needed for an
            :class:`EventList`.
        maxlag : int, optional
            Maximum lag of the 3rd-order cumulant. Defaults to half the
            light-curve length.
        window : str, optional
            Lag window to taper the cumulant (``"parzen"``, ``"hamming"``,
            ``"hanning"``, ...); ``None`` applies no window.
        scale : {"biased", "unbiased"}, default "biased"
            Normalization of the 3rd-order cumulant.
        """
        lc = self._cumulant_lightcurve(data, dt)
        counts = np.asarray(lc.counts, dtype=float)
        n = counts.size
        maxlag, window, scale = self._validate_cumulant_params(n, maxlag, window, scale)
        signal = (counts - counts.mean()).reshape(1, n)
        cum3 = self._normalize_cumulant3(self._cumulant3(signal, n, maxlag), n, maxlag, scale)
        self.dt = lc.dt
        self.m = 1
        self.method = "cumulant"
        self._finalize_cumulant(cum3, n, maxlag, lc.dt, window, scale)
        return

    def plot_cum3(self, ax=None, save=False, filename=None):
        """
        Plot the 3rd-order cumulant as a function of lag (``method="cumulant"``
        only).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If ``None``, a new figure and axes are
            created.
        save : bool, default False
            If ``True``, save the figure to a file.
        filename : str, optional
            The filename to save the figure to. If ``None``, defaults to
            "bispec_cum3.png".
        """
        if getattr(self, "cum3", None) is None:
            raise ValueError("plot_cum3 is only available for method='cumulant'.")
        if ax is None:
            _, ax = plt.subplots()
        cont = ax.contourf(self.lags, self.lags, self.cum3, 100, cmap=plt.cm.Spectral_r)
        ax.figure.colorbar(cont, ax=ax)
        ax.set_title("3rd Order Cumulant")
        ax.set_xlabel("Lag 1")
        ax.set_ylabel("Lag 2")
        if save:
            ax.figure.savefig(filename if filename is not None else "bispec_cum3.png")
        return ax


class AveragedBispectrum(AveragedCrossBispectrum, Bispectrum):
    type = "bispectrum"

    r"""Make an averaged (auto-)bispectrum from a light curve or event list.

    The light curve is split into segments of length ``segment_size``, an
    auto-bispectrum is computed per segment, and the results are averaged (see
    :class:`Bispectrum`). Averaging is what makes the bicoherence and biphase
    statistically meaningful.

    Parameters
    ----------
    data : :class:`stingray.Lightcurve`, iterable of them, or :class:`stingray.events.EventList`
        The light curve data to be Fourier-transformed.

    segment_size : float
        The size, in seconds, of each averaged segment.

    Other Parameters
    ----------------
    gti : ``[[gti0_0, gti0_1], ...]``
        Good time intervals.
    dt : float
        The time resolution of the light curve (needed for an :class:`EventList`).
    silent : bool, default False
        Do not show a progress bar.
    save_all : bool, default False
        Save all intermediate 2D bispectra (under ``bispec_all``, shape
        ``m x nf x nf``). Large; use with care.
    save_diagonal : bool, default False
        Save only the diagonal of each intermediate bispectrum (under
        ``bispec_diagonal``, shape ``m x nf``). Enough for
        :meth:`plot_jellyfish`, at a fraction of the ``save_all`` memory.
    skip_checks : bool, default False
        Skip initial checks.
    lc : :class:`stingray.Lightcurve`, optional
        For backwards compatibility only. Deprecated; use ``data``.
    """

    def __init__(
        self,
        data=None,
        segment_size=None,
        gti=None,
        dt=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        method="fourier",
        maxlag=None,
        window=None,
        scale="biased",
        silent=False,
        save_all=False,
        save_diagonal=False,
        skip_checks=False,
        lc=None,
    ):
        _warn_interface_change()
        self._type = None
        if lc is not None:
            warnings.warn("The lc keyword is now deprecated. Use data instead", DeprecationWarning)
        if data is None:
            data = lc

        if method not in BISPECTRUM_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose one of {BISPECTRUM_METHODS}.")
        self.method = method

        good_input = data is not None
        if good_input and not skip_checks:
            good_input = self.initial_checks(data=data, dt=dt, segment_size=segment_size)

        self.dt = dt
        self.gti = gti
        self.bicoherence_norm = bicoherence_norm
        self.poisson_subtracted = poisson_subtract
        self.bias_subtract = bias_subtract
        self.channels_overlap = True
        self.segment_size = segment_size
        self.save_all = save_all

        if not good_input:
            return self._initialize_empty()

        if method == "cumulant":
            return self._make_cumulant_bispectrum_averaged(
                data, dt=dt, segment_size=segment_size, maxlag=maxlag, window=window, scale=scale
            )

        if isinstance(data, Generator):
            warnings.warn(
                "The averaged bispectrum from a generator of light curves "
                "pre-allocates the full list of light curves, losing the "
                "advantage of lazy loading. If that matters to you, use the "
                "AveragedBispectrum.from_lc_iterable static method, specifying "
                "the sampling time dt."
            )
            data = list(data)

        return self._initialize_from_any_input(
            data,
            dt=dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )

    def initial_checks(self, data=None, dt=None, segment_size=None):
        if data is not None and segment_size is None:
            raise ValueError("segment_size must be specified for an AveragedBispectrum.")
        return Bispectrum.initial_checks(self, data=data, dt=dt, segment_size=segment_size)

    def _make_cumulant_bispectrum_averaged(
        self, data, dt=None, segment_size=None, maxlag=None, window=None, scale="biased"
    ):
        """Segment-averaged cumulant bispectrum: average the per-segment 3rd-order
        cumulant over segments, then Fourier-transform."""
        lc = self._cumulant_lightcurve(data, dt)
        dt = lc.dt
        n_bin = int(np.rint(segment_size / dt))
        maxlag, window, scale = self._validate_cumulant_params(n_bin, maxlag, window, scale)
        gti = self.gti if self.gti is not None else lc.gti
        flux_iterable = get_flux_iterable_from_segments(
            lc.time, np.asarray(gti), segment_size, n_bin, dt=dt, fluxes=lc.counts
        )
        cum3_sum = None
        m = 0
        for flux in flux_iterable:
            flux = np.asarray(flux, dtype=float)
            if flux.size < n_bin or np.all(flux == 0):
                continue
            signal = (flux - flux.mean()).reshape(1, flux.size)
            c = self._normalize_cumulant3(
                self._cumulant3(signal, flux.size, maxlag), flux.size, maxlag, scale
            )
            cum3_sum = c if cum3_sum is None else cum3_sum + c
            m += 1
        if cum3_sum is None:
            raise ValueError("No usable segments were found for the cumulant bispectrum.")
        cum3 = cum3_sum / m
        self.dt = dt
        self.m = m
        self.method = "cumulant"
        self._finalize_cumulant(cum3, n_bin, maxlag, dt, window, scale)
        return

    @staticmethod
    def from_lightcurve(
        lc,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from a light curve."""
        return bispectrum_from_lightcurve(
            lc,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )

    @staticmethod
    def from_events(
        events,
        dt,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from an event list."""
        return bispectrum_from_events(
            events,
            dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )

    @staticmethod
    def from_time_array(
        times,
        dt,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from an array of event times."""
        return bispectrum_from_time_array(
            times,
            dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )

    @staticmethod
    def from_stingray_timeseries(
        ts,
        flux_attr,
        segment_size,
        error_flux_attr=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from a time series."""
        return bispectrum_from_stingray_timeseries(
            ts,
            flux_attr,
            error_flux_attr=error_flux_attr,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )

    @staticmethod
    def from_lc_iterable(
        iter_lc,
        dt,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        silent=False,
        save_all=False,
        save_diagonal=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from an iterable of light curves."""
        return bispectrum_from_lc_iterable(
            iter_lc,
            dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            silent=silent,
            save_all=save_all,
            save_diagonal=save_diagonal,
        )


def _pixel_edges(x):
    """Bin edges for ``pcolormesh`` from (possibly non-uniform) bin centers."""
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    mid = 0.5 * (x[1:] + x[:-1])
    first = x[0] - (mid[0] - x[0])
    last = x[-1] + (x[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


class DynamicalCrossBispectrum(AveragedCrossBispectrum):
    type = "crossbispectrum"

    r"""Make a time-resolved (dynamical) cross-bispectrum.

    This is the higher-order analogue of :class:`stingray.DynamicalCrossspectrum`.
    The observation is divided into time bins of length ``bin_size``; within each
    bin an :class:`AveragedCrossBispectrum` is computed over segments of length
    ``segment_size``, and the results are stacked as a function of both time and
    frequency. It traces how quadratic phase coupling between channels evolves.

    Unlike the dynamical *power* spectrum, the time unit is a **block of several
    segments**, not a single segment: a bicoherence built from one segment is
    identically 1 and carries no information, so it only becomes meaningful once
    several segments are averaged. Hence the two time scales ``segment_size``
    (the FFT length, which sets ``df`` and the Nyquist frequency) and ``bin_size``
    (the length of each dynamical row); the time resolution is ``bin_size``.

    Because each time bin holds a full 2-D ``(f1, f2)`` map, the stored object is
    a 3-D cube. That is expensive, so by default only the **diagonal**
    ``f1 = f2 = nu`` is kept (``store="diagonal"``), giving a ``nu x time`` image
    directly analogous to ``dyn_ps``; pass ``store="full"`` to keep the whole
    cube (needed for :meth:`plot_slice`, :meth:`plot_frame`, :meth:`plot_montage`).

    Parameters
    ----------
    data1, data2, data3 : :class:`stingray.Lightcurve` or :class:`stingray.events.EventList`
        The three channels, mapped to ``X(f1)``, ``Y(f2)``, ``Z(f1+f2)``.
        ``data2``/``data3`` default to ``data1`` (the auto case). For event
        lists, ``sample_time`` must be given.
    segment_size : float
        Length, in seconds, of the FFT segments averaged inside each time bin.
    bin_size : float
        Length, in seconds, of each dynamical time bin. Must be at least
        ``segment_size``; ``bin_size / segment_size`` segments are averaged per
        bin, and a warning is issued if that is small (the bicoherence bias
        floor is ``~1/sqrt(m)``).

    Other Parameters
    ----------------
    bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
        Bicoherence normalization (see :class:`Bispectrum`).
    poisson_subtract : bool, default False
        Subtract the per-bin Poisson-noise bias. For the cross case only has an
        effect when ``channels_overlap`` is True. Note the bias scales as
        ``~1/sqrt(N_bin)`` and a bin holds fewer photons than the whole
        observation, so the correction matters more per row than for a static
        bispectrum.
    channels_overlap : bool, default False
        Whether the three channels share the same photons.
    store : {"diagonal", "full"}, default "diagonal"
        Whether to keep only the diagonal of each time bin (light) or the whole
        ``(f1, f2)`` map (heavy).
    gti : ``[[gti0_0, gti0_1], ...]``
        Good time intervals. Defaults to the intersection of the inputs' GTIs.
    sample_time : float
        Time resolution of the light curves created from event lists. Required
        for :class:`EventList` inputs.

    Attributes
    ----------
    freq : numpy.ndarray
        The frequency axis (signed for the cross case, positive for the auto
        case).
    time : numpy.ndarray
        Mid-point time of each dynamical bin.
    dyn_bicoherence : numpy.ndarray
        The bicoherence per time bin: ``(n_time, nf)`` if ``store="diagonal"``,
        else ``(n_time, nf, nf)``.
    dyn_bispec : numpy.ndarray
        The complex cross-bispectrum per time bin, same shape as
        ``dyn_bicoherence``.
    dyn_biphase : numpy.ndarray
        The biphase per time bin, same shape.
    valid : numpy.ndarray
        ``(nf, nf)`` boolean mask of the resolved region.
    valid_diag : numpy.ndarray
        ``(nf,)`` boolean mask of the resolved diagonal frequencies.
    df, dt : float
        Frequency resolution, and time resolution (``= bin_size``).
    m : int
        Number of segments averaged per time bin.
    nseg : numpy.ndarray
        Number of segments actually averaged in each bin.
    """

    def __init__(
        self,
        data1=None,
        data2=None,
        data3=None,
        segment_size=None,
        bin_size=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        channels_overlap=False,
        store="diagonal",
        gti=None,
        sample_time=None,
        skip_checks=False,
    ):
        _warn_interface_change()
        self._type = None
        self.segment_size = segment_size
        self.bin_size = bin_size
        self.sample_time = sample_time
        self.gti = gti
        self.bicoherence_norm = bicoherence_norm
        self.poisson_subtract = poisson_subtract
        self.bias_subtract = bias_subtract
        self.channels_overlap = channels_overlap
        self.poisson_subtracted = poisson_subtract and channels_overlap
        if store not in ("diagonal", "full"):
            raise ValueError("store must be 'diagonal' or 'full'")
        self.store = store

        if data1 is not None:
            if data2 is None:
                data2 = data1
            if data3 is None:
                data3 = data1

        if segment_size is None and bin_size is None and data1 is None:
            return self._initialize_empty()

        if not skip_checks:
            self._dyn_checks(data1, data2, data3, sample_time, segment_size, bin_size)

        self._make_matrix(data1, data2, data3)

    def _initialize_empty(self):
        self.freq = None
        self.time = None
        self.valid = None
        self.valid_diag = None
        self.dyn_bispec = None
        self.dyn_bicoherence = None
        self.dyn_biphase = None
        self.nseg = None
        self.nphots = None
        self.nphots1 = None
        self.nphots2 = None
        self.nphots3 = None
        self.df = None
        self.dt = None
        self.m = None
        self._sum_bispec = None
        self._sum_denom1 = None
        self._sum_denom2 = None
        self._sum_absT = None
        return

    def _dyn_checks(self, data1, data2, data3, sample_time, segment_size, bin_size):
        if data1 is None:
            raise TypeError("data1 must be specified")
        if segment_size is None or bin_size is None:
            raise TypeError("segment_size and bin_size must both be specified")
        for data in (data1, data2, data3):
            if isinstance(data, EventList) and sample_time is None:
                raise ValueError("To pass event lists, please specify sample_time")
        if not (isinstance(data1, type(data2)) and isinstance(data1, type(data3))):
            raise ValueError("The input channels must all be of the same kind.")
        st = data1.dt if isinstance(data1, Lightcurve) else sample_time
        if segment_size < 2 * st:
            raise ValueError("segment_size must be at least 2 * sample_time.")
        if bin_size < segment_size:
            raise ValueError("bin_size must be at least as long as segment_size.")
        n_per_bin = int(round(bin_size / segment_size))
        if n_per_bin < 10:
            warnings.warn(
                f"Only ~{n_per_bin} segment(s) per time bin: the bicoherence bias "
                "floor (~1/sqrt(m)) will be large. Consider a larger bin_size."
            )
        return True

    def _build_bin(self, data1, data2, data3, bin_gti):
        """Build the averaged cross-bispectrum for one time bin."""
        return AveragedCrossBispectrum(
            data1,
            data2,
            data3,
            segment_size=self.segment_size,
            gti=bin_gti,
            dt=self.sample_time,
            bicoherence_norm=self.bicoherence_norm,
            poisson_subtract=self.poisson_subtract,
            bias_subtract=self.bias_subtract,
            channels_overlap=self.channels_overlap,
            silent=True,
        )

    def _resolve_gti(self, data1, data2, data3):
        if self.gti is not None:
            return np.asarray(self.gti)
        gti = data1.gti
        for data in (data2, data3):
            gti = cross_two_gtis(gti, data.gti)
        return np.asarray(gti)

    @staticmethod
    def _bin_nphots(avg):
        n = getattr(avg, "nphots", None)
        n1 = getattr(avg, "nphots1", None)
        n2 = getattr(avg, "nphots2", None)
        n3 = getattr(avg, "nphots3", None)
        n1 = n if n1 is None else n1
        n2 = n if n2 is None else n2
        n3 = n if n3 is None else n3
        return n, n1, n2, n3

    def _make_matrix(self, data1, data2, data3):
        """Fill the dynamical cube, iterating over time bins."""
        gti = self._resolve_gti(data1, data2, data3)
        tstart, tstop = time_intervals_from_gtis(gti, self.bin_size)

        diag = self.store == "diagonal"
        sum_bispec, sum_d1, sum_d2, sum_absT = [], [], [], []
        times, nseg, nph, nph1, nph2, nph3 = [], [], [], [], [], []
        freq = valid = None
        n_skipped = 0

        for ts, te in zip(tstart, tstop):
            bin_gti = cross_two_gtis(gti, np.array([[ts, te]]))
            try:
                avg = self._build_bin(data1, data2, data3, bin_gti)
            except (ValueError, AssertionError):
                # a bin with no usable segments (too short, or all zero) is skipped
                n_skipped += 1
                continue
            if getattr(avg, "freq", None) is None:  # pragma: no cover
                n_skipped += 1
                continue

            if freq is None:
                freq = avg.freq
                valid = avg.valid

            triple_sum = avg.bispec * avg.m  # complex sum of the per-segment triples
            d1, d2, aT = avg._bicoh_denom1, avg._bicoh_denom2, avg._bicoh_sum_abs
            if diag:
                sum_bispec.append(np.diag(triple_sum).copy())
                sum_d1.append(np.diag(d1).copy())
                sum_d2.append(np.diag(d2).copy())
                sum_absT.append(np.diag(aT).copy())
            else:
                sum_bispec.append(triple_sum)
                sum_d1.append(d1)
                sum_d2.append(d2)
                sum_absT.append(aT)

            n, n1, n2, n3 = self._bin_nphots(avg)
            times.append(0.5 * (ts + te))
            nseg.append(avg.m)
            nph.append(n)
            nph1.append(n1)
            nph2.append(n2)
            nph3.append(n3)

        if freq is None:
            raise ValueError("No usable time bins were found for the dynamical bispectrum.")
        if n_skipped:
            warnings.warn(f"{n_skipped} time bin(s) had no usable segments and were dropped.")

        self.freq = np.asarray(freq)
        self.valid = valid
        self.valid_diag = np.diag(valid)
        self.time = np.asarray(times)
        self.nseg = np.asarray(nseg)
        self.nphots = np.asarray(nph, dtype=float)
        self.nphots1 = np.asarray(nph1, dtype=float)
        self.nphots2 = np.asarray(nph2, dtype=float)
        self.nphots3 = np.asarray(nph3, dtype=float)
        self._sum_bispec = np.asarray(sum_bispec)
        self._sum_denom1 = np.asarray(sum_d1)
        self._sum_denom2 = np.asarray(sum_d2)
        self._sum_absT = np.asarray(sum_absT)
        self.df = float(self.freq[1] - self.freq[0]) if self.freq.size > 1 else None
        self.dt = self.bin_size
        self.m = int(self.nseg[0]) if self.nseg.size else None

        self._recompute()

    def _recompute(self):
        """Derive ``dyn_bispec``/``dyn_bicoherence``/``dyn_biphase`` from the sums."""
        if self.store == "diagonal":
            valid = np.broadcast_to(self.valid_diag[None, :], self._sum_bispec.shape)
            m = self.nseg[:, None]
        else:
            valid = np.broadcast_to(self.valid[None, :, :], self._sum_bispec.shape)
            m = self.nseg[:, None, None]

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_bispec = self._sum_bispec / m
        self.dyn_bispec = np.where(valid, mean_bispec, np.nan)
        self.dyn_biphase = np.where(valid, np.angle(self._sum_bispec), np.nan)
        self.dyn_bicoherence = bicoherence_from_sums(
            self.bicoherence_norm,
            np.abs(self._sum_bispec),
            self._sum_denom1,
            self._sum_denom2,
            self._sum_absT,
            valid=valid,
        )

    # -- helpers -----------------------------------------------------------

    def _diagonal(self, arr):
        """Return the ``(n_time, nf)`` diagonal of a per-bin quantity."""
        if self.store == "diagonal":
            return arr
        return np.diagonal(arr, axis1=1, axis2=2)

    def _freq_index(self, f):
        return int(np.argmin(np.abs(self.freq - f)))

    # -- plotting ----------------------------------------------------------

    @staticmethod
    def _mesh_norm(matrix, vmin, vmax, log):
        """Masked data and ``pcolormesh`` colour-scale kwargs, linear or log.

        For ``log=True`` a :class:`matplotlib.colors.LogNorm` is used, which keeps
        the true bicoherence values on a log-spaced colour scale; zero and
        unresolved (``NaN``) bins are masked out (``LogNorm`` cannot show them). A
        non-positive ``vmin`` is treated as "autoscale" for the log case.
        """
        if log:
            data = np.ma.masked_less_equal(np.ma.masked_invalid(matrix), 0.0)
            lo = vmin if (vmin is not None and vmin > 0) else None
            return data, {"norm": LogNorm(vmin=lo, vmax=vmax)}
        return np.ma.masked_invalid(matrix), {"vmin": vmin, "vmax": vmax}

    def plot_diagonal(self, ax=None, cmap="viridis", vmin=0.0, vmax=1.0, colorbar=True, log=False):
        r"""Plot the diagonal dynamical bicoherence ``b(nu, nu, t)``.

        This is the closest analogue of the dynamical power spectrum: a
        ``nu`` (with ``f1 = f2 = nu``) versus time image, where ``nu`` couples to
        its harmonic ``2 nu``. Works from either store. Set ``log=True`` for a
        logarithmic (``LogNorm``) colour scale.
        """
        if self.dyn_bicoherence is None:
            raise ValueError("This dynamical bispectrum has no data to plot.")
        diag = self._diagonal(self.dyn_bicoherence)
        if ax is None:
            _, ax = plt.subplots()
        data, norm_kw = self._mesh_norm(diag.T, vmin, vmax, log)
        pc = ax.pcolormesh(
            _pixel_edges(self.time), _pixel_edges(self.freq), data, cmap=cmap, **norm_kw
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\nu$ (Hz), $f_1 = f_2 = \nu$")
        ax.set_title("Diagonal dynamical bicoherence")
        if colorbar:
            ax.figure.colorbar(pc, ax=ax, label="bicoherence")
        return ax

    def plot_slice(self, f1, ax=None, cmap="viridis", vmin=0.0, vmax=1.0, colorbar=True, log=False):
        """Plot the bicoherence at a fixed ``f1`` as a function of ``(f2, time)``.

        Requires ``store="full"``. Set ``log=True`` for a logarithmic
        (``LogNorm``) colour scale.
        """
        if self.store != "full":
            raise ValueError("plot_slice needs store='full'.")
        i1 = self._freq_index(f1)
        sl = self.dyn_bicoherence[:, i1, :]
        if ax is None:
            _, ax = plt.subplots()
        data, norm_kw = self._mesh_norm(sl.T, vmin, vmax, log)
        pc = ax.pcolormesh(
            _pixel_edges(self.time), _pixel_edges(self.freq), data, cmap=cmap, **norm_kw
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$f_2$ (Hz)")
        ax.set_title(f"Bicoherence at $f_1 = {self.freq[i1]:g}$ Hz")
        if colorbar:
            ax.figure.colorbar(pc, ax=ax, label="bicoherence")
        return ax

    def plot_frame(self, t, ax=None, cmap="viridis", vmin=0.0, vmax=1.0, colorbar=True, log=False):
        """Plot the full ``(f1, f2)`` bicoherence map at the time bin nearest ``t``.

        Requires ``store="full"``. Set ``log=True`` for a logarithmic
        (``LogNorm``) colour scale.
        """
        if self.store != "full":
            raise ValueError("plot_frame needs store='full'.")
        k = int(np.argmin(np.abs(self.time - t)))
        if ax is None:
            _, ax = plt.subplots()
        data, norm_kw = self._mesh_norm(self.dyn_bicoherence[k].T, vmin, vmax, log)
        pc = ax.pcolormesh(
            _pixel_edges(self.freq), _pixel_edges(self.freq), data, cmap=cmap, **norm_kw
        )
        ax.set_xlabel("$f_1$ (Hz)")
        ax.set_ylabel("$f_2$ (Hz)")
        ax.set_title(f"Bicoherence at t = {self.time[k]:g} s")
        if colorbar:
            ax.figure.colorbar(pc, ax=ax, label="bicoherence")
        return ax

    def plot_montage(self, times=None, ncols=5, cmap="viridis", vmin=0.0, vmax=1.0, log=False):
        """Plot a grid of full ``(f1, f2)`` maps over time. Requires ``store="full"``.

        Set ``log=True`` for a logarithmic (``LogNorm``) colour scale.
        """
        if self.store != "full":
            raise ValueError("plot_montage needs store='full'.")
        if times is None:
            idx = np.arange(self.time.size)
        else:
            idx = [int(np.argmin(np.abs(self.time - t))) for t in np.atleast_1d(times)]
        n = len(idx)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True, squeeze=False
        )
        for ax in axes.flat:
            ax.set_visible(False)
        pc = None
        for k, ax in zip(idx, axes.flat):
            ax.set_visible(True)
            data, norm_kw = self._mesh_norm(self.dyn_bicoherence[k].T, vmin, vmax, log)
            pc = ax.pcolormesh(
                _pixel_edges(self.freq), _pixel_edges(self.freq), data, cmap=cmap, **norm_kw
            )
            ax.set_title(f"t = {self.time[k]:g} s", fontsize=9)
        # figure-level axis labels (fig.supxlabel/supylabel need matplotlib >= 3.4,
        # below the project's floor, so use fig.text instead).
        fig.text(0.5, 0.04, "$f_1$ (Hz)", ha="center")
        fig.text(0.04, 0.5, "$f_2$ (Hz)", va="center", rotation="vertical")
        if pc is not None:
            fig.colorbar(pc, ax=axes, fraction=0.02, pad=0.01, label="bicoherence")
        return axes

    def trace(self, f1, f2):
        r"""Return ``(time, bicoherence(t), biphase(t))`` at a fixed ``(f1, f2)``.

        For ``store="diagonal"`` only diagonal points (``f1 = f2``) are available.
        """
        i1, i2 = self._freq_index(f1), self._freq_index(f2)
        if self.store == "diagonal":
            if i1 != i2:
                raise ValueError(
                    "store='diagonal' only keeps f1 = f2; pass equal frequencies "
                    "or rebuild with store='full'."
                )
            bic = self.dyn_bicoherence[:, i1]
            bip = self.dyn_biphase[:, i1]
        else:
            bic = self.dyn_bicoherence[:, i1, i2]
            bip = self.dyn_biphase[:, i1, i2]
        return self.time, bic, bip

    def plot_trace(self, f1, f2, axes=None):
        """Plot the bicoherence and biphase at a fixed ``(f1, f2)`` versus time."""
        time, bic, bip = self.trace(f1, f2)
        if axes is None:
            _, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
        ax0, ax1 = axes
        ax0.plot(time, bic, "o-", color="tab:blue")
        ax0.axhline(1.0 / np.sqrt(self.m), color="0.5", ls=":", label=r"$1/\sqrt{m}$ floor")
        ax0.set_ylabel("bicoherence")
        ax0.set_ylim(0, 1.05)
        ax0.legend(fontsize="small")
        ax0.set_title(
            f"Coupling at $(f_1, f_2) = ({self.freq[self._freq_index(f1)]:g}, "
            f"{self.freq[self._freq_index(f2)]:g})$ Hz"
        )
        ax1.plot(time, np.degrees(bip), "o-", color="tab:orange")
        ax1.set_ylabel("biphase (deg)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylim(-190, 190)
        return axes

    # -- rebinning ---------------------------------------------------------

    def _rebinned_by_n_time(self, n):
        """Return a copy with consecutive time bins summed in groups of ``n``."""
        import copy as _copy

        if n <= 1:
            return _copy.deepcopy(self)
        new = _copy.deepcopy(self)
        nt = self.time.size
        edges = range(0, nt - n + 1, n)

        def group(arr, reducer):
            return np.array([reducer(arr[i : i + n], axis=0) for i in edges])

        new._sum_bispec = group(self._sum_bispec, np.sum)
        new._sum_denom1 = group(self._sum_denom1, np.sum)
        new._sum_denom2 = group(self._sum_denom2, np.sum)
        new._sum_absT = group(self._sum_absT, np.sum)
        new.time = group(self.time, np.mean)
        new.nseg = group(self.nseg, np.sum)
        for attr in ("nphots", "nphots1", "nphots2", "nphots3"):
            new_attr = group(getattr(self, attr), np.mean)
            setattr(new, attr, new_attr)
        new.dt = self.dt * n
        new.m = int(new.nseg[0]) if new.nseg.size else None
        new._recompute()
        return new

    def rebin_by_n_intervals(self, n, method="sum"):
        """Rebin in time by combining ``n`` consecutive intervals.

        The additive bispectrum sums are combined (not the finished
        bicoherence, which is a ratio and cannot be averaged), then the
        bicoherence and biphase are recomputed. ``method`` is accepted for API
        parity with :class:`DynamicalCrossspectrum` but the sums are always
        summed.
        """
        if not np.issubdtype(type(n), np.integer):
            warnings.warn("n must be an integer. Casting to int")
            n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        return self._rebinned_by_n_time(n)

    def rebin_time(self, dt_new, method="sum"):
        """Rebin to a coarser time resolution ``dt_new`` (an integer multiple of ``dt``).

        Combines the additive sums over consecutive bins and recomputes the
        bicoherence, so the result is the correct averaged bicoherence over the
        wider bin -- not an average of bicoherence values.
        """
        if dt_new < self.dt:
            raise ValueError("New time resolution must be larger than the current one.")
        n = int(round(dt_new / self.dt))
        return self._rebinned_by_n_time(max(n, 1))

    def rebin_frequency(self, df_new, method="sum"):
        """Rebin the (diagonal) frequency axis to a coarser ``df_new``.

        Only implemented for ``store="diagonal"``. As for :meth:`rebin_time`, the
        additive sums are combined and the bicoherence recomputed.
        """
        if self.store != "diagonal":
            raise NotImplementedError(
                "rebin_frequency is currently only implemented for store='diagonal'."
            )
        if df_new < self.df:
            raise ValueError("New frequency resolution must be larger than the current one.")
        import copy as _copy

        n = int(round(df_new / self.df))
        if n <= 1:
            return _copy.deepcopy(self)
        nf = self.freq.size
        edges = range(0, nf - n + 1, n)

        def group(arr, reducer, axis):
            return np.stack([reducer(arr[:, i : i + n], axis=axis) for i in edges], axis=1)

        new = _copy.deepcopy(self)
        new._sum_bispec = group(self._sum_bispec, np.sum, 1)
        new._sum_denom1 = group(self._sum_denom1, np.sum, 1)
        new._sum_denom2 = group(self._sum_denom2, np.sum, 1)
        new._sum_absT = group(self._sum_absT, np.sum, 1)
        new.freq = np.array([np.mean(self.freq[i : i + n]) for i in edges])
        # A rebinned diagonal frequency is resolved if all its members were.
        new.valid_diag = np.array([np.all(self.valid_diag[i : i + n]) for i in edges])
        new.df = self.df * n
        new._recompute()
        return new

    # -- tracking ----------------------------------------------------------

    def trace_maximum(self, min_freq=None, max_freq=None):
        """Trace the peak-bicoherence diagonal frequency index in each time bin.

        Returns the array of frequency indices (into ``freq``) of the maximum
        diagonal bicoherence between ``min_freq`` and ``max_freq`` per bin -- the
        bicoherence analogue of :meth:`DynamicalCrossspectrum.trace_maximum`.
        """
        if min_freq is None:
            min_freq = np.min(self.freq)
        if max_freq is None:
            max_freq = np.max(self.freq)
        band = (self.freq >= min_freq) & (self.freq <= max_freq) & self.valid_diag
        diag = self._diagonal(self.dyn_bicoherence)
        max_positions = []
        for row in diag:
            masked = np.where(band, row, -np.inf)
            max_positions.append(int(np.nanargmax(masked)))
        return np.array(max_positions)

    def shift_and_add(self, f0_list, nbins=None):
        r"""Shift-and-add the diagonal bicoherence, aligning each bin to its ``f0``.

        For each time bin ``i`` the diagonal sums are shifted so that
        ``f0_list[i]`` lands on a common reference bin, then co-added; the
        bicoherence is recomputed from the co-added sums. This tracks a drifting
        coupling (e.g. a QPO fundamental) the way the kHz-QPO shift-and-add
        tracks a drifting Lorentzian.

        Parameters
        ----------
        f0_list : iterable of float
            The reference frequency in each time bin (length ``n_time``).

        Other Parameters
        ----------------
        nbins : int, optional
            Length of the output (relative-frequency) axis. Defaults to the
            number of frequency bins.

        Returns
        -------
        rel_freq : numpy.ndarray
            Frequency relative to ``f0`` (0 at the reference).
        bicoherence : numpy.ndarray
            The shifted-and-added bicoherence.
        biphase : numpy.ndarray
            The corresponding biphase.
        """
        f0_list = np.atleast_1d(f0_list)
        if f0_list.size != self.time.size:
            raise ValueError("f0_list must have one entry per time bin.")
        # Reference bin index of each f0 on the actual frequency axis (which does
        # not start at 0, and is signed for the cross case), not round(f0/df).
        k0 = np.array([self._freq_index(f) for f in f0_list])

        st = self._diagonal(self._sum_bispec)
        d1 = self._diagonal(self._sum_denom1)
        d2 = self._diagonal(self._sum_denom2)
        aT = self._diagonal(self._sum_absT)

        L = int(nbins) if nbins is not None else self.freq.size
        center = L // 2
        out_st = np.zeros(L, dtype=complex)
        out_d1 = np.zeros(L)
        out_d2 = np.zeros(L)
        out_aT = np.zeros(L)

        src = np.arange(self.freq.size)
        for i in range(self.time.size):
            dst = src + (center - k0[i])
            keep = (dst >= 0) & (dst < L) & self.valid_diag
            out_st[dst[keep]] += st[i, keep]
            out_d1[dst[keep]] += d1[i, keep]
            out_d2[dst[keep]] += d2[i, keep]
            out_aT[dst[keep]] += aT[i, keep]

        bic = bicoherence_from_sums(self.bicoherence_norm, np.abs(out_st), out_d1, out_d2, out_aT)
        rel_freq = (np.arange(L) - center) * self.df
        return rel_freq, bic, np.angle(out_st)


class DynamicalBispectrum(DynamicalCrossBispectrum):
    type = "bispectrum"

    r"""Make a time-resolved (dynamical) auto-bispectrum from a single input.

    The auto special case of :class:`DynamicalCrossBispectrum` (the three
    channels are the same light curve), analogous to
    :class:`stingray.DynamicalPowerspectrum`. See
    :class:`DynamicalCrossBispectrum` for the full description of the two time
    scales (``segment_size``/``bin_size``), the ``store`` option, and the
    plotting/rebinning/tracking methods.

    Parameters
    ----------
    data : :class:`stingray.Lightcurve` or :class:`stingray.events.EventList`
        The light curve or event list. For an event list, ``sample_time`` must
        be given.
    segment_size : float
        Length, in seconds, of the FFT segments averaged inside each time bin.
    bin_size : float
        Length, in seconds, of each dynamical time bin.

    Other Parameters
    ----------------
    See :class:`DynamicalCrossBispectrum` (``channels_overlap`` is always True
    here).
    """

    def __init__(
        self,
        data=None,
        segment_size=None,
        bin_size=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        bias_subtract=False,
        store="diagonal",
        gti=None,
        sample_time=None,
        skip_checks=False,
    ):
        super().__init__(
            data1=data,
            segment_size=segment_size,
            bin_size=bin_size,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            bias_subtract=bias_subtract,
            channels_overlap=True,
            store=store,
            gti=gti,
            sample_time=sample_time,
            skip_checks=skip_checks,
        )

    def _build_bin(self, data1, data2, data3, bin_gti):
        """Build the averaged auto-bispectrum for one time bin."""
        return AveragedBispectrum(
            data1,
            segment_size=self.segment_size,
            gti=bin_gti,
            dt=self.sample_time,
            bicoherence_norm=self.bicoherence_norm,
            poisson_subtract=self.poisson_subtract,
            bias_subtract=self.bias_subtract,
            silent=True,
        )


def _populate_bispectrum_from_result_table(bs, table):
    """Copy the columns and metadata from a fourier result table onto ``bs``.

    Shared by the auto- and cross-bispectrum allocators.
    """
    bs.freq = np.asarray(table.meta["freq"])
    bs.bispec = table.meta["bispec"]
    bs.bicoherence = table.meta["bicoherence"]
    bs.bicoherence_norm = table.meta["bicoherence_norm"]
    bs.poisson_subtracted = table.meta.get("poisson_subtract", False)
    bs.bias_subtract = table.meta.get("bias_subtract", False)
    bs.biphase = table.meta["biphase"]
    bs.bispec_err = table.meta["bispec_err"]
    bs.biphase_err = table.meta["biphase_err"]
    bs.valid = table.meta["valid"]
    bs.bispec_mag = np.abs(bs.bispec)
    bs.bispec_phase = bs.biphase

    # Raw accumulated sums, kept so the bicoherence can be recomputed under a
    # different normalization via ``recompute_bicoherence`` without redoing FFTs.
    bs._bicoh_abs_bispec_sum = table.meta["bicoh_abs_bispec_sum"]
    bs._bicoh_denom1 = table.meta["bicoh_denom1"]
    bs._bicoh_denom2 = table.meta["bicoh_denom2"]
    bs._bicoh_sum_abs = table.meta["bicoh_sum_abs"]

    for attr in [
        "n",
        "m",
        "dt",
        "df",
        "nphots",
        "nphots1",
        "nphots2",
        "nphots3",
        "channels_overlap",
        "segment_size",
        "gti",
    ]:
        if attr in table.meta:
            setattr(bs, attr, table.meta[attr])

    if "subbs" in table.meta:
        bs.bispec_all = table.meta["subbs"]
    if "subbs_diagonal" in table.meta:
        bs.bispec_diagonal = table.meta["subbs_diagonal"]
    return bs


def _create_bispectrum_from_result_table(table, force_averaged=False):
    """Allocate a :class:`Bispectrum` / :class:`AveragedBispectrum` from a
    result table produced by ``stingray.fourier.avg_bispectrum_from_XX``."""
    if table is None:  # pragma: no cover
        raise ValueError("No usable segments were found to compute the bispectrum.")
    cls = AveragedBispectrum if (table.meta["m"] > 1 or force_averaged) else Bispectrum
    return _populate_bispectrum_from_result_table(cls(), table)


def _create_crossbispectrum_from_result_table(table, force_averaged=False):
    """Allocate a :class:`CrossBispectrum` / :class:`AveragedCrossBispectrum`
    from a result table produced by ``avg_cross_bispectrum_from_XX``."""
    if table is None:  # pragma: no cover
        raise ValueError("No usable segments were found to compute the cross-bispectrum.")
    cls = AveragedCrossBispectrum if (table.meta["m"] > 1 or force_averaged) else CrossBispectrum
    return _populate_bispectrum_from_result_table(cls(), table)


# ---------------------------------------------------------------------------
# Auto-bispectrum module functions
# ---------------------------------------------------------------------------


def bispectrum_from_time_array(
    times,
    dt,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate an auto-bispectrum from an array of event times.

    Parameters
    ----------
    times : `np.array`
        Event arrival times.
    dt : float
        The time resolution of the intermediate light curves.

    Other Parameters
    ----------------
    segment_size : float
        The length, in seconds, of the light curve segments to average. Only
        relevant (and required) for an :class:`AveragedBispectrum`.
    gti : ``[[gti0_0, gti0_1], ...]``
        Good time intervals.
    bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
        The bicoherence normalization (see :class:`Bispectrum`).
    poisson_subtract : bool, default False
        Subtract the Poisson-noise bias.
    silent : bool, default False
        Silence the progress bars.
    save_all, save_diagonal : bool, default False
        Store per-segment bispectra (full cube / diagonal only).

    Returns
    -------
    spec : :class:`AveragedBispectrum` or :class:`Bispectrum`
        The output bispectrum.
    """
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    table = avg_bispectrum_from_timeseries(
        times,
        gti,
        segment_size,
        dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        silent=silent,
        return_subbs=save_all,
        save_diagonal=save_diagonal,
    )
    return _create_bispectrum_from_result_table(table, force_averaged=force_averaged)


def bispectrum_from_events(
    events,
    dt,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate an auto-bispectrum from an event list. See
    `bispectrum_from_time_array` for the parameters."""
    if gti is None:
        gti = events.gti
    dt = events.suggest_compatible_dt(dt)
    return bispectrum_from_time_array(
        events.time,
        dt,
        segment_size=segment_size,
        gti=gti,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        silent=silent,
        save_all=save_all,
        save_diagonal=save_diagonal,
    )


def bispectrum_from_lightcurve(
    lc,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate an auto-bispectrum from a light curve. See
    `bispectrum_from_time_array` for the parameters."""
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    if gti is None:
        gti = lc.gti
    err = None
    if lc.err_dist == "gauss":
        err = lc.counts_err
    table = avg_bispectrum_from_timeseries(
        lc.time,
        gti,
        segment_size,
        lc.dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        silent=silent,
        fluxes=lc.counts,
        errors=err,
        return_subbs=save_all,
        save_diagonal=save_diagonal,
    )
    return _create_bispectrum_from_result_table(table, force_averaged=force_averaged)


def bispectrum_from_stingray_timeseries(
    ts,
    flux_attr,
    error_flux_attr=None,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate an auto-bispectrum from a time series. See
    `bispectrum_from_time_array` for the parameters."""
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    if gti is None:
        gti = ts.gti
    err = None
    if error_flux_attr is not None:
        err = getattr(ts, error_flux_attr)
    table = avg_bispectrum_from_timeseries(
        ts.time,
        gti,
        segment_size,
        ts.dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        silent=silent,
        fluxes=getattr(ts, flux_attr),
        errors=err,
        return_subbs=save_all,
        save_diagonal=save_diagonal,
    )
    return _create_bispectrum_from_result_table(table, force_averaged=force_averaged)


def bispectrum_from_lc_iterable(
    iter_lc,
    dt,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate an average auto-bispectrum from an iterable of light curves.

    See `bispectrum_from_time_array` for the parameters.
    """
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    common_gti = gti

    def iterate_lc_counts(iter_lc):
        for lc in iter_lc:
            if hasattr(lc, "counts"):
                n_bin = (
                    np.rint(segment_size / lc.dt).astype(int) if segment_size else lc.counts.size
                )
                lc_gti = lc.gti
                if common_gti is not None:
                    lc_gti = cross_two_gtis(common_gti, lc.gti)
                err = None
                if lc.err_dist == "gauss":
                    err = lc.counts_err
                flux_iterable = get_flux_iterable_from_segments(
                    lc.time, lc_gti, segment_size, n_bin, fluxes=lc.counts, errors=err
                )
                for out in flux_iterable:
                    yield out
            elif isinstance(lc, Iterable):
                yield lc
            else:
                raise TypeError(
                    "The inputs to bispectrum_from_lc_iterable must be "
                    "Lightcurve objects or arrays."
                )

    table = avg_bispectrum_from_iterable(
        iterate_lc_counts(iter_lc),
        dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        silent=silent,
        return_subbs=save_all,
        save_diagonal=save_diagonal,
    )
    return _create_bispectrum_from_result_table(table, force_averaged=force_averaged)


# ---------------------------------------------------------------------------
# Cross-bispectrum module functions
# ---------------------------------------------------------------------------


def crossbispectrum_from_time_array(
    times1,
    times2,
    times3,
    dt,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    channels_overlap=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate a cross-bispectrum from three arrays of event times.

    The three channels must be simultaneous. See `crossbispectrum_from_lightcurve`
    and :class:`CrossBispectrum` for the parameters.
    """
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    table = avg_cross_bispectrum_from_timeseries(
        times1,
        times2,
        times3,
        gti,
        segment_size,
        dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        channels_overlap=channels_overlap,
        silent=silent,
        return_subbs=save_all,
        save_diagonal=save_diagonal,
    )
    return _create_crossbispectrum_from_result_table(table, force_averaged=force_averaged)


def crossbispectrum_from_events(
    events1,
    events2,
    events3,
    dt,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    channels_overlap=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate a cross-bispectrum from three event lists. See
    `crossbispectrum_from_lightcurve` for the parameters."""
    if gti is None:
        gti = cross_two_gtis(cross_two_gtis(events1.gti, events2.gti), events3.gti)
    dt = events1.suggest_compatible_dt(dt)
    return crossbispectrum_from_time_array(
        events1.time,
        events2.time,
        events3.time,
        dt,
        segment_size=segment_size,
        gti=gti,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        channels_overlap=channels_overlap,
        silent=silent,
        save_all=save_all,
        save_diagonal=save_diagonal,
    )


def crossbispectrum_from_lightcurve(
    lc1,
    lc2,
    lc3,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    channels_overlap=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate a cross-bispectrum from three simultaneous light curves.

    Parameters
    ----------
    lc1, lc2, lc3 : :class:`stingray.Lightcurve`
        The three channels, mapped to ``X(f1)``, ``Y(f2)``, ``Z(f1+f2)``. They
        must share the same time bins.

    Other Parameters
    ----------------
    See :class:`CrossBispectrum` and `bispectrum_from_time_array`.

    Returns
    -------
    spec : :class:`AveragedCrossBispectrum` or :class:`CrossBispectrum`
        The output cross-bispectrum.
    """
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    if not (
        lc1.time.size == lc2.time.size == lc3.time.size
        and np.allclose(lc1.time, lc2.time)
        and np.allclose(lc1.time, lc3.time)
    ):
        raise ValueError("The three light curves must share the same time bins.")
    if gti is None:
        gti = cross_two_gtis(cross_two_gtis(lc1.gti, lc2.gti), lc3.gti)
    table = avg_cross_bispectrum_from_timeseries(
        lc1.time,
        lc2.time,
        lc3.time,
        gti,
        segment_size,
        lc1.dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        channels_overlap=channels_overlap,
        silent=silent,
        fluxes1=lc1.counts,
        fluxes2=lc2.counts,
        fluxes3=lc3.counts,
        return_subbs=save_all,
        save_diagonal=save_diagonal,
    )
    return _create_crossbispectrum_from_result_table(table, force_averaged=force_averaged)


def crossbispectrum_from_stingray_timeseries(
    ts1,
    ts2,
    ts3,
    flux_attr,
    error_flux_attr=None,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    channels_overlap=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate a cross-bispectrum from three time series. See
    `crossbispectrum_from_lightcurve` for the parameters."""
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    if gti is None:
        gti = cross_two_gtis(cross_two_gtis(ts1.gti, ts2.gti), ts3.gti)
    table = avg_cross_bispectrum_from_timeseries(
        ts1.time,
        ts2.time,
        ts3.time,
        gti,
        segment_size,
        ts1.dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        channels_overlap=channels_overlap,
        silent=silent,
        fluxes1=getattr(ts1, flux_attr),
        fluxes2=getattr(ts2, flux_attr),
        fluxes3=getattr(ts3, flux_attr),
        return_subbs=save_all,
        save_diagonal=save_diagonal,
    )
    return _create_crossbispectrum_from_result_table(table, force_averaged=force_averaged)


def crossbispectrum_from_lc_iterable(
    iter_lc1,
    iter_lc2,
    iter_lc3,
    dt,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    bias_subtract=False,
    channels_overlap=False,
    silent=False,
    save_all=False,
    save_diagonal=False,
):
    """Calculate an average cross-bispectrum from three iterables of light curves.

    See `crossbispectrum_from_lightcurve` for the parameters.
    """
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    common_gti = gti

    def iterate_lc_counts(iter_lc):
        for lc in iter_lc:
            if hasattr(lc, "counts"):
                n_bin = (
                    np.rint(segment_size / lc.dt).astype(int) if segment_size else lc.counts.size
                )
                lc_gti = lc.gti
                if common_gti is not None:
                    lc_gti = cross_two_gtis(common_gti, lc.gti)
                flux_iterable = get_flux_iterable_from_segments(
                    lc.time, lc_gti, segment_size, n_bin, fluxes=lc.counts
                )
                for out in flux_iterable:
                    yield out
            elif isinstance(lc, Iterable):
                yield lc
            else:
                raise TypeError(
                    "The inputs to crossbispectrum_from_lc_iterable must be "
                    "Lightcurve objects or arrays."
                )

    table = avg_cross_bispectrum_from_iterables(
        iterate_lc_counts(iter_lc1),
        iterate_lc_counts(iter_lc2),
        iterate_lc_counts(iter_lc3),
        dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        bias_subtract=bias_subtract,
        channels_overlap=channels_overlap,
        silent=silent,
        return_subbs=save_all,
        save_diagonal=save_diagonal,
    )
    return _create_crossbispectrum_from_result_table(table, force_averaged=force_averaged)

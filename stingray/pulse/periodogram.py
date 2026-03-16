import numpy as np
import matplotlib.pyplot as plt
from stingray.pulse.search import epoch_folding_search, z_n_search

class PulsarPeriodogram:
    """
    A class to generate and store Epoch Folding and Z^2_n periodograms.

    Parameters
    ----------
    events : stingray.EventList
        The event list to search for pulsations.
    frequencies : array-like
        The frequencies to search.
    nbin : int, optional, default 32
        The number of bins for the pulse profile (used in Epoch Folding).
    nharm : int, optional, default 1
        The number of harmonics for the Z^2_n search.

    Attributes
    ----------
    ef_freq : numpy.ndarray
        The frequencies used for Epoch Folding.
    ef_stat : numpy.ndarray
        The Epoch Folding statistics.
    z_freq : numpy.ndarray
        The frequencies used for Z^2_n search.
    z_stat : numpy.ndarray
        The Z^2_n statistics.
    """
    def __init__(self, events, frequencies, nbin=32, nharm=1):
        self.events = events
        self.frequencies = frequencies
        self.nbin = nbin
        self.nharm = nharm

        # Run Epoch Folding Search
        # epoch_folding_search returns (freqs, stats)
        self.ef_freq, self.ef_stat = epoch_folding_search(
            events.time, frequencies, nbin=nbin
        )

        # Run Z_n Search
        # z_n_search returns (freqs, stats)
        self.z_freq, self.z_stat = z_n_search(
            events.time, frequencies, nbin=nbin, nharm=nharm
        )

    def plot(self, ax=None, show=True):
        """
        Plot the periodograms.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If provided, plot on this axis. Otherwise, create new figures.
        show : bool, optional
            If True, show the plot.
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        else:
            # If user provides axes, we expect a list/array of 2 axes
            ax1, ax2 = ax

        # Plot Epoch Folding
        ax1.plot(self.ef_freq, self.ef_stat, label='Epoch Folding')
        ax1.axhline(self.nbin - 1, ls='--', color='red', label='Noise Level (Approx)')
        ax1.set_ylabel('EF Statistics')
        ax1.set_title('Pulsar Periodogram')
        ax1.legend()

        # Plot Z_n
        ax2.plot(self.z_freq, self.z_stat, color='orange', label=f'Z_{self.nharm} Statistics')
        ax2.set_ylabel(f'Z_{self.nharm} Statistics')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.legend()
        
        if show:
            plt.show()
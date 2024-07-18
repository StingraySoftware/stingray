import pytest
import numpy as np
from stingray.power_colors import (
    power_color,
    hue_from_power_color,
    plot_hues,
    plot_power_colors,
    DEFAULT_COLOR_CONFIGURATION,
)

rng = np.random.RandomState(1259723)


class TestPowerColor(object):
    @classmethod
    def setup_class(cls):
        cls.freq = np.arange(0.0001, 17, 0.00001)
        cls.power = 1 / cls.freq
        cls.pc0, cls.pc0e, cls.pc1, cls.pc1e = power_color(cls.freq, cls.power)
        cls.lpc0, _, cls.lpc1, _ = power_color(cls.freq, cls.power, return_log=True)
        cls.rms, cls.rmse = 0.1, 0.01
        cls.configuration = DEFAULT_COLOR_CONFIGURATION
        cls.configuration["state_definitions"]["HSS"]["hue_limits"] = [300, 370]

    def test_power_color(self):
        # The colors calculated with these frequency edges on a 1/f spectrum should be 1
        assert np.isclose(self.pc0, 1)
        assert np.isclose(self.pc1, 1)

    def test_return_log(self):
        # The colors calculated with these frequency edges on a 1/f spectrum should be 1
        assert np.isclose(self.lpc0, 0, atol=0.001)
        assert np.isclose(self.lpc1, 0, atol=0.001)

    def test_bad_edges(self):
        good = self.freq > 1 / 255  # the smallest frequency is 1/256
        with pytest.raises(ValueError, match="The minimum frequency is larger "):
            power_color(self.freq[good], self.power[good])

        good = self.freq < 15  # the smallest frequency is 1/256
        with pytest.raises(ValueError, match="The maximum frequency is lower "):
            power_color(self.freq[good], self.power[good])

        with pytest.raises(ValueError, match="freq_edges must have 5 elements"):
            power_color(self.freq, self.power, freq_edges=[1])
        with pytest.raises(ValueError, match="freq_edges must have 5 elements"):
            power_color(self.freq, self.power, freq_edges=[1, 2, 3, 4, 5, 6])

    def test_bad_excluded_interval(self):
        for fte in ([1, 1.1, 3.0], [4], [[1, 1.1, 3.0]], 0, [[[1, 3]]]):
            with pytest.raises(ValueError, match="freqs_to_exclude must be of "):
                power_color(self.freq, self.power, freqs_to_exclude=fte)

    def test_excluded_frequencies(self):
        pc0, _, pc1, _ = power_color(self.freq, self.power, freqs_to_exclude=[1, 1.1])
        # The colors calculated with these frequency edges on a 1/f spectrum should be 1
        # The excluded frequency interval is small enough that the approximation should work
        assert np.isclose(pc0, 1, atol=0.001)
        assert np.isclose(pc1, 1, atol=0.001)

    def test_with_power_err(self):
        pc0, pc0_err, pc1, pc1_err = power_color(
            self.freq,
            self.power,
            power_err=self.power / 2,
        )
        pc0e, pc0e_err, pc1e, pc1e_err = power_color(
            self.freq,
            self.power,
            power_err=self.power,
        )
        assert np.isclose(pc0, 1, atol=0.001)
        assert np.isclose(pc1, 1, atol=0.001)
        assert np.isclose(pc0e, 1, atol=0.001)
        assert np.isclose(pc1e, 1, atol=0.001)
        assert np.isclose(pc0e_err / pc0_err, 2, atol=0.001)
        assert np.isclose(pc1e_err / pc1_err, 2, atol=0.001)

    def test_hue(self):
        center = (4.51920, 0.453724)
        log_center = np.log10(np.asanyarray(center))
        for angle in np.radians(np.arange(0, 380, 20)):
            factor = rng.uniform(0.1, 10)
            x = factor * np.cos(3 / 4 * np.pi - angle) + log_center[0]
            y = factor * np.sin(3 / 4 * np.pi - angle) + log_center[1]
            hue = hue_from_power_color(10**x, 10**y, center)
            # Compare the angles in a safe way
            c2 = (np.sin(hue) - np.sin(angle)) ** 2 + (np.cos(hue) - np.cos(angle)) ** 2
            angle_diff = np.arccos((2.0 - c2) / 2.0)
            assert np.isclose(angle_diff, 0, atol=0.001)

    @pytest.mark.parametrize("plot_spans", [True, False])
    def test_plot_color(self, plot_spans):
        plot_power_colors(
            self.pc0,
            self.pc0e,
            self.pc1,
            self.pc1e,
            plot_spans=plot_spans,
            configuration=self.configuration,
        )

    @pytest.mark.parametrize("plot_spans", [True, False])
    @pytest.mark.parametrize("polar", [True, False])
    def test_hues(self, plot_spans, polar):
        plot_hues(
            self.rms,
            self.rmse,
            self.pc0,
            self.pc1,
            plot_spans=plot_spans,
            configuration=self.configuration,
            polar=polar,
        )

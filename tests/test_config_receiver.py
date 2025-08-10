"""
A Python module for radar simulation

---

- Copyright (C) 2018 - PRESENT  radarsimx.com
- E-mail: info@radarsimx.com
- Website: https://radarsimx.com

::

    ██████╗  █████╗ ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ███╗██╗  ██╗
    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗ ████║╚██╗██╔╝
    ██████╔╝███████║██║  ██║███████║██████╔╝███████╗██║██╔████╔██║ ╚███╔╝
    ██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗╚════██║██║██║╚██╔╝██║ ██╔██╗
    ██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║███████║██║██║ ╚═╝ ██║██╔╝ ██╗
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

"""

import pytest

import numpy as np

from radarsimpy import Receiver


class TestReceiver:
    """Test suite for the Receiver class."""

    def setup_method(self):
        """Set up common test parameters."""
        self.fs = 10e6
        self.noise_figure = 8
        self.rf_gain = 15
        self.load_resistor = 100
        self.baseband_gain = 10

    def test_init_basic(self):
        """Test initialization with basic parameters."""
        rx = Receiver(fs=self.fs)
        assert rx.bb_prop["fs"] == self.fs
        assert rx.rf_prop["noise_figure"] == 10  # default
        assert rx.rf_prop["rf_gain"] == 0  # default
        assert rx.bb_prop["load_resistor"] == 500  # default
        assert rx.bb_prop["baseband_gain"] == 0  # default
        assert rx.bb_prop["bb_type"] == "complex"  # default
        assert rx.bb_prop["noise_bandwidth"] == self.fs
        assert rx.rxchannel_prop["size"] == 1
        np.testing.assert_allclose(rx.rxchannel_prop["locations"], [[0, 0, 0]])

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        rx = Receiver(
            fs=self.fs,
            noise_figure=self.noise_figure,
            rf_gain=self.rf_gain,
            load_resistor=self.load_resistor,
            baseband_gain=self.baseband_gain,
            bb_type="real",
        )
        assert rx.bb_prop["fs"] == self.fs
        assert rx.rf_prop["noise_figure"] == self.noise_figure
        assert rx.rf_prop["rf_gain"] == self.rf_gain
        assert rx.bb_prop["load_resistor"] == self.load_resistor
        assert rx.bb_prop["baseband_gain"] == self.baseband_gain
        assert rx.bb_prop["bb_type"] == "real"
        assert rx.bb_prop["noise_bandwidth"] == self.fs / 2  # real type divides by 2

    def test_input_validation(self):
        """Test input validation in constructor."""
        # Test negative sampling rate
        with pytest.raises(ValueError, match="Sampling rate \\(fs\\) must be positive"):
            Receiver(fs=-1e6)

        # Test zero sampling rate
        with pytest.raises(ValueError, match="Sampling rate \\(fs\\) must be positive"):
            Receiver(fs=0)

        # Test invalid noise figure type
        with pytest.raises(ValueError, match="noise_figure must be a number"):
            Receiver(fs=self.fs, noise_figure="invalid")

        # Test invalid rf_gain type
        with pytest.raises(ValueError, match="rf_gain must be a number"):
            Receiver(fs=self.fs, rf_gain="invalid")

        # Test negative load resistor
        with pytest.raises(ValueError, match="load_resistor must be positive"):
            Receiver(fs=self.fs, load_resistor=-100)

        # Test zero load resistor
        with pytest.raises(ValueError, match="load_resistor must be positive"):
            Receiver(fs=self.fs, load_resistor=0)

        # Test invalid baseband gain type
        with pytest.raises(ValueError, match="baseband_gain must be a number"):
            Receiver(fs=self.fs, baseband_gain="invalid")

        # Test invalid baseband type
        with pytest.raises(
            ValueError,
            match="Invalid baseband type 'invalid'. Must be one of: complex, real",
        ):
            Receiver(fs=self.fs, bb_type="invalid")

    def test_validate_bb_prop(self):
        """Test validation of baseband properties."""
        # Test invalid bb_type
        rx = Receiver(fs=self.fs)
        with pytest.raises(
            ValueError,
            match="Invalid baseband type 'invalid'. Must be one of: complex, real",
        ):
            rx.bb_prop["bb_type"] = "invalid"
            rx.validate_bb_prop(rx.bb_prop)

        # Test negative sampling rate - use fresh receiver
        rx2 = Receiver(fs=self.fs)
        with pytest.raises(ValueError, match="Sampling rate \\(fs\\) must be positive"):
            rx2.bb_prop["fs"] = -1
            rx2.validate_bb_prop(rx2.bb_prop)

        # Test negative load resistor - use fresh receiver
        rx3 = Receiver(fs=self.fs)
        with pytest.raises(ValueError, match="Load resistor must be positive"):
            rx3.bb_prop["load_resistor"] = -100
            rx3.validate_bb_prop(rx3.bb_prop)

    def test_property_methods(self):
        """Test all property getter methods."""
        channels = [
            {"location": [1, 2, 3]},
            {"location": [4, 5, 6]},
        ]
        rx = Receiver(
            fs=self.fs,
            noise_figure=self.noise_figure,
            rf_gain=self.rf_gain,
            load_resistor=self.load_resistor,
            baseband_gain=self.baseband_gain,
            bb_type="complex",
            channels=channels,
        )

        # Test sampling_rate property
        assert rx.sampling_rate == self.fs

        # Test noise_bandwidth property (complex type)
        assert rx.noise_bandwidth == self.fs

        # Test num_channels property
        assert rx.num_channels == 2

        # Test channel_locations property
        expected_locations = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(rx.channel_locations, expected_locations)

        # Test noise_bandwidth for real type
        rx_real = Receiver(fs=self.fs, bb_type="real")
        assert rx_real.noise_bandwidth == self.fs / 2

    def test_get_channel_info(self):
        """Test the get_channel_info method."""
        channels = [
            {
                "location": (1, 2, 3),
                "polarization": [1, 0, 0],
                "azimuth_angle": [-45, 45],
                "azimuth_pattern": [-3, 0],
                "elevation_angle": [-30, 30],
                "elevation_pattern": [-6, 0],
            },
            {
                "location": (4, 5, 6),
                "polarization": [0, 1, 0],
                "azimuth_angle": [-60, 60],
                "azimuth_pattern": [-5, -2],
                "elevation_angle": [-45, 45],
                "elevation_pattern": [-8, -1],
            },
        ]
        rx = Receiver(fs=self.fs, channels=channels)

        # Test first channel info
        ch0_info = rx.get_channel_info(0)
        np.testing.assert_array_equal(ch0_info["location"], [1, 2, 3])
        np.testing.assert_array_equal(ch0_info["polarization"], [1, 0, 0])
        assert ch0_info["antenna_gain"] == 0  # max of [-3, 0]
        np.testing.assert_array_equal(ch0_info["azimuth_angles"], [-45, 45])
        np.testing.assert_array_equal(
            ch0_info["azimuth_pattern"], [-3, 0]
        )  # normalized
        np.testing.assert_array_equal(ch0_info["elevation_angles"], [-30, 30])
        np.testing.assert_array_equal(ch0_info["elevation_pattern"], [-6, 0])

        # Test second channel info
        ch1_info = rx.get_channel_info(1)
        np.testing.assert_array_equal(ch1_info["location"], [4, 5, 6])
        np.testing.assert_array_equal(ch1_info["polarization"], [0, 1, 0])
        assert ch1_info["antenna_gain"] == -2  # max of [-5, -2]

        # Test invalid channel index
        with pytest.raises(IndexError, match="Channel index 2 out of range \\[0, 1\\]"):
            rx.get_channel_info(2)

        with pytest.raises(
            IndexError, match="Channel index -1 out of range \\[0, 1\\]"
        ):
            rx.get_channel_info(-1)

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        rx = Receiver(
            fs=self.fs,
            noise_figure=self.noise_figure,
            rf_gain=self.rf_gain,
            load_resistor=self.load_resistor,
            baseband_gain=self.baseband_gain,
            bb_type="complex",
        )

        # Test __str__ method
        str_repr = str(rx)
        assert "Receiver" in str_repr
        assert "channels=1" in str_repr
        assert f"fs={self.fs/1e6:.1f} MHz" in str_repr
        assert f"noise_figure={self.noise_figure} dB" in str_repr
        assert "bb_type=complex" in str_repr

        # Test __repr__ method
        repr_str = repr(rx)
        assert "Receiver(" in repr_str
        assert f"fs={self.fs}" in repr_str
        assert f"noise_figure={self.noise_figure}" in repr_str
        assert f"rf_gain={self.rf_gain}" in repr_str
        assert f"load_resistor={self.load_resistor}" in repr_str
        assert f"baseband_gain={self.baseband_gain}" in repr_str
        assert "bb_type='complex'" in repr_str
        assert "channels=1" in repr_str

    def test_process_rxchannel_prop_basic(self):
        """Test processing of receiver channel properties with basic parameters."""
        rx = Receiver(fs=self.fs)
        channels = [{"location": (1, 2, 3)}]
        rxch_prop = rx.process_rxchannel_prop(channels)
        assert rxch_prop["size"] == 1
        np.testing.assert_allclose(rxch_prop["locations"], [[1, 2, 3]])
        np.testing.assert_allclose(rxch_prop["polarization"], [[0, 0, 1]])
        assert rxch_prop["antenna_gains"][0] == 0
        np.testing.assert_allclose(rxch_prop["az_angles"][0], [-90, 90])
        np.testing.assert_allclose(rxch_prop["az_patterns"][0], [0, 0])
        np.testing.assert_allclose(rxch_prop["el_angles"][0], [-90, 90])
        np.testing.assert_allclose(rxch_prop["el_patterns"][0], [0, 0])

    def test_process_rxchannel_prop_custom_parameters(self):
        """Test processing of receiver channel properties with custom parameters."""
        rx = Receiver(fs=self.fs)
        channels = [
            {
                "location": (4, 5, 6),
                "polarization": [1, 0, 0],
                "azimuth_angle": [-45, 45],
                "azimuth_pattern": [-3, -3],
                "elevation_angle": [-60, 60],
                "elevation_pattern": [-5, -5],
            }
        ]
        rxch_prop = rx.process_rxchannel_prop(channels)
        assert rxch_prop["size"] == 1
        np.testing.assert_allclose(rxch_prop["locations"], [[4, 5, 6]])
        np.testing.assert_allclose(rxch_prop["polarization"], [[1, 0, 0]])
        assert rxch_prop["antenna_gains"][0] == -3
        np.testing.assert_allclose(rxch_prop["az_angles"][0], [-45, 45])
        np.testing.assert_allclose(rxch_prop["az_patterns"][0], [0, 0])
        np.testing.assert_allclose(rxch_prop["el_angles"][0], [-60, 60])
        np.testing.assert_allclose(rxch_prop["el_patterns"][0], [0, 0])

    def test_process_rxchannel_prop_multiple_channels(self):
        """Test processing of receiver channel properties with multiple channels."""
        rx = Receiver(fs=self.fs)
        channels = [
            {"location": (0, 0, 0)},
            {
                "location": (10, 0, 0),
                "polarization": [0, 1, 0],
                "azimuth_angle": [-90, 90],
                "azimuth_pattern": [-10, -10],
                "elevation_angle": [-90, 90],
                "elevation_pattern": [-10, -10],
            },
        ]
        rxch_prop = rx.process_rxchannel_prop(channels)
        assert rxch_prop["size"] == 2
        np.testing.assert_allclose(rxch_prop["locations"], [[0, 0, 0], [10, 0, 0]])
        np.testing.assert_allclose(rxch_prop["polarization"], [[0, 0, 1], [0, 1, 0]])
        np.testing.assert_allclose(rxch_prop["antenna_gains"], [0, -10])

    def test_process_rxchannel_prop_invalid_pattern_length(self):
        """Test processing of receiver channel properties with invalid pattern length."""
        rx = Receiver(fs=self.fs)
        channels = [
            {
                "location": (0, 0, 0),
                "azimuth_angle": [-90, 0, 90],
                "azimuth_pattern": [-10, -5],
            }
        ]
        with pytest.raises(
            ValueError,
            match="Length mismatch for channel 0: azimuth_angle \\(3\\) and azimuth_pattern \\(2\\) must have same length",
        ):
            rx.process_rxchannel_prop(channels)

    def test_process_rxchannel_prop_invalid_elevation_pattern_length(self):
        """Test processing of receiver channel properties with invalid elevation pattern length."""
        rx = Receiver(fs=self.fs)
        channels = [
            {
                "location": (0, 0, 0),
                "elevation_angle": [-90, 0, 90],
                "elevation_pattern": [-10, -5],
            }
        ]
        with pytest.raises(
            ValueError,
            match="Length mismatch for channel 0: elevation_angle \\(3\\) and elevation_pattern \\(2\\) must have same length",
        ):
            rx.process_rxchannel_prop(channels)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test very small positive values
        rx = Receiver(fs=1e-6, load_resistor=1e-6)
        assert rx.sampling_rate == 1e-6
        assert rx.bb_prop["load_resistor"] == 1e-6

        # Test large values
        rx_large = Receiver(fs=1e12, load_resistor=1e6)
        assert rx_large.sampling_rate == 1e12
        assert rx_large.bb_prop["load_resistor"] == 1e6

        # Test single channel with complex patterns
        channels = [
            {
                "location": (0, 0, 0),
                "azimuth_angle": np.linspace(-180, 180, 37),
                "azimuth_pattern": np.random.uniform(-20, 0, 37),
                "elevation_angle": np.linspace(-90, 90, 19),
                "elevation_pattern": np.random.uniform(-15, 0, 19),
            }
        ]
        rx_complex = Receiver(fs=self.fs, channels=channels)
        assert rx_complex.num_channels == 1
        assert len(rx_complex.get_channel_info(0)["azimuth_angles"]) == 37
        assert len(rx_complex.get_channel_info(0)["elevation_angles"]) == 19

    def test_constants_usage(self):
        """Test that constants are properly used as defaults."""
        rx = Receiver(fs=self.fs)
        channel_info = rx.get_channel_info(0)

        # Check default polarization
        np.testing.assert_array_equal(channel_info["polarization"], [0, 0, 1])

        # Check default azimuth range
        np.testing.assert_array_equal(channel_info["azimuth_angles"], [-90, 90])

        # Check default elevation range
        np.testing.assert_array_equal(channel_info["elevation_angles"], [-90, 90])

        # Check default pattern values
        np.testing.assert_array_equal(channel_info["azimuth_pattern"], [0, 0])
        np.testing.assert_array_equal(channel_info["elevation_pattern"], [0, 0])

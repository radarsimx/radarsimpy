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
    def test_init_basic(self):
        """Test initialization with basic parameters."""
        rx = Receiver(fs=10e6)
        assert rx.bb_prop["fs"] == 10e6
        assert rx.rf_prop["noise_figure"] == 10
        assert rx.rf_prop["rf_gain"] == 0
        assert rx.bb_prop["load_resistor"] == 500
        assert rx.bb_prop["baseband_gain"] == 0
        assert rx.bb_prop["bb_type"] == "complex"
        assert rx.bb_prop["noise_bandwidth"] == 10e6
        assert rx.rxchannel_prop["size"] == 1
        np.testing.assert_allclose(rx.rxchannel_prop["locations"], [[0, 0, 0]])

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        rx = Receiver(
            fs=20e6,
            noise_figure=5,
            rf_gain=10,
            load_resistor=100,
            baseband_gain=20,
            bb_type="real",
        )
        assert rx.bb_prop["fs"] == 20e6
        assert rx.rf_prop["noise_figure"] == 5
        assert rx.rf_prop["rf_gain"] == 10
        assert rx.bb_prop["load_resistor"] == 100
        assert rx.bb_prop["baseband_gain"] == 20
        assert rx.bb_prop["bb_type"] == "real"
        assert rx.bb_prop["noise_bandwidth"] == 10e6

    def test_validate_bb_prop(self):
        """Test validation of baseband properties."""
        rx = Receiver(fs=10e6)
        with pytest.raises(ValueError):
            rx.bb_prop["bb_type"] = "invalid"
            rx.validate_bb_prop(rx.bb_prop)

    def test_process_rxchannel_prop_basic(self):
        """Test processing of receiver channel properties with basic parameters."""
        rx = Receiver(fs=10e6)
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
        rx = Receiver(fs=10e6)
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
        rx = Receiver(fs=10e6)
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
        rx = Receiver(fs=10e6)
        channels = [
            {
                "location": (0, 0, 0),
                "azimuth_angle": [-90, 90],
                "azimuth_pattern": [-10],
            }
        ]
        with pytest.raises(ValueError):
            rx.process_rxchannel_prop(channels)

    def test_process_rxchannel_prop_invalid_elevation_pattern_length(self):
        """Test processing of receiver channel properties with invalid elevation pattern length."""
        rx = Receiver(fs=10e6)
        channels = [
            {
                "location": (0, 0, 0),
                "elevation_angle": [-90, 90],
                "elevation_pattern": [-10],
            }
        ]
        with pytest.raises(ValueError):
            rx.process_rxchannel_prop(channels)

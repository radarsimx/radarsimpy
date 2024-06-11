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

import scipy.constants as const

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


def cw_rx():
    """
    Creates a continuous wave (CW) radar receiver.
    """
    return Receiver(
        fs=20,
        noise_figure=12,
        rf_gain=20,
        baseband_gain=50,
        load_resistor=1000,
        channels=[{"location": (0, 0, 0)}],
    )


def test_cw_rx():
    """
    Test the CW radar receiver.
    """
    print("#### CW receiver ####")
    cw = cw_rx()

    print("# CW receiver parameters #")
    assert cw.bb_prop["fs"] == 20
    assert cw.rf_prop["noise_figure"] == 12
    assert cw.rf_prop["rf_gain"] == 20
    assert cw.bb_prop["load_resistor"] == 1000
    assert cw.bb_prop["baseband_gain"] == 50
    assert cw.bb_prop["noise_bandwidth"] == cw.bb_prop["fs"]

    print("# CW receiver channel #")
    assert cw.rxchannel_prop["size"] == 1
    assert np.array_equal(cw.rxchannel_prop["locations"], np.array([[0, 0, 0]]))
    assert np.array_equal(cw.rxchannel_prop["az_angles"], [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.rxchannel_prop["az_patterns"], [np.zeros(2)])
    assert np.array_equal(cw.rxchannel_prop["el_angles"], [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.rxchannel_prop["el_patterns"], [np.zeros(2)])


def fmcw_rx():
    """
    Creates an FMCW radar receiver.
    """
    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6

    rx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
    }

    return Receiver(
        fs=2e6,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )


def test_fmcw_rx():
    """
    Test the FMCW radar receiver.
    """
    print("#### FMCW receiver ####")
    fmcw = fmcw_rx()

    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6
    pattern = pattern - np.max(pattern)

    print("# FMCW receiver parameters #")
    assert fmcw.bb_prop["fs"] == 2e6
    assert fmcw.rf_prop["noise_figure"] == 12
    assert fmcw.rf_prop["rf_gain"] == 20
    assert fmcw.bb_prop["load_resistor"] == 500
    assert fmcw.bb_prop["baseband_gain"] == 30
    assert fmcw.bb_prop["noise_bandwidth"] == fmcw.bb_prop["fs"]

    print("# FMCW receiver channel #")
    assert fmcw.rxchannel_prop["size"] == 1
    assert np.array_equal(fmcw.rxchannel_prop["locations"], np.array([[0, 0, 0]]))
    assert np.array_equal(fmcw.rxchannel_prop["az_angles"], [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.rxchannel_prop["az_patterns"], [pattern])
    assert np.array_equal(fmcw.rxchannel_prop["el_angles"], [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.rxchannel_prop["el_patterns"], [pattern])


def tdm_fmcw_rx():
    """
    Creates a TDM-FMCW radar receiver.
    """
    wavelength = const.c / 24.125e9
    channels = []
    for idx in range(0, 8):
        channels.append({"location": (0, wavelength / 2 * idx, 0)})

    return Receiver(
        fs=2e6,
        noise_figure=4,
        rf_gain=20,
        baseband_gain=50,
        load_resistor=500,
        channels=channels,
    )


def test_tdm_fmcw_rx():
    """
    Test the TDM-FMCW radar receiver.
    """
    print("#### TDM FMCW receiver ####")
    tdm = tdm_fmcw_rx()

    print("# TDM FMCW receiver parameters #")
    assert tdm.bb_prop["fs"] == 2e6
    assert tdm.rf_prop["noise_figure"] == 4
    assert tdm.rf_prop["rf_gain"] == 20
    assert tdm.bb_prop["load_resistor"] == 500
    assert tdm.bb_prop["baseband_gain"] == 50
    assert tdm.bb_prop["noise_bandwidth"] == tdm.bb_prop["fs"]

    print("# TDM FMCW receiver channel #")
    half_wavelength = const.c / 24.125e9 / 2
    assert tdm.rxchannel_prop["size"] == 8
    assert np.array_equal(
        tdm.rxchannel_prop["locations"],
        np.array(
            [
                [0, 0, 0],
                [0, half_wavelength, 0],
                [0, half_wavelength * 2, 0],
                [0, half_wavelength * 3, 0],
                [0, half_wavelength * 4, 0],
                [0, half_wavelength * 5, 0],
                [0, half_wavelength * 6, 0],
                [0, half_wavelength * 7, 0],
            ]
        ),
    )
    assert np.array_equal(
        tdm.rxchannel_prop["az_angles"],
        [
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
        ],
    )
    assert np.array_equal(
        tdm.rxchannel_prop["az_patterns"],
        [
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
        ],
    )
    assert np.array_equal(
        tdm.rxchannel_prop["el_angles"],
        [
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
        ],
    )
    assert np.array_equal(
        tdm.rxchannel_prop["el_patterns"],
        [
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
        ],
    )


def pmcw_rx():
    """
    Creates a PMCW radar receiver.
    """
    angle = np.arange(-90, 91, 1)
    pattern = np.ones(181) * 12

    return Receiver(
        fs=250e6,
        noise_figure=10,
        rf_gain=20,
        baseband_gain=30,
        load_resistor=1000,
        channels=[
            {
                "location": (0, 0, 0),
                "azimuth_angle": angle,
                "azimuth_pattern": pattern,
                "elevation_angle": angle,
                "elevation_pattern": pattern,
            }
        ],
    )


def test_pmcw_rx():
    """
    Test the PMCW radar receiver.
    """
    print("#### PMCW receiver ####")
    pmcw = pmcw_rx()

    print("# PMCW receiver parameters #")
    assert pmcw.bb_prop["fs"] == 250e6
    assert pmcw.rf_prop["noise_figure"] == 10
    assert pmcw.rf_prop["rf_gain"] == 20
    assert pmcw.bb_prop["load_resistor"] == 1000
    assert pmcw.bb_prop["baseband_gain"] == 30
    assert pmcw.bb_prop["noise_bandwidth"] == pmcw.bb_prop["fs"]

    print("# PMCW receiver channel #")
    assert pmcw.rxchannel_prop["size"] == 1
    assert np.array_equal(pmcw.rxchannel_prop["locations"], np.array([[0, 0, 0]]))
    assert np.array_equal(pmcw.rxchannel_prop["az_angles"], [np.arange(-90, 91, 1)])
    assert np.array_equal(pmcw.rxchannel_prop["az_patterns"], [np.zeros(181)])
    assert np.array_equal(pmcw.rxchannel_prop["el_angles"], [np.arange(-90, 91, 1)])
    assert np.array_equal(pmcw.rxchannel_prop["el_patterns"], [np.zeros(181)])

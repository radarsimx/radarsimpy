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
import scipy.constants as const
import numpy as np
import numpy.testing as npt

from radarsimpy import Transmitter


class TestTransmitter:
    def test_init_single_tone(self):
        """Test initialization with a single-tone waveform."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        assert tx.rf_prop["tx_power"] == 10
        assert tx.waveform_prop["f"][0] == 10e9
        assert tx.waveform_prop["f"][1] == 10e9
        assert tx.waveform_prop["t"][0] == 0
        assert tx.waveform_prop["t"][1] == 1e-6
        assert tx.waveform_prop["pulses"] == 10
        assert tx.waveform_prop["prp"][0] == 2e-6
        assert tx.waveform_prop["pulse_start_time"][0] == 0
        assert tx.waveform_prop["pulse_start_time"][9] == 1.8e-5

    def test_init_linear_modulation(self):
        """Test initialization with linear frequency modulation."""
        tx = Transmitter(f=[9e9, 11e9], t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        assert tx.waveform_prop["f"][0] == 9e9
        assert tx.waveform_prop["f"][1] == 11e9
        assert tx.waveform_prop["bandwidth"] == 2e9

    def test_init_arbitrary_waveform(self):
        """Test initialization with an arbitrary waveform."""
        f = np.linspace(9e9, 11e9, 100)
        t = np.linspace(0, 1e-6, 100)
        tx = Transmitter(f=f, t=t, tx_power=10, pulses=10, prp=2e-6)
        np.testing.assert_allclose(tx.waveform_prop["f"], f)
        np.testing.assert_allclose(tx.waveform_prop["t"], t)

    def test_init_frequency_offset(self):
        """Test initialization with frequency offset."""
        f_offset = np.linspace(0, 1e6, 10)
        tx = Transmitter(
            f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6, f_offset=f_offset
        )
        np.testing.assert_allclose(tx.waveform_prop["f_offset"], f_offset)

    def test_init_phase_noise(self):
        """Test initialization with phase noise."""
        pn_f = np.array([1e3, 1e4, 1e5])
        pn_power = np.array([-100, -110, -120])
        tx = Transmitter(
            f=10e9,
            t=1e-6,
            tx_power=10,
            pulses=10,
            prp=2e-6,
            pn_f=pn_f,
            pn_power=pn_power,
        )
        np.testing.assert_allclose(tx.rf_prop["pn_f"], pn_f)
        np.testing.assert_allclose(tx.rf_prop["pn_power"], pn_power)

    def test_validate_rf_prop(self):
        """Test validation of RF properties."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        with pytest.raises(ValueError):
            tx.rf_prop["pn_f"] = np.array([1e3, 1e4])
            tx.rf_prop["pn_power"] = None
            tx.validate_rf_prop(tx.rf_prop)
        with pytest.raises(ValueError):
            tx.rf_prop["pn_f"] = None
            tx.rf_prop["pn_power"] = np.array([-100, -110])
            tx.validate_rf_prop(tx.rf_prop)
        with pytest.raises(ValueError):
            tx.rf_prop["pn_f"] = np.array([1e3, 1e4])
            tx.rf_prop["pn_power"] = np.array([-100])
            tx.validate_rf_prop(tx.rf_prop)

    def test_validate_waveform_prop(self):
        """Test validation of waveform properties."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        with pytest.raises(ValueError):
            tx.waveform_prop["f"] = np.array([10e9, 10e9, 10e9])
            tx.validate_waveform_prop(tx.waveform_prop)
        with pytest.raises(ValueError):
            tx.waveform_prop["f_offset"] = np.linspace(0, 1e6, 9)
            tx.validate_waveform_prop(tx.waveform_prop)
        with pytest.raises(ValueError):
            tx.waveform_prop["prp"] = np.linspace(1e-6, 2e-6, 9)
            tx.validate_waveform_prop(tx.waveform_prop)
        with pytest.raises(ValueError):
            tx.waveform_prop["prp"] = np.array(
                [1e-7, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6]
            )
            tx.validate_waveform_prop(tx.waveform_prop)

    def test_process_waveform_modulation(self):
        """Test processing of waveform modulation parameters."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        mod_t = np.linspace(0, 1e-6, 10)
        amp = np.linspace(0, 1, 10)
        phs = np.linspace(0, 360, 10)
        mod = tx.process_waveform_modulation(mod_t, amp, phs)
        assert mod["enabled"]
        np.testing.assert_allclose(mod["t"], mod_t)
        np.testing.assert_allclose(np.abs(mod["var"]), amp)
        np.testing.assert_allclose(np.unwrap(np.angle(mod["var"])) / np.pi * 180, phs)

        # Test with only amplitude modulation
        mod = tx.process_waveform_modulation(mod_t, amp, None)
        assert mod["enabled"]
        np.testing.assert_allclose(mod["t"], mod_t)
        np.testing.assert_allclose(np.abs(mod["var"]), amp)
        np.testing.assert_allclose(
            np.angle(mod["var"]) / np.pi * 180, np.zeros_like(amp)
        )

        # Test with only phase modulation
        mod = tx.process_waveform_modulation(mod_t, None, phs)
        assert mod["enabled"]
        np.testing.assert_allclose(mod["t"], mod_t)
        np.testing.assert_allclose(np.abs(mod["var"]), np.ones_like(phs))
        np.testing.assert_allclose(np.unwrap(np.angle(mod["var"])) / np.pi * 180, phs)

        # Test with no modulation
        mod = tx.process_waveform_modulation(None, None, None)
        assert not mod["enabled"]

        with pytest.raises(ValueError):
            tx.process_waveform_modulation(mod_t, amp[:-1], phs)
        with pytest.raises(ValueError):
            tx.process_waveform_modulation(mod_t[:-1], amp, phs)

    def test_process_pulse_modulation(self):
        """Test processing of pulse modulation parameters."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        pulse_amp = np.linspace(0, 1, 10)
        pulse_phs = np.linspace(0, 360, 10)
        pulse_mod = tx.process_pulse_modulation(pulse_amp, pulse_phs)
        np.testing.assert_allclose(np.abs(pulse_mod), pulse_amp)
        np.testing.assert_allclose(
            np.unwrap(np.angle(pulse_mod)) / np.pi * 180, pulse_phs
        )

        with pytest.raises(ValueError):
            tx.process_pulse_modulation(pulse_amp[:-1], pulse_phs)
        with pytest.raises(ValueError):
            tx.process_pulse_modulation(pulse_amp, pulse_phs[:-1])

    def test_process_txchannel_prop(self):
        """Test processing of transmitter channel properties."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        channels = [
            {
                "location": (0, 0, 0),
                "polarization": [1, 0, 0],
                "delay": 1e-6,
                "azimuth_angle": [-90, 90],
                "azimuth_pattern": [0, 0],
                "elevation_angle": [-90, 90],
                "elevation_pattern": [0, 0],
                "pulse_amp": np.linspace(0, 1, 10),
                "pulse_phs": np.linspace(0, 360, 10),
                "mod_t": np.linspace(0, 1e-6, 10),
                "amp": np.linspace(0, 1, 10),
                "phs": np.linspace(0, 360, 10),
            },
            {
                "location": (10, 0, 0),
                "polarization": [0, 1, 0],
                "delay": 2e-6,
                "azimuth_angle": [-90, 90],
                "azimuth_pattern": [-10, -10],
                "elevation_angle": [-90, 90],
                "elevation_pattern": [-10, -10],
                "pulse_amp": np.linspace(1, 0, 10),
                "pulse_phs": np.linspace(360, 0, 10),
            },
        ]
        txch_prop = tx.process_txchannel_prop(channels)
        assert txch_prop["size"] == 2
        np.testing.assert_allclose(txch_prop["delay"], [1e-6, 2e-6])
        np.testing.assert_allclose(txch_prop["locations"], [[0, 0, 0], [10, 0, 0]])
        np.testing.assert_allclose(txch_prop["polarization"], [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(txch_prop["antenna_gains"], [0, -10])
        np.testing.assert_allclose(
            np.abs(txch_prop["pulse_mod"][0, :]), np.linspace(0, 1, 10)
        )
        np.testing.assert_allclose(
            np.unwrap(np.angle(txch_prop["pulse_mod"][0, :])) / np.pi * 180,
            np.linspace(0, 360, 10),
        )
        np.testing.assert_allclose(
            np.abs(txch_prop["pulse_mod"][1, :]), np.linspace(1, 0, 10)
        )
        np.testing.assert_allclose(
            np.unwrap(np.angle(txch_prop["pulse_mod"][1, :])) / np.pi * 180 + 360,
            np.linspace(360, 0, 10),
        )
        assert txch_prop["waveform_mod"][0]["enabled"]
        assert txch_prop["waveform_mod"][1]["enabled"] is False

        with pytest.raises(ValueError):
            channels[0]["azimuth_angle"] = [-90, 90, 0]
            tx.process_txchannel_prop(channels)
        with pytest.raises(ValueError):
            channels[0]["elevation_angle"] = [-90, 90, 0]
            tx.process_txchannel_prop(channels)

    def test_init_channels(self):
        """Test initialization with multiple channels."""
        channels = [
            {"location": (0, 0, 0)},
            {"location": (10, 0, 0)},
        ]
        tx = Transmitter(
            f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6, channels=channels
        )
        assert tx.txchannel_prop["size"] == 2
        np.testing.assert_allclose(
            tx.txchannel_prop["locations"], [[0, 0, 0], [10, 0, 0]]
        )

    def test_init_channels_with_modulation(self):
        """Test initialization with multiple channels and modulation."""
        channels = [
            {
                "location": (0, 0, 0),
                "pulse_amp": np.linspace(0, 1, 10),
                "pulse_phs": np.linspace(0, 360, 10),
                "mod_t": np.linspace(0, 1e-6, 10),
                "amp": np.linspace(0, 1, 10),
                "phs": np.linspace(0, 360, 10),
            },
            {
                "location": (10, 0, 0),
                "pulse_amp": np.linspace(1, 0, 10),
                "pulse_phs": np.linspace(360, 0, 10),
            },
        ]
        tx = Transmitter(
            f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6, channels=channels
        )
        assert tx.txchannel_prop["size"] == 2
        np.testing.assert_allclose(
            tx.txchannel_prop["locations"], [[0, 0, 0], [10, 0, 0]]
        )
        np.testing.assert_allclose(
            np.abs(tx.txchannel_prop["pulse_mod"][0, :]), np.linspace(0, 1, 10)
        )
        np.testing.assert_allclose(
            np.unwrap(np.angle(tx.txchannel_prop["pulse_mod"][0, :])) / np.pi * 180,
            np.linspace(0, 360, 10),
        )
        np.testing.assert_allclose(
            np.abs(tx.txchannel_prop["pulse_mod"][1, :]), np.linspace(1, 0, 10)
        )
        np.testing.assert_allclose(
            np.unwrap(np.angle(tx.txchannel_prop["pulse_mod"][1, :])) / np.pi * 180
            + 360,
            np.linspace(360, 0, 10),
        )
        assert tx.txchannel_prop["waveform_mod"][0]["enabled"]
        assert tx.txchannel_prop["waveform_mod"][1]["enabled"] is False


def cw_tx():
    """
    Creates a continuous wave (CW) radar transmitter.
    """
    return Transmitter(f=24e9, t=10, tx_power=10, pulses=2)


def test_cw_tx():
    """
    Test the CW radar transmitter.
    """
    print("#### CW transmitter ####")
    cw = cw_tx()

    print("# CW transmitter parameters #")
    # assert np.array_equal(cw.fc_vect, np.ones(2)*24e9)
    # assert cw.pulse_length == 10
    # assert cw.bandwidth == 0
    assert cw.rf_prop["tx_power"] == 10
    assert cw.waveform_prop["prp"][0] == 10
    assert cw.waveform_prop["pulses"] == 2

    print("# CW transmitter channel #")
    assert cw.txchannel_prop["size"] == 1
    assert np.array_equal(cw.txchannel_prop["locations"], np.array([[0, 0, 0]]))
    assert np.array_equal(cw.txchannel_prop["az_angles"], [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.txchannel_prop["az_patterns"], [np.zeros(2)])
    assert np.array_equal(cw.txchannel_prop["el_angles"], [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.txchannel_prop["el_patterns"], [np.zeros(2)])

    print("# CW transmitter modulation #")
    assert np.array_equal(
        cw.txchannel_prop["pulse_mod"], [np.ones(cw.waveform_prop["pulses"])]
    )


def fmcw_tx():
    """
    Creates an FMCW radar transmitter.
    """
    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6

    tx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
    }

    return Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=256,
        channels=[tx_channel],
    )


def test_fmcw_tx():
    """
    Test the FMCW radar transmitter.
    """
    print("#### FMCW transmitter ####")
    fmcw = fmcw_tx()

    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6
    pattern = pattern - np.max(pattern)

    print("# FMCW transmitter parameters #")
    # assert np.array_equal(fmcw.fc_vect, np.ones(256) * 24.125e9)
    assert fmcw.waveform_prop["pulse_length"] == 80e-6
    assert fmcw.waveform_prop["bandwidth"] == 100e6
    assert fmcw.rf_prop["tx_power"] == 10
    assert fmcw.waveform_prop["prp"][0] == 100e-6
    assert fmcw.waveform_prop["pulses"] == 256

    print("# FMCW transmitter channel #")
    assert fmcw.txchannel_prop["size"] == 1
    assert np.array_equal(fmcw.txchannel_prop["locations"], np.array([[0, 0, 0]]))
    assert np.array_equal(fmcw.txchannel_prop["az_angles"], [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.txchannel_prop["az_patterns"], [pattern])
    assert np.array_equal(fmcw.txchannel_prop["el_angles"], [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.txchannel_prop["el_patterns"], [pattern])

    print("# FMCW transmitter modulation #")
    assert np.array_equal(
        fmcw.txchannel_prop["pulse_mod"], [np.ones(fmcw.waveform_prop["pulses"])]
    )


def tdm_fmcw_tx():
    """
    Creates a TDM-FMCW radar transmitter.
    """
    wavelength = const.c / 24.125e9

    tx_channel_1 = {"location": (0, -4 * wavelength, 0), "delay": 0}
    tx_channel_2 = {"location": (0, 0, 0), "delay": 100e-6}

    return Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=20,
        prp=200e-6,
        pulses=2,
        channels=[tx_channel_1, tx_channel_2],
    )


def test_tdm_fmcw_tx():
    """
    Test the TDM-FMCW radar transmitter.
    """
    print("#### TDM FMCW transmitter ####")
    tdm = tdm_fmcw_tx()

    print("# TDM FMCW transmitter parameters #")
    # assert np.array_equal(tdm.fc_vect, np.ones(2) * 24.125e9)
    assert tdm.waveform_prop["pulse_length"] == 80e-6
    assert tdm.waveform_prop["bandwidth"] == 100e6
    assert tdm.rf_prop["tx_power"] == 20
    assert tdm.waveform_prop["prp"][0] == 200e-6
    assert tdm.waveform_prop["pulses"] == 2

    print("# TDM FMCW transmitter channel #")
    assert tdm.txchannel_prop["size"] == 2
    assert np.array_equal(
        tdm.txchannel_prop["locations"],
        np.array([[0, -4 * const.c / 24.125e9, 0], [0, 0, 0]]),
    )
    assert np.array_equal(
        tdm.txchannel_prop["az_angles"],
        [np.arange(-90, 91, 180), np.arange(-90, 91, 180)],
    )
    assert np.array_equal(tdm.txchannel_prop["az_patterns"], [np.zeros(2), np.zeros(2)])
    assert np.array_equal(
        tdm.txchannel_prop["el_angles"],
        [np.arange(-90, 91, 180), np.arange(-90, 91, 180)],
    )
    assert np.array_equal(tdm.txchannel_prop["el_patterns"], [np.zeros(2), np.zeros(2)])

    print("# TDM FMCW transmitter modulation #")
    assert np.array_equal(
        tdm.txchannel_prop["pulse_mod"],
        [np.ones(tdm.waveform_prop["pulses"]), np.ones(tdm.waveform_prop["pulses"])],
    )


def pmcw_tx(code1, code2):
    """
    Creates a PMCW radar transmitter.
    """
    angle = np.arange(-90, 91, 1)
    pattern = np.ones(181) * 12

    pulse_phs1 = np.zeros(np.shape(code1))
    pulse_phs2 = np.zeros(np.shape(code2))
    pulse_phs1[np.where(code1 == 1)] = 0
    pulse_phs1[np.where(code1 == -1)] = 180
    pulse_phs2[np.where(code2 == 1)] = 0
    pulse_phs2[np.where(code2 == -1)] = 180

    mod_t1 = np.arange(0, len(code1)) * 4e-9
    mod_t2 = np.arange(0, len(code2)) * 4e-9

    tx_channel_1 = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
        "mod_t": mod_t1,
        "phs": pulse_phs1,
    }

    tx_channel_2 = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
        "mod_t": mod_t2,
        "phs": pulse_phs2,
    }

    return Transmitter(
        f=24.125e9,
        t=2.1e-6,
        tx_power=20,
        pulses=256,
        channels=[tx_channel_1, tx_channel_2],
    )


def test_pmcw_tx():
    """
    Test the PMCW radar transmitter.
    """
    code1 = np.array(
        [
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
        ]
    )
    code2 = np.array(
        [
            1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
        ]
    )

    print("#### PMCW transmitter ####")
    pmcw = pmcw_tx(code1, code2)

    print("# PMCW transmitter parameters #")
    # assert np.array_equal(pmcw.fc_vect, np.ones(256) * 24.125e9)
    assert pmcw.waveform_prop["pulse_length"] == 2.1e-6
    assert pmcw.waveform_prop["bandwidth"] == 0
    assert pmcw.rf_prop["tx_power"] == 20
    assert pmcw.waveform_prop["prp"][0] == 2.1e-6
    assert pmcw.waveform_prop["pulses"] == 256

    print("# PMCW transmitter channel #")
    assert pmcw.txchannel_prop["size"] == 2
    assert np.array_equal(
        pmcw.txchannel_prop["locations"], np.array([[0, 0, 0], [0, 0, 0]])
    )
    assert np.array_equal(
        pmcw.txchannel_prop["az_angles"], [np.arange(-90, 91, 1), np.arange(-90, 91, 1)]
    )
    assert np.array_equal(
        pmcw.txchannel_prop["az_patterns"], [np.zeros(181), np.zeros(181)]
    )
    assert np.array_equal(
        pmcw.txchannel_prop["el_angles"], [np.arange(-90, 91, 1), np.arange(-90, 91, 1)]
    )
    assert np.array_equal(
        pmcw.txchannel_prop["el_patterns"], [np.zeros(181), np.zeros(181)]
    )

    print("# PMCW transmitter modulation #")
    npt.assert_almost_equal(pmcw.txchannel_prop["waveform_mod"][0]["var"], code1)
    npt.assert_almost_equal(pmcw.txchannel_prop["waveform_mod"][1]["var"], code2)

    npt.assert_almost_equal(
        pmcw.txchannel_prop["waveform_mod"][0]["t"], np.arange(0, len(code1)) * 4e-9
    )
    npt.assert_almost_equal(
        pmcw.txchannel_prop["waveform_mod"][1]["t"], np.arange(0, len(code2)) * 4e-9
    )

    # assert np.array_equal(pmcw.chip_length, [4e-9, 4e-9])

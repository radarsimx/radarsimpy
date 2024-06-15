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
            atol=1e-5,
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
            atol=1e-5,
        )
        assert tx.txchannel_prop["waveform_mod"][0]["enabled"]
        assert tx.txchannel_prop["waveform_mod"][1]["enabled"] is False

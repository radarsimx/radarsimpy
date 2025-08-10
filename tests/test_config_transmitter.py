"""
A Python module for radar simulation - Transmitter Unit Tests

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
    """Test suite for the Transmitter class."""

    def setup_method(self):
        """Set up common test parameters."""
        self.f_single = 10e9
        self.t_single = 1e-6
        self.tx_power = 10
        self.pulses = 10
        self.prp = 2e-6

    def test_init_single_tone(self):
        """Test initialization with a single-tone waveform."""
        tx = Transmitter(
            f=self.f_single,
            t=self.t_single,
            tx_power=self.tx_power,
            pulses=self.pulses,
            prp=self.prp,
        )
        assert tx.rf_prop["tx_power"] == self.tx_power
        assert tx.waveform_prop["f"][0] == self.f_single
        assert tx.waveform_prop["f"][1] == self.f_single
        assert tx.waveform_prop["t"][0] == 0
        assert tx.waveform_prop["t"][1] == self.t_single
        assert tx.waveform_prop["pulses"] == self.pulses
        assert tx.waveform_prop["prp"][0] == self.prp
        assert tx.waveform_prop["pulse_start_time"][0] == 0
        assert tx.waveform_prop["pulse_start_time"][9] == 1.8e-5

    def test_property_methods(self):
        """Test all property getter methods."""
        tx = Transmitter(
            f=[9e9, 11e9],
            t=self.t_single,
            tx_power=self.tx_power,
            pulses=self.pulses,
            prp=self.prp,
        )

        # Test frequency property
        np.testing.assert_array_equal(tx.frequency, [9e9, 11e9])

        # Test bandwidth property
        assert tx.bandwidth == 2e9

        # Test pulse_length property
        assert tx.pulse_length == self.t_single

        # Test num_pulses property
        assert tx.num_pulses == self.pulses

        # Test num_channels property (default single channel)
        assert tx.num_channels == 1

        # Test channel_locations property
        expected_locations = np.array([[0, 0, 0]])
        np.testing.assert_array_equal(tx.channel_locations, expected_locations)

    def test_get_channel_info(self):
        """Test the get_channel_info method."""
        channels = [
            {
                "location": (1, 2, 3),
                "polarization": [1, 0, 0],
                "delay": 1e-6,
                "grid": 2.0,
                "azimuth_angle": [-45, 45],
                "azimuth_pattern": [-3, 0],
                "elevation_angle": [-30, 30],
                "elevation_pattern": [-6, 0],
            },
            {
                "location": (4, 5, 6),
                "polarization": [0, 1, 0],
                "delay": 2e-6,
            },
        ]

        tx = Transmitter(f=self.f_single, t=self.t_single, channels=channels)

        # Test valid channel info
        info0 = tx.get_channel_info(0)
        np.testing.assert_array_equal(info0["location"], [1, 2, 3])
        np.testing.assert_array_equal(info0["polarization"], [1, 0, 0])
        assert info0["delay"] == 1e-6
        assert info0["grid"] == 2.0
        assert info0["antenna_gain"] == 0  # max of [-3, 0]
        np.testing.assert_array_equal(info0["azimuth_angles"], [-45, 45])
        np.testing.assert_array_equal(info0["azimuth_pattern"], [-3, 0])

        info1 = tx.get_channel_info(1)
        np.testing.assert_array_equal(info1["location"], [4, 5, 6])
        np.testing.assert_array_equal(info1["polarization"], [0, 1, 0])
        assert info1["delay"] == 2e-6
        assert info1["grid"] == 1.0  # default value

        # Test invalid channel index
        with pytest.raises(IndexError, match="Channel index 2 out of range"):
            tx.get_channel_info(2)

        with pytest.raises(IndexError, match="Channel index -1 out of range"):
            tx.get_channel_info(-1)

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        tx = Transmitter(f=[24e9, 24.5e9], t=100e-6, tx_power=20, pulses=5, prp=200e-6)

        # Test __str__ method
        str_repr = str(tx)
        assert "Transmitter(channels=1" in str_repr
        assert "pulses=5" in str_repr
        assert "bandwidth=0.500 GHz" in str_repr
        assert "pulse_length=100.0 μs" in str_repr

        # Test __repr__ method
        repr_str = repr(tx)
        assert "Transmitter(f=" in repr_str
        assert "tx_power=20" in repr_str
        assert "pulses=5" in repr_str
        assert "channels=1" in repr_str

    def test_input_validation(self):
        """Test input validation in constructor."""
        # Test invalid pulses
        with pytest.raises(ValueError, match="Number of pulses must be at least 1"):
            Transmitter(f=self.f_single, t=self.t_single, pulses=0)

        with pytest.raises(ValueError, match="Number of pulses must be at least 1"):
            Transmitter(f=self.f_single, t=self.t_single, pulses=-1)

        # Test invalid tx_power type (we'll test this with an actual runtime error)
        # Note: This test verifies runtime validation beyond type hints

    def test_init_linear_modulation(self):
        """Test initialization with linear frequency modulation."""
        tx = Transmitter(
            f=[9e9, 11e9],
            t=self.t_single,
            tx_power=self.tx_power,
            pulses=self.pulses,
            prp=self.prp,
        )
        assert tx.frequency[0] == 9e9  # Using property instead of direct access
        assert tx.frequency[1] == 11e9
        assert tx.bandwidth == 2e9  # Using property

    def test_init_arbitrary_waveform(self):
        """Test initialization with an arbitrary waveform."""
        f = np.linspace(9e9, 11e9, 100)
        t = np.linspace(0, 1e-6, 100)
        tx = Transmitter(
            f=f, t=t, tx_power=self.tx_power, pulses=self.pulses, prp=self.prp
        )
        np.testing.assert_allclose(tx.frequency, f)  # Using property
        np.testing.assert_allclose(tx.waveform_prop["t"], t)
        assert tx.pulse_length == pytest.approx(1e-6)  # Using property

    def test_init_frequency_offset(self):
        """Test initialization with frequency offset."""
        f_offset = np.linspace(0, 1e6, 10)
        tx = Transmitter(
            f=self.f_single,
            t=self.t_single,
            tx_power=self.tx_power,
            pulses=self.pulses,
            prp=self.prp,
            f_offset=f_offset,
        )
        np.testing.assert_allclose(tx.waveform_prop["f_offset"], f_offset)

        # Test scalar f_offset
        tx_scalar = Transmitter(
            f=self.f_single, t=self.t_single, pulses=5, f_offset=1e6
        )
        expected_offset = np.full(5, 1e6)
        np.testing.assert_allclose(tx_scalar.waveform_prop["f_offset"], expected_offset)

        # Test length mismatch
        with pytest.raises(
            ValueError, match="f_offset length \\(3\\) must match pulses \\(5\\)"
        ):
            Transmitter(
                f=self.f_single,
                t=self.t_single,
                pulses=5,
                f_offset=np.array(
                    [1e6, 2e6, 3e6]
                ),  # Wrong length: 3 elements for 5 pulses
            )

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
        tx = Transmitter(
            f=self.f_single,
            t=self.t_single,
            tx_power=self.tx_power,
            pulses=self.pulses,
            prp=self.prp,
        )

        # Test when only pn_f is provided
        with pytest.raises(
            ValueError, match="Both `pn_f` and `pn_power` must be provided together"
        ):
            tx.rf_prop["pn_f"] = np.array([1e3, 1e4])
            tx.rf_prop["pn_power"] = None
            tx.validate_rf_prop(tx.rf_prop)

        # Test when only pn_power is provided
        with pytest.raises(
            ValueError, match="Both `pn_f` and `pn_power` must be provided together"
        ):
            tx.rf_prop["pn_f"] = None
            tx.rf_prop["pn_power"] = np.array([-100, -110])
            tx.validate_rf_prop(tx.rf_prop)

        # Test length mismatch
        with pytest.raises(
            ValueError, match="Lengths of `pn_f` and `pn_power` should be the same"
        ):
            tx.rf_prop["pn_f"] = np.array([1e3, 1e4])
            tx.rf_prop["pn_power"] = np.array([-100])
            tx.validate_rf_prop(tx.rf_prop)

    def test_validate_waveform_prop(self):
        """Test validation of waveform properties."""
        tx = Transmitter(
            f=self.f_single,
            t=self.t_single,
            tx_power=self.tx_power,
            pulses=self.pulses,
            prp=self.prp,
        )

        # Test f and t length mismatch
        with pytest.raises(
            ValueError,
            match="Lengths of `f` \\(3\\) and `t` \\(2\\) should be the same",
        ):
            tx.waveform_prop["f"] = np.array([10e9, 10e9, 10e9])
            tx.validate_waveform_prop(tx.waveform_prop)

        # Test f_offset length mismatch with better error message
        with pytest.raises(
            ValueError, match="f_offset length \\(9\\) must match pulses \\(10\\)"
        ):
            # Reset f to correct length first
            tx.waveform_prop["f"] = np.array([10e9, 10e9])
            tx.waveform_prop["f_offset"] = np.linspace(0, 1e6, 9)
            tx.validate_waveform_prop(tx.waveform_prop)

        # Test prp length mismatch with better error message
        with pytest.raises(
            ValueError, match="prp length \\(9\\) must match pulses \\(10\\)"
        ):
            # Reset f and f_offset to correct lengths first
            tx.waveform_prop["f"] = np.array([10e9, 10e9])
            tx.waveform_prop["f_offset"] = np.zeros(10)
            tx.waveform_prop["prp"] = np.linspace(1e-6, 2e-6, 9)
            tx.validate_waveform_prop(tx.waveform_prop)

        # Test prp smaller than pulse_length with better error message
        with pytest.raises(
            ValueError,
            match="All PRP values \\(1\\.00e-07 s\\) must be >= pulse_length \\(1\\.00e-06 s\\)",
        ):
            # Reset f, f_offset, and prp to correct lengths first
            tx.waveform_prop["f"] = np.array([10e9, 10e9])
            tx.waveform_prop["f_offset"] = np.zeros(10)
            tx.waveform_prop["prp"] = np.array(
                [1e-7, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6]
            )
            tx.validate_waveform_prop(tx.waveform_prop)

    def test_process_waveform_modulation(self):
        """Test processing of waveform modulation parameters."""
        tx = Transmitter(
            f=self.f_single,
            t=self.t_single,
            tx_power=self.tx_power,
            pulses=self.pulses,
            prp=self.prp,
        )
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

        # Test length validation with improved error messages
        with pytest.raises(
            ValueError,
            match="Lengths of `amp` \\(9\\) and `phs` \\(10\\) should be the same",
        ):
            tx.process_waveform_modulation(mod_t, amp[:-1], phs)
        with pytest.raises(
            ValueError,
            match="Lengths of `mod_t` \\(9\\) and `amp` \\(10\\) should be the same",
        ):
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
            f=self.f_single,
            t=self.t_single,
            tx_power=self.tx_power,
            pulses=self.pulses,
            prp=self.prp,
            channels=channels,
        )
        assert tx.num_channels == 2  # Using property
        np.testing.assert_allclose(
            tx.channel_locations, [[0, 0, 0], [10, 0, 0]]  # Using property
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
            f=self.f_single,
            t=self.t_single,
            tx_power=self.tx_power,
            pulses=self.pulses,
            prp=self.prp,
            channels=channels,
        )
        assert tx.num_channels == 2  # Using property
        np.testing.assert_allclose(
            tx.channel_locations, [[0, 0, 0], [10, 0, 0]]  # Using property
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

    def test_comprehensive_integration(self):
        """Test comprehensive integration of all features."""
        # Create a complex transmitter with all features
        f_offset = np.linspace(0, 1e6, 5)
        pn_f = np.array([1e3, 1e4, 1e5])
        pn_power = np.array([-100, -110, -120])

        channels = [
            {
                "location": [0, 0, 0],
                "polarization": [1, 0, 0],
                "delay": 1e-6,
                "grid": 2.0,
                "azimuth_angle": [-60, 60],
                "azimuth_pattern": [-3, 0],
                "elevation_angle": [-30, 30],
                "elevation_pattern": [-6, 0],
                "pulse_amp": np.ones(5),
                "pulse_phs": np.zeros(5),
                "mod_t": np.linspace(0, 1e-6, 10),
                "amp": np.ones(10),
                "phs": np.zeros(10),
            },
            {
                "location": [0.05, 0, 0],
                "polarization": [0, 1, 0],
                "delay": 1.5e-6,
            },
        ]

        tx = Transmitter(
            f=[24e9, 24.5e9],
            t=100e-6,
            tx_power=30,
            pulses=5,
            prp=200e-6,
            f_offset=f_offset,
            pn_f=pn_f,
            pn_power=pn_power,
            channels=channels,
        )

        # Test all properties work correctly
        assert tx.num_channels == 2
        assert tx.num_pulses == 5
        assert tx.bandwidth == 0.5e9
        assert tx.pulse_length == 100e-6
        np.testing.assert_array_equal(tx.frequency, [24e9, 24.5e9])

        # Test RF properties
        np.testing.assert_array_equal(tx.rf_prop["pn_f"], pn_f)
        np.testing.assert_array_equal(tx.rf_prop["pn_power"], pn_power)

        # Test channel information
        ch0_info = tx.get_channel_info(0)
        assert ch0_info["delay"] == 1e-6
        assert ch0_info["grid"] == 2.0
        np.testing.assert_array_equal(ch0_info["polarization"], [1, 0, 0])

        ch1_info = tx.get_channel_info(1)
        assert ch1_info["delay"] == 1.5e-6
        assert ch1_info["grid"] == 1.0  # default
        np.testing.assert_array_equal(ch1_info["polarization"], [0, 1, 0])

        # Test string representations
        str_repr = str(tx)
        assert "channels=2" in str_repr
        assert "pulses=5" in str_repr
        assert "bandwidth=0.500 GHz" in str_repr

    def test_constants_and_defaults(self):
        """Test that default constants are used properly."""
        tx = Transmitter(f=self.f_single, t=self.t_single)

        # Test default channel uses constants
        channel_info = tx.get_channel_info(0)
        np.testing.assert_array_equal(channel_info["location"], [0, 0, 0])
        np.testing.assert_array_equal(
            channel_info["polarization"], [0, 0, 1]
        )  # DEFAULT_POLARIZATION
        assert channel_info["grid"] == 1.0  # DEFAULT_GRID_SIZE

        # Test with custom channel that should use defaults
        channels = [{"location": [1, 2, 3]}]  # Only location specified
        tx_custom = Transmitter(f=self.f_single, t=self.t_single, channels=channels)

        channel_info_custom = tx_custom.get_channel_info(0)
        np.testing.assert_array_equal(channel_info_custom["location"], [1, 2, 3])
        np.testing.assert_array_equal(
            channel_info_custom["polarization"], [0, 0, 1]
        )  # DEFAULT_POLARIZATION
        assert channel_info_custom["grid"] == 1.0  # DEFAULT_GRID_SIZE
        np.testing.assert_array_equal(
            channel_info_custom["azimuth_angles"], [-90, 90]
        )  # DEFAULT_AZIMUTH_RANGE
        np.testing.assert_array_equal(
            channel_info_custom["elevation_angles"], [-90, 90]
        )  # DEFAULT_ELEVATION_RANGE

    def test_improved_error_messages(self):
        """Test that error messages are descriptive and helpful."""
        channels = [
            {
                "location": (0, 0, 0),
                "azimuth_angle": [-90, 90, 0],  # Wrong length
                "azimuth_pattern": [0, 0],
            }
        ]

        # Test improved error message with channel information
        with pytest.raises(ValueError, match="Length mismatch for channel 0"):
            Transmitter(f=self.f_single, t=self.t_single, channels=channels)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test single pulse
        tx_single = Transmitter(f=self.f_single, t=self.t_single, pulses=1)
        assert tx_single.num_pulses == 1
        assert len(tx_single.waveform_prop["prp"]) == 1

        # Test very small values
        tx_small = Transmitter(f=1e9, t=1e-9, pulses=1, prp=2e-9)
        assert tx_small.pulse_length == 1e-9
        assert tx_small.bandwidth == 0.0  # Single tone

        # Test large arrays
        f_large = np.linspace(1e9, 2e9, 1000)
        t_large = np.linspace(0, 1e-3, 1000)
        tx_large = Transmitter(f=f_large, t=t_large, pulses=1)
        assert len(tx_large.frequency) == 1000
        assert tx_large.pulse_length == 1e-3

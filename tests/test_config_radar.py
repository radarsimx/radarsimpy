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

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.radar import (
    cal_phase_noise,
    _interpolate_phase_noise_power,
    _generate_noise_spectrum,
    BOLTZMANN_CONSTANT,
    SQRT_HALF,
    MILLIWATTS_TO_WATTS,
)


class TestRadar:
    """
    Test suite for the Radar class and related functions.

    This test suite covers:
    - Basic Radar initialization and properties
    - Phase noise calculation and validation
    - Input validation and error handling
    - Helper function testing
    - String representations and property accessors
    """

    @pytest.fixture
    def radar_setup(self):
        """Fixture for setting up a basic radar system."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        rx = Receiver(fs=10e6)
        radar = Radar(transmitter=tx, receiver=rx)
        return radar

    def test_init_basic(self, radar_setup):
        """Test initialization with basic parameters."""
        radar = radar_setup
        # assert radar.time_prop["frame_size"] == 2
        # assert np.allclose(radar.time_prop["frame_start_time"], [0, 1e-3])
        assert radar.sample_prop["samples_per_pulse"] == 10
        assert radar.array_prop["size"] == 1
        np.testing.assert_allclose(radar.array_prop["virtual_array"], [[0, 0, 0]])
        assert radar.radar_prop["transmitter"].waveform_prop["f"][0] == 10e9
        assert radar.radar_prop["receiver"].bb_prop["fs"] == 10e6
        assert np.allclose(radar.radar_prop["location"], [0, 0, 0])
        assert np.allclose(radar.radar_prop["speed"], [0, 0, 0])
        assert np.allclose(radar.radar_prop["rotation"], [0, 0, 0])
        assert np.allclose(radar.radar_prop["rotation_rate"], [0, 0, 0])

        # Test new properties
        assert radar.num_channels == 1
        assert radar.samples_per_pulse == 10
        np.testing.assert_allclose(radar.virtual_array_locations, [[0, 0, 0]])
        assert isinstance(radar.transmitter, type(radar.radar_prop["transmitter"]))
        assert isinstance(radar.receiver, type(radar.radar_prop["receiver"]))

    def test_origin_timestamp(self, radar_setup):
        """Test timestamp generation."""
        radar = radar_setup
        timestamp = radar.time_prop["origin_timestamp"]
        assert timestamp.shape == (1, 10, 10)
        assert np.allclose(timestamp[0, 0, 0], 0)
        assert np.allclose(timestamp[0, 9, 9], 1.89e-05)

    def test_final_timestamp(self, radar_setup):
        """Test final timestamp generation with frame start time."""
        radar = radar_setup

        # Test single frame case (default frame_time=0)
        timestamp = radar.time_prop["timestamp"]
        origin_timestamp = radar.time_prop["origin_timestamp"]
        assert timestamp.shape == (1, 10, 10)
        # With frame_time=0, final timestamp should equal origin timestamp
        np.testing.assert_allclose(timestamp, origin_timestamp)

        # Test with non-zero frame start time
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        rx = Receiver(fs=10e6)
        radar_with_offset = Radar(transmitter=tx, receiver=rx, frame_time=1e-3)  # type: ignore

        timestamp_offset = radar_with_offset.time_prop["timestamp"]
        origin_timestamp_offset = radar_with_offset.time_prop["origin_timestamp"]
        frame_start_time = radar_with_offset.time_prop["frame_start_time"]

        # Final timestamp should be origin timestamp + frame start time
        expected_timestamp = origin_timestamp_offset + frame_start_time
        np.testing.assert_allclose(timestamp_offset, expected_timestamp)

        # Verify specific values
        assert np.allclose(
            timestamp_offset[0, 0, 0], 1e-3
        )  # First sample at frame start time
        assert np.allclose(timestamp_offset[0, 9, 9], 1e-3 + 1.89e-05)  # Last sample

        # Test multi-frame case
        radar_multi_frame = Radar(transmitter=tx, receiver=rx, frame_time=[0, 1e-3, 2e-3])  # type: ignore
        timestamp_multi = radar_multi_frame.time_prop["timestamp"]

        # Should have shape [3*channels, pulses, samples] for 3 frames
        assert timestamp_multi.shape == (3, 10, 10)

        # Verify frame timing
        assert np.allclose(timestamp_multi[0, 0, 0], 0)  # Frame 0 start
        assert np.allclose(timestamp_multi[1, 0, 0], 1e-3)  # Frame 1 start
        assert np.allclose(timestamp_multi[2, 0, 0], 2e-3)  # Frame 2 start

    def test_motion_validation_through_public_interface(self, radar_setup):
        """Test motion validation through public interface (constructor and set_motion)."""
        radar = radar_setup
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        rx = Receiver(fs=10e6)

        # Test invalid input lengths through constructor
        with pytest.raises(ValueError, match="location must have 3 elements"):
            Radar(transmitter=tx, receiver=rx, location=[0, 0])  # Wrong length

        with pytest.raises(ValueError, match="speed must have 3 elements"):
            Radar(transmitter=tx, receiver=rx, speed=[0, 0, 0, 0])  # Wrong length

        # Test invalid input lengths through set_motion
        with pytest.raises(ValueError, match="rotation must have 3 elements"):
            radar.set_motion(rotation=[0, 0])  # Wrong length

        with pytest.raises(ValueError, match="rotation_rate must have 3 elements"):
            radar.set_motion(rotation_rate=[0, 0, 0, 0])  # Wrong length

    def test_time_varying_motion_validation_through_public_interface(self, radar_setup):
        """Test time-varying motion validation through public interface."""
        # Create radar with multi-frame setup for time-varying motion
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        rx = Receiver(fs=10e6)
        radar = Radar(transmitter=tx, receiver=rx, frame_time=[0, 1e-3, 2e-3])

        # Create time-varying location array with correct shape
        timestamp_shape = radar.time_prop["timestamp_shape"]
        time_varying_location = np.ones(timestamp_shape)

        # Test that non-zero speed is rejected with time-varying location through constructor
        with pytest.raises(ValueError, match="speed must be \\[0, 0, 0\\]"):
            Radar(
                transmitter=tx, 
                receiver=rx, 
                frame_time=[0, 1e-3, 2e-3],
                location=[time_varying_location, 0, 0],  # type: ignore
                speed=[1, 0, 0],  # Non-zero speed should be rejected
            )

        # Test successful case: zero velocities with time-varying motion
        try:
            test_radar = Radar(
                transmitter=tx, 
                receiver=rx, 
                frame_time=[0, 1e-3, 2e-3],
                location=[time_varying_location, 0, 0],  # type: ignore
                speed=[0, 0, 0],  # Zero speed should be accepted
                rotation_rate=[0, 0, 0],  # Zero rotation_rate should be accepted
            )
            # If we get here, the validation passed
            assert test_radar is not None
        except ValueError:
            pytest.fail("Zero velocities should be accepted with time-varying motion")

    def test_validate_radar_motion_edge_cases(self, radar_setup):
        """Test edge cases that are difficult to trigger through public interface."""
        radar = radar_setup

        # These tests are kept for edge cases that are hard to test through public interface
        # Test invalid array shape (complex nested structures)
        with pytest.raises(ValueError):
            radar._validate_radar_motion(
                location=[0, 0, [1, 2]],  # Invalid nested structure
                speed=[0, 0, 0],
                rotation=[0, 0, 0],
                rotation_rate=[0, 0, 0],
            )

        # Test complex array shape validation
        with pytest.raises(ValueError, match="speed\\[x\\] must be a scalar"):
            wrong_shape_array = np.ones((2, 3, 4))  # Wrong shape
            radar._validate_radar_motion(
                location=[0, 0, 0],
                speed=[wrong_shape_array, 0, 0],
                rotation=[0, 0, 0],
                rotation_rate=[0, 0, 0],
            )

    def test_process_radar_motion_scalar(self, radar_setup):
        """Test processing of radar motion with scalar inputs through public interface."""
        radar = radar_setup
        radar.set_motion(
            location=[1, 2, 3],
            speed=[4, 5, 6],
            rotation=[7, 8, 9],
            rotation_rate=[10, 11, 12],
        )
        assert np.allclose(radar.radar_prop["location"], [1, 2, 3])
        assert np.allclose(radar.radar_prop["speed"], [4, 5, 6])
        assert np.allclose(radar.radar_prop["rotation"], np.radians([7, 8, 9]))
        assert np.allclose(radar.radar_prop["rotation_rate"], np.radians([10, 11, 12]))

    def test_cal_phase_noise(self):
        """Test phase noise calculation."""
        fs = 10e6
        freq = np.array([1e3, 1e4, 1e5])
        power = np.array([-100, -110, -120])
        signal = np.ones((10, 100), dtype=complex)
        phase_noise = cal_phase_noise(signal, fs, freq, power)
        assert phase_noise.shape == (10, 100)
        assert np.allclose(np.abs(phase_noise), 1)

    def test_cal_phase_noise_validation(self):
        """Test phase noise calculation with validation."""
        fs = 10e6
        freq = np.array([1e3, 1e4, 1e5])
        power = np.array([-100, -110, -120])
        signal = np.ones((10, 100), dtype=complex)
        phase_noise = cal_phase_noise(signal, fs, freq, power, validation=True)
        assert phase_noise.shape == (10, 100)
        assert np.allclose(np.abs(phase_noise), 1)

    def test_cal_phase_noise_input_validation(self):
        """Test phase noise calculation input validation."""
        signal = np.ones((10, 100), dtype=complex)

        # Test negative sampling frequency
        with pytest.raises(ValueError, match="Sampling frequency must be positive"):
            cal_phase_noise(signal, -10e6, np.array([1e3]), np.array([-100]))

        # Test mismatched freq and power arrays
        with pytest.raises(
            ValueError, match="freq and power arrays must have same length"
        ):
            cal_phase_noise(signal, 10e6, np.array([1e3, 1e4]), np.array([-100]))

        # Test negative frequency values
        with pytest.raises(
            ValueError, match="All frequency values must be non-negative"
        ):
            cal_phase_noise(signal, 10e6, np.array([-1e3, 1e4]), np.array([-100, -110]))

    def test_helper_functions(self):
        """Test the extracted helper functions."""
        # Test _interpolate_phase_noise_power
        freq = np.array([1e3, 1e4, 1e5])
        power = np.array([-100, -110, -120])
        f_grid = np.linspace(0, 5e6, 100)
        realmin = 1e-30

        p_interp = _interpolate_phase_noise_power(freq, power, f_grid, realmin)
        assert len(p_interp) == len(f_grid)
        assert np.all(p_interp >= 0)  # Power should be non-negative

        # Test _generate_noise_spectrum
        rng = np.random.default_rng(42)
        delta_f = np.ones(100) * 1e3
        shape = (5, 100)

        spec_noise = _generate_noise_spectrum(
            p_interp, delta_f, shape, rng, validation=False
        )
        assert spec_noise.shape == shape
        assert np.iscomplexobj(spec_noise)

        # Test with validation=True
        spec_noise_val = _generate_noise_spectrum(
            p_interp, delta_f, shape, rng, validation=True
        )
        assert spec_noise_val.shape == shape
        assert np.iscomplexobj(spec_noise_val)

    def test_error_message_formatting(self, radar_setup):
        """Test that error messages are properly formatted and informative."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        rx = Receiver(fs=10e6)

        # Test that speed arrays are not supported (only scalars allowed) through constructor
        with pytest.raises(ValueError) as exc_info:
            wrong_shape_array = np.ones((2, 3, 4))
            Radar(
                transmitter=tx,
                receiver=rx,
                speed=[wrong_shape_array, 0, 0],  # type: ignore
            )

        error_msg = str(exc_info.value)
        assert "speed[x] must be a scalar" in error_msg
        assert "Time-varying speed arrays are not supported" in error_msg
        assert "Use time-varying location arrays instead" in error_msg

    def test_constants(self):
        """Test that constants are properly defined and have expected values."""
        # Test Boltzmann constant
        assert BOLTZMANN_CONSTANT == 1.38064852e-23

        # Test sqrt(0.5) constant
        assert abs(SQRT_HALF - (0.5**0.5)) < 1e-15

        # Test milliwatts to watts conversion
        assert MILLIWATTS_TO_WATTS == 1e-3

    def test_radar_with_complex_configuration(self):
        """Test radar initialization with a more complex configuration."""
        # Create transmitter with multiple channels and phase noise
        tx = Transmitter(
            f=24e9,
            t=2e-6,
            tx_power=20,
            pulses=128,
            prp=1e-5,
            pn_f=np.array([1e3, 10e3, 100e3, 1e6]),
            pn_power=np.array([-90, -100, -110, -120]),
            channels=[
                {"location": (0, 0, 0)},
                {"location": (0.5, 0, 0)},
                {"location": (1.0, 0, 0)},
            ],
        )

        # Create receiver with multiple channels
        rx = Receiver(
            fs=50e6,
            noise_figure=8,
            rf_gain=30,
            channels=[
                {"location": (0, 0, 0)},
                {"location": (0, 0.5, 0)},
            ],
        )

        # Create radar with motion
        radar = Radar(
            transmitter=tx,
            receiver=rx,
            location=(100, 200, 10),
            speed=(50, 0, 0),
            rotation=(45, 0, 0),
            rotation_rate=(2, 0, 0),
            seed=42,
        )

        # Test properties
        assert radar.num_channels == 6  # 3 tx * 2 rx
        assert radar.samples_per_pulse == 100  # 2e-6 * 50e6
        assert radar.virtual_array_locations.shape == (6, 3)

        # Test that phase noise was generated
        assert radar.sample_prop["phase_noise"] is not None
        assert isinstance(radar.sample_prop["phase_noise"], np.ndarray)

        # Test radar properties
        np.testing.assert_allclose(radar.radar_prop["location"], [100, 200, 10])
        np.testing.assert_allclose(radar.radar_prop["speed"], [50, 0, 0])

        # Test string representation
        str_repr = str(radar)
        assert "channels=6" in str_repr
        assert "samples_per_pulse=100" in str_repr

    def test_edge_cases_and_boundary_conditions(self, radar_setup):
        """Test edge cases and boundary conditions."""
        radar = radar_setup

        # Test with zero motion values through public interface
        radar.set_motion(
            location=[0.0, 0.0, 0.0],
            speed=[0.0, 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0],
            rotation_rate=[0.0, 0.0, 0.0],
        )

        # Test with very small values through public interface
        radar.set_motion(
            location=[1e-10, 1e-10, 1e-10],
            speed=[1e-15, 1e-15, 1e-15],
            rotation=[1e-6, 1e-6, 1e-6],
            rotation_rate=[1e-9, 1e-9, 1e-9],
        )

        # Verify all values are properly converted to numpy arrays
        assert isinstance(radar.radar_prop["location"], np.ndarray)
        assert isinstance(radar.radar_prop["speed"], np.ndarray)
        assert isinstance(radar.radar_prop["rotation"], np.ndarray)
        assert isinstance(radar.radar_prop["rotation_rate"], np.ndarray)

    def test_phase_noise_without_parameters(self):
        """Test radar initialization when phase noise parameters are None."""
        # Create transmitter without phase noise
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        rx = Receiver(fs=10e6)
        radar = Radar(transmitter=tx, receiver=rx)

        # Phase noise should be None
        assert radar.sample_prop["phase_noise"] is None

    def test_init_with_phase_noise(self):
        """Test initialization with phase noise."""
        tx = Transmitter(
            f=10e9,
            t=1e-6,
            tx_power=10,
            pulses=10,
            prp=2e-6,
            pn_f=np.array([1e3, 1e4, 1e5]),
            pn_power=np.array([-100, -110, -120]),
        )
        rx = Receiver(fs=10e6)
        radar = Radar(transmitter=tx, receiver=rx)
        assert radar.sample_prop["phase_noise"] is not None
        assert isinstance(radar.sample_prop["phase_noise"], np.ndarray)
        assert radar.sample_prop["phase_noise"].shape == (191,)

    def test_init_with_phase_noise_validation(self):
        """Test initialization with phase noise and validation."""
        tx = Transmitter(
            f=10e9,
            t=1e-6,
            tx_power=10,
            pulses=10,
            prp=2e-6,
            pn_f=np.array([1e3, 1e4, 1e5]),
            pn_power=np.array([-100, -110, -120]),
        )
        rx = Receiver(fs=10e6)
        radar = Radar(transmitter=tx, receiver=rx, validation=True)
        assert radar.sample_prop["phase_noise"] is not None
        assert isinstance(radar.sample_prop["phase_noise"], np.ndarray)
        assert radar.sample_prop["phase_noise"].shape == (191,)

    def test_init_with_multiple_channels(self):
        """Test initialization with multiple channels."""
        tx = Transmitter(
            f=10e9,
            t=1e-6,
            tx_power=10,
            pulses=10,
            prp=2e-6,
            channels=[{"location": (0, 0, 0)}, {"location": (10, 0, 0)}],
        )
        rx = Receiver(
            fs=10e6,
            channels=[{"location": (0, 0, 0)}, {"location": (0, 10, 0)}],
        )
        radar = Radar(transmitter=tx, receiver=rx, time=[0, 1e-3])
        assert radar.array_prop["size"] == 4
        np.testing.assert_allclose(
            radar.array_prop["virtual_array"],
            [[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]],
        )

    def test_string_representations(self, radar_setup):
        """Test string representations of Radar object."""
        radar = radar_setup

        # Test __str__
        str_repr = str(radar)
        assert "Radar(" in str_repr
        assert "channels=1" in str_repr
        assert "samples_per_pulse=10" in str_repr
        assert "MHz" in str_repr

        # Test __repr__
        repr_str = repr(radar)
        assert "Radar(" in repr_str
        assert "channels=1" in repr_str
        assert "samples_per_pulse=10" in repr_str

    def test_property_accessors(self, radar_setup):
        """Test property accessor methods."""
        radar = radar_setup

        # Test num_channels property
        assert radar.num_channels == 1
        assert radar.num_channels == radar.array_prop["size"]

        # Test samples_per_pulse property
        assert radar.samples_per_pulse == 10
        assert radar.samples_per_pulse == radar.sample_prop["samples_per_pulse"]

        # Test transmitter and receiver properties
        assert radar.transmitter is radar.radar_prop["transmitter"]
        assert radar.receiver is radar.radar_prop["receiver"]

        # Test virtual_array_locations property
        np.testing.assert_allclose(
            radar.virtual_array_locations, radar.array_prop["virtual_array"]
        )

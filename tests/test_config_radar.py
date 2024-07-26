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
from radarsimpy.radar import cal_phase_noise


class TestRadar:
    @pytest.fixture
    def radar_setup(self):
        """Fixture for setting up a basic radar system."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        rx = Receiver(fs=10e6)
        radar = Radar(transmitter=tx, receiver=rx, time=[0, 1e-3])
        return radar

    def test_init_basic(self, radar_setup):
        """Test initialization with basic parameters."""
        radar = radar_setup
        assert radar.time_prop["frame_size"] == 2
        assert np.allclose(radar.time_prop["frame_start_time"], [0, 1e-3])
        assert radar.sample_prop["samples_per_pulse"] == 10
        assert radar.array_prop["size"] == 1
        np.testing.assert_allclose(radar.array_prop["virtual_array"], [[0, 0, 0]])
        assert radar.radar_prop["transmitter"].waveform_prop["f"][0] == 10e9
        assert radar.radar_prop["receiver"].bb_prop["fs"] == 10e6
        assert np.allclose(radar.radar_prop["location"], [0, 0, 0])
        assert np.allclose(radar.radar_prop["speed"], [0, 0, 0])
        assert np.allclose(radar.radar_prop["rotation"], [0, 0, 0])
        assert np.allclose(radar.radar_prop["rotation_rate"], [0, 0, 0])

    def test_gen_timestamp(self, radar_setup):
        """Test timestamp generation."""
        radar = radar_setup
        timestamp = radar.gen_timestamp()
        assert timestamp.shape == (2, 10, 10)
        assert np.allclose(timestamp[0, 0, 0], 0)
        assert np.allclose(timestamp[0, 9, 9], 1.89e-05)

    def test_gen_timestamp_multiple_frames(self):
        """Test timestamp generation with multiple frames."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        rx = Receiver(fs=10e6)
        radar = Radar(transmitter=tx, receiver=rx, time=[0, 1e-3, 2e-3])
        timestamp = radar.gen_timestamp()
        assert timestamp.shape == (3, 10, 10)
        assert np.allclose(timestamp[0, 0, 0], 0)
        assert np.allclose(timestamp[1, 0, 0], 1e-3)
        assert np.allclose(timestamp[2, 9, 9], 0.0020189)

    def test_cal_noise(self, radar_setup):
        """Test noise calculation."""
        radar = radar_setup
        noise = radar.cal_noise()
        assert isinstance(noise, float)

    def test_validate_radar_motion(self, radar_setup):
        """Test validation of radar motion inputs."""
        radar = radar_setup
        with pytest.raises(ValueError):
            radar.validate_radar_motion(
                location=[0, 0, [1, 2]],
                speed=[0, 0, 0],
                rotation=[0, 0, 0],
                rotation_rate=[0, 0, 0],
            )

    def test_process_radar_motion_scalar(self, radar_setup):
        """Test processing of radar motion with scalar inputs."""
        radar = radar_setup
        radar.process_radar_motion(
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
        radar = Radar(transmitter=tx, receiver=rx, time=[0, 1e-3])
        assert radar.sample_prop["phase_noise"].shape == (10191, )

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
        radar = Radar(transmitter=tx, receiver=rx, time=[0, 1e-3], validation=True)
        assert radar.sample_prop["phase_noise"].shape == (10191, )

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

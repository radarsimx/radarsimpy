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

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.radar import cal_phase_noise
from .test_config_transmitter import cw_tx, fmcw_tx, tdm_fmcw_tx, pmcw_tx
from .test_config_receiver import cw_rx, fmcw_rx, tdm_fmcw_rx, pmcw_rx


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
        assert radar.radar_prop["interf"] is None
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
        assert radar.sample_prop["phase_noise"].shape == (2, 10, 10)

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
        assert radar.sample_prop["phase_noise"].shape == (2, 10, 10)

    def test_init_with_interference(self):
        """Test initialization with interference radar."""
        tx = Transmitter(f=10e9, t=1e-6, tx_power=10, pulses=10, prp=2e-6)
        rx = Receiver(fs=10e6)
        interf_tx = Transmitter(f=11e9, t=1e-6, tx_power=5, pulses=10, prp=2e-6)
        interf_rx = Receiver(fs=10e6)
        interf_radar = Radar(transmitter=interf_tx, receiver=interf_rx, time=[0, 1e-3])
        radar = Radar(transmitter=tx, receiver=rx, time=[0, 1e-3], interf=interf_radar)
        assert (
            radar.radar_prop["interf"].radar_prop["transmitter"].waveform_prop["f"][0]
            == 11e9
        )

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


def cw_radar():
    """
    Creates a continuous wave (CW) radar system.

    :return: A Radar object with a transmitter and receiver.
    :rtype: Radar
    """
    return Radar(transmitter=cw_tx(), receiver=cw_rx())


def test_cw_radar():
    """
    Test the CW radar
    """
    cw = cw_radar()

    assert cw.sample_prop["samples_per_pulse"] == 10 * 20
    assert cw.array_prop["size"] == 1
    assert np.array_equal(cw.array_prop["virtual_array"], np.array([[0, 0, 0]]))


def fmcw_radar():
    """
    Creates an FMCW radar system.

    :return: A Radar object with a transmitter and receiver.
    :rtype: Radar
    """
    return Radar(transmitter=fmcw_tx(), receiver=fmcw_rx(), time=[0, 1])


def test_fmcw_radar():
    """
    Test the FMCW radar
    """
    fmcw = fmcw_radar()

    assert fmcw.sample_prop["samples_per_pulse"] == 80e-6 * 2e6
    assert fmcw.array_prop["size"] == 1
    assert np.array_equal(fmcw.array_prop["virtual_array"], np.array([[0, 0, 0]]))


def tdm_fmcw_radar():
    """
    Creates a TDM-FMCW radar system.

    :return: A Radar object with a transmitter and receiver.
    :rtype: Radar
    """
    return Radar(transmitter=tdm_fmcw_tx(), receiver=tdm_fmcw_rx())


def test_tdm_fmcw_radar():
    """
    Test the TDM FMCW radar
    """
    half_wavelength = const.c / 24.125e9 / 2
    tdm = tdm_fmcw_radar()

    assert tdm.sample_prop["samples_per_pulse"] == 80e-6 * 2e6
    assert tdm.array_prop["size"] == 16
    npt.assert_almost_equal(
        tdm.array_prop["virtual_array"],
        np.array(
            [
                [0, -8 * half_wavelength, 0],
                [0, -7 * half_wavelength, 0],
                [0, -6 * half_wavelength, 0],
                [0, -5 * half_wavelength, 0],
                [0, -4 * half_wavelength, 0],
                [0, -3 * half_wavelength, 0],
                [0, -2 * half_wavelength, 0],
                [0, -1 * half_wavelength, 0],
                [0, 0 * half_wavelength, 0],
                [0, 1 * half_wavelength, 0],
                [0, 2 * half_wavelength, 0],
                [0, 3 * half_wavelength, 0],
                [0, 4 * half_wavelength, 0],
                [0, 5 * half_wavelength, 0],
                [0, 6 * half_wavelength, 0],
                [0, 7 * half_wavelength, 0],
            ]
        ),
    )


def pmcw_radar():
    """
    Creates a PMCW radar system.

    :return: A Radar object with a transmitter and receiver.
    :rtype: Radar
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
    return Radar(transmitter=pmcw_tx(code1, code2), receiver=pmcw_rx())


def test_pmcw_radar():
    """
    Test the PMCW radar
    """
    pmcw = pmcw_radar()

    assert pmcw.sample_prop["samples_per_pulse"] == 2.1e-6 * 250e6
    assert pmcw.array_prop["size"] == 2
    assert np.array_equal(
        pmcw.array_prop["virtual_array"], np.array([[0, 0, 0], [0, 0, 0]])
    )

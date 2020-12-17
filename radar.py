#!python
# cython: language_level=3

# This script contains classes that define all the parameters for
# a radar system

# This script requires that 'numpy' be installed within the Python
# environment you are running this script in.

# This file can be imported as a module and contains the following
# class:

# * Transmitter - A class defines parameters of a radar transmitter
# * Receiver - A class defines parameters of a radar receiver
# * Radar - A class defines basic parameters of a radar system

# ----------
# RadarSimPy - A Radar Simulator Built with Python
# Copyright (C) 2018 - 2020  Zhengyu Peng
# E-mail: zpeng.me@gmail.com
# Website: https://zpeng.me

# `                      `
# -:.                  -#:
# -//:.              -###:
# -////:.          -#####:
# -/:.://:.      -###++##:
# ..   `://:-  -###+. :##:
#        `:/+####+.   :##:
# .::::::::/+###.     :##:
# .////-----+##:    `:###:
#  `-//:.   :##:  `:###/.
#    `-//:. :##:`:###/.
#      `-//:+######/.
#        `-/+####/.
#          `+##+.
#           :##:
#           :##:
#           :##:
#           :##:
#           :##:
#            .+:


import numpy as np
import scipy.constants as const
from scipy.interpolate import interp1d


class Transmitter:
    """
    A class defines basic parameters of a radar transmitter

    :param fc:
        Center frequency for each pulse (Hz).
        If ``fc`` is a single number, all the pulses have
        the same center frequency.
        ``fc`` can alse be a 1-D array to specify different
        center frequency for each pulse. In this case, the
        length of the 1-D array should equals to the length
        of ``pulses``
    :type fc: float or numpy.1darray
    :param float pulse_length:
        Dwell time of each pulse (seconds)
    :param float bandwidth:
        Bandwith of each pulse (Hz)
    :param float tx_power:
        Transmitter power (dBm)
    :param float repetition_period:
        Pulse repetition period (s). ``repetition_period >=
        pulse_length``. If it is ``None``, ``repetition_period =
        pulse_length``
    :param int pulses:
        Total number of pulses
    :param str slop_type:
        ``rising`` or ``falling`` slop for the frequency modulation
    :param list[dict] channels:
        Properties of transmitter channels

        [
            {

            - **location** (*numpy.1darray*) -
                3D location of the channel <x. y. z> (m)
            - **delay** (*float*) -
                Transmit delay (s)
            - **azimuth_angle** (*numpy.1darray*) -
                Angles for azimuth pattern (deg)
            - **azimuth_pattern** (*numpy.1darray*) -
                Azimuth pattern (dB)
            - **elevation_angle** (*numpy.1darray*) -
                Angles for elevation pattern (deg)
            - **elevation_pattern** (*numpy.1darray*) -
                Elevation pattern (dB)
            - **phase_code** (*numpy.1darray*) -
                Phase code sequence for phase modulation (deg).
                If ``chip_length == 0``, or ``chip_length`` is not defined,
                length of ``phase_code`` should be equal to ``pulses``
            - **chip_length** (*float*) -
                Length for each phase code (s). If ``chip_length ==
                0``, one pulse will have one ``phase_code``. If
                ``chip_length != 0``, all ``phase_code`` will be
                applied to each pulse

            }

        ]

    :ivar int channel_size:
        Total number of transmitter channels
    :ivar numpy.2darray locations:
        3D location of the channel <x. y. z> (m)
    :ivar numpy.1darray az_angles:
        Angles for azimuth patterns (deg)
    :type az_angles: list[numpy.1darray]
    :ivar numpy.1darray az_patterns:
        Azimuth pattern (dB)
    :type az_patterns: list[numpy.1darray]
    :ivar numpy.1darray el_angles:
        Angles for elevation pattern (deg)
    :type el_angles: list[numpy.1darray]
    :ivar el_patterns:
        Elevation pattern (dB)
    :type el_patterns: list[numpy.1darray]
    :ivar phase_code:
        Phase code sequence for phase modulation (deg).
        If ``chip_length == 0``, or ``chip_length`` is not
        defined, length of ``phase_code`` should be equal
        to ``pulses``
    :type phase_code: list[numpy.1darray]
    :ivar numpy.1darray chip_length:
        Length for each phase code (s). If ``chip_length ==
        0``, one pulse will have one ``phase_code``. If
        ``chip_length != 0``, all ``phase_code`` will be
        applied to each pulse
    :ivar numpy.1darray delay:
        Transmit delay (s)
    :ivar float wavelength:
        Wavelength (m)
    :ivar float slope:
        Waveform slope, ``bandwidth / pulse_length``

    **Waveform**

    .. code-block:: python

        #                repetition_period
        #                  +-----------+
        #
        #                          X            X            X          +
        #                         X            X            X           |
        #                        X            X            X            |
        #                       X            X            X             |
        #          +---fc--->  X            X            X     ...   bandwidth
        #                     X            X            X               |
        #                    X            X            X                |
        #                   X            X            X                 |
        #                  X            X            X                  +
        #
        #                  +-------+
        #                 pulse_length
        #
        #                  +-------------------------------------+
        # chip_length == 0 |  phase 0  ||  phase 1  ||  phase 2  |  ...
        #                  +-------------------------------------+
        #
        #                  +-------------------------------------+
        # chip_length != 0 | phase 0:N || phase 0:N || phase 0:N |  ...
        #                  +-------------------------------------+

    Tips:

    - Set ``bandwidth`` to 0 get a tone waveform for a Doppler radar

    """

    def __init__(self, fc,
                 pulse_length,
                 bandwidth=0,
                 tx_power=0,
                 repetition_period=None,
                 pulses=1,
                 slop_type='rising',
                 channels=[dict(location=(0, 0, 0))]):
        # self.fc = np.array(fc)
        if isinstance(fc, (list, tuple, np.ndarray)):
            if len(fc) != pulses:
                raise ValueError('Length of `fc` should be equal to `pulses`.')
            else:
                self.fc = np.array(fc)
        else:
            self.fc = fc+np.zeros(pulses)

        self.pulse_length = pulse_length
        self.bandwidth = bandwidth
        self.tx_power = tx_power
        if repetition_period is None:
            self.repetition_period = self.pulse_length + np.zeros(pulses)
        else:
            if isinstance(repetition_period, (list, tuple, np.ndarray)):
                self.repetition_period = repetition_period
            else:
                self.repetition_period = repetition_period + np.zeros(pulses)

        self.chirp_start_time = np.cumsum(
            self.repetition_period)-self.repetition_period[0]
        self.pulses = pulses

        # self.az_fov = np.array(az_fov)
        # self.el_fov = np.array(el_fov)
        # self.grid = grid
        self.max_code_length = 0

        self.channels = channels
        self.channel_size = len(self.channels)
        self.locations = np.zeros((self.channel_size, 3))
        self.az_patterns = []
        self.az_angles = []
        self.el_patterns = []
        self.el_angles = []
        self.az_func = []
        self.el_func = []
        self.phase_code = []
        self.chip_length = []
        self.polarization = np.zeros((self.channel_size, 3))
        self.antenna_gains = np.zeros((self.channel_size))
        self.grid = []
        self.delay = np.zeros(self.channel_size)
        for tx_idx, tx_element in enumerate(self.channels):
            self.delay[tx_idx] = self.channels[tx_idx].get('delay', 0)
            self.locations[tx_idx, :] = np.array(
                tx_element.get('location'))
            self.polarization[tx_idx, :] = np.array(
                tx_element.get('polarization', np.array([0, 0, 1])))
            self.az_angles.append(
                np.array(self.channels[tx_idx].get('azimuth_angle',
                                                   np.arange(-90, 91, 1))))
            self.az_patterns.append(
                np.array(self.channels[tx_idx].get('azimuth_pattern',
                                                   np.zeros(181))))

            self.antenna_gains[tx_idx] = np.max(self.az_patterns[-1])

            self.az_patterns[-1] = self.az_patterns[-1] - \
                np.max(self.az_patterns[-1])
            self.az_func.append(
                interp1d(self.az_angles[-1], self.az_patterns[-1],
                         kind='linear')
            )
            self.el_angles.append(
                np.array(self.channels[tx_idx].get('elevation_angle',
                                                   np.arange(-90, 91, 1))))
            self.el_patterns.append(
                np.array(self.channels[tx_idx].get('elevation_pattern',
                                                   np.zeros(181))))
            self.el_patterns[-1] = self.el_patterns[-1] - \
                np.max(self.el_patterns[-1])
            self.el_func.append(
                interp1d(
                    self.el_angles[-1],
                    self.el_patterns[-1]-np.max(self.el_patterns[-1]),
                    kind='linear')
            )
            self.phase_code.append(
                np.exp(1j * self.channels[tx_idx].get(
                    'phase_code', np.zeros((self.pulses))) / 180 * np.pi))

            self.max_code_length = max(
                self.max_code_length, np.shape(self.phase_code[-1])[0])

            self.chip_length.append(self.channels[tx_idx].get(
                'chip_length', 0))

            self.grid.append(self.channels[tx_idx].get('grid', 0.5))

            if self.chip_length[tx_idx] == 0:
                self.modulation = 'frame'
            else:
                self.modulation = 'pulse'

        # additional transmitter parameters
        self.wavelength = const.c / self.fc[0]

        if slop_type == 'rising':
            self.slope = (self.bandwidth / self.pulse_length)
        else:
            self.slope = (-self.bandwidth / self.pulse_length)

        self.box_min = np.min(self.locations, axis=0)
        self.box_max = np.max(self.locations, axis=0)


class Receiver:
    """
    A class defines basic parameters of a radar receiver

    :param float fs:
        Sampling rate (sps)
    :param float noise_figure:
        Noise figure (dB)
    :param float rf_gain:
        RF gain (dB)
    :param float load_resistor:
        Load resistor to convert power to voltage (Ohm)
    :param float baseband_gain:
        Baseband gain (dB)
    :param int noise_bandwidth:
        Noise bandwith in calculating thermal noise, ``fs / 2``
    :param list[dict] channels:
        Properties of transmitter channels

        [
            {

            - **location** (*numpy.1darray*) -
                3D location of the channel <x. y. z> (m)
            - **azimuth_angle** (*numpy.1darray*) -
                Angles for azimuth pattern (deg)
            - **azimuth_pattern** (*numpy.1darray*) -
                Azimuth pattern (dB)
            - **elevation_angle** (*numpy.1darray*) -
                Angles for elevation pattern (deg)
            - **elevation_pattern** (*numpy.1darray*) -
                Elevation pattern (dB)

            }

        ]

    :ivar int channel_size:
        Total number of transmitter channels
    :ivar numpy.2darray locations:
        3D location of the channel <x. y. z> (m)
    :ivar list[numpy.1darray] az_angles:
        Angles for azimuth patterns (deg)
    :ivar list[numpy.1darray] az_patterns:
        Azimuth pattern (dB)
    :ivar list[numpy.1darray] el_angles:
        Angles for elevation pattern (deg)
    :ivar list[numpy.1darray] el_patterns:
        Elevation pattern (dB)
    """

    def __init__(self, fs,
                 noise_figure=10,
                 rf_gain=0,
                 load_resistor=500,
                 baseband_gain=0,
                 channels=[dict(location=(0, 0, 0))]):
        self.fs = fs
        self.noise_figure = noise_figure
        self.rf_gain = rf_gain
        self.load_resistor = load_resistor
        self.baseband_gain = baseband_gain
        self.noise_bandwidth = self.fs / 2

        # additional receiver parameters

        self.channels = channels
        self.channel_size = len(self.channels)
        self.locations = np.zeros((self.channel_size, 3))
        self.az_patterns = []
        self.az_angles = []
        self.az_func = []
        self.el_patterns = []
        self.el_angles = []
        self.antenna_gains = np.zeros((self.channel_size))
        self.el_func = []
        for rx_idx, rx_element in enumerate(self.channels):
            self.locations[rx_idx, :] = np.array(
                rx_element.get('location'))
            self.az_angles.append(
                np.array(self.channels[rx_idx].get('azimuth_angle',
                                                   np.arange(-90, 91, 1))))
            self.az_patterns.append(
                np.array(self.channels[rx_idx].get('azimuth_pattern',
                                                   np.zeros(181))))
            self.antenna_gains[rx_idx] = np.max(self.az_patterns[-1])
            self.az_patterns[-1] = self.az_patterns[-1] - \
                np.max(self.az_patterns[-1])
            self.az_func.append(
                interp1d(self.az_angles[-1], self.az_patterns[-1],
                         kind='linear')
            )
            self.el_angles.append(
                np.array(self.channels[rx_idx].get('elevation_angle',
                                                   np.arange(-90, 91, 1))))
            self.el_patterns.append(
                np.array(self.channels[rx_idx].get('elevation_pattern',
                                                   np.zeros(181))))
            self.el_patterns[-1] = self.el_patterns[-1] - \
                np.max(self.el_patterns[-1])
            self.el_func.append(
                interp1d(
                    self.el_angles[-1],
                    self.el_patterns[-1]-np.max(self.el_patterns[-1]),
                    kind='linear')
            )

        self.box_min = np.min(self.locations, axis=0)
        self.box_max = np.max(self.locations, axis=0)


class Radar:
    """
    A class defines basic parameters of a radar system

    :param Transmitter transmitter:
        Radar transmiter
    :param Receiver receiver:
        Radar Receiver
    :ivar int samples_per_pulse:
        Number of samples in one pulse
    :ivar int channel_size:
        Total number of channels
    :ivar numpy.2darray virtual_array:
        Locations of virtual array elements. [channel_size, 3 <x, y, z>]
    """

    def __init__(self,
                 transmitter,
                 receiver,
                 time=0,
                 type=None,
                 aperture=None):

        self.transmitter = transmitter
        self.receiver = receiver

        self.samples_per_pulse = int(self.transmitter.pulse_length *
                                     self.receiver.fs)

        self.t_offset = np.array(time)
        self.frames = np.size(time)
        # system parameters for FMCW radar
        if type == 'FMCW':
            self.max_range = (const.c * self.receiver.fs *
                              self.transmitter.pulse_length /
                              self.transmitter.bandwidth / 2)
            self.unambiguous_speed = const.c / \
                self.transmitter.repetition_period[0] / \
                self.transmitter.fc[0] / 2
            self.range_resolution = const.c / 2 / self.transmitter.bandwidth

        # virtual array
        self.channel_size = self.transmitter.channel_size * \
            self.receiver.channel_size
        self.virtual_array = np.repeat(
            self.transmitter.locations, self.receiver.channel_size,
            axis=0) + np.tile(self.receiver.locations,
                              (self.transmitter.channel_size, 1))

        self.box_min = np.min(
            [self.transmitter.box_min, self.receiver.box_min], axis=0)
        self.box_max = np.max(
            [self.transmitter.box_min, self.receiver.box_max], axis=0)

        if aperture is None:
            self.aperture_mesh = np.zeros((2, 3, 3))
            self.aperture_mesh[0, 0, 0] = self.box_max[0]
            self.aperture_mesh[0, 0, 1] = self.box_max[1]
            self.aperture_mesh[0, 0, 2] = self.box_max[2]

            self.aperture_mesh[0, 1, 0] = self.box_max[0]
            self.aperture_mesh[0, 1, 1] = self.box_max[1]
            self.aperture_mesh[0, 1, 2] = self.box_min[2]

            self.aperture_mesh[0, 2, 0] = self.box_min[0]
            self.aperture_mesh[0, 2, 1] = self.box_min[1]
            self.aperture_mesh[0, 2, 2] = self.box_max[2]

            self.aperture_mesh[1, 0, 0] = self.box_min[0]
            self.aperture_mesh[1, 0, 1] = self.box_min[1]
            self.aperture_mesh[1, 0, 2] = self.box_min[2]

            self.aperture_mesh[1, 1, 0] = self.box_max[0]
            self.aperture_mesh[1, 1, 1] = self.box_max[1]
            self.aperture_mesh[1, 1, 2] = self.box_min[2]

            self.aperture_mesh[1, 2, 0] = self.box_min[0]
            self.aperture_mesh[1, 2, 1] = self.box_min[1]
            self.aperture_mesh[1, 2, 2] = self.box_max[2]
        else:
            self.aperture_mesh = None
            self.aperture_phi = aperture.get('phi')
            self.aperture_theta = aperture.get('theta')
            self.aperture_location = np.array(aperture.get('location'))
            self.aperture_extension = np.array(aperture.get('extension'))

        self.timestamp = self.gen_timestamp()
        self.phase_code = self.cal_frame_phases()
        self.code_timestamp = self.cal_code_timestamp()
        self.noise = self.cal_noise()

        # if hasattr(self.transmitter.fc, '__len__'):
        self.fc_mat = np.tile(
            self.transmitter.fc[np.newaxis, :, np.newaxis],
            (self.channel_size, 1, self.samples_per_pulse)
        )
        # else:
        #     self.fc_mat = np.full_like(self.timestamp, self.transmitter.fc)

        beat_time_samples = np.arange(-self.samples_per_pulse / 2,
                                      self.samples_per_pulse / 2,
                                      1) / self.receiver.fs
        self.beat_time = np.tile(
            beat_time_samples[np.newaxis, np.newaxis, ...],
            (self.channel_size, self.transmitter.pulses, 1)
        )

    def gen_timestamp(self):
        """
        Generate timestamp

        Parameters
        ----------
        radar : Radar (radarsimpy.Radar)
            Refer to 'radar' parameter in 'run_simulator'

        Returns
        -------
        3D array
            A 3D array of timestamp, '[channels, pulses, adc_samples]'
        """

        channel_size = self.channel_size
        rx_channel_size = self.receiver.channel_size
        pulses = self.transmitter.pulses
        samples = self.samples_per_pulse
        crp = self.transmitter.repetition_period
        delay = self.transmitter.delay
        fs = self.receiver.fs

        chirp_delay = np.tile(
            np.expand_dims(
                np.expand_dims(np.cumsum(crp)-crp[0], axis=1),
                axis=0),
            (channel_size, 1, samples))

        tx_idx = np.arange(0, channel_size)/rx_channel_size
        tx_delay = np.tile(
            np.expand_dims(
                np.expand_dims(delay[tx_idx.astype(int)], axis=1),
                axis=2),
            (1, pulses, samples))

        timestamp = tx_delay+chirp_delay+np.tile(
            np.expand_dims(
                np.expand_dims(np.arange(0, samples), axis=0),
                axis=0),
            (channel_size, pulses, 1))/fs

        if self.frames > 1:
            toffset = np.repeat(
                np.tile(
                    np.expand_dims(
                        np.expand_dims(self.t_offset, axis=1), axis=2), (
                        1, self.transmitter.pulses, self.samples_per_pulse
                    )), self.channel_size, axis=0)

            timestamp = np.tile(timestamp, (self.frames, 1, 1)) + toffset
        elif self.frames == 1:
            timestamp = timestamp + self.t_offset

        return timestamp

    def cal_frame_phases(self):
        """
        Calculate phase sequence for frame level modulation

        Parameters
        ----------
        radar : Radar (radarsimpy.Radar)
            Refer to 'radar' parameter in 'run_simulator'
        targets : list of dict
            Refer to 'targets' parameter in 'run_simulator'

        Returns
        -------
        3D array
            Phase sequence
        """

        phase_code = np.array(self.transmitter.phase_code, dtype=complex)
        phase_code = np.repeat(phase_code, self.receiver.channel_size, axis=0)
        phase_code = np.repeat(phase_code, self.frames, axis=0)
        return phase_code

    def cal_code_timestamp(self):
        """
        Calculate phase sequence for pulse level modulation

        Parameters
        ----------
        radar : Radar (radarsimpy.Radar)
            Refer to 'radar' parameter in 'run_simulator'
        targets : list of dict
            Refer to 'targets' parameter in 'run_simulator'

        Returns
        -------
        dict
            {
                code_timestamp : list of 1D array
                    [
                        1D array
                            Timestamp for the edge of each code
                    ]
                phase_code : list of 1D array
                    [
                        1D array
                            Phase sequence
                    ]
            }
        """

        chip_length = np.expand_dims(
            np.array(self.transmitter.chip_length),
            axis=1)
        code_sequence = chip_length*np.tile(
            np.expand_dims(
                np.arange(0, self.transmitter.max_code_length),
                axis=0),
            (self.transmitter.channel_size, 1))

        code_timestamp = np.repeat(
            code_sequence, self.receiver.channel_size, axis=0)

        code_timestamp = np.repeat(
            code_timestamp, self.frames, axis=0)

        return code_timestamp

    def cal_noise(self):
        """
        Calculate noise amplitudes

        Parameters
        ----------
        radar : Radar (radarsimpy.Radar)
            Refer to 'radar' parameter in 'run_simulator'
        targets : list of dict
            Refer to 'targets' parameter in 'run_simulator'

        Returns
        -------
        3D array
            Noise amplitudes, '[channels, pulses, adc_samples]'
        """

        noise_amp = np.zeros([
            self.channel_size,
            self.transmitter.pulses,
            self.samples_per_pulse,
        ])

        Boltzmann_const = 1.38064852e-23
        Ts = 290
        input_noise_dbm = 10 * np.log10(Boltzmann_const * Ts * 1000)  # dBm/Hz
        receiver_noise_dbm = (input_noise_dbm + self.receiver.rf_gain +
                              self.receiver.noise_figure +
                              10 * np.log10(self.receiver.noise_bandwidth) +
                              self.receiver.baseband_gain)  # dBm/Hz
        receiver_noise_watts = 1e-3 * 10**(receiver_noise_dbm / 10
                                           )  # Watts/sqrt(hz)
        noise_amplitude_mixer = np.sqrt(receiver_noise_watts *
                                        self.receiver.load_resistor)
        noise_amplitude_peak = np.sqrt(2) * noise_amplitude_mixer + noise_amp
        return noise_amplitude_peak

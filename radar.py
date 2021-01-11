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
        Dwell time of each pulse (s)
    :param float bandwidth:
        Bandwith of each pulse (Hz)
    :param float tx_power:
        Transmitter power (dBm)
    :param float repetition_period:
        Pulse repetition period (s). ``repetition_period >=
        pulse_length``. If it is ``None``, ``repetition_period =
        pulse_length``.

        ``repetition_period`` can alse be a 1-D array to specify
        different repetition period for each pulse. In this case, the
        length of the 1-D array should equals to the length
        of ``pulses``
    :type repetitions_period: float or numpy.1darray
    :param int pulses:
        Total number of pulses
    :param str slop_type:
        ``rising`` or ``falling`` slope for the frequency modulation
    :param list[dict] channels:
        Properties of transmitter channels

        [{

        - **location** (*numpy.1darray*) --
            3D location of the channel [x. y. z] (m)
        - **delay** (*float*) --
            Transmit delay (s). ``default 0``
        - **azimuth_angle** (*numpy.1darray*) --
            Angles for azimuth pattern (deg). ``default [-90, 90]``
        - **azimuth_pattern** (*numpy.1darray*) --
            Azimuth pattern (dB). ``default [0, 0]``
        - **elevation_angle** (*numpy.1darray*) --
            Angles for elevation pattern (deg). ``default [-90, 90]``
        - **elevation_pattern** (*numpy.1darray*) --
            Elevation pattern (dB). ``default [0, 0]``
        - **phase_code** (*numpy.1darray*) --
            Phase code sequence for phase modulation (deg).
            If ``chip_length == 0``, or ``chip_length`` is not defined,
            length of ``phase_code`` should be equal to ``pulses``.
            ``default 0``
        - **chip_length** (*float*) --
            Length for each phase code (s). If ``chip_length ==
            0``, one pulse will have one ``phase_code``. If
            ``chip_length != 0``, all ``phase_code`` will be
            applied to each pulse. ``default 0``

        }]

    :ivar numpy.1darray fc:
        Center frequency array for the pulses
    :ivar float pulse_length:
        Dwell time of each pulse (s)
    :ivar float bandwidth:
        Bandwith of each pulse (Hz)
    :ivar float tx_power:
        Transmitter power (dBm)
    :ivar numpy.1darray repetition_period:
        Pulse repetition period (s)
    :ivar int pulses:
        Total number of pulses
    :ivar list[dict] channels:
        Properties of transmitter channels
    :ivar int channel_size:
        Number of transmitter channels
    :ivar numpy.2darray locations:
        3D location of the channels. Size of the aray is
        ``[channel_size, 3 <x, y, z>]`` (m)
    :ivar list[numpy.1darray] az_angles:
        Angles for each channel's azimuth pattern (deg)
    :ivar list[numpy.1darray] az_patterns:
        Azimuth pattern for each channel (dB)
    :ivar list[numpy.1darray] el_angles:
        Angles for each channel's elevation pattern (deg)
    :ivar list[numpy.1darray] el_patterns:
        Elevation pattern for each channel (dB)
    :ivar list az_func:
        Azimuth patterns' interpolation functions
    :ivar list el_func:
        Elevation patterns' interpolation functions
    :ivar numpy.1darray antenna_gains:
        Antenna gain for each channel (dB).
        Antenna gain is ``max(az_pattern)``
    :ivar list[numpy.1darray] phase_code:
        Phase code sequence for phase modulation (deg)
    :ivar numpy.1darray chip_length:
        Length for each phase code (s)
    :ivar numpy.1darray delay:
        Delay for each channel (s)
    :ivar numpy.1darray polarization:
        Antenna polarization ``[x, y, z]``.

        - Horizontal polarization: ``[1, 0, 0]``
        - Vertical polarization: ``[0, 0, 1]``

    :ivar float wavelength:
        Wavelength (m)
    :ivar float slope:
        Waveform slope, ``bandwidth / pulse_length``

    **Waveform**

    ::

        |                repetition_period
        |                  +-----------+
        |
        |                          /            /            /          +
        |                         /            /            /           |
        |                        /            /            /            |
        |                       /            /            /             |
        |          +---fc--->  /            /            /     ...   bandwidth
        |                     /            /            /               |
        |                    /            /            /                |
        |                   /            /            /                 |
        |                  /            /            /                  +
        |
        |                  +-------+
        |                 pulse_length
        |
        |                  +-------------------------------------+
        | chip_length == 0 |  phase 0  ||  phase 1  ||  phase 2  |  ...
        |                  +-------------------------------------+
        |
        |                  +-------------------------------------+
        | chip_length != 0 | phase 0:N || phase 0:N || phase 0:N |  ...
        |                  +-------------------------------------+

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

        self.pulse_length = pulse_length
        self.bandwidth = bandwidth
        self.tx_power = tx_power
        self.pulses = pulses
        self.channels = channels

        # Extend `fc` to a numpy.1darray. Length equels to `pulses`
        if isinstance(fc, (list, tuple, np.ndarray)):
            if len(fc) != pulses:
                raise ValueError(
                    'Length of `fc` should equal to the length of `pulses`.')
            else:
                self.fc = np.array(fc)
        else:
            self.fc = fc+np.zeros(pulses)

        # Extend `repetition_period` to a numpy.1darray.
        # Length equels to `pulses`
        if repetition_period is None:
            self.repetition_period = self.pulse_length + np.zeros(pulses)
        else:
            if isinstance(repetition_period, (list, tuple, np.ndarray)):
                if len(repetition_period) != pulses:
                    raise ValueError(
                        'Length of `repetition_period` should equal to the \
                            length of `pulses`.')
                else:
                    self.repetition_period = repetition_period
            else:
                self.repetition_period = repetition_period + np.zeros(pulses)

        if np.min(self.repetition_period < self.pulse_length):
            raise ValueError(
                '`repetition_period` should be larger than `pulse_length`.')

        self.chirp_start_time = np.cumsum(
            self.repetition_period)-self.repetition_period[0]

        self.max_code_length = 0

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
        Total RF gain (dB)
    :param float load_resistor:
        Load resistor to convert power to voltage (Ohm)
    :param float baseband_gain:
        Total baseband gain (dB)
    :param list[dict] channels:
        Properties of transmitter channels

        [{

        - **location** (*numpy.1darray*) --
            3D location of the channel [x. y. z] (m)
        - **azimuth_angle** (*numpy.1darray*) --
            Angles for azimuth pattern (deg). ``default [-90, 90]``
        - **azimuth_pattern** (*numpy.1darray*) --
            Azimuth pattern (dB). ``default [0, 0]``
        - **elevation_angle** (*numpy.1darray*) --
            Angles for elevation pattern (deg). ``default [-90, 90]``
        - **elevation_pattern** (*numpy.1darray*) --
            Elevation pattern (dB). ``default [0, 0]``

        }]

    :ivar float fs:
        Sampling rate (sps)
    :ivar float noise_figure:
        Noise figure (dB)
    :ivar float rf_gain:
        Total RF gain (dB)
    :ivar float load_resistor:
        Load resistor to convert power to voltage (Ohm)
    :ivar float baseband_gain:
        Total baseband gain (dB)
    :ivar float noise_bandwidth:
        Bandwidth in calculating the noise (Hz).
        ``noise_bandwidth = fs / 2``
    :ivar list[dict] channels:
        Properties of receiver channels
    :ivar int channel_size:
        Total number of receiver channels
    :ivar numpy.2darray locations:
        3D location of the channels. Size of the aray is
        ``[channel_size, 3 <x, y, z>]`` (m)
    :ivar list[numpy.1darray] az_angles:
        Angles for each channel's azimuth pattern (deg)
    :ivar list[numpy.1darray] az_patterns:
        Azimuth pattern for each channel (dB)
    :ivar list[numpy.1darray] el_angles:
        Angles for each channel's elevation pattern (deg)
    :ivar list[numpy.1darray] el_patterns:
        Elevation pattern for each channel (dB)
    :ivar list az_func:
        Azimuth patterns' interpolation functions
    :ivar list el_func:
        Elevation patterns' interpolation functions
    :ivar numpy.1darray antenna_gains:
        Antenna gain for each channel (dB).
        Antenna gain is ``max(az_pattern)``

    **Receiver noise**

    ::

        |           + n1 = 10*log10(Boltzmann_constant * Ts * 1000)
        |           |      + 10*log10(noise_bandwidth)  (dBm)
        |           v
        |    +------+------+
        |    |rf_gain      |
        |    +------+------+
        |           | n2 = n1 + noise_figure + rf_gain (dBm)
        |           v n3 = 1e-3 * 10^(n2/10) (Watts)
        |    +------+------+
        |    |mixer        |
        |    +------+------+
        |           | n4 = sqrt(n3 * load_resistor) (V)
        |           v
        |    +------+------+
        |    |baseband_gain|
        |    +------+------+
        |           | noise amplitude (peak to peak)
        |           v n5 = n4 * 10^(baseband_gain / 20) * sqrt(2) (V)

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
    :param time:
        Radar firing time instances / frames
    :type time: float or numpy.1darray
    :param dict aperture:
        Radar receiver aperture for ray tracing simulation

        {

        - **phi** (*float*) --
            phi angle of the aperture's normal (deg)
        - **theta** (*float*) --
            theta angle of the aperture's normal (deg)
        - **location** (*numpy.1darray*) --
            aperture's center location ``[x, y, z]`` (m)
        - **extension** (*numpy.1darray*) --
            aperture's extension of ``[left, right, top, bottom]``
            when facing towards its normal (m)

        }

    :ivar Transmitter transmitter:
        Radar transmiter
    :ivar Receiver receiver:
        Radar Receiver
    :ivar int samples_per_pulse:
        Number of samples in one pulse
    :ivar int channel_size:
        Total number of channels.
        ``channel_size = transmitter.channel_size * receiver.channel_size``
    :ivar numpy.2darray virtual_array:
        Locations of virtual array elements. [channel_size, 3 <x, y, z>]
    :ivar float max_range:
        Maximum range for an FMCW mode (m).
        ``max_range = c * fs * pulse_length / bandwidth / 2``
    :ivar float unambiguous_speed:
        Unambiguous speed (m/s).
        ``unambiguous_speed = c / repetition_period / fc / 2``
    :ivar float range_resolution:
        Range resolution (m).
        ``range_resolution = c / 2 / bandwidth``
    :ivar float aperture_phi:
        phi angle of the aperture's normal (deg)
    :ivar float aperture_theta:
        theta angle of the aperture's normal (deg)
    :ivar numpy.1darray aperture_location:
        Aperture's center location ``[x, y, z]`` (m)
    :ivar numpy.1darray aperture_extension:
        Aperture's extension of ``[left, right, top, bottom]``
        when facing towards its normal (m)
    :ivar numpy.3darray timestamp:
        Timestamp for each samples. Frame start time is
        defined in ``time``.
        ``[channes/frames, pulses, samples]``

        *Channel/frame order in timestamp*

        *[0]* ``Frame[0] -- Tx[0] -- Rx[0]``

        *[1]* ``Frame[0] -- Tx[0] -- Rx[1]``

        ...

        *[N]* ``Frame[0] -- Tx[1] -- Rx[0]``

        *[N+1]* ``Frame[0] -- Tx[1] -- Rx[1]``

        ...

        *[M]* ``Frame[1] -- Tx[0] -- Rx[0]``

        *[M+1]* ``Frame[1] -- Tx[0] -- Rx[1]``

    """

    def __init__(self,
                 transmitter,
                 receiver,
                 time=0,
                 aperture=None):

        self.transmitter = transmitter
        self.receiver = receiver

        self.samples_per_pulse = int(self.transmitter.pulse_length *
                                     self.receiver.fs)

        self.t_offset = np.array(time)
        self.frames = np.size(time)

        if self.transmitter.bandwidth > 0:
            self.max_range = (const.c * self.receiver.fs *
                              self.transmitter.pulse_length /
                              self.transmitter.bandwidth / 2)
            self.unambiguous_speed = const.c / \
                self.transmitter.repetition_period[0] / \
                self.transmitter.fc[0] / 2
            self.range_resolution = const.c / 2 / self.transmitter.bandwidth
        else:
            self.max_range = 0
            self.unambiguous_speed = 0
            self.range_resolution = 0

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

        :return:
            Timestamp for each samples. Frame start time is
            defined in ``time``.
            ``[channes/frames, pulses, samples]``
        :rtype: numpy.3darray
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

        :return:
            Phase sequence. ``[channes/frames, pulses, samples]``
        :rtype: numpy.2darray
        """

        phase_code = np.array(self.transmitter.phase_code, dtype=complex)
        phase_code = np.repeat(phase_code, self.receiver.channel_size, axis=0)
        phase_code = np.repeat(phase_code, self.frames, axis=0)
        return phase_code

    def cal_code_timestamp(self):
        """
        Calculate phase code timing for pulse level modulation

        :return:
            Timing at the start position of each phase code.
            ``[channes/frames, max_code_length]``
        :rtype: numpy.2darray
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

        :return:
            Peak to peak amplitude of noise.
            ``[channes/frames, pulses, samples]``
        :rtype: numpy.3darray
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

    def phase_noise(Sin, Fs, phase_noise_freq, phase_noise_power):
        #
        # Oscillator Phase Noise Model
        #
        #  INPUT:
        #     Sin - input COMPLEX signal, mxn matrix, column vectors are signals.
        #     Fs  - sampling frequency ( in Hz ) of Sin
        #     phase_noise_freq  - frequencies at which SSB Phase Noise is defined (offset from carrier in Hz)
        #     phase_noise_power - SSB Phase Noise power ( in dBc/Hz )
        #     VALIDATION_ON  - 1 - perform validation, 0 - don't perfrom validation
        #
        #  OUTPUT:
        #     Sout - output COMPLEX phase noised signal
        #
        #  NOTE:
        #     Input signal should be complex
        #
        #  EXAMPLE ( How to use add_phase_noise ):
        #         Assume SSB Phase Noise is specified as follows:
        #      -------------------------------------------------------
        #      |  Offset From Carrier      |        Phase Noise      |
        #      -------------------------------------------------------
        #      |        1   kHz            |        -84  dBc/Hz      |
        #      |        10  kHz            |        -100 dBc/Hz      |
        #      |        100 kHz            |        -96  dBc/Hz      |
        #      |        1   MHz            |        -109 dBc/Hz      |
        #      |        10  MHz            |        -122 dBc/Hz      |
        #      -------------------------------------------------------
        #
        #      Assume that we have 10000 samples of complex sinusoid of frequency 3 KHz
        #      sampled at frequency 40MHz:
        #
        #       Fc = 3e3; carrier frequency
        #       Fs = 40e6; sampling frequency
        #       t = 0:9999;
        #       S = exp(j*2*pi*Fc/Fs*t); complex sinusoid
        #
        #      Then, to produce phase noised signal S1 from the original signal S run follows:
        #
        #       Fs = 40e6;
        #       phase_noise_freq = [ 1e3, 10e3, 100e3, 1e6, 10e6 ]; Offset From Carrier
        #       phase_noise_power = [ -84, -100, -96, -109, -122 ]; Phase Noise power
        #       S1 = add_phase_noise( S, Fs, phase_noise_freq, phase_noise_power );

        # Version 1.0
        # Alex Bur-Guy, October 2005
        # alex@wavion.co.il
        #
        # Revisions:
        #       Version 1.5 -   Comments. Validation.
        #       Version 1.0 -   initial version

        # NOTES:
        # 1)  The presented model is a simple VCO phase noise model based on the following consideration:
        # If the output of an oscillator is given as  V(t) = V0 * cos( w0*t + phi(t) ),
        # then phi(t)  is defined as the phase noise.  In cases of small noise
        # sources (a valid assumption in any usable system), a narrowband modulation approximation can
        # be used to express the oscillator output as:
        #
        # V(t) = V0 * cos( w0*t + phi(t) )
        #
        #        = V0 * [cos(w0*t)*cos(phi(t)) - sin(w0*t)*sin(phi(t)) ]
        #
        #        ~ V0 * [cos(w0*t) - sin(w0*t)*phi(t)]
        #
        # This shows that phase noise will be mixed with the carrier to produce sidebands around the carrier.
        #
        #
        # 2) In other words, exp(j*x) ~ (1+j*x) for small x
        #
        # 3) Phase noise = 0 dBc/Hz at freq. offset of 0 Hz
        #
        # 4) The lowest phase noise level is defined by the input SSB phase noise power at the maximal
        #    freq. offset from DC. (IT DOES NOT BECOME EQUAL TO ZERO )
        #
        # The generation process is as follows:
        #  First of all we interpolate (in log-scale) SSB phase noise power spectrum in M
        #  equally spaced points (on the interval [0 Fs/2] including bounds ).
        #
        #  After that we calculate required frequency shape of the phase noise by X(m) = sqrt(P(m)*dF(m))
        #  and after that complement it by the symmetrical negative part of the spectrum.
        #
        #  After that we generate AWGN of power 1 in the freq domain and multiply it sample-by-sample to
        #  the calculated shape
        #
        #  Finally we perform  2*M-2 points IFFT to such generated noise
        #  ( See comments inside the code )
        #
        #  0 dBc/Hz
        #  \                                                          /
        #   \                                                        /
        #    \                                                      /
        #     \P dBc/Hz                                            /
        #     .\                                                  /
        #     . \                                                /
        #     .  \                                              /
        #     .   \____________________________________________/  /_ This level is defined by the phase_noise_power at the maximal freq. offset from DC defined in phase_noise_freq
        #     .                                                   \
        #  |__| _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__   (N points)
        #  0   dF                       Fs/2                          Fs
        #  DC
        #
        #
        #  For some basics about Oscillator phase noise see:
        #     http://www.circuitsage.com/pll/plldynamics.pdf
        #
        #     http://www.wj.com/pdf/technotes/LO_phase_noise.pdf

        # Sin = 1j * np.ones((128, 256))
        # Fs = 2e6
        # phase_noise_freq = np.array([1, 10, 100, 1000, 10000])*1000
        # phase_noise_power = np.array([-84, -100, -96, -109, -122])

        # if max(phase_noise_freq) >= Fs/2
        #      error( 'Maximal frequency offset should be less than Fs/2');
        # end

        # Make sure phase_noise_freq and  phase_noise_power are the row vectors

        # if length( phase_noise_freq ) ~= length( phase_noise_power )
        #      error('phase_noise_freq and phase_noise_power should be of the same length');
        # end

        # Sort phase_noise_freq and phase_noise_power
        sort_idx = np.argsort(phase_noise_freq)
        phase_noise_freq = phase_noise_freq[sort_idx]
        phase_noise_power = phase_noise_power[sort_idx]

        cut_idx = np.where(phase_noise_freq < Fs/2)
        phase_noise_freq = phase_noise_freq[cut_idx]
        phase_noise_power = phase_noise_power[cut_idx]

        # Add 0 dBc/Hz @ DC
        if not np.any(np.isin(phase_noise_freq, 0)):
            phase_noise_freq = np.concatenate(([0], phase_noise_freq))
            phase_noise_power = np.concatenate(([0], phase_noise_power))

        # Calculate input length
        [numCol, N] = np.shape(Sin)
        # Define M number of points (frequency resolution) in the positive spectrum
        #  (M equally spaced points on the interval [0 Fs/2] including bounds),
        # then the number of points in the negative spectrum will be M-2
        #  ( interval (Fs/2, Fs) not including bounds )
        #
        # The total number of points in the frequency domain will be 2*M-2, and if we want
        #  to get the same length as the input signal, then
        #   2*M-2 = N
        #   M-1 = N/2
        #   M = N/2 + 1
        #
        #  So, if N is even then M = N/2 + 1, and if N is odd we will take  M = (N+1)/2 + 1
        #
        if np.remainder(N, 2):
            M = int((N+1)/2 + 1)
        else:
            M = int(N/2 + 1)

        # Equally spaced partitioning of the half spectrum
        F = np.linspace(0, Fs/2, int(M))    # Freq. Grid
        dF = np.concatenate((np.diff(F), [F[-1]-F[-2]]))  # Delta F

        realmin = np.finfo(float).tiny

        # Perform interpolation of phase_noise_power in log-scale
        intrvlNum = len(phase_noise_freq)
        logP = np.zeros(int(M))
        # for intrvlIndex = 1 : intrvlNum,
        for intrvlIndex in range(0, intrvlNum):
            leftBound = phase_noise_freq[intrvlIndex]
            t1 = phase_noise_power[intrvlIndex]
            if intrvlIndex == intrvlNum-1:
                rightBound = Fs/2
                t2 = phase_noise_power[-1]
                inside = np.where(np.logical_and(
                    F >= leftBound, F <= rightBound))
            else:
                rightBound = phase_noise_freq[intrvlIndex+1]
                t2 = phase_noise_power[intrvlIndex+1]
                inside = np.where(np.logical_and(
                    F >= leftBound, F < rightBound))

            logP[inside] = t1 + (np.log10(F[inside] + realmin) - np.log10(leftBound + realmin)) / (
                np.log10(rightBound + 2*realmin) - np.log10(leftBound + realmin)) * (t2-t1)

        # Interpolated P ( half spectrum [0 Fs/2] ) [ dBc/Hz ]
        P = 10**(np.real(logP)/10)

        # Now we will generate AWGN of power 1 in frequency domain and shape it by the desired shape
        # as follows:
        #
        #    At the frequency offset F(m) from DC we want to get power Ptag(m) such that P(m) = Ptag/dF(m),
        #     that is we have to choose X(m) =  sqrt( P(m)*dF(m) );
        #
        # Due to the normalization factors of FFT and IFFT defined as follows:
        #     For length K input vector x, the DFT is a length K vector X,
        #     with elements
        #                      K
        #        X(k) =       sum  x(n)*exp(-j*2*pi*(k-1)*(n-1)/K), 1 <= k <= K.
        #                     n=1
        #     The inverse DFT (computed by IFFT) is given by
        #                      K
        #        x(n) = (1/K) sum  X(k)*exp( j*2*pi*(k-1)*(n-1)/K), 1 <= n <= K.
        #                     k=1
        #
        # we have to compensate normalization factor (1/K) multiplying X(k) by K.
        # In our case K = 2*M-2.

        # Generate AWGN of power 1

        awgn_P1 = (np.sqrt(0.5)*(np.random.randn(numCol, M) +
                                 1j*np.random.randn(numCol, M)))
    #     awgn_P1 = (np.sqrt(0.5)*(np.ones((numCol, M)) +
    #                              1j*np.ones((numCol, M))))

        # Shape the noise on the positive spectrum [0, Fs/2] including bounds ( M points )
        X = (2*M-2) * np.sqrt(dF * P) * awgn_P1

        # X = np.transpose(X)
        # Complete symmetrical negative spectrum  (Fs/2, Fs) not including bounds (M-2 points)
        tmp_X = np.zeros((numCol, int(M*2-2)), dtype=complex)
        tmp_X[:, 0:M] = X
        tmp_X[:, M:(2*M-2)] = np.fliplr(np.conjugate(X[:, 1:-1]))

        X = tmp_X

        # X( M + (1:M-2),: ) = flipud( conj(X(2:end-1,:)) );

        # Remove DC
        X[:, 0] = 0

        # Perform IFFT
        x_t = np.fft.ifft(X, axis=1)

        # Calculate phase noise
        phase_noise = np.exp(1j * np.real(x_t[:, 0:N]))
    #     phase_noise = x[:, 0:N]
        # phase_noise = 1 + 1i*real(x(1:N));

        # Add phase noise
        Sout = Sin * phase_noise

        return Sout

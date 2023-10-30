"""
This script contains classes that define all the parameters for
a radar system

This script requires that 'numpy' be installed within the Python
environment you are running this script in.

This file can be imported as a module and contains the following
class:

* Radar - A class defines basic parameters of a radar system

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


import numpy as np

from .util import cal_phase_noise


class Radar:
    """
    A class defines basic parameters of a radar system

    :param Transmitter transmitter:
        Radar transmiter
    :param Receiver receiver:
        Radar Receiver
    :param list location:
        3D location of the radar [x, y, z] (m). ``default
        [0, 0, 0]``
    :param list speed:
        Speed of the radar (m/s), [vx, vy, vz]. ``default
        [0, 0, 0]``
    :param list rotation:
        Radar's angle (deg), [yaw, pitch, roll].
        ``default [0, 0, 0]``
    :param list rotation_rate:
        Radar's rotation rate (deg/s),
        [yaw rate, pitch rate, roll rate]
        ``default [0, 0, 0]``
    :param time:
        Radar firing time instances / frames
        :type time: float or list
    :param Radar interf:
        Interference radar. ``default None``
    :param int seed:
        Seed for noise generator

    :ivar dict time_prop: Time properties

        - **frame_size**: Number of frames

        - **frame_start_time**: Frame start time

        - **timestamp_shape**: Shape of timestamp

        - **timestamp**: Timestamp for each samples

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

    :ivar dict sample_prop: Sample properties

        - **samples_per_pulse**: Number of samples in one pulse

        - **noise**: Noise amplitude

        - **phase_noise**: Phase noise matrix

    :ivar dict array_prop: Array properties

        - **size**: Number of virtual array elements

        - **virtual_array**: Locations of virtual array elements. [channel_size, 3 <x, y, z>]

    :ivar dict radar_prop: Radar properties

        - **transmitter**: Radar transmitter

        - **receiver**: Radar receiver

        - **interf**: Interference radar

        - **location**: Radar location (m)

        - **speed**: Radar speed (m/s)

        - **rotation**: Radar rotation (rad)

        - **rotation_rate**: Radar rotation rate (rad/s)

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        transmitter,
        receiver,
        location=(0, 0, 0),
        speed=(0, 0, 0),
        rotation=(0, 0, 0),
        rotation_rate=(0, 0, 0),
        time=0,
        interf=None,
        seed=None,
        **kwargs
    ):
        self.time_prop = {
            "frame_size": np.size(time),
            "frame_start_time": np.array(time),
        }
        self.sample_prop = {
            "samples_per_pulse": int(
                transmitter.waveform_prop["pulse_length"] * receiver.bb_prop["fs"]
            )
        }
        self.array_prop = {
            "size": (
                transmitter.txchannel_prop["size"] * receiver.rxchannel_prop["size"]
            ),
            "virtual_array": np.repeat(
                transmitter.txchannel_prop["locations"],
                receiver.rxchannel_prop["size"],
                axis=0,
            )
            + np.tile(
                receiver.rxchannel_prop["locations"],
                (transmitter.txchannel_prop["size"], 1),
            ),
        }
        self.radar_prop = {
            "transmitter": transmitter,
            "receiver": receiver,
            "interf": interf,
        }

        # timing properties
        self.time_prop["timestamp"] = self.gen_timestamp()
        self.time_prop["timestamp_shape"] = np.shape(self.time_prop["timestamp"])

        # sample properties
        self.sample_prop["noise"] = self.cal_noise()

        if (
            transmitter.rf_prop["pn_f"] is not None
            and transmitter.rf_prop["pn_power"] is not None
        ):
            dummy_sig = np.ones(
                (
                    self.array_prop["size"]
                    * self.time_prop["frame_size"]
                    * transmitter.waveform_prop["pulses"],
                    self.sample_prop["samples_per_pulse"],
                )
            )
            self.sample_prop["phase_noise"] = cal_phase_noise(
                dummy_sig,
                receiver.bb_prop["fs"],
                transmitter.rf_prop["pn_f"],
                transmitter.rf_prop["pn_power"],
                seed=seed,
                validation=kwargs.get("validation", False),
            )
            self.sample_prop["phase_noise"] = np.reshape(
                self.sample_prop["phase_noise"],
                (
                    self.array_prop["size"] * self.time_prop["frame_size"],
                    transmitter.waveform_prop["pulses"],
                    self.sample_prop["samples_per_pulse"],
                ),
            )
        else:
            self.sample_prop["phase_noise"] = None

        self.process_radar_motion(
            location,
            speed,
            rotation,
            rotation_rate,
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

        channel_size = self.array_prop["size"]
        rx_channel_size = self.radar_prop["receiver"].rxchannel_prop["size"]
        pulses = self.radar_prop["transmitter"].waveform_prop["pulses"]
        samples = self.sample_prop["samples_per_pulse"]
        crp = self.radar_prop["transmitter"].waveform_prop["prp"]
        delay = self.radar_prop["transmitter"].txchannel_prop["delay"]
        fs = self.radar_prop["receiver"].bb_prop["fs"]

        chirp_delay = np.tile(
            np.expand_dims(np.expand_dims(np.cumsum(crp) - crp[0], axis=1), axis=0),
            (channel_size, 1, samples),
        )

        tx_idx = np.arange(0, channel_size) / rx_channel_size
        tx_delay = np.tile(
            np.expand_dims(np.expand_dims(delay[tx_idx.astype(int)], axis=1), axis=2),
            (1, pulses, samples),
        )

        timestamp = (
            tx_delay
            + chirp_delay
            + np.tile(
                np.expand_dims(np.expand_dims(np.arange(0, samples), axis=0), axis=0),
                (channel_size, pulses, 1),
            )
            / fs
        )

        if self.time_prop["frame_size"] > 1:
            toffset = np.repeat(
                np.tile(
                    np.expand_dims(
                        np.expand_dims(self.time_prop["frame_start_time"], axis=1),
                        axis=2,
                    ),
                    (
                        1,
                        self.radar_prop["transmitter"].waveform_prop["pulses"],
                        self.sample_prop["samples_per_pulse"],
                    ),
                ),
                channel_size,
                axis=0,
            )

            timestamp = (
                np.tile(timestamp, (self.time_prop["frame_size"], 1, 1)) + toffset
            )
        elif self.time_prop["frame_size"] == 1:
            timestamp = timestamp + self.time_prop["frame_start_time"]

        return timestamp

    def cal_noise(self, noise_temp=290):
        """
        Calculate noise amplitudes

        :return:
            Peak to peak amplitude of noise.
            ``[channes/frames, pulses, samples]``
        :rtype: numpy.3darray
        """

        noise_amp = np.zeros(
            [
                self.array_prop["size"],
                self.radar_prop["transmitter"].waveform_prop["pulses"],
                self.sample_prop["samples_per_pulse"],
            ]
        )

        boltzmann_const = 1.38064852e-23

        input_noise_dbm = 10 * np.log10(boltzmann_const * noise_temp * 1000)  # dBm/Hz
        receiver_noise_dbm = (
            input_noise_dbm
            + self.radar_prop["receiver"].rf_prop["rf_gain"]
            + self.radar_prop["receiver"].rf_prop["noise_figure"]
            + 10 * np.log10(self.radar_prop["receiver"].bb_prop["noise_bandwidth"])
            + self.radar_prop["receiver"].bb_prop["baseband_gain"]
        )  # dBm/Hz
        receiver_noise_watts = 1e-3 * 10 ** (receiver_noise_dbm / 10)  # Watts/sqrt(hz)
        noise_amplitude_mixer = np.sqrt(
            receiver_noise_watts * self.radar_prop["receiver"].bb_prop["load_resistor"]
        )
        noise_amplitude_peak = np.sqrt(2) * noise_amplitude_mixer + noise_amp
        return noise_amplitude_peak

    def validate_radar_motion(self, location, speed, rotation, rotation_rate):
        """
        Validate radar motion inputs

        :param list location: 3D location of the radar [x, y, z] (m)
        :param list speed: Speed of the radar (m/s), [vx, vy, vz]
        :param list rotation: Radar's angle (deg), [yaw, pitch, roll]
        :param list rotation_rate: Radar's rotation rate (deg/s),
        [yaw rate, pitch rate, roll rate]

        :raises ValueError: speed[x] must be a scalar or have the same shape as timestamp
        :raises ValueError: location[x] must be a scalar or have the same shape as timestamp
        :raises ValueError: rotation_rate[x] must be a scalar or have the same shape as timestamp
        :raises ValueError: rotation[x] must be a scalar or have the same shape as timestamp
        """

        for idx in range(0, 3):
            if np.size(speed[idx]) > 1:
                if np.shape(speed[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        "speed ["
                        + str(idx)
                        + "] must be a scalar or have the same shape as timestamp"
                    )

            if np.size(location[idx]) > 1:
                if np.shape(location[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        "location["
                        + str(idx)
                        + "] must be a scalar or have the same shape as timestamp"
                    )

            if np.size(rotation_rate[idx]) > 1:
                if np.shape(rotation_rate[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        "rotation_rate["
                        + str(idx)
                        + "] must be a scalar or have the same shape as timestamp"
                    )

            if np.size(rotation[idx]) > 1:
                if np.shape(rotation[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        "rotation["
                        + str(idx)
                        + "] must be a scalar or have the same shape as timestamp"
                    )

    def process_radar_motion(self, location, speed, rotation, rotation_rate):
        """
        Process radar motion parameters

        :param list location: 3D location of the radar [x, y, z] (m)
        :param list speed: Speed of the radar (m/s), [vx, vy, vz]
        :param list rotation: Radar's angle (deg), [yaw, pitch, roll]
        :param list rotation_rate: Radar's rotation rate (deg/s),
        [yaw rate, pitch rate, roll rate]

        """
        shape = self.time_prop["timestamp_shape"]

        if any(
            np.size(var) > 1
            for var in list(location)
            + list(speed)
            + list(rotation)
            + list(rotation_rate)
        ):
            self.validate_radar_motion(location, speed, rotation, rotation_rate)
            self.radar_prop["location"] = np.zeros(shape + (3,))
            self.radar_prop["speed"] = np.zeros(shape + (3,))
            self.radar_prop["rotation"] = np.zeros(shape + (3,))
            self.radar_prop["rotation_rate"] = np.zeros(shape + (3,))

            for idx in range(0, 3):
                if np.size(speed[idx]) > 1:
                    self.radar_prop["speed"][:, :, :, idx] = speed[idx]
                else:
                    self.radar_prop["speed"][:, :, :, idx] = np.full(shape, speed[idx])

                if np.size(location[idx]) > 1:
                    self.radar_prop["location"][:, :, :, idx] = location[idx]
                else:
                    self.radar_prop["location"][:, :, :, idx] = (
                        location[idx] + speed[idx] * self.time_prop["timestamp"]
                    )

                if np.size(rotation_rate[idx]) > 1:
                    self.radar_prop["rotation_rate"][:, :, :, idx] = np.radians(
                        rotation_rate[idx]
                    )

                else:
                    self.radar_prop["rotation_rate"][:, :, :, idx] = np.full(
                        shape, np.radians(rotation_rate[idx])
                    )

                if np.size(rotation[idx]) > 1:
                    self.radar_prop["rotation"][:, :, :, idx] = np.radians(
                        rotation[idx]
                    )
                else:
                    self.radar_prop["rotation"][:, :, :, idx] = (
                        np.radians(rotation[idx])
                        + np.radians(rotation_rate[idx]) * self.time_prop["timestamp"]
                    )

        else:
            self.radar_prop["speed"] = np.array(speed)
            self.radar_prop["location"] = np.array(location)
            self.radar_prop["rotation"] = np.radians(rotation)
            self.radar_prop["rotation_rate"] = np.radians(rotation_rate)

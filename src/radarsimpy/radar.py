"""
This script contains classes that define all the parameters for
a radar system

This script requires that 'numpy' be installed within the Python
environment you are running this script in.

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


def cal_phase_noise(  # pylint: disable=too-many-arguments, too-many-locals
    signal, fs, freq, power, seed=None, validation=False
):
    """
    Oscillator Phase Noise Model

    :param numpy.2darray signal:
        Input signal
    :param float fs:
        Sampling frequency
    :param numpy.1darray freq:
        Frequency of the phase noise
    :param numpy.1darray power:
        Power of the phase noise
    :param int seed:
        Seed for noise generator
    :param boolean validation:
        Validate phase noise

    :return:
        Signal with phase noise
    :rtype: numpy.2darray

    **NOTES**

    - The presented model is a simple VCO phase noise model based
    on the following consideration:
        If the output of an oscillator is given as
        V(t) = V0 * cos( w0*t + phi(t) ), then phi(t) is defined
        as the phase noise.  In cases of small noise sources (a valid
        assumption in any usable system), a narrowband modulatio
        approximation can be used to express the oscillator output as:

        V(t) = V0 * cos( w0*t + phi(t) )
            = V0 * [cos(w0*t)*cos(phi(t)) - signal(w0*t)*signal(phi(t)) ]
            ~ V0 * [cos(w0*t) - signal(w0*t)*phi(t)]

        This shows that phase noise will be mixed with the carrier
        to produce sidebands around the carrier.

    - In other words, exp(j*x) ~ (1+j*x) for small x

    - Phase noise = 0 dBc/Hz at freq. offset of 0 Hz

    - The lowest phase noise level is defined by the input SSB phase
    noise power at the maximal freq. offset from DC.
    (IT DOES NOT BECOME EQUAL TO ZERO )

    The generation process is as follows:

    First of all we interpolate (in log-scale) SSB phase noise power
    spectrum in num_f_points equally spaced points
    (on the interval [0 fs/2] including bounds ).

    After that we calculate required frequency shape of the phase
    noise by spec_noise(m) = sqrt(P(m)*delta_f(m)) and after that complement it
    by the symmetrical negative part of the spectrum.

    After that we generate AWGN of power 1 in the freq domain and
    multiply it sample-by-sample to the calculated shape

    Finally we perform  2*num_f_points-2 points IFFT to such generated noise

    ::

        █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █
        █ 0 dBc/Hz                                                        █
        █ \\                                                    /         █
        █  \\                                                  /          █
        █   \\                                                /           █
        █    \\P dBc/Hz                                      /            █
        █    .\\                                            /             █
        █    . \\                                          /              █
        █    .  \\                                        /               █
        █    .   \\______________________________________/ <- This level  █
        █    .              is defined by the power at the maximal freq   █
        █  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__  (N) █
        █  0   delta_f                    fs/2                       fs   █
        █  DC                                                             █
        █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █

    """

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    signal = signal.astype(complex)

    # Sort freq and power
    sort_idx = np.argsort(freq)
    freq = freq[sort_idx]
    power = power[sort_idx]

    cut_idx = np.where(freq < fs / 2)
    freq = freq[cut_idx]
    power = power[cut_idx]

    # Add 0 dBc/Hz @ DC
    if not np.any(np.isin(freq, 0)):
        freq = np.concatenate(([0], freq))
        power = np.concatenate(([0], power))

    # Calculate input length
    [row, num_samples] = np.shape(signal)
    # Define num_f_points number of points (frequency resolution) in the
    # positive spectrum (num_f_points equally spaced points on the interval
    # [0 fs/2] including bounds), then the number of points in the
    # negative spectrum will be num_f_points-2 ( interval (fs/2, fs) not
    # including bounds )
    #
    # The total number of points in the frequency domain will be
    # 2*num_f_points-2, and if we want to get the same length as the input
    # signal, then
    #   2*num_f_points-2 = num_samples
    #   num_f_points-1 = num_samples/2
    #   num_f_points = num_samples/2 + 1
    #
    # So, if num_samples is even then num_f_points = num_samples/2 + 1,
    # and if num_samples is odd we will take
    # num_f_points = (num_samples+1)/2 + 1
    #
    if np.remainder(num_samples, 2):
        num_f_points = int((num_samples + 1) / 2 + 1)
    else:
        num_f_points = int(num_samples / 2 + 1)

    # Equally spaced partitioning of the half spectrum
    f_grid = np.linspace(0, fs / 2, int(num_f_points))  # Freq. Grid
    delta_f = np.concatenate((np.diff(f_grid), [f_grid[-1] - f_grid[-2]]))  # Delta F

    realmin = np.finfo(np.float64).tiny
    # realmin = 1e-30

    # Perform interpolation of power in log-scale
    intrvl_num = len(freq)
    log_p = np.zeros(int(num_f_points))
    # for intrvl_index = 1 : intrvl_num,
    for intrvl_index in range(0, intrvl_num):
        left_bound = freq[intrvl_index]
        t1 = power[intrvl_index]
        if intrvl_index == intrvl_num - 1:
            right_bound = fs / 2
            t2 = power[-1]
            inside = np.where(
                np.logical_and(f_grid >= left_bound, f_grid <= right_bound)
            )
        else:
            right_bound = freq[intrvl_index + 1]
            t2 = power[intrvl_index + 1]
            inside = np.where(
                np.logical_and(f_grid >= left_bound, f_grid < right_bound)
            )

        log_p[inside] = t1 + (
            np.log10(f_grid[inside] + realmin) - np.log10(left_bound + realmin)
        ) / (np.log10(right_bound + 2 * realmin) - np.log10(left_bound + realmin)) * (
            t2 - t1
        )

    # Interpolated P ( half spectrum [0 fs/2] ) [ dBc/Hz ]
    p_interp = 10 ** (np.real(log_p) / 10)

    # Now we will generate AWGN of power 1 in frequency domain and shape
    # it by the desired shape as follows:
    #
    #    At the frequency offset f_grid(m) from DC we want to get power Ptag(m)
    #    such that p_interp(m) = Ptag/delta_f(m), that is we have to choose
    #    spec_noise(m) = sqrt( p_interp(m)*delta_f(m) );
    #
    # Due to the normalization factors of FFT and IFFT defined as follows:
    #     For length K input vector x, the DFT is a length K vector spec_noise,
    #     with elements
    #                K
    #      spec_noise(k) =   sum  x(n)*exp(-j*2*pi*(k-1)*(n-1)/K), 1 <= k <= K.
    #               n=1
    #     The inverse DFT (computed by IFFT) is given by
    #                      K
    #      x(n) = (1/K) sum  spec_noise(k)*exp( j*2*pi*(k-1)*(n-1)/K), 1 <= n <= K.
    #                     k=1
    #
    # we have to compensate normalization factor (1/K) multiplying spec_noise(k)
    # by K. In our case K = 2*num_f_points-2.

    # Generate AWGN of power 1
    if validation:
        awgn_p1 = np.sqrt(0.5) * (
            np.ones((row, num_f_points)) + 1j * np.ones((row, num_f_points))
        )
    else:
        awgn_p1 = np.sqrt(0.5) * (
            rng.standard_normal((row, num_f_points))
            + 1j * rng.standard_normal((row, num_f_points))
        )

    # Shape the noise on the positive spectrum [0, fs/2] including bounds
    # ( num_f_points points )
    # spec_noise = (2*num_f_points-2) * np.sqrt(delta_f * p_interp) * awgn_p1
    spec_noise = num_f_points * np.sqrt(delta_f * p_interp) * awgn_p1
    ## NOTE: this normalization should be num_f_points vs. (2*num_f_points-2)
    # since on line 222 he creates the two-sided spectrum by adding the negative frequency spectrum.

    # spec_noise = np.transpose(spec_noise)
    # Complete symmetrical negative spectrum  (fs/2, fs) not including
    # bounds (num_f_points-2 points)
    tmp_spec_noise = np.zeros((row, int(num_f_points * 2 - 2)), dtype=complex)
    tmp_spec_noise[:, 0:num_f_points] = spec_noise
    tmp_spec_noise[:, num_f_points : (2 * num_f_points - 2)] = np.fliplr(
        np.conjugate(spec_noise[:, 1:-1])
    )

    spec_noise = tmp_spec_noise

    # Remove DC
    spec_noise[:, 0] = 0

    # Perform IFFT
    x_t = np.fft.ifft(spec_noise, axis=1)

    # Calculate phase noise
    phase_noise = np.exp(-1j * np.real(x_t[:, 0:num_samples]))

    # Add phase noise
    return signal * phase_noise


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
            self.sample_prop["phase_noise"] = np.reshape(
                cal_phase_noise(
                    np.ones(
                        (
                            self.array_prop["size"]
                            * self.time_prop["frame_size"]
                            * transmitter.waveform_prop["pulses"],
                            self.sample_prop["samples_per_pulse"],
                        )
                    ),
                    receiver.bb_prop["fs"],
                    transmitter.rf_prop["pn_f"],
                    transmitter.rf_prop["pn_power"],
                    seed=seed,
                    validation=kwargs.get("validation", False),
                ),
                self.time_prop["timestamp_shape"],
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
        :rtype: float
        """

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
        noise_amplitude_peak = np.sqrt(2) * noise_amplitude_mixer
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

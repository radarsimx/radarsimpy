"""
Radar Configuration and Phase Noise Modeling in Python

This module contains classes and functions to define and simulate
the parameters and behavior of radar systems. It includes tools for
configuring radar system properties, modeling oscillator phase noise,
and simulating radar motion and noise characteristics. A major focus
of the module is on accurately modeling radar signal properties,
including phase noise and noise amplitudes.

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

from typing import List, Union, Tuple, Optional, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .transmitter import Transmitter
    from .receiver import Receiver

# Constants
BOLTZMANN_CONSTANT = 1.38064852e-23  # J/K
REALMIN_FALLBACK = 1e-30
SQRT_HALF = 0.5**0.5
MILLIWATTS_TO_WATTS = 1e-3


def _interpolate_phase_noise_power(
    freq: NDArray, power: NDArray, f_grid: NDArray, realmin: float
) -> NDArray:
    """
    Interpolate phase noise power in log-scale.

    :param freq: Frequency array
    :param power: Power array
    :param f_grid: Frequency grid for interpolation
    :param realmin: Minimum value to avoid log(0)
    :return: Interpolated power values
    """
    intrvl_num = len(freq)
    log_p = np.zeros(len(f_grid))

    for intrvl_index in range(intrvl_num):
        left_bound = freq[intrvl_index]
        t1 = power[intrvl_index]

        if intrvl_index == intrvl_num - 1:
            right_bound = f_grid[-1] * 2  # fs/2
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

    return 10 ** (np.real(log_p) / 10)


def _generate_noise_spectrum(
    p_interp: NDArray,
    delta_f: NDArray,
    shape: Tuple[int, int],
    rng,
    validation: bool = False,
) -> NDArray:
    """
    Generate noise spectrum with proper shape.

    :param p_interp: Interpolated power values
    :param delta_f: Frequency spacing array
    :param shape: Shape (row, num_f_points)
    :param rng: Random number generator
    :param validation: Use deterministic values for validation
    :return: Generated noise spectrum
    """
    row, num_f_points = shape

    # Generate AWGN of power 1
    if validation:
        awgn_p1 = SQRT_HALF * (
            np.ones((row, num_f_points)) + 1j * np.ones((row, num_f_points))
        )
    else:
        awgn_p1 = SQRT_HALF * (
            rng.standard_normal((row, num_f_points))
            + 1j * rng.standard_normal((row, num_f_points))
        )

    # Shape the noise on the positive spectrum [0, fs/2] including bounds
    spec_noise = num_f_points * np.sqrt(delta_f * p_interp) * awgn_p1

    return spec_noise


def cal_phase_noise(  # pylint: disable=too-many-arguments, too-many-locals
    signal: NDArray,
    fs: float,
    freq: NDArray,
    power: NDArray,
    seed: Optional[int] = None,
    validation: bool = False,
) -> NDArray:
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
    # Input validation
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if len(freq) != len(power):
        raise ValueError(
            f"freq and power arrays must have same length: "
            f"freq={len(freq)}, power={len(power)}"
        )
    if np.any(freq < 0):
        raise ValueError("All frequency values must be non-negative")

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

    # Perform interpolation of power in log-scale
    realmin = float(np.finfo(np.float64).tiny)
    # Alternative: realmin = REALMIN_FALLBACK
    p_interp = _interpolate_phase_noise_power(freq, power, f_grid, realmin)

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

    # Generate shaped noise spectrum
    spec_noise = _generate_noise_spectrum(
        p_interp, delta_f, (row, num_f_points), rng, validation
    )

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
    Defines the basic parameters and properties of a radar system.

    This class represents the overall configuration of a radar system,
    including its transmitter, receiver, spatial properties (location, speed, orientation),
    and associated metadata.

    :param Transmitter transmitter: The radar transmitter instance.
    :param Receiver receiver: The radar receiver instance.
    :param list location:
        The 3D location of the radar relative to a global coordinate system [x, y, z] in meters (m).
        Default: ``[0, 0, 0]``.
    :param list speed:
        The velocity of the radar in meters per second (m/s), specified as [vx, vy, vz].
        Default: ``[0, 0, 0]``.
    :param list rotation:
        The radar's orientation in degrees (°), specified as [yaw, pitch, roll].
        Default: ``[0, 0, 0]``.
    :param list rotation_rate:
        The radar's angular velocity in degrees per second (°/s),
        specified as [yaw rate, pitch rate, roll rate].
        Default: ``[0, 0, 0]``.
    :param int seed:
        Seed for the random noise generator to ensure reproducibility.

    :ivar dict time_prop:
        Time-related properties of the radar system:

        - **timestamp_shape** (*tuple*): The shape of the timestamp array.
        - **timestamp** (*numpy.ndarray*): The timestamp for each sample in a frame,
          structured as ``[channels, pulses, samples]``.
          Channel order in timestamp (with ``M`` Tx channels and ``N`` Rx channels):

            - [0, :, :] ``Tx[0] → Rx[0]``
            - [1, :, :] ``Tx[0] → Rx[1]``
            - ...
            - [N-1, :, :] ``Tx[0] → Rx[N-1]``
            - [N, :, :] ``Tx[1] → Rx[0]``
            - ...
            - [M·N-1, :, :] ``Tx[M-1] → Rx[N-1]``

    :ivar dict sample_prop:
        Sample-related properties:

        - **samples_per_pulse** (*int*): Number of samples in a single pulse.
        - **noise** (*float*): Noise amplitude.
        - **phase_noise** (*numpy.ndarray*): Phase noise matrix for pulse samples.

    :ivar dict array_prop:
        Metadata related to the radar's virtual array:

        - **size** (*int*): Total number of virtual array elements.
        - **virtual_array** (*numpy.ndarray*): 3D locations of each virtual array element,
          structured as ``[channel_size, 3]`` where each row corresponds to an [x, y, z] position.

    :ivar dict radar_prop:
        Radar system properties:

        - **transmitter** (*Transmitter*): Instance of the radar transmitter.
        - **receiver** (*Receiver*): Instance of the radar receiver.
        - **location** (*list*): The radar's 3D location in meters (m).
        - **speed** (*list*): The radar's velocity in meters per second (m/s).
        - **rotation** (*list*): The radar's orientation in radians (rad).
        - **rotation_rate** (*list*): Angular velocity of the radar in radians per second (rad/s).

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        transmitter: "Transmitter",
        receiver: "Receiver",
        location: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        speed: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        rotation: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        rotation_rate: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.time_prop: dict[str, Any] = {}

        # Calculate samples per pulse and validate
        samples_per_pulse = int(
            transmitter.waveform_prop["pulse_length"] * receiver.bb_prop["fs"]
        )
        if samples_per_pulse <= 0:
            pulse_length = transmitter.waveform_prop["pulse_length"]
            fs = receiver.bb_prop["fs"]
            product = pulse_length * fs
            raise ValueError(
                f"samples_per_pulse must be greater than 0, got {samples_per_pulse}. "
                f"This occurs when pulse_length ({pulse_length}) * fs ({fs}) = {product:.6f} < 1. "
                f"Either increase the pulse_length or increase the sampling frequency."
            )

        self.sample_prop: dict[str, Union[int, float, NDArray, None]] = {
            "samples_per_pulse": samples_per_pulse
        }
        self.array_prop: dict[str, Union[int, NDArray]] = {
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
        self.radar_prop: dict[str, Any] = {
            "transmitter": transmitter,
            "receiver": receiver,
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
            num_pn_samples = (
                np.ceil(
                    (
                        np.max(self.time_prop["timestamp"])
                        - np.min(self.time_prop["timestamp"])
                    )
                    * self.radar_prop["receiver"].bb_prop["fs"]
                ).astype(int)
                + 1
            )
            self.sample_prop["phase_noise"] = cal_phase_noise(
                np.ones(
                    (
                        1,
                        num_pn_samples,
                    )
                ),
                receiver.bb_prop["fs"],
                transmitter.rf_prop["pn_f"],
                transmitter.rf_prop["pn_power"],
                seed=seed,
                validation=kwargs.get("validation", False),
            )
            self.sample_prop["phase_noise"] = self.sample_prop["phase_noise"].flatten()
        else:
            self.sample_prop["phase_noise"] = None

        self.process_radar_motion(
            list(location),
            list(speed),
            list(rotation),
            list(rotation_rate),
        )

    def gen_timestamp(self) -> NDArray:
        """
        Generate timestamp

        :return:
            Timestamp for each samples. Frame start time is
            defined in ``time``.
            Shape: ``[channels/frames, pulses, samples]``
        :rtype: numpy.3darray
        """

        channel_size = int(self.array_prop["size"])
        rx_channel_size = int(self.radar_prop["receiver"].rxchannel_prop["size"])
        pulses = self.radar_prop["transmitter"].waveform_prop["pulses"]
        samples_per_pulse = self.sample_prop["samples_per_pulse"]
        assert isinstance(
            samples_per_pulse, int
        ), "samples_per_pulse must be an integer"
        samples = samples_per_pulse
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

        return timestamp

    def cal_noise(self, noise_temp: float = 290) -> float:
        """
        Calculate noise amplitudes

        :param float noise_temp: Noise temperature in Kelvin
        :return:
            Peak to peak amplitude of noise.
        :rtype: float
        """

        input_noise_dbm = 10 * np.log10(
            BOLTZMANN_CONSTANT * noise_temp * 1000
        )  # dBm/Hz
        receiver_noise_dbm = (
            input_noise_dbm
            + self.radar_prop["receiver"].rf_prop["rf_gain"]
            + self.radar_prop["receiver"].rf_prop["noise_figure"]
            + 10 * np.log10(self.radar_prop["receiver"].bb_prop["noise_bandwidth"])
            + self.radar_prop["receiver"].bb_prop["baseband_gain"]
        )  # dBm/Hz
        receiver_noise_watts = MILLIWATTS_TO_WATTS * 10 ** (
            receiver_noise_dbm / 10
        )  # Watts/sqrt(hz)
        noise_amplitude_mixer = np.sqrt(
            receiver_noise_watts * self.radar_prop["receiver"].bb_prop["load_resistor"]
        )
        # noise_amplitude_peak = np.sqrt(2) * noise_amplitude_mixer
        return noise_amplitude_mixer

    def validate_radar_motion(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
    ) -> None:
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
        # Validate input lengths
        if len(location) != 3:
            raise ValueError(f"location must have 3 elements, got {len(location)}")
        if len(speed) != 3:
            raise ValueError(f"speed must have 3 elements, got {len(speed)}")
        if len(rotation) != 3:
            raise ValueError(f"rotation must have 3 elements, got {len(rotation)}")
        if len(rotation_rate) != 3:
            raise ValueError(
                f"rotation_rate must have 3 elements, got {len(rotation_rate)}"
            )

        for idx in range(3):
            # More descriptive coordinate names
            coord_names = ["x", "y", "z"]
            coord = coord_names[idx]

            if np.size(speed[idx]) > 1:
                if np.shape(speed[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        f"speed[{coord}] must be a scalar or have the same shape as timestamp. "
                        f"Got shape {np.shape(speed[idx])}, expected {self.time_prop['timestamp_shape']}"
                    )

            if np.size(location[idx]) > 1:
                if np.shape(location[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        f"location[{coord}] must be a scalar or have the same shape as timestamp. "
                        f"Got shape {np.shape(location[idx])}, expected {self.time_prop['timestamp_shape']}"
                    )

            if np.size(rotation_rate[idx]) > 1:
                if np.shape(rotation_rate[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        f"rotation_rate[{coord}] must be a scalar or have the same shape as timestamp. "
                        f"Got shape {np.shape(rotation_rate[idx])}, expected {self.time_prop['timestamp_shape']}"
                    )

            if np.size(rotation[idx]) > 1:
                if np.shape(rotation[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        f"rotation[{coord}] must be a scalar or have the same shape as timestamp. "
                        f"Got shape {np.shape(rotation[idx])}, expected {self.time_prop['timestamp_shape']}"
                    )

    def process_radar_motion(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
    ) -> None:
        """
        Process radar motion parameters

        :param list location: 3D location of the radar [x, y, z] (m)
        :param list speed: Speed of the radar (m/s), [vx, vy, vz]
        :param list rotation: Radar's angle (deg), [yaw, pitch, roll]
        :param list rotation_rate: Radar's rotation rate (deg/s),
        [yaw rate, pitch rate, roll rate]

        """
        shape = self.time_prop["timestamp_shape"]

        # Check if any location or rotation parameters are time-varying (arrays)
        has_time_varying_motion = any(
            np.size(var) > 1 for var in list(location) + list(rotation)
        )

        if has_time_varying_motion:
            self.validate_radar_motion(location, speed, rotation, rotation_rate)
            self._setup_time_varying_motion(
                location, speed, rotation, rotation_rate, shape
            )
        else:
            self._setup_static_motion(location, speed, rotation, rotation_rate)

    def _setup_time_varying_motion(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
        shape: Tuple[int, ...],
    ) -> None:
        """
        Setup radar motion parameters for time-varying motion.

        :param location: 3D location parameters
        :param speed: Velocity parameters
        :param rotation: Rotation angle parameters
        :param rotation_rate: Angular velocity parameters
        :param shape: Shape of the timestamp array
        """
        # Initialize arrays for time-varying motion
        self.radar_prop["location"] = np.zeros(shape + (3,))
        self.radar_prop["rotation"] = np.zeros(shape + (3,))

        # Convert speed and rotation_rate to arrays (these are constant for time-varying case)
        self.radar_prop["speed"] = np.array(speed)
        self.radar_prop["rotation_rate"] = np.radians(np.array(rotation_rate))

        # Process each spatial dimension (x, y, z)
        for idx in range(3):
            self._process_location_dimension(location, speed, idx)
            self._process_rotation_dimension(rotation, rotation_rate, idx)

    def _process_location_dimension(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        idx: int,
    ) -> None:
        """Process a single dimension of location (x, y, or z)."""
        if np.size(location[idx]) > 1:
            # Time-varying position directly specified
            self.radar_prop["location"][:, :, :, idx] = location[idx]
        else:
            # Calculate position from initial location + velocity * time
            self.radar_prop["location"][:, :, :, idx] = (
                location[idx] + speed[idx] * self.time_prop["timestamp"]
            )

    def _process_rotation_dimension(
        self,
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
        idx: int,
    ) -> None:
        """Process a single dimension of rotation (yaw, pitch, or roll)."""
        if np.size(rotation[idx]) > 1:
            # Time-varying rotation directly specified
            self.radar_prop["rotation"][:, :, :, idx] = np.radians(rotation[idx])
        else:
            # Calculate rotation from initial angle + angular velocity * time
            self.radar_prop["rotation"][:, :, :, idx] = (
                np.radians(rotation[idx])
                + np.radians(rotation_rate[idx]) * self.time_prop["timestamp"]
            )

    def _setup_static_motion(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
    ) -> None:
        """
        Setup radar motion parameters for static (non-time-varying) motion.

        :param location: 3D location parameters
        :param speed: Velocity parameters
        :param rotation: Rotation angle parameters
        :param rotation_rate: Angular velocity parameters
        """
        self.radar_prop["speed"] = np.array(speed)
        self.radar_prop["location"] = np.array(location)
        self.radar_prop["rotation"] = np.radians(np.array(rotation))
        self.radar_prop["rotation_rate"] = np.radians(np.array(rotation_rate))

    @property
    def num_channels(self) -> int:
        """Get the total number of virtual array channels."""
        return int(self.array_prop["size"])

    @property
    def samples_per_pulse(self) -> int:
        """Get the number of samples per pulse."""
        samples = self.sample_prop["samples_per_pulse"]
        assert isinstance(samples, int)
        return samples

    @property
    def transmitter(self):
        """Get the transmitter instance."""
        return self.radar_prop["transmitter"]

    @property
    def receiver(self):
        """Get the receiver instance."""
        return self.radar_prop["receiver"]

    @property
    def virtual_array_locations(self) -> NDArray:
        """Get the 3D locations of virtual array elements."""
        virtual_array = self.array_prop["virtual_array"]
        assert isinstance(virtual_array, np.ndarray)
        return virtual_array

    def __str__(self) -> str:
        """String representation of the Radar."""
        return (
            f"Radar(channels={self.num_channels}, "
            f"samples_per_pulse={self.samples_per_pulse}, "
            f"fs={self.receiver.bb_prop['fs']/1e6:.1f} MHz)"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the Radar."""
        return (
            f"Radar(transmitter={self.transmitter.__class__.__name__}, "
            f"receiver={self.receiver.__class__.__name__}, "
            f"channels={self.num_channels}, "
            f"samples_per_pulse={self.samples_per_pulse})"
        )

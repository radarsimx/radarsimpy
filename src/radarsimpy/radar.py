r"""
This script contains classes that define all the parameters for
a radar system

This script requires that 'numpy' be installed within the Python
environment you are running this script in.

This file can be imported as a module and contains the following
class:

* Transmitter - A class defines parameters of a radar transmitter
* Receiver - A class defines parameters of a radar receiver
* Radar - A class defines basic parameters of a radar system

----------
RadarSimPy - A Radar Simulator Built with Python
Copyright (C) 2018 - PRESENT  radarsimx.com
E-mail: info@radarsimx.com
Website: https://radarsimx.com

 ____           _            ____  _          __  __
|  _ \ __ _  __| | __ _ _ __/ ___|(_)_ __ ___ \ \/ /
| |_) / _` |/ _` |/ _` | '__\___ \| | '_ ` _ \ \  /
|  _ < (_| | (_| | (_| | |   ___) | | | | | | |/  \
|_| \_\__,_|\__,_|\__,_|_|  |____/|_|_| |_| |_/_/\_\

"""


import numpy as np

from .util import cal_phase_noise


class Transmitter:
    """
    A class defines basic parameters of a radar transmitter

    :param f:
        Waveform frequency (Hz).
        If ``f`` is a single number, radar transmits a single-tone waveform.

        For linear modulation, specify ``f`` with ``[f_start, f_stop]``.

        ``f`` can alse be a 1-D array of an arbitrary waveform, specify
        the time with ``t``.
    :type f: float or numpy.1darray
    :param t:
        Timing of each pulse (s).
    :type t: float or numpy.1darray
    :param float tx_power:
        Transmitter power (dBm)
    :param int pulses:
        Total number of pulses
    :param float prp:
        Pulse repetition period (s). ``prp >=
        pulse_length``. If it is ``None``, ``prp =
        pulse_length``.

        ``prp`` can alse be a 1-D array to specify
        different repetition period for each pulse. In this case, the
        length of the 1-D array should equals to the length
        of ``pulses``
    :type repetitions_period: float or numpy.1darray
    :param numpy.1darray f_offset:
        Frequency offset for each pulse (Hz). The length must be the same
        as ``pulses``.
    :param numpy.1darray pn_f:
        Frequency of the phase noise (Hz)
    :param numpy.1darray pn_power:
        Power of the phase noise (dB/Hz)
    :param list[dict] channels:
        Properties of transmitter channels

        [{

        - **location** (*numpy.1darray*) --
            3D location of the channel [x, y, z] (m)
        - **polarization** (*numpy.1darray*) --
            Antenna polarization [x, y, z].
            ``default = [0, 0, 1] (vertical polarization)``
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
        - **pulse_amp** (*numpy.1darray*) --
            Relative amplitude sequence for pulse's amplitude modulation.
            The array length should be the same as `pulses`. ``default 1``
        - **pulse_phs** (*numpy.1darray*) --
            Phase code sequence for pulse's phase modulation (deg).
            The array length should be the same as `pulses`. ``default 0``
        - **mod_t** (*numpy.1darray*) --
            Time stamps for waveform modulation (s). ``default None``
        - **phs** (*numpy.1darray*) --
            Phase scheme for waveform modulation (deg). ``default None``
        - **amp** (*numpy.1darray*) --
            Relative amplitude scheme for waveform modulation. ``default None``

        }]

    :ivar numpy.1darray fc_vect:
        Center frequency array for the pulses (Hz)
    :ivar float fc_frame:
        Center frequency of the whole frame (Hz)
    :ivar float bandwidth:
        Bandwith of each pulse (Hz), calculated from ``max(f) - min(f)``
    :ivar float pulse_length:
        Dwell time of each pulse (s), calculated from ``t[-1] - t[0]``
    :ivar int channel_size:
        Number of transmitter channels
    :ivar numpy.2darray locations:
        3D location of the channels. Size of the aray is
        ``[channel_size, 3 <x, y, z>]`` (m)
    :ivar numpy.1darray delay:
        Delay for each channel (s)
    :ivar numpy.1darray polarization:
        Antenna polarization ``[x, y, z]``.

        - Horizontal polarization: ``[1, 0, 0]``
        - Vertical polarization: ``[0, 0, 1]``
    :ivar list[numpy.1darray] az_angles:
        Angles for each channel's azimuth pattern (deg)
    :ivar list[numpy.1darray] az_patterns:
        Azimuth pattern for each channel (dB)
    :ivar list[numpy.1darray] el_angles:
        Angles for each channel's elevation pattern (deg)
    :ivar list[numpy.1darray] el_patterns:
        Elevation pattern for each channel (dB)
    :ivar numpy.1darray antenna_gains:
        Antenna gain for each channel (dB).
        Antenna gain is ``max(az_pattern)``
    :ivar list[numpy.1darray] pulse_mod:
        Complex modulation code sequence for phase modulation.
        Lentgh of ``pulse_mod`` is the same as ``pulses``
    :ivar list[dict] waveform_mod:
        Waveform modulation properties for each channel.
        {
            ``enabled`` (*bool*) -- Enable waveform modulation
            ``var`` (*numpy.1darray*) -- Variance of the modulation
            ``t`` (*numpy.1darray*) -- Time stamps for waveform modulation
        }

    **Waveform**

    ::

        |                       prp
        |                  +-----------+
        |
        |            +---f[1]--->  /            /            /
        |                         /            /            /
        |                        /            /            /
        |                       /            /            /
        |                      /            /            /     ...
        |                     /            /            /
        |                    /            /            /
        |                   /            /            /
        |      +---f[0]--->/            /            /
        |
        |                  +-------+
        |                 t[0]    t[1]
        |
        |    Pulse         +--------------------------------------+
        |    modulation    |pulse_amp[0]|pulse_amp[1]|pulse_amp[2]|  ...
        |                  |pulse_phs[0]|pulse_phs[1]|pulse_phs[2]|  ...
        |                  +--------------------------------------+
        |
        |    Waveform      +--------------------------------------+
        |    modulation    |           amp / phs / mod_t          |  ...
        |                  +--------------------------------------+

    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-branches, too-many-statements
        self,
        f,
        t,
        tx_power=0,
        pulses=1,
        prp=None,
        f_offset=None,
        pn_f=None,
        pn_power=None,
        channels=None,
    ):
        self.rf_prop = {}
        self.waveform_prop = {}
        self.txchannel_prop = {}

        self.rf_prop["tx_power"] = tx_power
        self.rf_prop["pn_f"] = pn_f
        self.rf_prop["pn_power"] = pn_power
        self.validate_rf_prop(self.rf_prop)

        # get `f(t)`
        # the lenght of `f` should be the same as `t`
        if isinstance(f, (list, tuple, np.ndarray)):
            f = np.array(f)
        else:
            f = np.array([f, f])

        if isinstance(t, (list, tuple, np.ndarray)):
            t = np.array(t) - t[0]
        else:
            t = np.array([0, t])

        self.waveform_prop["f"] = f
        self.waveform_prop["t"] = t
        self.waveform_prop["bandwidth"] = np.max(f) - np.min(f)
        self.waveform_prop["pulse_length"] = t[-1]
        self.waveform_prop["pulses"] = pulses

        # frequency offset for each pulse
        # the length of `f_offset` should be the same as `pulses`
        if f_offset is None:
            f_offset = np.zeros(pulses)
        else:
            if isinstance(f_offset, (list, tuple, np.ndarray)):
                f_offset = np.array(f_offset)
            else:
                f_offset = f_offset + np.zeros(pulses)
        self.waveform_prop["f_offset"] = f_offset

        # Extend `prp` to a numpy.1darray.
        # Length equels to `pulses`
        if prp is None:
            prp = self.waveform_prop["pulse_length"] + np.zeros(pulses)
        else:
            if isinstance(prp, (list, tuple, np.ndarray)):
                prp = np.array(prp)
            else:
                prp = prp + np.zeros(pulses)
        self.waveform_prop["prp"] = prp

        # start time of each pulse, without considering the delay
        self.waveform_prop["pulse_start_time"] = np.cumsum(prp) - prp[0]

        self.validate_waveform_prop(self.waveform_prop)

        if channels is None:
            channels = [{"location": (0, 0, 0)}]

        # number of transmitter channels
        self.txchannel_prop["size"] = len(channels)

        # firing delay for each channel
        self.txchannel_prop["delay"] = np.zeros(self.txchannel_prop["size"])
        self.txchannel_prop["grid"] = np.zeros(self.txchannel_prop["size"])
        self.txchannel_prop["locations"] = np.zeros((self.txchannel_prop["size"], 3))
        self.txchannel_prop["polarization"] = np.zeros((self.txchannel_prop["size"], 3))

        # waveform modulation parameters
        self.txchannel_prop["waveform_mod"] = []

        # pulse modulation parameters
        self.txchannel_prop["pulse_mod"] = np.ones(
            (self.txchannel_prop["size"], pulses), dtype=complex
        )

        # azimuth patterns
        self.txchannel_prop["az_patterns"] = []
        self.txchannel_prop["az_angles"] = []

        # elevation patterns
        self.txchannel_prop["el_patterns"] = []
        self.txchannel_prop["el_angles"] = []

        # antenna peak gain
        # antenna gain is calculated based on azimuth pattern
        self.txchannel_prop["antenna_gains"] = np.zeros((self.txchannel_prop["size"]))

        for tx_idx, tx_element in enumerate(channels):
            self.txchannel_prop["delay"][tx_idx] = tx_element.get("delay", 0)
            self.txchannel_prop["grid"][tx_idx] = tx_element.get("grid", 1)

            self.txchannel_prop["locations"][tx_idx, :] = np.array(
                tx_element.get("location")
            )
            self.txchannel_prop["polarization"][tx_idx, :] = np.array(
                tx_element.get("polarization", [0, 0, 1])
            )

            self.txchannel_prop["waveform_mod"].append(
                self.process_waveform_modulation(
                    tx_element.get("mod_t", None),
                    tx_element.get("amp", None),
                    tx_element.get("phs", None),
                )
            )

            self.txchannel_prop["pulse_mod"][tx_idx, :] = self.process_pulse_modulation(
                tx_element.get("pulse_amp", np.ones((pulses))),
                tx_element.get("pulse_phs", np.zeros((pulses))),
            )

            self.process_patterns(tx_element, tx_idx)

    def validate_rf_prop(self, rf_prop):
        """_summary_

        Args:
            rf_prop (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        if rf_prop["pn_f"] is not None and rf_prop["pn_power"] is None:
            raise ValueError("Lengths of `pn_f` and `pn_power` should be the same")
        if rf_prop["pn_f"] is None and rf_prop["pn_power"] is not None:
            raise ValueError("Lengths of `pn_f` and `pn_power` should be the same")
        if rf_prop["pn_f"] is not None and rf_prop["pn_power"] is not None:
            if len(rf_prop["pn_f"]) != len(rf_prop["pn_power"]):
                raise ValueError("Lengths of `pn_f` and `pn_power` should be the same")

    def validate_waveform_prop(self, waveform_prop):
        """_summary_

        Args:
            waveform_prop (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        if len(waveform_prop["f"]) != len(waveform_prop["t"]):
            raise ValueError("Lengths of `f` and `t` should be the same")

        if len(waveform_prop["f_offset"]) != waveform_prop["pulses"]:
            raise ValueError("Lengths of `f_offset` and `pulses` should be the same")

        if len(waveform_prop["prp"]) != waveform_prop["pulses"]:
            raise ValueError("Length of `prp` should equal to the length of `pulses`.")

        if np.min(waveform_prop["prp"]) < waveform_prop["pulse_length"]:
            raise ValueError("`prp` should be larger than `pulse_length`")

    def process_waveform_modulation(self, mod_t, amp, phs):
        """_summary_

        Args:
            mod_t (_type_): _description_
            amp (_type_): _description_
            phs (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        if phs is not None and amp is None:
            amp = np.ones_like(phs)
        elif phs is None and amp is not None:
            phs = np.zeros_like(amp)

        if not all([mod_t, amp, phs]):
            return {"enabled": False, "var": None, "t": None}

        if isinstance(amp, (list, tuple, np.ndarray)):
            amp = np.array(amp)
        else:
            amp = np.array([amp, amp])

        if isinstance(phs, (list, tuple, np.ndarray)):
            phs = np.array(phs)
        else:
            phs = np.array([phs, phs])

        if isinstance(mod_t, (list, tuple, np.ndarray)):
            mod_t = np.array(mod_t)
        else:
            mod_t = np.array([0, mod_t])

        if len(amp) != len(phs):
            raise ValueError("Lengths of `amp` and `phs` should be the same")

        mod_var = amp * np.exp(1j * phs / 180 * np.pi)

        if len(mod_t) != len(mod_var):
            raise ValueError("Lengths of `mod_t`, `amp`, and `phs` should be the same")

        return {"enabled": True, "var": mod_var, "t": mod_t}

    def process_pulse_modulation(self, pulse_amp, pulse_phs):
        """_summary_

        Args:
            pulse_amp (_type_): _description_
            pulse_phs (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if len(pulse_amp) != self.waveform_prop["pulses"]:
            raise ValueError("Lengths of `pulse_amp` and `pulses` should be the same")
        if len(pulse_phs) != self.waveform_prop["pulses"]:
            raise ValueError("Length of `pulse_phs` and `pulses` should be the same")

        return pulse_amp * np.exp(1j * (pulse_phs / 180 * np.pi))

    def process_patterns(self, tx_channel, tx_idx):
        """_summary_

        Args:
            tx_channel (_type_): _description_
            tx_idx (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        # azimuth pattern
        az_angle = np.array(tx_channel.get("azimuth_angle", [-90, 90]))
        az_pattern = np.array(tx_channel.get("azimuth_pattern", [0, 0]))
        if len(az_angle) != len(az_pattern):
            raise ValueError(
                "Lengths of `azimuth_angle` and `azimuth_pattern` \
                    should be the same"
            )

        self.txchannel_prop["antenna_gains"][tx_idx] = np.max(az_pattern)
        az_pattern = az_pattern - self.txchannel_prop["antenna_gains"][tx_idx]

        self.txchannel_prop["az_angles"].append(az_angle)
        self.txchannel_prop["az_patterns"].append(az_pattern)

        # elevation pattern
        el_angle = np.array(tx_channel.get("elevation_angle", [-90, 90]))
        el_pattern = np.array(tx_channel.get("elevation_pattern", [0, 0]))
        if len(el_angle) != len(el_pattern):
            raise ValueError(
                "Lengths of `elevation_angle` and `elevation_pattern` \
                    should be the same"
            )
        el_pattern = el_pattern - np.max(el_pattern)

        self.txchannel_prop["el_angles"].append(el_angle)
        self.txchannel_prop["el_patterns"].append(el_pattern)


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
    :param string bb_type:
        Baseband data type, ``complex`` or ``real``.
        Deafult is ``complex``
    :param list[dict] channels:
        Properties of transmitter channels

        [{

        - **location** (*numpy.1darray*) --
            3D location of the channel [x, y, z] (m)
        - **polarization** (*numpy.1darray*) --
            Antenna polarization [x, y, z].
            ``default = [0, 0, 1] (vertical polarization)``
        - **azimuth_angle** (*numpy.1darray*) --
            Angles for azimuth pattern (deg). ``default [-90, 90]``
        - **azimuth_pattern** (*numpy.1darray*) --
            Azimuth pattern (dB). ``default [0, 0]``
        - **elevation_angle** (*numpy.1darray*) --
            Angles for elevation pattern (deg). ``default [-90, 90]``
        - **elevation_pattern** (*numpy.1darray*) --
            Elevation pattern (dB). ``default [0, 0]``

        }]

    :ivar float noise_bandwidth:
        Bandwidth in calculating the noise (Hz).
        ``noise_bandwidth = fs / 2``
    :ivar int channel_size:
        Total number of receiver channels
    :ivar numpy.2darray locations:
        3D location of the channels. Size of the aray is
        ``[channel_size, 3 <x, y, z>]`` (m)
    :ivar numpy.1darray polarization:
        Antenna polarization ``[x, y, z]``.

        - Horizontal polarization: ``[1, 0, 0]``
        - Vertical polarization: ``[0, 0, 1]``
    :ivar list[numpy.1darray] az_angles:
        Angles for each channel's azimuth pattern (deg)
    :ivar list[numpy.1darray] az_patterns:
        Azimuth pattern for each channel (dB)
    :ivar list[numpy.1darray] el_angles:
        Angles for each channel's elevation pattern (deg)
    :ivar list[numpy.1darray] el_patterns:
        Elevation pattern for each channel (dB)
    :ivar numpy.1darray antenna_gains:
        Antenna gain for each channel (dB).
        Antenna gain is ``max(az_pattern)``

    **Receiver noise**

    ::

        |           + n1 = 10*log10(Boltzmann_constant * noise_temperature * 1000)
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

    def __init__(
        self,
        fs,
        noise_figure=10,
        rf_gain=0,
        load_resistor=500,
        baseband_gain=0,
        bb_type="complex",
        channels=None,
    ):
        self.fs = fs
        self.noise_figure = noise_figure
        self.rf_gain = rf_gain
        self.load_resistor = load_resistor
        self.baseband_gain = baseband_gain
        self.bb_type = bb_type
        if bb_type == "complex":
            self.noise_bandwidth = self.fs
        elif bb_type == "real":
            self.noise_bandwidth = self.fs / 2
        else:
            raise ValueError("Invalid baseband type")

        # additional receiver parameters
        if channels is None:
            self.channels = [{"location": (0, 0, 0)}]
        else:
            self.channels = channels

        self.channel_size = len(self.channels)

        self.locations = np.zeros((self.channel_size, 3))
        self.polarization = np.zeros((self.channel_size, 3))

        self.az_patterns = []
        self.az_angles = []

        self.el_patterns = []
        self.el_angles = []

        self.antenna_gains = np.zeros((self.channel_size))

        for rx_idx, rx_element in enumerate(self.channels):
            self.locations[rx_idx, :] = np.array(rx_element.get("location"))
            self.polarization[rx_idx, :] = np.array(
                rx_element.get("polarization", [0, 0, 1])
            )

            # azimuth pattern
            self.az_angles.append(
                np.array(
                    self.channels[rx_idx].get("azimuth_angle", np.arange(-90, 91, 180))
                )
            )
            self.az_patterns.append(
                np.array(self.channels[rx_idx].get("azimuth_pattern", np.zeros(2)))
            )
            if len(self.az_angles[-1]) != len(self.az_patterns[-1]):
                raise ValueError(
                    "Lengths of `azimuth_angle` and `azimuth_pattern` \
                        should be the same"
                )

            self.antenna_gains[rx_idx] = np.max(self.az_patterns[-1])
            self.az_patterns[-1] = self.az_patterns[-1] - np.max(self.az_patterns[-1])

            # elevation pattern
            self.el_angles.append(
                np.array(
                    self.channels[rx_idx].get(
                        "elevation_angle", np.arange(-90, 91, 180)
                    )
                )
            )
            self.el_patterns.append(
                np.array(self.channels[rx_idx].get("elevation_pattern", np.zeros(2)))
            )
            if len(self.el_angles[-1]) != len(self.el_patterns[-1]):
                raise ValueError(
                    "Lengths of `elevation_angle` and `elevation_pattern` \
                        should be the same"
                )

            self.el_patterns[-1] = self.el_patterns[-1] - np.max(self.el_patterns[-1])


class Radar:
    """
    A class defines basic parameters of a radar system

    :param Transmitter transmitter:
        Radar transmiter
    :param Receiver receiver:
        Radar Receiver
    :param numpy.1darray location:
        3D location of the radar [x, y, z] (m). ``default
        [0, 0, 0]``
    :param numpy.1darray speed:
        Speed of the radar (m/s), [vx, vy, vz]. ``default
        [0, 0, 0]``
    :param numpy.1darray rotation:
        Radar's angle (deg), [yaw, pitch, roll].
        ``default [0, 0, 0]``
    :param numpy.1darray rotation_rate:
        Radar's rotation rate (deg/s),
        [yaw rate, pitch rate, roll rate]
        ``default [0, 0, 0]``
    :param time:
        Radar firing time instances / frames
        :type time: float or numpy.1darray
    :param Radar interf:
        Interference radar. ``default None``
    :param int seed:
        Seed for noise generator

    :ivar int samples_per_pulse:
        Number of samples in one pulse
    :ivar int channel_size:
        Total number of channels.
        ``channel_size = transmitter.channel_size * receiver.channel_size``
    :ivar numpy.2darray virtual_array:
        Locations of virtual array elements. [channel_size, 3 <x, y, z>]
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

    def __init__(
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
        self.transmitter = transmitter
        self.receiver = receiver

        self.validation = kwargs.get("validation", False)

        self.samples_per_pulse = int(self.transmitter.waveform_prop["pulse_length"] * self.receiver.fs)

        self.t_offset = np.array(time)
        self.frames = np.size(time)

        # virtual array
        self.channel_size = self.transmitter.txchannel_prop["size"] * self.receiver.channel_size
        self.virtual_array = np.repeat(
            self.transmitter.txchannel_prop["locations"], self.receiver.channel_size, axis=0
        ) + np.tile(self.receiver.locations, (self.transmitter.txchannel_prop["size"], 1))

        self.timestamp = self.gen_timestamp()
        self.pulse_phs = self.cal_frame_phases()

        self.noise = self.cal_noise()

        # if hasattr(self.transmitter.fc, '__len__'):
        # self.fc_mat = np.tile(
        #     self.transmitter.fc_vect[np.newaxis, :, np.newaxis],
        #     (self.channel_size, 1, self.samples_per_pulse),
        # )

        # self.f_offset_mat = np.tile(
        #     self.transmitter.f_offset[np.newaxis, :, np.newaxis],
        #     (self.channel_size, 1, self.samples_per_pulse),
        # )

        beat_time_samples = np.arange(0, self.samples_per_pulse, 1) / self.receiver.fs
        self.beat_time = np.tile(
            beat_time_samples[np.newaxis, np.newaxis, ...],
            (self.channel_size, self.transmitter.waveform_prop["pulses"], 1),
        )

        if self.transmitter.rf_prop["pn_f"] is not None and self.transmitter.rf_prop["pn_power"] is not None:
            dummy_sig = np.ones(
                (
                    self.channel_size * self.frames * self.transmitter.waveform_prop["pulses"],
                    self.samples_per_pulse,
                )
            )
            self.phase_noise = cal_phase_noise(
                dummy_sig,
                self.receiver.fs,
                self.transmitter.rf_prop["pn_f"],
                self.transmitter.rf_prop["pn_power"],
                seed=seed,
                validation=self.validation,
            )
            self.phase_noise = np.reshape(
                self.phase_noise,
                (
                    self.channel_size * self.frames,
                    self.transmitter.waveform_prop["pulses"],
                    self.samples_per_pulse,
                ),
            )
        else:
            self.phase_noise = None

        self.location = np.array(location)
        self.speed = np.array(speed)
        self.rotation = np.array(rotation)
        self.rotation_rate = np.array(rotation_rate)
        shape = np.shape(self.timestamp)

        if any(
            np.size(var) > 1
            for var in list(location)
            + list(speed)
            + list(rotation)
            + list(rotation_rate)
        ):
            self.location = np.zeros(shape + (3,))
            self.speed = np.zeros(shape + (3,))
            self.rotation = np.zeros(shape + (3,))
            self.rotation_rate = np.zeros(shape + (3,))

            if np.size(speed[0]) > 1:
                if np.shape(speed[0]) != shape:
                    raise ValueError(
                        "speed[0] must be a scalar or have the same shape as "
                        "timestamp"
                    )
                self.speed[:, :, :, 0] = speed[0]
            else:
                self.speed[:, :, :, 0] = np.full(shape, speed[0])

            if np.size(speed[1]) > 1:
                if np.shape(speed[1]) != shape:
                    raise ValueError(
                        "speed[1] must be a scalar or have the same shape as "
                        "timestamp"
                    )
                self.speed[:, :, :, 1] = speed[1]
            else:
                self.speed[:, :, :, 1] = np.full(shape, speed[1])

            if np.size(speed[2]) > 1:
                if np.shape(speed[2]) != shape:
                    raise ValueError(
                        "speed[2] must be a scalar or have the same shape as "
                        "timestamp"
                    )
                self.speed[:, :, :, 2] = speed[2]
            else:
                self.speed[:, :, :, 2] = np.full(shape, speed[2])

            if np.size(location[0]) > 1:
                if np.shape(location[0]) != shape:
                    raise ValueError(
                        "location[0] must be a scalar or have the same shape "
                        "as timestamp"
                    )
                self.location[:, :, :, 0] = location[0]
            else:
                self.location[:, :, :, 0] = location[0] + speed[0] * self.timestamp

            if np.size(location[1]) > 1:
                if np.shape(location[1]) != shape:
                    raise ValueError(
                        "location[1] must be a scalar or have the same shape "
                        "as timestamp"
                    )
                self.location[:, :, :, 1] = location[1]
            else:
                self.location[:, :, :, 1] = location[1] + speed[1] * self.timestamp

            if np.size(location[2]) > 1:
                if np.shape(location[2]) != shape:
                    raise ValueError(
                        "location[2] must be a scalar or have the same shape "
                        "as timestamp"
                    )
                self.location[:, :, :, 2] = location[2]
            else:
                self.location[:, :, :, 2] = location[2] + speed[2] * self.timestamp

            if np.size(rotation_rate[0]) > 1:
                if np.shape(rotation_rate[0]) != shape:
                    raise ValueError(
                        "rotation_rate[0] must be a scalar or have the same "
                        "shape as timestamp"
                    )
                self.rotation_rate[:, :, :, 0] = np.radians(rotation_rate[0])
            else:
                self.rotation_rate[:, :, :, 0] = np.full(
                    shape, np.radians(rotation_rate[0])
                )

            if np.size(rotation_rate[1]) > 1:
                if np.shape(rotation_rate[1]) != shape:
                    raise ValueError(
                        "rotation_rate[1] must be a scalar or have the same "
                        "shape as timestamp"
                    )
                self.rotation_rate[:, :, :, 1] = np.radians(rotation_rate[1])
            else:
                self.rotation_rate[:, :, :, 1] = np.full(
                    shape, np.radians(rotation_rate[1])
                )

            if np.size(rotation_rate[2]) > 1:
                if np.shape(rotation_rate[2]) != shape:
                    raise ValueError(
                        "rotation_rate[2] must be a scalar or have the same "
                        "shape as timestamp"
                    )
                self.rotation_rate[:, :, :, 2] = np.radians(rotation_rate[2])
            else:
                self.rotation_rate[:, :, :, 2] = np.full(
                    shape, np.radians(rotation_rate[2])
                )

            if np.size(rotation[0]) > 1:
                if np.shape(rotation[0]) != shape:
                    raise ValueError(
                        "rotation[0] must be a scalar or have the same shape "
                        "as timestamp"
                    )
                self.rotation[:, :, :, 0] = np.radians(rotation[0])
            else:
                self.rotation[:, :, :, 0] = np.radians(
                    rotation[0] + rotation_rate[0] * self.timestamp
                )

            if np.size(rotation[1]) > 1:
                if np.shape(rotation[1]) != shape:
                    raise ValueError(
                        "rotation[1] must be a scalar or have the same shape "
                        "as timestamp"
                    )
                self.rotation[:, :, :, 1] = np.radians(rotation[1])
            else:
                self.rotation[:, :, :, 1] = np.radians(
                    rotation[1] + rotation_rate[1] * self.timestamp
                )

            if np.size(rotation[2]) > 1:
                if np.shape(rotation[2]) != shape:
                    raise ValueError(
                        "rotation[2] must be a scalar or have the same shape "
                        "as timestamp"
                    )
                self.rotation[:, :, :, 2] = np.radians(rotation[2])
            else:
                self.rotation[:, :, :, 2] = np.radians(
                    rotation[2] + rotation_rate[2] * self.timestamp
                )
        else:
            self.speed = np.array(speed)
            self.loccation = np.array(location)
            self.rotation = np.array(np.radians(rotation))
            self.rotation_rate = np.array(np.radians(rotation_rate))

        self.interf = interf

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
        pulses = self.transmitter.waveform_prop["pulses"]
        samples = self.samples_per_pulse
        crp = self.transmitter.waveform_prop["prp"]
        delay = self.transmitter.txchannel_prop["delay"]
        fs = self.receiver.fs

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

        if self.frames > 1:
            toffset = np.repeat(
                np.tile(
                    np.expand_dims(np.expand_dims(self.t_offset, axis=1), axis=2),
                    (1, self.transmitter.pulses, self.samples_per_pulse),
                ),
                self.channel_size,
                axis=0,
            )

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

        pulse_phs = self.transmitter.txchannel_prop["pulse_mod"]
        pulse_phs = np.repeat(pulse_phs, self.receiver.channel_size, axis=0)
        pulse_phs = np.repeat(pulse_phs, self.frames, axis=0)
        return pulse_phs

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
                self.channel_size,
                self.transmitter.waveform_prop["pulses"],
                self.samples_per_pulse,
            ]
        )

        boltzmann_const = 1.38064852e-23

        input_noise_dbm = 10 * np.log10(boltzmann_const * noise_temp * 1000)  # dBm/Hz
        receiver_noise_dbm = (
            input_noise_dbm
            + self.receiver.rf_gain
            + self.receiver.noise_figure
            + 10 * np.log10(self.receiver.noise_bandwidth)
            + self.receiver.baseband_gain
        )  # dBm/Hz
        receiver_noise_watts = 1e-3 * 10 ** (receiver_noise_dbm / 10)  # Watts/sqrt(hz)
        noise_amplitude_mixer = np.sqrt(
            receiver_noise_watts * self.receiver.load_resistor
        )
        noise_amplitude_peak = np.sqrt(2) * noise_amplitude_mixer + noise_amp
        return noise_amplitude_peak

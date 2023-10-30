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

    :ivar dict rf_prop: RF properties

        - **tx_power**: Transmitter power (dBm)

        - **pn_f**: Frequency of the phase noise (Hz)

        - **pn_power**: Power of the phase noise (dB/Hz)

    :ivar dict waveform_prop: Waveform properties

        - **f**: Waveform frequency (Hz)

        - **t**: Timing of each pulse (s)

        - **bandwidth**: Transmitting bandwidth (Hz)

        - **pulse_length**: Transmitting length (s)

        - **pulses**: Number of pulses

        - **f_offset**: Frequency offset for each pulse

        - **prp**: Pulse repetition time (s)

        - **pulse_start_time**: Start time of each pulse

    :ivar dict txchannel_prop: Transmitter channels

        - **size**: Number of transmitter channels

        - **delay**: Tx start delay (s)

        - **grid**: Ray tracing grid size (deg)

        - **locations**: Location of the Tx channel [x, y, z] m

        - **polarization**: Polarization of the Tx channel

        - **waveform_mod**: Waveform modulation parameters

        - **pulse_mod**: Pulse modulation parameters

        - **az_angles**: Azimuth angles (deg)

        - **az_patterns**: Azimuth pattern (dB)

        - **el_angles**: Elevation angles (deg)

        - **el_patterns**: Elevation pattern (dB)

        - **antenna_gains**: Tx antenna gain (dB)

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

    def __init__(  # pylint: disable=too-many-arguments
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

        self.txchannel_prop = self.process_txchannel_prop(channels)

    def validate_rf_prop(self, rf_prop):
        """
        Validate RF properties

        :param dict rf_prop: RF properties

        :raises ValueError: Lengths of `pn_f` and `pn_power` should be the same
        :raises ValueError: Lengths of `pn_f` and `pn_power` should be the same
        :raises ValueError: Lengths of `pn_f` and `pn_power` should be the same
        """
        if rf_prop["pn_f"] is not None and rf_prop["pn_power"] is None:
            raise ValueError("Lengths of `pn_f` and `pn_power` should be the same")
        if rf_prop["pn_f"] is None and rf_prop["pn_power"] is not None:
            raise ValueError("Lengths of `pn_f` and `pn_power` should be the same")
        if rf_prop["pn_f"] is not None and rf_prop["pn_power"] is not None:
            if len(rf_prop["pn_f"]) != len(rf_prop["pn_power"]):
                raise ValueError("Lengths of `pn_f` and `pn_power` should be the same")

    def validate_waveform_prop(self, waveform_prop):
        """
        Validate waveform properties

        :param waveform_prop (dict): Wavefrom properties

        :raises ValueError: Lengths of `f` and `t` should be the same
        :raises ValueError: Lengths of `f_offset` and `pulses` should be the same
        :raises ValueError: Length of `prp` should equal to the length of `pulses`
        :raises ValueError: `prp` should be larger than `pulse_length`
        """
        if len(waveform_prop["f"]) != len(waveform_prop["t"]):
            raise ValueError("Lengths of `f` and `t` should be the same")

        if len(waveform_prop["f_offset"]) != waveform_prop["pulses"]:
            raise ValueError("Lengths of `f_offset` and `pulses` should be the same")

        if len(waveform_prop["prp"]) != waveform_prop["pulses"]:
            raise ValueError("Length of `prp` should equal to the length of `pulses`")

        if np.min(waveform_prop["prp"]) < waveform_prop["pulse_length"]:
            raise ValueError("`prp` should be larger than `pulse_length`")

    def process_waveform_modulation(self, mod_t, amp, phs):
        """
        Process waveform modulation parameters

        :param numpy.1darray mod_t: Time stamps for waveform modulation (s). ``default None``
        :param numpy.1darray amp:
            Relative amplitude scheme for waveform modulation. ``default None``
        :param numpy.1darray phs: Phase scheme for waveform modulation (deg). ``default None``

        :raises ValueError: Lengths of `amp` and `phs` should be the same
        :raises ValueError: Lengths of `mod_t`, `amp`, and `phs` should be the same

        :return:
            Waveform modulation
        :rtype: dict
        """

        if phs is not None and amp is None:
            amp = np.ones_like(phs)
        elif phs is None and amp is not None:
            phs = np.zeros_like(amp)

        if mod_t is None or amp is None or phs is None:
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
        """
        Process pulse modulation parameters

        :param numpy.1darray pulse_amp:
            Relative amplitude sequence for pulse's amplitude modulation.
            The array length should be the same as `pulses`. ``default 1``
        :param numpy.1darray pulse_phs:
            Phase code sequence for pulse's phase modulation (deg).
            The array length should be the same as `pulses`. ``default 0``

        :raises ValueError: Lengths of `pulse_amp` and `pulses` should be the same
        :raises ValueError: Length of `pulse_phs` and `pulses` should be the same

        :return:
            Pulse modulation array
        :rtype: numpy.1darray
        """
        if len(pulse_amp) != self.waveform_prop["pulses"]:
            raise ValueError("Lengths of `pulse_amp` and `pulses` should be the same")
        if len(pulse_phs) != self.waveform_prop["pulses"]:
            raise ValueError("Length of `pulse_phs` and `pulses` should be the same")

        return pulse_amp * np.exp(1j * (pulse_phs / 180 * np.pi))

    def process_txchannel_prop(self, channels):
        """
        Process transmitter channel parameters

        :param dict channels: Dictionary of transmitter channels

        :raises ValueError: Lengths of `azimuth_angle` and `azimuth_pattern`
            should be the same
        :raises ValueError: Lengths of `elevation_angle` and `elevation_pattern`
            should be the same

        :return:
            Transmitter channel properties
        :rtype: dict
        """
        # number of transmitter channels
        txch_prop = {}

        txch_prop["size"] = len(channels)

        # firing delay for each channel
        txch_prop["delay"] = np.zeros(txch_prop["size"])
        txch_prop["grid"] = np.zeros(txch_prop["size"])
        txch_prop["locations"] = np.zeros((txch_prop["size"], 3))
        txch_prop["polarization"] = np.zeros((txch_prop["size"], 3))

        # waveform modulation parameters
        txch_prop["waveform_mod"] = []

        # pulse modulation parameters
        txch_prop["pulse_mod"] = np.ones(
            (txch_prop["size"], self.waveform_prop["pulses"]), dtype=complex
        )

        # azimuth patterns
        txch_prop["az_patterns"] = []
        txch_prop["az_angles"] = []

        # elevation patterns
        txch_prop["el_patterns"] = []
        txch_prop["el_angles"] = []

        # antenna peak gain
        # antenna gain is calculated based on azimuth pattern
        txch_prop["antenna_gains"] = np.zeros((txch_prop["size"]))

        for tx_idx, tx_element in enumerate(channels):
            txch_prop["delay"][tx_idx] = tx_element.get("delay", 0)
            txch_prop["grid"][tx_idx] = tx_element.get("grid", 1)

            txch_prop["locations"][tx_idx, :] = np.array(tx_element.get("location"))
            txch_prop["polarization"][tx_idx, :] = np.array(
                tx_element.get("polarization", [0, 0, 1])
            )

            txch_prop["waveform_mod"].append(
                self.process_waveform_modulation(
                    tx_element.get("mod_t", None),
                    tx_element.get("amp", None),
                    tx_element.get("phs", None),
                )
            )

            txch_prop["pulse_mod"][tx_idx, :] = self.process_pulse_modulation(
                tx_element.get("pulse_amp", np.ones((self.waveform_prop["pulses"]))),
                tx_element.get("pulse_phs", np.zeros((self.waveform_prop["pulses"]))),
            )

            # azimuth pattern
            az_angle = np.array(tx_element.get("azimuth_angle", [-90, 90]))
            az_pattern = np.array(tx_element.get("azimuth_pattern", [0, 0]))
            if len(az_angle) != len(az_pattern):
                raise ValueError(
                    "Lengths of `azimuth_angle` and `azimuth_pattern` \
                        should be the same"
                )

            txch_prop["antenna_gains"][tx_idx] = np.max(az_pattern)
            az_pattern = az_pattern - txch_prop["antenna_gains"][tx_idx]

            txch_prop["az_angles"].append(az_angle)
            txch_prop["az_patterns"].append(az_pattern)

            # elevation pattern
            el_angle = np.array(tx_element.get("elevation_angle", [-90, 90]))
            el_pattern = np.array(tx_element.get("elevation_pattern", [0, 0]))
            if len(el_angle) != len(el_pattern):
                raise ValueError(
                    "Lengths of `elevation_angle` and `elevation_pattern` \
                        should be the same"
                )
            el_pattern = el_pattern - np.max(el_pattern)

            txch_prop["el_angles"].append(el_angle)
            txch_prop["el_patterns"].append(el_pattern)

        return txch_prop


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

    :ivar dict rf_prop: RF properties

        - **rf_gain**: RF gain of the receiver (dB)

        - **noise_figure**: Receiver noise figure (dB)

    :ivar dict bb_prop: Baseband properties

        - **fs**: Sampling rate

        - **load_resistor**: Load resistor (ohm)

        - **baseband_gain**: Baseband gain (dB)

        - **bb_type**: Baseband type, ``real`` or ``complex``

    :ivar dict rxchannel_prop: Receiver channels

        - **size**: Number of receiver channels

        - **locations**: Location of the Rx channel [x, y, z] m

        - **polarization**: Polarization of the Rx channel

        - **az_angles**: Azimuth angles (deg)

        - **az_patterns**: Azimuth pattern (dB)

        - **el_angles**: Elevation angles (deg)

        - **el_patterns**: Elevation pattern (dB)

        - **antenna_gains**: Rx antenna gain (dB)

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

    def __init__(  # pylint: disable=too-many-arguments
        self,
        fs,
        noise_figure=10,
        rf_gain=0,
        load_resistor=500,
        baseband_gain=0,
        bb_type="complex",
        channels=None,
    ):
        self.rf_prop = {}
        self.bb_prop = {}
        self.rxchannel_prop = {}

        self.rf_prop["rf_gain"] = rf_gain
        self.rf_prop["noise_figure"] = noise_figure

        self.bb_prop["fs"] = fs
        self.bb_prop["load_resistor"] = load_resistor
        self.bb_prop["baseband_gain"] = baseband_gain
        self.bb_prop["bb_type"] = bb_type
        if bb_type == "complex":
            self.bb_prop["noise_bandwidth"] = fs
        elif bb_type == "real":
            self.bb_prop["noise_bandwidth"] = fs / 2

        self.validate_bb_prop(self.bb_prop)

        # additional receiver parameters
        if channels is None:
            channels = [{"location": (0, 0, 0)}]

        self.rxchannel_prop = self.process_rxchannel_prop(channels)

    def validate_bb_prop(self, bb_prop):
        """
        Validate baseband properties

        :param dict bb_prop: Baseband properties

        :raises ValueError: Invalid baseband type
        """
        if bb_prop["bb_type"] != "complex" and bb_prop["bb_type"] != "real":
            raise ValueError("Invalid baseband type")

    def process_rxchannel_prop(self, channels):
        """
        Process receiver channel parameters

        :param dict channels: Dictionary of receiver channels

        :raises ValueError: Lengths of `azimuth_angle` and `azimuth_pattern`
            should be the same
        :raises ValueError: Lengths of `elevation_angle` and `elevation_pattern`
            should be the same

        :return:
            Receiver channel properties
        :rtype: dict
        """
        rxch_prop = {}

        rxch_prop["size"] = len(channels)

        rxch_prop["locations"] = np.zeros((rxch_prop["size"], 3))
        rxch_prop["polarization"] = np.zeros((rxch_prop["size"], 3))

        rxch_prop["az_patterns"] = []
        rxch_prop["az_angles"] = []

        rxch_prop["el_patterns"] = []
        rxch_prop["el_angles"] = []

        rxch_prop["antenna_gains"] = np.zeros((rxch_prop["size"]))

        for rx_idx, rx_element in enumerate(channels):
            rxch_prop["locations"][rx_idx, :] = np.array(rx_element.get("location"))
            rxch_prop["polarization"][rx_idx, :] = np.array(
                rx_element.get("polarization", [0, 0, 1])
            )

            # azimuth pattern
            az_angle = np.array(rx_element.get("azimuth_angle", [-90, 90]))
            az_pattern = np.array(rx_element.get("azimuth_pattern", [0, 0]))
            if len(az_angle) != len(az_pattern):
                raise ValueError(
                    "Lengths of `azimuth_angle` and `azimuth_pattern` \
                        should be the same"
                )

            rxch_prop["antenna_gains"][rx_idx] = np.max(az_pattern)
            az_pattern = az_pattern - rxch_prop["antenna_gains"][rx_idx]

            rxch_prop["az_angles"].append(az_angle)
            rxch_prop["az_patterns"].append(az_pattern)

            # elevation pattern
            el_angle = np.array(rx_element.get("elevation_angle", [-90, 90]))
            el_pattern = np.array(rx_element.get("elevation_pattern", [0, 0]))
            if len(el_angle) != len(el_pattern):
                raise ValueError(
                    "Lengths of `elevation_angle` and `elevation_pattern` \
                        should be the same"
                )
            el_pattern = el_pattern - np.max(el_pattern)

            rxch_prop["el_angles"].append(el_angle)
            rxch_prop["el_patterns"].append(el_pattern)

        return rxch_prop


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
                        + "] must be a scalar or have the same shape as "
                        "timestamp"
                    )

            if np.size(location[idx]) > 1:
                if np.shape(location[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        "location["
                        + str(idx)
                        + "] must be a scalar or have the same shape "
                        "as timestamp"
                    )

            if np.size(rotation_rate[idx]) > 1:
                if np.shape(rotation_rate[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        "rotation_rate["
                        + str(idx)
                        + "] must be a scalar or have the same "
                        "shape as timestamp"
                    )

            if np.size(rotation[idx]) > 1:
                if np.shape(rotation[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        "rotation["
                        + str(idx)
                        + "] must be a scalar or have the same shape "
                        "as timestamp"
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

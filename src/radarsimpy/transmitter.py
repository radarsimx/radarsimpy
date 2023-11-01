"""
This script contains classes that define all the parameters for
a radar transmitter

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

        █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █
        █                        prp                                            █
        █                   +-----------+                                       █
        █       +---f[1]--------->  /            /            /                 █
        █                          /            /            /                  █
        █                         /            /            /                   █
        █                        /            /            /                    █
        █                       /            /            /     ...             █
        █                      /            /            /                      █
        █                     /            /            /                       █
        █                    /            /            /                        █
        █       +---f[0]--->/            /            /                         █
        █                   +-------+                                           █
        █                  t[0]    t[1]                                         █
        █                                                                       █
        █     Pulse         +--------------------------------------+            █
        █     modulation    |pulse_amp[0]|pulse_amp[1]|pulse_amp[2]|  ...       █
        █                   |pulse_phs[0]|pulse_phs[1]|pulse_phs[2]|  ...       █
        █                   +--------------------------------------+            █
        █                                                                       █
        █     Waveform      +--------------------------------------+            █
        █     modulation    |           amp / phs / mod_t          |  ...       █
        █                   +--------------------------------------+            █
        █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █
        
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

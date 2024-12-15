"""
This script contains classes that define all the parameters for
a radar receiver

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


class Receiver:
    """
    Represents the basic parameters and properties of a radar receiver.

    This class defines the RF and baseband properties of a radar receiver,
    along with the characteristics of its receiver channels.

    :param float fs: Sampling rate in samples per second (sps).
    :param float noise_figure: Noise figure of the receiver in decibels (dB).
    :param float rf_gain: Total RF gain of the receiver in decibels (dB).
    :param float load_resistor: Load resistance to convert power to voltage, in ohms (Ω).
    :param float baseband_gain: Total baseband gain in decibels (dB).
    :param str bb_type: Baseband data type, either ``complex`` or ``real``.
    Defaults to ``complex``.
    :param list[dict] channels: A list of dictionaries defining the properties of receiver channels,
    where each dictionary contains the following keys:

        - **location** (*numpy.ndarray*): 3D location of the channel relative to the radar's position [x, y, z] in meters.
        - **polarization** (*numpy.ndarray*): Antenna polarization vector [x, y, z].
        Defaults to ``[0, 0, 1]`` (vertical polarization).
        Examples:

            - Vertical polarization: ``[0, 0, 1]``
            - Horizontal polarization: ``[0, 1, 0]``
            - Right-handed circular polarization: ``[0, 1, 1j]``
            - Left-handed circular polarization: ``[0, 1, -1j]``
        - **azimuth_angle** (*numpy.ndarray*): Azimuth pattern angles in degrees.
        Defaults to ``[-90, 90]``.
        - **azimuth_pattern** (*numpy.ndarray*): Azimuth pattern in decibels (dB).
        Defaults to ``[0, 0]``.
        - **elevation_angle** (*numpy.ndarray*): Elevation pattern angles in degrees.
        Defaults to ``[-90, 90]``.
        - **elevation_pattern** (*numpy.ndarray*): Elevation pattern in decibels (dB).
        Defaults to ``[0, 0]``.

    :ivar dict rf_prop: RF properties of the receiver:

        - **rf_gain** (*float*): RF gain in decibels (dB).
        - **noise_figure** (*float*): Noise figure in decibels (dB).

    :ivar dict bb_prop: Baseband properties of the receiver:

        - **fs** (*float*): Sampling rate in samples per second (sps).
        - **load_resistor** (*float*): Load resistance in ohms (Ω).
        - **baseband_gain** (*float*): Baseband gain in decibels (dB).
        - **bb_type** (*str*): Baseband data type, either ``real`` or ``complex``.

    :ivar dict rxchannel_prop: Properties of the receiver channels:

        - **size** (*int*): Number of receiver channels.
        - **locations** (*numpy.ndarray*): 3D locations of the receiver channels [x, y, z] in meters.
        - **polarization** (*numpy.ndarray*): Polarization vectors of the receiver channels.
        - **az_angles** (*numpy.ndarray*): Azimuth angles in degrees.
        - **az_patterns** (*numpy.ndarray*): Azimuth pattern in decibels (dB).
        - **el_angles** (*numpy.ndarray*): Elevation angles in degrees.
        - **el_patterns** (*numpy.ndarray*): Elevation pattern in decibels (dB).
        - **antenna_gains** (*numpy.ndarray*): Antenna gains of the receiver channels in decibels (dB).

    **Receiver Noise Model**:

    The following diagram illustrates the radar receiver noise model and calculations:

    ::

        █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █
        █  +-------------+                                              █
        █  | Rx Antenna  |                                              █
        █  +------+------+                                              █
        █         | n1 = 10*log10(boltzmann_const * noise_temp * 1000)  █
        █         ↓      + 10*log10(noise_bandwidth)  (dBm)             █
        █  +------+------+                                              █
        █  |    RF Amp   |                                              █
        █  +------+------+                                              █
        █         | n2 = n1 + noise_figure + rf_gain (dBm)              █
        █         ↓ n3 = 1e-3 * 10^(n2/10) (Watts)                      █
        █  +------+------+                                              █
        █  |    Mixer    |                                              █
        █  +------+------+                                              █
        █         | n4 = sqrt(n3 * load_resistor) (V)                   █
        █         ↓                                                     █
        █  +------+------+                                              █
        █  |Baseband Amp |                                              █
        █  +------+------+                                              █
        █         | noise amplitude (peak to peak)                      █
        █         ↓ n5 = n4 * 10^(baseband_gain / 20) * sqrt(2) (V)     █
        █  +------+------+                                              █
        █  |     ADC     |                                              █
        █  +-------------+                                              █
        █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █

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

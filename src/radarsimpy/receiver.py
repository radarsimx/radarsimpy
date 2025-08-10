"""
Radar Receiver Configuration and Noise Modeling

This module provides the `Receiver` class, which defines the parameters and
properties of a radar receiver. It includes tools for configuring receiver
channels, modeling noise properties, and validating baseband and RF characteristics.
The module is intended to support radar system simulation by accurately modeling
receiver behavior.

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

from typing import List, Dict, Optional
import numpy as np
from numpy.typing import NDArray

# Constants
DEFAULT_POLARIZATION = [0, 0, 1]  # Vertical polarization
DEFAULT_AZIMUTH_RANGE = [-90, 90]
DEFAULT_ELEVATION_RANGE = [-90, 90]
DEFAULT_PATTERN_DB = [0, 0]
VALID_BB_TYPES = {"complex", "real"}


class Receiver:
    """
    Represents the basic parameters and properties of a radar receiver.

    This class defines the RF and baseband properties of a radar receiver,
    along with the characteristics of its receiver channels.

    :param float fs:
        Sampling rate in samples per second (sps).
    :param float noise_figure:
        Noise figure of the receiver in decibels (dB).
    :param float rf_gain:
        Total RF gain of the receiver in decibels (dB).
    :param float load_resistor:
        Load resistance to convert power to voltage, in ohms (Ω).
    :param float baseband_gain:
        Total baseband gain in decibels (dB).
    :param str bb_type:
        Baseband data type, either ``complex`` or ``real``.
        Defaults to ``complex``.
    :param list[dict] channels:
        A list of dictionaries defining the properties of receiver channels,
        where each dictionary contains the following keys:

        - **location** (*numpy.ndarray*):
          3D location of the channel relative to the radar's position [x, y, z] in meters.
        - **polarization** (*numpy.ndarray*):
          Antenna polarization vector [x, y, z].
          Defaults to ``[0, 0, 1]`` (vertical polarization).
          Examples:

            - Vertical polarization: ``[0, 0, 1]``
            - Horizontal polarization: ``[0, 1, 0]``
            - Right-handed circular polarization: ``[0, 1, 1j]``
            - Left-handed circular polarization: ``[0, 1, -1j]``

        - **azimuth_angle** (*numpy.ndarray*):
          Azimuth pattern angles in degrees.
          Defaults to ``[-90, 90]``.
        - **azimuth_pattern** (*numpy.ndarray*):
          Azimuth pattern in decibels (dB).
          Defaults to ``[0, 0]``.
        - **elevation_angle** (*numpy.ndarray*):
          Elevation pattern angles in degrees.
          Defaults to ``[-90, 90]``.
        - **elevation_pattern** (*numpy.ndarray*):
          Elevation pattern in decibels (dB).
          Defaults to ``[0, 0]``.

    :ivar dict rf_prop:
        RF properties of the receiver:

        - **rf_gain** (*float*): RF gain in decibels (dB).
        - **noise_figure** (*float*): Noise figure in decibels (dB).

    :ivar dict bb_prop:
        Baseband properties of the receiver:

        - **fs** (*float*): Sampling rate in samples per second (sps).
        - **load_resistor** (*float*): Load resistance in ohms (Ω).
        - **baseband_gain** (*float*): Baseband gain in decibels (dB).
        - **bb_type** (*str*): Baseband data type, either ``real`` or ``complex``.

    :ivar dict rxchannel_prop:
        Properties of the receiver channels:

        - **size** (*int*): Number of receiver channels.
        - **locations** (*numpy.ndarray*):
          3D locations of the receiver channels [x, y, z] in meters.
        - **polarization** (*numpy.ndarray*): Polarization vectors of the receiver channels.
        - **az_angles** (*numpy.ndarray*): Azimuth angles in degrees.
        - **az_patterns** (*numpy.ndarray*): Azimuth pattern in decibels (dB).
        - **el_angles** (*numpy.ndarray*): Elevation angles in degrees.
        - **el_patterns** (*numpy.ndarray*): Elevation pattern in decibels (dB).
        - **antenna_gains** (*numpy.ndarray*):
          Antenna gains of the receiver channels in decibels (dB).

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
        fs: float,
        noise_figure: float = 10,
        rf_gain: float = 0,
        load_resistor: float = 500,
        baseband_gain: float = 0,
        bb_type: str = "complex",
        channels: Optional[List[Dict]] = None,
    ):
        # Input validation
        if fs <= 0:
            raise ValueError("Sampling rate (fs) must be positive")
        if not isinstance(noise_figure, (int, float)):
            raise ValueError("noise_figure must be a number")
        if not isinstance(rf_gain, (int, float)):
            raise ValueError("rf_gain must be a number")
        if load_resistor <= 0:
            raise ValueError("load_resistor must be positive")
        if not isinstance(baseband_gain, (int, float)):
            raise ValueError("baseband_gain must be a number")
        if bb_type not in VALID_BB_TYPES:
            raise ValueError(
                f"Invalid baseband type '{bb_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_BB_TYPES))}"
            )

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

    @staticmethod
    def _validate_array_lengths(
        arr1: NDArray,
        arr2: NDArray,
        name1: str,
        name2: str,
        channel_idx: Optional[int] = None,
    ) -> None:
        """Helper method to validate that two arrays have the same length."""
        if len(arr1) != len(arr2):
            channel_info = (
                f" for channel {channel_idx}" if channel_idx is not None else ""
            )
            raise ValueError(
                f"Length mismatch{channel_info}: {name1} ({len(arr1)}) "
                f"and {name2} ({len(arr2)}) must have same length"
            )

    def validate_bb_prop(self, bb_prop: Dict) -> None:
        """
        Validate baseband properties

        :param dict bb_prop: Baseband properties

        :raises ValueError: Invalid baseband type
        :raises ValueError: Invalid sampling rate
        :raises ValueError: Invalid load resistor value
        """
        if bb_prop["bb_type"] not in VALID_BB_TYPES:
            raise ValueError(
                f"Invalid baseband type '{bb_prop['bb_type']}'. "
                f"Must be one of: {', '.join(sorted(VALID_BB_TYPES))}"
            )

        if bb_prop["fs"] <= 0:
            raise ValueError("Sampling rate (fs) must be positive")

        if bb_prop["load_resistor"] <= 0:
            raise ValueError("Load resistor must be positive")

    def process_rxchannel_prop(self, channels: List[Dict]) -> Dict:
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
                rx_element.get("polarization", DEFAULT_POLARIZATION)
            )

            # azimuth pattern
            az_angle = np.array(rx_element.get("azimuth_angle", DEFAULT_AZIMUTH_RANGE))
            az_pattern = np.array(rx_element.get("azimuth_pattern", DEFAULT_PATTERN_DB))
            self._validate_array_lengths(
                az_angle, az_pattern, "azimuth_angle", "azimuth_pattern", rx_idx
            )

            rxch_prop["antenna_gains"][rx_idx] = np.max(az_pattern)
            az_pattern = az_pattern - rxch_prop["antenna_gains"][rx_idx]

            rxch_prop["az_angles"].append(az_angle)
            rxch_prop["az_patterns"].append(az_pattern)

            # elevation pattern
            el_angle = np.array(
                rx_element.get("elevation_angle", DEFAULT_ELEVATION_RANGE)
            )
            el_pattern = np.array(
                rx_element.get("elevation_pattern", DEFAULT_PATTERN_DB)
            )
            self._validate_array_lengths(
                el_angle, el_pattern, "elevation_angle", "elevation_pattern", rx_idx
            )
            el_pattern = el_pattern - np.max(el_pattern)

            rxch_prop["el_angles"].append(el_angle)
            rxch_prop["el_patterns"].append(el_pattern)

        return rxch_prop

    @property
    def sampling_rate(self) -> float:
        """Get the sampling rate."""
        return self.bb_prop["fs"]

    @property
    def noise_bandwidth(self) -> float:
        """Get the noise bandwidth."""
        return self.bb_prop["noise_bandwidth"]

    @property
    def num_channels(self) -> int:
        """Get the number of receiver channels."""
        return self.rxchannel_prop["size"]

    @property
    def channel_locations(self) -> NDArray:
        """Get the 3D locations of receiver channels."""
        return self.rxchannel_prop["locations"]

    def get_channel_info(self, channel_idx: int) -> Dict[str, NDArray]:
        """
        Get comprehensive information about a specific channel.

        :param int channel_idx: Index of the channel (0-based)
        :return: Dictionary containing channel information
        :rtype: dict
        :raises IndexError: If channel_idx is out of range
        """
        if not 0 <= channel_idx < self.num_channels:
            raise IndexError(
                f"Channel index {channel_idx} out of range [0, {self.num_channels-1}]"
            )

        return {
            "location": self.rxchannel_prop["locations"][channel_idx],
            "polarization": self.rxchannel_prop["polarization"][channel_idx],
            "antenna_gain": self.rxchannel_prop["antenna_gains"][channel_idx],
            "azimuth_angles": self.rxchannel_prop["az_angles"][channel_idx],
            "azimuth_pattern": self.rxchannel_prop["az_patterns"][channel_idx],
            "elevation_angles": self.rxchannel_prop["el_angles"][channel_idx],
            "elevation_pattern": self.rxchannel_prop["el_patterns"][channel_idx],
        }

    def __str__(self) -> str:
        """String representation of the Receiver."""
        return (
            f"Receiver(channels={self.num_channels}, "
            f"fs={self.sampling_rate/1e6:.1f} MHz, "
            f"noise_figure={self.rf_prop['noise_figure']} dB, "
            f"bb_type={self.bb_prop['bb_type']})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the Receiver."""
        return (
            f"Receiver(fs={self.sampling_rate}, "
            f"noise_figure={self.rf_prop['noise_figure']}, "
            f"rf_gain={self.rf_prop['rf_gain']}, "
            f"load_resistor={self.bb_prop['load_resistor']}, "
            f"baseband_gain={self.bb_prop['baseband_gain']}, "
            f"bb_type='{self.bb_prop['bb_type']}', "
            f"channels={self.num_channels})"
        )

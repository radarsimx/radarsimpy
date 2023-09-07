#!python
# distutils: language = c++

# Simulate the baseband output of a radar system with defined targets

# ----------
# RadarSimPy - A Radar Simulator Built with Python
# Copyright (C) 2018 - PRESENT  Zhengyu Peng
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
import warnings


def cal_locations(radar, target):
    """
    Calculate target location versus timestamp

    :param Radar radar:
        Refer to `radar` parameter in `run_simulator`
    :param dict target:
        Refer to `target` parameter in `run_simulator`

    :return:
        {

        - **tgx_t** (*numpy.3darray*) --
            Target location versus time in x axis,
            ``[channels, pulses, adc_samples]``
        - **tgy_t** (*numpy.3darray*) --
            Target location versus time in y axis,
            ``[channels, pulses, adc_samples]``
        - **tgz_t** (*numpy.3darray*) --
            Target location versus time in z axis,
            ``[channels, pulses, adc_samples]``
        - **rcs_t** (*numpy.3darray*) --
            Target RCS versus time,
            ``[channels, pulses, adc_samples]``
        - **phs_t** (*numpy.3darray*) --
            Target phase versus time,
            ``[channels, pulses, adc_samples]``

        }
    :rtype: dict
    """

    timestamp = radar.timestamp
    # Target locations versus time
    matrix_shape = np.shape(timestamp)
    tgx_t = np.zeros(matrix_shape)
    tgy_t = np.zeros(matrix_shape)
    tgz_t = np.zeros(matrix_shape)
    rcs_t = np.zeros(matrix_shape)
    phs_t = np.zeros(matrix_shape)

    location = target["location"]
    speed = target.get("speed", (0, 0, 0))
    rcs = target["rcs"]
    phase = target.get("phase", 0)

    if type(location[0]) == str:
        temp_x = eval(location[0].replace("math", "np").replace("t", "timestamp"))
        tgx_t = temp_x + speed[0] * timestamp
    else:
        tgx_t = location[0] + speed[0] * timestamp

    if type(location[1]) == str:
        temp_y = eval(location[1].replace("math", "np").replace("t", "timestamp"))
        tgy_t = temp_y + speed[1] * timestamp
    else:
        tgy_t = location[1] + speed[1] * timestamp

    if type(location[2]) == str:
        temp_z = eval(location[2].replace("math", "np").replace("t", "timestamp"))
        tgz_t = temp_z + speed[2] * timestamp
    else:
        tgz_t = location[2] + speed[2] * timestamp

    if type(rcs) == str:
        rcs_t = eval(rcs.replace("math", "np").replace("t", "timestamp"))
    else:
        rcs_t = np.full_like(timestamp, rcs)

    if type(phase) == str:
        phs_t = eval(phase.replace("math", "np").replace("t", "timestamp"))
    else:
        phs_t = np.full_like(timestamp, phase)

    return {
        "tgx_t": tgx_t,
        "tgy_t": tgy_t,
        "tgz_t": tgz_t,
        "rcs_t": rcs_t,
        "phs_t": phs_t,
    }


def cal_range_angle(radar, target):
    """
    Calculate relative ranges and angles between target and radar

    :param Radar radar:
        Refer to `radar` parameter in `run_simulator`
    :param dict target:
        Refer to `target` parameter in `run_simulator`

    :return:
        {

        - **rcs_t** (*numpy.3darray*) --
            Target RCS versus time,
            ``[channels, pulses, adc_samples]``
        - **phs_t** (*numpy.3darray*) --
            Target phase versus time,
            ``[channels, pulses, adc_samples]``
        - **tg_to_tx_rng** (*numpy.3darray*) --
            Range between target and radar transmitter,
            ``[channels, pulses, adc_samples]``
        - **tg_to_rx_rng** (*numpy.3darray*) --
            Range between target and radar receiver,
            ``[channels, pulses, adc_samples]``
        - **tg_to_tx_azi** (*numpy.3darray*) --
            Azimuth angle between target and radar transmitter,
            ``[channels, pulses, adc_samples]``
        - **tg_to_rx_azi** (*numpy.3darray*) --
            Azimuth angle between target and radar receiver,
            ``[channels, pulses, adc_samples]``
        - **tg_to_tx_ele** (*numpy.3darray*) --
            Elevation angle between target and radar transmitter,
            ``[channels, pulses, adc_samples]``
        - **tg_to_rx_ele** (*numpy.3darray*) --
            Elevation angle between target and radar receiver,
            ``[channels, pulses, adc_samples]``

        }
    :rtype: dict
    """

    target_locations = cal_locations(radar, target)
    tgx_t = target_locations["tgx_t"]
    tgy_t = target_locations["tgy_t"]
    tgz_t = target_locations["tgz_t"]

    matrix_shape = np.shape(tgx_t)

    # Target ranges versus time
    tx_tgx_t = np.zeros(matrix_shape)
    rx_tgx_t = np.zeros(matrix_shape)
    tx_tgy_t = np.zeros(matrix_shape)
    rx_tgy_t = np.zeros(matrix_shape)
    tx_tgz_t = np.zeros(matrix_shape)
    rx_tgz_t = np.zeros(matrix_shape)

    for i_tx in range(0, radar.transmitter.channel_size):
        for i_rx in range(0, radar.receiver.channel_size):
            ch = i_tx * radar.receiver.channel_size + i_rx

            tx_tgx_t[ch, :, :] = radar.transmitter.locations[i_tx, 0]
            rx_tgx_t[ch, :, :] = radar.receiver.locations[i_rx, 0]
            tx_tgy_t[ch, :, :] = radar.transmitter.locations[i_tx, 1]
            rx_tgy_t[ch, :, :] = radar.receiver.locations[i_rx, 1]
            tx_tgz_t[ch, :, :] = radar.transmitter.locations[i_tx, 2]
            rx_tgz_t[ch, :, :] = radar.receiver.locations[i_rx, 2]

    tg_to_tx_rng = np.sqrt(
        (tgx_t - tx_tgx_t) ** 2 + (tgy_t - tx_tgy_t) ** 2 + (tgz_t - tx_tgz_t) ** 2
    )
    tg_to_rx_rng = np.sqrt(
        (tgx_t - rx_tgx_t) ** 2 + (tgy_t - rx_tgy_t) ** 2 + (tgz_t - rx_tgz_t) ** 2
    )
    tg_to_tx_azi = np.arctan2((tgy_t - tx_tgy_t), (tgx_t - tx_tgx_t))
    tg_to_rx_azi = np.arctan2((tgy_t - rx_tgy_t), (tgx_t - rx_tgx_t))
    tg_to_tx_ele = np.arctan2(
        (tgz_t - tx_tgz_t),
        np.sqrt((tgx_t - tx_tgx_t) ** 2 + (tgy_t - tx_tgy_t) ** 2),
    )
    tg_to_rx_ele = np.arctan2(
        (tgz_t - rx_tgz_t),
        np.sqrt((tgx_t - rx_tgx_t) ** 2 + (tgy_t - rx_tgy_t) ** 2),
    )

    return {
        "rcs_t": target_locations["rcs_t"],
        "phs_t": target_locations["phs_t"],
        "tg_to_tx_rng": tg_to_tx_rng,
        "tg_to_rx_rng": tg_to_rx_rng,
        "tg_to_tx_azi": tg_to_tx_azi,
        "tg_to_rx_azi": tg_to_rx_azi,
        "tg_to_tx_ele": tg_to_tx_ele,
        "tg_to_rx_ele": tg_to_rx_ele,
    }


def sim_target(radar, target):
    """
    Simulate target

    :param Radar radar:
        Refer to `radar` parameter in `run_simulator`
    :param dict target:
        Refer to `target` parameter in `run_simulator`

    :return:
        {

        - **amplitude** (*numpy.3darray*) --
            Signal peak amplitudes,
            ``[channels, pulses, adc_samples]``
        - **phs_t** (*numpy.3darray*) --
            Target phase versus time,
            ``[channels, pulses, adc_samples]``
        - **delta_t** (*numpy.3darray*) --
            Round trip delay,
            ``[channels, pulses, adc_samples]``

        }
    :rtype: dict
    """

    fc_matrix = radar.fc_mat

    range_angle = cal_range_angle(radar, target)
    antenna_gain = np.zeros_like(radar.timestamp)

    for i_tx in range(0, radar.transmitter.channel_size):
        for i_rx in range(0, radar.receiver.channel_size):
            ch = i_tx * radar.receiver.channel_size + i_rx

            tx_gain_azi = radar.transmitter.az_func[i_tx](
                180 / np.pi * range_angle["tg_to_tx_azi"][ch, :, :]
            )

            tx_gain_ele = (
                radar.transmitter.el_func[i_tx](
                    180 / np.pi * range_angle["tg_to_tx_ele"][ch, :, :]
                )
                + radar.transmitter.antenna_gains[i_tx]
            )

            rx_gain_azi = radar.receiver.az_func[i_rx](
                180 / np.pi * range_angle["tg_to_rx_azi"][ch, :, :]
            )

            rx_gain_ele = (
                radar.receiver.el_func[i_rx](
                    180 / np.pi * range_angle["tg_to_rx_ele"][ch, :, :]
                )
                + radar.receiver.antenna_gains[i_rx]
            )

            antenna_gain[ch, :, :] = (
                tx_gain_azi + tx_gain_ele + rx_gain_azi + rx_gain_ele
            )

    pr_dbm = (
        radar.transmitter.tx_power
        + antenna_gain
        - 10 * np.log10(4 * np.pi * range_angle["tg_to_tx_rng"] ** 2)
        + range_angle["rcs_t"]
        - 10 * np.log10(4 * np.pi * range_angle["tg_to_rx_rng"] ** 2)
        + 10 * np.log10((const.c / fc_matrix) ** 2 / (4 * np.pi))
        + radar.receiver.rf_gain
    )
    pr_watts = 1e-3 * 10 ** (pr_dbm / 10)
    amp_mixer = np.sqrt(pr_watts * radar.receiver.load_resistor)
    peak_amp = amp_mixer * 10 ** (radar.receiver.baseband_gain / 20) * np.sqrt(2)

    delta_t = (range_angle["tg_to_tx_rng"] + range_angle["tg_to_rx_rng"]) / const.c

    phs_t = range_angle["phs_t"]

    return {
        "amplitude": peak_amp,
        "phs_t": phs_t,
        "delta_t": delta_t,
    }


def simpy(radar, targets, noise=True):
    """
    simpy(radar, targets, noise=True) ``deprecated``

    Simulate baseband response of a radar with defined targets. Python engine.

    :param Radar radar:
        Radar model
    :param list[dict] targets:
        Ideal point target list

        [{

        - **location** (*numpy.1darray*) --
            Location of the target (m), [x, y, z]
        - **rcs** (*float*) --
            Target RCS (dBsm)
        - **speed** (*numpy.1darray*) --
            Speed of the target (m/s), [vx, vy, vz]. ``default
            [0, 0, 0]``
        - **phase** (*float*) --
            Target phase (deg). ``default 0``

        }]

        *Note*: Target's parameters can be specified with
        ``Radar.timestamp`` to customize the time varying property.
        Ex: ``location=(1e-3*np.sin(2*np.pi*1*radar.timestamp), 0, 0)``
    :param bool noise:
        Flag to enable noise calculation. ``default True``

    :return:
        {

        - **baseband** (*numpy.3darray*) --
            Time domain complex (I/Q) baseband data.
            ``[channes/frames, pulses, samples]``

            *Channel/frame order in baseband*

            *[0]* ``Frame[0] -- Tx[0] -- Rx[0]``

            *[1]* ``Frame[0] -- Tx[0] -- Rx[1]``

            ...

            *[N]* ``Frame[0] -- Tx[1] -- Rx[0]``

            *[N+1]* ``Frame[0] -- Tx[1] -- Rx[1]``

            ...

            *[M]* ``Frame[1] -- Tx[0] -- Rx[0]``

            *[M+1]* ``Frame[1] -- Tx[0] -- Rx[1]``

        - **timestamp** (*numpy.3darray*) --
            Refer to Radar.timestamp

        }
    :rtype: dict
    """

    warnings.warn(
        "`simpy` has been deprecated, please use `simc` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # noise = radar.noise
    channels = radar.channel_size
    pulses = radar.transmitter.pulses
    samples = radar.samples_per_pulse
    # slope = radar.transmitter.slope
    # fc_matrix = radar.fc_mat
    beat_time = radar.beat_time

    f = radar.f
    f_offset = radar.transmitter.f_offset
    k = radar.k
    d_pt = radar.delta_t

    p0 = np.zeros_like(beat_time)
    p1 = np.zeros_like(beat_time)

    phase_noise = radar.phase_noise
    pn_mod = np.zeros_like(radar.phase_noise)

    target_number = len(targets)

    baseband = np.zeros(np.shape(beat_time), dtype=complex)
    for n in range(0, target_number):
        raw_data = sim_target(radar, targets[n])
        delta_t = raw_data["delta_t"]

        if radar.phase_noise is not None:
            shift_mean = np.round(np.mean(delta_t, axis=2) * radar.receiver.fs)
            shift_mean = shift_mean.astype(int)

            for f_idx in range(0, channels):
                for p_idx in range(0, pulses):
                    pn_mod[f_idx, p_idx, :] = np.roll(
                        phase_noise[f_idx, p_idx, :], shift_mean[f_idx, p_idx]
                    ) * np.conjugate(phase_noise[f_idx, p_idx, :])
        else:
            pn_mod = 1

        # if radar.transmitter.modulation == 'pulse':
        phase = np.ones(
            [
                channels,
                pulses,
                samples,
            ],
            dtype=complex,
        )

        """ use vector method to calcuate phase, seems take longer time
        temp = np.arange(
            0, samples, 1) / radar.receiver.fs

        temp_time = np.expand_dims(
            np.expand_dims(temp, axis=0),
            axis=0)
        temp_time = np.tile(
            temp_time, (
                channels, pulses, 1
            )) - delta_t[n, :, :, :]

        phase_timestamp = np.tile(
            np.expand_dims(
                radar.code_timestamp, axis=1
            ), (1, pulses, 1))

        for i_code in range(0, np.shape(phase_timestamp)[2]):
            temp_phase_timestamp = np.tile(
                phase_timestamp[:, :, i_code, np.newaxis],
                (1, 1, samples))
            addr = np.where(temp_time > temp_phase_timestamp)
            new_addr = (addr[0], np.ones_like(
                addr[0], dtype='int64')*i_code)
            phase[addr] = radar.pulse_phs[new_addr]
        """

        for i_ch in range(0, channels):
            tx_idx = int(i_ch / radar.receiver.channel_size)
            if radar.transmitter.waveform_mod[tx_idx]["enabled"]:
                for i_pulse in range(0, pulses):
                    temp_time = (
                        np.arange(0, samples) / radar.receiver.fs
                        - delta_t[i_ch, i_pulse, :]
                    )
                    phase_timestamp = radar.transmitter.waveform_mod[tx_idx]["t"]
                    for i_code in range(0, np.shape(phase_timestamp)[0]):
                        phase[
                            i_ch, i_pulse, np.where(temp_time > phase_timestamp[i_code])
                        ] = radar.transmitter.waveform_mod[tx_idx]["var"][i_code]

        r"""
        Core equation to calculate baseband data $s_b(t)$:
        $$
        s_b(t) = A \exp{\left( 2 \pi j k t_d t + 2 \pi j f_c t_d +
            2 \pi j k t_d^2 + j \phi \right)}
            \exp{\left( j \Phi(t) \right)}
        $$

        where $A$ is amplitude, $k=\frac{B}{T_c}$ is slope, $B$ is
        bandwidth, $T_c$ is pulse lenght, $t_d$ is round-trip delay,
        $f_c$ is center frequency, $\phi$ is target phase, $\Phi(t)$
        is phase modulation sequence.
        """
        # baseband = baseband + pn_mod*raw_data['amplitude'] * np.exp(
        #     1j * (
        #         2 * np.pi * (slope * delta_t * beat_time +
        #                      fc_matrix * delta_t -
        #                      slope * (delta_t**2)/2) +
        #         raw_data['phs_t'] / 180 * np.pi
        #     )) * phase

        if len(k) == 2:
            baseband = (
                baseband
                + pn_mod
                * raw_data["amplitude"]
                * np.exp(
                    1j
                    * (
                        2
                        * np.pi
                        * (
                            (radar.f_offset_mat + f[0] + 0.5 * k[1] * beat_time)
                            * beat_time
                            - (
                                radar.f_offset_mat
                                + f[0]
                                + 0.5 * k[1] * (beat_time - delta_t)
                            )
                            * (beat_time - delta_t)
                        )
                        + raw_data["phs_t"] / 180 * np.pi
                    )
                )
                * phase
            )
        else:
            for p_idx in range(0, radar.transmitter.pulses):
                freq_t = (f_offset[p_idx] + f[0:-1] + 0.5 * k[1:] * (d_pt[1:])) * (
                    d_pt[1:]
                )
                phase_t = np.cumsum(np.concatenate(([0], freq_t)))

                # Can't use linear interp here
                fun_phase_t = interp1d(
                    radar.t,
                    phase_t,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(phase_t[0], phase_t[-1]),
                )

                p0[:, p_idx, :] = fun_phase_t(beat_time[:, p_idx, :])
                p1[:, p_idx, :] = fun_phase_t(
                    beat_time[:, p_idx, :] - delta_t[:, p_idx, :]
                )

            baseband = (
                baseband
                + pn_mod
                * raw_data["amplitude"]
                * np.exp(1j * (2 * np.pi * (p0 - p1) + raw_data["phs_t"] / 180 * np.pi))
                * phase
            )

    if noise:
        baseband = baseband + radar.noise * (
            np.random.randn(
                channels,
                pulses,
                samples,
            )
            + 1j
            * np.random.randn(
                channels,
                pulses,
                samples,
            )
        )

    # if radar.transmitter.modulation == 'frame':
    baseband = baseband * np.expand_dims(radar.pulse_phs, axis=2)

    return {"baseband": baseband, "timestamp": radar.timestamp}

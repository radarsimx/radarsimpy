"""
System level tests for range-gated deramp (stretch) processing

Covers the long-range FMCW/stretch case: an X-band radar imaging targets at
~111 km, where the round-trip delay is ~15 chirp lengths and the un-gated beat
tone (4.45 GHz) is far past Nyquist. With a range gate the target deramps to a
recoverable beat about the gate.

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

import warnings

import numpy as np
import numpy.testing as npt
import pytest
import scipy.constants as const

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module

# X-band stretch radar imaging ships at 60 nmi
CARRIER = 9e9
BANDWIDTH = 300e6
CHIRP_LENGTH = 50e-6
CHIRP_SLOPE = BANDWIDTH / CHIRP_LENGTH  # 6e12 Hz/s
FS = 40e6
GATE_RANGE = 111.12e3  # 60 nautical miles
GATE_DELAY = 2 * GATE_RANGE / const.c


def _build_radar(gate_delay=GATE_DELAY, pulses=1, prp=None, speed=(0, 0, 0)):
    """Build the reference X-band stretch radar."""
    tx = Transmitter(
        f=[CARRIER - BANDWIDTH / 2, CARRIER + BANDWIDTH / 2],
        t=CHIRP_LENGTH,
        tx_power=40,
        prp=prp,
        pulses=pulses,
        channels=[{"location": (0, 0, 0)}],
    )
    rx = Receiver(
        fs=FS,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        gate_delay=gate_delay,
        channels=[{"location": (0, 0, 0)}],
    )
    return Radar(transmitter=tx, receiver=rx, speed=speed)


def _range_profile(baseband):
    """Range FFT magnitude of the first channel/pulse."""
    return np.abs(np.fft.fft(baseband[0, 0, :]))


def _beat_bin_to_range(bin_idx, num_samples, gate_range):
    """Convert a range-FFT bin to absolute range."""
    freqs = np.fft.fftfreq(num_samples, d=1 / FS)
    return gate_range + freqs[bin_idx] * const.c / (2 * CHIRP_SLOPE)


def test_gate_delay_defaults_to_zero():
    """An unspecified gate_delay leaves the receiver un-gated."""
    rx = Receiver(fs=FS)
    assert rx.gate_delay == 0.0
    assert rx.gate_range == 0.0
    assert rx.bb_prop["gate_delay"] == 0.0


def test_gate_range_round_trip():
    """gate_range is the range that produces a DC beat."""
    rx = Receiver(fs=FS, gate_delay=GATE_DELAY)
    npt.assert_allclose(rx.gate_range, GATE_RANGE, rtol=1e-12)


def test_negative_gate_delay_rejected():
    """A negative gate delay is not physical."""
    with pytest.raises(ValueError, match="non-negative"):
        Receiver(fs=FS, gate_delay=-1e-6)


def test_timestamp_starts_at_gate():
    """The receive window opens at the gate, so sampling starts there."""
    radar = _build_radar()
    timestamp = radar.time_prop["timestamp"]

    npt.assert_allclose(timestamp[0, 0, 0], GATE_DELAY, rtol=1e-12)
    # Sample spacing is unaffected by the gate
    npt.assert_allclose(timestamp[0, 0, 1] - timestamp[0, 0, 0], 1 / FS, rtol=1e-9)


def test_unambiguous_range_window_gated():
    """Gated, the signed residual gives a two-sided +/-500 m window."""
    radar = _build_radar()
    npt.assert_allclose(radar.chirp_slope, CHIRP_SLOPE, rtol=1e-12)
    npt.assert_allclose(radar.unambiguous_range_span, 999.3, rtol=1e-3)

    low, high = radar.unambiguous_range_window
    npt.assert_allclose(low, GATE_RANGE - 499.65, rtol=1e-3)
    npt.assert_allclose(high, GATE_RANGE + 499.65, rtol=1e-3)


def test_unambiguous_range_window_ungated():
    """
    Un-gated, every beat is positive so the whole [0, fs) band is usable.

    The window is one-sided [0, span] -- twice the half-width a two-sided
    reading would give, which is why ordinary short-range FMCW configurations
    are not flagged.
    """
    radar = _build_radar(gate_delay=0.0)
    low, high = radar.unambiguous_range_window
    assert low == 0.0
    npt.assert_allclose(high, 999.3, rtol=1e-3)


def test_target_at_gate_produces_dc():
    """
    A target sitting exactly on the gate deramps to DC.

    This is the defining property of a gated deramp receiver, and the thing that
    is impossible without one at this range.
    """
    radar = _build_radar()
    target = {"location": (GATE_RANGE, 0, 0), "rcs": 30}

    result = sim_radar(radar, [target], device="cpu")
    profile = _range_profile(result["baseband"])

    assert np.argmax(profile) == 0


def test_target_offset_from_gate_lands_in_correct_bin():
    """
    A target 100 m beyond the gate beats at slope*2*100/c = 4 MHz.

    At fs = 40 MHz over 2000 samples that is bin 200.
    """
    offset = 100.0
    radar = _build_radar()
    target = {"location": (GATE_RANGE + offset, 0, 0), "rcs": 30}

    result = sim_radar(radar, [target], device="cpu")
    profile = _range_profile(result["baseband"])
    peak_bin = int(np.argmax(profile))

    expected_beat = CHIRP_SLOPE * 2 * offset / const.c
    expected_bin = int(round(expected_beat * profile.size / FS))
    assert peak_bin == expected_bin

    recovered = _beat_bin_to_range(peak_bin, profile.size, GATE_RANGE)
    npt.assert_allclose(recovered, GATE_RANGE + offset, atol=1.0)


def test_target_inside_gate_mirrors():
    """A target 100 m short of the gate produces the mirrored (negative) bin."""
    offset = 100.0
    radar = _build_radar()

    beyond = sim_radar(radar, [{"location": (GATE_RANGE + offset, 0, 0), "rcs": 30}],
                       device="cpu")
    inside = sim_radar(radar, [{"location": (GATE_RANGE - offset, 0, 0), "rcs": 30}],
                       device="cpu")

    beyond_bin = int(np.argmax(_range_profile(beyond["baseband"])))
    inside_bin = int(np.argmax(_range_profile(inside["baseband"])))
    num_samples = beyond["baseband"].shape[2]

    # Negative beat frequencies live in the upper half of the FFT
    assert beyond_bin == num_samples - inside_bin


def test_amplitude_reflects_true_range_not_gate_offset():
    """
    The gate must not change the radar equation.

    A target at the gate is 111 km away and must be attenuated as such, even
    though it deramps to DC. Compare against the same target seen by a radar
    gated 100 m short of it: both are ~111 km away, so their echo amplitudes
    must match to well within a dB.
    """
    target_range = GATE_RANGE
    on_gate = _build_radar(gate_delay=GATE_DELAY)
    off_gate = _build_radar(gate_delay=2 * (GATE_RANGE - 100.0) / const.c)
    target = {"location": (target_range, 0, 0), "rcs": 30}

    peak_on = np.max(_range_profile(sim_radar(on_gate, [target], device="cpu")["baseband"]))
    peak_off = np.max(_range_profile(sim_radar(off_gate, [target], device="cpu")["baseband"]))

    ratio_db = 20 * np.log10(peak_on / peak_off)
    assert abs(ratio_db) < 0.5


def test_ungated_long_range_target_aliases():
    """
    Without a gate the same target is unrecoverable.

    This is the bug the feature exists to fix: the beat tone is 4.45 GHz against
    a 40 MHz sampling rate, so the peak lands in an essentially arbitrary bin
    rather than the true one.
    """
    radar = _build_radar(gate_delay=0.0)
    target = {"location": (GATE_RANGE, 0, 0), "rcs": 30}

    with pytest.warns(RuntimeWarning, match="alias"):
        result = sim_radar(radar, [target], device="cpu")

    profile = _range_profile(result["baseband"])
    # The un-gated swath is +/-500 m about zero range, so a 111 km target
    # cannot land anywhere meaningful.
    recovered = _beat_bin_to_range(int(np.argmax(profile)), profile.size, 0.0)
    assert abs(recovered - GATE_RANGE) > 1e3


def test_ordinary_short_range_fmcw_not_flagged():
    """
    A conventional un-gated short-range FMCW setup must not warn.

    Regression test: a two-sided +/-fs/2 reading of the swath would flag this
    perfectly ordinary configuration, because it ignores that un-gated beats are
    always positive and so span the full [0, fs) band. Mirrors the geometry in
    test_system_fmcw_radar.py.
    """
    tx = Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=4,
    )
    rx = Receiver(fs=2e6, noise_figure=12, rf_gain=20, baseband_gain=30)
    radar = Radar(transmitter=tx, receiver=rx)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim_radar(radar, [{"location": (200, 0, 0), "rcs": 20}], device="cpu")

    aliasing = [w for w in caught if "alias" in str(w.message)]
    assert not aliasing, f"ordinary FMCW config wrongly flagged: {aliasing}"


def test_no_warning_when_gated_correctly():
    """A target inside the gated swath must not warn."""
    radar = _build_radar()
    target = {"location": (GATE_RANGE + 100.0, 0, 0), "rcs": 30}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim_radar(radar, [target], device="cpu")

    aliasing = [w for w in caught if "alias" in str(w.message)]
    assert not aliasing, f"unexpected aliasing warning: {aliasing}"


# Ray tracing at 111 km is prohibitively expensive (ray count grows with the
# arc subtended by a grid cell at the target). The gate mechanism is
# range-independent, so the mesh path is exercised at short range instead.
MESH_TARGET_RANGE = 30.0
MESH_GATE_RANGE = 10.0


def test_mesh_gate_shifts_peak_by_predicted_bins():
    """
    Gating the mesh path moves the peak by exactly the predicted bin count.

    Exercises simulator_mesh.cpp together with Snapshot::time_, which carry the
    gate differently from the point simulator: the snapshot holds the gate and
    mesh move_time is elapsed-since-snapshot, so the gate cancels there. A sign
    error or a double count in either shows up directly as a wrong shift.
    """
    target = {"model": "./models/plate5x5.stl", "location": (MESH_TARGET_RANGE, 0, 0)}

    ungated = sim_radar(_build_radar(gate_delay=0.0), [target], density=0.4,
                        device="cpu")
    gated = sim_radar(
        _build_radar(gate_delay=2 * MESH_GATE_RANGE / const.c),
        [target],
        density=0.4,
        device="cpu",
    )

    ungated_bin = int(np.argmax(_range_profile(ungated["baseband"])))
    gated_bin = int(np.argmax(_range_profile(gated["baseband"])))
    num_samples = ungated["baseband"].shape[2]

    expected_shift = int(
        round(CHIRP_SLOPE * 2 * MESH_GATE_RANGE / const.c * num_samples / FS)
    )
    assert ungated_bin - gated_bin == expected_shift, (
        f"ungated bin {ungated_bin}, gated bin {gated_bin}, "
        f"expected shift {expected_shift}"
    )


def test_mesh_agrees_with_point_under_gate():
    """A gated mesh target lands in the same range bin as an equivalent point."""
    gate_delay = 2 * MESH_GATE_RANGE / const.c
    radar = _build_radar(gate_delay=gate_delay)
    location = (MESH_TARGET_RANGE, 0, 0)

    point = sim_radar(radar, [{"location": location, "rcs": 20}], device="cpu")
    mesh = sim_radar(
        radar,
        [{"model": "./models/plate5x5.stl", "location": location}],
        density=0.4,
        device="cpu",
    )

    point_bin = int(np.argmax(_range_profile(point["baseband"])))
    mesh_bin = int(np.argmax(_range_profile(mesh["baseband"])))

    # The plate has extent, so allow a couple of 0.5 m range bins of slack
    assert abs(mesh_bin - point_bin) <= 4, f"point {point_bin}, mesh {mesh_bin}"


def test_cpu_gpu_parity():
    """
    The gated path must agree between CPU and GPU.

    gate_delay_ is a double member on a Receiver<float> that gets memcpy'd
    wholesale to the device, so a layout or precision mismatch would show up
    here as a shifted peak. Skipped automatically on CPU-only builds, where
    device="gpu" falls back to CPU and the comparison is vacuous.
    """
    radar = _build_radar()
    targets = [{"location": (GATE_RANGE + 100.0, 0, 0), "rcs": 30}]

    cpu = sim_radar(radar, targets, device="cpu")["baseband"]
    gpu = sim_radar(radar, targets, device="gpu")["baseband"]

    cpu_bin = int(np.argmax(_range_profile(cpu)))
    gpu_bin = int(np.argmax(_range_profile(gpu)))
    assert cpu_bin == gpu_bin

    relative = np.max(np.abs(cpu - gpu)) / np.max(np.abs(cpu))
    assert relative < 1e-6, f"CPU/GPU relative difference {relative:.3e}"


def test_doppler_preserved_across_gate():
    """
    The gate is a constant time shift, so it must not perturb Doppler.

    A closing target seen through a gate must show the same pulse-to-pulse phase
    advance as the identical geometry simulated at short range with no gate.
    """
    velocity = -30.0  # m/s, closing
    pulses = 16
    prp = 200e-6
    # Comfortably inside the +/-499.65 m un-gated swath (500 m would sit
    # exactly at Nyquist).
    short_range = 300.0

    gated = _build_radar(pulses=pulses, prp=prp)
    gated_result = sim_radar(
        gated,
        [{"location": (GATE_RANGE, 0, 0), "speed": (velocity, 0, 0), "rcs": 30}],
        device="cpu",
    )

    ungated = _build_radar(gate_delay=0.0, pulses=pulses, prp=prp)
    ungated_result = sim_radar(
        ungated,
        [{"location": (short_range, 0, 0), "speed": (velocity, 0, 0), "rcs": 30}],
        device="cpu",
    )

    def doppler_bin(baseband):
        # Peak of the Doppler FFT at the target's range bin
        profile = np.abs(np.fft.fft(baseband[0, 0, :]))
        rbin = int(np.argmax(profile))
        return int(np.argmax(np.abs(np.fft.fft(baseband[0, :, rbin]))))

    assert doppler_bin(gated_result["baseband"]) == doppler_bin(
        ungated_result["baseband"]
    )

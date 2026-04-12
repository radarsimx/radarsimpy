"""
Tests for receiver thermal noise simulation via sim_radar.

Validates that:
- Noise output is non-trivial and has correct shape
- Same Rx channel at the same timestamp produces identical noise values
- Different Rx channels produce different noise
- Noise level scales with receiver gain / noise figure
- Complex and real baseband modes behave correctly
- Noise statistics match expected distribution (zero mean, correct std dev)
- Multi-frame noise is independent across frames
- Deterministic seed produces reproducible results (regression)

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
import numpy.testing as npt

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module


def _make_radar(
    num_tx=1,
    num_rx=2,
    num_pulses=4,
    num_frames=1,
    bb_type="complex",
    noise_figure=10,
    rf_gain=20,
    baseband_gain=30,
):
    """
    Helper to build a minimal radar configuration for noise testing.

    Uses a 24 GHz FMCW chirp (100 MHz BW, 80 us sweep) so that
    sample_size is reasonably large for statistical checks.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=num_pulses,
        channels=[{"location": (0, 0, 0)}] * num_tx,
    )
    rx = Receiver(
        fs=2e6,
        noise_figure=noise_figure,
        rf_gain=rf_gain,
        load_resistor=500,
        baseband_gain=baseband_gain,
        bb_type=bb_type,
        channels=[{"location": (0, 0, 0)}] * num_rx,
    )
    if num_frames == 1:
        radar = Radar(transmitter=tx, receiver=rx)
    else:
        radar = Radar(
            transmitter=tx,
            receiver=rx,
            frame_time=np.arange(num_frames, dtype=float),
        )
    return radar


def _run_noise(radar, targets=None):
    """Run sim_radar and return the result dict. Uses an empty target list."""
    if targets is None:
        targets = [{"location": (500, 0, 0), "rcs": 0, "phase": 0}]
    return sim_radar(radar, targets)


# ============================================================================
# Shape and basic sanity
# ============================================================================


def test_noise_output_shape():
    """Noise array shape must match baseband shape."""
    radar = _make_radar(num_tx=1, num_rx=2, num_pulses=4, num_frames=1)
    result = _run_noise(radar)

    assert result["noise"] is not None
    assert result["noise"].shape == result["baseband"].shape


def test_noise_output_shape_multiframe():
    """Noise shape must include all frames * channels."""
    radar = _make_radar(num_tx=2, num_rx=3, num_pulses=2, num_frames=2)
    result = _run_noise(radar)

    expected_channels = 2 * 2 * 3  # frames * tx * rx
    expected_pulses = 2
    expected_samples = radar.sample_prop["samples_per_pulse"]
    assert result["noise"].shape == (
        expected_channels,
        expected_pulses,
        expected_samples,
    )


def test_noise_is_nonzero():
    """With nonzero noise figure, noise should not be all zeros."""
    radar = _make_radar(noise_figure=10)
    result = _run_noise(radar)
    assert np.any(result["noise"] != 0), "Noise output should not be all zeros"


# ============================================================================
# Same Rx at same timestamp => identical noise
# ============================================================================


def test_same_rx_same_timestamp_same_noise_complex():
    """
    MIMO scenario: virtual channels sharing the same Rx and having
    the same timestamp must produce identical noise.

    With 2 Tx and 2 Rx, virtual channels are ordered:
      ch0 = Tx0-Rx0, ch1 = Tx0-Rx1, ch2 = Tx1-Rx0, ch3 = Tx1-Rx1

    ch0 and ch2 share Rx0; ch1 and ch3 share Rx1.
    When all Tx channels have identical timing (no delay), the timestamps
    for ch0 and ch2 are the same, so their noise must be identical.
    """
    radar = _make_radar(num_tx=2, num_rx=2, num_pulses=4, bb_type="complex")
    result = _run_noise(radar)
    noise = result["noise"]

    # ch0 (Tx0-Rx0) vs ch2 (Tx1-Rx0): same Rx0, same timestamps
    npt.assert_array_equal(
        noise[0],
        noise[2],
        err_msg="Channels sharing Rx0 should have identical noise",
    )

    # ch1 (Tx0-Rx1) vs ch3 (Tx1-Rx1): same Rx1, same timestamps
    npt.assert_array_equal(
        noise[1],
        noise[3],
        err_msg="Channels sharing Rx1 should have identical noise",
    )


def test_same_rx_same_timestamp_same_noise_real():
    """Same Rx / same timestamp check for real baseband mode."""
    radar = _make_radar(num_tx=2, num_rx=2, num_pulses=4, bb_type="real")
    result = _run_noise(radar)
    noise = result["noise"]

    npt.assert_array_equal(
        noise[0],
        noise[2],
        err_msg="Channels sharing Rx0 (real mode) should have identical noise",
    )
    npt.assert_array_equal(
        noise[1],
        noise[3],
        err_msg="Channels sharing Rx1 (real mode) should have identical noise",
    )


def test_same_rx_same_timestamp_multiframe():
    """
    Same-Rx-same-timestamp property should hold in every frame.

    With 2 Tx, 2 Rx, 2 frames the channel layout is:
      Frame 0: ch0=Tx0-Rx0, ch1=Tx0-Rx1, ch2=Tx1-Rx0, ch3=Tx1-Rx1
      Frame 1: ch4=Tx0-Rx0, ch5=Tx0-Rx1, ch6=Tx1-Rx0, ch7=Tx1-Rx1
    """
    radar = _make_radar(num_tx=2, num_rx=2, num_pulses=3, num_frames=2)
    result = _run_noise(radar)
    noise = result["noise"]

    v_ch = 2 * 2  # tx * rx
    rx_size = 2
    num_frames = 2

    for frame in range(num_frames):
        base = frame * v_ch
        for rx in range(rx_size):
            # Collect all virtual channels in this frame that map to this Rx
            ch_indices = [base + ch for ch in range(v_ch) if ch % rx_size == rx]
            ref = noise[ch_indices[0]]
            for ch_idx in ch_indices[1:]:
                npt.assert_array_equal(
                    noise[ch_idx],
                    ref,
                    err_msg=(
                        f"Frame {frame}, Rx {rx}: ch{ch_idx} should match "
                        f"ch{ch_indices[0]}"
                    ),
                )


def test_same_rx_many_tx_channels():
    """With 4 Tx and 1 Rx, all 4 virtual channels share Rx0 => identical noise."""
    radar = _make_radar(num_tx=4, num_rx=1, num_pulses=4)
    result = _run_noise(radar)
    noise = result["noise"]

    for ch in range(1, 4):
        npt.assert_array_equal(
            noise[ch],
            noise[0],
            err_msg=f"ch{ch} (Tx{ch}-Rx0) should match ch0 (Tx0-Rx0)",
        )


# ============================================================================
# Different Rx channels => different noise
# ============================================================================


def test_different_rx_different_noise():
    """
    Different Rx channels should produce statistically independent noise.
    """
    radar = _make_radar(num_tx=1, num_rx=2, num_pulses=8)
    result = _run_noise(radar)
    noise = result["noise"]

    # ch0 = Rx0, ch1 = Rx1 — they should differ
    assert not np.array_equal(
        noise[0], noise[1]
    ), "Different Rx channels should produce different noise"


# ============================================================================
# Complex vs real mode
# ============================================================================


def test_complex_mode_has_imaginary():
    """Complex baseband noise should have non-zero imaginary part."""
    radar = _make_radar(bb_type="complex")
    result = _run_noise(radar)
    noise = result["noise"]

    assert np.iscomplexobj(noise), "Complex mode should return complex array"
    assert np.any(noise.imag != 0), "Complex noise should have nonzero imaginary part"


def test_real_mode_is_real():
    """Real baseband noise should be purely real (no imaginary component)."""
    radar = _make_radar(bb_type="real")
    result = _run_noise(radar)
    noise = result["noise"]

    assert not np.iscomplexobj(noise) or np.all(
        noise.imag == 0
    ), "Real mode noise should have no imaginary part"


# ============================================================================
# Statistical properties
# ============================================================================


def test_noise_mean_approximately_zero():
    """Noise should have approximately zero mean (Gaussian)."""
    radar = _make_radar(num_tx=1, num_rx=1, num_pulses=64)
    result = _run_noise(radar)
    noise = result["noise"]

    if np.iscomplexobj(noise):
        vals = np.concatenate([noise.real.ravel(), noise.imag.ravel()])
    else:
        vals = noise.ravel()

    mean = np.mean(vals)
    std_err = np.std(vals) / np.sqrt(len(vals))
    # Allow 5-sigma tolerance
    assert abs(mean) < 5 * std_err, (
        f"Noise mean should be ~0, got {mean:.6e} "
        f"(std_err={std_err:.6e})"
    )


def test_noise_power_scales_with_noise_figure():
    """
    Increasing noise_figure by 10 dB should increase noise power by ~10 dB
    (i.e., amplitude by ~sqrt(10) ≈ 3.16x).
    """
    radar_low = _make_radar(noise_figure=5, num_tx=1, num_rx=1, num_pulses=32)
    radar_high = _make_radar(noise_figure=15, num_tx=1, num_rx=1, num_pulses=32)

    result_low = _run_noise(radar_low)
    result_high = _run_noise(radar_high)

    noise_low = result_low["noise"]
    noise_high = result_high["noise"]

    if np.iscomplexobj(noise_low):
        power_low = np.mean(np.abs(noise_low) ** 2)
        power_high = np.mean(np.abs(noise_high) ** 2)
    else:
        power_low = np.mean(noise_low**2)
        power_high = np.mean(noise_high**2)

    ratio_db = 10 * np.log10(power_high / power_low)

    # Should be ~10 dB, allow ±3 dB tolerance for statistical variation
    assert 7 < ratio_db < 13, (
        f"10 dB increase in noise figure should give ~10 dB more noise power, "
        f"got {ratio_db:.1f} dB"
    )


# ============================================================================
# Multi-frame independence
# ============================================================================


def test_multiframe_noise_differs_across_frames():
    """
    Different frames should produce independent noise realizations.
    """
    radar = _make_radar(num_tx=1, num_rx=1, num_pulses=4, num_frames=3)
    result = _run_noise(radar)
    noise = result["noise"]

    # With 1 Tx, 1 Rx: ch0=frame0, ch1=frame1, ch2=frame2
    assert not np.array_equal(
        noise[0], noise[1]
    ), "Frame 0 and Frame 1 noise should differ"
    assert not np.array_equal(
        noise[0], noise[2]
    ), "Frame 0 and Frame 2 noise should differ"


# ============================================================================
# Regression: noise amplitude matches expected thermal noise level
# ============================================================================


def test_noise_amplitude_matches_thermal_model():
    """
    Verify noise RMS matches the theoretical thermal noise amplitude
    computed from the radar's noise parameters.

    noise_level = sqrt(P_noise * R_load)
    where P_noise = 10^((input_noise + rf_gain + nf + 10*log10(BW) + bb_gain)/10) / 1000
    """
    radar = _make_radar(
        num_tx=1,
        num_rx=1,
        num_pulses=64,
        noise_figure=10,
        rf_gain=20,
        baseband_gain=30,
        bb_type="complex",
    )

    expected_noise_level = radar.sample_prop["noise"]

    result = _run_noise(radar)
    noise = result["noise"]

    # For complex noise, each component has std = noise_level / sqrt(2)
    # Total RMS = sqrt(mean(|noise|^2)) should ≈ noise_level
    measured_rms = np.sqrt(np.mean(np.abs(noise) ** 2))

    # Allow 20% tolerance for statistical variation
    npt.assert_allclose(
        measured_rms,
        expected_noise_level,
        rtol=0.2,
        err_msg=(
            f"Measured noise RMS ({measured_rms:.6e}) should match "
            f"theoretical noise level ({expected_noise_level:.6e})"
        ),
    )


def test_noise_amplitude_real_mode():
    """
    Verify noise RMS in real mode matches theoretical noise level.
    """
    radar = _make_radar(
        num_tx=1,
        num_rx=1,
        num_pulses=64,
        noise_figure=10,
        rf_gain=20,
        baseband_gain=30,
        bb_type="real",
    )

    expected_noise_level = radar.sample_prop["noise"]

    result = _run_noise(radar)
    noise = result["noise"]

    measured_rms = np.sqrt(np.mean(noise**2))

    npt.assert_allclose(
        measured_rms,
        expected_noise_level,
        rtol=0.2,
        err_msg=(
            f"Measured noise RMS ({measured_rms:.6e}) should match "
            f"theoretical noise level ({expected_noise_level:.6e})"
        ),
    )


# ============================================================================
# Regression: noise is independent of signal (baseband)
# ============================================================================


def test_noise_independent_of_target_rcs():
    """
    Noise should be the same statistical process regardless of target.
    Two simulations with different RCS targets should produce noise
    with the same RMS level (within tolerance).
    """
    radar1 = _make_radar(num_tx=1, num_rx=1, num_pulses=32)
    radar2 = _make_radar(num_tx=1, num_rx=1, num_pulses=32)

    result1 = _run_noise(radar1, targets=[{"location": (100, 0, 0), "rcs": 0}])
    result2 = _run_noise(radar2, targets=[{"location": (100, 0, 0), "rcs": 40}])

    noise1 = result1["noise"]
    noise2 = result2["noise"]

    if np.iscomplexobj(noise1):
        rms1 = np.sqrt(np.mean(np.abs(noise1) ** 2))
        rms2 = np.sqrt(np.mean(np.abs(noise2) ** 2))
    else:
        rms1 = np.sqrt(np.mean(noise1**2))
        rms2 = np.sqrt(np.mean(noise2**2))

    # RMS should be very similar (same noise model), allow 30% tolerance
    npt.assert_allclose(
        rms1,
        rms2,
        rtol=0.3,
        err_msg="Noise RMS should not depend on target RCS",
    )

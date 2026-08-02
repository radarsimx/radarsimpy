"""
Tests for the FFT front end of ``radarsimpy.processing``

Covers ``range_fft``, ``doppler_fft`` and ``range_doppler_fft``: windowing,
zero padding, the axes they operate on, and their equivalence.

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
import pytest
from scipy import signal

import radarsimpy.processing as proc

N_CHANNELS = 2
N_PULSES = 16
N_SAMPLES = 64


@pytest.fixture
def tone_baseband():
    """
    Baseband cube ``[channels, pulses, samples]`` holding a single complex
    tone whose range bin is 8 and Doppler bin is 3.
    """
    range_bin = 8
    doppler_bin = 3

    samples = np.arange(N_SAMPLES)
    pulses = np.arange(N_PULSES)
    fast_time = np.exp(2j * np.pi * range_bin * samples / N_SAMPLES)
    slow_time = np.exp(2j * np.pi * doppler_bin * pulses / N_PULSES)

    cube = np.einsum("p,s->ps", slow_time, fast_time)
    return np.repeat(cube[np.newaxis, ...], N_CHANNELS, axis=0), range_bin, doppler_bin


class TestRangeFFT:
    """``proc.range_fft`` operates on the fast-time (last) axis."""

    def test_no_window_locates_the_tone(self, tone_baseband):
        """Without a window the tone lands exactly on its range bin."""
        data, range_bin, _ = tone_baseband

        profile = proc.range_fft(data)

        assert profile.shape == data.shape
        for channel in range(N_CHANNELS):
            assert np.argmax(np.abs(profile[channel, 0, :])) == range_bin

    def test_matches_plain_fft(self, tone_baseband):
        """The unwindowed result is exactly ``fft`` along axis 2."""
        data, _, _ = tone_baseband

        npt.assert_allclose(proc.range_fft(data), np.fft.fft(data, axis=2), atol=1e-9)

    def test_window_is_applied_along_fast_time(self, tone_baseband):
        """A window multiplies each range line before transforming."""
        data, _, _ = tone_baseband
        window = signal.windows.chebwin(N_SAMPLES, at=60)

        windowed = proc.range_fft(data, rwin=window)
        expected = np.fft.fft(data * window[np.newaxis, np.newaxis, :], axis=2)

        npt.assert_allclose(windowed, expected, atol=1e-9)

    def test_window_lowers_sidelobes(self, tone_baseband):
        """Windowing trades main-lobe width for sidelobe suppression."""
        # Offset the tone by half a bin so it straddles two bins and leaks.
        samples = np.arange(N_SAMPLES)
        leaky = np.exp(2j * np.pi * 8.5 * samples / N_SAMPLES)
        data = leaky[np.newaxis, np.newaxis, :]

        rect = np.abs(proc.range_fft(data))[0, 0]
        cheb = np.abs(proc.range_fft(data, rwin=signal.windows.chebwin(N_SAMPLES, 80)))[
            0, 0
        ]

        # Look well away from the main lobe.
        far = np.r_[0:5, 13:N_SAMPLES]
        assert (cheb[far].max() / cheb.max()) < (rect[far].max() / rect.max())

    def test_zero_padding_increases_resolution(self, tone_baseband):
        """``n`` larger than the sample count zero-pads the transform."""
        data, range_bin, _ = tone_baseband
        n_fft = 4 * N_SAMPLES

        padded = proc.range_fft(data, n=n_fft)

        assert padded.shape == (N_CHANNELS, N_PULSES, n_fft)
        # The tone stays at the same normalised frequency.
        assert np.argmax(np.abs(padded[0, 0, :])) == range_bin * n_fft // N_SAMPLES

    def test_truncation(self, tone_baseband):
        """``n`` smaller than the sample count truncates the input."""
        data, _, _ = tone_baseband

        truncated = proc.range_fft(data, n=N_SAMPLES // 2)

        assert truncated.shape == (N_CHANNELS, N_PULSES, N_SAMPLES // 2)


class TestDopplerFFT:
    """``proc.doppler_fft`` operates on the slow-time (pulse) axis."""

    def test_no_window_locates_the_tone(self, tone_baseband):
        """Without a window the tone lands exactly on its Doppler bin."""
        data, _, doppler_bin = tone_baseband

        doppler = proc.doppler_fft(data)

        assert doppler.shape == data.shape
        assert np.argmax(np.abs(doppler[0, :, 0])) == doppler_bin

    def test_matches_plain_fft(self, tone_baseband):
        """The unwindowed result is exactly ``fft`` along axis 1."""
        data, _, _ = tone_baseband

        npt.assert_allclose(proc.doppler_fft(data), np.fft.fft(data, axis=1), atol=1e-9)

    def test_window_is_applied_along_slow_time(self, tone_baseband):
        """A window multiplies each Doppler line before transforming."""
        data, _, _ = tone_baseband
        window = signal.windows.chebwin(N_PULSES, at=60)

        windowed = proc.doppler_fft(data, dwin=window)
        expected = np.fft.fft(data * window[np.newaxis, :, np.newaxis], axis=1)

        npt.assert_allclose(windowed, expected, atol=1e-9)

    def test_zero_padding(self, tone_baseband):
        """``n`` zero-pads along the Doppler axis."""
        data, _, doppler_bin = tone_baseband
        n_fft = 4 * N_PULSES

        padded = proc.doppler_fft(data, n=n_fft)

        assert padded.shape == (N_CHANNELS, n_fft, N_SAMPLES)
        assert np.argmax(np.abs(padded[0, :, 0])) == doppler_bin * n_fft // N_PULSES


class TestRangeDopplerFFT:
    """``proc.range_doppler_fft`` chains the two transforms."""

    def test_peak_is_at_the_expected_bin(self, tone_baseband):
        """The 2-D peak sits at (Doppler bin, range bin)."""
        data, range_bin, doppler_bin = tone_baseband

        rdm = proc.range_doppler_fft(data)

        assert rdm.shape == data.shape
        peak = np.unravel_index(np.argmax(np.abs(rdm[0])), rdm[0].shape)
        assert peak == (doppler_bin, range_bin)

    def test_equivalent_to_chaining_manually(self, tone_baseband):
        """The convenience wrapper matches an explicit range-then-Doppler chain."""
        data, _, _ = tone_baseband
        rwin = signal.windows.chebwin(N_SAMPLES, at=60)
        dwin = signal.windows.chebwin(N_PULSES, at=50)

        combined = proc.range_doppler_fft(data, rwin=rwin, dwin=dwin)
        chained = proc.doppler_fft(proc.range_fft(data, rwin=rwin), dwin=dwin)

        npt.assert_allclose(combined, chained, atol=1e-9)

    def test_independent_fft_sizes(self, tone_baseband):
        """``rn`` and ``dn`` size the range and Doppler axes independently."""
        data, _, _ = tone_baseband

        rdm = proc.range_doppler_fft(data, rn=2 * N_SAMPLES, dn=8 * N_PULSES)

        assert rdm.shape == (N_CHANNELS, 8 * N_PULSES, 2 * N_SAMPLES)

    def test_real_input_is_accepted(self):
        """Real-valued baseband produces a conjugate-symmetric range profile."""
        samples = np.arange(N_SAMPLES)
        data = np.cos(2 * np.pi * 8 * samples / N_SAMPLES)
        data = np.broadcast_to(data, (1, N_PULSES, N_SAMPLES))

        rdm = proc.range_doppler_fft(data)

        assert rdm.shape == (1, N_PULSES, N_SAMPLES)
        # A real cosine has mirrored range peaks at bins 8 and N-8.
        profile = np.abs(rdm[0, 0, :])
        assert np.argmax(profile[: N_SAMPLES // 2]) == 8
        npt.assert_allclose(profile[8], profile[N_SAMPLES - 8], rtol=1e-9)

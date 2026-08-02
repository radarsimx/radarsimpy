"""
System level test for CFAR

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

import radarsimpy.processing as proc
from radarsimpy.tools import log_factorial


def test_ca_cfar():
    """
    This function tests the CA-CFAR algorithm.
    """
    sig = np.ones((2, 32))
    sig[0, 16] = 20
    sig[1, 10] = 30

    ca_cfar = proc.cfar_ca_1d(sig, guard=2, trailing=10, pfa=1e-2, axis=1)

    npt.assert_almost_equal(
        ca_cfar[0, :],
        np.array(
            [
                2.58925412,
                2.58925412,
                2.58925412,
                2.84817953,
                8.02668777,
                8.28561318,
                8.54453859,
                8.803464,
                9.06238941,
                9.32131482,
                9.58024024,
                9.83916565,
                10.09809106,
                10.09809106,
                5.17850824,
                5.17850824,
                5.17850824,
                5.17850824,
                5.17850824,
                10.09809106,
                9.83916565,
                9.58024024,
                9.32131482,
                9.06238941,
                8.803464,
                8.54453859,
                8.28561318,
                8.02668777,
                7.76776235,
                2.58925412,
                2.58925412,
                2.58925412,
            ]
        ),
        decimal=3,
    )
    npt.assert_almost_equal(
        ca_cfar[1, :],
        np.array(
            [
                10.09809106,
                10.09809106,
                10.09809106,
                10.35701647,
                10.61594188,
                10.8748673,
                11.13379271,
                11.39271812,
                4.14280659,
                4.401732,
                4.66065741,
                4.91958282,
                5.17850824,
                12.68734518,
                12.68734518,
                12.68734518,
                12.68734518,
                12.68734518,
                12.68734518,
                12.68734518,
                12.42841977,
                12.16949435,
                11.91056894,
                4.14280659,
                3.88388118,
                3.62495577,
                3.36603035,
                3.10710494,
                2.84817953,
                2.58925412,
                2.58925412,
                2.58925412,
            ]
        ),
        decimal=3,
    )

    sig = np.ones((32, 2))
    sig[16, 0] = 20
    sig[10, 1] = 30

    ca_cfar = proc.cfar_ca_1d(sig, guard=2, trailing=10, pfa=1e-2, axis=0)

    npt.assert_almost_equal(
        ca_cfar[:, 0],
        np.array(
            [
                2.58925412,
                2.58925412,
                2.58925412,
                2.84817953,
                8.02668777,
                8.28561318,
                8.54453859,
                8.803464,
                9.06238941,
                9.32131482,
                9.58024024,
                9.83916565,
                10.09809106,
                10.09809106,
                5.17850824,
                5.17850824,
                5.17850824,
                5.17850824,
                5.17850824,
                10.09809106,
                9.83916565,
                9.58024024,
                9.32131482,
                9.06238941,
                8.803464,
                8.54453859,
                8.28561318,
                8.02668777,
                7.76776235,
                2.58925412,
                2.58925412,
                2.58925412,
            ]
        ),
        decimal=3,
    )
    npt.assert_almost_equal(
        ca_cfar[:, 1],
        np.array(
            [
                10.09809106,
                10.09809106,
                10.09809106,
                10.35701647,
                10.61594188,
                10.8748673,
                11.13379271,
                11.39271812,
                4.14280659,
                4.401732,
                4.66065741,
                4.91958282,
                5.17850824,
                12.68734518,
                12.68734518,
                12.68734518,
                12.68734518,
                12.68734518,
                12.68734518,
                12.68734518,
                12.42841977,
                12.16949435,
                11.91056894,
                4.14280659,
                3.88388118,
                3.62495577,
                3.36603035,
                3.10710494,
                2.84817953,
                2.58925412,
                2.58925412,
                2.58925412,
            ]
        ),
        decimal=3,
    )


def test_os_cfar():
    """
    This function tests the OS-CFAR algorithm.
    """
    sig = np.ones((2, 32))
    sig[0, 16] = 20
    sig[1, 10] = 30

    os_cfar = proc.cfar_os_1d(sig, guard=0, trailing=4, k=6, pfa=1e-2, axis=1)

    npt.assert_almost_equal(
        os_cfar[0, :],
        np.array(
            [
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
            ]
        ),
        decimal=3,
    )
    npt.assert_almost_equal(
        os_cfar[1, :],
        np.array(
            [
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
            ]
        ),
        decimal=3,
    )

    sig = np.ones((32, 2))
    sig[16, 0] = 20
    sig[10, 1] = 30

    os_cfar = proc.cfar_os_1d(sig, guard=0, trailing=4, k=6, pfa=1e-2, axis=0)

    npt.assert_almost_equal(
        os_cfar[:, 0],
        np.array(
            [
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
            ]
        ),
        decimal=3,
    )
    npt.assert_almost_equal(
        os_cfar[:, 1],
        np.array(
            [
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
                5.86979834,
            ]
        ),
        decimal=3,
    )


# =============================================================================
# Threshold factor
# =============================================================================


class TestOSCFARThreshold:
    """``proc.os_cfar_threshold`` solves for the OS-CFAR scale factor."""

    def test_known_value(self):
        """Reference value for the classic k=18, N=32, Pfa=1e-6 case."""
        npt.assert_almost_equal(
            proc.os_cfar_threshold(18, 32, 1e-6), 26.1270827, decimal=5
        )

    def test_threshold_is_positive(self):
        """The scale factor is a positive multiplier."""
        assert proc.os_cfar_threshold(12, 16, 1e-4) > 0

    def test_lower_pfa_needs_a_larger_threshold(self):
        """Demanding fewer false alarms raises the required threshold."""
        loose = proc.os_cfar_threshold(12, 16, 1e-3)
        tight = proc.os_cfar_threshold(12, 16, 1e-8)
        assert tight > loose

    def test_higher_rank_needs_a_smaller_threshold(self):
        """A higher rank ``k`` already suppresses more noise."""
        low_rank = proc.os_cfar_threshold(9, 16, 1e-5)
        high_rank = proc.os_cfar_threshold(15, 16, 1e-5)
        assert high_rank < low_rank

    def test_satisfies_the_defining_equation(self):
        """The returned root drives the OS-CFAR Pfa equation to ~zero."""
        k, n, pfa = 12, 16, 1e-5
        t_os = proc.os_cfar_threshold(k, n, pfa)

        residual = (
            log_factorial(n)
            - log_factorial(n - k)
            - np.sum(np.log(np.arange(n, n - k, -1) + t_os))
            - np.log(pfa)
        )
        assert abs(residual) < 1e-3


# =============================================================================
# Input validation
# =============================================================================

#: ``(function, minimal working keyword arguments)`` for all four variants.
CFAR_VARIANTS = [
    (proc.cfar_ca_1d, {"guard": 1, "trailing": 4}),
    (proc.cfar_ca_2d, {"guard": 1, "trailing": 4}),
    (proc.cfar_os_1d, {"guard": 1, "trailing": 4, "k": 6}),
    (proc.cfar_os_2d, {"guard": 1, "trailing": 4, "k": 100}),
]


@pytest.mark.parametrize("func, kwargs", CFAR_VARIANTS)
def test_complex_input_is_rejected(func, kwargs):
    """CFAR operates on amplitude/power, so complex data is an error."""
    data = np.ones((24, 24), dtype=complex)

    with pytest.raises(ValueError, match="should not be complex"):
        func(data, **kwargs)


@pytest.mark.parametrize("func, kwargs", CFAR_VARIANTS)
def test_unknown_detector_is_rejected(func, kwargs):
    """Only ``linear`` and ``squarelaw`` detectors exist."""
    data = np.ones((24, 24))

    with pytest.raises(ValueError, match="`linear` or `squarelaw`"):
        func(data, detector="magnitude", **kwargs)


@pytest.mark.parametrize("func", [proc.cfar_ca_2d, proc.cfar_os_2d])
def test_no_trailing_bins_is_rejected(func):
    """With zero trailing cells there is nothing to average over."""
    data = np.ones((16, 16))
    kwargs = {"k": 4} if func is proc.cfar_os_2d else {}

    with pytest.raises(ValueError, match="No trailing bins"):
        func(data, guard=2, trailing=0, **kwargs)


def test_os_cfar_2d_rejects_no_trailing_bins_with_an_offset():
    """
    An empty training set is invalid however the threshold factor was
    obtained. Without this check ``samples[k]`` fails deep in the sliding
    window with a bare IndexError.
    """
    data = np.ones((16, 16))

    with pytest.raises(ValueError, match="No trailing bins"):
        proc.cfar_os_2d(data, guard=2, trailing=0, k=4, offset=1.0)


def test_os_cfar_1d_warns_on_out_of_range_rank():
    """``k`` outside ``N/2 < k < N`` triggers a guidance warning."""
    data = np.ones((32,))

    with pytest.warns(UserWarning, match="usuall chosen to satisfy"):
        proc.cfar_os_1d(data, guard=0, trailing=8, k=2, offset=1.0)


@pytest.mark.parametrize("threshold_kwargs", [{"pfa": 1e-3}, {"offset": 1.0}])
def test_os_cfar_2d_warns_on_out_of_range_rank(threshold_kwargs):
    """
    The 2-D variant warns on the same condition, and the warning depends on
    the window geometry rather than on how the threshold factor was derived --
    so it must fire on the ``offset`` path too.
    """
    data = np.ones((12, 12))

    with pytest.warns(UserWarning, match="usuall chosen to satisfy"):
        proc.cfar_os_2d(data, guard=1, trailing=2, k=2, **threshold_kwargs)


def test_os_cfar_2d_accepts_explicit_offset():
    """An explicit ``offset`` bypasses the Pfa-derived threshold."""
    data = np.full((16, 16), 2.0)

    # guard=1 / trailing=3 gives N = 9*9 - 3*3 = 72 training cells; on a flat
    # field every order statistic is the field level, so cfar == 2.0 * offset.
    cfar = proc.cfar_os_2d(data, guard=1, trailing=3, k=54, offset=1.5)

    npt.assert_allclose(cfar, 3.0, rtol=1e-9)


def test_os_cfar_2d_offset_ignores_pfa_and_detector():
    """
    Once ``offset`` is given, neither ``pfa`` nor ``detector`` may influence
    the result -- they only feed the threshold factor that ``offset`` replaces.
    """
    data = np.full((16, 16), 2.0)
    params = {"guard": 1, "trailing": 3, "k": 54, "offset": 1.5}

    baseline = proc.cfar_os_2d(data, **params)

    npt.assert_allclose(proc.cfar_os_2d(data, pfa=1e-9, **params), baseline)
    npt.assert_allclose(proc.cfar_os_2d(data, detector="linear", **params), baseline)


# =============================================================================
# Detector and offset behaviour
# =============================================================================


@pytest.mark.parametrize("func, kwargs", CFAR_VARIANTS)
def test_linear_detector_threshold_is_sqrt_of_squarelaw(func, kwargs):
    """
    The linear detector uses the square root of the square-law factor.

    On a flat unity field the noise estimate is 1, so away from the
    convolution edges the threshold is the scale factor itself.
    """
    data = np.ones((32, 32))

    squarelaw = func(data, pfa=1e-3, **kwargs)
    linear = func(data, pfa=1e-3, detector="linear", **kwargs)

    interior = (slice(12, -12), slice(12, -12))
    npt.assert_allclose(linear[interior], np.sqrt(squarelaw[interior]), rtol=1e-9)


@pytest.mark.parametrize("func, kwargs", CFAR_VARIANTS)
def test_explicit_offset_bypasses_pfa(func, kwargs):
    """An explicit ``offset`` scales the estimate and ignores ``pfa``."""
    data = np.ones((24, 24))

    once = func(data, offset=1.0, **kwargs)
    twice = func(data, offset=2.0, **kwargs)

    npt.assert_allclose(twice, 2.0 * once, rtol=1e-9)
    # On a flat unity field the noise estimate is the field level itself.
    # The CA variants convolve in "same" mode, so only the interior is free
    # of the implicit zeros at the borders.
    interior = (slice(8, -8), slice(8, -8))
    npt.assert_allclose(once[interior], 1.0, rtol=1e-9)


def test_ca_cfar_1d_axis0_matches_axis1_transposed(rng):
    """``axis=0`` and ``axis=1`` are transposes of one another."""
    data = rng.random((24, 20)) + 1.0

    along_rows = proc.cfar_ca_1d(data, guard=1, trailing=5, axis=0)
    along_cols = proc.cfar_ca_1d(data.T, guard=1, trailing=5, axis=1)

    npt.assert_allclose(along_rows, along_cols.T, rtol=1e-9)


def test_ca_cfar_1d_accepts_1d_input():
    """A 1-D input is handled by the ``data.ndim == 1`` branch."""
    data = np.ones(64)
    data[32] = 50

    cfar = proc.cfar_ca_1d(data, guard=2, trailing=8, pfa=1e-3)

    assert cfar.shape == data.shape
    assert data[32] > cfar[32], "the strong target should exceed its threshold"


def test_os_cfar_1d_1d_and_2d_inputs_agree(rng):
    """The 1-D OS-CFAR branch mirrors the 2-D one for a single column."""
    column = rng.random((48, 1)) + 1.0

    one_d = proc.cfar_os_1d(column[:, 0], guard=1, trailing=6, k=9, offset=2.0)
    two_d = proc.cfar_os_1d(column, guard=1, trailing=6, k=9, offset=2.0)

    npt.assert_allclose(one_d, two_d[:, 0], rtol=1e-9)


# =============================================================================
# 2-D CFAR
# =============================================================================

#: 2-D variants. The OS rank follows the recommended ~0.75 N, where a
#: guard=2 / trailing=6 window has N = 17*17 - 5*5 = 264 training cells.
CFAR_2D_VARIANTS = [(proc.cfar_ca_2d, {}), (proc.cfar_os_2d, {"k": 198})]


@pytest.fixture
def range_doppler_map():
    """A flat noise field with one strong target at (12, 20)."""
    field = np.ones((24, 40))
    field[12, 20] = 500.0
    return field, (12, 20)


class TestCFAR2D:
    """``proc.cfar_ca_2d`` and ``proc.cfar_os_2d``."""

    @pytest.mark.parametrize("func, extra", CFAR_2D_VARIANTS)
    def test_detects_the_target(self, func, extra, range_doppler_map):
        """The target exceeds its own threshold on a flat noise field."""
        field, (row, col) = range_doppler_map

        cfar = func(field, guard=2, trailing=6, pfa=1e-3, **extra)

        assert cfar.shape == field.shape
        assert field[row, col] > cfar[row, col]

    @pytest.mark.parametrize("func, extra", CFAR_2D_VARIANTS)
    def test_scalar_and_list_guard_trailing_agree(self, func, extra, range_doppler_map):
        """A scalar ``guard``/``trailing`` is equivalent to a two-element list."""
        field, _ = range_doppler_map

        scalar = func(field, guard=2, trailing=6, pfa=1e-3, **extra)
        listed = func(field, guard=[2, 2], trailing=[6, 6], pfa=1e-3, **extra)

        npt.assert_allclose(scalar, listed, rtol=1e-9)

    # guard=[1, 3] / trailing=[4, 8] gives N = 11*23 - 3*7 = 232 training cells.
    @pytest.mark.parametrize(
        "func, extra", [(proc.cfar_ca_2d, {}), (proc.cfar_os_2d, {"k": 174})]
    )
    def test_asymmetric_windows(self, func, extra, range_doppler_map):
        """Different guard/trailing sizes per axis are supported."""
        field, (row, col) = range_doppler_map

        cfar = func(field, guard=[1, 3], trailing=[4, 8], pfa=1e-3, **extra)

        assert cfar.shape == field.shape
        assert field[row, col] > cfar[row, col]

    def test_ca_cfar_2d_flat_field_threshold(self):
        """On a constant field the CA-CFAR threshold equals ``offset * level``."""
        field = np.full((20, 20), 4.0)

        cfar = proc.cfar_ca_2d(field, guard=1, trailing=4, offset=3.0)

        # Ignore the convolution edges, which see implicit zeros.
        interior = cfar[6:-6, 6:-6]
        npt.assert_allclose(interior, 12.0, rtol=1e-9)

    def test_os_cfar_2d_is_robust_to_an_interfering_target(self):
        """
        A second strong cell inside the training window biases CA-CFAR but
        leaves the order statistic — and therefore OS-CFAR — untouched.
        """
        clean = np.ones((24, 24))
        clean[12, 12] = 200.0  # target under test

        interfered = clean.copy()
        interfered[12, 17] = 200.0  # interferer inside the training window

        params = {"guard": 2, "trailing": 5, "pfa": 1e-3}
        ca_clean = proc.cfar_ca_2d(clean, **params)[12, 12]
        ca_interfered = proc.cfar_ca_2d(interfered, **params)[12, 12]
        os_clean = proc.cfar_os_2d(clean, k=150, **params)[12, 12]
        os_interfered = proc.cfar_os_2d(interfered, k=150, **params)[12, 12]

        assert ca_interfered > ca_clean, "CA-CFAR should be pulled up by the interferer"
        npt.assert_allclose(os_interfered, os_clean, rtol=1e-9)
        assert clean[12, 12] > os_interfered

    def test_os_cfar_2d_wraps_around_the_edges(self):
        """Edge cells roll over, so a uniform field gives a uniform threshold."""
        field = np.full((16, 16), 2.0)

        # guard=1 / trailing=3 gives N = 9*9 - 3*3 = 72 training cells.
        cfar = proc.cfar_os_2d(field, guard=1, trailing=3, k=54, pfa=1e-3)

        # Every cell sees the same rolled-over training set.
        npt.assert_allclose(cfar, cfar[0, 0], rtol=1e-9)
        npt.assert_allclose(cfar[0, 0], 2.0 * proc.os_cfar_threshold(54, 72, 1e-3))

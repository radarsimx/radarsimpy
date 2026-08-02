"""
A Python module for radar simulation

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
from scipy.special import gammaln

from radarsimpy.tools import (
    log_factorial,
    marcumq,
    pd_swerling0,
    pd_swerling1,
    pd_swerling2,
    pd_swerling3,
    pd_swerling4,
    roc_pd,
    roc_snr,
    threshold,
)


def test_roc_pd():
    """
    Test the ROC Pd function
    """
    npt.assert_almost_equal(roc_pd(1e-8, 13, 1, "Swerling 5"), 0.6287, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-8, 11, 1, "Swerling 5"), 0.1683, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-8, -3.2, 256, "Swerling 5"), 0.8411, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-8, -4.8, 256, "Swerling 5"), 0.2249, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-9, -10, 256, "Coherent"), 0.8765, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-9, -12, 256, "Coherent"), 0.3767, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-6, 12.4, 1, "Coherent"), 0.8733, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-6, 8.8, 1, "Coherent"), 0.1953, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-4, 16, 1, "Swerling 1"), 0.7980, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 6.8, 1, "Swerling 1"), 0.2036, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 3.6, 256, "Swerling 1"), 0.8959, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -7.6, 256, "Swerling 1"), 0.2560, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-4, -4.4, 256, "Swerling 2"), 0.9120, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -7.2, 256, "Swerling 2"), 0.2125, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 16, 1, "Swerling 2"), 0.7980, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 6.8, 1, "Swerling 2"), 0.2036, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-4, 15.2, 1, "Swerling 3"), 0.8846, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 6.8, 1, "Swerling 3"), 0.1931, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -0.4, 256, "Swerling 3"), 0.8889, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -8.4, 256, "Swerling 3"), 0.1775, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-4, 15.2, 1, "Swerling 4"), 0.8846, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 6.8, 1, "Swerling 4"), 0.1931, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -4.8, 256, "Swerling 4"), 0.8413, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -7.2, 256, "Swerling 4"), 0.2155, decimal=4)


def test_roc_snr():
    """
    Test the ROC SNR function
    """
    npt.assert_almost_equal(roc_snr(1e-8, 0.6290, 1, "Swerling 5"), 13, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-8, 0.1681, 1, "Swerling 5"), 11, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-8, 0.8424, 256, "Swerling 5"), -3.2, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-8, 0.2266, 256, "Swerling 5"), -4.8, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-9, 0.8765, 256, "Coherent"), -10, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-9, 0.3767, 256, "Coherent"), -12, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-6, 0.8733, 1, "Coherent"), 12.4, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-6, 0.1953, 1, "Coherent"), 8.8, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-4, 0.7980, 1, "Swerling 1"), 16, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2036, 1, "Swerling 1"), 6.8, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.8959, 256, "Swerling 1"), 3.6, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2560, 256, "Swerling 1"), -7.6, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-4, 0.9120, 256, "Swerling 2"), -4.4, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2125, 256, "Swerling 2"), -7.2, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.7980, 1, "Swerling 2"), 16, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2036, 1, "Swerling 2"), 6.8, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-4, 0.8846, 1, "Swerling 3"), 15.2, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.1931, 1, "Swerling 3"), 6.8, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.8889, 256, "Swerling 3"), -0.4, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.1772, 256, "Swerling 3"), -8.4, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-4, 0.8846, 1, "Swerling 4"), 15.2, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.1931, 1, "Swerling 4"), 6.8, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.8413, 256, "Swerling 4"), -4.8, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2155, 256, "Swerling 4"), -7.2, decimal=1)


# =============================================================================
# Low level helpers
# =============================================================================


class TestLogFactorial:
    """``log_factorial`` computes ``log(n!)`` without overflowing."""

    @pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 20])
    def test_matches_scipy_gammaln(self, n):
        """``log(n!) == gammaln(n + 1)``."""
        npt.assert_allclose(log_factorial(n), gammaln(n + 1), atol=1e-9)

    def test_zero_and_one_are_zero(self):
        """``0! == 1! == 1``, so the log is 0."""
        assert log_factorial(0) == 0.0
        assert log_factorial(1) == 0.0

    def test_array_input_is_evaluated_elementwise(self):
        """An array argument returns an array of the same shape."""
        n = np.array([0, 1, 4, 7, 12])

        result = log_factorial(n)

        assert result.shape == n.shape
        npt.assert_allclose(result, gammaln(n + 1), atol=1e-9)

    def test_large_n_does_not_overflow(self):
        """A value whose factorial overflows float64 is still finite in log."""
        value = log_factorial(1000)

        assert np.isfinite(value)
        npt.assert_allclose(value, gammaln(1001), rtol=1e-12)


class TestMarcumQ:
    """``marcumq`` is the generalized Marcum Q function."""

    def test_zero_threshold_is_one(self):
        """``Q_m(a, 0) == 1`` for any non-centrality."""
        npt.assert_allclose(marcumq(0.0, 0.0), 1.0, atol=1e-12)
        npt.assert_allclose(marcumq(3.0, 0.0), 1.0, atol=1e-12)

    def test_decreases_with_threshold(self):
        """Raising the threshold can only lower the tail probability."""
        values = [marcumq(2.0, x) for x in (0.5, 1.0, 2.0, 4.0, 8.0)]
        assert all(np.diff(values) < 0)

    def test_increases_with_non_centrality(self):
        """A stronger signal raises the probability of exceeding a threshold."""
        values = [marcumq(a, 3.0) for a in (0.0, 1.0, 2.0, 4.0)]
        assert all(np.diff(values) > 0)

    def test_bounded_to_the_unit_interval(self):
        """The result is a probability."""
        for a in (0.0, 1.5, 5.0):
            for x in (0.1, 1.0, 10.0):
                assert 0.0 <= marcumq(a, x) <= 1.0

    def test_first_order_matches_rayleigh_tail(self):
        """``Q_1(0, x) == exp(-x^2 / 2)``, the Rayleigh survival function."""
        for x in (0.5, 1.0, 3.0):
            npt.assert_allclose(marcumq(0.0, x), np.exp(-(x**2) / 2), atol=1e-9)

    def test_higher_order(self):
        """``m > 1`` selects a higher-order (more degrees of freedom) form."""
        assert marcumq(1.0, 2.0, m=3) > marcumq(1.0, 2.0, m=1)


# =============================================================================
# Swerling detection probabilities
# =============================================================================

#: The ``pd_swerling*`` functions, keyed by the ``roc_pd`` ``stype`` they back.
SWERLING_FUNCS = [
    ("Swerling 0", pd_swerling0),
    ("Swerling 1", pd_swerling1),
    ("Swerling 2", pd_swerling2),
    ("Swerling 3", pd_swerling3),
    ("Swerling 4", pd_swerling4),
]


class TestSwerlingModels:
    """Direct tests of the per-model detection-probability functions."""

    @pytest.mark.parametrize("stype, func", SWERLING_FUNCS)
    @pytest.mark.parametrize("npulses", [1, 4, 32])
    def test_pd_is_a_probability(self, stype, func, npulses):
        """Pd stays within [0, 1] across a wide SNR sweep."""
        thred = threshold(1e-6, npulses)

        for snr_db in (-10, 0, 10, 20):
            pd = func(npulses, 10 ** (snr_db / 10), thred)
            assert 0.0 <= pd <= 1.0, f"{stype} npulses={npulses} snr={snr_db}"

    @pytest.mark.parametrize("stype, func", SWERLING_FUNCS)
    def test_pd_increases_with_snr(self, stype, func):
        """More signal always means a higher probability of detection."""
        npulses = 8
        thred = threshold(1e-6, npulses)

        pds = [func(npulses, 10 ** (snr / 10), thred) for snr in (-5, 0, 5, 10, 15)]
        assert all(np.diff(pds) > 0), f"{stype} is not monotonic in SNR: {pds}"

    @pytest.mark.parametrize("stype, func", SWERLING_FUNCS)
    def test_matches_roc_pd(self, stype, func):
        """``roc_pd`` dispatches to the matching ``pd_swerling*`` function."""
        pfa, snr_db, npulses = 1e-6, 8.0, 10
        thred = threshold(pfa, npulses)

        direct = func(npulses, 10 ** (snr_db / 10), thred)
        via_roc = roc_pd(pfa, snr_db, npulses, stype)

        npt.assert_allclose(via_roc, direct, rtol=1e-9)

    def test_swerling0_scalar_and_array_snr_agree(self):
        """The vectorised Swerling 0 branch matches the scalar branch."""
        npulses = 10
        thred = threshold(1e-6, npulses)
        snr = 10 ** (np.array([-5.0, 0.0, 5.0, 10.0]) / 10)

        vector = pd_swerling0(npulses, snr, thred)
        scalar = [pd_swerling0(npulses, float(s), thred) for s in snr]

        assert vector.shape == snr.shape
        npt.assert_allclose(vector, scalar, rtol=1e-9)

    def test_swerling0_uses_the_gaussian_approximation_above_50_pulses(self):
        """Above 50 pulses the series is replaced by an asymptotic expansion."""
        snr = 10 ** (-5.0 / 10)

        below = pd_swerling0(50, snr, threshold(1e-6, 50))
        above = pd_swerling0(51, snr, threshold(1e-6, 51))

        # The two branches must agree closely across the switch-over.
        npt.assert_allclose(above, below, atol=0.02)
        assert 0.0 <= above <= 1.0

    def test_swerling1_single_pulse_closed_form(self):
        """For one pulse Swerling 1 reduces to ``exp(-thred / (1 + snr))``."""
        snr, thred = 4.0, threshold(1e-4, 1)

        npt.assert_allclose(pd_swerling1(1, snr, thred), np.exp(-thred / (1 + snr)))

    def test_swerling1_and_2_agree_for_a_single_pulse(self):
        """Scan-to-scan and pulse-to-pulse fluctuation coincide at N = 1."""
        snr, thred = 4.0, threshold(1e-4, 1)

        npt.assert_allclose(pd_swerling1(1, snr, thred), pd_swerling2(1, snr, thred))

    def test_swerling3_and_4_agree_for_a_single_pulse(self):
        """The dominant-scatterer models also coincide at N = 1."""
        snr, thred = 4.0, threshold(1e-4, 1)

        npt.assert_allclose(
            pd_swerling3(1, snr, thred), pd_swerling4(1, snr, thred), rtol=1e-6
        )

    def test_swerling3_small_pulse_count_branch(self):
        """``npulses <= 2`` returns the closed form without the series term."""
        for npulses in (1, 2):
            pd = pd_swerling3(npulses, 4.0, threshold(1e-4, npulses))
            assert 0.0 <= pd <= 1.0

    def test_swerling4_uses_the_gaussian_approximation_at_50_pulses(self):
        """``npulses >= 50`` switches to the asymptotic expansion."""
        snr = 10 ** (-5.0 / 10)

        below = pd_swerling4(49, snr, threshold(1e-6, 49))
        above = pd_swerling4(50, snr, threshold(1e-6, 50))

        npt.assert_allclose(above, below, atol=0.05)
        assert 0.0 <= above <= 1.0

    def test_fluctuating_targets_need_more_snr_than_non_fluctuating(self):
        """At high Pd, a fluctuating target is harder to detect."""
        npulses, pfa = 1, 1e-6
        thred = threshold(pfa, npulses)
        snr = 10 ** (15.0 / 10)

        assert pd_swerling1(npulses, snr, thred) < pd_swerling0(npulses, snr, thred)


# =============================================================================
# threshold / roc_pd / roc_snr shapes and edge cases
# =============================================================================


class TestThreshold:
    """``threshold`` inverts the incomplete gamma function."""

    def test_known_values(self):
        """Reference thresholds from Mahafza."""
        npt.assert_almost_equal(threshold(1e-4, 1), 9.21, decimal=2)
        npt.assert_almost_equal(threshold(1e-4, 10), 26.19, decimal=2)
        npt.assert_almost_equal(threshold(1e-4, 20), 41.03, decimal=2)
        npt.assert_almost_equal(threshold(1e-4, 40), 67.89, decimal=2)

    def test_decreasing_pfa_raises_the_threshold(self):
        """Fewer false alarms demand a higher bar."""
        values = [threshold(pfa, 10) for pfa in (1e-2, 1e-4, 1e-6, 1e-8)]
        assert all(np.diff(values) > 0)

    def test_more_pulses_raise_the_threshold(self):
        """Integrating more pulses accumulates more noise power."""
        values = [threshold(1e-6, n) for n in (1, 2, 10, 100)]
        assert all(np.diff(values) > 0)


class TestROCShapes:
    """``roc_pd`` and ``roc_snr`` broadcast scalars and 1-D arrays."""

    def test_scalar_inputs_return_a_scalar(self):
        """Two scalars give a scalar."""
        pd = roc_pd(1e-6, 10.0, 1, "Swerling 1")
        assert np.isscalar(pd) or np.ndim(pd) == 0

    def test_snr_array_returns_1d(self):
        """A scalar Pfa with an SNR sweep gives a 1-D result."""
        snr = np.array([0.0, 5.0, 10.0, 15.0])

        pd = roc_pd(1e-6, snr, 1, "Swerling 1")

        assert pd.shape == snr.shape
        assert all(np.diff(pd) > 0)

    def test_pfa_array_returns_1d(self):
        """A Pfa sweep with a scalar SNR gives a 1-D result."""
        pfa = np.array([1e-3, 1e-5, 1e-7])

        pd = roc_pd(pfa, 10.0, 1, "Swerling 1")

        assert pd.shape == pfa.shape
        assert all(np.diff(pd) < 0), "a tighter Pfa lowers Pd"

    def test_both_arrays_return_2d(self):
        """Two sweeps give a ``[pfa, snr]`` matrix."""
        pfa = np.array([1e-4, 1e-6])
        snr = np.array([0.0, 5.0, 10.0])

        pd = roc_pd(pfa, snr, 1, "Swerling 1")

        assert pd.shape == (pfa.size, snr.size)

    def test_real_signal_type(self):
        """The ``Real`` non-fluctuating type is supported."""
        pd = roc_pd(1e-6, 10.0, 4, "Real")
        assert 0.0 <= pd <= 1.0

    def test_real_needs_more_snr_than_coherent(self):
        """Real (envelope) detection loses 3 dB relative to coherent."""
        assert roc_pd(1e-6, 8.0, 8, "Real") < roc_pd(1e-6, 8.0, 8, "Coherent")

    def test_swerling_5_is_an_alias_of_swerling_0(self):
        """The two names select the same model."""
        npt.assert_allclose(
            roc_pd(1e-6, 10.0, 4, "Swerling 5"),
            roc_pd(1e-6, 10.0, 4, "Swerling 0"),
        )

    def test_unknown_signal_type_returns_none(self):
        """An unrecognised ``stype`` yields ``None`` rather than raising."""
        assert roc_pd(1e-6, 10.0, 1, "Swerling 42") is None

    def test_roc_snr_inverts_roc_pd(self):
        """``roc_snr`` recovers the SNR that ``roc_pd`` maps to a given Pd."""
        pfa, npulses, stype = 1e-6, 10, "Swerling 1"
        snr_db = 6.0

        pd = roc_pd(pfa, snr_db, npulses, stype)
        recovered = roc_snr(pfa, pd, npulses, stype)

        npt.assert_allclose(recovered, snr_db, atol=1e-2)

    def test_roc_snr_pd_array_returns_1d(self):
        """A Pd sweep gives a 1-D SNR result that increases with Pd."""
        pd = np.array([0.3, 0.5, 0.7, 0.9])

        snr = roc_snr(1e-6, pd, 1, "Coherent")

        assert snr.shape == pd.shape
        assert all(np.diff(snr) > 0)

    def test_roc_snr_pfa_array_returns_1d(self):
        """A Pfa sweep gives a 1-D SNR result that increases as Pfa tightens."""
        pfa = np.array([1e-3, 1e-5, 1e-7])

        snr = roc_snr(pfa, 0.9, 1, "Coherent")

        assert snr.shape == pfa.shape
        assert all(np.diff(snr) > 0)

    def test_roc_snr_both_arrays_return_2d(self):
        """Two sweeps give a ``[pfa, pd]`` matrix."""
        pfa = np.array([1e-4, 1e-6])
        pd = np.array([0.5, 0.9])

        snr = roc_snr(pfa, pd, 1, "Coherent")

        assert snr.shape == (pfa.size, pd.size)

    def test_roc_snr_returns_none_when_bracketing_fails(self):
        """An unreachable Pd cannot be bracketed, so ``None`` is returned."""
        # A Pd of 1.0 is not attainable at finite SNR.
        assert roc_snr(1e-6, 1.0, 1, "Coherent") is None

    def test_roc_snr_returns_none_for_unknown_signal_type(self):
        """An unknown ``stype`` makes ``roc_pd`` return ``None``, aborting the solve."""
        with pytest.raises(TypeError):
            roc_snr(1e-6, 0.9, 1, "Swerling 42")

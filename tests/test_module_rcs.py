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

import pytest
import numpy as np
import numpy.testing as npt

from radarsimpy.rt import rcs_sbr  # pylint: disable=no-name-in-module


# def test_rcs_sbr_basic():
#     """
#     Basic test case with a single target and simple parameters.
#     """
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([0, 0, 0]),
#         }
#     ]
#     f = 1e9
#     inc_phi = 0
#     inc_theta = 0
#     rcs = rcs_sbr(targets, f, inc_phi, inc_theta)
#     assert isinstance(rcs, float)  # Check if RCS is a float


# def test_rcs_sbr_multiple_targets():
#     """
#     Test with multiple targets.
#     """
#     targets = [
#         {
#             "model": "path/to/model1.obj",
#             "location": np.array([0, 0, 0]),
#         },
#         {
#             "model": "path/to/model2.obj",
#             "location": np.array([10, 0, 0]),
#         },
#     ]
#     f = 1e9
#     inc_phi = 0
#     inc_theta = 0
#     rcs = rcs_sbr(targets, f, inc_phi, inc_theta)
#     assert isinstance(rcs, float)  # Check if RCS is a float


# def test_rcs_sbr_different_angles():
#     """
#     Test with different incidence and observation angles.
#     """
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([0, 0, 0]),
#         }
#     ]
#     f = 1e9
#     inc_phi = 45
#     inc_theta = 30
#     obs_phi = 60
#     obs_theta = 45
#     rcs = rcs_sbr(targets, f, inc_phi, inc_theta, obs_phi=obs_phi, obs_theta=obs_theta)
#     assert isinstance(rcs, float)  # Check if RCS is a float


# def test_rcs_sbr_polarization():
#     """
#     Test with different polarizations.
#     """
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([0, 0, 0]),
#         }
#     ]
#     f = 1e9
#     inc_phi = 0
#     inc_theta = 0
#     inc_pol = [1, 0, 0]
#     obs_pol = [0, 1, 0]
#     rcs = rcs_sbr(targets, f, inc_phi, inc_theta, inc_pol=inc_pol, obs_pol=obs_pol)
#     assert isinstance(rcs, float)  # Check if RCS is a float


# def test_rcs_sbr_density():
#     """
#     Test with different ray densities.
#     """
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([0, 0, 0]),
#         }
#     ]
#     f = 1e9
#     inc_phi = 0
#     inc_theta = 0
#     density = 5
#     rcs = rcs_sbr(targets, f, inc_phi, inc_theta, density=density)
#     assert isinstance(rcs, float)  # Check if RCS is a float


# def test_rcs_sbr_freetier_limit():
#     """
#     Test the FreeTier limit on the number of targets.
#     """
#     targets = [
#         {
#             "model": "path/to/model1.obj",
#             "location": np.array([0, 0, 0]),
#         },
#         {
#             "model": "path/to/model2.obj",
#             "location": np.array([10, 0, 0]),
#         },
#         {
#             "model": "path/to/model3.obj",
#             "location": np.array([20, 0, 0]),
#         },
#         {
#             "model": "path/to/model4.obj",
#             "location": np.array([30, 0, 0]),
#         },
#     ]
#     f = 1e9
#     inc_phi = 0
#     inc_theta = 0
#     with pytest.raises(Exception):
#         rcs_sbr(targets, f, inc_phi, inc_theta)


def test_rcs_momostatic():
    """
    Tests the radar cross-section (RCS) calculation for a monostatic radar system.
    """
    phi = 0
    theta = 90
    freq = np.array([1e9, 3e9])
    pol = np.array([0, 0, 1])
    density = 1
    rcs = np.zeros_like(freq)
    target = {
        "model": "./models/plate5x5.stl",
        "location": (0, 0, 0),
    }
    for f_idx, f in enumerate(freq):
        rcs[f_idx] = 10 * np.log10(
            rcs_sbr([target], f, phi, theta, inc_pol=pol, density=density)
        )

    npt.assert_almost_equal(rcs, np.array([48.3, 59.2]), decimal=1)


def test_rcs_bistatic():
    """
    Tests the radar cross-section (RCS) calculation for a bistatic radar system.
    """
    phi = np.array([-30, -24, 65])
    theta = 90

    inc_phi = 30
    inc_theta = 90

    freq = 1e9
    pol = np.array([0, 0, 1])
    density = 1
    rcs = np.zeros_like(phi)

    target = {
        "model": "./models/plate5x5.stl",
        "location": (0, 0, 0),
    }
    for phi_idx, phi_ang in enumerate(phi):
        rcs[phi_idx] = 10 * np.log10(
            rcs_sbr(
                [target],
                freq,
                inc_phi=inc_phi,
                inc_theta=inc_theta,
                inc_pol=pol,
                obs_phi=phi_ang,
                obs_theta=theta,
                density=density,
            )
        )

    npt.assert_almost_equal(rcs, np.array([47, 34, 6]), decimal=0)

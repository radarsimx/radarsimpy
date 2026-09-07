# distutils: language = c++

"""
Execution device selection shared by the simulators

The execution policies are compile-time tags, so a GPU-enabled build has to
decide at runtime whether CUDA is actually usable. ``sim_radar``, ``sim_rcs``
and ``sim_lidar`` all face that question and must answer it identically, so the
rule lives here rather than in any one of them.

This file is textually included by ``simulator.pyx`` ahead of the three
simulators, which is what makes ``resolve_device()`` visible to all of them.

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

# Standard library imports
import warnings

# RadarSimX core components
from radarsimpy.includes.radarsimc cimport (
    gpu_available as _gpu_available_c,
    CUDA_BUILD
)


def gpu_available():
    """
    gpu_available()

    Whether this build can actually run a simulation on the GPU.

    Two separate things have to be true, and this reports their conjunction:
    the module has to have been compiled with CUDA support, and the machine has
    to expose at least one usable CUDA device. Either one missing means the
    simulators run on the CPU.

    This is what ``device="auto"`` resolves against, so it also answers "what
    will auto pick?" for ``sim_radar``, ``sim_rcs`` and ``sim_lidar``.

    The underlying device probe runs once and is cached for the lifetime of the
    process, so this is cheap to call repeatedly. It follows
    ``CUDA_VISIBLE_DEVICES``, which means a child process started with that set
    to ``-1`` will report ``False`` here even on a CUDA build.

    :return:
        ``True`` when a CUDA device is usable, ``False`` otherwise.
    :rtype: bool

    :example:
        >>> from radarsimpy.simulator import gpu_available
        >>> gpu_available()
        True
    """
    # Not `bool(...)`: the simulator files cimport the C++ `bool`, which shadows
    # the Python builtin across this shared namespace. A `bint` converts to a
    # genuine Python bool on return.
    cdef bint available = _gpu_available_c()
    return available


cdef str resolve_device(device):
    """
    Turn a user-facing device name into the execution policy to dispatch on.

    Shared by ``sim_radar``, ``sim_rcs`` and ``sim_lidar`` so the three agree on
    what each name means and on when a fallback is worth reporting.

    The execution policies are compile-time tags, so a GPU-enabled build would
    otherwise launch CUDA kernels on a machine that has no CUDA device.
    ``gpu_available()`` is false on a CPU-only build and caches its device
    probe, so this costs nothing to ask on every call.

    Call it exactly once per simulation: it may warn, and callers that need the
    answer in two places should keep the result rather than ask again.

    :param str device: ``"auto"``, ``"gpu"`` or ``"cpu"``, in any case.
    :return: ``"gpu"`` or ``"cpu"`` -- never ``"auto"``.
    :rtype: str
    :raises ValueError: If the name is not one of the three.
    """
    device_lower = device.lower()
    if device_lower not in ("auto", "gpu", "cpu"):
        raise ValueError(
            f"\nInvalid Device Selection\n"
            f"------------------------\n"
            f"The specified device '{device}' is not recognized.\n\n"
            f"Available devices:\n"
            f"- 'auto': Use the GPU when one is usable, otherwise the CPU\n"
            f"- 'gpu': Execute simulation on GPU (CUDA)\n"
            f"- 'cpu': Execute simulation on CPU\n\n"
            f"Please choose 'auto', 'gpu' or 'cpu'."
        )

    if device_lower == "auto":
        # Picking the CPU here is the documented behaviour, not a failed
        # request, so it is silent. That is what keeps the warning below
        # meaningful: it fires only when a caller asked for the GPU by name and
        # did not get it.
        return "gpu" if _gpu_available_c() else "cpu"

    if device_lower == "gpu" and not _gpu_available_c():
        # Warn in both cases. Staying silent on a CPU-only build means anyone
        # benchmarking with device="gpu" on a CPU wheel records CPU timings
        # believing they are GPU timings, with nothing on screen to say so.
        if CUDA_BUILD:
            reason = "No CUDA device was detected on this machine."
        else:
            reason = (
                "This build of radarsimpy was compiled without CUDA support "
                "(CPU-only build)."
            )
        warnings.warn(
            f"{reason} Running the simulation on the CPU instead, so any "
            "timing from this run is a CPU timing.",
            RuntimeWarning,
            # 1 is here, 2 is the sim_* function, 3 is the caller's own line.
            stacklevel=3
        )
        return "cpu"

    return device_lower

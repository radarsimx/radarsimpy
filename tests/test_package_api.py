"""
Tests for the top-level ``radarsimpy`` package surface

Covers the public metadata, the introspection helpers (``get_version``,
``get_info``, ``print_info``, ``check_installation``, ``hello``) and the
consistency of ``__all__`` with what the package actually exports.

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

import builtins
import re
import sys
import types
from importlib import metadata

import pytest

import radarsimpy as rs
from radarsimpy import lib


class TestPackageMetadata:
    """Version strings and other package metadata."""

    def test_get_version_matches_dunder(self):
        """``get_version()`` is the single source of truth for ``__version__``."""
        assert rs.get_version() == rs.__version__

    def test_version_is_semver(self):
        """The version follows ``MAJOR.MINOR.PATCH``."""
        assert re.fullmatch(r"\d+\.\d+\.\d+", rs.__version__), rs.__version__

    def test_metadata_fields_are_populated(self):
        """Author/email/url/license/description are non-empty strings."""
        for attr in (
            "__author__",
            "__email__",
            "__url__",
            "__license__",
            "__description__",
        ):
            value = getattr(rs, attr)
            assert isinstance(value, str)
            assert value.strip()

    @pytest.mark.parametrize("name", rs.__all__)
    def test_all_entries_are_exported(self, name):
        """Every name in ``__all__`` actually exists on the package."""
        assert hasattr(rs, name), f"__all__ advertises missing attribute {name!r}"


class TestGetInfo:
    """``radarsimpy.get_info``."""

    @staticmethod
    @pytest.fixture(scope="class")
    def info():
        """The info dictionary, built once for the class."""
        return rs.get_info()

    def test_top_level_keys(self, info):
        """All documented keys are present."""
        assert set(info) == {
            "package",
            "version",
            "author",
            "website",
            "python_version",
            "platform",
            "modules",
            "simulation_engines",
            "dependencies",
        }

    def test_values_track_package_metadata(self, info):
        """Reported metadata matches the module-level constants."""
        assert info["package"] == "RadarSimPy"
        assert info["version"] == rs.__version__
        assert info["author"] == rs.__author__
        assert info["website"] == rs.__url__

    def test_modules_and_engines_are_described(self, info):
        """Each advertised module/engine maps to a non-empty description."""
        assert {"radar", "transmitter", "receiver", "processing", "tools"} <= set(
            info["modules"]
        )
        assert set(info["simulation_engines"]) == {
            "sim_radar",
            "sim_lidar",
            "sim_rcs",
        }
        for mapping in (info["modules"], info["simulation_engines"]):
            for key, description in mapping.items():
                assert isinstance(description, str) and description.strip(), key

    def test_required_dependencies_are_reported_installed(self, info):
        """numpy and scipy are hard requirements, so they must be found."""
        deps = info["dependencies"]
        assert {"numpy", "scipy", "pymeshlab", "pyvista"} == set(deps)
        assert deps["numpy"] != "Not installed"
        assert deps["scipy"] != "Not installed"

    def test_optional_dependencies_report_version_or_absence(self, info):
        """Optional deps report either a version string or 'Not installed'."""
        for name in ("pymeshlab", "pyvista"):
            value = info["dependencies"][name]
            assert isinstance(value, str) and value.strip()

    def test_uninstallable_dependencies_are_reported_as_missing(self, monkeypatch):
        """Every dependency probe degrades to 'Not installed' on ImportError."""
        blocked = {"numpy", "scipy", "pymeshlab", "pyvista"}
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in blocked:
                raise ImportError(f"simulated missing {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        deps = rs.get_info()["dependencies"]

        assert set(deps) == blocked
        assert all(value == "Not installed" for value in deps.values())

    def test_dependency_without_dunder_version_does_not_raise(self, monkeypatch):
        """
        Not every distribution sets ``__version__`` on its top-level module --
        pymeshlab is one that does not. ``get_info`` must fall back to the
        distribution metadata instead of raising ``AttributeError``.
        """
        stub = types.ModuleType("pymeshlab")  # deliberately has no __version__
        monkeypatch.setitem(sys.modules, "pymeshlab", stub)

        version = rs.get_info()["dependencies"]["pymeshlab"]

        assert isinstance(version, str) and version.strip()
        assert version != "Not installed"

    def test_module_version_prefers_dunder_version(self):
        """A module that declares ``__version__`` is reported verbatim."""
        stub = types.ModuleType("stub_pkg")
        stub.__version__ = "1.2.3"

        assert rs._module_version(stub) == "1.2.3"

    def test_module_version_falls_back_to_distribution_metadata(self):
        """Without ``__version__`` the installed distribution is consulted."""
        stub = types.ModuleType("pytest")  # installed, so metadata resolves

        assert rs._module_version(stub) == metadata.version("pytest")

    def test_module_version_gives_up_gracefully(self):
        """An unknown module with no metadata yields a placeholder, not an error."""
        stub = types.ModuleType("not_a_real_distribution_abc123")

        assert rs._module_version(stub) == "Unknown version"

    def test_returns_a_fresh_dict(self, info):
        """Mutating the result must not corrupt later calls."""
        info["modules"]["radar"] = "tampered"
        assert rs.get_info()["modules"]["radar"] != "tampered"


class TestConsoleHelpers:
    """Functions whose product is printed output."""

    def test_print_info_reports_version_and_sections(self, capsys):
        """``print_info`` echoes the version and every section heading."""
        rs.print_info()
        out = capsys.readouterr().out

        assert rs.__version__ in out
        assert rs.__author__ in out
        assert "Core Modules" in out
        assert "Simulation Engines" in out
        assert "Dependencies" in out
        # Installed dependencies are rendered with a version prefix.
        assert "numpy" in out

    def test_check_installation_succeeds(self, capsys):
        """numpy/scipy are installed, so the check passes and says so."""
        assert rs.check_installation() is True
        assert "complete" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "missing, expected", [("scipy", "SciPy"), ("numpy", "NumPy")]
    )
    def test_check_installation_reports_missing_dependency(
        self, missing, expected, monkeypatch, capsys
    ):
        """A missing hard dependency is reported and flips the return value."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == missing:
                raise ImportError(f"simulated missing {missing}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert rs.check_installation() is False
        out = capsys.readouterr().out
        assert "Installation Issues Found" in out
        assert expected in out

    def test_hello_mentions_quick_start(self, capsys):
        """``hello`` prints the quick-start banner."""
        rs.hello()
        out = capsys.readouterr().out

        assert "Welcome to RadarSimPy" in out
        assert "Quick Start" in out
        assert "sim_radar" in out


# =============================================================================
# radarsimpy.lib - compiled extension diagnostics
# =============================================================================


class TestLibModule:
    """``radarsimpy.lib`` reports the state of the compiled Cython bridge."""

    def test_extensions_are_available(self):
        """The test suite requires a built package."""
        assert lib.is_available() is True

    def test_all_matches_availability(self):
        """The compiled entry points are exported when they are importable."""
        assert {"is_available", "get_lib_info"} <= set(lib.__all__)
        assert {"cp_GetTargetMesh", "cp_GetSceneStateChannels"} <= set(lib.__all__)

    def test_get_lib_info_available_branch(self):
        """The available branch describes the exported functions and types."""
        info = lib.get_lib_info()

        assert info["module"] == "radarsimpy.lib"
        assert info["available"] is True
        assert info["compiled_extensions"] is True
        assert set(info["functions"]) == {
            "cp_GetTargetMesh",
            "cp_GetSceneStateChannels",
        }
        assert "precision" in info["data_types"]
        assert "error" not in info

    def test_get_lib_info_unavailable_branch(self, monkeypatch):
        """The unavailable branch reports causes and remedies instead."""
        monkeypatch.setattr(lib, "_lib_available", False)
        monkeypatch.setattr(lib, "_import_error", "simulated import failure")

        info = lib.get_lib_info()

        assert info["available"] is False
        assert info["error"] == "simulated import failure"
        assert info["possible_causes"] and info["solutions"]
        assert "functions" not in info

    def test_print_lib_status_available(self, capsys):
        """Status output lists the available functions."""
        lib.print_lib_status()
        out = capsys.readouterr().out

        assert "Available" in out
        assert "cp_GetTargetMesh" in out
        assert "cp_GetSceneStateChannels" in out

    def test_print_lib_status_unavailable(self, monkeypatch, capsys):
        """Status output falls back to troubleshooting advice."""
        monkeypatch.setattr(lib, "_lib_available", False)
        monkeypatch.setattr(lib, "_import_error", "simulated import failure")

        lib.print_lib_status()
        out = capsys.readouterr().out

        assert "Not Available" in out
        assert "simulated import failure" in out
        assert "build_ext" in out

    def test_check_compilation_passes(self, capsys):
        """``check_compilation`` agrees with ``is_available``."""
        assert lib.check_compilation() is True
        assert "Passed" in capsys.readouterr().out

    def test_check_compilation_fails(self, monkeypatch, capsys):
        """A missing extension is reported with build instructions."""
        monkeypatch.setattr(lib, "_lib_available", False)

        assert lib.check_compilation() is False
        out = capsys.readouterr().out
        assert "Failed" in out
        assert "build_ext" in out

    def test_function_signatures_are_documented(self):
        """Every exported function has a signature, parameters and returns."""
        signatures = lib.get_function_signatures()

        assert set(signatures) == {"cp_GetTargetMesh", "cp_GetSceneStateChannels"}
        for name, entry in signatures.items():
            assert entry["signature"].startswith(name)
            assert entry["parameters"]
            assert entry["returns"]

    def test_function_signatures_when_unavailable(self, monkeypatch):
        """Without the extension there are no signatures to report."""
        monkeypatch.setattr(lib, "_lib_available", False)

        assert lib.get_function_signatures() == {"error": "Library not available"}

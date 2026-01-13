#!/usr/bin/env python3
"""
RadarSimPy Source Code Packaging Script

This script packages the RadarSimPy source code into a release-ready zip file.
It automatically extracts the version from the package and creates a clean
source distribution excluding build artifacts and temporary files.

Usage:
    python package_source.py

Output:
    dist/radarsimpy_source_<version>.zip

Author: RadarSimX
Website: https://radarsimx.com
"""

import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Set


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color

    @staticmethod
    def is_windows():
        return sys.platform == "win32"

    @classmethod
    def disable_on_windows(cls):
        """Disable colors on Windows if not supported"""
        if cls.is_windows():
            cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.NC = ""


def print_info(message: str) -> None:
    """Print info message"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def print_success(message: str) -> None:
    """Print success message"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def print_warning(message: str) -> None:
    """Print warning message"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def print_error(message: str) -> None:
    """Print error message"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}", file=sys.stderr)


def get_version() -> str:
    """
    Extract version from package __init__.py file

    Returns:
        Version string

    Raises:
        RuntimeError: If version cannot be extracted
    """
    version_file = Path("src") / "radarsimpy" / "__init__.py"

    if not version_file.exists():
        raise RuntimeError(f"Version file not found: {version_file}")

    try:
        content = version_file.read_text(encoding="utf-8")
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)

        if match:
            return match.group(1)
        else:
            raise RuntimeError(f"Could not find __version__ in {version_file}")

    except Exception as e:
        raise RuntimeError(f"Could not read version from {version_file}: {e}") from e


def get_exclude_patterns() -> tuple[Set[str], Set[str]]:
    """
    Get patterns for files and directories to exclude from packaging

    Returns:
        Tuple of (directory names to exclude, file patterns to exclude)
    """
    exclude_dirs = {
        # Simple directory names to exclude anywhere
        "__pycache__",
        "build",
        "dist",
        ".git",
        ".github",
        ".vscode",
        ".idea",
        "debug",
        "release",
        "build_logs",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".eggs",
        "node_modules",
        "env",
        "venv",
        ".env",
        ".venv",
        "htmlcov",
        ".ipynb_checkpoints",
        "_build",
        "cmake-build-debug",
        "cmake-build-release",
        # Specific paths to exclude (relative to project root)
        "./radarsimpy",  # Top-level radarsimpy folder (not src/radarsimpy)
        "./references",
        "./src/radarsimcpp/hdf5-lib-build/hdf5",
        "./src/radarsimcpp/hdf5-lib-build/hdf5lib",
        "./src/radarsimcpp/autocoder",
        "./src/radarsimcpp/entry",
    }

    exclude_files = {
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.dll",
        "*.dylib",
        "*.o",
        "*.obj",
        "*.exp",
        "*.log",
        "*.h5",
        ".DS_Store",
        "Thumbs.db",
        "*.swp",
        "*.swo",
        "*~",
        ".gitignore",
        ".gitmodules",
        "batch_build.bat",
        "batch_build.sh",
        "package_source.bat",
        "package_source.py",
        "package_source.sh",
    }

    return exclude_dirs, exclude_files


def should_exclude(path: Path, exclude_dirs: Set[str], exclude_files: Set[str]) -> bool:
    """
    Check if a path should be excluded from packaging

    Args:
        path: Path to check
        exclude_dirs: Set of directory names to exclude
        exclude_files: Set of file patterns to exclude

    Returns:
        True if path should be excluded
    """
    # Convert to string with forward slashes for consistent comparison
    path_str = path.as_posix()

    # Check if path matches any exclude pattern
    for exclude_pattern in exclude_dirs:
        # Handle full path patterns (e.g., './src/radarsimcpp/hdf5-lib-build/hdf5')
        if "/" in exclude_pattern or "\\" in exclude_pattern:
            # Normalize the pattern
            normalized_pattern = exclude_pattern.lstrip("./").lstrip(".\\")
            # Check if path starts with this pattern
            if (
                path_str.startswith(normalized_pattern + "/")
                or path_str == normalized_pattern
            ):
                return True
        # Handle simple directory names
        elif exclude_pattern in path.parts:
            return True

    # Check if filename matches any exclude pattern
    name = path.name
    for pattern in exclude_files:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif pattern.endswith("*"):
            if name.startswith(pattern[:-1]):
                return True
        elif name == pattern:
            return True

    return False


def copy_source_tree(
    src_dir: Path, dest_dir: Path, exclude_dirs: Set[str], exclude_files: Set[str]
) -> int:
    """
    Copy source tree to destination, excluding specified patterns

    Args:
        src_dir: Source directory
        dest_dir: Destination directory
        exclude_dirs: Set of directory names to exclude
        exclude_files: Set of file patterns to exclude

    Returns:
        Number of files copied
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_count = 0

    for item in src_dir.rglob("*"):
        relative_path = item.relative_to(src_dir)

        # Skip excluded paths
        if should_exclude(relative_path, exclude_dirs, exclude_files):
            continue

        dest_path = dest_dir / relative_path

        if item.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_path)
            file_count += 1

    return file_count


def create_zip_archive(source_dir: Path, output_path: Path) -> None:
    """
    Create zip archive from source directory

    Args:
        source_dir: Directory to archive
        output_path: Output zip file path
    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in source_dir.rglob("*"):
            if item.is_file():
                arcname = item.relative_to(source_dir.parent)
                zipf.write(item, arcname)


def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def main() -> int:
    """
    Main entry point for packaging script

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Disable colors on Windows if not in a modern terminal
    if Colors.is_windows() and not os.environ.get("WT_SESSION"):
        Colors.disable_on_windows()

    print_info("Starting RadarSimPy source code packaging...")

    try:
        # Get project root (script location)
        script_dir = Path(__file__).parent.resolve()
        os.chdir(script_dir)

        # Extract version
        version = get_version()
        print_info(f"Package version: {version}")

        # Define paths
        output_dir = Path("dist")
        version_clean = version.replace(".", "_").replace("-", "_")
        archive_name = f"radarsimpy_source_{version_clean}.zip"
        output_path = output_dir / archive_name
        temp_dir_name = f"radarsimpy_{version}".replace("-", "_").replace(".", "_")
        temp_dir = Path(temp_dir_name)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Output directory: {output_dir}")

        # Remove old archive if exists
        if output_path.exists():
            print_warning(f"Removing existing archive: {output_path}")
            output_path.unlink()

        # Remove temporary directory if exists
        if temp_dir.exists():
            print_info("Cleaning up old temporary directory...")
            shutil.rmtree(temp_dir)

        # Get exclusion patterns
        exclude_dirs, exclude_files = get_exclude_patterns()

        # Add the temp directory to exclusions to prevent infinite loop
        exclude_dirs.add(temp_dir.name)

        # Copy source files
        print_info("Copying source files...")
        file_count = copy_source_tree(Path("."), temp_dir, exclude_dirs, exclude_files)
        print_info(f"Copied {file_count} files")

        # Create zip archive
        print_info(f"Creating zip archive: {archive_name}")
        create_zip_archive(temp_dir, output_path)

        # Cleanup temporary directory
        print_info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)

        # Get archive size
        archive_size = output_path.stat().st_size
        size_display = format_size(archive_size)

        # Success message
        print_success("Package created successfully!")
        print()
        print("=" * 50)
        print(f"  Package: {archive_name}")
        print(f"  Version: {version}")
        print(f"  Size:    {size_display}")
        print(f"  Path:    {output_path}")
        print("=" * 50)
        print()
        print_info("Source package is ready for release!")

        return 0

    except Exception as e:
        print_error(f"Packaging failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

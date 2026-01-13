#!/bin/bash
# ==============================================================================
# RadarSimPy Source Code Packaging Script (Linux/macOS)
# ==============================================================================
#
# This script is a simple wrapper that calls the Python packaging script.
#
# Usage:
#   ./package_source.sh
#
# Output:
#   dist/radarsimpy_source_<version>.zip
#
# ==============================================================================

set -e  # Exit on error

# Get project root directory (script location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is required but not found. Please install Python 3.10 or higher."
    exit 1
fi

# Run the Python packaging script
python3 package_source.py

exit $?

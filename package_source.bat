@echo off
REM ==============================================================================
REM RadarSimPy Source Code Packaging Script (Windows)
REM ==============================================================================
REM
REM This script is a simple wrapper that calls the Python packaging script.
REM
REM Requirements:
REM   - Python 3.10 or higher
REM
REM Usage:
REM   package_source.bat
REM
REM Output:
REM   dist\radarsimpy_source_<version>.zip
REM
REM ==============================================================================

setlocal

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is required but not found. Please install Python 3.10 or higher.
    exit /b 1
)

REM Run the Python packaging script
python package_source.py

endlocal
exit /b %errorlevel%

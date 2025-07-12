#!/bin/bash

#===============================================================================
# Build Script for RadarSimPy - Linux Platform
#===============================================================================
#
# DESCRIPTION:
#   This script automates the build process for RadarSimPy on Linux systems.
#   It compiles the C++ library (libradarsimcpp.so) and builds Python extensions
#   using Cython, providing a complete build pipeline with comprehensive error
#   handling and logging.
#
# REQUIREMENTS:
#   - Linux operating system
#   - CMake 3.15 or higher
#   - Python 3.7 or higher
#   - GCC/G++ compiler
#   - CUDA toolkit (for GPU builds)
#   - Python packages: setuptools, Cython
#   - pytest (for running tests)
#
# FEATURES:
#   - Automatic dependency checking
#   - Parallel build support with auto-detection of CPU cores
#   - Comprehensive logging with timestamped log files
#   - Color-coded console output
#   - Robust error handling with cleanup procedures
#   - Support for both CPU and GPU architectures
#   - Configurable build tiers (standard/free)
#   - Optional unit testing with Google Test and pytest
#   - Customizable CMake arguments
#   - Build artifact management
#
# USAGE:
#   ./build_linux.sh [OPTIONS]
#
# OPTIONS:
#   --help              Show help message
#   --tier=TIER         Build tier: 'standard' or 'free' (default: standard)
#   --arch=ARCH         Build architecture: 'cpu' or 'gpu' (default: cpu)
#   --test=TEST         Enable unit tests: 'on' or 'off' (default: on)
#   --jobs=N            Number of parallel build jobs (default: auto-detect)
#   --clean=CLEAN       Clean build artifacts: 'true' or 'false' (default: true)
#   --verbose           Enable verbose output (default: true)
#   --cmake-args=ARGS   Additional CMake arguments
#
# EXAMPLES:
#   ./build_linux.sh                                    # Default build
#   ./build_linux.sh --tier=free --arch=gpu           # GPU build with free tier
#   ./build_linux.sh --jobs=8 --verbose               # 8-core parallel build
#   ./build_linux.sh --cmake-args="-DCUSTOM_FLAG=ON"  # Custom CMake flags
#
# EXIT CODES:
#   0  - Success
#   1  - General error (missing dependencies, validation failure, etc.)
#   130 - Interrupted by user (Ctrl+C)
#   >1 - Test failures (number indicates failed test suites)
#
# FILES CREATED:
#   - ./radarsimpy/                    # Output directory with built libraries
#   - ./build_YYYYMMDD_HHMMSS.log     # Timestamped build log
#   - ./src/radarsimcpp/build/        # CMake build directory
#
# NOTES:
#   - The script uses 'set -euo pipefail' for strict error handling
#   - All build artifacts are cleaned up automatically on failure
#   - Log files are preserved for debugging purposes
#   - GPU builds require NVIDIA CUDA toolkit to be installed
#   - The script validates all parameters before starting the build
#
#===============================================================================

# Exit on any error, undefined variables, and pipe failures
set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="${SCRIPT_DIR}/build_$(date +%Y%m%d_%H%M%S).log"
readonly BUILD_START_TIME=$(date +%s)

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
#
# log_info() - Displays an informational message with blue coloring
# Arguments:
#   $1 - The message to display
# Output:
#   Writes the message to both stdout and the log file with [INFO] prefix
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

#
# log_success() - Displays a success message with green coloring
# Arguments:
#   $1 - The success message to display
# Output:
#   Writes the message to both stdout and the log file with [SUCCESS] prefix
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

#
# log_warning() - Displays a warning message with yellow coloring
# Arguments:
#   $1 - The warning message to display
# Output:
#   Writes the message to both stdout and the log file with [WARNING] prefix
log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${LOG_FILE}"
}

#
# log_error() - Displays an error message with red coloring
# Arguments:
#   $1 - The error message to display
# Output:
#   Writes the message to both stdout and the log file with [ERROR] prefix
log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

#
# Help() - Displays comprehensive usage information and command line options
# Description:
#   Shows the script's usage syntax, available command line options with descriptions,
#   and practical examples of how to use the script with different configurations.
# Arguments:
#   None
# Output:
#   Prints formatted help text to stdout
# Exit:
#   This function is typically called before script exit when --help is specified
Help() {
    cat << EOF

Usage: build_linux.sh [OPTIONS]

Build script for RadarSimPy - A Radar Simulator Built with Python

OPTIONS:
    --help              Show this help message
    --tier=TIER         Build tier: 'standard' or 'free' (default: standard)
    --arch=ARCH         Build architecture: 'cpu' or 'gpu' (default: cpu)
    --test=TEST         Enable unit tests: 'on' or 'off' (default: on)
    --jobs=N            Number of parallel build jobs (default: auto-detect)
    --clean             Clean build artifacts before building (default: true)
    --verbose           Enable verbose output (default: false)
    --cmake-args=ARGS   Additional CMake arguments

EXAMPLES:
    $0                                  # Default build
    $0 --tier=free --arch=gpu         # GPU build with free tier
    $0 --jobs=4 --verbose              # Parallel build with verbose output
    $0 --cmake-args="-DCUSTOM_FLAG=ON" # Custom CMake arguments

EOF
}

#
# cleanup() - Signal handler for build process cleanup
# Description:
#   Handles cleanup operations when the script exits, either normally or due to
#   errors/interruptions. Logs appropriate error messages and ensures proper
#   exit code propagation.
# Arguments:
#   None (uses $? to get the exit code)
# Global Variables:
#   LOG_FILE - Path to the log file for error reporting
# Exit:
#   Exits with the same code that triggered the cleanup
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Build failed with exit code $exit_code"
        log_info "Check log file: ${LOG_FILE}"
    fi
    exit $exit_code
}

#
# detect_cores() - Automatically detects the number of CPU cores available
# Description:
#   Attempts to determine the number of CPU cores using multiple methods
#   for maximum compatibility across different Linux distributions and systems.
#   Falls back to a safe default if detection fails.
# Arguments:
#   None
# Output:
#   Prints the number of CPU cores to stdout
# Return:
#   Always returns 0 (success)
# Methods used (in order):
#   1. nproc command (most reliable on modern systems)
#   2. /proc/cpuinfo parsing (fallback for older systems)
#   3. Hard-coded fallback value of 4
detect_cores() {
    if command -v nproc &> /dev/null; then
        nproc
    elif [ -f /proc/cpuinfo ]; then
        grep -c ^processor /proc/cpuinfo
    else
        echo "4" # fallback
    fi
}

#
# command_exists() - Checks if a command is available in the system PATH
# Description:
#   Verifies whether a given command/executable is available and can be executed.
#   Used for dependency checking before attempting to use external tools.
# Arguments:
#   $1 - The command name to check for existence
# Return:
#   0 if command exists and is executable
#   1 if command is not found or not executable
# Example:
#   if command_exists cmake; then echo "CMake is available"; fi
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

#
# check_requirements() - Validates all system dependencies and requirements
# Description:
#   Performs comprehensive system dependency checking including:
#   - Required build tools (cmake, python3, gcc, g++)
#   - GPU-specific requirements (nvcc for CUDA builds)
#   - Python packages (setuptools, Cython)
#   Exits the script with error code 1 if any dependencies are missing.
# Arguments:
#   None
# Global Variables:
#   ARCH - Build architecture (used to determine if GPU tools are needed)
# Dependencies Checked:
#   - cmake: Build system generator
#   - python3: Python interpreter
#   - gcc/g++: C/C++ compilers
#   - nvcc: NVIDIA CUDA compiler (GPU builds only)
#   - Python setuptools and Cython packages
# Exit:
#   Exits with code 1 if any required dependencies are missing
check_requirements() {
    log_info "Checking system requirements..."
    
    local missing_deps=()
    
    # Check for required commands
    for cmd in cmake python3 gcc g++; do
        if ! command_exists "$cmd"; then
            missing_deps+=("$cmd")
        fi
    done
    
    # Check for GPU-specific requirements
    if [ "${ARCH,,}" == "gpu" ]; then
        if ! command_exists nvcc; then
            missing_deps+=("nvcc (CUDA toolkit)")
        fi
    fi
    
    # Check for Python packages
    if ! python3 -c "import setuptools, Cython" 2>/dev/null; then
        missing_deps+=("Python setuptools and Cython")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies:"
        printf '  - %s\n' "${missing_deps[@]}"
        exit 1
    fi
    
    log_success "All system requirements satisfied"
}

#
# parse_arguments() - Parses command line arguments and sets global configuration
# Description:
#   Processes all command line arguments passed to the script and sets global
#   configuration variables. Handles both parameter validation and default value
#   assignment. Also handles special cases like --help and automatic CPU detection.
# Arguments:
#   $@ - All command line arguments passed to the script
# Global Variables Set:
#   TIER - Build tier (standard/free)
#   ARCH - Build architecture (cpu/gpu)  
#   TEST - Unit test flag (on/off)
#   JOBS - Number of parallel build jobs
#   CLEAN - Clean build artifacts flag (true/false)
#   VERBOSE - Verbose output flag (true/false)
#   CMAKE_ARGS - Additional CMake arguments
# Supported Options:
#   --help: Shows help and exits
#   --tier=VALUE: Sets build tier
#   --arch=VALUE: Sets architecture
#   --test=VALUE: Enables/disables tests
#   --jobs=VALUE: Sets parallel job count
#   --clean=VALUE: Enables/disables cleanup
#   --verbose: Enables verbose output
#   --cmake-args=VALUE: Passes additional CMake arguments
# Exit:
#   Exits with code 0 on --help
#   Exits with code 1 on unknown options
parse_arguments() {
    # Default configuration values (make them global)
    TIER="standard"         # Build tier (standard/free)
    ARCH="cpu"             # Build architecture (cpu/gpu)
    TEST="on"              # Unit test flag (on/off)
    JOBS=""                # Number of parallel jobs (auto-detect if empty)
    CLEAN="true"           # Clean build artifacts before building
    VERBOSE="true"        # Enable verbose output
    CMAKE_ARGS=""          # Additional CMake arguments
    
    # Parse command line arguments
    # Supports --help, --tier, --arch, --test, --jobs, --clean, --verbose, and --cmake-args parameters
    for i in "$@"; do
        case $i in
            --help*)
                Help
                exit 0
                ;;
            --tier=*)
                TIER="${i#*=}"
                shift
                ;;
            --arch=*)
                ARCH="${i#*=}"
                shift
                ;;
            --test=*)
                TEST="${i#*=}"
                shift
                ;;
            --jobs=*)
                JOBS="${i#*=}"
                shift
                ;;
            --clean=*)
                CLEAN="${i#*=}"
                shift
                ;;
            --verbose*)
                VERBOSE="true"
                shift
                ;;
            --cmake-args=*)
                CMAKE_ARGS="${i#*=}"
                shift
                ;;
            --*)
                log_error "Unknown option: $i"
                echo "Use --help for usage information"
                exit 1
                ;;
            *)
                ;;
        esac
    done
    
    # Set number of jobs if not specified
    if [ -z "$JOBS" ]; then
        JOBS=$(detect_cores)
        log_info "Auto-detected $JOBS CPU cores for parallel build"
    fi
}

#
# validate_parameters() - Validates all parsed command line parameters
# Description:
#   Performs comprehensive validation of all configuration parameters set by
#   parse_arguments(). Checks for valid values, proper formatting, and logical
#   constraints. Accumulates all validation errors before reporting them.
# Arguments:
#   None
# Global Variables Used:
#   TIER - Validated against 'standard' and 'free'
#   ARCH - Validated against 'cpu' and 'gpu'
#   TEST - Validated against 'on' and 'off'
#   JOBS - Validated as positive integer
#   CLEAN - Validated against 'true' and 'false'
# Validation Rules:
#   - TIER: Must be 'standard' or 'free' (case insensitive)
#   - ARCH: Must be 'cpu' or 'gpu' (case insensitive)
#   - TEST: Must be 'on' or 'off' (case insensitive)
#   - JOBS: Must be positive integer >= 1
#   - CLEAN: Must be 'true' or 'false' (case insensitive)
# Exit:
#   Exits with code 1 if any validation errors are found
validate_parameters() {
    local errors=0
    
    # Validate tier parameter
    if [[ "${TIER,,}" != "standard" && "${TIER,,}" != "free" ]]; then
        log_error "Invalid --tier parameter: '$TIER'. Choose 'free' or 'standard'"
        errors=$((errors + 1))
    fi
    
    # Validate architecture parameter
    if [[ "${ARCH,,}" != "cpu" && "${ARCH,,}" != "gpu" ]]; then
        log_error "Invalid --arch parameter: '$ARCH'. Choose 'cpu' or 'gpu'"
        errors=$((errors + 1))
    fi
    
    # Validate test parameter
    if [[ "${TEST,,}" != "on" && "${TEST,,}" != "off" ]]; then
        log_error "Invalid --test parameter: '$TEST'. Choose 'on' or 'off'"
        errors=$((errors + 1))
    fi
    
    # Validate jobs parameter
    if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
        log_error "Invalid --jobs parameter: '$JOBS'. Must be a positive integer"
        errors=$((errors + 1))
    fi
    
    # Validate clean parameter
    if [[ "${CLEAN,,}" != "true" && "${CLEAN,,}" != "false" ]]; then
        log_error "Invalid --clean parameter: '$CLEAN'. Choose 'true' or 'false'"
        errors=$((errors + 1))
    fi
    
    if [ $errors -gt 0 ]; then
        log_error "Parameter validation failed with $errors error(s)"
        exit 1
    fi
    
    log_success "All parameters validated successfully"
}

#
# display_banner() - Shows project banner and current build configuration
# Description:
#   Displays the RadarSimPy project banner with ASCII art logo and shows
#   the current build configuration settings. Provides visual confirmation
#   of all build parameters before the build process begins.
# Arguments:
#   None
# Global Variables Used:
#   TIER - Build tier setting
#   ARCH - Architecture setting  
#   TEST - Test execution setting
#   JOBS - Number of parallel jobs
#   CLEAN - Clean build setting
#   VERBOSE - Verbose output setting
#   LOG_FILE - Log file path
#   CMAKE_ARGS - Additional CMake arguments (if any)
# Output:
#   Prints formatted banner and configuration to stdout
display_banner() {
    echo
    echo "=========================================="
    echo "RadarSimPy - A Radar Simulator Built with Python"
    echo "Copyright (C) 2018 - PRESENT  radarsimx.com"
    echo "E-mail: info@radarsimx.com"
    echo "Website: https://radarsimx.com"
    echo "=========================================="
    echo
    echo "██████╗  █████╗ ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ███╗██╗  ██╗"
    echo "██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗ ████║╚██╗██╔╝"
    echo "██████╔╝███████║██║  ██║███████║██████╔╝███████╗██║██╔████╔██║ ╚███╔╝ "
    echo "██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗╚════██║██║██║╚██╔╝██║ ██╔██╗ "
    echo "██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║███████║██║██║ ╚═╝ ██║██╔╝ ██╗"
    echo "╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝"
    echo
    echo "Build Configuration:"
    echo "  - Tier: ${TIER^^}"
    echo "  - Architecture: ${ARCH^^}"
    echo "  - Tests: ${TEST^^}"
    echo "  - Parallel Jobs: ${JOBS}"
    echo "  - Clean Build: ${CLEAN^^}"
    echo "  - Verbose: ${VERBOSE^^}"
    echo "  - Log File: ${LOG_FILE}"
    [ -n "$CMAKE_ARGS" ] && echo "  - CMake Args: ${CMAKE_ARGS}"
    echo
}

#
# clean_build_artifacts() - Removes previous build artifacts and generated files
# Description:
#   Performs cleanup of build directories and generated files from previous
#   build attempts. Only executes if CLEAN is set to 'true'. Removes both
#   build directories and generated source files to ensure clean builds.
# Arguments:
#   None
# Global Variables Used:
#   CLEAN - Controls whether cleanup is performed
# Directories/Files Removed:
#   - ./src/radarsimcpp/build/ - C++ build directory
#   - ./radarsimpy/ - Python package output directory
#   - ./build/ - General build directory
#   - *.c, *.cpp, *.html files in ./src/radarsimpy/ - Generated files
# Behavior:
#   - If CLEAN='true': Performs full cleanup
#   - If CLEAN='false': Skips cleanup and logs message
clean_build_artifacts() {
    if [ "${CLEAN,,}" == "true" ]; then
        log_info "Cleaning previous build artifacts..."
        
        # Remove build directories and files
        rm -rf ./src/radarsimcpp/build
        rm -rf ./radarsimpy
        rm -rf ./build
        
        # Remove generated source files
        find ./src/radarsimpy -name "*.c" -delete
        find ./src/radarsimpy -name "*.cpp" -delete
        find ./src/radarsimpy -name "*.html" -delete
        
        log_success "Build artifacts cleaned"
    else
        log_info "Skipping build artifact cleanup"
    fi
}

#
# build_cpp_library() - Builds the C++ library (libradarsimcpp.so)
# Description:
#   Compiles the RadarSimCpp C++ library using CMake build system. Configures
#   build options based on architecture (CPU/GPU) and test settings. Supports
#   parallel compilation and both verbose and quiet build modes.
# Arguments:
#   None
# Global Variables Used:
#   ARCH - Determines GPU build flags
#   TEST - Controls Google Test compilation
#   CMAKE_ARGS - Additional CMake arguments
#   JOBS - Number of parallel compilation jobs
#   VERBOSE - Controls build output verbosity
#   WORKPATH - Working directory to return to
#   LOG_FILE - Log file for quiet builds
# Build Process:
#   1. Creates build directory at ./src/radarsimcpp/build
#   2. Configures CMake with appropriate flags
#   3. Builds with specified parallel job count
#   4. Times the build process
#   5. Returns to original working directory
# CMake Flags Set:
#   - CMAKE_BUILD_TYPE=Release (always)
#   - GPU_BUILD=ON (if ARCH=gpu)
#   - GTEST=ON/OFF (based on TEST setting)
#   - Custom flags from CMAKE_ARGS
build_cpp_library() {
    local build_start=$(date +%s)
    log_info "Building libradarsimcpp.so with ${ARCH^^} architecture..."
    
    # Create build directory
    mkdir -p ./src/radarsimcpp/build
    cd ./src/radarsimcpp/build
    
    # Prepare CMake arguments
    local cmake_args="-DCMAKE_BUILD_TYPE=Release"
    
    # Add architecture-specific flags
    if [ "${ARCH,,}" == "gpu" ]; then
        cmake_args+=" -DGPU_BUILD=ON"
    fi
    
    # Add test flags
    if [ "${TEST,,}" == "on" ]; then
        cmake_args+=" -DGTEST=ON"
    else
        cmake_args+=" -DGTEST=OFF"
    fi
    
    # Add custom CMake arguments
    if [ -n "$CMAKE_ARGS" ]; then
        cmake_args+=" $CMAKE_ARGS"
    fi
    
    # Configure with CMake
    log_info "Configuring CMake with args: $cmake_args"
    if [ "$VERBOSE" == "true" ]; then
        cmake $cmake_args ..
    else
        cmake $cmake_args .. >> "${LOG_FILE}" 2>&1
    fi
    
    # Build with parallel jobs
    log_info "Building with $JOBS parallel jobs..."
    if [ "$VERBOSE" == "true" ]; then
        cmake --build . --parallel "$JOBS"
    else
        cmake --build . --parallel "$JOBS" >> "${LOG_FILE}" 2>&1
    fi
    
    local build_end=$(date +%s)
    local build_time=$((build_end - build_start))
    log_success "C++ library built successfully in ${build_time}s"
    
    cd "$WORKPATH"
}

#
# build_python_extensions() - Builds Python extensions using Cython
# Description:
#   Compiles Python extension modules using Cython and the setup.py build system.
#   Links against the previously built C++ library and creates platform-specific
#   binary extensions for Python import.
# Arguments:
#   None
# Global Variables Used:
#   WORKPATH - Working directory for build
#   VERBOSE - Controls build output verbosity
#   TIER - Build tier passed to setup.py
#   ARCH - Architecture setting passed to setup.py
#   LOG_FILE - Log file for quiet builds
# Build Process:
#   1. Changes to working directory
#   2. Invokes setup.py build_ext with tier and architecture flags
#   3. Times the build process
#   4. Handles both verbose and quiet build modes
# Output:
#   Creates compiled Python extension modules in the current directory
build_python_extensions() {
    local build_start=$(date +%s)
    log_info "Building Python extensions with Cython..."
    
    cd "$WORKPATH"
    
    # Build Python extensions
    if [ "$VERBOSE" == "true" ]; then
        python3 setup.py build_ext -b ./ --tier "${TIER}" --arch "${ARCH}"
    else
        python3 setup.py build_ext -b ./ --tier "${TIER}" --arch "${ARCH}" >> "${LOG_FILE}" 2>&1
    fi
    
    local build_end=$(date +%s)
    local build_time=$((build_end - build_start))
    log_success "Python extensions built successfully in ${build_time}s"
}

#
# install_libraries() - Copies built libraries to final installation directory
# Description:
#   Installs all built components (Python files, shared libraries, and support
#   files) into the ./radarsimpy/ directory structure. Creates the necessary
#   directory hierarchy and handles missing files gracefully.
# Arguments:
#   None
# Installation Process:
#   1. Creates ./radarsimpy/lib/ directory structure
#   2. Copies Python source files from ./src/radarsimpy/
#   3. Copies library __init__.py file
#   4. Copies shared libraries (.so files) from build directory
# File Sources:
#   - Python files: ./src/radarsimpy/*.py
#   - Library init: ./src/radarsimpy/lib/__init__.py  
#   - Shared libs: ./src/radarsimcpp/build/*.so
# Error Handling:
#   Uses '|| true' to continue gracefully if some files are missing
install_libraries() {
    log_info "Installing library files to ./radarsimpy..."
    
    # Create radarsimpy directory structure
    mkdir -p ./radarsimpy/lib
    
    # Copy Python files
    if [ -d "./src/radarsimpy" ]; then
        cp ./src/radarsimpy/*.py ./radarsimpy/ 2>/dev/null || true
    fi
    
    # Copy lib init file
    if [ -f "./src/radarsimpy/lib/__init__.py" ]; then
        cp ./src/radarsimpy/lib/__init__.py ./radarsimpy/lib/
    fi
    
    # Copy shared libraries
    if [ -d "./src/radarsimcpp/build" ]; then
        find ./src/radarsimcpp/build -name "*.so" -exec cp {} ./radarsimpy/ \; 2>/dev/null || true
    fi
    
    log_success "Library files installed successfully"
}

#
# cleanup_build_files() - Removes intermediate build files and directories
# Description:
#   Cleans up intermediate build artifacts that are no longer needed after
#   the build process completes. Removes temporary directories and generated
#   source files while preserving the final installation.
# Arguments:
#   None
# Files/Directories Removed:
#   - ./build/ - Intermediate build directory
#   - *.c files in ./src/radarsimpy/ - Generated C source files
#   - *.cpp files in ./src/radarsimpy/ - Generated C++ source files  
#   - *.html files in ./src/radarsimpy/ - Generated documentation files
# Error Handling:
#   Uses '|| true' to continue gracefully if files don't exist
# Purpose:
#   Reduces disk space usage by removing temporary build artifacts
cleanup_build_files() {
    log_info "Cleaning up intermediate build files..."
    
    # Remove intermediate build directories
    rm -rf build
    
    # Remove generated source files
    find ./src/radarsimpy -name "*.c" -delete 2>/dev/null || true
    find ./src/radarsimpy -name "*.cpp" -delete 2>/dev/null || true
    find ./src/radarsimpy -name "*.html" -delete 2>/dev/null || true
    
    log_success "Intermediate build files cleaned"
}

#
# run_tests() - Executes test suites if testing is enabled
# Description:
#   Runs both C++ unit tests (Google Test) and Python unit tests (pytest)
#   if the TEST flag is set to 'on'. Tracks test failures and provides
#   comprehensive test result reporting with timing information.
# Arguments:
#   None
# Global Variables Used:
#   TEST - Controls whether tests are executed
#   VERBOSE - Controls test output verbosity
#   LOG_FILE - Log file for quiet test runs
# Test Suites:
#   1. C++ Unit Tests:
#      - Executable: ./src/radarsimcpp/build/radarsimcpp_test
#      - Framework: Google Test
#   2. Python Unit Tests:
#      - Command: pytest
#      - Framework: pytest
# Return Codes:
#   0 - All tests passed or tests disabled
#   >0 - Number of failed test suites
# Behavior:
#   - If TEST='off': Skips all tests
#   - If TEST='on': Runs available test suites
#   - Missing test executables are reported as warnings, not errors
run_tests() {
    if [ "${TEST,,}" == "on" ]; then
        local test_start=$(date +%s)
        local test_failures=0
        
        log_info "Running test suite..."
        
        # Run C++ unit tests using Google Test
        if [ -f "./src/radarsimcpp/build/radarsimcpp_test" ]; then
            log_info "Running C++ unit tests..."
            if [ "$VERBOSE" == "true" ]; then
                ./src/radarsimcpp/build/radarsimcpp_test
            else
                ./src/radarsimcpp/build/radarsimcpp_test >> "${LOG_FILE}" 2>&1
            fi
            
            if [ $? -eq 0 ]; then
                log_success "C++ tests passed"
            else
                log_error "C++ tests failed"
                test_failures=$((test_failures + 1))
            fi
        else
            log_warning "C++ test executable not found, skipping C++ tests"
        fi
        
        # Run Python unit tests using pytest
        if command_exists pytest; then
            log_info "Running Python unit tests..."
            if [ "$VERBOSE" == "true" ]; then
                pytest -v
            else
                pytest >> "${LOG_FILE}" 2>&1
            fi
            
            if [ $? -eq 0 ]; then
                log_success "Python tests passed"
            else
                log_error "Python tests failed"
                test_failures=$((test_failures + 1))
            fi
        else
            log_warning "pytest not found, skipping Python tests"
        fi
        
        local test_end=$(date +%s)
        local test_time=$((test_end - test_start))
        
        if [ $test_failures -eq 0 ]; then
            log_success "All tests passed in ${test_time}s"
        else
            log_error "$test_failures test suite(s) failed"
            return $test_failures
        fi
    else
        log_info "Tests disabled, skipping test execution"
    fi
    
    return 0
}

#
# display_summary() - Shows final build summary and statistics
# Description:
#   Displays a comprehensive summary of the completed build process including
#   total build time, configuration settings, and output locations. Provides
#   a final status report for the user.
# Arguments:
#   None
# Global Variables Used:
#   BUILD_START_TIME - Start time for total duration calculation
#   TIER - Build tier setting
#   ARCH - Architecture setting
#   TEST - Test execution setting
#   JOBS - Number of parallel jobs used
#   VERBOSE - Verbose output setting (affects log file display)
#   LOG_FILE - Log file location
# Output Information:
#   - Total build time in seconds
#   - Configuration summary
#   - Output directory location
#   - Log file location (if not in verbose mode)
display_summary() {
    local build_end=$(date +%s)
    local total_time=$((build_end - BUILD_START_TIME))
    
    echo
    echo "=========================================="
    echo "BUILD SUMMARY"
    echo "=========================================="
    echo "  Total Build Time: ${total_time}s"
    echo "  Configuration:"
    echo "    - Tier: ${TIER^^}"
    echo "    - Architecture: ${ARCH^^}"
    echo "    - Tests: ${TEST^^}"
    echo "    - Parallel Jobs: ${JOBS}"
    echo "  Output Directory: ./radarsimpy"
    [ "$VERBOSE" != "true" ] && echo "  Log File: ${LOG_FILE}"
    echo "=========================================="
    echo
}

#
# main() - Main execution function that orchestrates the entire build process
# Description:
#   Central coordinator function that manages the complete build workflow.
#   Sets up signal handlers, parses arguments, validates configuration,
#   and executes all build steps in the correct order. Handles error
#   propagation and provides final status reporting.
# Arguments:
#   $@ - All command line arguments passed to the script
# Global Variables Set:
#   WORKPATH - Current working directory
#   (All other globals set by parse_arguments)
# Build Workflow:
#   1. Set up signal handlers for cleanup
#   2. Parse and validate command line arguments
#   3. Check system requirements
#   4. Display build configuration
#   5. Clean previous build artifacts
#   6. Build C++ library
#   7. Build Python extensions
#   8. Install libraries to final location
#   9. Clean up intermediate files
#   10. Run test suites (if enabled)
#   11. Display build summary
# Return Codes:
#   0 - Build completed successfully
#   >0 - Build failed (error code indicates failure type)
# Error Handling:
#   - Signal handlers ensure cleanup on interruption
#   - Test failures are captured and reported
#   - All errors are logged with appropriate messages
main() {
    local return_code=0

    # Set up signal handlers
    trap cleanup EXIT
    trap 'log_error "Build interrupted by user"; exit 130' INT TERM

    # Parse command line arguments
    parse_arguments "$@"

    # Store current working directory
    readonly WORKPATH=$(pwd)
    log_info "Working directory: ${WORKPATH}"

    # Run parameter validation and system checks
    validate_parameters
    check_requirements

    display_banner

    clean_build_artifacts

    build_cpp_library

    build_python_extensions

    install_libraries

    cleanup_build_files

    # Run tests and capture return code
    if ! run_tests; then
        return_code=$?
    fi
    
    # Display build summary
    display_summary
    
    if [ $return_code -eq 0 ]; then
        log_success "Build completed successfully!"
    else
        log_error "Build completed with errors (exit code: $return_code)"
    fi
    
    return $return_code
}

# Execute main function
main "$@"
exit $?

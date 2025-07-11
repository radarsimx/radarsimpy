#!/bin/bash

#===============================================================================
# Build Script for RadarSimPy - macOS Platform
#===============================================================================
#
# DESCRIPTION:
#   This script automates the build process for RadarSimPy on macOS systems.
#   It compiles the C++ library (libradarsimcpp.dylib) and builds Python 
#   extensions using Cython, providing a complete build pipeline with 
#   comprehensive error handling and logging optimized for macOS.
#
# REQUIREMENTS:
#   - macOS 10.14 (Mojave) or later
#   - Xcode Command Line Tools
#   - CMake 3.15 or higher
#   - Python 3.7 or higher
#   - Clang/Clang++ compiler (comes with Xcode)
#   - CUDA toolkit (for GPU builds, if supported)
#   - Python packages: setuptools, Cython
#   - pytest (for running tests)
#
# FEATURES:
#   - Automatic dependency checking including macOS-specific requirements
#   - Parallel build support with auto-detection of CPU cores via sysctl
#   - Comprehensive logging with timestamped log files
#   - Color-coded console output
#   - Robust error handling with cleanup procedures
#   - Support for both CPU and GPU architectures (where supported)
#   - Configurable build tiers (standard/free)
#   - Optional unit testing with Google Test and pytest
#   - Customizable CMake arguments
#   - Build artifact management
#   - macOS-specific dynamic library handling (.dylib files)
#
# USAGE:
#   ./build_macos.sh [OPTIONS]
#
# OPTIONS:
#   --help              Show help message
#   --tier=TIER         Build tier: 'standard' or 'free' (default: standard)
#   --arch=ARCH         Build architecture: 'cpu' or 'gpu' (default: cpu)
#   --test=TEST         Enable unit tests: 'on' or 'off' (default: on)
#   --jobs=N            Number of parallel build jobs (default: auto-detect)
#   --clean=CLEAN       Clean build artifacts: 'true' or 'false' (default: true)
#   --verbose           Enable verbose output (default: false)
#   --cmake-args=ARGS   Additional CMake arguments
#
# EXAMPLES:
#   ./build_macos.sh                                    # Default build
#   ./build_macos.sh --tier=free --arch=gpu           # GPU build with free tier
#   ./build_macos.sh --jobs=8 --verbose               # 8-core parallel build
#   ./build_macos.sh --cmake-args="-DCUSTOM_FLAG=ON"  # Custom CMake flags
#
# EXIT CODES:
#   0  - Success
#   1  - General error (missing dependencies, validation failure, etc.)
#   130 - Interrupted by user (Ctrl+C)
#   >1 - Test failures (number indicates failed test suites)
#
# FILES CREATED:
#   - ./radarsimpy/                         # Output directory with built libraries
#   - ./build_macos_YYYYMMDD_HHMMSS.log    # Timestamped build log
#   - ./src/radarsimcpp/build/             # CMake build directory
#
# MACOS-SPECIFIC NOTES:
#   - Uses Clang/Clang++ compiler instead of GCC
#   - Creates .dylib files instead of .so files
#   - Requires Xcode Command Line Tools for development headers
#   - Uses sysctl for CPU core detection
#   - Optimized for Apple Silicon and Intel processors
#   - GPU support depends on CUDA availability and compatibility
#
# NOTES:
#   - The script uses 'set -euo pipefail' for strict error handling
#   - All build artifacts are cleaned up automatically on failure
#   - Log files are preserved for debugging purposes
#   - The script validates all parameters before starting the build
#   - Compatible with both Intel and Apple Silicon Macs
#
#===============================================================================

# Exit on any error, undefined variables, and pipe failures
set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="${SCRIPT_DIR}/build_macos_$(date +%Y%m%d_%H%M%S).log"
readonly BUILD_START_TIME=$(date +%s)

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

# Function to display help information and usage instructions
Help() {
    cat << EOF

Usage: build_macos.sh [OPTIONS]

Build script for RadarSimPy - A Radar Simulator Built with Python (macOS)

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

# Default configuration values
TIER="standard"         # Build tier (standard/free)
ARCH="cpu"             # Build architecture (cpu/gpu)
TEST="on"              # Unit test flag (on/off)
JOBS=""                # Number of parallel jobs (auto-detect if empty)
CLEAN="true"           # Clean build artifacts before building
VERBOSE="false"        # Enable verbose output
CMAKE_ARGS=""          # Additional CMake arguments

# Cleanup function for signal handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Build failed with exit code $exit_code"
        log_info "Check log file: ${LOG_FILE}"
    fi
    exit $exit_code
}

# Set up signal handlers
trap cleanup EXIT
trap 'log_error "Build interrupted by user"; exit 130' INT TERM

# Function to detect number of CPU cores
detect_cores() {
    if command -v sysctl &> /dev/null; then
        sysctl -n hw.ncpu
    elif command -v nproc &> /dev/null; then
        nproc
    else
        echo "4" # fallback
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    local missing_deps=()
    
    # Check for required commands
    for cmd in cmake python3 clang++; do
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
    
    # Check for macOS-specific requirements
    if ! xcode-select -p &> /dev/null; then
        missing_deps+=("Xcode Command Line Tools")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies:"
        printf '  - %s\n' "${missing_deps[@]}"
        exit 1
    fi
    
    log_success "All system requirements satisfied"
}
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

# Validate parameters
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

# Run parameter validation and system checks
validate_parameters
check_requirements

# Display project banner and build configuration
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
    echo "Build Configuration (macOS):"
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

display_banner

# Store current working directory
readonly WORKPATH=$(pwd)
log_info "Working directory: ${WORKPATH}"

# Clean up previous build artifacts
clean_build_artifacts() {
    if [ "${CLEAN,,}" == "true" ]; then
        log_info "Cleaning previous build artifacts..."
        
        # Remove build directories and files
        rm -rf ./src/radarsimcpp/build
        rm -rf ./radarsimpy
        rm -rf ./build
        
        # Remove generated source files
        find ./src -name "*.c" -delete
        find ./src -name "*.cpp" -delete
        find ./src -name "*.html" -delete
        
        log_success "Build artifacts cleaned"
    else
        log_info "Skipping build artifact cleanup"
    fi
}

clean_build_artifacts

# Build the C++ library (macOS specific - creates .dylib files)
build_cpp_library() {
    local build_start=$(date +%s)
    log_info "Building libradarsimcpp.dylib with ${ARCH^^} architecture..."
    
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

build_cpp_library

# Build Python extensions using Cython
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

build_python_extensions

# Copy library files to radarsimpy directory (macOS specific - handles .dylib files)
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
    
    # Copy dynamic libraries (macOS uses .dylib instead of .so)
    if [ -d "./src/radarsimcpp/build" ]; then
        find ./src/radarsimcpp/build -name "*.dylib" -exec cp {} ./radarsimpy/ \; 2>/dev/null || true
    fi
    
    log_success "Library files installed successfully"
}

install_libraries

# Clean up intermediate build files
cleanup_build_files() {
    log_info "Cleaning up intermediate build files..."
    
    # Remove intermediate build directories
    rm -rf build
    
    # Remove generated source files
    find ./src -name "*.c" -delete 2>/dev/null || true
    find ./src -name "*.cpp" -delete 2>/dev/null || true
    find ./src -name "*.html" -delete 2>/dev/null || true
    
    log_success "Intermediate build files cleaned"
}

cleanup_build_files

# Run tests if enabled
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

# Display build summary
display_summary() {
    local build_end=$(date +%s)
    local total_time=$((build_end - BUILD_START_TIME))
    
    echo
    echo "=========================================="
    echo "BUILD SUMMARY (macOS)"
    echo "=========================================="
    echo "  Total Build Time: ${total_time}s"
    echo "  Configuration:"
    echo "    - Tier: ${TIER^^}"
    echo "    - Architecture: ${ARCH^^}"
    echo "    - Tests: ${TEST^^}"
    echo "    - Parallel Jobs: ${JOBS}"
    echo "  Output Directory: ./radarsimpy"
    echo "  Log File: ${LOG_FILE}"
    echo "=========================================="
    echo
}

# Main execution
main() {
    local return_code=0
    
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
main
exit $?

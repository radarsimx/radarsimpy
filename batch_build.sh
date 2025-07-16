#!/bin/bash
#
# batch_build.sh - Linux Batch Build Script for RadarSimPy
#
# DESCRIPTION:
#   Automates the RadarSimPy build across multiple Python conda environments
#   on Linux. Compiles the C++ library, builds Python extensions, and packages
#   distributions for different tiers (free/standard) and architectures (CPU/GPU).
#
# USAGE:
#   ./batch_build.sh [OPTIONS]
#
# OPTIONS:
#   --tier=<free|standard|both>   Build tier (default: both)
#   --arch=<cpu|gpu|both>         Target architecture (default: cpu)
#   --skip-tests                  Skip running unit tests (default: off)
#   --verbose                     Enable verbose logging (default: off)
#   --jobs=<auto|N>               Number of parallel jobs (default: auto)
#   --help                        Show this help message and exit
#
# EXAMPLES:
#   ./batch_build.sh --tier=free --arch=gpu --jobs=8
#   ./batch_build.sh --arch=both --skip-tests
#   ./batch_build.sh --tier=both --arch=both --verbose
#
# EXIT CODES:
#   0   Success
#   1   General error (invalid args, missing deps, build failure)
#  >1   Test failures (number of failed test suites)
#
# FILES GENERATED:
#   build_logs/linux_batch_build_log_YYYYMMDD_HHMMSS.log
#
#===============================================================================

# Configuration
SCRIPT_VERSION="2.0"
PYTHON_VERSIONS="py39 py310 py311 py312 py313"
BUILD_LOG_DATE=$(date +%Y%m%d)
BUILD_LOG_TIME=$(date +%H%M%S)
BUILD_LOG="$(dirname "$(readlink -f "$0")")/build_logs/linux_batch_build_log_${BUILD_LOG_DATE}_${BUILD_LOG_TIME}.log"

# Create build_logs directory if it doesn't exist
mkdir -p "$(dirname "$(readlink -f "$0")")/build_logs"

# Parse command line arguments
TIER="both"
ARCH="cpu"
SKIP_TESTS="false"
VERBOSE="false"
JOBS="auto"

# Function to show help
# show_help() - Display usage information, options, and examples
# Description:
#   Prints detailed help text including script usage, options, examples,
#   and exit codes.
# Arguments:
#   None
# Exit:
#   Exits with code 0 when invoked.
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
  --tier=<free|standard|both>   Build tier (default: both)
  --arch=<cpu|gpu|both>         Target architecture (default: cpu)
  --skip-tests                  Skip running unit tests
  --verbose                     Enable verbose output
  --jobs=<auto|N>               Number of parallel jobs (default: auto)
  --help                        Show this help message and exit

EXAMPLES:
  $0 --tier=free --arch=cpu
  $0 --tier=standard --arch=gpu --jobs=8
  $0 --arch=both --skip-tests --jobs=auto
  $0 --tier=both --arch=both --jobs=4
  $0 --verbose

EXIT CODES:
  0   Success
  1   Invalid arguments or build failure
 >1   Number of failed test suites

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tier=*)
            TIER="${1#*=}"
            shift
            ;;
        --arch=*)
            ARCH="${1#*=}"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        --jobs=*)
            JOBS="${1#*=}"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ "$TIER" != "free" && "$TIER" != "standard" && "$TIER" != "both" ]]; then
    echo "[ERROR] Invalid tier specified: $TIER. Must be free, standard, or both."
    exit 1
fi

if [[ "$ARCH" != "cpu" && "$ARCH" != "gpu" && "$ARCH" != "both" ]]; then
    echo "[ERROR] Invalid architecture specified: $ARCH. Must be cpu, gpu, or both."
    exit 1
fi

# Validate and set jobs parameter
if [[ "$JOBS" == "auto" ]]; then
    # Auto-detect number of CPU cores
    if command -v nproc &> /dev/null; then
        JOBS=$(nproc)
    elif [[ -f /proc/cpuinfo ]]; then
        JOBS=$(grep -c ^processor /proc/cpuinfo)
    else
        JOBS=4
    fi
    log_info "Auto-detected $JOBS CPU cores for parallel build"
elif [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    log_info "Using $JOBS parallel jobs for build"
else
    echo "[ERROR] Invalid jobs parameter: $JOBS. Must be 'auto' or a positive integer."
    exit 1
fi

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

# log_info() - Log an informational message
# Arguments:
#   $1: Message string
# Behavior:
#   Prints the message prefixed with [INFO] to stdout and appends to the build log.
log_info() {
    local msg="$1"
    echo "[INFO] $msg"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $msg" >> "$BUILD_LOG"
}

# log_error() - Log an error message
# Arguments:
#   $1: Message string
# Behavior:
#   Prints the message prefixed with [ERROR] to stdout and appends to the build log.
log_error() {
    local msg="$1"
    echo "[ERROR] $msg"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $msg" >> "$BUILD_LOG"
}

# log_warning() - Log a warning message
# Arguments:
#   $1: Message string
# Behavior:
#   Prints the message prefixed with [WARNING] to stdout and appends to the build log.
log_warning() {
    local msg="$1"
    echo "[WARNING] $msg"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $msg" >> "$BUILD_LOG"
}

# Display header and copyright information
log_info "RadarSimPy Build Script v$SCRIPT_VERSION"
log_info "Copyright 2018 - PRESENT  radarsimx.com"
log_info "E-mail: info@radarsimx.com"
log_info "Website: https://radarsimx.com"
log_info ""
log_info "██████╗  █████╗ ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ███╗██╗  ██╗"
log_info "██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗ ████║╚██╗██╔╝"
log_info "██████╔╝███████║██║  ██║███████║██████╔╝███████╗██║██╔████╔██║ ╚███╔╝ "
log_info "██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗╚════██║██║██║╚██╔╝██║ ██╔██╗ "
log_info "██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║███████║██║██║ ╚═╝ ██║██╔╝ ██╗"
log_info "╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝"
log_info ""
log_info "Build Configuration:"
log_info "  Tier: $TIER"
log_info "  Architecture: $ARCH"
log_info "  Skip Tests: $SKIP_TESTS"
log_info "  Verbose: $VERBOSE"
log_info "  Parallel Jobs: $JOBS"
log_info "  Log File: $BUILD_LOG"
log_info ""

# Store current directory path
workpath=$(pwd)

# ============================================================================
# PREREQUISITE CHECK FUNCTIONS
# ============================================================================

# check_prerequisites() - Verify required commands and environments
# Description:
#   Ensures CMake is installed, checks CUDA if GPU builds are requested,
#   and verifies that all required Python conda environments exist.
# Returns:
#   0 if all prerequisites are met; non-zero otherwise.
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if cmake is available
    log_info "Checking CMake..."
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake and add it to PATH."
        return 1
    fi
    log_info "CMake found."

    # Check CUDA availability for GPU builds
    if [[ "$ARCH" == "gpu" ]] || [[ "$ARCH" == "both" ]]; then
        log_info "GPU architecture requested, checking CUDA..."
        check_cuda_prerequisites
        if [[ $? -ne 0 ]]; then
            return 1
        fi
    fi

    # Check Python environments
    log_info "Checking Python environments..."
    local missing_envs=false
    for version in $PYTHON_VERSIONS; do
        log_info "Checking environment $version..."
        if ! conda info --envs | grep -q "$version"; then
            log_error "Python environment $version not found. This is required for the build."
            missing_envs=true
        else
            log_info "Environment $version found."
        fi
    done

    if [[ "$missing_envs" == "true" ]]; then
        log_error "One or more required Python environments are missing. Please create all required environments before building."
        return 1
    fi

    log_info "Prerequisites check completed."
    return 0
}

# check_cuda_prerequisites() - Verify CUDA toolkit prerequisites
# Description:
#   Checks for nvcc and nvidia-smi availability to support GPU builds.
# Returns:
#   0 if CUDA prerequisites are met; non-zero otherwise.
check_cuda_prerequisites() {
    log_info "Checking CUDA prerequisites..."

    # Check if nvcc is available
    log_info "Checking NVCC..."
    if ! command -v nvcc &> /dev/null; then
        log_error "NVCC not found. Please install CUDA Toolkit and add it to PATH."
        return 1
    fi
    log_info "NVCC found."

    # Check if nvidia-smi is available
    log_info "Checking nvidia-smi..."
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found. GPU may not be available."
    else
        log_info "nvidia-smi found."
    fi

    log_info "CUDA prerequisites check completed."
    return 0
}

# ============================================================================
# CLEANUP FUNCTIONS
# ============================================================================

# cleanup_build_artifacts() - Remove previous build artifacts
# Description:
#   Deletes C++ build directories, the radarsimpy output directory, and general build folders.
# Returns:
#   Always returns 0.
cleanup_build_artifacts() {
    log_info "Cleaning up previous build artifacts..."

    # Clean build directories
    rm -rf "./src/radarsimcpp/build"
    rm -rf "./radarsimpy"
    rm -rf "./build"

    cleanup_temp_files
    return 0
}

# cleanup_temp_files() - Remove temporary generated files
# Description:
#   Deletes generated C/C++/HTML files in the Python source tree.
# Returns:
#   Always returns 0.
cleanup_temp_files() {
    log_info "Cleaning temporary files..."

    # Clean generated files
    rm -f ./src/radarsimpy/*.c
    rm -f ./src/radarsimpy/*.cpp
    rm -f ./src/radarsimpy/*.html
    rm -f ./src/radarsimpy/raytracing/*.c
    rm -f ./src/radarsimpy/raytracing/*.cpp
    rm -f ./src/radarsimpy/raytracing/*.html
    rm -f ./src/radarsimpy/lib/*.cpp
    rm -f ./src/radarsimpy/lib/*.html
    rm -f ./src/*.cpp
    rm -f ./src/*.html
    return 0
}

# ============================================================================
# CORE BUILD FUNCTIONS
# ============================================================================

# build_all_combinations() - Build for all requested tier/architecture combinations
# Description:
#   Determines the list of tiers and architectures to build and invokes build_single_combination().
# Returns:
#   0 on success; non-zero if any combination fails.
build_all_combinations() {
    log_info "Building all requested combinations..."

    # Determine which architectures to build
    local arch_list
    if [[ "$ARCH" == "both" ]]; then
        arch_list="cpu gpu"
    else
        arch_list="$ARCH"
    fi

    # Determine which tiers to build
    local tier_list
    if [[ "$TIER" == "both" ]]; then
        tier_list="free standard"
    else
        tier_list="$TIER"
    fi

    # Build each combination
    for arch in $arch_list; do
        for tier in $tier_list; do
            build_single_combination "$arch" "$tier"
            if [[ $? -ne 0 ]]; then
                return 1
            fi
        done
    done

    return 0
}

# build_single_combination() - Build a specific tier and architecture
# Arguments:
#   $1: Architecture ("cpu" or "gpu")
#   $2: Tier ("free" or "standard")
# Returns:
#   0 on success; non-zero on failure.
build_single_combination() {
    local build_arch="$1"
    local build_tier="$2"
    log_info "Building $build_tier tier for $build_arch architecture..."

    # Build C++ library for this architecture
    build_cpp_library "$build_arch"
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    # Build Python extensions for this combination
    build_python_extensions "$build_tier" "$build_arch"
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    # Copy built files
    copy_built_files
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    # Clean up intermediate files
    cleanup_temp_files

    # Create distribution folder
    create_distribution "$build_tier" "$build_arch"
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    # Clean up for next combination
    rm -rf "./radarsimpy"

    return 0
}

# build_cpp_library() - Compile the C++ library for a given architecture
# Arguments:
#   $1: Architecture ("cpu" or "gpu")
# Returns:
#   0 on success; non-zero on failure.
build_cpp_library() {
    local build_arch="$1"
    log_info "Building C++ library for $build_arch architecture..."

    # Clean and recreate build directory
    rm -rf "./src/radarsimcpp/build"
    mkdir -p "./src/radarsimcpp/build"
    cd "./src/radarsimcpp/build"

    # Configure CMake with appropriate flags
    local cmake_flags="-DCMAKE_BUILD_TYPE=Release -DGTEST=ON"
    if [[ "$build_arch" == "gpu" ]]; then
        cmake_flags="$cmake_flags -DGPU_BUILD=ON"
        log_info "Configuring with GPU/CUDA support..."
    else
        log_info "Configuring with CPU support..."
    fi

    # Configure and build
    cmake $cmake_flags .. 2>&1
    if [[ $? -ne 0 ]]; then
        cd "$workpath"
        log_error "CMake configuration failed for $build_arch"
        return 1
    fi

    cmake --build . --parallel $JOBS 2>&1
    if [[ $? -ne 0 ]]; then
        cd "$workpath"
        log_error "C++ library build failed for $build_arch"
        return 1
    fi

    cd "$workpath"
    log_info "C++ library built successfully for $build_arch."
    return 0
}

# build_python_extensions() - Build Python extension modules
# Arguments:
#   $1: Tier ("free" or "standard")
#   $2: Architecture ("cpu" or "gpu")
# Returns:
#   Always returns 0 (warnings on individual env failures).
build_python_extensions() {
    local build_tier="$1"
    local build_arch="$2"
    log_info "Building Python extensions for $build_tier tier with $build_arch architecture..."

    for version in $PYTHON_VERSIONS; do
        log_info "Building for $version..."
        if conda info --envs | grep -q "$version"; then
            conda run -n "$version" python setup.py build_ext -b ./ --tier "$build_tier" --arch "$build_arch" 2>&1
            if [[ $? -ne 0 ]]; then
                log_warning "Failed to build for $version, continuing..."
            fi
        else
            log_warning "Environment $version not found, skipping..."
        fi
    done
    return 0
}

# copy_built_files() - Copy built artifacts to the radarsimpy directory
# Description:
#   Copies shared libraries, Python modules, and the lib/__init__.py file
#   into the distribution directory.
# Returns:
#   0 on success; non-zero on failure.
copy_built_files() {
    log_info "Copying built files..."

    # Ensure destination directory exists
    mkdir -p "./radarsimpy/lib"

    # Copy files with error checking
    if ! cp "./src/radarsimcpp/build/"*.so "./radarsimpy/" 2>/dev/null; then
        log_error "Failed to copy shared library"
        return 1
    fi

    if ! cp "./src/radarsimpy/"*.py "./radarsimpy/"; then
        log_error "Failed to copy Python files"
        return 1
    fi

    if ! cp "./src/radarsimpy/lib/__init__.py" "./radarsimpy/lib/"; then
        log_error "Failed to copy lib __init__.py"
        return 1
    fi

    return 0
}

# create_distribution() - Package build into release directory
# Arguments:
#   $1: Tier ("free" or "standard")
#   $2: Architecture ("cpu" or "gpu")
# Returns:
#   0 on success; non-zero on failure.
create_distribution() {
    local build_tier="$1"
    local build_arch="$2"
    log_info "Creating $build_tier distribution for $build_arch architecture..."

    # Detect Linux distribution
    local linux_distro=$(detect_linux_distribution)
    log_info "Detected Linux distribution: $linux_distro"

    # Determine architecture suffix for folder name
    local arch_suffix="CPU"
    if [[ "$build_arch" == "gpu" ]]; then
        arch_suffix="GPU"
    fi

    local dist_folder
    if [[ "$build_tier" == "free" ]]; then
        dist_folder="./release/trial/${linux_distro}_x86_64_$arch_suffix"
    else
        dist_folder="./release/standard/${linux_distro}_x86_64_$arch_suffix"
    fi

    # Clean and create distribution folder
    rm -rf "$dist_folder"
    mkdir -p "$dist_folder"
    cp -rf "./radarsimpy" "$dist_folder/" 2>&1
    if [[ $? -ne 0 ]]; then
        log_error "Failed to create $build_tier distribution for $build_arch"
        return 1
    fi

    return 0
}

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

# run_tests() - Execute unit tests
# Description:
#   Runs Google Test via CTest for the C++ library and reports pass/fail.
# Returns:
#   0 if tests pass or are skipped; non-zero if tests fail.
run_tests() {
    log_info "Running unit tests..."

    # Run Google Test (C++) using CTest with parallel jobs
    if [[ -f "./src/radarsimcpp/build/radarsimcpp_test" ]]; then
        log_info "Running Google Test for C++ using CTest with $JOBS parallel jobs..."
        ctest --test-dir "./src/radarsimcpp/build" --parallel $JOBS --verbose 2>&1
        if [[ $? -ne 0 ]]; then
            log_error "Google Test failed"
            return 1
        else
            log_info "Google Test passed"
        fi
    else
        log_warning "Google Test executable not found, skipping C++ tests"
    fi

    log_info "Unit tests completed successfully."
    return 0
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# detect_linux_distribution() - Identify the Linux distribution ID
# Description:
#   Reads /etc/os-release (or fallbacks) to compute a sanitized distro identifier.
# Output:
#   Echoes a string like "Ubuntu22", "Debian10", "Arch", or "Linux".
detect_linux_distribution() {
    local distro_id=""
    
    # Try to detect using /etc/os-release (most modern distributions)
    if [[ -f "/etc/os-release" ]]; then
        source /etc/os-release
        case "$ID" in
            ubuntu)
                # Extract major version from VERSION_ID (e.g., "22.04" -> "22")
                local version_major=$(echo "$VERSION_ID" | cut -d'.' -f1)
                distro_id="Ubuntu${version_major}"
                ;;
            debian)
                local version_major=$(echo "$VERSION_ID" | cut -d'.' -f1)
                distro_id="Debian${version_major}"
                ;;
            centos)
                local version_major=$(echo "$VERSION_ID" | cut -d'.' -f1)
                distro_id="CentOS${version_major}"
                ;;
            rhel)
                local version_major=$(echo "$VERSION_ID" | cut -d'.' -f1)
                distro_id="RHEL${version_major}"
                ;;
            fedora)
                distro_id="Fedora${VERSION_ID}"
                ;;
            opensuse-leap)
                local version_major=$(echo "$VERSION_ID" | cut -d'.' -f1)
                distro_id="OpenSUSE${version_major}"
                ;;
            arch)
                distro_id="Arch"
                ;;
            *)
                distro_id="${ID^}${VERSION_ID}"
                ;;
        esac
    # Fallback to /etc/lsb-release (older Ubuntu/Debian systems)
    elif [[ -f "/etc/lsb-release" ]]; then
        source /etc/lsb-release
        case "$DISTRIB_ID" in
            Ubuntu)
                local version_major=$(echo "$DISTRIB_RELEASE" | cut -d'.' -f1)
                distro_id="Ubuntu${version_major}"
                ;;
            *)
                distro_id="${DISTRIB_ID}${DISTRIB_RELEASE}"
                ;;
        esac
    # Fallback to checking specific release files
    elif [[ -f "/etc/redhat-release" ]]; then
        if grep -q "CentOS" /etc/redhat-release; then
            local version=$(grep -o "[0-9]\+\.[0-9]\+" /etc/redhat-release | head -1 | cut -d'.' -f1)
            distro_id="CentOS${version}"
        elif grep -q "Red Hat" /etc/redhat-release; then
            local version=$(grep -o "[0-9]\+\.[0-9]\+" /etc/redhat-release | head -1 | cut -d'.' -f1)
            distro_id="RHEL${version}"
        fi
    elif [[ -f "/etc/debian_version" ]]; then
        local version=$(cat /etc/debian_version | cut -d'.' -f1)
        distro_id="Debian${version}"
    # Final fallback to generic Linux
    else
        distro_id="Linux"
    fi
    
    # Remove any dots or special characters and ensure no spaces
    distro_id=$(echo "$distro_id" | sed 's/[^a-zA-Z0-9]//g')
    
    echo "$distro_id"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Verify prerequisites
check_prerequisites
if [[ $? -ne 0 ]]; then
    log_error "Prerequisites check failed"
    exit 1
fi

# Main build process
log_info "Starting build process..."

# Clean up previous build artifacts
cleanup_build_artifacts
if [[ $? -ne 0 ]]; then
    log_error "Failed to clean build artifacts"
    exit 1
fi

# Build all combinations of tier and architecture
build_all_combinations
if [[ $? -ne 0 ]]; then
    log_error "Build failed"
    exit 1
fi

# Run unit tests if not skipped
if [[ "$SKIP_TESTS" == "false" ]]; then
    run_tests
    if [[ $? -ne 0 ]]; then
        log_error "Tests failed"
        exit 1
    fi
fi

log_info "Build completed successfully!"
exit 0

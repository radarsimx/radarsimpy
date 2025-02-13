#!/bin/bash

# Function to display help information and usage instructions
Help()
{
    # Display Help
    echo
    echo "Usages:"
    echo
    echo "Syntax: build_linux.sh --tier=[standard|free] --arch=[cpu|gpu] --test=[on|off]"
    echo "options:"
    echo "    --help    Show the usages of the parameters"
    echo "    --tier    Build tier, choose 'standard' or 'free'. Default is 'standard'"
    echo "    --arch    Build architecture, choose 'cpu' or 'gpu'. Default is 'cpu'"
    echo "    --test    Enable or disable unit test, choose 'on' or 'off'. Default is 'on'"
    echo
}

# Default configuration values
TIER="standard"    # Build tier (standard/free)
ARCH="cpu"        # Build architecture (cpu/gpu)
TEST="on"         # Unit test flag (on/off)

# Parse command line arguments
# Supports --help, --tier, --arch, and --test parameters
for i in "$@"; do
    case $i in
        --help*)
            Help
            exit;;
        --tier=*)
            TIER="${i#*=}"
            shift # past argument
            ;;
        --arch=*)
            ARCH="${i#*=}"
            shift # past argument
            ;;
        --test=*)
            TEST="${i#*=}"
            shift # past argument
            ;;
        --*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            ;;
    esac
done

# Validate the tier parameter
if [ "${TIER,,}" != "standard" ] && [ "${TIER,,}" != "free" ]; then
    echo "ERROR: Invalid --tier parameters, please choose 'free' or 'standard'"
    exit 1
fi

# Validate the architecture parameter
if [ "${ARCH,,}" != "cpu" ] && [ "${ARCH,,}" != "gpu" ]; then
    echo "ERROR: Invalid --arch parameters, please choose 'cpu' or 'gpu'"
    exit 1
fi

# Validate the test parameter
if [ "${TEST,,}" != "on" ] && [ "${TEST,,}" != "off" ]; then
    echo "ERROR: Invalid --test parameters, please choose 'on' or 'off'"
    exit 1
fi

# Display project banner and copyright information
echo "Automatic build script of radarsimcpp/radarsimpy for Linux"
echo ""
echo "----------"
echo "RadarSimPy - A Radar Simulator Built with Python"
echo "Copyright (C) 2018 - PRESENT  radarsimx.com"
echo "E-mail: info@radarsimx.com"
echo "Website: https://radarsimx.com"
echo ""
echo "██████╗  █████╗ ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ███╗██╗  ██╗"
echo "██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗ ████║╚██╗██╔╝"
echo "██████╔╝███████║██║  ██║███████║██████╔╝███████╗██║██╔████╔██║ ╚███╔╝ "
echo "██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗╚════██║██║██║╚██╔╝██║ ██╔██╗ "
echo "██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║███████║██║██║ ╚═╝ ██║██╔╝ ██╗"
echo "╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝"

# Store current working directory
workpath=$(pwd)

# Clean up previous build artifacts
echo "## Clean old build files ##"
rm -rf ./src/radarsimcpp/build
rm -rf ./radarsimpy

# Build libradarsimcpp.so
echo "## Building libradarsimcpp.so with ${ARCH^^} ##"
mkdir ./src/radarsimcpp/build 
cd ./src/radarsimcpp/build

# Configure CMake based on architecture and test settings
if [ "${ARCH,,}" == "gpu" ]; then
    if [ "${TEST,,}" == "on" ]; then
        # GPU build with tests enabled
        cmake -DCMAKE_BUILD_TYPE=Release -DGPU_BUILD=ON -DGTEST=ON ..
    elif [ "${TEST,,}" == "off" ]; then
        # GPU build without tests
        cmake -DCMAKE_BUILD_TYPE=Release -DGPU_BUILD=ON -DGTEST=OFF ..
    fi
elif [ "${ARCH,,}" == "cpu" ]; then
    if [ "${TEST,,}" == "on" ]; then
        # CPU build with tests enabled
        cmake -DCMAKE_BUILD_TYPE=Release -DGTEST=ON ..
    elif [ "${TEST,,}" == "off" ]; then
        # CPU build without tests
        cmake -DCMAKE_BUILD_TYPE=Release -DGTEST=OFF ..
    fi
fi

# Build the project using CMake
cmake --build .

# Build Python extensions using Cython
echo "## Building radarsimpy with Cython ##"
cd $workpath
python setup.py build_ext -b ./ --tier "${TIER}" --arch "${ARCH}"

# Copy library files to radarsimpy directory
echo "## Copying lib files to ./radarsimpy ##"
cp ./src/radarsimpy/*.py ./radarsimpy
cp ./src/radarsimpy/lib/__init__.py ./radarsimpy/lib
cp ./src/radarsimcpp/build/*.so ./radarsimpy

# Clean up intermediate build files
echo "## Cleaning radarsimpy builds ##"
rm -rf build

# Remove generated C/C++ source files and HTML documentation
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

echo "## Build completed ##"

return_code = 0
# Run tests if enabled
if [ "${TEST,,}" == "on" ]; then
    # Run C++ unit tests using Google Test
    echo "## Run Google test ##"
    ./src/radarsimcpp/build/radarsimcpp_test
    return_code = $(($return_code + $?))

    # Run Python unit tests using pytest
    echo "## Pytest ##"
    pytest
    return_code = $(($return_code + $?))
fi

exit $return_code

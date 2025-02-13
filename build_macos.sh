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

# Parse command line arguments
# Supported arguments:
# --tier: Build tier (standard/free)
# --arch: Build architecture (cpu/gpu)
# --test: Enable/disable unit testing (on/off)
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
# Must be either 'standard' or 'free'
if [ "${TIER}" != "standard" ] && [ "${TIER}" != "free" ]; then
    echo "ERROR: Invalid --tier parameters, please choose 'free' or 'standard'"
    exit 1
fi

# Validate the architecture parameter
# Must be either 'cpu' or 'gpu'
if [ "${ARCH}" != "cpu" ] && [ "${ARCH}" != "gpu" ]; then
    echo "ERROR: Invalid --arch parameters, please choose 'cpu' or 'gpu'"
    exit 1
fi

# Validate the test parameter
# Must be either 'on' or 'off'
if [ "${TEST}" != "on" ] && [ "${TEST}" != "off" ]; then
    echo "ERROR: Invalid --test parameters, please choose 'on' or 'off'"
    exit 1
fi

# Display welcome message and copyright information
echo "Automatic build script of radarsimcpp/radarsimpy for macOS"
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

# Build the C++ library
echo "## Building libradarsimcpp.so with ${ARCH} ##"
mkdir ./src/radarsimcpp/build 
cd ./src/radarsimcpp/build

# Convert architecture and test parameters to lowercase for consistency
ARCH_LOWER=$(echo "$ARCH" | tr '[:upper:]' '[:lower:]')
TEST_LOWER=$(echo "$TEST" | tr '[:upper:]' '[:lower:]')

# Configure CMake based on build parameters
# GPU build with testing enabled/disabled
if [ "${ARCH_LOWER}" = "gpu" ]; then
    if [ "${TEST_LOWER}" = "on" ]; then
        cmake -DCMAKE_BUILD_TYPE=Release -DGPU_BUILD=ON -DGTEST=ON ..
    elif [ "${TEST_LOWER}" = "off" ]; then
        cmake -DCMAKE_BUILD_TYPE=Release -DGPU_BUILD=ON -DGTEST=OFF ..
    fi
# CPU build with testing enabled/disabled
elif [ "${ARCH_LOWER}" = "cpu" ]; then
    if [ "${TEST_LOWER}" = "on" ]; then
        cmake -DCMAKE_BUILD_TYPE=Release -DGTEST=ON ..
    elif [ "${TEST_LOWER}" = "off" ]; then
        cmake -DCMAKE_BUILD_TYPE=Release -DGTEST=OFF ..
    fi
fi

# Build the project using CMake
cmake --build .

# Build Python extensions using Cython
echo "## Building radarsimpy with Cython ##"
cd $workpath
python setup.py build_ext -b ./ --tier "${TIER}" --arch "${ARCH}"

# Copy built files to the final location
echo "## Copying lib files to ./radarsimpy ##"
cp ./src/radarsimpy/*.py ./radarsimpy
cp ./src/radarsimpy/lib/__init__.py ./radarsimpy/lib
cp ./src/radarsimcpp/build/*.dylib ./radarsimpy

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

return_code=0
# Run tests if testing is enabled
if [ "${TEST}" == "on" ]; then
    # Run C++ unit tests
    echo "## Run Google test ##"
    ./src/radarsimcpp/build/radarsimcpp_test
    return_code=$(($return_code + $?))

    # Run Python unit tests
    echo "## Pytest ##"
    pytest
    return_code=$(($return_code + $?))
fi

exit $return_code

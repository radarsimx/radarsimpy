#!/bin/bash

# Display banner and copyright information
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

# Store current working directory for later reference
workpath=$(pwd)

# Clean previous build artifacts and directories
echo "## Clean old build files ##"
rm -rf ./src/radarsimcpp/build
rm -rf ./radarsimpy

# Build CPU-only C++ library
echo "## Building libradarsimcpp.so with CPU ##"
mkdir ./src/radarsimcpp/build 
cd ./src/radarsimcpp/build

# Configure CMake for CPU-only build
cmake -DCMAKE_BUILD_TYPE=Release -DGTEST=ON ..
cmake --build .

# Build Python extensions for multiple Python versions (Free Tier)
echo "## Building radarsimpy with Cython ##"
cd $workpath
# Build CPU-only extensions for Python 3.9-3.12
conda run -n py312 python setup.py build_ext -b ./ --tier free --arch cpu
conda run -n py311 python setup.py build_ext -b ./ --tier free --arch cpu
conda run -n py310 python setup.py build_ext -b ./ --tier free --arch cpu
conda run -n py39 python setup.py build_ext -b ./ --tier free --arch cpu

# Copy built libraries and Python files
echo "## Copying lib files to ./radarsimpy ##"
cp ./src/radarsimpy/*.py ./radarsimpy
cp ./src/radarsimpy/lib/__init__.py ./radarsimpy/lib
cp ./src/radarsimcpp/build/*.so ./radarsimpy

# Clean intermediate build files
echo "## Cleaning radarsimpy builds ##"
rm -rf build
# Remove generated C/C++/HTML files
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

# Package Free Tier CPU release
echo "## Copying lib files to freetier release folder ##"
rm -rf ./Linux_x86_64_CPU_FreeTier
mkdir ./Linux_x86_64_CPU_FreeTier
mkdir ./Linux_x86_64_CPU_FreeTier/radarsimpy
cp -rf ./radarsimpy/* ./Linux_x86_64_CPU_FreeTier/radarsimpy

# Clean and rebuild for Standard Tier
rm -rf ./radarsimpy
# Build CPU-only extensions for Standard Tier
conda run -n py312 python setup.py build_ext -b ./ --tier standard --arch cpu
conda run -n py311 python setup.py build_ext -b ./ --tier standard --arch cpu
conda run -n py310 python setup.py build_ext -b ./ --tier standard --arch cpu
conda run -n py39 python setup.py build_ext -b ./ --tier standard --arch cpu

# Copy files for Standard Tier
echo "## Copying lib files to ./radarsimpy ##"
cp ./src/radarsimpy/*.py ./radarsimpy
cp ./src/radarsimpy/lib/__init__.py ./radarsimpy/lib
cp ./src/radarsimcpp/build/*.so ./radarsimpy

# Final cleanup of build artifacts
echo "## Cleaning radarsimpy builds ##"
rm -rf build
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

# Package Standard Tier CPU release
echo "## Copying lib files to standard release folder ##"
rm -rf ./Linux_x86_64_CPU
mkdir ./Linux_x86_64_CPU
mkdir ./Linux_x86_64_CPU/radarsimpy
cp -rf ./radarsimpy/* ./Linux_x86_64_CPU/radarsimpy

# Run unit tests
echo "## Build completed ##"
echo "## Run Google test ##"
./src/radarsimcpp/build/radarsimcpp_test

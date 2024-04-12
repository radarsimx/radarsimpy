#!/bin/bash

Help()
{
   # Display Help
   echo
   echo "Usages:"
   echo
   echo "Syntax: build_linux.sh --tier=[standard|free] --arch=[cpu|gpu] --test=[on|off]"
   echo "options:"
   echo "   --help	Show the usages of the parameters"
   echo "   --tier	Build tier, choose 'standard' or 'free'. Default is 'standard'"
   echo "   --arch	Build architecture, choose 'cpu' or 'gpu'. Default is 'cpu'"
   echo "   --test	Enable or disable unit test, choose 'on' or 'off'. Default is 'on'"
   echo
}

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

if [ "${TIER,,}" != "standard" ] && [ "${TIER,,}" != "free" ]; then
    echo "ERROR: Invalid --tier parameters, please choose 'free' or 'standard'"
    exit 1
fi

if [ "${ARCH,,}" != "cpu" ] && [ "${ARCH,,}" != "gpu" ]; then
    echo "ERROR: Invalid --arch parameters, please choose 'cpu' or 'gpu'"
    exit 1
fi

if [ "${TEST,,}" != "on" ] && [ "${TEST,,}" != "off" ]; then
    echo "ERROR: Invalid --test parameters, please choose 'on' or 'off'"
    exit 1
fi

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

workpath=$(pwd)

echo "## Clean old build files ##"
rm -rf ./src/radarsimcpp/build
rm -rf ./radarsimpy

echo "## Building libradarsimcpp.so with ${ARCH^^} ##"
mkdir ./src/radarsimcpp/build 
cd ./src/radarsimcpp/build

if [ "${ARCH,,}" == "gpu" ]; then
    cmake -DCMAKE_BUILD_TYPE=Release -DGPU_BUILD=ON -DGTEST=ON ..
elif [ "${ARCH,,}" == "cpu" ]; then
    cmake -DCMAKE_BUILD_TYPE=Release -DGTEST=ON ..
fi
cmake --build .

echo "## Building radarsimpy with Cython ##"
cd $workpath
python setup.py build_ext -b ./ --tier "${TIER}" --arch "${ARCH}"

echo "## Copying lib files to ./radarsimpy ##"
# mkdir ./radarsimpy/lib
cp ./src/radarsimpy/*.py ./radarsimpy
cp ./src/radarsimpy/lib/__init__.py ./radarsimpy/lib
cp ./src/radarsimcpp/build/*.dylib ./radarsimpy

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

echo "## Build completed ##"

if [ "${TEST,,}" == "on" ]; then
    echo "## Run Google test ##"
    ./src/radarsimcpp/build/radarsimcpp_test

    echo "## Pytest ##"
    pytest
fi
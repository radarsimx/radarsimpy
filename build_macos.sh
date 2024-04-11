#!/bin/bash

for i in "$@"; do
  case $i in
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

echo "## Building libradarsimcpp.so with CP${ARCH}U ##"
mkdir ./src/radarsimcpp/build 
cd ./src/radarsimcpp/build

if [ "${ARCH}" == "gpu" ]; then
    cmake -DCMAKE_BUILD_TYPE=Release -DGPU_BUILD=ON -DGTEST=ON ..
elif [ "${ARCH}" == "cpu" ]; then
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

echo "## Copying lib files to unit test folder ##"
rm -rf ./tests/radarsimpy
mkdir ./tests/radarsimpy
cp -rf ./radarsimpy/* ./tests/radarsimpy

echo "## Build completed ##"

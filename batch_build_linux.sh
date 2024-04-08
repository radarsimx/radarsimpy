#!/bin/sh

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

workpath=$(pwd)

echo "## Clean old build files ##"
rm -rf ./src/radarsimcpp/build
rm -rf ./radarsimpy

echo "## Building libradarsimcpp.so with CPU ##"
mkdir ./src/radarsimcpp/build 
cd ./src/radarsimcpp/build

cmake -DCMAKE_BUILD_TYPE=Release -DGTEST=ON ..
cmake --build .

echo "## Building radarsimpy with Cython ##"
cd $workpath
conda run -n py38 python setup_freetier.py build_ext -b ./
conda run -n py39 python setup_freetier.py build_ext -b ./
conda run -n py310 python setup_freetier.py build_ext -b ./
conda run -n py311 python setup_freetier.py build_ext -b ./
conda run -n py312 python setup_freetier.py build_ext -b ./

echo "## Copying lib files to ./radarsimpy ##"

cp ./src/radarsimpy/*.py ./radarsimpy
cp ./src/radarsimpy/lib/__init__.py ./radarsimpy/lib
cp ./src/radarsimcpp/build/*.so ./radarsimpy

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

echo "## Copying lib files to freetier release folder ##"

rm -rf ./Linux_x86_64_CPU_FreeTier
mkdir ./Linux_x86_64_CPU_FreeTier
mkdir ./Linux_x86_64_CPU_FreeTier/radarsimpy
cp -rf ./radarsimpy/* ./Linux_x86_64_CPU_FreeTier/radarsimpy

rm -rf ./radarsimpy

conda run -n py38 python setup.py build_ext -b ./
conda run -n py39 python setup.py build_ext -b ./
conda run -n py310 python setup.py build_ext -b ./
conda run -n py311 python setup.py build_ext -b ./
conda run -n py312 python setup.py build_ext -b ./

echo "## Copying lib files to ./radarsimpy ##"

cp ./src/radarsimpy/*.py ./radarsimpy
cp ./src/radarsimpy/lib/__init__.py ./radarsimpy/lib
cp ./src/radarsimcpp/build/*.so ./radarsimpy

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

echo "## Copying lib files to standard release folder ##"

rm -rf ./Linux_x86_64_CPU
mkdir ./Linux_x86_64_CPU
mkdir ./Linux_x86_64_CPU/radarsimpy
cp -rf ./radarsimpy/* ./Linux_x86_64_CPU/radarsimpy

echo "## Build completed ##"

echo "## Run Google test ##"
./src/radarsimcpp/build/radarsimcpp_test

echo "## Pytest ##"
pytest

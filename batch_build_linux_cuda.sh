#!/bin/bash

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

echo "## Building libradarsimcpp.so with GPU ##"
mkdir ./src/radarsimcpp/build 
cd ./src/radarsimcpp/build

cmake -DCMAKE_BUILD_TYPE=Release -DGPU_BUILD=ON -DGTEST=ON ..
cmake --build .

echo "## Building radarsimpy with Cython ##"
cd $workpath
conda run -n py38 python setup.py build_ext -b ./ --tier free --arch gpu
conda run -n py39 python setup.py build_ext -b ./ --tier free --arch gpu
conda run -n py310 python setup.py build_ext -b ./ --tier free --arch gpu
conda run -n py311 python setup.py build_ext -b ./ --tier free --arch gpu
conda run -n py312 python setup.py build_ext -b ./ --tier free --arch gpu

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

rm -rf ./Linux_x86_64_GPU_FreeTier
mkdir ./Linux_x86_64_GPU_FreeTier
mkdir ./Linux_x86_64_GPU_FreeTier/radarsimpy
cp -rf ./radarsimpy/* ./Linux_x86_64_GPU_FreeTier/radarsimpy

rm -rf ./radarsimpy

conda run -n py38 python setup.py build_ext -b ./ --tier standard --arch gpu
conda run -n py39 python setup.py build_ext -b ./ --tier standard --arch gpu
conda run -n py310 python setup.py build_ext -b ./ --tier standard --arch gpu
conda run -n py311 python setup.py build_ext -b ./ --tier standard --arch gpu
conda run -n py312 python setup.py build_ext -b ./ --tier standard --arch gpu

echo "## Copying lib files to ./radarsimpy ##"
# mkdir ./radarsimpy
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

rm -rf ./Linux_x86_64_GPU
mkdir ./Linux_x86_64_GPU
mkdir ./Linux_x86_64_GPU/radarsimpy
cp -rf ./radarsimpy/* ./Linux_x86_64_GPU/radarsimpy

echo "## Build completed ##"

echo "## Run Google test ##"
./src/radarsimcpp/build/radarsimcpp_test

echo "## Pytest ##"
pytest

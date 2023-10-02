#!/bin/sh

echo "Automatic build script of radarsimcpp/radarsimpy for Linux"
echo ""
echo "----------"
echo "RadarSimPy - A Radar Simulator Built with Python"
echo "Copyright (C) 2018 - PRESENT  radarsimx.com"
echo "E-mail: info@radarsimx.com"
echo "Website: https://radarsimx.com"
echo ""
echo " ____           _            ____  _          __  __ "
echo "|  _ \ __ _  __| | __ _ _ __/ ___|(_)_ __ ___ \ \/ / "
echo "| |_) / _' |/ _' |/ _' | '__\___ \| | '_ ' _ \ \  /  "
echo "|  _ < (_| | (_| | (_| | |   ___) | | | | | | |/  \  "
echo "|_| \_\__,_|\__,_|\__,_|_|  |____/|_|_| |_| |_/_/\_\ "

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
python setup_cuda.py build_ext -b ./

echo "## Copying lib files to ./radarsimpy ##"
# mkdir ./radarsimpy/lib
cp ./src/radarsimpy/__init__.py ./radarsimpy
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

echo "## Copying lib files to unit test folder ##"
rm -rf ./tests/radarsimpy
mkdir ./tests/radarsimpy
cp -rf ./radarsimpy/* ./tests/radarsimpy

echo "## Build completed ##"

echo "## Run Google test ##"
./src/radarsimcpp/build/radarsimcpp_test

echo "## Pytest ##"
pytest

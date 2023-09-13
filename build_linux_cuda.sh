#!/bin/sh

echo "Automatic build script of radarsimc/radarsimpy for Linux"
echo ""
echo "----------"
echo "RadarSimPy - A Radar Simulator Built with Python"
echo "Copyright (C) 2018 - PRESENT  Zhengyu Peng"
echo "E-mail: zpeng.me@gmail.com"
echo "Website: https://zpeng.me"
echo ""
echo "'                      '"
echo "-:.                  -#:"
echo "-//:.              -###:"
echo "-////:.          -#####:"
echo "-/:.://:.      -###++##:"
echo "..   '://:-  -###+. :##:"
echo "       ':/+####+.   :##:"
echo ".::::::::/+###.     :##:"
echo ".////-----+##:    ':###:"
echo " '-//:.   :##:  ':###/."
echo "   '-//:. :##:':###/."
echo "     '-//:+######/."
echo "       '-/+####/."
echo "         '+##+."
echo "          :##:"
echo "          :##:"
echo "          :##:"
echo "          :##:"
echo "          :##:"
echo "           .+:"

workpath=$(pwd)

echo "## Clean old build files ##"
rm -rf ./src/radarsimc/build
rm -rf ./radarsimpy

echo "## Building libradarsimcpp.so with GPU ##"
mkdir ./src/radarsimc/build 
cd ./src/radarsimc/build

cmake -DCMAKE_BUILD_TYPE=Release -DGPU_BUILD=ON -DGTEST=ON ..
cmake --build .

echo "## Building radarsimpy with Cython ##"
cd $workpath
python setup_cuda.py build_ext -b ./

echo "## Copying lib files to ./radarsimpy ##"
# mkdir ./radarsimpy/lib
cp ./src/radarsimpy/__init__.py ./radarsimpy
cp ./src/radarsimpy/lib/__init__.py ./radarsimpy/lib
cp ./src/radarsimc/build/*.so ./radarsimpy

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
./src/radarsimc/build/radarsimc_test

echo "## Pytest ##"
pytest

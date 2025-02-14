REM Display header and copyright information
@ECHO OFF

ECHO Automatic build script of radarsimcpp/radarsimpy for Windows
ECHO:
ECHO ----------
ECHO RadarSimPy - A Radar Simulator Built with Python
ECHO Copyright (C) 2018 - PRESENT  radarsimx.com
ECHO E-mail: info@radarsimx.com
ECHO Website: https://radarsimx.com
ECHO:
ECHO  ######                               #####           #     # 
ECHO  #     #   ##   #####    ##   #####  #     # # #    #  #   #  
ECHO  #     #  #  #  #    #  #  #  #    # #       # ##  ##   # #   
ECHO  ######  #    # #    # #    # #    #  #####  # # ## #    #    
ECHO  #   #   ###### #    # ###### #####        # # #    #   # #   
ECHO  #    #  #    # #    # #    # #   #  #     # # #    #  #   #  
ECHO  #     # #    # #####  #    # #    #  #####  # #    # #     # 
ECHO:

REM Store current directory path
SET pwd=%cd%

REM Clean up previous build artifacts
ECHO clean old build files
RMDIR /Q/S .\src\radarsimcpp\build

ECHO clean old radarsimpy module
RMDIR /Q/S .\radarsimpy

REM Create and navigate to build directory
MD ".\src\radarsimcpp\build"
CD ".\src\radarsimcpp\build"

REM Build C++ library with CMake (with CUDA support)
ECHO ## Building radarsimcpp.dll with MSVC ##
cmake -DGPU_BUILD=ON -DGTEST=ON ..
cmake --build . --config Release

REM Build Python extensions for multiple Python versions (Free Tier)
ECHO ## Building radarsimpy with Cython ##
CD %pwd%
REM Build for Python 3.9-3.13 with GPU support
conda.exe run -n py313 python setup.py build_ext -b ./ --tier=free --arch=gpu
conda.exe run -n py312 python setup.py build_ext -b ./ --tier=free --arch=gpu
conda.exe run -n py311 python setup.py build_ext -b ./ --tier=free --arch=gpu
conda.exe run -n py310 python setup.py build_ext -b ./ --tier=free --arch=gpu
conda.exe run -n py39 python setup.py build_ext -b ./ --tier=free --arch=gpu

REM Copy built files to radarsimpy directory
ECHO ## Copying dll files to ./radarsimpy ##
XCOPY ".\src\radarsimcpp\build\Release\radarsimcpp.dll" ".\radarsimpy\"
XCOPY ".\src\radarsimpy\*.py" ".\radarsimpy\"
XCOPY ".\src\radarsimpy\lib\__init__.py" ".\radarsimpy\lib\"

REM Clean up intermediate build files
ECHO ## Cleaning radarsimpy builds ##
RMDIR build /s /q

DEL ".\src\radarsimpy\*.c"
DEL ".\src\radarsimpy\*.cpp"
DEL ".\src\radarsimpy\*.html"
DEL ".\src\radarsimpy\raytracing\*.c"
DEL ".\src\radarsimpy\raytracing\*.cpp"
DEL ".\src\radarsimpy\raytracing\*.html"
DEL ".\src\radarsimpy\lib\*.cpp"
DEL ".\src\radarsimpy\lib\*.html"
DEL ".\src\*.cpp"
DEL ".\src\*.html"

REM Create FreeTier GPU distribution
ECHO ## Copying lib files to freetier release folder ##
RMDIR /Q/S .\Windows_x86_64_GPU_FreeTier
XCOPY /E /I .\radarsimpy .\Windows_x86_64_GPU_FreeTier\radarsimpy

RMDIR /Q/S .\radarsimpy

REM Build Standard Tier GPU version
REM Build Python extensions for multiple Python versions (Standard Tier)
conda.exe run -n py313 python setup.py build_ext -b ./ --tier=standard --arch=gpu
conda.exe run -n py312 python setup.py build_ext -b ./ --tier=standard --arch=gpu
conda.exe run -n py311 python setup.py build_ext -b ./ --tier=standard --arch=gpu
conda.exe run -n py310 python setup.py build_ext -b ./ --tier=standard --arch=gpu
conda.exe run -n py39 python setup.py build_ext -b ./ --tier=standard --arch=gpu

REM Copy built files to radarsimpy directory
ECHO ## Copying dll files to ./radarsimpy ##
XCOPY ".\src\radarsimcpp\build\Release\radarsimcpp.dll" ".\radarsimpy\"
XCOPY ".\src\radarsimpy\*.py" ".\radarsimpy\"
XCOPY ".\src\radarsimpy\lib\__init__.py" ".\radarsimpy\lib\"

REM Clean up intermediate build files
ECHO ## Cleaning radarsimpy builds ##
RMDIR build /s /q

DEL ".\src\radarsimpy\*.c"
DEL ".\src\radarsimpy\*.cpp"
DEL ".\src\radarsimpy\*.html"
DEL ".\src\radarsimpy\raytracing\*.c"
DEL ".\src\radarsimpy\raytracing\*.cpp"
DEL ".\src\radarsimpy\raytracing\*.html"
DEL ".\src\radarsimpy\lib\*.cpp"
DEL ".\src\radarsimpy\lib\*.html"
DEL ".\src\*.cpp"
DEL ".\src\*.html"

REM Create Standard Tier GPU distribution
ECHO ## Copying lib files to standard release folder ##
RMDIR /Q/S .\Windows_x86_64_GPU
XCOPY /E /I .\radarsimpy .\Windows_x86_64_GPU\radarsimpy

REM Run unit tests
ECHO ## Run Google test ##
.\src\radarsimcpp\build\Release\radarsimcpp_test.exe

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

REM Build C++ library with CMake
ECHO ## Building radarsimcpp.dll with MSVC ##
cmake -DGTEST=ON ..
cmake --build . --config Release

REM Build Python extensions for multiple Python versions (Free Tier)
ECHO ## Building radarsimpy with Cython ##
CD %pwd%
REM Build for Python 3.9-3.12 with CPU support
conda.exe run -n py312 python setup.py build_ext -b ./ --tier free --arch cpu
conda.exe run -n py311 python setup.py build_ext -b ./ --tier free --arch cpu
conda.exe run -n py310 python setup.py build_ext -b ./ --tier free --arch cpu
conda.exe run -n py39 python setup.py build_ext -b ./ --tier free --arch cpu

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

REM Create FreeTier distribution
ECHO ## Copying lib files to freetier release folder ##
RMDIR /Q/S .\Windows_x86_64_CPU_FreeTier
XCOPY /E /I .\radarsimpy .\Windows_x86_64_CPU_FreeTier\radarsimpy

RMDIR /Q/S .\radarsimpy

REM Build Standard Tier version
REM Build Python extensions for multiple Python versions (Standard Tier)
conda.exe run -n py312 python setup.py build_ext -b ./ --tier standard --arch cpu
conda.exe run -n py311 python setup.py build_ext -b ./ --tier standard --arch cpu
conda.exe run -n py310 python setup.py build_ext -b ./ --tier standard --arch cpu
conda.exe run -n py39 python setup.py build_ext -b ./ --tier standard --arch cpu

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

REM Create Standard Tier distribution
ECHO ## Copying lib files to standard release folder ##
RMDIR /Q/S .\Windows_x86_64_CPU
XCOPY /E /I .\radarsimpy .\Windows_x86_64_CPU\radarsimpy

REM Build completed
ECHO ## Build completed ##

REM Run unit tests
ECHO ## Run Google test ##
.\src\radarsimcpp\build\Release\radarsimcpp_test.exe

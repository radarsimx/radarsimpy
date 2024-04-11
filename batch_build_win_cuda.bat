@ECHO OFF

ECHO Automatic build script of radarsimcpp/radarsimpy for Windows
ECHO:
ECHO ----------
ECHO RadarSimPy - A Radar Simulator Built with Python
ECHO Copyright (C) 2018 - PRESENT  radarsimx.com
ECHO E-mail: info@radarsimx.com
ECHO Website: https://radarsimx.com
ECHO:
ECHO  *******                  **                   ******** **             **     **
ECHO /**////**                /**                  **////// //             //**   **
ECHO /**   /**   ******       /**  ******   ******/**        ** **********  //** **
ECHO /*******   //////**   ****** //////** //**//*/*********/**//**//**//**  //***
ECHO /**///**    *******  **///**  *******  /** / ////////**/** /** /** /**   **/**
ECHO /**  //**  **////** /**  /** **////**  /**          /**/** /** /** /**  ** //**
ECHO /**   //**//********//******//********/***    ******** /** *** /** /** **   //**
ECHO //     //  ////////  //////  //////// ///    ////////  // ///  //  // //     //
ECHO:

SET pwd=%cd%

ECHO clean old build files
RMDIR /Q/S .\src\radarsimcpp\build

ECHO clean old radarsimpy module
RMDIR /Q/S .\radarsimpy

@REM go to the build folder
MD ".\src\radarsimcpp\build"
CD ".\src\radarsimcpp\build"

ECHO ## Building radarsimcpp.dll with MSVC ##
@REM MSVC needs to set the build type using '--config Relesae' 
cmake -DGPU_BUILD=ON -DGTEST=ON ..
cmake --build . --config Release

ECHO ## Building radarsimpy with Cython ##
CD %pwd%
conda.exe run -n py38 python setup.py build_ext -b ./ --tier free --arch gpu
conda.exe run -n py39 python setup.py build_ext -b ./ --tier free --arch gpu
conda.exe run -n py310 python setup.py build_ext -b ./ --tier free --arch gpu
conda.exe run -n py311 python setup.py build_ext -b ./ --tier free --arch gpu
conda.exe run -n py312 python setup.py build_ext -b ./ --tier free --arch gpu

ECHO ## Copying dll files to ./radarsimpy ##
XCOPY ".\src\radarsimcpp\build\Release\radarsimcpp.dll" ".\radarsimpy\"
XCOPY ".\src\radarsimpy\*.py" ".\radarsimpy\"
XCOPY ".\src\radarsimpy\lib\__init__.py" ".\radarsimpy\lib\"

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

ECHO ## Copying lib files to freetier release folder ##

RMDIR /Q/S .\Windows_x86_64_GPU_FreeTier
XCOPY /E /I .\radarsimpy .\Windows_x86_64_GPU_FreeTier\radarsimpy

RMDIR /Q/S .\radarsimpy

conda.exe run -n py38 python setup.py build_ext -b ./ --tier standard --arch gpu
conda.exe run -n py39 python setup.py build_ext -b ./ --tier standard --arch gpu
conda.exe run -n py310 python setup.py build_ext -b ./ --tier standard --arch gpu
conda.exe run -n py311 python setup.py build_ext -b ./ --tier standard --arch gpu
conda.exe run -n py312 python setup.py build_ext -b ./ --tier standard --arch gpu

ECHO ## Copying dll files to ./radarsimpy ##
XCOPY ".\src\radarsimcpp\build\Release\radarsimcpp.dll" ".\radarsimpy\"
XCOPY ".\src\radarsimpy\*.py" ".\radarsimpy\"
XCOPY ".\src\radarsimpy\lib\__init__.py" ".\radarsimpy\lib\"

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

ECHO ## Copying lib files to standard release folder ##

RMDIR /Q/S .\Windows_x86_64_GPU
XCOPY /E /I .\radarsimpy .\Windows_x86_64_GPU\radarsimpy

ECHO ## Build completed ##

ECHO ## Run Google test ##
.\src\radarsimcpp\build\Release\radarsimcpp_test.exe

ECHO ## Pytest ##
pytest

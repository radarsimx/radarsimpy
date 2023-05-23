@ECHO OFF

ECHO Automatic build script of radarsimc/radarsimpy for Windows
ECHO:
ECHO ----------
ECHO RadarSimPy - A Radar Simulator Built with Python
ECHO Copyright (C) 2018 - PRESENT  Zhengyu Peng
ECHO E-mail: zpeng.me@gmail.com
ECHO Website: https://zpeng.me
ECHO:
ECHO `                      `
ECHO -:.                  -#:
ECHO -//:.              -###:
ECHO -////:.          -#####:
ECHO -/:.://:.      -###++##:
ECHO ..   `://:-  -###+. :##:
ECHO        `:/+####+.   :##:
ECHO .::::::::/+###.     :##:
ECHO .////-----+##:    `:###:
ECHO  `-//:.   :##:  `:###/.
ECHO    `-//:. :##:`:###/.
ECHO      `-//:+######/.
ECHO        `-/+####/.
ECHO          `+##+.
ECHO           :##:
ECHO           :##:
ECHO           :##:
ECHO           :##:
ECHO           :##:
ECHO            .+:

SET pwd=%cd%

ECHO clean old build files
RMDIR /Q/S .\src\radarsimc\build

ECHO clean old radarsimpy module
RMDIR /Q/S .\radarsimpy

@REM go to the build folder
MD ".\src\radarsimc\build"
CD ".\src\radarsimc\build"

ECHO ## Building radarsimc.dll with MSVC ##
@REM MSVC needs to set the build type using '--config Relesae' 
cmake -DGTEST=ON ..
cmake --build . --config Release

ECHO ## Building radarsimpy with Cython ##
CD %pwd%
python setup.py build_ext -b ./

ECHO ## Copying dll files to ./radarsimpy ##
XCOPY ".\src\radarsimc\build\Release\radarsimc.dll" ".\radarsimpy\"
XCOPY ".\src\radarsimpy\__init__.py" ".\radarsimpy\"
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

ECHO ## Copying lib files to unit test folder ##

RMDIR /Q/S .\tests\radarsimpy
XCOPY /E /I .\radarsimpy .\tests\radarsimpy

ECHO ## Build completed ##

@REM ECHO ## Run Google test ##
@REM .\src\radarsimc\build\Release\radarsimc_test.exe

@REM ECHO ## Pytest ##
@REM pytest

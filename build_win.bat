@ECHO OFF

set TIER=standard
set ARCH=cpu
set TEST=on

goto GETOPTS

:Help
    ECHO:
    ECHO Usages:
    ECHO    --help    Show the usages of the parameters
    ECHO    --tier    Build tier, choose 'standard' or 'free'. Default is 'standard'
    ECHO    --arch    Build architecture, choose 'cpu' or 'gpu'. Default is 'cpu'
    ECHO    --test    Enable or disable unit test, choose 'on' or 'off'. Default is 'on'
    ECHO:
    goto EOF

:GETOPTS
    if /I "%1" == "--help" goto Help
    if /I "%1" == "--tier" set TIER=%2 & shift
    if /I "%1" == "--arch" set ARCH=%2 & shift
    if /I "%1" == "--test" set TEST=%2 & shift
    shift
    if not "%1" == "" goto GETOPTS

    if /I NOT %TIER% == free (
        if /I NOT %TIER% == standard (
            ECHO ERROR: Invalid --tier parameters, please choose 'free' or 'standard'
            goto EOF
        )
    )

    if /I NOT %ARCH% == cpu (
        if /I NOT %ARCH% == gpu (
            ECHO ERROR: Invalid --arch parameters, please choose 'cpu' or 'gpu'
            goto EOF
        )
    )

    if /I NOT %TEST% == on (
        if /I NOT %TEST% == off (
            ECHO ERROR: Invalid --test parameters, please choose 'on' or 'off'
            goto EOF
        )
    )

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
if /I %ARCH% == gpu (
    if /I %TEST% == on (
        cmake -DGPU_BUILD=ON -DGTEST=ON ..
    ) else (
        cmake -DGPU_BUILD=ON -DGTEST=OFF ..
    )
) else if /I %ARCH% == cpu (
    if /I %TEST% == on (
        cmake -DGTEST=ON ..
    ) else (
        cmake -DGTEST=OFF ..
    )
)
cmake --build . --config Release

ECHO ## Building radarsimpy with Cython ##
CD %pwd%
python setup.py build_ext -b ./ --tier %TIER% --arch %ARCH%

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

ECHO ## Build completed ##

if /I %TEST% == on (
    ECHO ## Run Google test ##
    .\src\radarsimcpp\build\Release\radarsimcpp_test.exe

    ECHO ## Pytest ##
    pytest
)

:EOF
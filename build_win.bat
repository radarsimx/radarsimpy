@ECHO OFF

REM Default build configuration
set TIER=standard
set ARCH=cpu
set TEST=on

goto GETOPTS

REM Help section - displays command line parameter usage
:Help
    ECHO:
    ECHO Usages:
    ECHO    --help    Show the usages of the parameters
    ECHO    --tier    Build tier, choose 'standard' or 'free'. Default is 'standard'
    ECHO    --arch    Build architecture, choose 'cpu' or 'gpu'. Default is 'cpu'
    ECHO    --test    Enable or disable unit test, choose 'on' or 'off'. Default is 'on'
    ECHO:
    goto EOF

REM Command line parameter parsing section
:GETOPTS
    REM Parse command line arguments
    if /I "%1" == "--help" goto Help
    if /I "%1" == "--tier" set TIER=%2 & shift
    if /I "%1" == "--arch" set ARCH=%2 & shift
    if /I "%1" == "--test" set TEST=%2 & shift
    shift
    if not "%1" == "" goto GETOPTS

    REM Validate tier parameter
    if /I NOT %TIER% == free (
        if /I NOT %TIER% == standard (
            ECHO ERROR: Invalid --tier parameters, please choose 'free' or 'standard'
            goto EOF
        )
    )

    REM Validate architecture parameter
    if /I NOT %ARCH% == cpu (
        if /I NOT %ARCH% == gpu (
            ECHO ERROR: Invalid --arch parameters, please choose 'cpu' or 'gpu'
            goto EOF
        )
    )

    REM Validate test parameter
    if /I NOT %TEST% == on (
        if /I NOT %TEST% == off (
            ECHO ERROR: Invalid --test parameters, please choose 'on' or 'off'
            goto EOF
        )
    )

REM Display banner and copyright information
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

REM Store current directory for later use
SET pwd=%cd%

REM Clean up previous build artifacts
ECHO clean old build files
RMDIR /Q/S .\src\radarsimcpp\build

ECHO clean old radarsimpy module
RMDIR /Q/S .\radarsimpy

REM Create and enter build directory
@REM Create fresh build directory and change to it
MD ".\src\radarsimcpp\build"
CD ".\src\radarsimcpp\build"

REM Configure CMake build based on architecture and test settings
ECHO ## Building radarsimcpp.dll with MSVC ##
@REM MSVC requires explicit Release configuration
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

REM Build Python extensions using Cython
ECHO ## Building radarsimpy with Cython ##
CD %pwd%
python setup.py build_ext -b ./ --tier %TIER% --arch %ARCH%

REM Copy built artifacts to final locations
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

SET TEST_FAILED=0
REM Run tests if enabled
if /I %TEST% == on (
    ECHO ## Run Google test ##
    .\src\radarsimcpp\build\Release\radarsimcpp_test.exe
    if errorlevel 1 (
        echo Google test failed!
        SET TEST_FAILED=1
    )

    ECHO ## Pytest ##
    pytest
    if errorlevel 1 (
        echo Pytest failed!
        SET TEST_FAILED=1
    )
)

exit /b %TEST_FAILED%

:EOF
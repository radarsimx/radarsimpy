@echo off

REM ==============================================================================
REM RadarSimPy Build Script for Windows
REM ==============================================================================
REM
REM This script builds the RadarSimPy radar simulation library for Windows.
REM It supports both CPU (OpenMP) and GPU (CUDA) architectures with optional
REM unit testing using Google Test and pytest.
REM
REM Requirements:
REM   - CMake 3.18 or higher
REM   - Visual Studio 2019 or higher (with C++ tools)
REM   - Python 3.7 or higher with Cython
REM   - CUDA SDK (for GPU builds)
REM   - pytest (for Python tests)
REM
REM Usage:
REM   build_win.bat [--tier=<standard|free>] [--arch=<cpu|gpu>] [--test=<on|off>] [--jobs=<auto|number>]
REM
REM ==============================================================================

REM Default build configuration
set TIER=standard
set ARCH=cpu
set TEST=on
set BUILD_TYPE=Release
set SCRIPT_DIR=%~dp0
set BUILD_FAILED=0
set JOBS=0

REM Initialize error tracking
set CMAKE_FAILED=0
set PYTHON_BUILD_FAILED=0
set TEST_FAILED=0

goto GETOPTS

REM Help section - displays command line parameter usage
:Help
    echo.
    echo RadarSimPy Build Script for Windows
    echo.
    echo Usage:
    echo   build_win.bat [OPTIONS]
    echo.
    echo Options:
    echo   --help    Show this help message
    echo   --tier    Build tier: 'standard' or 'free' - default: standard
    echo   --arch    Build architecture: 'cpu' or 'gpu' - default: cpu
    echo   --test    Enable unit tests: 'on' or 'off' - default: on
    echo   --jobs    Number of parallel jobs: 'auto' or number - default: auto
    echo.
    echo Examples:
    echo   build_win.bat --arch=gpu --test=off
    echo   build_win.bat --tier=free --arch=cpu --jobs=4
    echo   build_win.bat --jobs=auto
    echo.
    goto EOF

REM Command line parameter parsing section
:GETOPTS
    REM Parse command line arguments
    if /I "%1" == "--help" goto Help
    if /I "%1" == "-h" goto Help
    if /I "%1" == "--tier" (
        set TIER=%2
        shift
        shift
        goto GETOPTS
    )
    if /I "%1" == "--arch" (
        set ARCH=%2
        shift
        shift
        goto GETOPTS
    )
    if /I "%1" == "--test" (
        set TEST=%2
        shift
        shift
        goto GETOPTS
    )
    if /I "%1" == "--jobs" (
        set JOBS=%2
        shift
        shift
        goto GETOPTS
    )
    if not "%1" == "" (
        echo ERROR: Unknown parameter: %1
        echo Use --help for usage information
        goto ERROR_EXIT
    )

    REM Validate tier parameter
    if /I NOT "%TIER%" == "free" (
        if /I NOT "%TIER%" == "standard" (
            echo ERROR: Invalid --tier parameter '%TIER%'. Please choose 'free' or 'standard'
            goto ERROR_EXIT
        )
    )

    REM Validate architecture parameter
    if /I NOT "%ARCH%" == "cpu" (
        if /I NOT "%ARCH%" == "gpu" (
            echo ERROR: Invalid --arch parameter '%ARCH%'. Please choose 'cpu' or 'gpu'
            goto ERROR_EXIT
        )
    )

    REM Validate test parameter
    if /I NOT "%TEST%" == "on" (
        if /I NOT "%TEST%" == "off" (
            echo ERROR: Invalid --test parameter '%TEST%'. Please choose 'on' or 'off'
            goto ERROR_EXIT
        )
    )

    REM Validate and set jobs parameter
    if /I "%JOBS%" == "0" (
        set JOBS=auto
    )
    if /I "%JOBS%" == "auto" (
        REM Auto-detect number of CPU cores
        if defined NUMBER_OF_PROCESSORS (
            set JOBS=%NUMBER_OF_PROCESSORS%
        ) else (
            set JOBS=4
        )
        echo INFO: Auto-detected %JOBS% CPU cores for parallel build
    ) else (
        REM Validate that jobs is a positive integer
        echo %JOBS%| findstr /r "^[1-9][0-9]*$" >nul
        if %errorlevel% neq 0 (
            echo ERROR: Invalid --jobs parameter '%JOBS%'. Please provide 'auto' or a positive integer
            goto ERROR_EXIT
        )
    )

REM Validate build environment
:VALIDATE_ENVIRONMENT
    echo INFO: Validating build environment...
    
    REM Check for CMake
    cmake --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: CMake is not installed or not in PATH
        echo Please install CMake 3.18 or higher
        goto ERROR_EXIT
    )
    
    REM Check for Python
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.7 or higher
        goto ERROR_EXIT
    )
    
    REM Check for CUDA if GPU build is requested
    if /I "%ARCH%" == "gpu" (
        nvcc --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo ERROR: CUDA toolkit is not installed or not in PATH
            echo Please install CUDA SDK for GPU builds
            goto ERROR_EXIT
        )
    )
    
    REM Check for pytest if tests are enabled
    if /I "%TEST%" == "on" (
        pytest --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo WARNING: pytest is not installed or not in PATH
            echo Python tests will be skipped
        )
    )

REM Display banner and copyright information
:DISPLAY_BANNER
    echo.
    echo ====================================================================
    echo RadarSimPy - A Radar Simulator Built with Python
    echo Copyright (C) 2018 - PRESENT  radarsimx.com
    echo E-mail: info@radarsimx.com
    echo Website: https://radarsimx.com
    echo ====================================================================
    echo.
    echo Build Configuration:
    echo   Tier: %TIER%
    echo   Architecture: %ARCH%
    echo   Tests: %TEST%
    echo   Build Type: %BUILD_TYPE%
    echo   Parallel Jobs: %JOBS%
    echo.
    echo ######                               #####           #     # 
    echo #     #   ##   #####    ##   #####  #     # # #    #  #   #  
    echo #     #  #  #  #    #  #  #  #    # #       # ##  ##   # #   
    echo ######  #    # #    # #    # #    #  #####  # # ## #    #    
    echo #   #   ###### #    # ###### #####        # # #    #   # #   
    echo #    #  #    # #    # #    # #   #  #     # # #    #  #   #  
    echo #     # #    # #####  #    # #    #  #####  # #    # #     # 
    echo.

REM Store current directory for later use
set PWD=%cd%

REM Start build process
:BUILD_START
    echo INFO: Starting build process...
    echo INFO: Current directory: %PWD%

REM Clean up previous build artifacts
:CLEAN_BUILD
    echo INFO: Cleaning previous build artifacts...
    
    if exist ".\src\radarsimcpp\build" (
        rmdir /q /s ".\src\radarsimcpp\build" 2>nul
        if %errorlevel% neq 0 (
            echo WARNING: Could not fully clean C++ build directory
        )
    )
    
    if exist ".\radarsimpy" (
        rmdir /q /s ".\radarsimpy" 2>nul
        if %errorlevel% neq 0 (
            echo WARNING: Could not fully clean Python module directory
        )
    )
    
    if exist ".\build" (
        rmdir /q /s ".\build" 2>nul
        if %errorlevel% neq 0 (
            echo WARNING: Could not fully clean Python build directory
        )
    )

REM Build C++ library
:BUILD_CPP
    echo INFO: Building RadarSimCpp library...
    
    REM Create build directory
    if not exist ".\src\radarsimcpp\build" (
        mkdir ".\src\radarsimcpp\build"
        if %errorlevel% neq 0 (
            echo ERROR: Failed to create C++ build directory
            goto ERROR_EXIT
        )
    )
    
    REM Change to build directory
    pushd ".\src\radarsimcpp\build"
    
    REM Configure CMake build based on architecture and test settings
    echo INFO: Configuring CMake build - Architecture: %ARCH%, Tests: %TEST%...
    
    if /I "%ARCH%" == "gpu" (
        if /I "%TEST%" == "on" (
            cmake -DGPU_BUILD=ON -DGTEST=ON ..
        ) else (
            cmake -DGPU_BUILD=ON -DGTEST=OFF ..
        )
    ) else (
        if /I "%TEST%" == "on" (
            cmake -DGTEST=ON ..
        ) else (
            cmake -DGTEST=OFF ..
        )
    )
    
    if %errorlevel% neq 0 (
        echo ERROR: CMake configuration failed
        set CMAKE_FAILED=1
        popd
        goto ERROR_EXIT
    )
    
    REM Build the C++ library
    echo INFO: Building C++ library with %BUILD_TYPE% configuration using %JOBS% parallel jobs...
    cmake --build . --config %BUILD_TYPE% --parallel %JOBS%
    
    if %errorlevel% neq 0 (
        echo ERROR: C++ library build failed
        set CMAKE_FAILED=1
        popd
        goto ERROR_EXIT
    )
    
    popd
    echo INFO: C++ library build completed successfully

REM Build Python extensions
:BUILD_PYTHON
    echo INFO: Building Python extensions with Cython...
    
    cd /d "%PWD%"
    python setup.py build_ext -b ./ --tier %TIER% --arch %ARCH%
    
    if %errorlevel% neq 0 (
        echo ERROR: Python extension build failed
        set PYTHON_BUILD_FAILED=1
        goto ERROR_EXIT
    )
    
    echo INFO: Python extension build completed successfully

REM Copy built artifacts
:COPY_ARTIFACTS
    echo INFO: Copying build artifacts...
    
    REM Ensure radarsimpy directory exists
    if not exist ".\radarsimpy" (
        mkdir ".\radarsimpy"
        if %errorlevel% neq 0 (
            echo ERROR: Failed to create radarsimpy directory
            goto ERROR_EXIT
        )
    )
    
    REM Copy DLL files
    if exist ".\src\radarsimcpp\build\%BUILD_TYPE%\radarsimcpp.dll" (
        copy ".\src\radarsimcpp\build\%BUILD_TYPE%\radarsimcpp.dll" ".\radarsimpy\" >nul
        if %errorlevel% neq 0 (
            echo ERROR: Failed to copy radarsimcpp.dll
            goto ERROR_EXIT
        )
    ) else (
        echo ERROR: radarsimcpp.dll not found in build directory
        goto ERROR_EXIT
    )
    
    REM Copy Python files
    if exist ".\src\radarsimpy\*.py" (
        copy ".\src\radarsimpy\*.py" ".\radarsimpy\" >nul
        if %errorlevel% neq 0 (
            echo WARNING: Some Python files may not have been copied
        )
    )
    
    REM Create lib directory and copy files
    if not exist ".\radarsimpy\lib" (
        mkdir ".\radarsimpy\lib"
    )
    
    if exist ".\src\radarsimpy\lib\__init__.py" (
        copy ".\src\radarsimpy\lib\__init__.py" ".\radarsimpy\lib\" >nul
        if %errorlevel% neq 0 (
            echo WARNING: Failed to copy lib/__init__.py
        )
    )
    
    echo INFO: Artifacts copied successfully

REM Clean up intermediate build files
:CLEANUP_INTERMEDIATE
    echo INFO: Cleaning intermediate build files...
    
    REM Remove Python build directory
    if exist ".\build" (
        rmdir /q /s ".\build" 2>nul
        if %errorlevel% neq 0 (
            echo WARNING: Could not fully clean Python build directory
        )
    )
    
    REM Clean up generated files in src directories
    if exist ".\src\radarsimpy\*.c" del /q ".\src\radarsimpy\*.c" 2>nul
    if exist ".\src\radarsimpy\*.cpp" del /q ".\src\radarsimpy\*.cpp" 2>nul
    if exist ".\src\radarsimpy\*.html" del /q ".\src\radarsimpy\*.html" 2>nul
    if exist ".\src\radarsimpy\raytracing\*.c" del /q ".\src\radarsimpy\raytracing\*.c" 2>nul
    if exist ".\src\radarsimpy\raytracing\*.cpp" del /q ".\src\radarsimpy\raytracing\*.cpp" 2>nul
    if exist ".\src\radarsimpy\raytracing\*.html" del /q ".\src\radarsimpy\raytracing\*.html" 2>nul
    if exist ".\src\radarsimpy\lib\*.cpp" del /q ".\src\radarsimpy\lib\*.cpp" 2>nul
    if exist ".\src\radarsimpy\lib\*.html" del /q ".\src\radarsimpy\lib\*.html" 2>nul
    if exist ".\src\*.cpp" del /q ".\src\*.cpp" 2>nul
    if exist ".\src\*.html" del /q ".\src\*.html" 2>nul
    
    echo INFO: Intermediate files cleaned

REM Run tests if enabled
:RUN_TESTS
    if /I "%TEST%" == "off" (
        echo INFO: Tests are disabled, skipping test execution
        goto BUILD_SUCCESS
    )
    
    echo INFO: Running tests...
    
    REM Run Google Test (C++) using CTest
    if exist ".\src\radarsimcpp\build\%BUILD_TYPE%\radarsimcpp_test.exe" (
        echo INFO: Running Google Test for C++ using CTest...
        ctest --test-dir ".\src\radarsimcpp\build" -C %BUILD_TYPE% --verbose
        if %errorlevel% neq 0 (
            echo ERROR: Google Test failed
            set TEST_FAILED=1
        ) else (
            echo INFO: Google Test passed
        )
    ) else (
        echo WARNING: Google Test executable not found, skipping C++ tests
    )
    
    REM Run Python tests with pytest
    pytest --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo INFO: Running Python tests with pytest...
        pytest
        if %errorlevel% neq 0 (
            echo ERROR: Python tests failed
            set TEST_FAILED=1
        ) else (
            echo INFO: Python tests passed
        )
    ) else (
        echo WARNING: pytest not available, skipping Python tests
    )
    
    REM Check overall test results
    if %TEST_FAILED% neq 0 (
        echo ERROR: Some tests failed
        goto ERROR_EXIT
    )
    
    echo INFO: All tests passed successfully

REM Build completion
:BUILD_SUCCESS
    echo.
    echo ====================================================================
    echo BUILD COMPLETED SUCCESSFULLY
    echo ====================================================================
    echo.
    echo Build Summary:
    echo   Tier: %TIER%
    echo   Architecture: %ARCH%
    echo   Tests: %TEST%
    echo   Build Type: %BUILD_TYPE%
    echo   Parallel Jobs: %JOBS%
    echo.
    echo Output Location:
    echo   C++ Library: .\src\radarsimcpp\build\%BUILD_TYPE%\
    echo   Python Module: .\radarsimpy\
    echo.
    if /I "%TEST%" == "on" (
        echo All tests passed successfully
    ) else (
        echo Tests were skipped - disabled
    )
    echo.
    echo ====================================================================
    
    exit /b 0

REM Error handling
:ERROR_EXIT
    echo.
    echo ====================================================================
    echo BUILD FAILED
    echo ====================================================================
    echo.
    echo Error Summary:
    if %CMAKE_FAILED% neq 0 echo   - CMake build failed
    if %PYTHON_BUILD_FAILED% neq 0 echo   - Python build failed
    if %TEST_FAILED% neq 0 echo   - Tests failed
    echo.
    echo Please check the error messages above and fix the issues.
    echo.
    echo ====================================================================
    
    exit /b 1

:EOF
    exit /b 0
REM ============================================================================
REM RadarSimPy Windows Build Script
REM Copyright (C) 2018 - PRESENT  radarsimx.com
REM ============================================================================
@ECHO OFF
SETLOCAL EnableDelayedExpansion

REM Configuration
SET SCRIPT_VERSION=2.0
SET PYTHON_VERSIONS=py39 py310 py311 py312 py313
SET BUILD_LOG_DATE=%DATE:~-4,4%%DATE:~-7,2%%DATE:~-10,2%
SET BUILD_LOG_TIME=%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
SET BUILD_LOG_TIME=%BUILD_LOG_TIME: =0%
SET BUILD_LOG=%~dp0build_logs\windows_batch_build_log_%BUILD_LOG_DATE%_%BUILD_LOG_TIME%.log

REM Create build_logs directory if it doesn't exist
IF NOT EXIST "%~dp0build_logs" MD "%~dp0build_logs"

REM Parse command line arguments
SET TIER=both
SET ARCH=cpu
SET SKIP_TESTS=false
SET VERBOSE=false

:parse_args
IF "%~1"=="" GOTO :args_done
IF /I "%~1"=="--tier" (
    SET TIER=%~2
    SHIFT
    SHIFT
    GOTO :parse_args
)
IF /I "%~1"=="--arch" (
    SET ARCH=%~2
    SHIFT
    SHIFT
    GOTO :parse_args
)
IF /I "%~1"=="--skip-tests" (
    SET SKIP_TESTS=true
    SHIFT
    GOTO :parse_args
)
IF /I "%~1"=="--verbose" (
    SET VERBOSE=true
    SHIFT
    GOTO :parse_args
)
IF /I "%~1"=="--help" (
    CALL :show_help
    EXIT /B 0
)
ECHO Unknown argument: %~1
CALL :show_help
EXIT /B 1

:args_done

REM Validate arguments
IF /I NOT "%TIER%"=="free" IF /I NOT "%TIER%"=="standard" IF /I NOT "%TIER%"=="both" (
    CALL :log_error "Invalid tier specified: %TIER%. Must be free, standard, or both."
    EXIT /B 1
)

IF /I NOT "%ARCH%"=="cpu" IF /I NOT "%ARCH%"=="gpu" IF /I NOT "%ARCH%"=="both" (
    CALL :log_error "Invalid architecture specified: %ARCH%. Must be cpu, gpu, or both."
    EXIT /B 1
)

REM Display header and copyright information
CALL :log_info "RadarSimPy Build Script v%SCRIPT_VERSION%"
CALL :log_info "Copyright 2018 - PRESENT  radarsimx.com"
CALL :log_info "E-mail: info@radarsimx.com"
CALL :log_info "Website: https://radarsimx.com"
CALL :log_info ""
CALL :log_info "  ######                               #####           #     #"
CALL :log_info "  #     #   ##   #####    ##   #####  #     # # #    #  #   #"
CALL :log_info "  #     #  #  #  #    #  #  #  #    # #       # ##  ##   # #"
CALL :log_info "  ######  #    # #    # #    # #    #  #####  # # ## #    #"
CALL :log_info "  #   #   ###### #    # ###### #####        # # #    #   # #"
CALL :log_info "  #    #  #    # #    # #    # #   #  #     # # #    #  #   #"
CALL :log_info "  #     # #    # #####  #    # #    #  #####  # #    # #     #"
CALL :log_info ""
CALL :log_info "Build Configuration:"
CALL :log_info "  Tier: %TIER%"
CALL :log_info "  Architecture: %ARCH%"
CALL :log_info "  Skip Tests: %SKIP_TESTS%"
CALL :log_info "  Verbose: %VERBOSE%"
CALL :log_info "  Log File: %BUILD_LOG%"
CALL :log_info ""

REM Store current directory path
SET pwd=%cd%

REM Verify prerequisites
CALL :check_prerequisites
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "Prerequisites check failed"
    EXIT /B 1
)

REM Main build process
CALL :log_info "Starting build process..."

REM Clean up previous build artifacts
CALL :cleanup_build_artifacts
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "Failed to clean build artifacts"
    EXIT /B 1
)

REM Build all combinations of tier and architecture
CALL :build_all_combinations
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "Build failed"
    EXIT /B 1
)

REM Run unit tests if not skipped
IF /I "%SKIP_TESTS%"=="false" (
    CALL :run_tests
    IF !ERRORLEVEL! NEQ 0 (
        CALL :log_error "Tests failed"
        EXIT /B 1
    )
)

CALL :log_info "Build completed successfully!"
EXIT /B 0

REM ============================================================================
REM CORE BUILD FUNCTIONS
REM ============================================================================

:build_all_combinations
CALL :log_info "Building all requested combinations..."

REM Determine which architectures to build
SET arch_list=
IF /I "%ARCH%"=="both" (
    SET arch_list=cpu gpu
) ELSE (
    SET arch_list=%ARCH%
)

REM Determine which tiers to build
SET tier_list=
IF /I "%TIER%"=="both" (
    SET tier_list=free standard
) ELSE (
    SET tier_list=%TIER%
)

REM Build each combination
FOR %%a IN (%arch_list%) DO (
    FOR %%t IN (%tier_list%) DO (
        CALL :build_single_combination "%%a" "%%t"
        IF !ERRORLEVEL! NEQ 0 EXIT /B 1
    )
)

EXIT /B 0

:build_single_combination
SET build_arch=%~1
SET build_tier=%~2
CALL :log_info "Building %build_tier% tier for %build_arch% architecture..."

REM Build C++ library for this architecture
CALL :build_cpp_library "%build_arch%"
IF !ERRORLEVEL! NEQ 0 EXIT /B 1

REM Build Python extensions for this combination
CALL :build_python_extensions "%build_tier%" "%build_arch%"
IF !ERRORLEVEL! NEQ 0 EXIT /B 1

REM Copy built files
CALL :copy_built_files
IF !ERRORLEVEL! NEQ 0 EXIT /B 1

REM Clean up intermediate files
CALL :cleanup_temp_files

REM Create distribution folder
CALL :create_distribution "%build_tier%" "%build_arch%"
IF !ERRORLEVEL! NEQ 0 EXIT /B 1

REM Clean up for next combination
RMDIR /Q/S ".\radarsimpy" 2>NUL

EXIT /B 0

:build_cpp_library
SET build_arch=%~1
CALL :log_info "Building C++ library for %build_arch% architecture..."

REM Clean and recreate build directory
IF EXIST ".\src\radarsimcpp\build" (
    RMDIR /Q/S ".\src\radarsimcpp\build" 2>NUL
)
IF NOT EXIST ".\src\radarsimcpp\build" (
    MD ".\src\radarsimcpp\build"
)
CD ".\src\radarsimcpp\build"

REM Configure CMake with appropriate flags
SET CMAKE_FLAGS=-DGTEST=ON
IF /I "%build_arch%"=="gpu" (
    SET CMAKE_FLAGS=%CMAKE_FLAGS% -DGPU_BUILD=ON
    CALL :log_info "Configuring with GPU/CUDA support..."
) ELSE (
    CALL :log_info "Configuring with CPU support..."
)

REM Configure and build
cmake %CMAKE_FLAGS% .. 2>&1
IF !ERRORLEVEL! NEQ 0 (
    CD "%pwd%"
    CALL :log_error "CMake configuration failed for %build_arch%"
    EXIT /B 1
)

cmake --build . --config Release 2>&1
IF !ERRORLEVEL! NEQ 0 (
    CD "%pwd%"
    CALL :log_error "C++ library build failed for %build_arch%"
    EXIT /B 1
)

CD "%pwd%"
CALL :log_info "C++ library built successfully for %build_arch%."
EXIT /B 0

:build_python_extensions
SET build_tier=%~1
SET build_arch=%~2
CALL :log_info "Building Python extensions for %build_tier% tier with %build_arch% architecture..."


FOR %%v IN (%PYTHON_VERSIONS%) DO (
    CALL :log_info "Building for %%v..."
    conda info --envs | findstr /C:"%%v" >NUL 2>&1
    IF !ERRORLEVEL! EQU 0 (
        conda.exe run -n %%v python setup.py build_ext -b ./ --tier %build_tier% --arch %build_arch% 2>&1
        IF !ERRORLEVEL! NEQ 0 (
            CALL :log_warning "Failed to build for %%v, continuing..."
        )
    ) ELSE (
        CALL :log_warning "Environment %%v not found, skipping..."
    )
)
EXIT /B 0

:copy_built_files
CALL :log_info "Copying built files..."

REM Ensure destination directory exists
IF NOT EXIST ".\radarsimpy" MD ".\radarsimpy"
IF NOT EXIST ".\radarsimpy\lib" MD ".\radarsimpy\lib"

REM Copy files with error checking
XCOPY ".\src\radarsimcpp\build\Release\radarsimcpp.dll" ".\radarsimpy\" /Y 2>NUL
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "Failed to copy radarsimcpp.dll"
    EXIT /B 1
)

XCOPY ".\src\radarsimpy\*.py" ".\radarsimpy\" /Y 2>NUL
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "Failed to copy Python files"
    EXIT /B 1
)

XCOPY ".\src\radarsimpy\lib\__init__.py" ".\radarsimpy\lib\" /Y 2>NUL
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "Failed to copy lib __init__.py"
    EXIT /B 1
)

EXIT /B 0

:create_distribution
SET build_tier=%~1
SET build_arch=%~2
CALL :log_info "Creating %build_tier% distribution for %build_arch% architecture..."

REM Determine architecture suffix for folder name
SET ARCH_SUFFIX=CPU
IF /I "%build_arch%"=="gpu" SET ARCH_SUFFIX=GPU

IF /I "%build_tier%"=="free" (
    SET dist_folder=.\release\trial\Windows_x86_64_%ARCH_SUFFIX%
) ELSE (
    SET dist_folder=.\release\standard\Windows_x86_64_%ARCH_SUFFIX%
)

REM Clean and create distribution folder
IF EXIST "%dist_folder%" (
    RMDIR /Q/S "%dist_folder%" 2>NUL
)
XCOPY /E /I ".\radarsimpy" "%dist_folder%\radarsimpy" 2>&1
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "Failed to create %build_tier% distribution for %build_arch%"
    EXIT /B 1
)

EXIT /B 0

REM ============================================================================
REM UTILITY FUNCTIONS
REM ============================================================================

:show_help
ECHO Usage: %~nx0 [options]
ECHO.
ECHO Options:
ECHO   --tier=^<free^|standard^|both^>  Build tier (default: both)
ECHO   --arch=^<cpu^|gpu^|both^>        Architecture (default: cpu)
ECHO   --skip-tests                   Skip running unit tests
ECHO   --verbose                      Enable verbose output
ECHO   --help                         Show this help message
ECHO.
ECHO Examples:
ECHO   %~nx0 --tier=free --arch=cpu
ECHO   %~nx0 --tier=standard --arch=gpu
ECHO   %~nx0 --arch=both --skip-tests
ECHO   %~nx0 --tier=both --arch=both
ECHO   %~nx0 --verbose
EXIT /B 0

REM ============================================================================
REM LOGGING FUNCTIONS
REM ============================================================================

:log_info
SET msg=%~1
ECHO [INFO] %msg%
ECHO [%DATE% %TIME%] [INFO] %msg% >> "%BUILD_LOG%"
EXIT /B 0

:log_error
SET msg=%~1
ECHO [ERROR] %msg%
ECHO [%DATE% %TIME%] [ERROR] %msg% >> "%BUILD_LOG%"
EXIT /B 0

:log_warning
SET msg=%~1
ECHO [WARNING] %msg%
ECHO [%DATE% %TIME%] [WARNING] %msg% >> "%BUILD_LOG%"
EXIT /B 0

REM ============================================================================
REM PREREQUISITE CHECK FUNCTIONS
REM ============================================================================

:check_prerequisites
CALL :log_info "Checking prerequisites..."

REM Check if cmake is available
CALL :log_info "Checking CMake..."
cmake --version >NUL 2>&1
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "CMake not found. Please install CMake and add it to PATH."
    EXIT /B 1
)
CALL :log_info "CMake found."

REM Check CUDA availability for GPU builds
IF /I "%ARCH%"=="gpu" (
    CALL :log_info "GPU architecture requested, checking CUDA..."
    CALL :check_cuda_prerequisites
    IF !ERRORLEVEL! NEQ 0 EXIT /B 1
) ELSE IF /I "%ARCH%"=="both" (
    CALL :log_info "Both architectures requested, checking CUDA..."
    CALL :check_cuda_prerequisites
    IF !ERRORLEVEL! NEQ 0 EXIT /B 1
)

REM Check Python environments
CALL :log_info "Checking Python environments..."
SET missing_envs=false
FOR %%v IN (%PYTHON_VERSIONS%) DO (
    CALL :log_info "Checking environment %%v..."
    conda info --envs | findstr /C:"%%v" >NUL 2>&1
    IF !ERRORLEVEL! NEQ 0 (
        CALL :log_error "Python environment %%v not found. This is required for the build."
        SET missing_envs=true
    ) ELSE (
        CALL :log_info "Environment %%v found."
    )
)

IF "%missing_envs%"=="true" (
    CALL :log_error "One or more required Python environments are missing. Please create all required environments before building."
    EXIT /B 1
)

CALL :log_info "Prerequisites check completed."
EXIT /B 0

:check_cuda_prerequisites
CALL :log_info "Checking CUDA prerequisites..."

REM Check if nvcc is available
CALL :log_info "Checking NVCC..."
nvcc --version >NUL 2>&1
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "NVCC not found. Please install CUDA Toolkit and add it to PATH."
    EXIT /B 1
)
CALL :log_info "NVCC found."

REM Check if nvidia-smi is available
CALL :log_info "Checking nvidia-smi..."
nvidia-smi >NUL 2>&1
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_warning "nvidia-smi not found. GPU may not be available."
) ELSE (
    CALL :log_info "nvidia-smi found."
)

CALL :log_info "CUDA prerequisites check completed."
EXIT /B 0

REM ============================================================================
REM CLEANUP FUNCTIONS
REM ============================================================================

:cleanup_build_artifacts
CALL :log_info "Cleaning up previous build artifacts..."

REM Clean build directories
IF EXIST ".\src\radarsimcpp\build" (
    RMDIR /Q/S ".\src\radarsimcpp\build" 2>NUL
)
IF EXIST ".\radarsimpy" (
    RMDIR /Q/S ".\radarsimpy" 2>NUL
)
IF EXIST ".\build" (
    RMDIR /Q/S ".\build" 2>NUL
)

CALL :cleanup_temp_files
EXIT /B 0

:cleanup_temp_files
CALL :log_info "Cleaning temporary files..."

REM Clean generated files
DEL ".\src\radarsimpy\*.c" 2>NUL
DEL ".\src\radarsimpy\*.cpp" 2>NUL
DEL ".\src\radarsimpy\*.html" 2>NUL
DEL ".\src\radarsimpy\raytracing\*.c" 2>NUL
DEL ".\src\radarsimpy\raytracing\*.cpp" 2>NUL
DEL ".\src\radarsimpy\raytracing\*.html" 2>NUL
DEL ".\src\radarsimpy\lib\*.c" 2>NUL
DEL ".\src\radarsimpy\lib\*.cpp" 2>NUL
DEL ".\src\radarsimpy\lib\*.html" 2>NUL
EXIT /B 0

REM ============================================================================
REM TEST FUNCTIONS
REM ============================================================================

:run_tests
CALL :log_info "Running unit tests..."

IF NOT EXIST ".\src\radarsimcpp\build\Release\radarsimcpp_test.exe" (
    CALL :log_error "Test executable not found"
    EXIT /B 1
)

".\src\radarsimcpp\build\Release\radarsimcpp_test.exe" 2>&1
IF !ERRORLEVEL! NEQ 0 (
    CALL :log_error "Unit tests failed"
    EXIT /B 1
)

CALL :log_info "Unit tests passed successfully."
EXIT /B 0

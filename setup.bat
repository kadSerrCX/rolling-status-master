@echo off
REM Installation script for codec5jchain on Windows 10
REM Optimized for NVIDIA RTX 3050 and AMD Ryzen 5 5600X

echo ================================================================================
echo codec5jchain - Automated Setup for Windows 10
echo Hardware: NVIDIA RTX 3050 + AMD Ryzen 5 5600X
echo ================================================================================
echo.

REM Check if Python is installed
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo Python is installed!
echo.

REM Check if we're in the project directory
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo Please run this script from the codec5jchain directory
    pause
    exit /b 1
)

REM Create virtual environment
echo [2/6] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created!
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)
echo.

REM Install dependencies
echo [5/6] Installing Python dependencies...
echo This may take several minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed!
echo.

REM Install PyTorch with CUDA support
echo [6/6] Installing PyTorch with CUDA support for RTX 3050...
echo This may take several minutes and download ~2GB of data...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo WARNING: PyTorch installation failed
    echo You can install it manually later with:
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo PyTorch with CUDA installed!
)
echo.

REM Run hardware configuration check
echo ================================================================================
echo Running hardware configuration check...
echo ================================================================================
echo.
python configure_hardware.py
echo.

REM Display next steps
echo ================================================================================
echo Installation Complete!
echo ================================================================================
echo.
echo Next steps:
echo   1. Review the configuration summary above
echo   2. Run 'python system_monitor.py' to monitor your system
echo   3. Run 'python benchmark.py' to test performance
echo   4. Read QUICKSTART.md for usage instructions
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate
echo.
echo Press any key to exit...
pause >nul

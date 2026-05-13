@echo off
SETLOCAL EnableDelayedExpansion

:: FluxRT installation script for Windows.
:: Run from the repository root: scripts\install.bat

:: ── sanity-check: running from repo root ──────────────────────────────────────
IF NOT EXIST "pyproject.toml" (
    echo [ERROR] This script must be run from the FluxRT repository root.
    exit /b 1
)

:: ── prerequisites ─────────────────────────────────────────────────────────────
echo [+] Checking prerequisites...

where git >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] 'git' is not installed. Install it from https://git-scm.com/download/win
    exit /b 1
)

where conda >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] 'conda' is not installed. Install Miniconda or Anaconda first.
    exit /b 1
)

git lfs version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] 'git-lfs' is not installed.
    echo         Install with: winget install GitHub.GitLFS
    echo         Or download from https://git-lfs.com
    exit /b 1
)

echo [+] All prerequisites found.

:: ── conda environment ─────────────────────────────────────────────────────────
SET CONDA_ENV=fluxrt

:: Locate conda base and load hooks so 'conda activate' works in this session.
FOR /F "delims=" %%i IN ('conda info --base 2^>nul') DO SET CONDA_BASE=%%i
IF "!CONDA_BASE!"=="" (
    echo [ERROR] Cannot determine conda base directory.
    exit /b 1
)
IF NOT EXIST "!CONDA_BASE!\Scripts\activate.bat" (
    echo [ERROR] Cannot find conda activation script at !CONDA_BASE!\Scripts\activate.bat
    exit /b 1
)
CALL "!CONDA_BASE!\Scripts\activate.bat" "!CONDA_BASE!"

:: Check env by directory — more reliable than parsing 'conda env list'.
IF EXIST "!CONDA_BASE!\envs\%CONDA_ENV%" (
    echo [+] Conda environment '%CONDA_ENV%' already exists.
) ELSE (
    echo [+] Creating conda environment '%CONDA_ENV%' (python=3.12^)...
    cmd /c conda create -n %CONDA_ENV% python=3.12 pip -y
    IF NOT EXIST "!CONDA_BASE!\envs\%CONDA_ENV%" (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)

CALL conda activate %CONDA_ENV%
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate conda environment '%CONDA_ENV%'.
    exit /b 1
)

:: ── PyTorch ───────────────────────────────────────────────────────────────────
python -c "import torch" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing PyTorch with CUDA 12.8 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install PyTorch.
        exit /b 1
    )
) ELSE (
    echo [+] PyTorch is already installed.
)

:: ── Python requirements ───────────────────────────────────────────────────────
:: Use 'diffusers' as a proxy — it's the heaviest transitive dependency.
python -c "import diffusers" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing Python requirements from requirements.txt...
    pip install -r requirements.txt
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install requirements.
        exit /b 1
    )
) ELSE (
    echo [+] Python requirements already installed.
)

:: ── triton-windows ────────────────────────────────────────────────────────────
:: Required for model compilation on Windows (auto-installed but explicit here).
python -c "import triton" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing triton-windows (required for model compilation^)...
    pip install triton-windows
    IF ERRORLEVEL 1 (
        echo [!] Warning: triton-windows installation failed.
        echo [!]          Model compilation may not work. Check compatibility at:
        echo [!]          https://github.com/woct0rdho/triton-windows/issues/158
    )
) ELSE (
    echo [+] triton already installed.
)

:: ── fluxrt package ────────────────────────────────────────────────────────────
python -c "import fluxrt" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing fluxrt package in editable mode...
    pip install -e .
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install fluxrt package.
        exit /b 1
    )
) ELSE (
    echo [+] fluxrt package already installed.
)

:: ── model downloads ───────────────────────────────────────────────────────────
:: Register LFS hooks for the current user (idempotent).
git lfs install

:: ── RIFE frame-interpolation model ───────────────────────────────────────────
SET RIFE_DIR=RIFE-safetensors
SET RIFE_SENTINEL=RIFE-safetensors\flownet.safetensors
IF EXIST "%RIFE_SENTINEL%" (
    echo [+] RIFE frame-interpolation model: already downloaded.
) ELSE IF EXIST "%RIFE_DIR%\.git" (
    echo [!] RIFE: directory exists but looks incomplete — resuming LFS download...
    git -C "%RIFE_DIR%" pull --ff-only
    git -C "%RIFE_DIR%" lfs pull
) ELSE IF EXIST "%RIFE_DIR%" (
    echo [!] RIFE: '%RIFE_DIR%' exists but is not a git repository.
    echo [!]       Remove it and re-run to download the model.
) ELSE (
    echo [+] Downloading RIFE frame-interpolation model...
    git clone https://huggingface.co/TensorForger/RIFE-safetensors
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to clone RIFE model.
        exit /b 1
    )
)

:: ── FLUX.2-klein-4B base model ────────────────────────────────────────────────
SET FLUX_DIR=FLUX.2-klein-4B
SET FLUX_SENTINEL=FLUX.2-klein-4B\transformer\diffusion_pytorch_model.safetensors
IF EXIST "%FLUX_SENTINEL%" (
    echo [+] FLUX.2-klein-4B base model: already downloaded.
) ELSE IF EXIST "%FLUX_DIR%\.git" (
    echo [!] FLUX.2-klein-4B: directory exists but looks incomplete — resuming LFS download...
    git -C "%FLUX_DIR%" pull --ff-only
    git -C "%FLUX_DIR%" lfs pull
) ELSE IF EXIST "%FLUX_DIR%" (
    echo [!] FLUX.2-klein-4B: '%FLUX_DIR%' exists but is not a git repository.
    echo [!]                  Remove it and re-run to download the model.
) ELSE (
    echo [+] Downloading FLUX.2-klein-4B base model...
    git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to clone FLUX.2-klein-4B model.
        exit /b 1
    )
)

:: ── FLUX.2-klein-4B-int8 model ────────────────────────────────────────────────
SET INT8_DIR=FLUX.2-klein-4B-int8
SET INT8_SENTINEL=FLUX.2-klein-4B-int8\diffusion_pytorch_model.safetensors
IF EXIST "%INT8_SENTINEL%" (
    echo [+] FLUX.2-klein-4B int8 model: already downloaded.
) ELSE IF EXIST "%INT8_DIR%\.git" (
    echo [!] FLUX.2-klein-4B-int8: directory exists but looks incomplete — resuming LFS download...
    git -C "%INT8_DIR%" pull --ff-only
    git -C "%INT8_DIR%" lfs pull
) ELSE IF EXIST "%INT8_DIR%" (
    echo [!] FLUX.2-klein-4B-int8: '%INT8_DIR%' exists but is not a git repository.
    echo [!]                       Remove it and re-run to download the model.
) ELSE (
    echo [+] Downloading FLUX.2-klein-4B int8 model...
    git clone https://huggingface.co/aydin99/FLUX.2-klein-4B-int8
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to clone FLUX.2-klein-4B-int8 model.
        exit /b 1
    )
)

:: ── done ──────────────────────────────────────────────────────────────────────
echo.
echo [+] Installation complete.
echo [!] Note: the GUI requires OBS to be installed for virtual webcam output.
echo [!]       Download from https://obsproject.com/download
echo.
echo [+] Activate the environment and start:  conda activate %CONDA_ENV%
echo [+] Then run, for example:               python scripts\run_gradio_demo.py

ENDLOCAL

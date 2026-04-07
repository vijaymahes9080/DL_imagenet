@echo off
title "ORIEN | Neural Hub [EXTREME]"
setlocal enabledelayedexpansion

:: 💎 ORIEN: EXTREME ONE-CLICK LAUNCHER
:: Optimized for high-fidelity real-time interaction
:: Bilingual: English + Tamil Support (Native Bridge)
color 0b
set "ROOT=%~dp0"
cd /d "%ROOT%"
set TF_ENABLE_ONEDNN_OPTS=0
set CUDA_VISIBLE_DEVICES=-1
set TF_CPP_MIN_LOG_LEVEL=3

echo.
echo "  🚀 ORIEN NEURAL SYNERGY ENGINE"
echo "  ══════════════════════════════════════════"
echo "  [STATUS] Initializing Neural Clusters..."

:: 🛠️ STAGE 0: SOCKET PURGE (PowerShell - Silent & Robust)
powershell -NoProfile -Command "Get-Process -Id (Get-NetTCPConnection -LocalPort 8000,8080 -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue" 2>nul

:: 🛠️ STAGE 1: PYTHON HYPER-ENV
set "PY=python"
if exist ".venv_training\Scripts\python.exe" (
    echo "  [ENV] Native Neural .venv Detected"
    set "PY=.venv_training\Scripts\python.exe"
)

:: 🛠️ STAGE 2: SYNERGY AUDIT (Silent)
echo "  [AUDIT] Verifying Neural Shard Integrity..."
!PY! scripts\verify_datasets.py >nul 2>&1

:: 🛠️ STAGE 3: LAUNCH ALL
echo "  [BOOT] Powering up ORIEN Neural Hub..."
echo "  [URL] Access HUD at http://localhost:8080"
echo.
echo "  -------------------------------------------------"
echo "  🌐 Mode: BILINGUAL (English / Tamil)"
echo "  👁️ Sensors: CAMERA + MICROPHONE (Real-time)"
echo "  -------------------------------------------------"
echo.
!PY! "scripts/run_all.py"

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo "  [ERR] Critical Launch Failure. Re-running with sync..."
    call scripts\HARD_DOWNLOAD.bat
    !PY! "scripts/run_all.py"
)

pause

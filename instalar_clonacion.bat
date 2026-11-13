@echo off
title texToVox - Instalador de Clonacion de Voz (OpenVoice v2 + XTTS-v2)
cd /d "%~dp0"

echo ============================================================
echo   texToVox - Configurando entorno dedicado (.venv_clone)
echo ============================================================
echo.

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Se requiere Python 3.12 instalado (py -3.12).
    pause
    exit /b 1
)

if not exist ".venv_clone\Scripts\python.exe" (
    echo Creando entorno virtual Python 3.12...
    py -3.12 -m venv .venv_clone
)

echo Actualizando pip e instalando PyTorch CPU + librerias de clonacion...
.venv_clone\Scripts\python.exe -m pip install --upgrade pip
.venv_clone\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv_clone\Scripts\pip.exe install coqui-tts librosa soundfile scipy huggingface-hub edge-tts six torchcodec

echo.
echo [OK] Entorno .venv_clone listo. Ya puedes abrir texToVox con iniciar_texToVox.bat.
pause


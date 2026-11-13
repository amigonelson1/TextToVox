@echo off
title Compilador de texToVox
cd /d "%~dp0"
echo Instalando / verificando PyInstaller...
pip install pyinstaller
echo.
python build_exe.py
pause


@echo off
setlocal
title TAG-R - Lanzador Rapido

:: --- ELEVACIÓN DE PRIVILEGIOS ---
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Solicitando permisos de administrador...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
cd /d "%~dp0"
:: -------------------------------


echo ================================================================
echo           TAG-R - INICIANDO APLICACION
echo ================================================================
echo.

REM 1. Verificar .venv
if not exist ".venv" (
    echo [INFO] El entorno virtual no existe. Ejecutando instalador...
    call "Instalar_Dependencias.bat"
)

REM 2. Activar y Ejecutar
echo [INFO] Activando entorno...
call .venv\Scripts\activate

echo [INFO] Ejecutando servidor...
python TAG-R_Launcher.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] La aplicacion se detuvo inesperadamente.
    pause
    exit /b 1
)

exit /b 0

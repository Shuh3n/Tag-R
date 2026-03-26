@echo off
setlocal
title TAG-R - Lanzador Rapido

:: --- ELEVACIÓN DE PRIVILEGIOS ---
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Solicitando administrador...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
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
echo [INFO] Activando entorno virtual (.venv)...
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] No se encontro el archivo de activacion en .venv\Scripts\activate.bat
    pause
    exit /b 1
)
call .venv\Scripts\activate

echo [INFO] Ejecutando servidor (python TAG-R_Launcher.py)...
python TAG-R_Launcher.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] La aplicacion se detuvo inesperadamente (Codigo de salida: %errorlevel%).
    echo Revisa si hay errores de Python arriba.
    pause
    exit /b 1
)

echo.
echo [INFO] El servidor se ha cerrado normalmente.
pause
exit /b 0

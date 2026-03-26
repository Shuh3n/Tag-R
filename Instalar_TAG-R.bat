@echo off
setlocal
title TAG-R - Instalador

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
echo           TAG-R - INSTALADOR DE DEPENDENCIAS
echo ================================================================
echo.
echo Este proceso preparara el reconocimiento facial en tu equipo.
echo Puede tardar unos minutos segun tu conexion a internet.
echo.

if not exist "Sistema_TAG-R\Instalar_Dependencias.bat" goto no_installer

echo [INFO] Accediendo a carpeta del sistema...
cd Sistema_TAG-R
echo [INFO] Llamando al instalador interno...
call "Instalar_Dependencias.bat"

if errorlevel 1 (
    echo.
    echo [ERROR] Hubo un problema durante la instalacion.
    pause
    exit /b 1
)
exit /b 0

:no_installer
echo [ERROR] No se encontro el instalador en 'Sistema_TAG-R\Instalar_Dependencias.bat'.
pause
exit /b 1

@echo off
setlocal
title TAG-R - Instalador

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
echo           TAG-R - INSTALADOR DE DEPENDENCIAS
echo ================================================================
echo.
echo Este proceso preparara el reconocimiento facial en tu equipo.
echo Puede tardar unos minutos segun tu conexion a internet.
echo.

if exist "Sistema_TAG-R\Instalar_Dependencias.bat" (
    cd Sistema_TAG-R
    call "Instalar_Dependencias.bat"
    exit /b %errorlevel%
)

echo [ERROR] No se encontro el instalador en 'Sistema_TAG-R\Instalar_Dependencias.bat'.
pause
exit /b 1

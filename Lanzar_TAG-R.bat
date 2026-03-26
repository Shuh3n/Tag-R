@echo off
setlocal
title TAG-R - Acceso Rapido

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
echo           TAG-R - LANZADOR DE APLICACION
echo ================================================================
echo.

REM Verificar si existe el ejecutable compilado
if not exist "Sistema_TAG-R\dist\TAG-R.exe" goto try_script

echo [INFO] Iniciando version optimizada (EXE)...
cd Sistema_TAG-R
"dist\TAG-R.exe"

if errorlevel 1 (
    echo.
    echo [ERROR] La aplicacion (EXE) se ha cerrado con errores.
    pause
    exit /b 1
)
echo [OK] Aplicacion finalizada.
exit /b 0

:try_script
REM Si no hay EXE, intentar lanzar via Python
if not exist "Sistema_TAG-R\Lanzar_Rapido.bat" goto no_system

echo [INFO] Iniciando via Script...
cd Sistema_TAG-R
call "Lanzar_Rapido.bat"

if errorlevel 1 (
    echo.
    echo [ERROR] Hubo un problema al ejecutar el script.
    pause
    exit /b 1
)
exit /b 0

:no_system
echo [ERROR] No se encontro el sistema de TAG-R. 
echo Asegurate de que la carpeta 'Sistema_TAG-R' exista.
pause
exit /b 1

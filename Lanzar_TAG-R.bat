@echo off
setlocal
title TAG-R - Acceso Rapido
cd /d "%~dp0"

echo ================================================================
echo           TAG-R - LANZADOR DE APLICACION
echo ================================================================
echo.

REM Verificar si existe el ejecutable compilado
if exist "Sistema_TAG-R\dist\TAG-R.exe" (
    echo [INFO] Iniciando version optimizada (EXE)...
    start "" "Sistema_TAG-R\dist\TAG-R.exe"
    exit /b 0
)

REM Si no hay EXE, intentar lanzar via Python
if exist "Sistema_TAG-R\Lanzar_Rapido.bat" (
    echo [INFO] Iniciando via Script...
    cd Sistema_TAG-R
    call "Lanzar_Rapido.bat"
    exit /b %errorlevel%
)

echo [ERROR] No se encontro el sistema de TAG-R. 
echo Asegurate de que la carpeta 'Sistema_TAG-R' exista.
pause
exit /b 1

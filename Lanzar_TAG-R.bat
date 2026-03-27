@echo off
setlocal enabledelayedexpansion
title TAG-R - Lanzador

:: =================================================================
:: ELEVACION DE PRIVILEGIOS (metodo robusto con rutas con espacios)
:: =================================================================
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Solicitando permisos de administrador...
    powershell -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/k \"%~s0\"' -Verb RunAs"
    exit /b 0
)
cd /d "%~dp0"
:: =================================================================


cls
echo ================================================================
echo           TAG-R - LANZADOR DE APLICACION
echo ================================================================
echo.

REM --- Verificar que la carpeta Sistema_TAG-R existe ---
if not exist "Sistema_TAG-R" (
    echo [ERROR] No se encontro la carpeta 'Sistema_TAG-R'.
    echo Asegurate de que los archivos esten descomprimidos correctamente.
    goto error_fatal
)

REM --- Opcion 1: EXE compilado (mas rapido) ---
if exist "Sistema_TAG-R\dist\TAG-R.exe" (
    echo [INFO] Iniciando version optimizada (EXE)...
    cd /d "%~dp0Sistema_TAG-R"
    "dist\TAG-R.exe"
    set EXIT_CODE=!errorlevel!
    cd /d "%~dp0"
    if !EXIT_CODE! neq 0 (
        echo.
        echo [ERROR] La aplicacion EXE se cerro con errores (codigo: !EXIT_CODE!).
        goto error_fatal
    )
    echo [OK] Aplicacion finalizada correctamente.
    pause
    exit /b 0
)

REM --- Opcion 2: Via script Python ---
if exist "Sistema_TAG-R\Lanzar_Rapido.bat" (
    echo [INFO] Iniciando via Script Python...
    cd /d "%~dp0Sistema_TAG-R"
    call "Lanzar_Rapido.bat"
    set EXIT_CODE=!errorlevel!
    cd /d "%~dp0"
    if !EXIT_CODE! neq 0 (
        echo [ERROR] El script de lanzamiento fallo.
        goto error_fatal
    )
    exit /b 0
)

REM --- No se encontro ninguna forma de lanzar ---
echo [ERROR] No se encontro TAG-R.exe ni Lanzar_Rapido.bat.
echo Ejecuta primero 'Instalar_TAG-R.bat' para configurar el sistema.

:error_fatal
echo.
echo ================================================================
echo  [FATAL] No se pudo iniciar TAG-R.
echo  Ejecuta 'Instalar_TAG-R.bat' para reparar la instalacion.
echo ================================================================
echo.
pause
exit /b 1

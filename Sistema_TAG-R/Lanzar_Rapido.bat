@echo off
setlocal enabledelayedexpansion
title TAG-R - Iniciando...

:: --- ELEVACION DE PRIVILEGIOS (metodo robusto con rutas con espacios) ---
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Solicitando permisos de administrador...
    powershell -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/k \"%~s0\"' -Verb RunAs"
    exit /b 0
)
cd /d "%~dp0"
:: -----------------------------------------------------------------------


echo ================================================================
echo           TAG-R - INICIANDO APLICACION
echo ================================================================
echo.

REM --- [PASO 1] Verificar que el entorno virtual existe ---
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Entorno virtual no encontrado. Ejecutando instalador...
    echo.
    if not exist "Instalar_Dependencias.bat" (
        echo [ERROR] No se encuentra Instalar_Dependencias.bat en esta carpeta.
        echo Asegurate de que los archivos del sistema esten completos.
        goto error_fatal
    )
    call "Instalar_Dependencias.bat"
    if errorlevel 1 (
        echo [ERROR] La instalacion fallo. No se puede iniciar TAG-R.
        goto error_fatal
    )
    echo.
)

REM --- [PASO 2] Verificar modulo principal ---
if not exist "TAG-R_Launcher.py" (
    echo [ERROR] No se encontro TAG-R_Launcher.py
    echo Verifica que todos los archivos del sistema esten presentes.
    goto error_fatal
)

REM --- [PASO 3] Activar entorno e iniciar ---
title TAG-R - En Ejecucion
echo [INFO] Activando entorno virtual...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] No se pudo activar el entorno virtual.
    echo Intenta ejecutar Instalar_TAG-R.bat para reparar la instalacion.
    goto error_fatal
)

echo [INFO] Iniciando servidor TAG-R...
echo [INFO] La ventana del navegador abrira automaticamente.
echo.
echo [Para cerrar TAG-R, cierra esta ventana o presiona Ctrl+C]
echo.
echo ================================================================
echo.

python TAG-R_Launcher.py

set EXIT_CODE=%errorlevel%

echo.
echo ================================================================

if %EXIT_CODE% neq 0 (
    echo [ERROR] TAG-R se cerro con un error (codigo: %EXIT_CODE%).
    echo.
    echo Posibles causas:
    echo   - Puerto 8000 ocupado por otro proceso
    echo   - Falta un modulo de Python (reinstala con Instalar_TAG-R.bat)
    echo   - Error en la configuracion de la aplicacion
    echo.
    goto error_fatal
)

echo [OK] TAG-R se cerro correctamente.
echo.
pause
exit /b 0

:error_fatal
echo.
echo ================================================================
echo  [FATAL] No se pudo iniciar TAG-R.
echo  Si el problema persiste, vuelve a ejecutar Instalar_TAG-R.bat
echo  o contacta al soporte tecnico.
echo ================================================================
echo.
pause
exit /b 1

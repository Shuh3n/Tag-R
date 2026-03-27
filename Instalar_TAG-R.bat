@echo off
setlocal enabledelayedexpansion
title TAG-R - Instalador de Dependencias

:: =================================================================
:: ELEVACION DE PRIVILEGIOS (metodo robusto con rutas con espacios)
:: =================================================================
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Solicitando permisos de administrador...
    :: Usamos %~s0 (short path 8.3 sin espacios) para evitar problemas con comillas
    powershell -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/k \"%~s0\"' -Verb RunAs"
    exit /b 0
)
cd /d "%~dp0"
:: =================================================================


cls
echo ================================================================
echo           TAG-R - INSTALADOR DE DEPENDENCIAS
echo ================================================================
echo.
echo Este proceso preparara el reconocimiento facial en tu equipo.
echo Puede tardar varios minutos segun tu conexion a internet.
echo.

:: Verificar que existe el instalador interno
if not exist "Sistema_TAG-R\Instalar_Dependencias.bat" (
    echo [ERROR] No se encontro el archivo:
    echo         Sistema_TAG-R\Instalar_Dependencias.bat
    echo.
    echo Asegurate de que todos los archivos esten descomprimidos correctamente.
    echo.
    echo Presiona cualquier tecla para cerrar...
    pause >nul
    exit /b 1
)

echo [INFO] Accediendo a carpeta del sistema...
cd /d "%~dp0Sistema_TAG-R"

echo [INFO] Iniciando instalador interno...
echo.
call "Instalar_Dependencias.bat"
set INSTALL_RESULT=%errorlevel%

:: Volver al directorio raiz
cd /d "%~dp0"

if %INSTALL_RESULT% neq 0 (
    echo.
    echo ================================================================
    echo  [ERROR] La instalacion termino con errores (codigo: %INSTALL_RESULT%).
    echo  Revisa los mensajes de arriba para mas detalles.
    echo ================================================================
    echo.
    echo Presiona cualquier tecla para cerrar...
    pause >nul
    exit /b 1
)

:: Exito - el Instalar_Dependencias.bat ya muestra su propio mensaje final
exit /b 0

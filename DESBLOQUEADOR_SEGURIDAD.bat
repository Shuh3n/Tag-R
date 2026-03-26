@echo off
setlocal
title TAG-R - Desbloqueador de Seguridad

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
echo           DESBLOQUEADOR DE SEGURIDAD (SmartScreen)
echo ================================================================
echo.
echo Se ha detectado que Windows a veces bloquea programas (.exe y .bat) 
echo porque fueron descargados de Internet. Este proceso quitara 
echo esa restriccion para que TAG-R funcione sin advertencias constantes.
echo.
echo Solo presiona una tecla para aplicar el desbloqueo...
pause

powershell -Command "Get-ChildItem -Recurse '%~dp0' | Unblock-File"

echo.
echo [OK] Archivos desbloqueados y listos para usar.
echo.
pause
exit /b 0

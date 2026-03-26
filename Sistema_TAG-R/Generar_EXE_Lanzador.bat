@echo off
setlocal
title TAG-R - Generador de Lanzador EXE

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
echo           TAG-R - GENERADOR DE LANZADOR .EXE
echo ================================================================
echo.

REM 1. Verificar .venv
if not exist ".venv" (
    echo [ERROR] No se encontro el entorno virtual (.venv).
    echo Por favor, ejecuta primero 'Instalar_Dependencias.bat'.
    pause
    exit /b 1
)

REM 2. Instalar PyInstaller en el entorno
echo [1/3] Preparando PyInstaller en el entorno virtual...
call .venv\Scripts\activate
python -m pip install pyinstaller

REM 3. Generar el EXE
echo [2/3] Generando ejecutable (esto puede tardar un poco)...
REM Explicacion de flags:
REM --onefile: Crea un unico archivo .exe
REM --name: Nombre del ejecutable
REM --add-data: Para incluir archivos estaticos si fuera necesario
REM --collect-all: Para asegurar que se incluyan dependencias complejas como insightface y onnxruntime

python -m PyInstaller --noconfirm --onefile --name "TAG-R" ^
    --collect-all onnxruntime ^
    --collect-all insightface ^
    --hidden-import "uvicorn.logging" ^
    --hidden-import "uvicorn.protocols" ^
    --hidden-import "uvicorn.protocols.http" ^
    --hidden-import "uvicorn.protocols.http.auto" ^
    --hidden-import "uvicorn.protocols.websockets" ^
    --hidden-import "uvicorn.protocols.websockets.auto" ^
    --hidden-import "uvicorn.lifespan" ^
    --hidden-import "uvicorn.lifespan.on" ^
    TAG-R_Launcher.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] No se pudo generar el archivo .exe.
    pause
    exit /b 1
)

echo.
echo [3/3] Limpieza de temporales...
if exist "dist\TAG-R.exe" (
    copy /Y "dist\TAG-R.exe" "TAG-R.exe"
    echo [OK] El archivo 'TAG-R.exe' ya esta en la carpeta principal.
)

echo.
echo ================================================================
echo          ¡PROCESO COMPLETADO CON EXITO!
echo ================================================================
echo.
echo Ahora tienes el archivo 'TAG-R.exe' listo para usar.
echo.
pause
exit /b 0

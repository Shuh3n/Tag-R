@echo off
setlocal
title TAG-R - Instalación de Dependencias
cd /d "%~dp0"

echo ================================================================
2: echo           TAG-R - INSTALADOR DE DEPENDENCIAS
3: echo ================================================================
echo.

REM 1. Verificar Python
echo [1/4] Verificando instalacion de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    echo Por favor, descarga Python 3.10 o 3.11 de https://www.python.org/
    echo Asegurate de marcar "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)
echo [OK] Python detectado.
echo.

REM 2. Crear Entorno Virtual
echo [2/4] Creando entorno virtual (.venv)...
if not exist ".venv" (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo [OK] Entorno virtual creado.
) else (
    echo [INFO] El entorno virtual ya existe.
)
echo.

REM 3. Instalar dependencias
echo [3/4] Instalando dependencias (esto puede tardar unos minutos)...
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --prefer-binary

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Hubo un error instalando algunas dependencias.
    echo Intentando instalacion alternativa para InsightFace...
    python -m pip install onnxruntime==1.16.3 insightface==0.7.3 numpy==1.24.3 --prefer-binary
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] No se pudieron instalar las dependencias correctamente.
    echo Por favor, revisa tu conexion a internet o errores arriba.
    pause
    exit /b 1
)
echo [OK] Dependencias instaladas.
echo.

REM 4. Comprobar instalacion
echo [4/4] Verificando instalacion de modulos criticos...
python -c "import fastapi, uvicorn, insightface, cv2, numpy; print('Todo OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Algunos modulos no responden. Revisa el log de arriba.
) else (
    echo [OK] Verificacion final exitosa.
)

echo.
echo ================================================================
echo          INSTALACION COMPLETADA CON EXITO
echo ================================================================
echo.
echo Ahora puedes cerrar esta ventana y usar el lanzador (TAG-R.exe)
echo o el script principal para iniciar la aplicacion.
echo.
pause
exit /b 0

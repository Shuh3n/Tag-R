@echo off
title TAG-R - Organizador de Fotos con IA
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo ================================================================
echo                       TAG-R v1.0
echo                Organizador de Fotos con IA
echo ================================================================
echo.

echo [1/5] Verificando Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no encontrado
    echo.
    echo SOLUCION:
    echo 1. Instala Python desde https://python.org
    echo 2. Durante instalacion marca "Add Python to PATH"
    echo 3. Reinicia el sistema
    echo 4. Vuelve a ejecutar este archivo
    echo.
    pause
    exit /b 1
)

echo [OK] Python encontrado

echo.
echo [2/5] Verificando FastAPI...
python -c "import fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Instalando FastAPI y Uvicorn...
    python -m pip install fastapi "uvicorn[standard]" --no-cache-dir
    if %errorlevel% neq 0 (
        echo [ERROR] No se pudo instalar FastAPI
        echo Intentalo manualmente: pip install fastapi uvicorn[standard]
        pause
        exit /b 1
    )
)

echo.
echo [3/5] Verificando OpenCV y NumPy...
python -c "import cv2, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Instalando OpenCV y NumPy...
    python -m pip install opencv-python-headless numpy --prefer-binary --no-cache-dir
    if %errorlevel% neq 0 (
        echo [ERROR] No se pudo instalar OpenCV/NumPy
        pause
        exit /b 1
    )
)

echo.
echo [4/5] Verificando dependencias ML...
python -c "import sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Instalando Scikit-learn...
    python -m pip install scikit-learn --prefer-binary --no-cache-dir
)

python -c "import multipart" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Instalando Python-multipart...
    python -m pip install python-multipart --no-cache-dir
)

echo.
echo [5/5] Verificando InsightFace...
python -c "import insightface" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Instalando InsightFace (puede tardar 5-10 minutos)...
    echo        IMPORTANTE: No cerrar esta ventana
    python -m pip install insightface --no-cache-dir --prefer-binary
    if %errorlevel% neq 0 (
        echo [WARNING] Error con InsightFace, intentando version alternativa...
        python -m pip install insightface==0.7.3 --no-cache-dir
    )
)

echo.
echo [OK] Todas las dependencias instaladas
echo.
echo ================================================================
echo                    INICIANDO APLICACION
echo ================================================================
echo.
echo IMPORTANTE:
echo - NO CERRAR ESTA VENTANA
echo - Se abrira automaticamente el navegador
echo - API ejecutandose en: http://localhost:8000
echo - Para detener: presiona Ctrl+C aqui
echo.

REM Verificar que main.py existe
if not exist "main.py" (
    echo [ERROR] Archivo main.py no encontrado
    echo Asegurate de que todos los archivos esten en la misma carpeta
    pause
    exit /b 1
)

REM Ejecutar la aplicacion con mejor manejo de errores
echo [INFO] Ejecutando main.py...
python main.py

REM Si llegamos aqui, la aplicacion se cerro
echo.
echo ================================================================
echo                   APLICACION CERRADA
echo ================================================================
echo.

REM Si hay error, mostrar ayuda
if %errorlevel% neq 0 (
    echo [ERROR] La aplicacion termino con errores
    echo.
    echo POSIBLES SOLUCIONES:
    echo 1. Ejecuta este archivo como ADMINISTRADOR
    echo 2. Verifica conexion a Internet
    echo 3. Desactiva temporalmente el antivirus
    echo 4. Reinstala Python desde python.org
    echo.
) else (
    echo Gracias por usar TAG-R!
)

pause

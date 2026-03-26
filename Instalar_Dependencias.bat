@echo off
setlocal
title TAG-R - Instalación de Dependencias
cd /d "%~dp0"

echo ================================================================
echo           TAG-R - INSTALADOR DE DEPENDENCIAS
echo ================================================================
echo.

REM 1. Verificar e instalar Python
echo [1/4] Verificando instalacion de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [AVISO] Python no detectado en el sistema. 
    echo Intentando instalacion automatica de Python 3.11...
    
    winget --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Instalando Python 3.11 via Winget (gestor oficial de Microsoft)...
        echo Espera a que termine, esto puede tardar unos minutos...
        
        winget install --id Python.Python.3.11 --source winget --silent --accept-package-agreements --accept-source-agreements
        
        if %errorlevel% equ 0 (
            echo.
            echo [OK] Python 3.11 se ha instalado satisfactoriamente.
            echo.
            echo ================================================================
            echo [IMPORTANTE] DEBES REINICIAR EL INSTALADOR
            echo Para que Windows reconozca el nuevo comando, cierra esta ventana
            echo y vuelve a abrir 'Instalar_Dependencias.bat'.
            echo ================================================================
            echo.
            pause
            exit /b 0
        ) else (
            echo [ERROR] La instalacion automatica fallo (Codigo: %errorlevel%).
            echo Por favor, instalalo manualmente desde: https://www.python.org/
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] Winget no disponible. Instala Python 3.11 manualmente.
        echo Asegurate de marcar la casilla "Add Python to PATH" al instalar.
        pause
        exit /b 1
    )
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
echo [4/5] Verificando instalacion de modulos criticos...
python -c "import fastapi, uvicorn, insightface, cv2, numpy, PIL, PyInstaller; print('Todo OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Algunos modulos no responden. Revisa el log de arriba.
) else (
    echo [OK] Verificacion exitosa.
)
echo.

REM 5. Generar Lanzador EXE con Icono
echo [5/5] Generando Lanzador EXE personalizado...
if exist "images\logo.png" (
    echo [INFO] Convirtiendo logo a formato de icono (.ico)...
    python -c "from PIL import Image; img = Image.open('images/logo.png'); img.save('logo.ico', format='ICO', sizes=[(256,256)])"
)

echo [INFO] Creando ejecutable (esto puede tardar 1-2 minutos)...
python -m PyInstaller --noconfirm --onefile --name "TAG-R" ^
    --icon="logo.ico" ^
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

if exist "dist\TAG-R.exe" (
    copy /Y "dist\TAG-R.exe" "TAG-R.exe" >nul
    echo [OK] El archivo 'TAG-R.exe' con icono ha sido creado.
)

echo.
echo ================================================================
echo          INSTALACION Y GENERACION COMPLETADAS
echo ================================================================
echo.
echo Ya puedes usar 'TAG-R.exe' para iniciar la aplicacion.
echo.
pause
exit /b 0

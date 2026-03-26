@echo off
setlocal
title TAG-R - Instalación de Dependencias

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
echo           TAG-R - INSTALADOR DE DEPENDENCIAS
echo ================================================================
echo.

REM 1. Verificar e instalar Python
echo [1/4] Verificando instalacion de Python 3.11...

set PYTHON_EXE=
py -3.11 --version >nul 2>&1
if errorlevel 1 goto check_python_path
set PYTHON_EXE=py -3.11
goto python_ok

:check_python_path
python --version 2>&1 | findstr /R /C:"Python 3\.11" >nul
if errorlevel 1 goto need_install
set PYTHON_EXE=python
goto python_ok

:need_install
echo [AVISO] Python 3.11 no detectado en el sistema (o tienes otra version incompatible). 
echo Intentando instalacion automatica de Python 3.11...

winget --version >nul 2>&1
if errorlevel 1 goto winget_fail

echo [INFO] Instalando Python 3.11 via Winget (gestor oficial de Microsoft)...
echo Espera a que termine, esto puede tardar unos minutos...

winget install --id Python.Python.3.11 --source winget --silent --accept-package-agreements --accept-source-agreements

if errorlevel 1 goto winget_install_fail

echo.
echo [OK] Python 3.11 se ha instalado satisfactoriamente.
echo.
echo ================================================================
echo [IMPORTANTE] DEBES REINICIAR EL INSTALADOR
echo Para que Windows reconozca el nuevo comando de Python 3.11, 
echo cierra esta ventana y vuelve a abrir el instalador.
echo ================================================================
echo.
pause
exit /b 0

:winget_install_fail
echo [ERROR] La instalacion automatica fallo.
echo Por favor, instalalo manualmente desde: https://www.python.org/downloads/release/python-3119/
pause
exit /b 1

:winget_fail
echo [ERROR] Winget no disponible. Instala Python 3.11 manualmente.
echo Asegurate de descargar la version 3.11.x y marcar "Add Python to PATH" al instalar.
pause
exit /b 1

:python_ok
echo [OK] Python 3.11 detectado correctamente.

echo.

REM 2. Crear Entorno Virtual
echo [2/4] Creando entorno virtual (.venv)...
if exist ".venv" goto venv_exists

%PYTHON_EXE% -m venv .venv
if errorlevel 1 goto venv_error

echo [OK] Entorno virtual creado.
goto venv_done

:venv_error
echo [ERROR] No se pudo crear el entorno virtual.
pause
exit /b 1

:venv_exists
echo [INFO] El entorno virtual ya existe.

:venv_done
echo.

REM 3. Instalar dependencias
echo [3/4] Instalando dependencias (esto puede tardar unos minutos)...
call .venv\Scripts\activate
python -m pip install --upgrade pip

echo [INFO] Descargando compilacion especial de InsightFace para Windows...
echo Esto prevendra el error de "Building wheel"...
python -m pip install https://github.com/Gourieff/sd-webui-reactor/releases/download/v1.1.2/insightface-0.7.3-cp311-cp311-win_amd64.whl

echo [INFO] Instalando el resto de requerimientos...
python -m pip install -r requirements.txt --prefer-binary

if errorlevel 1 goto install_warning
goto deps_ok

:install_warning
echo.
echo [WARNING] Cierto módulo reportó problemas, forzando instalación secundaria...
python -m pip install onnxruntime==1.16.3 numpy==1.24.3 --prefer-binary

if errorlevel 1 goto install_error
goto deps_ok

:install_error
echo.
echo [ERROR] No se pudieron instalar las dependencias correctamente.
echo Por favor, revisa tu conexion a internet o errores arriba.
pause
exit /b 1

:deps_ok

echo [OK] Dependencias instaladas.
echo.

REM 4. Comprobar instalacion
echo [4/5] Verificando instalacion de modulos criticos...
python -c "import fastapi, uvicorn, insightface, cv2, numpy, PIL, PyInstaller; print('Todo OK')" >nul 2>&1
if errorlevel 1 goto verify_warn

echo [OK] Verificacion exitosa.
goto verify_done

:verify_warn
echo [WARNING] Algunos modulos no responden. Revisa el log de arriba.

:verify_done
echo.

REM 5. Generar Lanzador EXE con Icono
echo [5/5] Generando Lanzador EXE personalizado...

if not exist "images\logo.png" goto skip_icon
echo [INFO] Convirtiendo logo a formato de icono (.ico)...
python -c "from PIL import Image; img = Image.open('images/logo.png'); img.save('logo.ico', format='ICO', sizes=[(256,256)])"

:skip_icon

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

if not exist "dist\TAG-R.exe" goto finish_install
copy /Y "dist\TAG-R.exe" "TAG-R.exe" >nul
echo [OK] El archivo 'TAG-R.exe' con icono ha sido creado.

:finish_install

echo.
echo ================================================================
echo          INSTALACION Y GENERACION COMPLETADAS
echo ================================================================
echo.
echo Ya puedes usar 'TAG-R.exe' para iniciar la aplicacion.
echo.
pause
exit /b 0

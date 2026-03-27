@echo off
setlocal enabledelayedexpansion
title TAG-R - Instalacion de Dependencias

:: --- ELEVACION DE PRIVILEGIOS (robusto con rutas con espacios) ---
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Solicitando permisos de administrador...
    powershell -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/k \"%~s0\"' -Verb RunAs"
    exit /b 0
)
cd /d "%~dp0"
:: -----------------------------------------------------------------


cls
echo ================================================================
echo           TAG-R - INSTALADOR DE DEPENDENCIAS
echo ================================================================
echo.

:: Rutas absolutas al venv (evita conflictos con Python del sistema)
set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"


REM ==================================================================
REM [1/5] VERIFICAR PYTHON 3.11
REM ==================================================================
echo [1/5] Verificando instalacion de Python 3.11...
set "PYTHON_EXE="

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py -3.11"
    goto python_ok
)

python --version 2>&1 | findstr /R "Python 3\.11" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto python_ok
)

echo [AVISO] Python 3.11 no detectado. Intentando instalar con Winget...
winget --version >nul 2>&1
if errorlevel 1 goto no_winget

winget install --id Python.Python.3.11 --source winget --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto winget_fail

echo.
echo [OK] Python 3.11 instalado. DEBES CERRAR ESTA VENTANA Y REINICIAR EL INSTALADOR.
echo.
pause
exit /b 0

:no_winget
echo [ERROR] Winget no disponible. Instala Python 3.11 manualmente:
echo   https://www.python.org/downloads/release/python-31110/
echo   IMPORTANTE: Marca "Add Python to PATH".
pause
exit /b 1

:winget_fail
echo [ERROR] La instalacion automatica de Python fallo.
echo Instala Python 3.11 manualmente desde:
echo   https://www.python.org/downloads/release/python-31110/
pause
exit /b 1

:python_ok
echo [OK] Python 3.11 detectado. (%PYTHON_EXE%)
echo.


REM ==================================================================
REM [2/5] CREAR ENTORNO VIRTUAL
REM ==================================================================
echo [2/5] Creando entorno virtual (.venv)...

if exist "%VENV_PY%" (
    echo [INFO] El entorno virtual ya existe.
    goto venv_done
)

if exist "%VENV_DIR%" (
    echo [INFO] Entorno virtual incompleto detectado. Recreando...
    rmdir /s /q "%VENV_DIR%"
)

%PYTHON_EXE% -m venv "%VENV_DIR%"
if errorlevel 1 goto venv_error
echo [OK] Entorno virtual creado.
goto venv_done

:venv_error
echo [ERROR] No se pudo crear el entorno virtual.
echo Verifica los permisos de escritura en esta carpeta.
pause
exit /b 1

:venv_done
echo.


REM ==================================================================
REM [3/5] INSTALAR DEPENDENCIAS
REM ==================================================================
echo [3/5] Instalando dependencias (puede tardar varios minutos)...
echo.

echo [INFO] Actualizando pip...
"%VENV_PY%" -m pip install --upgrade pip --quiet
echo.

:: Limpiar cache de insightface por si hay una version corrupta previa
echo [INFO] Limpiando cache de InsightFace (evita errores de instalaciones previas)...
"%VENV_PY%" -m pip cache remove insightface >nul 2>&1
echo.

echo [INFO] Instalando InsightFace pre-compilado para Windows (cp311)...
echo [INFO] Fuente: github.com/Gourieff/Assets
"%VENV_PY%" -m pip install "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl" --no-deps
if errorlevel 1 goto insightface_fallback

echo [OK] InsightFace instalado correctamente.
goto insightface_done

:insightface_fallback
echo.
echo [WARNING] No se pudo instalar InsightFace desde GitHub.
echo [INFO] Intentando instalacion alternativa desde PyPI...
"%VENV_PY%" -m pip install insightface==0.7.3 --prefer-binary
if errorlevel 1 goto insightface_error
echo [OK] InsightFace instalado (via PyPI).
goto insightface_done

:insightface_error
echo.
echo [ERROR] No se pudo instalar InsightFace por ninguna via.
echo Revisa tu conexion a internet e intenta de nuevo.
pause
exit /b 1

:insightface_done
echo.

echo [INFO] Instalando el resto de dependencias...
"%VENV_PY%" -m pip install -r requirements.txt --prefer-binary --quiet
if errorlevel 1 goto deps_warning
echo [OK] Dependencias instaladas correctamente.
goto deps_done

:deps_warning
echo [WARNING] Algunos paquetes reportaron advertencias. Instalando modulos criticos...
"%VENV_PY%" -m pip install onnxruntime==1.16.3 numpy==1.24.3 Pillow==10.1.0 --prefer-binary --quiet
if errorlevel 1 goto deps_error
echo [OK] Modulos criticos instalados.
goto deps_done

:deps_error
echo [ERROR] No se pudieron instalar las dependencias minimas.
pause
exit /b 1

:deps_done
echo.


REM ==================================================================
REM [4/5] VERIFICAR MODULOS CRITICOS
REM ==================================================================
echo [4/5] Verificando modulos criticos...

"%VENV_PY%" -c "import fastapi, uvicorn, insightface, cv2, numpy, PIL, PyInstaller; print('[OK] Todos los modulos verificados.')" 2>&1
if errorlevel 1 (
    echo [WARNING] Uno o mas modulos no se verificaron. Revisa los mensajes arriba.
    echo Presiona cualquier tecla para continuar de todas formas...
    pause >nul
) else (
    echo [OK] Verificacion exitosa.
)
echo.


REM ==================================================================
REM [5/5] GENERAR LANZADOR EXE
REM ==================================================================
echo [5/5] Generando Lanzador EXE personalizado...

if exist "images\logo.png" (
    echo [INFO] Convirtiendo logo a formato .ico...
    "%VENV_PY%" -c "from PIL import Image; img = Image.open('images/logo.png'); img.save('logo.ico', format='ICO', sizes=[(256,256)])"
)

if not exist "TAG-R_Launcher.py" (
    echo [ERROR] No se encontro TAG-R_Launcher.py. No se puede generar el EXE.
    goto skip_exe
)

echo [INFO] Creando ejecutable (esto puede tardar 1-2 minutos, por favor espera)...
echo.

if exist "logo.ico" (
    "%VENV_PY%" -m PyInstaller --noconfirm --onefile --name "TAG-R" --icon="logo.ico" --collect-all onnxruntime --collect-all insightface --hidden-import "uvicorn.logging" --hidden-import "uvicorn.protocols" --hidden-import "uvicorn.protocols.http" --hidden-import "uvicorn.protocols.http.auto" --hidden-import "uvicorn.protocols.websockets" --hidden-import "uvicorn.protocols.websockets.auto" --hidden-import "uvicorn.lifespan" --hidden-import "uvicorn.lifespan.on" TAG-R_Launcher.py
) else (
    "%VENV_PY%" -m PyInstaller --noconfirm --onefile --name "TAG-R" --collect-all onnxruntime --collect-all insightface --hidden-import "uvicorn.logging" --hidden-import "uvicorn.protocols" --hidden-import "uvicorn.protocols.http" --hidden-import "uvicorn.protocols.http.auto" --hidden-import "uvicorn.protocols.websockets" --hidden-import "uvicorn.protocols.websockets.auto" --hidden-import "uvicorn.lifespan" --hidden-import "uvicorn.lifespan.on" TAG-R_Launcher.py
)

if errorlevel 1 (
    echo.
    echo [WARNING] No se pudo generar el EXE. Esto no es critico.
    echo Puedes usar 'Lanzar_TAG-R.bat' para iniciar normalmente.
    goto skip_exe
)

if exist "dist\TAG-R.exe" (
    copy /Y "dist\TAG-R.exe" "TAG-R.exe" >nul
    echo [OK] Ejecutable TAG-R.exe generado correctamente.
) else (
    echo [WARNING] El EXE no se encontro tras la compilacion.
)

:skip_exe
echo.
echo ================================================================
echo           INSTALACION COMPLETADA EXITOSAMENTE
echo ================================================================
echo.
echo Cierra esta ventana e inicia TAG-R usando:
echo   - Lanzar_TAG-R.bat (en la carpeta raiz)
echo   - TAG-R.exe (si fue generado)
echo.
pause
exit /b 0

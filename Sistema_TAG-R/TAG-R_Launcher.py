import os
import sys
import time
import threading
import webbrowser
import uvicorn
from pathlib import Path

import socket

def find_free_port(start=8000):
    for port in range(start, start + 10):
        try:
            s = socket.socket()
            s.bind(('127.0.0.1', port))
            s.close()
            return port
        except:
            continue
    return None

# Configuración básica
HOST = "127.0.0.1"
PORT = find_free_port()
if not PORT:
    print("[ERROR] No se encontraron puertos libres.")
    sys.exit(1)

URL = f"http://{HOST}:{PORT}"

def open_browser():
    """Espera un momento a que el servidor esté listo y abre el navegador."""
    time.sleep(3)
    print(f"\n[TAG-R] Abriendo navegador en {URL}...")
    webbrowser.open(URL)

def run_server():
    """Inicia el servidor FastAPI usando Uvicorn."""
    print("="*60)
    print("    TAG-R - ORGANIZADOR DE FOTOS CON IA")
    print("="*60)
    print(f"\n[INFO] Iniciando servidor en {URL}...")
    
    # Importar la app de app.py
    # Se hace el import aquí para asegurar que el entorno esté listo
    try:
        from app import app
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    except Exception as e:
        print(f"\n[ERROR] No se pudo iniciar el servidor: {e}")
        input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    # Iniciar hilo para el navegador
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Iniciar el servidor (bloqueante)
    run_server()

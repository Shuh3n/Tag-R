import sys
import os
import time
import threading
import webbrowser
import subprocess
import socket
from pathlib import Path

def print_safe(msg):
    try:
        print(msg)
        sys.stdout.flush()
    except:
        pass

def check_dependencies():
    print_safe("\n[INFO] Verificando dependencias...")
    
    deps = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
    ]
    
    all_ok = True
    for dep, name in deps:
        try:
            __import__(dep)
            print_safe(f"   ✅ {name}")
        except:
            print_safe(f"   ❌ {name}")
            all_ok = False
    
    # Verificar InsightFace especialmente
    try:
        from insightface.app import FaceAnalysis
        print_safe(f"   ✅ InsightFace")
    except Exception as e:
        print_safe(f"   ❌ InsightFace: {e}")
        print_safe("\n❗ INSTALA INSIGHTFACE:")
        print_safe("   pip install insightface==0.7.3 onnxruntime==1.16.0")
        all_ok = False
    
    return all_ok

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

def main():
    print_safe("="*60)
    print_safe("    TAG-R - ORGANIZADOR CON IA")
    print_safe("="*60)
    
    if not check_dependencies():
        input("\nEnter para salir...")
        return
    
    port = find_free_port()
    if not port:
        print_safe("\n[ERROR] No hay puertos libres")
        input("Enter...")
        return
    
    print_safe(f"\n[INFO] Puerto: {port}")
    print_safe("[INFO] Iniciando servidor...\n")
    
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        def read():
            for line in iter(proc.stdout.readline, ''):
                if line.strip():
                    print_safe(f"[SERVER] {line.strip()}")
        
        threading.Thread(target=read, daemon=True).start()
        time.sleep(3)
        
        url = f"http://127.0.0.1:{port}"
        print_safe(f"\n{'='*60}")
        print_safe(f"✅ SERVIDOR LISTO: {url}")
        print_safe(f"{'='*60}\n")
        
        webbrowser.open(url)
        
        try:
            while proc.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            print_safe("\n[INFO] Deteniendo...")
            proc.terminate()
            proc.wait(5)
            print_safe("[OK] Detenido")
    except Exception as e:
        print_safe(f"\n[ERROR] {e}")

if __name__ == "__main__":
    try:
        main()
    except:
        pass
    input("\nEnter para salir...")
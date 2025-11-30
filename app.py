from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import zipfile
import uuid
import asyncio
import logging
import numpy as np
import cv2
import os
import gc
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tagr")

# App
app = FastAPI(title="TAG-R - Face Organizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Serve UI (place your index.html in ./static/index.html)
BASE = Path(".").resolve()
STATIC_DIR = BASE / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# TEMP dir
TEMP_DIR = BASE / "temp_processing"
TEMP_DIR.mkdir(exist_ok=True)

# Lazy init model
_face_app = None
_initializing = False

def get_face_app():
    """
    Inicializa InsightFace de forma robusta (CPU). Lanza HTTPException(503) con mensaje
    claro si faltan dependencias (ej numpy incompatible).
    """
    global _face_app, _initializing

    if _face_app is not None:
        return _face_app

    if _initializing:
        raise HTTPException(503, "Modelo inicializándose, inténtalo en 30 segundos")

    _initializing = True
    try:
        logger.info("Inicializando InsightFace (lazy)...")

        # Chequear numpy versión (evitar crash por ABI incompatibilidad)
        try:
            import numpy as _np_check
            major = int(_np_check.__version__.split(".")[0])
            if major >= 2:
                msg = (
                    "Incompatibilidad detectada: NumPy >= 2 está instalado.\n"
                    "Algunas dependencias nativas (onnxruntime/insightface) requieren numpy<2.\n"
                    "Solución: crea y activa un venv con Python 3.11.9 y ejecuta:\n"
                    "  pip install -r requirements.txt\n"
                )
                logger.error(msg)
                raise HTTPException(503, msg)
        except Exception:
            logger.debug("No se pudo comprobar versión de numpy (continúo).")

        # Importar onnxruntime e insightface con mensajes claros
        try:
            import onnxruntime  # noqa: F401
        except Exception as e:
            logger.error("Error importando onnxruntime: %s", e)
            raise HTTPException(
                503,
                "Error cargando onnxruntime: {}\nPrueba a instalar versiones compatibles: pip install \"numpy<2\" onnxruntime==1.16.3".format(e)
            )

        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            logger.error("Error importando insightface: %s", e)
            raise HTTPException(
                503,
                "Error importando insightface: {}\nPrueba: pip install insightface==0.7.3".format(e)
            )

        # Instanciar FaceAnalysis intentando firmas distintas
        inst = None
        try:
            inst = FaceAnalysis(allowed_modules=['detection', 'recognition'])
            logger.info("FaceAnalysis(allowed_modules=...) OK")
        except TypeError:
            try:
                inst = FaceAnalysis()
                logger.info("FaceAnalysis() OK")
            except Exception as e:
                logger.error("No se pudo instanciar FaceAnalysis: %s", e)
                raise HTTPException(503, f"No se pudo instanciar FaceAnalysis: {e}")

        # Forzar CPU
        try:
            inst.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("InsightFace preparado en CPU (ctx_id=-1)")
        except Exception as e:
            logger.error("Error en inst.prepare(): %s", e)
            raise HTTPException(503, f"Error preparando modelo: {e}")

        _face_app = inst
        return _face_app

    finally:
        _initializing = False
        gc.collect()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self.active[session_id] = ws
        logger.info(f"WS conectado: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active:
            try:
                del self.active[session_id]
            except KeyError:
                pass
            logger.info(f"WS desconectado: {session_id}")

    async def send(self, session_id: str, payload: dict):
        ws = self.active.get(session_id)
        if not ws:
            return
        try:
            await ws.send_json(payload)
        except Exception as e:
            logger.debug(f"Error enviando WS a {session_id}: {e}")

manager = ConnectionManager()

# Utility functions
def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    max_size = 800
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_face_embedding(img_path: Path):
    face_app = get_face_app()
    img = load_image(img_path)
    if img is None:
        return None
    try:
        faces = face_app.get(img)
        emb = faces[0].embedding if faces else None
        del img
        gc.collect()
        return emb
    except Exception as e:
        logger.error(f"Error procesando {img_path}: {e}")
        return None

def find_best_match(embedding, known_persons: dict, threshold=0.4):
    if not known_persons or embedding is None:
        return None
    best_match = None
    best_similarity = -1
    for name, emb in known_persons.items():
        sim = cosine_similarity([embedding], [emb])[0][0]
        if sim > best_similarity and sim > threshold:
            best_similarity = sim
            best_match = name
    return best_match

async def process_photos(work_dir: Path, threshold: float, session_id: str):
    known_dir = work_dir / "known_faces"
    input_dir = work_dir / "input_photos"
    output_dir = work_dir / "organized"

    if not known_dir.exists():
        raise ValueError("Falta carpeta: known_faces")
    if not input_dir.exists():
        raise ValueError("Falta carpeta: input_photos")

    output_dir.mkdir(exist_ok=True)

    await manager.send(session_id, {"type": "status", "message": "📚 Cargando personas conocidas..."})

    known_persons = {}
    for person_dir in sorted(known_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        embeddings = []
        for photo in person_dir.glob("*"):
            if photo.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                emb = get_face_embedding(photo)
                if emb is not None:
                    embeddings.append(emb)
        if embeddings:
            known_persons[person_dir.name] = np.mean(embeddings, axis=0)
            logger.info(f"✓ {person_dir.name}: {len(embeddings)} fotos")
        gc.collect()

    if not known_persons:
        raise ValueError("No se encontraron rostros en known_faces")

    await manager.send(session_id, {"type": "status", "message": f"✅ {len(known_persons)} personas cargadas"})

    photos = [p for p in input_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    total = len(photos)
    if total == 0:
        raise ValueError("No hay fotos en input_photos")

    await manager.send(session_id, {"type": "total_photos", "total": total})

    stats = defaultdict(int)
    update_interval = max(1, min(10, total // 20))

    for i, photo in enumerate(photos, 1):
        embedding = get_face_embedding(photo)
        if embedding is None:
            folder = "sin_rostros"
        else:
            match = find_best_match(embedding, known_persons, threshold)
            folder = match if match else "desconocidos"

        stats[folder] += 1

        dest_folder = output_dir / folder
        dest_folder.mkdir(parents=True, exist_ok=True)

        dest = dest_folder / photo.name
        counter = 1
        while dest.exists():
            dest = dest_folder / f"{photo.stem}_{counter}{photo.suffix}"
            counter += 1

        shutil.copy2(photo, dest)

        if i % update_interval == 0 or i == total:
            await manager.send(session_id, {"type": "processing", "current": i, "total": total, "progress": int((i/total)*100)})
            await asyncio.sleep(0.05)

        if i % 50 == 0:
            gc.collect()

    return dict(stats)

# Routes
@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>TAG-R</h1><p>index.html no encontrado</p>", status_code=404)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await manager.connect(session_id, ws)
    try:
        while True:
            await asyncio.sleep(10)
            try:
                await ws.send_json({"type": "ping"})
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(session_id)

@app.post("/process")
async def process_endpoint(file: UploadFile = File(...), threshold: float = 0.4, session_id: str | None = None):
    MAX_SIZE = 1024 * 1024 * 1024  # 1GB
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Solo archivos .zip")

    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, "Archivo demasiado grande (máx 1GB)")

    if not session_id:
        session_id = str(uuid.uuid4())

    work_dir = TEMP_DIR / session_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        upload_path = work_dir / file.filename
        upload_path.write_bytes(contents)
        logger.info(f"📥 {file.filename} ({len(contents):,} bytes)")

        await manager.send(session_id, {"type": "status", "message": "📦 Extrayendo archivo..."})
        with zipfile.ZipFile(upload_path, "r") as z:
            z.extractall(work_dir)
        await manager.send(session_id, {"type": "status", "message": "✓ Archivo extraído"})

        if not (work_dir / "known_faces").exists():
            raise HTTPException(400, "Falta carpeta 'known_faces'")
        if not (work_dir / "input_photos").exists():
            raise HTTPException(400, "Falta carpeta 'input_photos'")

        # Inicializar modelo temprano para detectar errores y notificar por WS
        try:
            get_face_app()
        except HTTPException as e:
            try:
                await manager.send(session_id, {"type": "error", "message": str(e.detail if hasattr(e, "detail") else e)})
            except Exception:
                logger.debug("No se pudo enviar mensaje WS (quizá no conectado).")
            raise

        stats = await process_photos(work_dir, threshold, session_id)

        await manager.send(session_id, {"type": "status", "message": "📦 Creando ZIP..."})
        output_zip = work_dir / "organized_photos.zip"
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fp in (work_dir / "organized").rglob("*"):
                if fp.is_file():
                    zipf.write(fp, fp.relative_to(work_dir / "organized"))

        await manager.send(session_id, {"type": "completed", "stats": stats})

        # Cleanup diferido
        async def cleanup():
            await asyncio.sleep(60)
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
                logger.info(f"🗑️ Limpieza: {work_dir}")
            except Exception as e:
                logger.debug(f"Error en cleanup: {e}")
        asyncio.create_task(cleanup())

        return FileResponse(path=output_zip, filename=f"organized_{file.filename.rsplit('.',1)[0]}.zip", media_type="application/zip")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error(traceback.format_exc())
        try:
            await manager.send(session_id, {"type": "error", "message": str(e)})
        except Exception:
            pass
        raise HTTPException(500, f"Error: {e}")
    finally:
        gc.collect()

if __name__ == "__main__":
    # Si quieres ejecutar con `python app.py` en lugar de uvicorn:
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
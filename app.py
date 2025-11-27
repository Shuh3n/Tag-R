from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import zipfile
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
# NO importes insightface aquí
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import traceback
import logging
import asyncio
import json
import os
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Organizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CAMBIO PRINCIPAL: Inicialización lazy del modelo
_face_app = None
_initializing = False

def get_face_app():
    """Inicializa el modelo solo cuando se necesita"""
    global _face_app, _initializing
    
    if _face_app is not None:
        return _face_app
    
    if _initializing:
        raise HTTPException(503, "Modelo inicializándose, inténtalo en 30 segundos")
    
    _initializing = True
    try:
        logger.info("Inicializando modelo de reconocimiento facial (lazy)...")
        from insightface.app import FaceAnalysis
        
        face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=-1, det_size=(512, 512))  # ctx_id=-1 para CPU forzado
        
        _face_app = face_app
        logger.info("Modelo inicializado correctamente")
        return _face_app
        
    except Exception as e:
        logger.error(f"Error inicializando modelo: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(503, f"Error inicializando modelo: {e}")
    finally:
        _initializing = False
        gc.collect()

TEMP_DIR = Path("temp_processing")
TEMP_DIR.mkdir(exist_ok=True)

# Gestor de conexiones WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"✅ WebSocket conectado: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"❌ WebSocket desconectado: {session_id}")

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"Error enviando mensaje: {e}")

manager = ConnectionManager()

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    # Redimensiona si es muy grande
    h, w = img.shape[:2]
    max_size = 800
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_face_embedding(img_path):
    """CAMBIO: Usa get_face_app() lazy"""
    face_app = get_face_app()  # Inicializa solo cuando se necesita
    
    img = load_image(img_path)
    if img is None:
        return None
    
    try:
        faces = face_app.get(img)
        result = faces[0].embedding if faces else None
        # Libera memoria
        del img
        gc.collect()
        return result
    except Exception as e:
        logger.error(f"Error procesando {img_path}: {e}")
        return None

# Asegúrate de que extract_archive solo use zipfile:
def extract_archive(archive_path: Path, extract_to: Path):
    """Extrae archivos ZIP"""
    logger.info(f"Extrayendo {archive_path} a {extract_to}")
    
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info("Extracción completada")

def create_zip(source_dir, output_path):
    logger.info(f"Creando ZIP: {output_path}")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in Path(source_dir).rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(source_dir))
    logger.info("ZIP creado exitosamente")

def find_best_match(embedding, known_persons, threshold=0.4):
    if not known_persons or embedding is None:
        return None
    
    best_match = None
    best_similarity = -1
    
    for name, known_emb in known_persons.items():
        similarity = cosine_similarity([embedding], [known_emb])[0][0]
        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match = name
    
    return best_match

async def process_photos(work_dir, threshold, session_id):
    known_dir = work_dir / "known_faces"
    input_dir = work_dir / "input_photos"
    output_dir = work_dir / "organized"
    
    if not known_dir.exists():
        raise ValueError(f"Falta carpeta: known_faces")
    if not input_dir.exists():
        raise ValueError(f"Falta carpeta: input_photos")
    
    output_dir.mkdir(exist_ok=True)
    
    await manager.send_message(session_id, {
        "type": "status",
        "message": "📚 Cargando personas conocidas..."
    })
    
    known_persons = {}
    if Path(known_dir).exists():
        persons = [d for d in Path(known_dir).iterdir() if d.is_dir()]
        
        for person_dir in persons:
            embeddings = []
            for photo in person_dir.glob("*"):
                if photo.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    emb = get_face_embedding(photo)
                    if emb is not None:
                        embeddings.append(emb)
            
            if embeddings:
                known_persons[person_dir.name] = np.mean(embeddings, axis=0)
            
            # Libera memoria entre personas
            gc.collect()
    
    if not known_persons:
        raise ValueError("No se encontraron personas con rostros válidos en known_faces")
    
    await manager.send_message(session_id, {
        "type": "status",
        "message": f"✅ {len(known_persons)} personas cargadas"
    })
    
    photos = [p for p in input_dir.rglob("*") 
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    
    total = len(photos)
    await manager.send_message(session_id, {
        "type": "total_photos",
        "total": total
    })
    
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
            await manager.send_message(session_id, {
                "type": "processing",
                "current": i,
                "total": total,
                "progress": int((i / total) * 100)
            })
            await asyncio.sleep(0.05)
        
        # Libera memoria cada 50 fotos
        if i % 50 == 0:
            gc.collect()
    
    return stats

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)

@app.post("/process")
async def process_archive(
    file: UploadFile = File(...),
    threshold: float = 0.4,
    session_id: str = None
):
    MAX_SIZE = 1024 * 1024 * 1024  # 1GB
    
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, f"Archivo demasiado grande. Máximo: 1GB")
    
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(400, "Solo se aceptan archivos ZIP")
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    work_dir = TEMP_DIR / session_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        upload_path = work_dir / file.filename
        with open(upload_path, "wb") as buffer:
            buffer.write(contents)
        
        await manager.send_message(session_id, {
            "type": "status",
            "message": "📦 Extrayendo archivo..."
        })
        
        extract_archive(upload_path, work_dir)
        
        if not (work_dir / "known_faces").exists():
            raise HTTPException(400, "Falta carpeta 'known_faces'")
        if not (work_dir / "input_photos").exists():
            raise HTTPException(400, "Falta carpeta 'input_photos'")
        
        stats = await process_photos(work_dir, threshold, session_id)
        
        await manager.send_message(session_id, {
            "type": "status",
            "message": "📦 Creando archivo ZIP..."
        })
        
        output_zip = work_dir / "organized_photos.zip"
        create_zip(work_dir / "organized", output_zip)
        
        await manager.send_message(session_id, {
            "type": "completed",
            "stats": dict(stats)
        })
        
        return FileResponse(
            path=output_zip,
            filename=f"organized_{file.filename.rsplit('.', 1)[0]}.zip",
            media_type="application/zip"
        )
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        await manager.send_message(session_id, {
            "type": "error",
            "message": str(e)
        })
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except:
            pass
        gc.collect()

@app.get("/")
async def root():
    return {"message": "Face Organizer API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
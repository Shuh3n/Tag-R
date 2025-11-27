from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import zipfile
import rarfile
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import traceback
import logging
import asyncio
import json

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

logger.info("Inicializando modelo de reconocimiento facial...")
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
logger.info("Modelo inicializado correctamente")

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
        logger.info(f"📊 Conexiones activas: {list(self.active_connections.keys())}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"❌ WebSocket desconectado: {session_id}")
            logger.info(f"📊 Conexiones activas: {list(self.active_connections.keys())}")

    async def send_message(self, session_id: str, message: dict):
        logger.info(f"📤 Intentando enviar a {session_id}: {message}")
        logger.info(f"📊 Conexiones disponibles: {list(self.active_connections.keys())}")
        
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
                logger.info(f"✅ Mensaje enviado exitosamente")
            except Exception as e:
                logger.error(f"❌ Error enviando mensaje: {e}")
        else:
            logger.warning(f"⚠️  No hay WebSocket para session_id: {session_id}")

manager = ConnectionManager()

def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def get_face_embedding(img_path):
    img = load_image(img_path)
    if img is None:
        return None
    faces = face_app.get(img)
    return faces[0].embedding if faces else None

def extract_archive(file_path, extract_to):
    file_path = Path(file_path)
    logger.info(f"Extrayendo {file_path} a {extract_to}")
    
    if file_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.suffix.lower() in ['.rar', '.cbr']:
        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(extract_to)
    else:
        raise ValueError("Formato no soportado. Usa ZIP o RAR")
    
    logger.info("Extracción completada")

def create_zip(source_dir, output_path):
    logger.info(f"Creando ZIP: {output_path}")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in Path(source_dir).rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(source_dir))
    logger.info("ZIP creado exitosamente")

async def load_known_persons(known_dir, session_id):
    await manager.send_message(session_id, {
        "type": "status",
        "message": "📚 Cargando personas conocidas..."
    })
    
    known = {}
    if not Path(known_dir).exists():
        return known
    
    persons = [d for d in Path(known_dir).iterdir() if d.is_dir()]
    
    for i, person_dir in enumerate(persons, 1):
        await manager.send_message(session_id, {
            "type": "status",
            "message": f"Cargando persona {i}/{len(persons)}: {person_dir.name}"
        })
        
        embeddings = []
        for photo in person_dir.glob("*"):
            if photo.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                emb = get_face_embedding(photo)
                if emb is not None:
                    embeddings.append(emb)
        
        if embeddings:
            known[person_dir.name] = np.mean(embeddings, axis=0)
            await manager.send_message(session_id, {
                "type": "person_loaded",
                "name": person_dir.name,
                "photos": len(embeddings)
            })
    
    return known

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
    
    # Cargar personas
    await manager.send_message(session_id, {
        "type": "status",
        "message": "📚 Cargando personas conocidas..."
    })
    await asyncio.sleep(0.1)  # Dar tiempo para que se envíe
    
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
    
    if not known_persons:
        raise ValueError("No se encontraron personas con rostros válidos en known_faces")
    
    await manager.send_message(session_id, {
        "type": "status",
        "message": f"✅ {len(known_persons)} personas cargadas"
    })
    await asyncio.sleep(0.1)
    
    # Buscar fotos
    photos = [p for p in input_dir.rglob("*") 
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    
    total = len(photos)
    await manager.send_message(session_id, {
        "type": "total_photos",
        "total": total
    })
    await asyncio.sleep(0.1)
    
    stats = defaultdict(int)
    
    # OPTIMIZACIÓN: Enviar actualizaciones cada 10 fotos o cada 5%
    update_interval = max(1, min(10, total // 20))
    last_update_time = asyncio.get_event_loop().time()
    
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
        
        # Enviar actualización y dar tiempo para que se procese
        current_time = asyncio.get_event_loop().time()
        should_update = (
            i % update_interval == 0 or
            i == total or
            (current_time - last_update_time) >= 1.0
        )
        
        if should_update:
            await manager.send_message(session_id, {
                "type": "processing",
                "current": i,
                "total": total,
                "progress": int((i / total) * 100)
            })
            last_update_time = current_time
            # Dar tiempo para que el mensaje se envíe antes de continuar
            await asyncio.sleep(0.05)
    
    return stats

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Mantener conexión viva
    except WebSocketDisconnect:
        manager.disconnect(session_id)

@app.post("/process")
async def process_archive(
    file: UploadFile = File(...),
    threshold: float = 0.4,
    session_id: str = None
):
    # Validar tamaño del archivo (1GB máximo)
    MAX_SIZE = 1024 * 1024 * 1024  # 1GB
    
    # Leer contenido
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, f"Archivo demasiado grande. Máximo: 1GB")
    
    # Si no hay session_id, generar uno nuevo
    if not session_id:
        session_id = str(uuid.uuid4())
    
    work_dir = TEMP_DIR / session_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Nueva petición - Session: {session_id}")
    
    try:
        # Guardar archivo
        upload_path = work_dir / file.filename
        with open(upload_path, "wb") as buffer:
            buffer.write(contents)
        
        logger.info(f"Archivo guardado, enviando mensaje WebSocket a {session_id}")
        
        await manager.send_message(session_id, {
            "type": "status",
            "message": "📦 Extrayendo archivo..."
        })
        
        # Dar tiempo para que el mensaje se envíe
        await asyncio.sleep(0.2)
        
        # Extraer
        extract_archive(upload_path, work_dir)
        
        # Validar estructura
        if not (work_dir / "known_faces").exists():
            raise HTTPException(400, "Falta carpeta 'known_faces'")
        if not (work_dir / "input_photos").exists():
            raise HTTPException(400, "Falta carpeta 'input_photos'")
        
        # Procesar con WebSocket
        stats = await process_photos(work_dir, threshold, session_id)
        
        await manager.send_message(session_id, {
            "type": "status",
            "message": "📦 Creando archivo ZIP..."
        })
        
        await asyncio.sleep(0.2)
        
        # Crear ZIP
        output_zip = work_dir / "organized_photos.zip"
        create_zip(work_dir / "organized", output_zip)
        
        await manager.send_message(session_id, {
            "type": "completed",
            "stats": dict(stats)
        })
        
        await asyncio.sleep(0.1)
        
        return FileResponse(
            path=output_zip,
            filename=f"organized_{file.filename.rsplit('.', 1)[0]}.zip",
            media_type="application/zip",
            headers={
                "X-Session-Id": session_id,
                "X-Process-Stats": str(dict(stats))
            }
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
        # Limpiar después de un tiempo
        # shutil.rmtree(work_dir, ignore_errors=True)
        pass

@app.get("/")
async def root():
    return {
        "message": "Face Organizer API",
        "endpoints": {
            "/process": "POST - Sube ZIP/RAR",
            "/ws/{session_id}": "WebSocket - Progreso en tiempo real",
            "/docs": "Documentación"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configuración
KNOWN_FACES_DIR = "known_faces"
INPUT_DIR = "input_photos"
OUTPUT_DIR = "output_organized"
SIMILARITY_THRESHOLD = 0.4  # menor = más estricto

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def get_face_embedding(app, img_path):
    """Extrae embedding del primer rostro detectado"""
    img = load_image(img_path)
    if img is None:
        return None
    faces = app.get(img)
    return faces[0].embedding if faces else None

def load_known_persons(app):
    """Carga embeddings de personas conocidas"""
    known = {}
    if not os.path.exists(KNOWN_FACES_DIR):
        return known
    
    print("📚 Cargando personas conocidas...")
    for person_dir in Path(KNOWN_FACES_DIR).iterdir():
        if not person_dir.is_dir():
            continue
        
        embeddings = []
        for photo in person_dir.glob("*"):
            if photo.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                emb = get_face_embedding(app, photo)
                if emb is not None:
                    embeddings.append(emb)
        
        if embeddings:
            # Promedio de embeddings para esa persona
            known[person_dir.name] = np.mean(embeddings, axis=0)
            print(f"   ✓ {person_dir.name}: {len(embeddings)} fotos")
    
    return known

def find_best_match(embedding, known_persons):
    """Encuentra la persona más similar"""
    if not known_persons or embedding is None:
        return None
    
    best_match = None
    best_similarity = -1
    
    for name, known_emb in known_persons.items():
        similarity = cosine_similarity([embedding], [known_emb])[0][0]
        if similarity > best_similarity and similarity > SIMILARITY_THRESHOLD:
            best_similarity = similarity
            best_match = name
    
    return best_match

def organize_photos():
    setup_directories()
    
    # Inicializar modelo
    print("🔧 Inicializando modelo de reconocimiento facial...")
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Cargar personas conocidas
    known_persons = load_known_persons(app)
    
    if not known_persons:
        print(f"\n⚠️  No se encontraron personas en '{KNOWN_FACES_DIR}'")
        print(f"   Estructura requerida:")
        print(f"   {KNOWN_FACES_DIR}/")
        print(f"   ├── Juan/")
        print(f"   │   └── foto1.jpg")
        print(f"   └── Maria/")
        print(f"       └── foto1.jpg")
        return
    
    # Buscar fotos
    input_photos = list(Path(INPUT_DIR).rglob("*"))
    input_photos = [p for p in input_photos if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    
    if not input_photos:
        print(f"\n⚠️  No hay fotos en '{INPUT_DIR}'")
        return
    
    print(f"\n📸 Procesando {len(input_photos)} fotos...\n")
    
    stats = defaultdict(int)
    
    for photo in tqdm(input_photos, desc="Organizando"):
        try:
            embedding = get_face_embedding(app, photo)
            
            if embedding is None:
                dest_folder = Path(OUTPUT_DIR) / "sin_rostros"
                stats["sin_rostros"] += 1
            else:
                match = find_best_match(embedding, known_persons)
                if match:
                    dest_folder = Path(OUTPUT_DIR) / match
                    stats[match] += 1
                else:
                    dest_folder = Path(OUTPUT_DIR) / "desconocidos"
                    stats["desconocidos"] += 1
            
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            # Copiar con nombre único si ya existe
            dest_path = dest_folder / photo.name
            counter = 1
            while dest_path.exists():
                dest_path = dest_folder / f"{photo.stem}_{counter}{photo.suffix}"
                counter += 1
            
            shutil.copy2(photo, dest_path)
            
        except Exception as e:
            print(f"\n❌ Error procesando {photo.name}: {e}")
    
    print("\n✅ Proceso completado:")
    for person, count in sorted(stats.items()):
        print(f"   {person}: {count} fotos")
    print(f"\n📁 Resultados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    print("🔍 Organizador de Fotos con Reconocimiento Facial\n")
    organize_photos()
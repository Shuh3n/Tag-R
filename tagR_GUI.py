import os
import shutil
import threading
import cv2
from pathlib import Path
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

class ModernButton(tk.Canvas):
    """Botón moderno con bordes redondeados"""
    def __init__(self, parent, text, command, bg_color="#3498db", fg_color="white", 
                 hover_color="#2980b9", width=200, height=40, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'], 
                        highlightthickness=0, **kwargs)
        
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        self.text = text
        self.width = width
        self.height = height
        
        self.draw_button(bg_color)
        self.bind("<Button-1>", lambda e: self.on_click())
        self.bind("<Enter>", lambda e: self.on_hover())
        self.bind("<Leave>", lambda e: self.on_leave())
        
    def draw_button(self, color):
        self.delete("all")
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                radius=10, fill=color, outline="")
        self.create_text(self.width/2, self.height/2, text=self.text, 
                        fill=self.fg_color, font=("Segoe UI", 10, "bold"))
        
    def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                 x2-radius, y1,
                 x2, y1,
                 x2, y1+radius,
                 x2, y2-radius,
                 x2, y2,
                 x2-radius, y2,
                 x1+radius, y2,
                 x1, y2,
                 x1, y2-radius,
                 x1, y1+radius,
                 x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)
        
    def on_hover(self):
        self.draw_button(self.hover_color)
        self.config(cursor="hand2")
        
    def on_leave(self):
        self.draw_button(self.bg_color)
        self.config(cursor="")
        
    def on_click(self):
        if self.command:
            self.command()

class FaceOrganizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Organizer")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#f5f6fa")
        
        # Variables
        self.known_faces_dir = tk.StringVar(value="known_faces")
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar(value="output_organized")
        self.tolerance = tk.DoubleVar(value=0.4)  # threshold para insightface
        self.is_processing = False
        self.app = None  # FaceAnalysis app
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header con gradiente simulado
        header = tk.Frame(self.root, bg="#667eea", height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        header_inner = tk.Frame(header, bg="#667eea")
        header_inner.place(relx=0.5, rely=0.5, anchor="center")
        
        # Icono y título
        icon = tk.Label(header_inner, text="🎭", font=("Segoe UI", 32), 
                       bg="#667eea", fg="white")
        icon.pack(side=tk.LEFT, padx=(0, 15))
        
        title_frame = tk.Frame(header_inner, bg="#667eea")
        title_frame.pack(side=tk.LEFT)
        
        title = tk.Label(title_frame, text="Face Organizer", 
                        font=("Segoe UI", 24, "bold"), bg="#667eea", fg="white")
        title.pack(anchor="w")
        
        subtitle = tk.Label(title_frame, text="Organiza tus fotos inteligentemente", 
                           font=("Segoe UI", 10), bg="#667eea", fg="#e0e6ff")
        subtitle.pack(anchor="w")
        
        # Contenedor principal con padding
        main = tk.Frame(self.root, bg="#f5f6fa")
        main.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # === Card: Configuración de Carpetas ===
        folders_card = self.create_card(main, "📁  Carpetas")
        folders_card.pack(fill=tk.X, pady=(0, 20))
        
        # Grid para carpetas
        for i, (label_text, var, btn_text) in enumerate([
            ("Personas conocidas", self.known_faces_dir, "Cambiar"),
            ("Fotos a organizar", self.input_dir, "Seleccionar"),
            ("Carpeta de salida", self.output_dir, "Cambiar")
        ]):
            row_frame = tk.Frame(folders_card, bg="white")
            row_frame.pack(fill=tk.X, pady=8)
            
            label = tk.Label(row_frame, text=label_text, 
                           font=("Segoe UI", 9), bg="white", fg="#2c3e50", width=18, anchor="w")
            label.pack(side=tk.LEFT, padx=(0, 10))
            
            entry_frame = tk.Frame(row_frame, bg="#ecf0f1", highlightthickness=1, 
                                  highlightbackground="#dfe4ea", highlightcolor="#667eea")
            entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            
            entry = tk.Entry(entry_frame, textvariable=var, font=("Segoe UI", 9),
                           bg="#ecf0f1", fg="#2c3e50", relief=tk.FLAT, 
                           state="readonly", bd=0)
            entry.pack(fill=tk.BOTH, padx=10, pady=8)
            
            btn = ModernButton(row_frame, f"📂 {btn_text}", 
                             lambda v=var: self.select_folder(v),
                             bg_color="#667eea", hover_color="#5568d3",
                             width=120, height=35)
            btn.pack(side=tk.LEFT)
        
        # === Card: Configuración ===
        config_card = self.create_card(main, "⚙️  Configuración")
        config_card.pack(fill=tk.X, pady=(0, 20))
        
        tolerance_frame = tk.Frame(config_card, bg="white")
        tolerance_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(tolerance_frame, text="Umbral de similitud (similarity threshold)", 
                font=("Segoe UI", 10), bg="white", fg="#2c3e50").pack(anchor="w")
        
        slider_container = tk.Frame(tolerance_frame, bg="white")
        slider_container.pack(fill=tk.X, pady=(8, 0))
        
        # Custom style para el slider
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TScale", 
                       background="#667eea", 
                       troughcolor="#ecf0f1",
                       borderwidth=0,
                       lightcolor="#667eea",
                       darkcolor="#667eea")
        
        tolerance_scale = ttk.Scale(slider_container, from_=0.2, to=0.7, 
                                   style="Custom.Horizontal.TScale",
                                   variable=self.tolerance, orient=tk.HORIZONTAL)
        tolerance_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
        
        self.tolerance_label = tk.Label(slider_container, text="0.40", 
                                       font=("Segoe UI", 10, "bold"), 
                                       bg="#667eea", fg="white", width=5,
                                       relief=tk.FLAT, padx=8, pady=4)
        self.tolerance_label.pack(side=tk.LEFT)
        
        tolerance_scale.config(command=lambda v: self.tolerance_label.config(text=f"{float(v):.2f}"))
        
        hint = tk.Label(tolerance_frame, text="← Más estricto  |  Más permisivo →", 
                       font=("Segoe UI", 8, "italic"), bg="white", fg="#95a5a6")
        hint.pack(anchor="w", pady=(5, 0))
        
        # === Botón Principal ===
        btn_frame = tk.Frame(main, bg="#f5f6fa")
        btn_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.start_button = ModernButton(btn_frame, "🚀  INICIAR ORGANIZACIÓN", 
                                        self.start_processing,
                                        bg_color="#10ac84", hover_color="#0e9470",
                                        width=300, height=50)
        self.start_button.pack(anchor="center")
        
        # === Card: Progreso ===
        progress_card = self.create_card(main, "📊  Progreso")
        progress_card.pack(fill=tk.X, pady=(0, 20))
        
        self.progress_label = tk.Label(progress_card, text="Esperando inicio...", 
                                      font=("Segoe UI", 9), bg="white", 
                                      fg="#7f8c8d", anchor="w")
        self.progress_label.pack(fill=tk.X, pady=(0, 8))
        
        # Progressbar con estilo
        style.configure("Custom.Horizontal.TProgressbar",
                       troughcolor="#ecf0f1",
                       background="#667eea",
                       borderwidth=0,
                       thickness=20)
        
        self.progress_bar = ttk.Progressbar(progress_card, 
                                           style="Custom.Horizontal.TProgressbar",
                                           mode='determinate')
        self.progress_bar.pack(fill=tk.X)
        
        # === Card: Log ===
        log_card = self.create_card(main, "📝  Registro de Actividad")
        log_card.pack(fill=tk.BOTH, expand=True)
        
        log_container = tk.Frame(log_card, bg="#2c3e50", highlightthickness=0)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_container, 
                                                 bg="#2c3e50", fg="#ecf0f1",
                                                 font=("Consolas", 9), 
                                                 relief=tk.FLAT, bd=0,
                                                 padx=10, pady=10,
                                                 insertbackground="#ecf0f1")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def create_card(self, parent, title):
        """Crea una tarjeta con sombra simulada y bordes redondeados"""
        # Sombra (simulada con frame gris desplazado)
        shadow = tk.Frame(parent, bg="#d2d8e0", bd=0)
        shadow.pack(fill=tk.X)
        
        # Card principal
        card = tk.Frame(shadow, bg="white", bd=0)
        card.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Header de la card
        header = tk.Frame(card, bg="white")
        header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        title_label = tk.Label(header, text=title, 
                              font=("Segoe UI", 12, "bold"), 
                              bg="white", fg="#2c3e50")
        title_label.pack(anchor="w")
        
        # Separador
        separator = tk.Frame(card, bg="#ecf0f1", height=1)
        separator.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # Contenido de la card
        content = tk.Frame(card, bg="white")
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        return content
        
    def select_folder(self, var):
        folder = filedialog.askdirectory(title="Seleccionar carpeta")
        if folder:
            var.set(folder)
            self.log(f"📁 Carpeta seleccionada: {folder}")
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("Advertencia", "Ya hay un proceso en ejecución")
            return
        
        if not self.input_dir.get():
            messagebox.showerror("Error", "Selecciona la carpeta de fotos a organizar")
            return
        
        self.is_processing = True
        self.start_button.draw_button("#95a5a6")
        self.log("\n" + "="*60)
        self.log("🚀 Iniciando proceso de organización...")
        
        thread = threading.Thread(target=self.process_photos, daemon=True)
        thread.start()
    
    def load_image(self, path):
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    
    def get_face_embedding(self, img_path):
        """Extrae embedding del primer rostro detectado"""
        img = self.load_image(img_path)
        if img is None:
            return None
        faces = self.app.get(img)
        return faces[0].embedding if faces else None
    
    def process_photos(self):
        try:
            known_dir = self.known_faces_dir.get()
            input_dir = self.input_dir.get()
            output_dir = self.output_dir.get()
            threshold = self.tolerance.get()
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Inicializar modelo insightface
            self.log("\n🔧 Inicializando modelo de reconocimiento facial...")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            self.log("\n📚 Cargando personas conocidas...")
            known_persons = self.load_known_persons(known_dir)
            
            if not known_persons:
                self.log("⚠️  No se encontraron personas conocidas")
                self.log(f"   Verifica que existan subcarpetas en: {known_dir}")
                self.log(f"   Cada subcarpeta debe tener fotos de una persona")
                messagebox.showwarning("Advertencia", 
                                      f"No hay personas en '{known_dir}'\n"
                                      "Crea subcarpetas por persona con fotos de referencia")
                self.finish_processing()
                return
            
            photos = [p for p in Path(input_dir).rglob("*") 
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            
            if not photos:
                self.log("⚠️  No hay fotos en la carpeta seleccionada")
                messagebox.showinfo("Info", "No se encontraron fotos para procesar")
                self.finish_processing()
                return
            
            total = len(photos)
            self.log(f"\n📸 Procesando {total} fotos...")
            self.progress_bar['maximum'] = total
            
            stats = defaultdict(int)
            
            for i, photo in enumerate(photos, 1):
                try:
                    self.progress_bar['value'] = i
                    percent = int((i / total) * 100)
                    self.progress_label.config(
                        text=f"Procesando: {photo.name} ({i}/{total}) - {percent}%"
                    )
                    
                    embedding = self.get_face_embedding(photo)
                    
                    if embedding is None:
                        folder = "sin_rostros"
                        self.log(f"   ❌ {photo.name} - sin rostros detectados")
                    else:
                        match = self.find_best_match(embedding, known_persons, threshold)
                        if match:
                            folder = match
                            self.log(f"   ✅ {photo.name} → {match}")
                        else:
                            folder = "desconocidos"
                            self.log(f"   ❓ {photo.name} - no coincide con nadie")
                    
                    stats[folder] += 1
                    
                    dest_folder = Path(output_dir) / folder
                    dest_folder.mkdir(parents=True, exist_ok=True)
                    
                    dest = dest_folder / photo.name
                    counter = 1
                    while dest.exists():
                        dest = dest_folder / f"{photo.stem}_{counter}{photo.suffix}"
                        counter += 1
                    
                    shutil.copy2(photo, dest)
                    
                except Exception as e:
                    self.log(f"   ❌ Error en {photo.name}: {e}")
            
            self.log("\n" + "="*60)
            self.log("✅ PROCESO COMPLETADO")
            self.log("\n📊 Resumen:")
            for person, count in sorted(stats.items()):
                self.log(f"   • {person}: {count} fotos")
            self.log(f"\n📁 Resultados guardados en: {output_dir}")
            
            messagebox.showinfo("Completado", 
                              f"✅ Se procesaron {total} fotos exitosamente\n\n"
                              f"📁 Resultados en: {output_dir}")
            
        except Exception as e:
            self.log(f"\n❌ ERROR CRÍTICO: {e}")
            messagebox.showerror("Error", f"Error durante el procesamiento:\n{e}")
        
        finally:
            self.finish_processing()
    
    def load_known_persons(self, known_dir):
        known = {}
        if not os.path.exists(known_dir):
            self.log(f"⚠️  La carpeta {known_dir} no existe")
            return known
        
        for person_dir in Path(known_dir).iterdir():
            if not person_dir.is_dir():
                continue
            
            embeddings = []
            for photo in person_dir.glob("*"):
                if photo.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    emb = self.get_face_embedding(photo)
                    if emb is not None:
                        embeddings.append(emb)
            
            if embeddings:
                known[person_dir.name] = np.mean(embeddings, axis=0)
                self.log(f"   ✓ {person_dir.name}: {len(embeddings)} foto(s)")
        
        return known
    
    def find_best_match(self, embedding, known_persons, threshold):
        """Encuentra la persona más similar usando cosine similarity"""
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
    
    def finish_processing(self):
        self.is_processing = False
        self.start_button.draw_button("#10ac84")
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Proceso finalizado ✓")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceOrganizerGUI(root)
    root.mainloop()
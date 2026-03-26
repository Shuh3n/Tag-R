# 🚀 TAG-R - Instrucciones de Uso

¡Bienvenido a **TAG-R**! Esta herramienta utiliza inteligencia artificial avanzada para organizar tus fotos automáticamente mediante reconocimiento facial.

## 🛠️ Paso 1: Instalación (Solo la primera vez)

Antes de usar el programa, debes realizar una configuración inicial automática. Este paso instalará Python si no lo tienes, creará un entorno seguro y generará el lanzador personalizado.

1.  Ejecuta el archivo **`Instalar_Dependencias.bat`**.
2.  Si el script instala Python, te pedirá cerrarlo y volverlo a ejecutar una vez más.
3.  Espera a que termine (puede tardar de 5 a 15 minutos). Al finalizar, verás en la carpeta un archivo llamado **`TAG-R.exe`** con el logo oficial.

## 🏃 Paso 2: Ejecución

Tienes dos formas de iniciar la aplicación:

### Opción A (Recomendada): Ejecutable con Logo
*   Haz doble clic en el archivo **`TAG-R.exe`** que se acaba de generar.
*   Esto abrirá el motor de IA y lanzará automáticamente tu navegador en `http://localhost:8000`.

### Opción B: Lanzador Alternativo
*   Si prefieres usar el script directo, ejecuta **`Lanzar_Rapido.bat`**.

## 📁 Cómo organizar tus fotos

Para que la IA funcione correctamente, debes subir un archivo **ZIP** con la siguiente estructura:

```text
mi_archivo.zip
├── Faces/                  <-- Fotos de referencia
│   ├── Juan/               (2-3 fotos claras de la cara de Juan)
│   └── Maria/              (2-3 fotos claras de la cara de Maria)
└── Photos/                 <-- Fotos que quieres organizar
    ├── Foto_Cumple.jpg
    ├── Vacaciones.png
    └── ...
```

1.  Arrastra el archivo **ZIP** a la zona punteada del navegador.
2.  Ajusta el **Umbral de similitud** (0.40 es el recomendado).
3.  Haz clic en **PROCESAR FOTOS**.
4.  Al finalizar, descarga el archivo ZIP con todas tus fotos organizadas en carpetas por persona.

## ⚠️ Notas Importantes

*   **Primer uso:** La primera vez que proceses fotos, la IA descargará automáticamente los modelos de reconocimiento facial (~500MB). Necesitarás internet para este primer paso.
*   **Privacidad:** Todo el proceso ocurre en **tu computadora**. Ninguna foto se sube a la nube.
*   **Error de NumPy:** Si ves errores raros, asegúrate de haber usado el instalador, ya que este fuerza versiones compatibles de los módulos.

---
*Desarrollado con ❤️ para organizar tus recuerdos.*

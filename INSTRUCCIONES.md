# 🚀 TAG-R - Instrucciones de Uso

¡Bienvenido a **TAG-R**! Esta herramienta utiliza inteligencia artificial avanzada para organizar tus fotos automáticamente mediante reconocimiento facial.

## 🛠️ Paso 1: Instalación (Solo la primera vez)

Antes de usar el programa, debes instalar las dependencias necesarias. Esto configurará un entorno virtual aislado para evitar problemas con otras versiones de Python.

1.  Asegúrate de tener **Python 3.10 o 3.11** instalado.
2.  Ejecuta el archivo **`Instalar_Dependencias.bat`**.
3.  Espera a que termine (puede tardar de 5 a 15 minutos dependiendo de tu conexión a internet).

## 🏃 Paso 2: Ejecución

Tienes varias formas de iniciar la aplicación:

### Opción A (Recomendada): Lanzador Rápido
*   Ejecuta **`Lanzar_Rapido.bat`**. 
*   Esto abrirá una consola con los logs del servidor y, automáticamente, se abrirá tu navegador en `http://localhost:8000`.

### Opción B: Ejecutable (Portable)
*   Si ya generaste el archivo **`TAG-R.exe`** (usando `Generar_EXE_Lanzador.bat`), simplemente haz doble clic en él.

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

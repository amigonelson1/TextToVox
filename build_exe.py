"""
Script de compilación automatizada a ejecutable (.exe) para Windows usando PyInstaller.
Incluye los modelos, configuraciones y datos fonéticos de espeak-ng.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def build():
    print("=== Compilador de texToVox a .EXE (Windows) ===")
    
    # 1. Localizar espeak-ng-data de piper
    import piper
    piper_dir = Path(piper.__file__).parent
    espeak_data = piper_dir / "espeak-ng-data"
    
    if not espeak_data.exists():
        print(f"Advertencia: no se encontró {espeak_data}")
    else:
        print(f"Datos de espeak-ng localizados: {espeak_data}")

    # 2. Comando PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", "texToVox",
        # Añadir modelos y configuraciones
        "--add-data", f"{BASE_DIR / 'models'};models",
        "--add-data", f"{BASE_DIR / 'config'};config",
    ]

    # Añadir espeak-ng-data si existe
    if espeak_data.exists():
        cmd.extend(["--add-data", f"{espeak_data};piper/espeak-ng-data"])

    # Hidden imports necesarios para CustomTkinter, lameenc, onnxruntime y sounddevice
    hidden_imports = [
        "customtkinter",
        "PIL",
        "PIL._imagingtk",
        "PIL.ImageTk",
        "darkdetect",
        "onnxruntime",
        "sounddevice",
        "soundfile",
        "lameenc",
        "edge_tts",
        "piper",
        "scipy",
        "parselmouth",
        "pathvalidate"
    ]
    for hi in hidden_imports:
        cmd.extend(["--hidden-import", hi])

    # Archivo de entrada principal
    cmd.append(str(BASE_DIR / "main.py"))

    print("\nEjecutando PyInstaller...")
    print(" ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode == 0:
        exe_path = BASE_DIR / "dist" / "texToVox" / "texToVox.exe"
        print("\n" + "=" * 50)
        print("¡COMPILACIÓN EXITOSA!")
        print(f"Ejecutable generado en: {exe_path}")
        print("Puedes distribuir la carpeta 'dist/texToVox' como paquete portátil.")
        print("=" * 50)
    else:
        print(f"\nError durante la compilación. Código de salida: {result.returncode}")

if __name__ == "__main__":
    build()


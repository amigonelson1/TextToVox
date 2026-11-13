import sys
from pathlib import Path
import time

# Asegurar que src sea importable
sys.path.insert(0, str(Path(__file__).parent))

from src.core.voice_manager import VoiceCatalog
from src.core.engine import PiperEngine
from src.core.text_processing import normalize_spanish_text, SmartTextChunker
from src.core.audio import AudioStitcher, AudioExporter


def main():
    print("=== TEST DE VERIFICACIÓN FASE 1 & 2: texToVox ===")
    models_dir = Path("models")
    catalog = VoiceCatalog(models_dir)

    voices = catalog.get_all_voices()
    print(f"Voces detectadas en catálogo: {len(voices)}")
    for v in voices:
        print(f" - {v.display_name} | Instalada: {v.is_installed}")

    # Seleccionar la primera voz en español latinoamericano (Claude)
    target_voice = next((v for v in voices if v.key == "es_MX-claude-high"), voices[0])
    print(f"\nVoz seleccionada para prueba: {target_voice.name} ({target_voice.key})")

    if not target_voice.is_installed:
        print(f"Descargando modelo {target_voice.name}...")
        def progress(p, msg):
            print(f"  [{p:5.1f}%] {msg}")
        catalog.download_voice(target_voice, progress_callback=progress)

    # 1. Cargar motor
    print("\n[1/5] Cargando motor Piper...")
    t0 = time.time()
    engine = PiperEngine()
    engine.load_model(target_voice.model_path, target_voice.config_path)
    t_load = time.time() - t0
    print(f"✓ Modelo cargado en {t_load:.2f} s. Sample rate: {engine.get_sample_rate()} Hz")
    print(f"  Locutores disponibles: {engine.get_speakers()}")

    # 2. Normalización de texto
    sample_text = """
    ¡Hola! Bienvenido a texToVox, el generador de voz local y offline.
    El Dr. Gómez viajó a 100 km/h hasta Bogotá para la conferencia.
    ¿Sabías que este sistema funciona completamente sin internet ni suscripciones?
    Pruébalo con audiolibros, locuciones y textos extensos con total libertad.
    """
    print("\n[2/5] Normalizando texto...")
    normalized = normalize_spanish_text(sample_text)
    print("Texto normalizado:\n" + normalized)

    # 3. Fragmentación (Chunking)
    print("\n[3/5] Segmentando texto en fragmentos...")
    chunker = SmartTextChunker(max_chars=200)
    chunks = chunker.split(normalized)
    print(f"✓ Total fragmentos generados: {len(chunks)}")
    for c in chunks:
        print(f"  [{c.index + 1}/{len(chunks)}] (Fin de párrafo: {c.is_paragraph_end}) -> {c.text}")

    # 4. Síntesis
    print("\n[4/5] Sintetizando fragmentos con Piper...")
    t_synth_start = time.time()
    chunk_results = []
    for c in chunks:
        audio_chunks = engine.synthesize(c.text, speed=1.0)
        chunk_results.append((audio_chunks, c.is_paragraph_end))
    t_synth = time.time() - t_synth_start
    print(f"✓ Síntesis completada en {t_synth:.2f} s.")

    # 5. Concatenación y Exportación
    print("\n[5/5] Concatenando audio y exportando...")
    stitcher = AudioStitcher(sentence_pause_ms=250, paragraph_pause_ms=550)
    final_audio = stitcher.stitch(chunk_results, engine.get_sample_rate())
    duration_sec = len(final_audio) / engine.get_sample_rate()
    rtf = t_synth / duration_sec if duration_sec > 0 else 0
    print(f"✓ Duración total del audio: {duration_sec:.2f} s")
    print(f"✓ Factor en tiempo real (RTF): {rtf:.3f} (¡Sintetizó {duration_sec/t_synth:.1f}x más rápido que tiempo real!)")

    # Exportar WAV y MP3
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    wav_path = AudioExporter.export_wav(final_audio, engine.get_sample_rate(), out_dir / "prueba.wav")
    mp3_path = AudioExporter.export_mp3(final_audio, engine.get_sample_rate(), out_dir / "prueba.mp3")

    print(f"✓ Archivo WAV exportado: {wav_path} ({wav_path.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Archivo MP3 exportado: {mp3_path} ({mp3_path.stat().st_size / 1024:.1f} KB)")
    print("\n=== ¡FASE 1 & 2 COMPLETADAS CON ÉXITO TOTAL! ===")


if __name__ == "__main__":
    main()


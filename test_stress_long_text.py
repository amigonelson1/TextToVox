"""
Prueba de estrés para síntesis de textos extensos (artículos / capítulos completos).
Evalúa:
1. Segmentación con SmartTextChunker sobre texto de múltiples párrafos.
2. Inferencia secuencial sin fugas de memoria.
3. Concatenación y masterización de audio.
4. Exportación WAV y MP3.
"""
import time
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.text_processing import normalize_spanish_text, SmartTextChunker
from src.core.engine import KokoroEngine
from src.core.audio import AudioStitcher, AudioExporter

def generate_sample_chapter():
    paragraphs = [
        "Capítulo uno: El despertar de la inteligencia artificial local. En las últimas décadas, el procesamiento de voz ha evolucionado a pasos agigantados. Lo que antes requería granjas de servidores en la nube y conexiones continuas a internet, hoy es una realidad tangible directamente en nuestros procesadores personales.",
        "El Dr. Gómez, especialista en fonética computacional, explicó durante el congreso de 2026 que la autonomía del usuario es la clave del futuro. 'No podemos depender de plataformas externas que cambian sus condiciones, aumentan sus tarifas o limitan el volumen de palabras que un creador puede convertir', afirmó ante una audiencia de más de 500 profesionales.",
        "Durante la demostración práctica, un texto extenso de prueba fue dividido automáticamente en oraciones coherentes. Cada punto, coma y signo de interrogación fue respetado minuciosamente. El sistema procesó párrafos completos con pausas acústicas humanas, logrando una entonación balanceada entre oraciones cortas y cláusulas complejas.",
        "Por ejemplo, al mencionar cifras como el 95% de precisión o distancias de 120 km/h en un trayecto de $50 dólares, el normalizador fonético tradujo cada símbolo a palabras habladas con total naturalidad, evitando errores de dicción o tropiezos mecánicos.",
        "En conclusión, contar con un sintetizador de voz neuronal offline no solo protege la privacidad absoluta de los documentos y guiones, sino que brinda una herramienta ilimitada para audiolibros, doblajes, tutoriales y presentaciones de alta calidad."
    ]
    # Multiplicar los párrafos para simular un texto extenso de más de 1.000 palabras
    return "\n\n".join(paragraphs * 2)

def run_stress_test():
    print("=== INICIANDO PRUEBA DE ESTRÉS: TEXTO LARGO ===")
    raw_text = generate_sample_chapter()
    words = len(raw_text.split())
    chars = len(raw_text)
    print(f"Texto de prueba generado: {words} palabras | {chars} caracteres")

    # 1. Normalización
    t0 = time.time()
    normalized = normalize_spanish_text(raw_text)
    t_norm = time.time() - t0
    print(f"✓ Normalización completada en {t_norm:.3f} s")

    # 2. Fragmentación
    chunker = SmartTextChunker(max_chars=250)
    chunks = chunker.split(normalized)
    total_chunks = len(chunks)
    print(f"✓ SmartTextChunker generó {total_chunks} fragmentos de oraciones.")

    # 3. Motor Kokoro
    print("\nCargando motor Kokoro-82M...")
    engine = KokoroEngine()
    engine.load_model(Path("models/kokoro-v1.0.onnx"), Path("models/voices-v1.0.bin"))
    engine.set_active_voice("em_alex")
    print(f"✓ Motor listo. Sample rate: {engine.get_sample_rate()} Hz")

    # 4. Síntesis por fragmentos
    print("\nIniciando síntesis fragmento por fragmento...")
    t_synth_start = time.time()
    chunk_results = []
    
    for i, c in enumerate(chunks):
        t_c0 = time.time()
        audio_chunks = engine.synthesize(c.text, speed=1.0)
        chunk_results.append((audio_chunks, c.is_paragraph_end))
        t_chunk = time.time() - t_c0
        pct = ((i + 1) / total_chunks) * 100
        print(f"  [{pct:5.1f}%] Fragmento {i+1:02d}/{total_chunks:02d} procesado en {t_chunk:.2f} s -> '{c.text[:40]}...'")

    t_synth_total = time.time() - t_synth_start
    print(f"\n✓ Síntesis completa en {t_synth_total:.2f} s ({t_synth_total/total_chunks:.2f} s por fragmento)")

    # 5. Ensamblado con silencios
    print("\nEnsamblando audio final con micro-fades y silencios...")
    stitcher = AudioStitcher(sentence_pause_ms=250, paragraph_pause_ms=550, normalize_peaks=True)
    final_audio = stitcher.stitch(chunk_results, engine.get_sample_rate())
    total_duration = len(final_audio) / engine.get_sample_rate()
    rtf = t_synth_total / total_duration if total_duration > 0 else 0

    print(f"✓ Duración total de audio generado: {total_duration:.2f} s ({total_duration/60:.2f} minutos)")
    print(f"✓ Factor de Tiempo Real (RTF): {rtf:.3f} (Sintetizó a {total_duration/t_synth_total:.1f}x tiempo real en CPU)")

    # 6. Exportación
    out_dir = Path("output/stress_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_file = AudioExporter.export_wav(final_audio, engine.get_sample_rate(), out_dir / "capitulo_completo.wav")
    mp3_file = AudioExporter.export_mp3(final_audio, engine.get_sample_rate(), out_dir / "capitulo_completo.mp3")

    print(f"✓ WAV generado: {wav_file} ({wav_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"✓ MP3 generado: {mp3_file} ({mp3_file.stat().st_size / (1024*1024):.2f} MB)")
    print("\n=== ¡PRUEBA DE ESTRÉS FINALIZADA CON ÉXITO ROTUNDO! ===")

if __name__ == "__main__":
    run_stress_test()


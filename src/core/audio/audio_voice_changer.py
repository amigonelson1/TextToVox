from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import uuid

import numpy as np
from scipy import signal
import soundfile as sf

from src.core.audio.stitcher import AudioStitcher
from src.core.engine.openvoice_engine import OpenVoiceEngine
from src.core.voice_manager.voice_info import VoiceInfo


@dataclass
class AudioFileInfo:
    """Metadatos básicos de un archivo de audio cargado para conversión vocal."""
    path: Path
    filename: str
    duration_sec: float
    sample_rate: int
    channels: int


def probe_audio_file(audio_path: Path) -> AudioFileInfo:
    """Inspecciona duración, frecuencia de muestreo y canales sin cargar todo en RAM si no hace falta."""
    src = Path(audio_path)
    if not src.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de audio: {src}")

    try:
        info = sf.info(str(src))
        duration = float(info.duration)
        sr = int(info.samplerate)
        channels = int(info.channels)
    except Exception:
        data, sr = sf.read(str(src), dtype="float32", always_2d=False)
        channels = 1 if data.ndim == 1 else int(data.shape[1])
        duration = float(len(data) / max(sr, 1))

    if duration <= 0.1 or sr <= 0:
        raise ValueError("El archivo de audio está vacío o es demasiado corto.")

    return AudioFileInfo(
        path=src.resolve(),
        filename=src.name,
        duration_sec=duration,
        sample_rate=sr,
        channels=channels,
    )


def load_mono_resampled(
    audio_path: Path,
    target_sr: int = 22050,
    max_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Carga WAV/MP3/OGG/FLAC a mono float32, recorta opcionalmente y remuestrea con scipy."""
    src = Path(audio_path)
    if not src.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de audio: {src}")

    if max_seconds is not None and max_seconds > 0:
        info = sf.info(str(src))
        max_orig_frames = int(max_seconds * info.samplerate)
        audio, orig_sr = sf.read(
            str(src),
            frames=max_orig_frames if max_orig_frames > 0 else -1,
            dtype="float32",
            always_2d=False,
        )
    else:
        audio, orig_sr = sf.read(str(src), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("El audio cargado no contiene muestras válidas.")

    if int(orig_sr) != int(target_sr):
        frac = Fraction(int(target_sr), int(orig_sr)).limit_denominator(500)
        audio = signal.resample_poly(audio, frac.numerator, frac.denominator).astype(np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        raise ValueError("El archivo no contiene señal de voz audible.")
    if peak > 0.92 or peak < 0.20:
        audio = (audio / peak) * 0.85

    return audio.astype(np.float32), int(target_sr)


def segment_audio_by_pauses(
    audio: np.ndarray,
    sr: int,
    max_chunk_sec: float = 12.0,
) -> List[Tuple[bool, np.ndarray]]:
    """
    Divide el audio en bloques cronológicos de voz activa (is_speech=True)
    y silencios originales (is_speech=False), preservando ritmo y pausas naturales
    y limitando cada fragmento hablado a <= max_chunk_sec para uso estable de RAM.
    """
    n_samples = len(audio)
    if n_samples == 0:
        return []

    win = max(int(0.025 * sr), 64)
    hop = max(int(0.010 * sr), 32)
    n_frames = max(1, 1 + max(0, n_samples - win) // hop)

    frame_rms = np.empty(n_frames, dtype=np.float32)
    for idx in range(n_frames):
        s = idx * hop
        frame = audio[s:min(s + win, n_samples)]
        frame_rms[idx] = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0

    peak = float(np.max(np.abs(audio)))
    thresh = max(peak * (10.0 ** (-34.0 / 20.0)), 0.004)
    active_frames = frame_rms >= thresh

    if not np.any(active_frames):
        raise ValueError("No se detectó voz audible en el audio cargado.")

    # Convertir frames activos en intervalos de muestras con padding anti-recorte
    pad_samples = int(0.08 * sr)
    merge_gap_samples = int(0.22 * sr)

    raw_intervals: List[List[int]] = []
    in_seg = False
    seg_start = 0
    for idx, is_act in enumerate(active_frames):
        if is_act and not in_seg:
            seg_start = max(0, idx * hop - pad_samples)
            in_seg = True
        elif not is_act and in_seg:
            seg_end = min(n_samples, idx * hop + win + pad_samples)
            raw_intervals.append([seg_start, seg_end])
            in_seg = False
    if in_seg:
        raw_intervals.append([seg_start, n_samples])

    merged: List[List[int]] = []
    for start, end in raw_intervals:
        if merged and start - merged[-1][1] <= merge_gap_samples:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # Subdividir tramos continuos muy largos por el valle de menor energía cercano a max_chunk_sec
    max_samples = int(max_chunk_sec * sr)
    min_cut_samples = int(max(3.0, max_chunk_sec * 0.65) * sr)
    final_speech_intervals: List[Tuple[int, int]] = []

    for start, end in merged:
        cur = start
        while end - cur > max_samples:
            search_lo = cur + min_cut_samples
            search_hi = min(end, cur + max_samples)
            lo_f = max(0, search_lo // hop)
            hi_f = min(n_frames - 1, max(lo_f + 1, search_hi // hop))
            best_f = lo_f + int(np.argmin(frame_rms[lo_f:hi_f + 1]))
            split_at = min(end, max(search_lo, best_f * hop))
            final_speech_intervals.append((cur, split_at))
            cur = split_at
        if end > cur:
            final_speech_intervals.append((cur, end))

    timeline: List[Tuple[bool, np.ndarray]] = []
    cursor = 0
    for s, e in final_speech_intervals:
        if s > cursor:
            timeline.append((False, np.zeros(s - cursor, dtype=np.float32)))
        timeline.append((True, audio[s:e]))
        cursor = e

    if cursor < n_samples:
        timeline.append((False, np.zeros(n_samples - cursor, dtype=np.float32)))

    return timeline


class AudioVoiceChanger:
    """
    Conversor de voz a voz (Speech-to-Speech) basado en OpenVoice v2 ToneColorConverter.
    Toma cualquier grabación del usuario (.wav, .mp3, .ogg, .flac) y devuelve la misma
    locución transformada al timbre de cualquier voz del catálogo o perfil clonado.
    """

    @classmethod
    def convert_file(
        cls,
        source_audio: Path,
        openvoice_engine: OpenVoiceEngine,
        openvoice_model_dir: Path,
        target_voice: Optional[VoiceInfo] = None,
        target_profile_audio: Optional[Path] = None,
        tau: float = 0.45,
        max_seconds: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[np.ndarray, int]:
        def _progress(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        if cancel_check and cancel_check():
            raise InterruptedError("Conversión cancelada por el usuario.")

        _progress(0.06, "Cargando motor neuronal OpenVoice v2...")
        if not openvoice_engine.is_loaded():
            openvoice_engine.load_model(openvoice_model_dir)

        target_sr = openvoice_engine.get_sample_rate() or 22050

        _progress(0.14, "Analizando audio de entrada...")
        mono_audio, sr = load_mono_resampled(
            source_audio,
            target_sr=target_sr,
            max_seconds=max_seconds,
        )

        # Extraer embedding de la voz origen (usa caché por huella del archivo)
        _progress(0.20, "Extrayendo timbre de tu audio de entrada...")
        src_se_path = openvoice_engine.ensure_audio_embedding(
            source_audio,
            prefix="vc_src",
            min_duration_sec=0.4,
        )

        if cancel_check and cancel_check():
            raise InterruptedError("Conversión cancelada por el usuario.")

        # Extraer o cargar embedding de la voz destino
        _progress(0.28, "Preparando firma vocal de la voz destino...")
        if target_profile_audio is not None:
            tgt_se_path = openvoice_engine.ensure_audio_embedding(
                target_profile_audio,
                prefix="ref",
                min_duration_sec=2.0,
            )
        elif target_voice is not None:
            tgt_se_path = openvoice_engine.ensure_catalog_voice_embedding(target_voice)
        else:
            raise ValueError("Debes seleccionar una voz destino válida.")

        timeline = segment_audio_by_pauses(mono_audio, sr, max_chunk_sec=12.0)
        speech_indices = [idx for idx, (is_sp, _) in enumerate(timeline) if is_sp]
        if not speech_indices:
            raise ValueError("No se encontraron segmentos de voz activos en el audio.")

        stitcher = AudioStitcher(normalize_peaks=True)
        out_parts: List[np.ndarray] = []
        total_speech = len(speech_indices)
        done_speech = 0

        for is_speech, seg_f32 in timeline:
            if cancel_check and cancel_check():
                raise InterruptedError("Conversión cancelada por el usuario.")

            if not is_speech:
                out_parts.append(np.zeros(len(seg_f32), dtype=np.int16))
                continue

            done_speech += 1
            pct = 0.30 + ((done_speech - 1) / total_speech) * 0.56
            _progress(pct, f"Transformando voz (bloque {done_speech} de {total_speech})...")

            token = uuid.uuid4().hex[:10]
            seg_wav = openvoice_engine.tmp_dir / f"vc_seg_{token}.wav"
            try:
                seg_i16 = np.clip(seg_f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
                sf.write(str(seg_wav), seg_i16, sr, subtype="PCM_16")
                converted_i16, conv_sr = openvoice_engine.convert_wav_segment(
                    src_wav=seg_wav,
                    src_se_path=src_se_path,
                    tgt_se_path=tgt_se_path,
                    tau=tau,
                )
                sr = conv_sr
                fade_len = min(110, max(12, len(converted_i16) // 8))
                out_parts.append(stitcher.apply_anti_click(converted_i16, fade_samples=fade_len))
            finally:
                try:
                    if seg_wav.exists():
                        seg_wav.unlink()
                except Exception:
                    pass

        combined = np.concatenate(out_parts) if out_parts else np.array([], dtype=np.int16)
        normalized = stitcher.normalize_audio_peak(combined, target_db=-1.0)
        return normalized, sr


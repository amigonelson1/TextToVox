import numpy as np
from typing import List
from src.core.engine.base import AudioChunkData


class AudioStitcher:
    """
    Concatena múltiples fragmentos de audio PCM insertando silencios acústicos
    entre oraciones y párrafos para una entonación natural, con micro-desvanecimiento
    anti-clic y masterización de pico a -1.0 dBFS.
    """

    def __init__(
        self,
        sentence_pause_ms: int = 200,
        paragraph_pause_ms: int = 500,
        normalize_peaks: bool = True
    ):
        self.sentence_pause_ms = sentence_pause_ms
        self.paragraph_pause_ms = paragraph_pause_ms
        self.normalize_peaks = normalize_peaks

    def create_silence(self, duration_ms: int, sample_rate: int) -> np.ndarray:
        """Crea un array de silencio int16 para la duración especificada."""
        num_samples = int((duration_ms / 1000.0) * sample_rate)
        return np.zeros(num_samples, dtype=np.int16)

    def apply_anti_click(self, arr: np.ndarray, fade_samples: int = 120) -> np.ndarray:
        """Aplica un micro-desvanecimiento al inicio y final para evitar chasquidos (pops/clicks)."""
        if len(arr) < fade_samples * 2:
            return arr
        result = arr.copy()
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        result[:fade_samples] = (result[:fade_samples].astype(np.float32) * fade_in).astype(np.int16)
        result[-fade_samples:] = (result[-fade_samples:].astype(np.float32) * fade_out).astype(np.int16)
        return result

    def normalize_audio_peak(self, arr: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        """Normaliza el pico de amplitud para que el audio suene con volumen óptimo sin distorsión."""
        if len(arr) == 0:
            return arr
        max_val = np.max(np.abs(arr))
        if max_val == 0:
            return arr
        target_amplitude = int(32767.0 * (10.0 ** (target_db / 20.0)))
        gain = target_amplitude / max_val
        gain = min(gain, 4.0)  # Límite de ganancia para evitar amplificar ruido de fondo
        normalized = np.clip(arr.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
        return normalized

    def stitch(
        self,
        chunk_results: List[tuple[List[AudioChunkData], bool]],
        sample_rate: int
    ) -> np.ndarray:
        """
        Concatena fragmentos de audio con silencios y suavizado acústico.
        chunk_results es una lista de tuplas: (lista de AudioChunkData, is_paragraph_end)
        """
        if not chunk_results:
            return np.array([], dtype=np.int16)

        audio_parts: List[np.ndarray] = []

        sentence_silence = self.create_silence(self.sentence_pause_ms, sample_rate)
        paragraph_silence = self.create_silence(self.paragraph_pause_ms, sample_rate)

        total_items = len(chunk_results)

        for i, (audio_chunks, is_paragraph_end) in enumerate(chunk_results):
            for chunk in audio_chunks:
                if chunk.audio_bytes:
                    arr = np.frombuffer(chunk.audio_bytes, dtype=np.int16)
                    arr_smooth = self.apply_anti_click(arr)
                    audio_parts.append(arr_smooth)

            # Insertar pausa si no es el último bloque
            if i < total_items - 1:
                if is_paragraph_end:
                    audio_parts.append(paragraph_silence)
                else:
                    audio_parts.append(sentence_silence)

        if not audio_parts:
            return np.array([], dtype=np.int16)

        concatenated = np.concatenate(audio_parts)

        if self.normalize_peaks:
            return self.normalize_audio_peak(concatenated, target_db=-1.0)

        return concatenated

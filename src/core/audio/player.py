import threading
from typing import Optional, Callable
import numpy as np
import sounddevice as sd


class AudioPlayer:
    """
    Reproductor de audio integrado no bloqueante basado en sounddevice.
    Permite reproducir arrays de audio PCM (int16), pausar, reanudar,
    detener, adelantar/rebobinar (seek) y monitorear la posición actual en segundos.
    """

    def __init__(self):
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 22050
        self._current_frame: int = 0
        self._is_playing: bool = False
        self._is_paused: bool = False
        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()
        self.on_finished: Optional[Callable[[], None]] = None

    def load(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Carga un nuevo buffer de audio para reproducir."""
        self.stop()
        with self._lock:
            if audio_data.dtype != np.int16:
                self._audio_data = (audio_data * 32767).astype(np.int16)
            else:
                self._audio_data = audio_data.copy()
            self._sample_rate = sample_rate
            self._current_frame = 0

    def play(self) -> None:
        """Inicia la reproducción desde el inicio o la posición actual."""
        with self._lock:
            if self._audio_data is None or len(self._audio_data) == 0:
                return

            if self._is_paused and self._stream and not self._stream.closed:
                self._is_paused = False
                self._is_playing = True
                self._stream.start()
                return

            self._stop_stream_locked()

            if self._current_frame >= len(self._audio_data):
                self._current_frame = 0

            self._is_playing = True
            self._is_paused = False

            def _callback(outdata, frames, time_info, status):
                with self._lock:
                    if not self._is_playing or self._is_paused or self._audio_data is None:
                        outdata.fill(0)
                        return

                    chunk_end = self._current_frame + frames
                    if chunk_end <= len(self._audio_data):
                        chunk = self._audio_data[self._current_frame:chunk_end]
                        outdata[:len(chunk), 0] = chunk
                        self._current_frame = chunk_end
                    else:
                        available = len(self._audio_data) - self._current_frame
                        if available > 0:
                            outdata[:available, 0] = self._audio_data[self._current_frame:]
                            outdata[available:, 0] = 0
                        else:
                            outdata.fill(0)
                        self._current_frame = len(self._audio_data)
                        self._is_playing = False
                        # Notificar fin en hilo separado
                        if self.on_finished:
                            threading.Thread(target=self.on_finished, daemon=True).start()
                        raise sd.CallbackStop

            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                callback=_callback
            )
            self._stream.start()

    def pause(self) -> None:
        """Pausa la reproducción manteniendo la posición actual."""
        with self._lock:
            if self._is_playing and not self._is_paused:
                self._is_paused = True
                self._is_playing = False
                if self._stream and not self._stream.closed:
                    self._stream.stop()

    def resume(self) -> None:
        """Reanuda la reproducción si estaba pausada."""
        self.play()

    def stop(self) -> None:
        """Detiene la reproducción y reinicia la posición al inicio."""
        with self._lock:
            self._stop_stream_locked()
            self._current_frame = 0
            self._is_playing = False
            self._is_paused = False

    def seek(self, position_seconds: float) -> None:
        """Cambia la posición de reproducción a los segundos indicados."""
        with self._lock:
            if self._audio_data is None:
                return
            target_frame = int(position_seconds * self._sample_rate)
            self._current_frame = max(0, min(target_frame, len(self._audio_data)))

    def get_position(self) -> float:
        """Retorna la posición actual en segundos."""
        with self._lock:
            if self._sample_rate > 0:
                return self._current_frame / self._sample_rate
            return 0.0

    def get_duration(self) -> float:
        """Retorna la duración total en segundos del audio cargado."""
        with self._lock:
            if self._audio_data is not None and self._sample_rate > 0:
                return len(self._audio_data) / self._sample_rate
            return 0.0

    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing

    def is_paused(self) -> bool:
        with self._lock:
            return self._is_paused

    def _stop_stream_locked(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


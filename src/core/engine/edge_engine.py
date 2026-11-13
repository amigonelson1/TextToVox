import asyncio
import io
from pathlib import Path
from typing import Optional, Dict, List
import soundfile as sf
import edge_tts

from src.core.engine.base import BaseTTSEngine, AudioChunkData


class EdgeTTSEngine(BaseTTSEngine):
    """
    Motor de síntesis neural de alta fidelidad basado en Edge-TTS.
    Ofrece voces nativas con dicción elegante humana (Colombianas, Mexicanas y Argentinas)
    sin suscripción, sin claves API ni límites de créditos.
    """

    def __init__(self, voice_name: str = "es-CO-GonzaloNeural"):
        self.voice_name = voice_name
        self._sample_rate: int = 24000
        self._is_ready: bool = True

    def load_model(self, model_path: Path, config_path: Optional[Path] = None) -> None:
        """En Edge-TTS, model_path representa el identificador de voz (ej. es-CO-GonzaloNeural)."""
        if isinstance(model_path, (str, Path)):
            self.voice_name = str(model_path).replace(".onnx", "").replace(".bin", "")
        self._is_ready = True

    def set_voice(self, voice_name: str) -> None:
        self.voice_name = voice_name

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        speaker_id: Optional[int] = None
    ) -> List[AudioChunkData]:
        """Sintetiza texto a audio PCM de 16 bits a 24 kHz."""
        text_clean = text.strip()
        if not text_clean:
            return []

        # Formatear velocidad para Edge-TTS: ej. speed=1.2 -> "+20%", speed=0.8 -> "-20%"
        pct_change = int((speed - 1.0) * 100)
        rate_str = f"{'+' if pct_change >= 0 else ''}{pct_change}%"

        async def _run():
            communicate = edge_tts.Communicate(text_clean, self.voice_name, rate=rate_str)
            mp3_buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.extend(chunk["data"])
            return mp3_buffer

        # Ejecutar corrutina de forma síncrona/segura
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Si estamos dentro de un hilo con event loop activo
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    mp3_bytes = executor.submit(asyncio.run, _run()).result()
            else:
                mp3_bytes = loop.run_until_complete(_run())
        except RuntimeError:
            mp3_bytes = asyncio.run(_run())

        if not mp3_bytes:
            return []

        # Decodificar MP3 en memoria a array PCM int16
        audio_data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="int16")
        self._sample_rate = sr

        return [
            AudioChunkData(
                audio_bytes=audio_data.tobytes(),
                sample_rate=sr,
                sample_width=2,
                channels=1
            )
        ]

    def is_loaded(self) -> bool:
        return self._is_ready

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_speakers(self) -> Dict[int, str]:
        return {0: self.voice_name}


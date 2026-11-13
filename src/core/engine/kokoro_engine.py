from pathlib import Path
from typing import Optional, Dict, List, Union
import logging
import numpy as np

try:
    from kokoro_onnx import Kokoro
except ImportError:
    Kokoro = None

from src.core.engine.base import BaseTTSEngine, AudioChunkData

logger = logging.getLogger(__name__)

# Catálogo ampliado de estilos de voz en español para Kokoro-82M
KOKORO_VOICE_DEFINITIONS: Dict[str, dict] = {
    # Voces estándar
    "em_alex": {"name": "Alex (Masculino - Profesional / Corporativo)", "code": "em_alex"},
    "ef_dora": {"name": "Dora (Femenina - Expresiva / Tutoriales)", "code": "ef_dora"},
    "af_bella": {"name": "Bella (Femenina - Suave / Juvenil)", "code": "af_bella"},
    "af_sarah": {"name": "Sarah (Femenina - Cálida / Narrativa)", "code": "af_sarah"},
    "af_nicole": {"name": "Nicole (Femenina - Clara / Comercial)", "code": "af_nicole"},
    "am_adam": {"name": "Adam (Masculino - Barítono / Documental)", "code": "am_adam"},
    "am_michael": {"name": "Michael (Masculino - Dinámico / Moderno)", "code": "am_michael"},
    "em_santa": {"name": "Santa (Masculino - Tono Grave / Relato)", "code": "em_santa"},
    # Voces híbridas (Blended)
    "blend_masc": {
        "name": "Híbrido Masculino (Alex + Adam - Equilibrado)",
        "blend": [("em_alex", 0.6), ("am_adam", 0.4)]
    },
    "blend_fem": {
        "name": "Híbrido Femenino (Dora + Bella - Natural y Cálido)",
        "blend": [("ef_dora", 0.6), ("af_bella", 0.4)]
    }
}


class KokoroEngine(BaseTTSEngine):
    """
    Implementación del motor TTS neural Kokoro-82M (StyleTTS 2 / iSTFT vía ONNX).
    Soporta múltiples timbres vocales masculinos y femeninos, y mezclas acústicas (voice blending).
    """

    def __init__(self):
        self._kokoro: Optional[Kokoro] = None
        self._model_path: Optional[Path] = None
        self._voices_path: Optional[Path] = None
        self._sample_rate: int = 24000
        self._active_voice_key: str = "em_alex"
        self._cached_blends: Dict[str, np.ndarray] = {}

    def load_model(self, model_path: Path, config_path: Optional[Path] = None) -> None:
        """Carga el modelo ONNX de Kokoro y su archivo de pesos vocales (.bin)."""
        if Kokoro is None:
            raise RuntimeError(
                "La librería 'kokoro-onnx' no está instalada. "
                "Por favor instálala ejecutando: pip install kokoro-onnx"
            )
        model_p = Path(model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"No se encontró el modelo Kokoro: {model_p}")

        if config_path is None:
            candidate = model_p.parent / "voices-v1.0.bin"
            if not candidate.exists():
                candidate = model_p.parent / "voices.bin"
            if not candidate.exists():
                raise FileNotFoundError(f"No se encontró el archivo de voces en {model_p.parent}")
            voices_p = candidate
        else:
            voices_p = Path(config_path)

        logger.info("Cargando motor Kokoro-82M: %s con voces: %s", model_p.name, voices_p.name)
        self._kokoro = Kokoro(model_path=str(model_p), voices_path=str(voices_p))
        self._model_path = model_p
        self._voices_path = voices_p
        self._cached_blends.clear()

    def set_active_voice(self, voice_key: str) -> None:
        """Configura la voz activa a sintetizar."""
        if voice_key in KOKORO_VOICE_DEFINITIONS:
            self._active_voice_key = voice_key

    def _get_voice_vector(self, voice_key: str) -> Union[str, np.ndarray]:
        """Obtiene el identificador de voz o el vector interpolado en caso de mezcla."""
        defn = KOKORO_VOICE_DEFINITIONS.get(voice_key, KOKORO_VOICE_DEFINITIONS["em_alex"])
        if "code" in defn:
            return defn["code"]

        # Es una mezcla de voces
        if voice_key in self._cached_blends:
            return self._cached_blends[voice_key]

        blends = defn.get("blend", [])
        total_vec = None
        for code, weight in blends:
            vec = self._kokoro.get_voice_style(code)
            weighted = vec * weight
            if total_vec is None:
                total_vec = weighted
            else:
                total_vec += weighted

        if total_vec is not None:
            self._cached_blends[voice_key] = total_vec.astype(np.float32)
            return self._cached_blends[voice_key]

        return "em_alex"

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        speaker_id: Optional[int] = None
    ) -> List[AudioChunkData]:
        """
        Sintetiza un fragmento de texto a audio con entonación natural a 24 kHz.
        """
        if not self.is_loaded() or self._kokoro is None:
            raise RuntimeError("No hay ningún modelo Kokoro cargado en el motor.")

        text_clean = text.strip()
        if not text_clean:
            return []

        # Determinar voz o mezcla
        voice_target = self._get_voice_vector(self._active_voice_key)

        speed = max(0.5, min(speed, 2.0))

        audio_float, sr = self._kokoro.create(
            text=text_clean,
            voice=voice_target,
            speed=speed,
            lang="es",
            trim=True
        )

        self._sample_rate = sr
        audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)

        return [
            AudioChunkData(
                audio_bytes=audio_int16.tobytes(),
                sample_rate=sr,
                sample_width=2,
                channels=1
            )
        ]

    def is_loaded(self) -> bool:
        return self._kokoro is not None

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_speakers(self) -> Dict[int, str]:
        keys = list(KOKORO_VOICE_DEFINITIONS.keys())
        return {idx: KOKORO_VOICE_DEFINITIONS[k]["name"] for idx, k in enumerate(keys)}

from pathlib import Path
from typing import Optional, Dict, List
import logging

from piper.voice import PiperVoice
from piper.config import SynthesisConfig

from src.core.engine.base import BaseTTSEngine, AudioChunkData

logger = logging.getLogger(__name__)


class PiperEngine(BaseTTSEngine):
    """Implementación del motor TTS usando Piper (VITS / ONNX Runtime)."""

    def __init__(self):
        self._voice: Optional[PiperVoice] = None
        self._model_path: Optional[Path] = None
        self._config_path: Optional[Path] = None

    def load_model(self, model_path: Path, config_path: Optional[Path] = None) -> None:
        """
        Carga un modelo de Piper en formato ONNX.
        Si config_path es None, busca model_path.with_suffix('.onnx.json') o model_path.name + '.json'.
        """
        model_p = Path(model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_p}")

        if config_path is None:
            # Buscar archivo de configuración json
            candidate1 = model_p.with_suffix(".onnx.json")
            candidate2 = Path(str(model_p) + ".json")
            candidate3 = model_p.with_suffix(".json")
            if candidate1.exists():
                cfg_p = candidate1
            elif candidate2.exists():
                cfg_p = candidate2
            elif candidate3.exists():
                cfg_p = candidate3
            else:
                raise FileNotFoundError(
                    f"No se encontró el archivo de configuración para el modelo: {model_p}"
                )
        else:
            cfg_p = Path(config_path)
            if not cfg_p.exists():
                raise FileNotFoundError(f"No se encontró la configuración: {cfg_p}")

        logger.info("Cargando modelo Piper: %s con config: %s", model_p.name, cfg_p.name)
        self._voice = PiperVoice.load(model_path=str(model_p), config_path=str(cfg_p))
        self._model_path = model_p
        self._config_path = cfg_p

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        speaker_id: Optional[int] = None
    ) -> List[AudioChunkData]:
        """
        Sintetiza un texto y devuelve una lista de AudioChunkData (PCM de 16 bits).
        """
        if not self.is_loaded() or self._voice is None:
            raise RuntimeError("No hay ningún modelo de voz cargado en el motor.")

        text_clean = text.strip()
        if not text_clean:
            return []

        # En Piper, length_scale es inversamente proporcional a la velocidad
        # speed = 1.0 -> length_scale = 1.0
        # speed = 1.25 -> length_scale = 0.8 (habla más rápido, dura menos)
        speed = max(0.2, min(speed, 3.0))
        length_scale = 1.0 / speed

        syn_config = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=length_scale
        )

        results: List[AudioChunkData] = []
        for chunk in self._voice.synthesize(text=text_clean, syn_config=syn_config):
            audio_bytes = chunk.audio_int16_bytes
            results.append(
                AudioChunkData(
                    audio_bytes=audio_bytes,
                    sample_rate=chunk.sample_rate,
                    sample_width=chunk.sample_width,
                    channels=chunk.sample_channels
                )
            )

        return results

    def is_loaded(self) -> bool:
        return self._voice is not None

    def get_sample_rate(self) -> int:
        if self._voice and self._voice.config:
            return self._voice.config.sample_rate
        return 22050

    def get_speakers(self) -> Dict[int, str]:
        if not self._voice or not self._voice.config:
            return {0: "Predeterminado"}

        cfg = self._voice.config
        if cfg.speaker_id_map:
            # Map {id: name}
            return {idx: name for name, idx in cfg.speaker_id_map.items()}

        if cfg.num_speakers and cfg.num_speakers > 1:
            return {i: f"Voz {i + 1}" for i in range(cfg.num_speakers)}

        return {0: "Predeterminado"}


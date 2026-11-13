from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List


@dataclass
class AudioChunkData:
    """Representa un fragmento de audio PCM sintetizado."""
    audio_bytes: bytes
    sample_rate: int
    sample_width: int = 2  # 16-bit PCM por defecto
    channels: int = 1       # Mono por defecto


class BaseTTSEngine(ABC):
    """Interfaz abstracta para motores locales de Text-to-Speech."""

    @abstractmethod
    def load_model(self, model_path: Path, config_path: Optional[Path] = None) -> None:
        """Carga los pesos y la configuración del modelo de voz."""
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        speaker_id: Optional[int] = None
    ) -> List[AudioChunkData]:
        """
        Sintetiza un fragmento de texto a fragmentos de audio PCM.
        
        Args:
            text: Texto a sintetizar.
            speed: Factor de velocidad (1.0 = normal, 1.2 = 20% más rápido, 0.8 = 20% más lento).
            speaker_id: ID del locutor si el modelo es multilocutor.
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Retorna True si hay un modelo cargado y listo para inferencia."""
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Retorna la frecuencia de muestreo del modelo cargado."""
        pass

    @abstractmethod
    def get_speakers(self) -> Dict[int, str]:
        """Retorna diccionario de locutores disponibles {id: nombre}."""
        pass


from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict


@dataclass
class VoiceInfo:
    """Información y metadatos de una voz TTS."""
    key: str
    name: str
    language: str
    language_name: str
    gender: str  # "Masculino", "Femenino", "Mixto"
    quality: str  # "Estudio (24 kHz)", "Alta (22 kHz)", etc.
    engine_type: str = "edge"  # "edge", "piper", "openvoice", "xtts"
    voice_identifier: str = ""  # ID para edge-tts (ej. es-CO-GonzaloNeural) o archivo piper
    is_installed: bool = True
    model_path: Optional[Path] = None
    config_path: Optional[Path] = None
    download_model_url: Optional[str] = None
    download_config_url: Optional[str] = None
    speakers: Dict[int, str] = field(default_factory=lambda: {0: "Predeterminado"})
    sample_rate: int = 24000
    description: str = ""

    @property
    def display_name(self) -> str:
        """Nombre amigable para mostrar en el selector de la UI."""
        if self.engine_type == "edge":
            badge = "💎 Elegante"
        elif self.engine_type == "piper":
            badge = "📁 100% Offline"
        elif self.engine_type == "openvoice":
            badge = "🧬 Clon LatAm"
        elif self.engine_type == "xtts":
            badge = "🧬 Clon Offline"
        else:
            badge = "🧪 Experimental"
        return f"[{badge}] {self.name} ({self.gender} - {self.language_name})"

import json
from dataclasses import dataclass, asdict
from pathlib import Path

CONFIG_PATH = Path("config/app_config.json")


@dataclass
class AppConfig:
    output_dir: str = "output"
    last_voice_key: str = "edge-es-CO-GonzaloNeural"
    speaker_id: int = 0
    speed: float = 1.0
    sentence_pause_ms: int = 250
    paragraph_pause_ms: int = 550
    export_format: str = "Ambos (WAV + MP3)"  # "WAV", "MP3", "Ambos (WAV + MP3)"
    theme_mode: str = "dark"                  # "dark", "light", "system"
    color_theme: str = "blue"
    cloned_profile_id: str = ""
    cloned_base_voice_key: str = "edge-es-CO-GonzaloNeural"
    openvoice_tau: float = 0.35
    voice_change_tau: float = 0.45
    synth_preset: str = "Natural (Sin cambios)"
    synth_pitch_semitones: float = 0.0
    synth_formant_ratio: float = 1.0
    synth_pitch_range: float = 1.0
    synth_warmth_db: float = 0.0
    synth_clarity_db: float = 0.0
    live_mode: str = "dsp"                     # "dsp" | "neural"
    live_input_device: str = ""
    live_output_device: str = ""
    live_monitor_device: str = ""
    live_monitor_enabled: bool = False
    live_monitor_volume: float = 0.70
    live_input_gain_db: float = 0.0
    live_noise_gate_db: float = -45.0
    live_tau: float = 0.45

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=4, ensure_ascii=False)
        except Exception:
            pass


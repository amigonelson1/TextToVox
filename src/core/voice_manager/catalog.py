import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
import urllib.request

from src.core.voice_manager.voice_info import VoiceInfo

logger = logging.getLogger(__name__)

HF_PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Catálogo centrado en ESPAÑOL NATIVO, NATURAL Y ELEGANTE (Sin acentos extranjeros)
DEFAULT_CATALOG: Dict[str, dict] = {
    # === VOCES CLONADAS (Tu Voz o Audio de Referencia) ===
    "clone-openvoice-v2": {
        "name": "Clon Híbrido LatAm (OpenVoice v2)",
        "language": "es_CO",
        "language_name": "Español Latino + Tu Timbre",
        "gender": "Personalizable",
        "quality": "Alta (22 kHz / Híbrido CPU)",
        "engine_type": "openvoice",
        "voice_identifier": "openvoice-v2",
        "sample_rate": 22050,
        "cloning_subdir": "openvoice_v2",
        "files": {
            "config.json": "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/config.json",
            "checkpoint.pth": "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/checkpoint.pth",
        },
        "description": "Conserva 100% el acento colombiano/latino y la prosodia de una voz base y transfiere el timbre de tu muestra (6-15 s). Ideal y rápido en CPU."
    },
    "clone-xtts-v2": {
        "name": "Clon 100% Offline (Coqui XTTS-v2)",
        "language": "es",
        "language_name": "Español Zero-Shot Offline",
        "gender": "Personalizable",
        "quality": "Estudio (24 kHz / Zero-Shot)",
        "engine_type": "xtts",
        "voice_identifier": "xtts-v2",
        "sample_rate": 24000,
        "cloning_subdir": "xtts_v2",
        "files": {
            "config.json": "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json",
            "vocab.json": "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
            "speakers_xtts.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/speakers_xtts.pth",
            "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
            "model.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
        },
        "description": "Clona timbre y entonación directamente sin internet desde una muestra de 6-15 s (uso personal CPML). Más lento en CPU sin GPU NVIDIA."
    },

    # === VOCES NEURALES ELEGANTES NATIVAS (Colombia, México y Argentina) ===
    "edge-es-CO-GonzaloNeural": {
        "name": "Gonzalo (Colombia)",
        "language": "es_CO",
        "language_name": "Español (Colombia)",
        "gender": "Masculino",
        "quality": "Estudio (24 kHz / Neural Nativo)",
        "engine_type": "edge",
        "voice_identifier": "es-CO-GonzaloNeural",
        "description": "Locutor colombiano con dicción impecable, tono cálido, elegante y natural. Excelente para presentaciones corporativas y narraciones."
    },
    "edge-es-CO-SalomeNeural": {
        "name": "Salomé (Colombia)",
        "language": "es_CO",
        "language_name": "Español (Colombia)",
        "gender": "Femenino",
        "quality": "Estudio (24 kHz / Neural Nativo)",
        "engine_type": "edge",
        "voice_identifier": "es-CO-SalomeNeural",
        "description": "Locutora colombiana con entonación humana dulce y profesional. Ideal para cursos, audiolibros y videos explicativos."
    },
    "edge-es-MX-DaliaNeural": {
        "name": "Dalia (México / Neutro)",
        "language": "es_MX",
        "language_name": "Español (México / LatAm)",
        "gender": "Femenino",
        "quality": "Estudio (24 kHz / Neural Nativo)",
        "engine_type": "edge",
        "voice_identifier": "es-MX-DaliaNeural",
        "description": "Voz femenina neutra latinoamericana, muy clara, equilibrada y sin modismos marcados."
    },
    "edge-es-MX-JorgeNeural": {
        "name": "Jorge (México / Neutro)",
        "language": "es_MX",
        "language_name": "Español (México / LatAm)",
        "gender": "Masculino",
        "quality": "Estudio (24 kHz / Neural Nativo)",
        "engine_type": "edge",
        "voice_identifier": "es-MX-JorgeNeural",
        "description": "Locutor masculino mexicano con tono firme y periodístico para tutoriales y documentales."
    },
    "edge-es-AR-TomasNeural": {
        "name": "Tomás (Argentina)",
        "language": "es_AR",
        "language_name": "Español (Argentina)",
        "gender": "Masculino",
        "quality": "Estudio (24 kHz / Neural Nativo)",
        "engine_type": "edge",
        "voice_identifier": "es-AR-TomasNeural",
        "description": "Locutor argentino nativo con cadencia rioplatense auténtica y fluida."
    },
    "edge-es-AR-ElenaNeural": {
        "name": "Elena (Argentina)",
        "language": "es_AR",
        "language_name": "Español (Argentina)",
        "gender": "Femenino",
        "quality": "Estudio (24 kHz / Neural Nativo)",
        "engine_type": "edge",
        "voice_identifier": "es-AR-ElenaNeural",
        "description": "Locutora argentina nativa con acento rioplatense natural y expresivo."
    },
    "edge-es-ES-AlvaroNeural": {
        "name": "Álvaro (España)",
        "language": "es_ES",
        "language_name": "Español (España)",
        "gender": "Masculino",
        "quality": "Estudio (24 kHz / Neural Nativo)",
        "engine_type": "edge",
        "voice_identifier": "es-ES-AlvaroNeural",
        "description": "Locutor castellano con dicción peninsular natural y elegante."
    },
    "edge-es-ES-ElviraNeural": {
        "name": "Elvira (España)",
        "language": "es_ES",
        "language_name": "Español (España)",
        "gender": "Femenino",
        "quality": "Estudio (24 kHz / Neural Nativo)",
        "engine_type": "edge",
        "voice_identifier": "es-ES-ElviraNeural",
        "description": "Locutora castellana con tono suave y formal."
    },

    # === VOCES 100% OFFLINE EN ESPAÑOL NATIVO (Piper TTS) ===
    "es_MX-claude-high": {
        "name": "Claude (México / Offline)",
        "language": "es_MX",
        "language_name": "Español (México / LatAm)",
        "gender": "Masculino",
        "quality": "Alta (22 kHz / VITS)",
        "engine_type": "piper",
        "voice_identifier": "es_MX-claude-high",
        "model_file": "es_MX-claude-high.onnx",
        "config_file": "es_MX-claude-high.onnx.json",
        "model_url": f"{HF_PIPER_BASE_URL}/es/es_MX/claude/high/es_MX-claude-high.onnx",
        "config_url": f"{HF_PIPER_BASE_URL}/es/es_MX/claude/high/es_MX-claude-high.onnx.json",
        "description": "Voz masculina mexicana nativa ejecutada 100% en local sin internet."
    },
    "es_AR-daniela-high": {
        "name": "Daniela (Argentina / Offline)",
        "language": "es_AR",
        "language_name": "Español (Argentina)",
        "gender": "Femenino",
        "quality": "Alta (22 kHz / VITS)",
        "engine_type": "piper",
        "voice_identifier": "es_AR-daniela-high",
        "model_file": "es_AR-daniela-high.onnx",
        "config_file": "es_AR-daniela-high.onnx.json",
        "model_url": f"{HF_PIPER_BASE_URL}/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
        "config_url": f"{HF_PIPER_BASE_URL}/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
        "description": "Voz femenina argentina nativa ejecutada 100% en local sin internet."
    },
    "es_MX-ald-medium": {
        "name": "Aldo (México / Offline)",
        "language": "es_MX",
        "language_name": "Español (México / LatAm)",
        "gender": "Masculino",
        "quality": "Media (22 kHz / VITS)",
        "engine_type": "piper",
        "voice_identifier": "es_MX-ald-medium",
        "model_file": "es_MX-ald-medium.onnx",
        "config_file": "es_MX-ald-medium.onnx.json",
        "model_url": f"{HF_PIPER_BASE_URL}/es/es_MX/ald/medium/es_MX-ald-medium.onnx",
        "config_url": f"{HF_PIPER_BASE_URL}/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json",
        "description": "Voz masculina mexicana en VITS ligera."
    }
}


class VoiceCatalog:
    """Administra el catálogo de voces nativas y modelos locales."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_all_voices(self) -> List[VoiceInfo]:
        """Retorna la lista completa de voces con prioridad para español nativo elegante."""
        voices_list: List[VoiceInfo] = []

        for key, entry in DEFAULT_CATALOG.items():
            engine_type = entry.get("engine_type", "edge")

            if engine_type == "edge":
                is_inst = True
                model_p = None
                config_p = None
            elif engine_type in ("openvoice", "xtts"):
                clone_dir = self.models_dir / "cloning" / entry.get("cloning_subdir", key)
                required_files = list(entry.get("files", {}).keys())
                is_inst = bool(required_files) and all((clone_dir / fname).is_file() for fname in required_files)
                model_p = clone_dir
                config_p = clone_dir / "config.json" if (clone_dir / "config.json").is_file() else None
            else:
                model_file = self.models_dir / entry.get("model_file", f"{key}.onnx")
                config_file = self.models_dir / entry.get("config_file", f"{key}.onnx.json")
                is_inst = model_file.exists() and config_file.exists()
                model_p = model_file if is_inst else None
                config_p = config_file if is_inst else None

            v_info = VoiceInfo(
                key=key,
                name=entry["name"],
                language=entry["language"],
                language_name=entry["language_name"],
                gender=entry["gender"],
                quality=entry["quality"],
                engine_type=engine_type,
                voice_identifier=entry.get("voice_identifier", key),
                is_installed=is_inst,
                model_path=model_p,
                config_path=config_p,
                download_model_url=entry.get("model_url"),
                download_config_url=entry.get("config_url"),
                sample_rate=entry.get("sample_rate", 24000 if engine_type == "edge" else 22050),
                description=entry.get("description", "")
            )
            voices_list.append(v_info)

        # Ordenar con prioridad: Colombia (Gonzalo, Salomé) -> México -> Argentina -> España -> Clonación -> Offline
        priority_order = {
            "edge-es-CO-GonzaloNeural": 1,
            "edge-es-CO-SalomeNeural": 2,
            "edge-es-MX-DaliaNeural": 3,
            "edge-es-MX-JorgeNeural": 4,
            "edge-es-AR-TomasNeural": 5,
            "edge-es-AR-ElenaNeural": 6,
            "edge-es-ES-AlvaroNeural": 7,
            "edge-es-ES-ElviraNeural": 8,
            "clone-openvoice-v2": 9,
            "clone-xtts-v2": 10,
            "es_MX-claude-high": 11,
            "es_AR-daniela-high": 12,
            "es_MX-ald-medium": 13,
        }
        voices_list.sort(key=lambda v: priority_order.get(v.key, 99))
        return voices_list

    def download_voice(
        self,
        voice_info: VoiceInfo,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> bool:
        if voice_info.engine_type == "edge":
            return True

        entry = DEFAULT_CATALOG.get(voice_info.key, {})

        # Descarga multi-archivo para motores de clonación (OpenVoice v2 / XTTS-v2)
        if voice_info.engine_type in ("openvoice", "xtts"):
            files_map: Dict[str, str] = entry.get("files", {})
            if not files_map:
                raise ValueError(f"El motor {voice_info.name} no tiene lista de archivos configurada.")

            dest_dir = self.models_dir / "cloning" / entry.get("cloning_subdir", voice_info.key)
            dest_dir.mkdir(parents=True, exist_ok=True)
            total_files = len(files_map)

            for idx, (fname, url) in enumerate(files_map.items()):
                dest_file = dest_dir / fname
                if dest_file.is_file() and dest_file.stat().st_size > 0:
                    continue

                tmp_file = dest_dir / f"{fname}.part"
                base_pct = (idx / total_files) * 100.0
                span_pct = (1.0 / total_files) * 100.0

                def _clone_reporthook(block_num, block_size, total_size):
                    if total_size > 0 and progress_callback:
                        downloaded = block_num * block_size
                        frac = min(1.0, downloaded / total_size)
                        percent = min(99.0, base_pct + frac * span_pct)
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        progress_callback(
                            percent,
                            f"Descargando {fname}: {mb_down:.1f} / {mb_total:.1f} MB ({percent:.0f}%)"
                        )

                urllib.request.urlretrieve(url, tmp_file, reporthook=_clone_reporthook)
                tmp_file.replace(dest_file)

            voice_info.is_installed = True
            voice_info.model_path = dest_dir
            voice_info.config_path = dest_dir / "config.json"
            if progress_callback:
                progress_callback(100.0, f"¡Modelo {voice_info.name} descargado y listo!")
            return True

        if not voice_info.download_model_url or not voice_info.download_config_url:
            raise ValueError(f"La voz {voice_info.name} no cuenta con URL de descarga.")

        entry = DEFAULT_CATALOG.get(voice_info.key, {})
        dest_model = self.models_dir / entry.get("model_file", f"{voice_info.key}.onnx")
        dest_config = self.models_dir / entry.get("config_file", f"{voice_info.key}.onnx.json")

        if not dest_config.exists():
            if progress_callback:
                progress_callback(5.0, f"Descargando metadatos de {voice_info.name}...")
            urllib.request.urlretrieve(voice_info.download_config_url, dest_config)

        if not dest_model.exists():
            def _reporthook(block_num, block_size, total_size):
                if total_size > 0 and progress_callback:
                    downloaded = block_num * block_size
                    percent = min(99.0, 5.0 + (downloaded / total_size) * 94.0)
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    progress_callback(percent, f"Descargando {voice_info.name}: {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.0f}%)")

            urllib.request.urlretrieve(
                voice_info.download_model_url,
                dest_model,
                reporthook=_reporthook
            )

        voice_info.is_installed = True
        voice_info.model_path = dest_model
        voice_info.config_path = dest_config

        if progress_callback:
            progress_callback(100.0, f"¡Voz {voice_info.name} lista para usar!")

        return True
